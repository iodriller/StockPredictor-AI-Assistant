from __future__ import annotations

from .config import Settings
from .contracts import (
    CalendarContext,
    ContextSummary,
    FeatureSet,
    MarketState,
    ModelPrediction,
    SectorContext,
    SignalDecision,
)
from .utils import clamp, dedupe_preserve_order, now_in_timezone_iso


def fuse_signals(
    symbol: str,
    features: FeatureSet,
    predictions: list[ModelPrediction],
    context: ContextSummary,
    settings: Settings,
    horizon: str | None = None,
    intraday_score: float = 0.0,
    intraday_reasons: list[str] | None = None,
    market_state: MarketState | None = None,
    sector_context: SectorContext | None = None,
    calendar_context: CalendarContext | None = None,
) -> SignalDecision:
    horizon_profile = settings.horizon_profile(horizon)
    weights = _resolve_weights(settings, horizon_profile)
    thresholds = settings.signal_fusion.get("thresholds", {})
    model_score, model_scores, disagreement = _model_component(predictions, settings)
    technical_score = clamp(features.technical_score, -1.0, 1.0)
    context_score = clamp(context.score, -1.0, 1.0)
    sentiment_score = {"bullish": 0.4, "bearish": -0.4}.get(context.sentiment, 0.0)
    intraday_component = clamp(float(intraday_score), -1.0, 1.0)
    # Record each weighted component so the UI can show, in a white-box way, how
    # much each input (including news-driven context/sentiment) moved the score.
    components = [
        ("models", model_score, float(weights.get("models", 0.0))),
        ("technicals", technical_score, float(weights.get("technicals", 0.0))),
        ("intraday", intraday_component, float(weights.get("intraday", 0.0))),
        ("context", context_score, float(weights.get("context", 0.0))),
        ("sentiment", sentiment_score, float(weights.get("sentiment", 0.0))),
    ]
    score_breakdown: list[dict] = [
        {"component": name, "raw_score": raw, "weight": weight, "contribution": weight * raw, "kind": "component"}
        for name, raw, weight in components
    ]
    score = sum(weight * raw for _, raw, weight in components)
    reasons = list(features.reasons)
    if intraday_reasons:
        reasons.extend(intraday_reasons[:3])
    if context.catalysts:
        reasons.append("context has catalyst input")
    if context.risks:
        reasons.append("context has risk input")
    reasons.extend(context.reasons_to_trade[:2])
    reasons.extend(context.reasons_to_skip[:2])
    if disagreement:
        penalty = float(thresholds.get("disagreement_penalty", 0.18))
        score = _record_penalty(score, 1 - penalty, "model disagreement", score_breakdown)
        reasons.append("model disagreement reduced confidence")

    # Calendar hard-blocks: earnings within 24h, market closed, high-impact macro event.
    hard_blockers: list[str] = []
    if calendar_context is not None and calendar_context.no_trade_flags:
        hard_blockers.extend(calendar_context.no_trade_flags)

    # Market/sector light penalty: shave confidence if market is risk-off or sector diverges.
    if market_state is not None and market_state.risk_environment == "elevated":
        score = _record_penalty(score, 0.85, "elevated VIX", score_breakdown)
        reasons.append("VIX is elevated; downweighting confidence")
    if sector_context is not None and sector_context.alignment == "divergent":
        score = _record_penalty(score, 0.90, f"sector divergence ({sector_context.sector_etf})", score_breakdown)
        reasons.append(f"symbol is diverging from sector ETF ({sector_context.sector_etf})")

    # Soft news penalty: LLM/heuristic no-trade flags from the news analysis shave
    # confidence and are surfaced as reasons, but never force an action on their own.
    news_flag_count = int(float(context.features.get("news_no_trade_flag_count", 0) or 0))
    if news_flag_count > 0:
        news_penalty = float(thresholds.get("news_no_trade_penalty", 0.15))
        score = _record_penalty(score, 1 - news_penalty, f"news no-trade flags ({news_flag_count})", score_breakdown)
        reasons.append("news no-trade flags reduced confidence")

    component_confidence = _average([prediction.confidence for prediction in predictions if prediction.confidence > 0])
    # Confidence weighting is config-driven so the no-trade calibration can be tuned
    # without code changes. Defaults preserve the original behavior.
    score_weight = float(thresholds.get("confidence_score_weight", 0.75))
    component_weight = float(thresholds.get("confidence_component_weight", 0.35))
    disagreement_conf_penalty = float(thresholds.get("disagreement_confidence_penalty", 0.12))
    confidence = clamp(
        abs(score) * score_weight + component_confidence * component_weight - (disagreement_conf_penalty if disagreement else 0),
        0.0,
        1.0,
    )
    action = _action_from_score(score, confidence, thresholds)
    if hard_blockers:
        action = "no_trade"
        reasons.extend(f"hard block: {flag}" for flag in hard_blockers)
    if action in {"no_trade", "low_confidence"}:
        reasons.append("setup is not actionable under configured thresholds")
    deduped_reasons = dedupe_preserve_order(reasons)

    return SignalDecision(
        symbol=symbol.upper(),
        action=action,
        confidence=confidence,
        score=score,
        timeframe=str(settings.data.get("interval", "1d")),
        reasons=deduped_reasons,
        model_scores=model_scores,
        feature_scores={
            "technicals": technical_score,
            "intraday": intraday_component,
            "sentiment": sentiment_score,
        },
        context_score=context_score,
        created_at=now_in_timezone_iso(str(settings.app.get("timezone", "UTC"))),
        top_reason=deduped_reasons[0] if deduped_reasons else "",
        score_breakdown=score_breakdown,
    )


def _record_penalty(score: float, factor: float, label: str, breakdown: list[dict]) -> float:
    """Apply a multiplicative penalty and record its signed contribution to the score."""
    new_score = score * factor
    breakdown.append(
        {
            "component": label,
            "raw_score": None,
            "weight": factor,
            "contribution": new_score - score,
            "kind": "penalty",
        }
    )
    return new_score


def _resolve_weights(settings: Settings, horizon_profile: dict) -> dict[str, float]:
    """Pick weights per horizon if defined; otherwise fall back to the global weights."""
    base = settings.signal_fusion.get("weights", {})
    overrides = horizon_profile.get("weights") if isinstance(horizon_profile, dict) else None
    if overrides:
        merged = {**base, **overrides}
    else:
        merged = dict(base)
    return _normalized_weights(merged)


def _model_component(predictions: list[ModelPrediction], settings: Settings) -> tuple[float, dict[str, float], bool]:
    """Turn model forecasts into a [-1, 1] directional score.

    Two calibration fixes over the old `expected_return / 0.05 * confidence`:
    1. The reference move scales with the forecast horizon (a 5-day forecast and a
       20-day forecast are not judged on the same yardstick), so a healthy trend
       forecast is no longer crushed toward zero by a flat 5% divisor.
    2. Confidence weights the vote down to a floor instead of multiplying it away,
       so a moderate-confidence but clearly directional model still counts.
    """
    cfg = settings.signal_fusion
    per_day = float(cfg.get("model_reference_move_per_day_pct", 0.002))
    floor = float(cfg.get("model_reference_move_floor_pct", 0.01))
    conf_floor = clamp(float(cfg.get("model_confidence_floor", 0.4)), 0.0, 1.0)
    scores: dict[str, float] = {}
    for prediction in predictions:
        reference = max(floor, per_day * max(1, int(prediction.horizon_days)))
        direction_strength = clamp(prediction.expected_return / reference, -1.0, 1.0)
        conf_weight = clamp(conf_floor + (1.0 - conf_floor) * clamp(prediction.confidence, 0.0, 1.0), 0.0, 1.0)
        scores[prediction.model] = direction_strength * conf_weight
    valid_scores = [score for score in scores.values() if score != 0]
    disagreement = any(score > 0.05 for score in valid_scores) and any(score < -0.05 for score in valid_scores)
    return (_average(valid_scores), scores, disagreement)


def _action_from_score(score: float, confidence: float, thresholds: dict[str, float]) -> str:
    min_confidence = float(thresholds.get("min_confidence", 0.35))
    if confidence < min_confidence:
        return "low_confidence"
    if score >= float(thresholds.get("long_score", 0.35)):
        return "long"
    if score <= float(thresholds.get("short_score", -0.35)):
        return "short"
    if abs(score) >= float(thresholds.get("watch_score", 0.18)):
        return "watch"
    return "no_trade"


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in weights.values()) or 1.0
    return {key: float(value) / total for key, value in weights.items()}


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
