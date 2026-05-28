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
    model_score, model_scores, disagreement = _model_component(predictions)
    technical_score = clamp(features.technical_score, -1.0, 1.0)
    context_score = clamp(context.score, -1.0, 1.0)
    sentiment_score = {"bullish": 0.4, "bearish": -0.4}.get(context.sentiment, 0.0)
    intraday_component = clamp(float(intraday_score), -1.0, 1.0)
    score = (
        weights.get("models", 0.0) * model_score
        + weights.get("technicals", 0.0) * technical_score
        + weights.get("intraday", 0.0) * intraday_component
        + weights.get("context", 0.0) * context_score
        + weights.get("sentiment", 0.0) * sentiment_score
    )
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
        score *= 1 - penalty
        reasons.append("model disagreement reduced confidence")

    # Calendar hard-blocks: earnings within 24h, market closed, high-impact macro event.
    hard_blockers: list[str] = []
    if calendar_context is not None and calendar_context.no_trade_flags:
        hard_blockers.extend(calendar_context.no_trade_flags)

    # Market/sector light penalty: shave confidence if market is risk-off or sector diverges.
    if market_state is not None and market_state.risk_environment == "elevated":
        score *= 0.85
        reasons.append("VIX is elevated; downweighting confidence")
    if sector_context is not None and sector_context.alignment == "divergent":
        score *= 0.90
        reasons.append(f"symbol is diverging from sector ETF ({sector_context.sector_etf})")

    component_confidence = _average([prediction.confidence for prediction in predictions if prediction.confidence > 0])
    confidence = clamp(abs(score) * 0.75 + component_confidence * 0.35 - (0.12 if disagreement else 0), 0.0, 1.0)
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
    )


def _resolve_weights(settings: Settings, horizon_profile: dict) -> dict[str, float]:
    """Pick weights per horizon if defined; otherwise fall back to the global weights."""
    base = settings.signal_fusion.get("weights", {})
    overrides = horizon_profile.get("weights") if isinstance(horizon_profile, dict) else None
    if overrides:
        merged = {**base, **overrides}
    else:
        merged = dict(base)
    return _normalized_weights(merged)


def _model_component(predictions: list[ModelPrediction]) -> tuple[float, dict[str, float], bool]:
    scores: dict[str, float] = {}
    for prediction in predictions:
        raw = clamp(prediction.expected_return / 0.05, -1.0, 1.0) * clamp(prediction.confidence, 0.0, 1.0)
        scores[prediction.model] = raw
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
