from __future__ import annotations

from .config import Settings
from .contracts import ContextSummary, FeatureSet, ModelPrediction, SignalDecision
from .utils import clamp, now_utc_iso


def fuse_signals(
    symbol: str,
    features: FeatureSet,
    predictions: list[ModelPrediction],
    context: ContextSummary,
    settings: Settings,
) -> SignalDecision:
    weights = _normalized_weights(settings.signal_fusion.get("weights", {}))
    thresholds = settings.signal_fusion.get("thresholds", {})
    model_score, model_scores, disagreement = _model_component(predictions)
    technical_score = clamp(features.technical_score, -1.0, 1.0)
    context_score = clamp(context.score, -1.0, 1.0)
    sentiment_score = {"bullish": 0.4, "bearish": -0.4}.get(context.sentiment, 0.0)
    score = (
        weights.get("models", 0.0) * model_score
        + weights.get("technicals", 0.0) * technical_score
        + weights.get("context", 0.0) * context_score
        + weights.get("sentiment", 0.0) * sentiment_score
    )
    reasons = list(features.reasons)
    if context.catalysts:
        reasons.append("context has catalyst input")
    if context.risks:
        reasons.append("context has risk input")
    if disagreement:
        penalty = float(thresholds.get("disagreement_penalty", 0.18))
        score *= 1 - penalty
        reasons.append("model disagreement reduced confidence")

    component_confidence = _average([prediction.confidence for prediction in predictions if prediction.confidence > 0])
    confidence = clamp(abs(score) * 0.75 + component_confidence * 0.35 - (0.12 if disagreement else 0), 0.0, 1.0)
    action = _action_from_score(score, confidence, thresholds)
    if action in {"no_trade", "low_confidence"}:
        reasons.append("setup is not actionable under configured thresholds")

    return SignalDecision(
        symbol=symbol.upper(),
        action=action,
        confidence=confidence,
        score=score,
        timeframe=str(settings.data.get("interval", "1d")),
        reasons=_dedupe(reasons),
        model_scores=model_scores,
        feature_scores={"technicals": technical_score, "sentiment": sentiment_score},
        context_score=context_score,
        created_at=now_utc_iso(),
    )


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


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output

