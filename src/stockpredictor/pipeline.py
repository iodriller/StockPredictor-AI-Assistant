from __future__ import annotations

import logging

import pandas as pd

from .config import Settings, load_settings
from .context import build_context_summary
from .contracts import AnalysisResult
from .data import MarketDataProvider, build_snapshot, fetch_market_data, get_market_data_provider
from .features import build_feature_set
from .models import run_models
from .risk import apply_risk_controls
from .signals import fuse_signals


LOGGER = logging.getLogger(__name__)


def analyze_symbol(
    symbol: str,
    settings: Settings | None = None,
    provider: MarketDataProvider | None = None,
    model_names: list[str] | None = None,
    data_frame: pd.DataFrame | None = None,
    include_context: bool = True,
) -> AnalysisResult:
    settings = settings or load_settings()
    provider = provider or get_market_data_provider(settings)
    symbol = symbol.upper()
    frame = data_frame.copy() if data_frame is not None else fetch_market_data(symbol, settings, provider)
    snapshot = build_snapshot(symbol, frame, settings, getattr(provider, "name", "configured"))
    features = build_feature_set(symbol, frame, settings)
    if data_frame is None:
        _add_benchmark_features(symbol, frame, features, settings, provider)
    predictions = run_models(symbol, frame, settings, model_names=model_names)
    context = build_context_summary(symbol, settings, include_live_sources=include_context)
    decision = fuse_signals(symbol, features, predictions, context, settings)
    decision, risk_plan = apply_risk_controls(decision, features, frame, settings)
    scanner_row = build_scanner_row(snapshot, features, context, decision, risk_plan)
    return AnalysisResult(
        snapshot=snapshot,
        features=features,
        predictions=predictions,
        context=context,
        decision=decision,
        risk_plan=risk_plan,
        scanner_row=scanner_row,
    )


def scan_symbols(
    settings: Settings | None = None,
    symbols: list[str] | None = None,
    provider: MarketDataProvider | None = None,
    max_symbols: int | None = None,
) -> list[AnalysisResult]:
    settings = settings or load_settings()
    provider = provider or get_market_data_provider(settings)
    symbols = symbols or settings.watchlist()
    selected_symbols = symbols[:max_symbols] if max_symbols is not None else symbols
    results = [
        analyze_symbol(symbol, settings=settings, provider=provider)
        for symbol in selected_symbols
    ]
    action_rank = {"long": 0, "short": 0, "watch": 1, "low_confidence": 2, "no_trade": 3}
    return sorted(
        results,
        key=lambda result: (
            action_rank.get(result.decision.action, 9),
            -result.decision.confidence,
            -float(result.scanner_row.get("rank_score", abs(result.decision.score))),
        ),
    )


def build_scanner_row(snapshot, features, context, decision, risk_plan) -> dict[str, float | str | bool | None]:
    volume_anomaly = features.indicators.get("volume_anomaly")
    gap_pct = features.indicators.get("gap_pct")
    atr_pct = features.indicators.get("atr_pct")
    vwap = features.indicators.get("vwap")
    support = features.indicators.get("support")
    resistance = features.indicators.get("resistance")
    benchmark_change_pct = features.indicators.get("benchmark_change_pct")
    relative_strength_pct = features.indicators.get("relative_strength_pct")
    risk_reward = risk_plan.risk_reward
    catalyst_flag = bool(context.catalysts)
    risk_flag = bool(context.risks or context.reasons_to_skip)
    extension_from_vwap_pct = _pct_distance(snapshot.latest_close, vwap)
    distance_to_support_pct = _pct_distance(snapshot.latest_close, support)
    distance_to_resistance_pct = _pct_distance(snapshot.latest_close, resistance)
    volume_anomaly_value = _float_default(volume_anomaly, 1.0)
    gap_pct_value = _float_default(gap_pct, 0.0)
    high_relative_volume = bool(volume_anomaly_value >= 1.5)
    meaningful_gap = bool(abs(gap_pct_value) >= 0.02)
    vwap_alignment = _vwap_alignment(snapshot.latest_close, vwap)
    # rank_score is a heuristic display-order metric, not a probability:
    #   abs(score)        : [0, 1]      weighted 0.45 (signal magnitude)
    #   confidence        : [0, 1]      weighted 0.30 (model agreement)
    #   abs(change_pct)   : fractional, weighted 4.0 so a 1% mover is worth 0.04
    #   max(rvol-1, 0)*.1 : RVOL above 1.0 boosts; missing volume -> 0
    #   abs(gap_pct)*1.5  : 1% gap -> 0.015
    # The catalyst/risk flags add small boosts/penalties. Constants are tuned to put
    # actionable signals near the top while letting strong movers without signal
    # still surface in the scanner.
    rank_score = (
        abs(decision.score) * 0.45
        + decision.confidence * 0.30
        + abs(float(snapshot.change_pct)) * 4.0
        + max(volume_anomaly_value - 1.0, 0.0) * 0.10
        + abs(gap_pct_value) * 1.5
        + (0.08 if catalyst_flag else 0.0)
        + (0.05 if high_relative_volume else 0.0)
        - (0.08 if risk_flag and decision.action in {"long", "short"} else 0.0)
    )
    return {
        "symbol": snapshot.symbol,
        "price": snapshot.latest_close,
        "change_pct": snapshot.change_pct,
        "volume": snapshot.latest_volume,
        "avg_volume": snapshot.avg_volume,
        "volume_anomaly": float(volume_anomaly) if volume_anomaly is not None else None,
        "gap_pct": float(gap_pct) if gap_pct is not None else None,
        "atr_pct": float(atr_pct) if atr_pct is not None else None,
        "prior_high": _float_or_none(features.indicators.get("prior_high")),
        "prior_low": _float_or_none(features.indicators.get("prior_low")),
        "session_open": _float_or_none(features.indicators.get("session_open")),
        "opening_range_high": _float_or_none(features.indicators.get("opening_range_high")),
        "opening_range_low": _float_or_none(features.indicators.get("opening_range_low")),
        "opening_range_status": str(features.indicators.get("opening_range_status", "")),
        "extension_from_vwap_pct": extension_from_vwap_pct,
        "distance_to_support_pct": distance_to_support_pct,
        "distance_to_resistance_pct": distance_to_resistance_pct,
        "benchmark": str(features.indicators.get("benchmark", "")),
        "benchmark_change_pct": _float_or_none(benchmark_change_pct),
        "relative_strength_pct": _float_or_none(relative_strength_pct),
        "liquidity_ok": risk_plan.liquidity_ok,
        "high_relative_volume": high_relative_volume,
        "meaningful_gap": meaningful_gap,
        "vwap_alignment": vwap_alignment,
        "regime": features.regime,
        "trend": str(features.indicators.get("trend", "unknown")),
        "action": decision.action,
        "confidence": decision.confidence,
        "score": decision.score,
        "risk_reward": risk_reward,
        "catalyst_flag": catalyst_flag,
        "risk_flag": risk_flag,
        "top_reason": decision.top_reason or (decision.reasons[0] if decision.reasons else ""),
        "rank_score": rank_score,
        "setup_quality": risk_plan.setup_quality,
        "skip_reasons": "; ".join(risk_plan.no_trade_reasons),
    }


def _pct_distance(price: float, level: object) -> float | None:
    try:
        if level is None or float(level) == 0:
            return None
        return (float(price) / float(level)) - 1
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_default(value: object, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _vwap_alignment(price: float, vwap: object) -> str:
    distance = _pct_distance(price, vwap)
    if distance is None:
        return "unknown"
    if abs(distance) <= 0.002:
        return "at_vwap"
    return "above_vwap" if distance > 0 else "below_vwap"


def _add_benchmark_features(symbol: str, frame: pd.DataFrame, features, settings: Settings, provider: MarketDataProvider) -> None:
    benchmark = str(settings.data.get("benchmark", "")).strip().upper()
    if not benchmark or benchmark == symbol.upper() or len(frame) < 2:
        return
    try:
        benchmark_frame = fetch_market_data(benchmark, settings, provider)
        if len(benchmark_frame) < 2:
            return
        symbol_change = (float(frame["Close"].iloc[-1]) / float(frame["Close"].iloc[-2])) - 1
        benchmark_change = (float(benchmark_frame["Close"].iloc[-1]) / float(benchmark_frame["Close"].iloc[-2])) - 1
    except Exception as exc:
        LOGGER.info("Benchmark %s comparison unavailable for %s: %s", benchmark, symbol, exc)
        return
    features.indicators["benchmark"] = benchmark
    features.indicators["benchmark_change_pct"] = benchmark_change
    features.indicators["relative_strength_pct"] = symbol_change - benchmark_change
