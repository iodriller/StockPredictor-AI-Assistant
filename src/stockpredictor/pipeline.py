from __future__ import annotations

import pandas as pd

from .config import Settings, load_settings
from .context import build_context_summary
from .contracts import AnalysisResult
from .data import MarketDataProvider, build_snapshot, fetch_market_data, get_market_data_provider
from .features import build_feature_set
from .models import run_models
from .risk import apply_risk_controls
from .signals import fuse_signals


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
) -> list[AnalysisResult]:
    settings = settings or load_settings()
    provider = provider or get_market_data_provider(settings)
    symbols = symbols or settings.watchlist()
    max_symbols = int(settings.dashboard.get("max_scan_symbols", len(symbols)))
    results = [
        analyze_symbol(symbol, settings=settings, provider=provider)
        for symbol in symbols[:max_symbols]
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
    risk_reward = risk_plan.risk_reward
    catalyst_flag = bool(context.catalysts)
    risk_flag = bool(context.risks or context.reasons_to_skip)
    rank_score = (
        abs(decision.score) * 0.45
        + decision.confidence * 0.30
        + abs(float(snapshot.change_pct)) * 4.0
        + max(float(volume_anomaly or 1.0) - 1.0, 0.0) * 0.10
        + (0.08 if catalyst_flag else 0.0)
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
    }
