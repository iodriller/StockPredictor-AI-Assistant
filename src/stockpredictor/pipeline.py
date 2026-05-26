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
    return AnalysisResult(
        snapshot=snapshot,
        features=features,
        predictions=predictions,
        context=context,
        decision=decision,
        risk_plan=risk_plan,
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
            -abs(result.decision.score),
        ),
    )

