from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd

from .calendar import build_calendar_context
from .config import Settings, load_settings
from .context import build_context_summary
from .contracts import AnalysisResult
from .data import MarketDataProvider, build_snapshot, fetch_intraday_data, fetch_market_data, get_market_data_provider
from .features import build_feature_set, build_intraday_features, intraday_technical_score
from .market import build_market_state, build_sector_context
from .models import run_models
from .news import analyze_symbol_news
from .risk import apply_risk_controls
from .session import build_session_context
from .signals import fuse_signals
from .snapshots import load_snapshots, record_snapshot


LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[float, str], None]


def analyze_symbol(
    symbol: str,
    settings: Settings | None = None,
    provider: MarketDataProvider | None = None,
    model_names: list[str] | None = None,
    data_frame: pd.DataFrame | None = None,
    include_context: bool = True,
    horizon: str | None = None,
    include_market_context: bool = True,
    include_session: bool = True,
    include_snapshot: bool = True,
    news_limit: int | None = None,
    include_news_analysis: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisResult:
    settings = settings or load_settings()
    provider = provider or get_market_data_provider(settings)
    symbol = symbol.upper()
    horizon_name = (horizon or str(settings.horizons.get("default", "swing"))).lower()
    _notify(progress_callback, 0.05, f"Fetching market data for {symbol}")
    frame = data_frame.copy() if data_frame is not None else fetch_market_data(symbol, settings, provider)
    snapshot = build_snapshot(symbol, frame, settings, getattr(provider, "name", "configured"))
    _notify(progress_callback, 0.28, "Building indicators (VWAP, RSI, MACD, ATR, trend)")
    features = build_feature_set(symbol, frame, settings)
    if data_frame is None:
        _add_benchmark_features(symbol, frame, features, settings, provider)

    # Intraday session + multi-timeframe features (only meaningful in a live request,
    # not in the backtest, where data_frame is supplied historically).
    session = None
    intraday_features: dict = {}
    intraday_score = 0.0
    intraday_reasons: list[str] = []
    if include_session and data_frame is None:
        _notify(progress_callback, 0.42, "Reading today's intraday session")
        intraday_frame = fetch_intraday_data(symbol, settings, provider)
        session = build_session_context(symbol, intraday_frame, settings)
        if intraday_frame is not None:
            intraday_features = build_intraday_features(symbol, intraday_frame, session, settings)
            intraday_score, intraday_reasons = intraday_technical_score(intraday_features)

    # Broad-market and sector cross-check (skipped in backtest).
    market_state = None
    sector_context = None
    calendar_context = None
    if include_market_context and data_frame is None:
        _notify(progress_callback, 0.55, "Cross-checking market & sector")
        allow_live_enrichment = getattr(provider, "name", "") != "synthetic"
        market_state = build_market_state(settings, provider)
        sector_context = build_sector_context(symbol, settings, provider, include_live_lookup=allow_live_enrichment)
        calendar_context = build_calendar_context(symbol, settings, include_live_sources=allow_live_enrichment)

    _notify(progress_callback, 0.66, "Running price models")
    predictions = run_models(symbol, frame, settings, model_names=model_names, horizon=horizon_name)
    # Deep-dive (live) requests fold the rich news analysis into the decision so the
    # trade plan both uses and shows the gathered news. Scans and backtests skip this
    # to stay fast and avoid per-symbol LLM calls.
    news_analysis = None
    news_enrichment = {"status": "skipped", "reason": "Rich news analysis was not requested for this run."}
    if include_context and include_news_analysis and data_frame is None and _news_in_decision_enabled(settings):
        try:
            _notify(progress_callback, 0.80, "Summarizing & scoring news with the AI model (slowest step)")
            news_analysis = analyze_symbol_news(symbol, settings, limit=news_limit)
            summary = news_analysis.get("summary", {})
            analysis_provider = str(summary.get("analysis_provider", "unknown"))
            news_enrichment = {
                "status": "degraded" if analysis_provider in {"heuristic_fallback", "llm_error"} else "available",
                "analysis_provider": analysis_provider,
                "headline_count": len(news_analysis.get("headlines", [])),
            }
            if summary.get("llm_error"):
                news_enrichment["error"] = str(summary["llm_error"])
        except Exception as exc:  # never let a news outage break the analysis
            LOGGER.info("News analysis unavailable for %s: %s", symbol, exc)
            news_analysis = None
            news_enrichment = {"status": "unavailable", "error": str(exc)}
    elif not _news_in_decision_enabled(settings):
        news_enrichment = {"status": "disabled", "reason": "Rich news-in-decision analysis is disabled by configuration."}
    _notify(progress_callback, 0.92, "Fusing the decision and building the risk plan")
    context = build_context_summary(symbol, settings, include_live_sources=include_context, news_analysis=news_analysis)
    decision = fuse_signals(
        symbol,
        features,
        predictions,
        context,
        settings,
        horizon=horizon_name,
        intraday_score=intraday_score,
        intraday_reasons=intraday_reasons,
        market_state=market_state,
        sector_context=sector_context,
        calendar_context=calendar_context,
    )
    decision, risk_plan = apply_risk_controls(
        decision,
        features,
        frame,
        settings,
        session=session,
        horizon=horizon_name,
        intraday_features=intraday_features,
    )
    scanner_row = build_scanner_row(snapshot, features, context, decision, risk_plan)
    previous = load_snapshots(settings, symbol, limit=int(settings.raw.get("snapshots", {}).get("compare_window", 5)))
    result = AnalysisResult(
        snapshot=snapshot,
        features=features,
        predictions=predictions,
        context=context,
        decision=decision,
        risk_plan=risk_plan,
        scanner_row=scanner_row,
        horizon=horizon_name,
        session=session,
        intraday_features=intraday_features,
        market_state=market_state,
        sector_context=sector_context,
        calendar=calendar_context,
        news_enrichment=news_enrichment,
        previous_snapshots=previous,
    )
    if include_snapshot and data_frame is None:
        result.snapshot_record = record_snapshot(settings, result, horizon=horizon_name)
    _notify(progress_callback, 1.0, f"{symbol} analysis ready")
    return result


def _notify(progress_callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if progress_callback is not None:
        progress_callback(float(fraction), message)


def scan_symbols(
    settings: Settings | None = None,
    symbols: list[str] | None = None,
    provider: MarketDataProvider | None = None,
    max_symbols: int | None = None,
    horizon: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[AnalysisResult]:
    settings = settings or load_settings()
    provider = provider or get_market_data_provider(settings)
    symbols = symbols or settings.watchlist()
    selected_symbols = symbols[:max_symbols] if max_symbols is not None else symbols
    results = []
    total = max(len(selected_symbols), 1)
    workers = max(1, min(int(settings.raw.get("scanner", {}).get("workers", 4)), total))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scanner") as executor:
        futures = {
            executor.submit(analyze_symbol, symbol, settings=settings, provider=provider, horizon=horizon, include_news_analysis=False): symbol
            for symbol in selected_symbols
        }
        for index, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                results.append(future.result())
                message = f"Analyzed {symbol} ({index}/{total})"
            except Exception as exc:
                LOGGER.warning("Scanner analysis failed for %s: %s", symbol, exc)
                message = f"Skipped {symbol}: analysis failed ({index}/{total})"
            if progress_callback is not None:
                progress_callback(index / total, message)
    if progress_callback is not None:
        progress_callback(1.0, f"Scanner finished for {len(results)} symbol(s)")
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


def _news_in_decision_enabled(settings: Settings) -> bool:
    news_cfg = settings.context_agent.get("news_analysis", {})
    return (
        bool(settings.context_agent.get("enabled", False))
        and bool(news_cfg.get("enabled", False))
        and bool(news_cfg.get("use_in_decision", True))
    )


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
