from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MarketSnapshot:
    symbol: str
    as_of: str
    timeframe: str
    provider: str
    rows: int
    latest_close: float
    latest_volume: float
    change_pct: float = 0.0
    avg_volume: float = 0.0


@dataclass
class SessionContext:
    """Today's intraday session state. Anchored on the regular session open in the configured timezone."""
    symbol: str
    as_of: str
    market_session: str = "unknown"
    minutes_since_open: int | None = None
    minutes_to_close: int | None = None
    live_price: float | None = None
    session_open: float | None = None
    session_high: float | None = None
    session_low: float | None = None
    session_vwap: float | None = None
    premarket_high: float | None = None
    premarket_low: float | None = None
    premarket_volume: float | None = None
    opening_range_high: float | None = None
    opening_range_low: float | None = None
    opening_range_status: str = "unavailable"
    time_of_day_rvol: float | None = None
    session_volume: float | None = None
    bars_loaded: int = 0
    interval: str = ""
    data_available: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class MarketState:
    """Broad-market state used for cross-checking a single-symbol decision."""
    as_of: str
    spy_change_pct: float | None = None
    qqq_change_pct: float | None = None
    iwm_change_pct: float | None = None
    vix_value: float | None = None
    vix_change_pct: float | None = None
    market_trend: str = "unknown"
    risk_environment: str = "unknown"
    notes: list[str] = field(default_factory=list)


@dataclass
class SectorContext:
    """Sector ETF used to confirm or contradict a symbol's setup."""
    symbol: str
    sector_name: str = "unknown"
    sector_etf: str = ""
    sector_change_pct: float | None = None
    sector_trend: str = "unknown"
    alignment: str = "unknown"
    notes: list[str] = field(default_factory=list)


@dataclass
class CalendarContext:
    """Time-of-day, earnings, and macro-event awareness for the analyzed symbol."""
    symbol: str
    as_of: str
    market_session: str = "unknown"
    minutes_to_open: int | None = None
    minutes_to_close: int | None = None
    next_earnings_date: str | None = None
    hours_to_earnings: float | None = None
    earnings_within_24h: bool = False
    macro_events_today: list[dict] = field(default_factory=list)
    macro_events_next_24h: list[dict] = field(default_factory=list)
    no_trade_flags: list[str] = field(default_factory=list)


@dataclass
class FeatureSet:
    symbol: str
    as_of: str
    latest_price: float
    indicators: dict[str, float | str | None] = field(default_factory=dict)
    levels: dict[str, float | None] = field(default_factory=dict)
    regime: str = "unknown"
    technical_score: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class ModelPrediction:
    model: str
    symbol: str
    horizon_days: int
    direction: str
    expected_return: float
    confidence: float
    predicted_price: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextSummary:
    symbol: str
    enabled: bool
    score: float
    sentiment: str
    catalysts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    raw_summary: str = ""
    features: dict[str, float | str | bool] = field(default_factory=dict)
    reasons_to_trade: list[str] = field(default_factory=list)
    reasons_to_skip: list[str] = field(default_factory=list)
    # White-box news provenance: the LLM/heuristic per-symbol summary that fed this
    # context (provider, grand_summary, day_trader_focus, dominant_category) and the
    # exact headlines used as evidence. Empty when no news analysis was supplied.
    news_analysis: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SignalDecision:
    symbol: str
    action: str
    confidence: float
    score: float
    timeframe: str
    reasons: list[str] = field(default_factory=list)
    model_scores: dict[str, float] = field(default_factory=dict)
    feature_scores: dict[str, float] = field(default_factory=dict)
    context_score: float = 0.0
    created_at: str = ""
    top_reason: str = ""
    # Per-component attribution so the UI can show, in a white-box way, how each
    # input (models/technicals/intraday/context/sentiment) and each penalty moved
    # the fused score. Each row: {component, raw_score, weight, contribution, kind}.
    score_breakdown: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RiskPlan:
    symbol: str
    action: str
    entry: float | None
    stop_loss: float | None
    targets: list[float] = field(default_factory=list)
    risk_reward: float | None = None
    max_position_risk: float | None = None
    position_size: int | None = None
    invalidation: str = ""
    notes: list[str] = field(default_factory=list)
    entry_zone: tuple[float, float] | None = None
    liquidity_ok: bool = True
    setup_quality: str = "unknown"
    risk_per_share: float | None = None
    planned_risk: float | None = None
    planned_position_value: float | None = None
    session_checks: dict[str, float | int | bool | str] = field(default_factory=dict)
    no_trade_reasons: list[str] = field(default_factory=list)
    stop_source: str = ""
    target_source: str = ""


@dataclass
class BacktestReport:
    strategy: str
    symbols: list[str]
    start: str
    end: str
    trades: int
    win_rate: float
    average_return: float
    max_drawdown: float
    sharpe_like: float
    no_trade_rate: float
    evaluations: int = 0
    no_trades: int = 0
    equity_curve: list[dict[str, float | str]] = field(default_factory=list)
    trade_log: list[dict[str, float | str | int]] = field(default_factory=list)


@dataclass
class AnalysisSnapshot:
    """Compact serializable record of a single analyze run, persisted for delta comparisons."""
    snapshot_id: str
    symbol: str
    timestamp: str
    horizon: str
    action: str
    score: float
    confidence: float
    live_price: float | None
    entry: float | None
    stop_loss: float | None
    target: float | None
    risk_reward: float | None
    market_session: str
    top_reason: str
    no_trade_flags: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    snapshot: MarketSnapshot
    features: FeatureSet
    predictions: list[ModelPrediction]
    context: ContextSummary
    decision: SignalDecision
    risk_plan: RiskPlan
    scanner_row: dict[str, float | str | bool | None] = field(default_factory=dict)
    horizon: str = "swing"
    session: SessionContext | None = None
    intraday_features: dict[str, float | str | None] = field(default_factory=dict)
    market_state: MarketState | None = None
    sector_context: SectorContext | None = None
    calendar: CalendarContext | None = None
    news_enrichment: dict[str, Any] = field(default_factory=dict)
    snapshot_record: AnalysisSnapshot | None = None
    previous_snapshots: list[AnalysisSnapshot] = field(default_factory=list)
