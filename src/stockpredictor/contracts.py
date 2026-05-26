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
    equity_curve: list[dict[str, float | str]] = field(default_factory=list)


@dataclass
class AnalysisResult:
    snapshot: MarketSnapshot
    features: FeatureSet
    predictions: list[ModelPrediction]
    context: ContextSummary
    decision: SignalDecision
    risk_plan: RiskPlan

