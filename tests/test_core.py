from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from stockpredictor.backtesting import run_backtest
from stockpredictor.config import load_settings
from stockpredictor.context import build_context_summary
from stockpredictor.contracts import ContextSummary, FeatureSet, ModelPrediction, SignalDecision
from stockpredictor.data import SyntheticProvider
from stockpredictor.features import build_feature_set
from stockpredictor.models import run_models
from stockpredictor.pipeline import analyze_symbol, scan_symbols
from stockpredictor.risk import apply_risk_controls
from stockpredictor.signals import fuse_signals


def test_default_config_loads() -> None:
    settings = load_settings("configs/default.yaml")
    assert "baseline" in settings.enabled_models()
    assert settings.watchlist()


def test_synthetic_data_and_features(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    frame = SyntheticProvider().fetch("TEST", "6mo", "1d")
    features = build_feature_set("TEST", frame, settings)
    assert features.latest_price > 0
    assert "rsi" in features.indicators
    assert features.regime in {"trending", "trending_high_volatility", "choppy", "choppy_high_volatility"}


def test_gaussian_process_model_outputs_prediction(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, enabled_models=["gaussian_process"])
    frame = SyntheticProvider().fetch("TEST", "6mo", "1d")
    predictions = run_models("TEST", frame, settings)
    assert len(predictions) == 1
    assert predictions[0].model == "gaussian_process"
    assert predictions[0].predicted_price > 0
    assert predictions[0].direction in {"up", "down", "flat"}


def test_signal_fusion_allows_non_actionable_output(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    features = FeatureSet(
        symbol="TEST",
        as_of="2026-01-01T00:00:00",
        latest_price=100.0,
        indicators={},
        regime="choppy",
        technical_score=0.0,
        reasons=["technical signal is neutral"],
    )
    predictions = [
        ModelPrediction(
            model="baseline",
            symbol="TEST",
            horizon_days=5,
            direction="flat",
            expected_return=0.0,
            confidence=0.1,
            predicted_price=100.0,
        )
    ]
    context = ContextSummary(symbol="TEST", enabled=False, score=0.0, sentiment="neutral")
    decision = fuse_signals("TEST", features, predictions, context, settings)
    assert decision.action in {"no_trade", "low_confidence"}


def test_risk_plan_for_actionable_long(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    frame = SyntheticProvider().fetch("TEST", "6mo", "1d")
    features = build_feature_set("TEST", frame, settings)
    features = replace(features, indicators={**features.indicators, "atr": 2.0, "atr_pct": 0.02})
    decision = SignalDecision(
        symbol="TEST",
        action="long",
        confidence=0.8,
        score=0.7,
        timeframe="1d",
    )
    adjusted, plan = apply_risk_controls(decision, features, frame, settings)
    assert adjusted.action == "long"
    assert plan.entry is not None
    assert plan.stop_loss is not None
    assert plan.position_size and plan.position_size > 0
    assert plan.risk_reward and plan.risk_reward >= 1.5


def test_context_manual_items(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    context = build_context_summary("TEST", settings, include_live_sources=False)
    assert context.enabled
    assert context.score > 0
    assert context.catalysts


def test_analyze_scan_and_backtest_with_synthetic_data(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    provider = SyntheticProvider()
    analysis = analyze_symbol("TEST", settings, provider=provider, include_context=False)
    assert analysis.snapshot.symbol == "TEST"
    assert analysis.predictions

    scan = scan_symbols(settings, symbols=["AAA", "BBB"], provider=provider)
    assert len(scan) == 2

    report = run_backtest(settings, symbols=["AAA"], provider=provider)
    assert report.symbols == ["AAA"]
    assert report.trades >= 0
    assert 0 <= report.no_trade_rate <= 1


def _test_settings(tmp_path: Path, enabled_models: list[str] | None = None):
    raw = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    raw["data"]["provider"] = "synthetic"
    raw["data"]["min_rows"] = 60
    raw["models"]["enabled"] = enabled_models or ["baseline"]
    raw["models"]["lookback_rows"] = 80
    raw["models"]["gaussian_process"]["max_train_rows"] = 55
    raw["models"]["arima"]["enabled"] = "arima" in raw["models"]["enabled"]
    raw["context_agent"]["sources"] = ["manual"]
    raw["context_agent"]["manual_items"] = [
        {"source": "fixture", "title": "TEST raises guidance after strong demand", "sentiment": "bullish", "impact": 0.5}
    ]
    raw["backtest"]["lookback_rows"] = 70
    raw["backtest"]["evaluation_step_days"] = 20
    raw["watchlists"]["default"] = ["AAA", "BBB"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_settings(config_path)

