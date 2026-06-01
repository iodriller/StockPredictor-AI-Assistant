from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import time

import pandas as pd
import pytest
import yaml

from stockpredictor.backtesting import _excursions, _simulate_exit, run_backtest
from stockpredictor.config import load_settings
from stockpredictor.context import build_context_summary
from stockpredictor.contracts import CalendarContext, ContextSummary, FeatureSet, ModelPrediction, SignalDecision
from stockpredictor.data import FallbackProvider, SyntheticProvider, _period_to_rows
from stockpredictor.features import build_feature_set
from stockpredictor.journal import append_journal_entry, load_journal_entries
from stockpredictor.models import run_models
from stockpredictor.pipeline import analyze_symbol, scan_symbols
from stockpredictor.risk import _merge_structural_target, apply_risk_controls
from stockpredictor.signals import fuse_signals


def test_education_help_enrichment_and_glossary() -> None:
    from stockpredictor import education as edu

    base = "A plain tooltip."
    assert edu.enrich_help("vwap", base, enabled=False) == base
    enriched = edu.enrich_help("vwap", base, enabled=True)
    assert enriched.startswith(base) and "💡" in enriched
    # Keys without a trader-usage note are returned unchanged even when enabled.
    assert edu.enrich_help("does_not_exist", base, enabled=True) == base
    # The glossary covers a broad set of terms and is alphabetized.
    groups = edu.glossary_groups()
    assert len(groups) > 25
    assert [term.lower() for term, _ in groups] == sorted(term.lower() for term, _ in groups)


def test_default_config_loads() -> None:
    settings = load_settings("configs/default.example.yaml")
    assert "baseline" in settings.enabled_models()
    assert settings.watchlist()


def test_horizon_fallbacks_keep_swing_and_position_vwap_guards(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/default.example.yaml").read_text(encoding="utf-8"))
    raw.pop("horizons", None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    settings = load_settings(config_path)

    assert settings.horizon_profile("swing")["max_entry_distance_from_vwap_pct"] == pytest.approx(0.60)
    assert settings.horizon_profile("position")["max_entry_distance_from_vwap_pct"] == pytest.approx(1.00)


def test_models_enabled_list_is_the_only_disable_mechanism(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/default.example.yaml").read_text(encoding="utf-8"))
    raw["models"]["enabled"] = ["baseline"]
    raw["models"]["baseline"] = {"enabled": False}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert load_settings(config_path).enabled_models() == ["baseline"]


def test_synthetic_data_and_features(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    frame = SyntheticProvider().fetch("TEST", "6mo", "1d")
    features = build_feature_set("TEST", frame, settings)
    assert features.latest_price > 0
    assert "rsi" in features.indicators
    assert features.regime in {"trending", "trending_high_volatility", "choppy", "choppy_high_volatility"}


def test_disabled_features_are_not_calculated(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["features"]["enabled"] = ["vwap"]
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    frame = SyntheticProvider().fetch("TEST", "6mo", "1d")
    features = build_feature_set("TEST", frame, settings)
    assert "vwap" in features.indicators
    assert "rsi" not in features.indicators
    assert "macd" not in features.indicators


def test_daily_volume_weighted_average_uses_recent_rolling_window(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["features"]["enabled"] = ["vwap"]
    raw["features"]["vwap_window"] = 3
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    frame = pd.DataFrame(
        {
            "Open": [100, 100, 100, 10, 20, 30],
            "High": [100, 100, 100, 10, 20, 30],
            "Low": [100, 100, 100, 10, 20, 30],
            "Close": [100, 100, 100, 10, 20, 30],
            "Volume": [1, 1, 1, 1, 1, 1],
        },
        index=pd.date_range("2026-01-01", periods=6, freq="D"),
    )

    features = build_feature_set("TEST", frame, settings)

    assert features.indicators["vwap"] == pytest.approx(20.0)


def _trend_frame(values) -> pd.DataFrame:
    import numpy as np

    close = np.asarray(values, dtype=float)
    index = pd.date_range("2025-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 2_000_000.0},
        index=index,
    )


def test_momentum_model_is_bullish_on_recent_reversal(tmp_path: Path) -> None:
    import numpy as np

    settings = _test_settings(tmp_path, enabled_models=["momentum"])
    # Down for 170 bars, then a clear reversal up over the last 30 — bullish "now".
    close = np.concatenate([np.linspace(140, 90, 170), np.linspace(90, 120, 30)])
    prediction = run_models("REV", _trend_frame(close), settings, horizon="swing")[0]

    assert prediction.model == "momentum"
    assert prediction.expected_return > 0  # reads the recent reversal, not the stale downtrend
    assert prediction.direction == "up"


def test_baseline_recency_weighting_softens_stale_downtrend(tmp_path: Path) -> None:
    import numpy as np

    from stockpredictor.models.baseline import BaselineTrendModel

    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    close = np.concatenate([np.linspace(140, 90, 170), np.linspace(90, 120, 30)])
    prediction = BaselineTrendModel().predict("REV", _trend_frame(close), settings)

    # The old unweighted fit returned roughly -0.23 here; recency weighting must pull
    # it much closer to flat so a fresh reversal is not reported as a strong decline.
    assert prediction.expected_return > -0.10


def test_model_component_scales_with_horizon(tmp_path: Path) -> None:
    from stockpredictor.contracts import ModelPrediction
    from stockpredictor.signals import _model_component

    settings = _test_settings(tmp_path)
    # A modest but real +1.5% / 5-day forecast should produce a meaningful (not
    # crushed) directional vote under the horizon-aware reference scale.
    prediction = ModelPrediction(
        model="momentum", symbol="X", horizon_days=5, direction="up",
        expected_return=0.015, confidence=0.6, predicted_price=101.5,
    )
    score, _, _ = _model_component([prediction], settings)
    assert score > 0.4


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


def _bullish_features() -> FeatureSet:
    return FeatureSet(
        symbol="TEST",
        as_of="2026-01-01T00:00:00",
        latest_price=100.0,
        indicators={},
        regime="trending",
        technical_score=0.5,
        reasons=["technical signal is positive"],
    )


def _bullish_prediction() -> ModelPrediction:
    return ModelPrediction(
        model="baseline",
        symbol="TEST",
        horizon_days=5,
        direction="up",
        expected_return=0.04,
        confidence=0.7,
        predicted_price=104.0,
    )


def test_score_breakdown_contributions_sum_to_score(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    context = ContextSummary(symbol="TEST", enabled=True, score=0.5, sentiment="bullish")
    decision = fuse_signals("TEST", _bullish_features(), [_bullish_prediction()], context, settings)

    assert decision.score_breakdown
    total = sum(float(row["contribution"]) for row in decision.score_breakdown)
    assert total == pytest.approx(decision.score, abs=1e-9)
    components = {row["component"] for row in decision.score_breakdown if row["kind"] == "component"}
    assert {"models", "technicals", "context", "sentiment"}.issubset(components)


def test_news_no_trade_flags_apply_soft_penalty(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    features = _bullish_features()
    predictions = [_bullish_prediction()]
    base_context = ContextSummary(symbol="TEST", enabled=True, score=0.5, sentiment="bullish")
    flagged_context = ContextSummary(
        symbol="TEST",
        enabled=True,
        score=0.5,
        sentiment="bullish",
        features={"news_no_trade_flag_count": 2.0},
        reasons_to_skip=["news no-trade flag: late extension"],
    )

    base = fuse_signals("TEST", features, predictions, base_context, settings)
    flagged = fuse_signals("TEST", features, predictions, flagged_context, settings)

    penalty = float(settings.signal_fusion["thresholds"]["news_no_trade_penalty"])
    assert flagged.score == pytest.approx(base.score * (1 - penalty), abs=1e-9)
    assert flagged.score < base.score  # soft shave, not a hard block
    assert any("news no-trade flags" in reason for reason in flagged.reasons)
    assert any(row.get("kind") == "penalty" and "news" in row["component"].lower() for row in flagged.score_breakdown)


def test_context_consumes_news_analysis_evidence(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    news_analysis = {
        "symbol": "TEST",
        "summary": {
            "symbol": "TEST",
            "grand_summary": "TEST momentum on AI demand",
            "dominant_category": "product_business",
            "analysis_provider": "localdeploy",
            "day_trader_focus": {
                "catalyst": "AI demand",
                "risk": "late extension",
                "tradeability": "confirm with VWAP",
                "no_trade_flags": ["late extension"],
            },
        },
        "headlines": [
            {
                "symbol": "TEST",
                "title": "TEST raises guidance",
                "url": "https://example.com/1",
                "impact": 0.5,
                "sentiment": "bullish",
                "freshness": 0.9,
                "category": "earnings_guidance",
            }
        ],
    }

    context = build_context_summary("TEST", settings, news_analysis=news_analysis)

    assert context.evidence and context.evidence[0]["title"] == "TEST raises guidance"
    assert context.news_analysis["grand_summary"] == "TEST momentum on AI demand"
    assert context.features["news_no_trade_flag_count"] == 1.0
    assert any("late extension" in reason for reason in context.reasons_to_skip)
    assert context.score > 0


def test_llm_stance_blends_into_context_score(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    summary = {
        "symbol": "TEST",
        "stance": {"direction": "bullish", "conviction": 0.8},
        "stance_score": 0.8,
        "day_trader_focus": {},
    }
    neutral_headlines = [
        {"symbol": "TEST", "title": "TEST mixed update", "url": "https://example.com/1", "impact": 0.0, "sentiment": "neutral", "freshness": 0.5}
    ]
    heuristic_news = {"symbol": "TEST", "summary": {**summary, "analysis_provider": "heuristic"}, "headlines": neutral_headlines}
    llm_news = {"symbol": "TEST", "summary": {**summary, "analysis_provider": "localdeploy"}, "headlines": neutral_headlines}

    heuristic_ctx = build_context_summary("TEST", settings, news_analysis=heuristic_news)
    llm_ctx = build_context_summary("TEST", settings, news_analysis=llm_news)

    # The LLM stance lifts the catalyst/context score; the heuristic stance does not.
    assert llm_ctx.score > heuristic_ctx.score
    assert llm_ctx.features["news_stance_score"] == pytest.approx(0.8)
    assert heuristic_ctx.features["news_stance_score"] == 0.0


def test_confidence_weights_are_config_driven(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["signal_fusion"]["thresholds"]["confidence_score_weight"] = 0.0
    raw["signal_fusion"]["thresholds"]["confidence_component_weight"] = 1.0
    raw["signal_fusion"]["thresholds"]["disagreement_confidence_penalty"] = 0.0
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    context = ContextSummary(symbol="TEST", enabled=True, score=0.5, sentiment="bullish")
    decision = fuse_signals("TEST", _bullish_features(), [_bullish_prediction()], context, settings)

    # confidence = abs(score)*0.0 + component_confidence(0.7)*1.0 - 0.0
    assert decision.confidence == pytest.approx(0.7, abs=1e-9)


def test_zero_model_weight_disables_disagreement_penalties_and_uses_ai_confidence(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline", "momentum"])
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["signal_fusion"]["weights"] = {
        "models": 0.0,
        "technicals": 0.0,
        "intraday": 0.0,
        "context": 0.8,
        "sentiment": 0.2,
    }
    raw["horizons"]["profiles"]["swing"]["weights"] = dict(raw["signal_fusion"]["weights"])
    raw["signal_fusion"]["thresholds"].update(
        {
            "long_score": 0.30,
            "short_score": -0.30,
            "confidence_score_weight": 0.60,
            "confidence_component_weight": 0.50,
            "disagreement_penalty": 0.18,
            "disagreement_confidence_penalty": 0.12,
        }
    )
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    context = ContextSummary(
        symbol="TEST",
        enabled=True,
        score=0.8,
        sentiment="bullish",
        features={"news_stance_score": 0.8, "context_confidence": 0.8},
    )
    predictions = [
        ModelPrediction(model="baseline", symbol="TEST", horizon_days=5, direction="up", expected_return=0.04, confidence=0.8, predicted_price=104.0),
        ModelPrediction(model="momentum", symbol="TEST", horizon_days=5, direction="down", expected_return=-0.04, confidence=0.8, predicted_price=96.0),
    ]

    decision = fuse_signals("TEST", replace(_bullish_features(), technical_score=-0.8), predictions, context, settings)

    assert decision.action == "long"
    assert decision.score == pytest.approx(0.72, abs=1e-9)
    assert decision.confidence == pytest.approx(0.832, abs=1e-9)
    assert not any(row["component"] == "model disagreement" for row in decision.score_breakdown)
    assert "model disagreement reduced confidence" not in decision.reasons


def test_closed_market_preserves_swing_setup_and_marks_execution_wait(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    ai_weights = {"models": 0.0, "technicals": 0.0, "intraday": 0.0, "context": 0.8, "sentiment": 0.2}
    raw["signal_fusion"]["weights"] = dict(ai_weights)
    raw["horizons"]["profiles"]["swing"]["weights"] = dict(ai_weights)
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    context = ContextSummary(
        symbol="TEST",
        enabled=True,
        score=0.8,
        sentiment="bullish",
        features={"news_stance_score": 0.8, "context_confidence": 0.8},
    )
    calendar = CalendarContext(
        symbol="TEST",
        as_of="2026-01-01T20:00:00-05:00",
        market_session="closed_overnight",
        no_trade_flags=["market is currently closed"],
    )

    decision = fuse_signals("TEST", _bullish_features(), [_bullish_prediction()], context, settings, calendar_context=calendar)

    assert decision.bias == "bullish"
    assert decision.signal_action == "long"
    assert decision.action == "long"
    assert decision.execution_blockers == ["market is currently closed"]


def test_closed_market_blocks_intraday_execution_but_preserves_signal(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    context = ContextSummary(symbol="TEST", enabled=True, score=0.5, sentiment="bullish")
    calendar = CalendarContext(
        symbol="TEST",
        as_of="2026-01-01T20:00:00-05:00",
        market_session="closed_overnight",
        no_trade_flags=["market is currently closed"],
    )

    decision = fuse_signals("TEST", _bullish_features(), [_bullish_prediction()], context, settings, horizon="intraday", calendar_context=calendar)

    assert decision.signal_action == "long"
    assert decision.action == "no_trade"
    assert decision.execution_blockers == ["market is currently closed"]


def test_risk_plan_for_actionable_long(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    frame = SyntheticProvider().fetch("TEST", "6mo", "1d")
    features = build_feature_set("TEST", frame, settings)
    latest = float(frame["Close"].iloc[-1])
    features = replace(
        features,
        indicators={
            **features.indicators,
            "atr": 2.0,
            "atr_pct": 0.02,
            "vwap": latest,
            "support": latest - 4,
            "resistance": latest + 8,
            "avg_volume": 2_000_000,
        },
    )
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
    assert plan.entry_zone is not None
    assert plan.liquidity_ok
    assert plan.setup_quality == "actionable"
    assert plan.risk_per_share and plan.risk_per_share > 0
    assert plan.planned_risk and plan.planned_risk > 0
    assert "max_daily_loss" in plan.session_checks
    assert len(plan.targets) == 1


def test_risk_blocks_low_liquidity(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    frame = SyntheticProvider().fetch("TEST", "6mo", "1d")
    features = build_feature_set("TEST", frame, settings)
    latest = float(frame["Close"].iloc[-1])
    features = replace(features, indicators={**features.indicators, "atr": 2.0, "atr_pct": 0.02, "vwap": latest, "avg_volume": 1})
    decision = SignalDecision(symbol="TEST", action="long", confidence=0.8, score=0.7, timeframe="1d")
    adjusted, plan = apply_risk_controls(decision, features, frame, settings)
    assert adjusted.action == "no_trade"
    assert not plan.liquidity_ok
    assert plan.setup_quality == "low_liquidity"


def test_risk_volume_fallback_uses_configured_window(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["features"]["volume_window"] = 3
    raw["risk"]["min_avg_volume"] = 900
    raw["risk"]["min_risk_reward"] = 0.1
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    frame = SyntheticProvider().fetch("TEST", "6mo", "1d")
    frame["Volume"] = 0.0
    frame.loc[frame.index[-3:], "Volume"] = 1000.0
    latest = float(frame["Close"].iloc[-1])
    features = build_feature_set("TEST", frame, settings)
    indicators = {
        **features.indicators,
        "atr": 1.0,
        "atr_pct": 0.01,
        "vwap": latest,
        "support": latest - 2,
        "resistance": latest + 10,
    }
    indicators.pop("avg_volume", None)
    features = replace(features, indicators=indicators)
    decision = SignalDecision(symbol="TEST", action="long", confidence=0.8, score=0.7, timeframe="1d")

    _, plan = apply_risk_controls(decision, features, frame, settings)

    assert plan.liquidity_ok
    assert plan.setup_quality != "low_liquidity"


def test_no_trade_reasons_do_not_mine_decision_text(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    frame = SyntheticProvider().fetch("TEST", "6mo", "1d")
    features = build_feature_set("TEST", frame, settings)
    decision = SignalDecision(
        symbol="TEST",
        action="no_trade",
        confidence=0.1,
        score=0.0,
        timeframe="1d",
        reasons=["noise in model output", "took no position"],
    )

    _, plan = apply_risk_controls(decision, features, frame, settings)

    assert plan.no_trade_reasons == ["fused signal is not actionable"]


def test_risk_plan_surfaces_stop_and_target_source(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    frame = SyntheticProvider().fetch("TEST", "6mo", "1d")
    features = build_feature_set("TEST", frame, settings)
    latest = float(frame["Close"].iloc[-1])
    features = replace(
        features,
        indicators={
            **features.indicators,
            "atr": 2.0,
            "atr_pct": 0.02,
            "vwap": latest - 1,
            "support": latest - 1.5,
            "resistance": latest + 0.5,
            "avg_volume": 2_000_000,
        },
    )
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["risk"]["min_risk_reward"] = 0.05
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    decision = SignalDecision(symbol="TEST", action="long", confidence=0.8, score=0.7, timeframe="1d")

    _, plan = apply_risk_controls(decision, features, frame, settings)

    assert plan.stop_source in {"support", "vwap", "atr_fallback"}
    assert plan.target_source in {"r_multiple", "structural_resistance"}


def test_structural_target_must_be_on_reward_side_of_entry() -> None:
    assert _merge_structural_target(115.0, 95.0, 100.0, long=True) == ([115.0], "r_multiple")
    assert _merge_structural_target(85.0, 105.0, 100.0, long=False) == ([85.0], "r_multiple")


def test_risk_long_stop_has_fallback_when_structural_levels_missing(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    frame = SyntheticProvider().fetch("TEST", "6mo", "1d")
    frame.loc[frame.index[-1], "Close"] = 0.25
    frame.loc[frame.index[-1], "Open"] = 0.25
    frame.loc[frame.index[-1], "High"] = 0.26
    frame.loc[frame.index[-1], "Low"] = 0.24
    features = build_feature_set("TEST", frame, settings)
    features = replace(
        features,
        indicators={
            **features.indicators,
            "atr": 2.0,
            "atr_pct": 8.0,
            "vwap": 0.0,
            "support": 0.0,
            "resistance": 2.0,
            "avg_volume": 2_000_000,
        },
    )
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["risk"]["skip_if_atr_pct_above"] = 10
    raw["risk"]["min_risk_reward"] = 0.1
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    decision = SignalDecision(symbol="TEST", action="long", confidence=0.8, score=0.7, timeframe="1d")

    _, plan = apply_risk_controls(decision, features, frame, settings)

    assert plan.stop_loss is not None


def test_context_manual_items(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    context = build_context_summary("TEST", settings, include_live_sources=False)
    assert context.enabled
    assert context.score > 0
    assert context.catalysts
    assert context.features["catalyst_score"] > 0
    assert context.reasons_to_trade


def test_context_summary_uses_loaded_checklist_once(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path)
    calls = 0

    def fake_load(path):
        nonlocal calls
        calls += 1
        return ["check catalyst", "check risk"]

    monkeypatch.setattr("stockpredictor.context._load_trader_checklist", fake_load)

    context = build_context_summary("TEST", settings, include_live_sources=False)

    assert calls == 1
    assert "2 items" in context.raw_summary


def test_context_sources_are_enforced(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["context_agent"]["sources"] = []
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    context = build_context_summary("TEST", settings, include_live_sources=False)
    assert context.catalysts == []
    assert context.features["catalyst_score"] == 0


def test_analyze_scan_and_backtest_with_synthetic_data(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    provider = SyntheticProvider()
    analysis = analyze_symbol("TEST", settings, provider=provider, include_context=False)
    assert analysis.snapshot.symbol == "TEST"
    assert analysis.predictions
    assert analysis.scanner_row["symbol"] == "TEST"
    assert analysis.scanner_row["benchmark"] == "SPY"
    assert analysis.scanner_row["relative_strength_pct"] is not None
    assert "volume_anomaly" in analysis.scanner_row
    assert "extension_from_vwap_pct" in analysis.scanner_row
    assert "liquidity_ok" in analysis.scanner_row

    scan = scan_symbols(settings, symbols=["AAA", "BBB"], provider=provider)
    assert len(scan) == 2
    assert all("rank_score" in result.scanner_row for result in scan)

    progress_updates = []
    report = run_backtest(settings, symbols=["AAA"], provider=provider, progress_callback=lambda value, message: progress_updates.append((value, message)))
    assert report.symbols == ["AAA"]
    assert report.trades >= 0
    assert 0 <= report.no_trade_rate <= 1
    assert report.evaluations > 0
    assert report.trade_log
    assert "setup_quality" in report.trade_log[0]
    assert report.initial_capital == 100000
    assert report.final_equity > 0
    assert report.symbol_stats[0]["symbol"] == "AAA"
    assert report.symbol_stats[0]["evaluations"] == report.evaluations
    assert progress_updates[-1] == (1.0, "Completed AAA")
    assert [value for value, _ in progress_updates] == sorted(value for value, _ in progress_updates)


def test_synthetic_analysis_skips_live_market_enrichment(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    provider = SyntheticProvider()
    monkeypatch.setattr("stockpredictor.market._yfinance_sector", lambda symbol: (_ for _ in ()).throw(AssertionError("live sector lookup called")))
    monkeypatch.setattr("stockpredictor.calendar._next_earnings_date", lambda symbol: (_ for _ in ()).throw(AssertionError("live earnings lookup called")))

    result = analyze_symbol("TEST", settings, provider=provider, include_context=False)

    assert result.snapshot.provider == "synthetic"


def test_disabled_context_agent_skips_rich_news_analysis(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["context_agent"]["enabled"] = False
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    monkeypatch.setattr("stockpredictor.pipeline.analyze_symbol_news", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rich news analysis called")))

    result = analyze_symbol("TEST", settings, provider=SyntheticProvider())

    assert result.context.enabled is False


def test_scan_symbols_does_not_apply_dashboard_cap(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    symbols = [f"TST{index}" for index in range(12)]
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["dashboard"]["max_scan_symbols"] = 2
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    class Result:
        def __init__(self, symbol: str) -> None:
            self.decision = SignalDecision(symbol=symbol, action="low_confidence", confidence=0.1, score=0.0, timeframe="1d")
            self.scanner_row = {"rank_score": 0.0}

    monkeypatch.setattr("stockpredictor.pipeline.analyze_symbol", lambda symbol, **kwargs: Result(symbol))

    assert len(scan_symbols(settings, symbols=symbols)) == len(symbols)
    assert len(scan_symbols(settings, symbols=symbols, max_symbols=2)) == 2


def test_scan_symbols_skips_expensive_news_analysis(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    calls = []

    class Result:
        def __init__(self, symbol: str) -> None:
            self.decision = SignalDecision(symbol=symbol, action="low_confidence", confidence=0.1, score=0.0, timeframe="1d")
            self.scanner_row = {"rank_score": 0.0}

    def fake_analyze(symbol, **kwargs):
        calls.append(kwargs)
        return Result(symbol)

    monkeypatch.setattr("stockpredictor.pipeline.analyze_symbol", fake_analyze)

    scan_symbols(settings, symbols=["AAA", "BBB"])

    assert calls
    assert all(call["include_news_analysis"] is False for call in calls)


def test_scan_symbols_uses_bounded_workers(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["scanner"]["workers"] = 2
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    lock = threading.Lock()
    active = 0
    max_active = 0

    class Result:
        def __init__(self, symbol: str) -> None:
            self.decision = SignalDecision(symbol=symbol, action="low_confidence", confidence=0.1, score=0.0, timeframe="1d")
            self.scanner_row = {"rank_score": 0.0}

    def fake_analyze(symbol, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return Result(symbol)

    monkeypatch.setattr("stockpredictor.pipeline.analyze_symbol", fake_analyze)

    scan_symbols(settings, symbols=["AAA", "BBB", "CCC"])

    assert max_active == 2


def test_analysis_surfaces_rich_news_outage(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    monkeypatch.setattr("stockpredictor.pipeline.analyze_symbol_news", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("LocalDeploy offline")))

    result = analyze_symbol("TEST", settings, provider=SyntheticProvider())

    assert result.news_enrichment["status"] == "unavailable"
    assert "LocalDeploy offline" in result.news_enrichment["error"]


def test_fallback_provider_logs_primary_failure(caplog) -> None:
    class BrokenProvider:
        name = "broken"

        def fetch(self, symbol: str, period: str, interval: str):
            raise RuntimeError("provider unavailable")

    provider = FallbackProvider(BrokenProvider(), SyntheticProvider(), min_rows=10)

    with caplog.at_level("WARNING", logger="stockpredictor.data"):
        frame = provider.fetch("TEST", "6mo", "1d")

    assert frame.attrs["provider"] == "synthetic"
    assert "provider unavailable" in caplog.text


def test_synthetic_intraday_row_count_respects_period() -> None:
    assert _period_to_rows("1y", "1m") == 252 * 390
    assert _period_to_rows("5d", "1h") == 5 * 6


def test_trade_journal_update_and_delete(tmp_path: Path) -> None:
    from stockpredictor.journal import delete_journal_entry, update_journal_entry

    settings = _test_settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["journal"] = {"enabled": True, "path": "journal.local.jsonl"}
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    record = append_journal_entry(settings, {"symbol": "test", "action": "long", "setup_type": "vwap_reclaim"})
    entry_id = record["id"]
    updated = update_journal_entry(settings, entry_id, {"outcome": "win", "notes": "good entry"})

    assert updated and updated["outcome"] == "win"
    assert updated["notes"] == "good entry"
    assert updated["id"] == entry_id
    assert load_journal_entries(settings)[-1]["outcome"] == "win"

    assert delete_journal_entry(settings, entry_id) is True
    assert load_journal_entries(settings) == []
    assert delete_journal_entry(settings, entry_id) is False


def test_trade_journal_roundtrip(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["journal"] = {"enabled": True, "path": "journal.local.jsonl"}
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    record = append_journal_entry(
        settings,
        {
            "symbol": "test",
            "action": "long",
            "setup_type": "vwap_reclaim",
            "followed_plan": True,
            "risk_respected": True,
            "entry_quality": 4,
            "exit_quality": 3,
            "outcome": "win",
        },
    )
    entries = load_journal_entries(settings)

    assert record["symbol"] == "TEST"
    assert entries[-1]["setup_type"] == "vwap_reclaim"


def test_session_context_from_synthetic_intraday(tmp_path: Path) -> None:
    from stockpredictor.data import SyntheticProvider
    from stockpredictor.session import build_session_context

    settings = _test_settings(tmp_path)
    provider = SyntheticProvider()
    intraday = provider.fetch_intraday("TEST", period="1d", interval="1m")
    session = build_session_context("TEST", intraday, settings)

    assert session.bars_loaded > 0
    assert session.live_price is not None
    # synthetic frame anchored to today's regular session in UTC translated to ET; session VWAP
    # must compute to a real number whenever bars are loaded.
    assert session.session_vwap is not None or session.session_open is not None


def test_stale_intraday_reference_is_not_exposed_as_live_price(tmp_path: Path, monkeypatch) -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from stockpredictor.session import build_session_context

    settings = _test_settings(tmp_path)
    monkeypatch.setattr(
        "stockpredictor.session._now_market",
        lambda tz_name: datetime(2026, 6, 1, 12, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    index = pd.date_range("2026-05-29 09:30", periods=3, freq="min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000.0, 1200.0, 1300.0],
        },
        index=index,
    )

    session = build_session_context("TEST", frame, settings)

    assert session.is_live is False
    assert session.session_date == "2026-05-29"
    assert session.live_price is None
    assert session.reference_price == 102.5
    assert session.time_of_day_rvol is None


def test_horizon_profile_overrides_atr_multiple(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["horizons"] = {
        "default": "swing",
        "profiles": {
            "intraday": {"horizon_days": 1, "lookback_rows": 30, "atr_stop_multiple": 0.5, "target_r_multiple": 1.0},
            "swing": {"horizon_days": 5, "lookback_rows": 80, "atr_stop_multiple": 1.5, "target_r_multiple": 1.5},
        },
    }
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    assert settings.horizon_profile("intraday")["atr_stop_multiple"] == 0.5
    assert settings.horizon_profile()["atr_stop_multiple"] == 1.5  # default
    assert settings.horizon_profile("nonexistent")["atr_stop_multiple"] == 1.5  # falls back


def test_missing_horizons_get_default_profiles(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw.pop("horizons", None)
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    assert {"intraday", "swing", "position"}.issubset(settings.horizons["profiles"])
    assert settings.horizon_profile("intraday")["name"] == "intraday"
    assert settings.horizon_profile("intraday")["horizon_days"] == 1


def test_snapshots_persist_and_diff(tmp_path: Path) -> None:
    from stockpredictor.contracts import AnalysisResult, ContextSummary, FeatureSet, MarketSnapshot, RiskPlan, SessionContext, SignalDecision
    from stockpredictor.snapshots import diff_snapshots, load_snapshots, record_snapshot

    settings = _test_settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["snapshots"] = {"enabled": True, "path": "snapshots.local.jsonl", "compare_window": 3}
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    def build(score: float, confidence: float, action: str, price: float) -> AnalysisResult:
        return AnalysisResult(
            snapshot=MarketSnapshot(symbol="TEST", as_of="t", timeframe="1d", provider="synthetic", rows=80, latest_close=price, latest_volume=1_000_000),
            features=FeatureSet(symbol="TEST", as_of="t", latest_price=price),
            predictions=[],
            context=ContextSummary(symbol="TEST", enabled=False, score=0.0, sentiment="neutral"),
            decision=SignalDecision(symbol="TEST", action=action, confidence=confidence, score=score, timeframe="1d", top_reason="trend is up"),
            risk_plan=RiskPlan(symbol="TEST", action=action, entry=price, stop_loss=price - 1, targets=[price + 1.5], risk_reward=1.5),
            session=SessionContext(symbol="TEST", as_of="t", market_session="regular_morning", live_price=price),
        )

    first = record_snapshot(settings, build(0.4, 0.55, "long", 100.0), horizon="swing")
    second = record_snapshot(settings, build(0.62, 0.71, "long", 102.5), horizon="swing")
    loaded = load_snapshots(settings, "TEST", limit=10)

    assert len(loaded) == 2
    assert loaded[-1].snapshot_id == second.snapshot_id
    diff = diff_snapshots(second, first)
    assert round(diff["score_delta"], 2) == 0.22
    assert round(diff["live_price_delta"], 2) == 2.5
    assert diff["action_changed"] is False


def test_news_freshness_promotes_recent_items(tmp_path: Path, monkeypatch) -> None:
    from datetime import datetime, timedelta, timezone

    from stockpredictor.news import build_news_feed

    settings = _test_settings(tmp_path)
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["context_agent"]["news_analysis"]["llm"]["enabled"] = False
    raw["context_agent"]["news_analysis"]["article_scraping"]["enabled"] = False
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)
    now = datetime.now(timezone.utc)
    fresh = (now - timedelta(minutes=5)).isoformat()
    stale = (now - timedelta(hours=12)).isoformat()
    monkeypatch.setattr(
        "stockpredictor.news.fetch_news_items",
        lambda symbols, limit=50, **kwargs: [
            {"symbol": "TEST", "title": "Old guidance recap", "url": "https://example.com/old", "published": stale, "impact": 0.4, "sentiment": "bullish"},
            {"symbol": "TEST", "title": "TEST raises guidance", "url": "https://example.com/new", "published": fresh, "impact": 0.5, "sentiment": "bullish"},
        ],
    )

    feed = build_news_feed(["TEST"], settings, limit=10)
    headlines = feed["headlines"]

    assert headlines[0]["title"] == "TEST raises guidance"
    assert headlines[0]["freshness"] > headlines[1]["freshness"]
    assert feed["fresh_catalyst_count"] == 1


def test_calendar_no_trade_flag_for_earnings_within_24h(tmp_path: Path, monkeypatch) -> None:
    from datetime import datetime, timedelta, timezone

    from stockpredictor.calendar import build_calendar_context

    settings = _test_settings(tmp_path)
    in_three_hours = datetime.now(timezone.utc) + timedelta(hours=3)
    monkeypatch.setattr("stockpredictor.calendar._next_earnings_date", lambda symbol: in_three_hours)

    context = build_calendar_context("TEST", settings)

    assert context.earnings_within_24h is True
    assert any("earnings inside 24h" in flag for flag in context.no_trade_flags)


def test_backtest_exit_simulation_paths() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    target_window = pd.DataFrame({"High": [101, 106, 107], "Low": [99, 100, 101]}, index=index)
    stop_window = pd.DataFrame({"High": [101, 102, 103], "Low": [99, 94, 93]}, index=index)
    time_window = pd.DataFrame({"High": [101, 102, 103], "Low": [99, 98, 97]}, index=index)
    assert _simulate_exit("long", target_window, stop_loss=95, target=105)[1] == "target_hit"
    assert _simulate_exit("long", stop_window, stop_loss=95, target=110)[1] == "stop_hit"
    assert _simulate_exit("long", time_window, stop_loss=95, target=110)[1] == "time_exit"


def test_backtest_session_guard_stops_after_consecutive_losses(tmp_path: Path, monkeypatch) -> None:
    settings = _test_settings(tmp_path, enabled_models=["baseline"])
    raw = yaml.safe_load(settings.path.read_text(encoding="utf-8"))
    raw["risk"]["stop_after_consecutive_losses"] = 1
    raw["risk"]["max_trades_per_day"] = 0
    raw["risk"]["max_daily_loss_pct"] = 0
    raw["backtest"]["evaluation_step_days"] = 5
    raw["backtest"]["lookback_rows"] = 70
    settings.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(settings.path)

    from stockpredictor.contracts import AnalysisResult, FeatureSet, MarketSnapshot, RiskPlan, SignalDecision
    from stockpredictor.data import SyntheticProvider

    def fake_analyze(symbol, settings=None, provider=None, model_names=None, data_frame=None, include_context=True):
        return AnalysisResult(
            snapshot=MarketSnapshot(symbol=symbol, as_of="t", timeframe="1d", provider="synthetic", rows=80, latest_close=100.0, latest_volume=1_000_000),
            features=FeatureSet(symbol=symbol, as_of="t", latest_price=100.0),
            predictions=[],
            context=__import__("stockpredictor.contracts", fromlist=["ContextSummary"]).ContextSummary(symbol=symbol, enabled=False, score=0.0, sentiment="neutral"),
            decision=SignalDecision(symbol=symbol, action="long", confidence=0.9, score=0.7, timeframe="1d"),
            risk_plan=RiskPlan(symbol=symbol, action="long", entry=100.0, stop_loss=95.0, targets=[110.0], risk_per_share=5.0, position_size=10, planned_risk=50.0, planned_position_value=1000.0, risk_reward=2.0),
        )

    monkeypatch.setattr("stockpredictor.backtesting.analyze_symbol", fake_analyze)

    # Force every trade to lose by making the future window dip below the stop.
    monkeypatch.setattr("stockpredictor.backtesting._simulate_exit", lambda action, window, stop_loss, target: (stop_loss, "stop_hit", "2026-01-01"))

    report = run_backtest(settings, symbols=["AAA"], provider=SyntheticProvider())
    blocked = [row for row in report.trade_log if row.get("exit_reason") == "session_blocked"]
    assert blocked, "expected at least one session-blocked entry once consecutive losses hit the limit"


def test_excursions_use_entry_denominator() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="D")
    window = pd.DataFrame({"High": [110, 108], "Low": [90, 95]}, index=index)

    long_mae, long_mfe = _excursions("long", window, entry=100)
    short_mae, short_mfe = _excursions("short", window, entry=100)

    assert long_mae == 0.1
    assert long_mfe == 0.1
    assert short_mae == 0.1
    assert short_mfe == 0.1


def _test_settings(tmp_path: Path, enabled_models: list[str] | None = None):
    raw = yaml.safe_load(Path("configs/default.example.yaml").read_text(encoding="utf-8"))
    raw["data"]["provider"] = "synthetic"
    raw["data"]["min_rows"] = 60
    raw["models"]["enabled"] = enabled_models or ["baseline"]
    raw["models"]["lookback_rows"] = 80
    raw["models"]["gaussian_process"]["max_train_rows"] = 55
    raw["context_agent"]["sources"] = ["manual"]
    raw["context_agent"]["manual_items"] = [
        {
            "source": "fixture",
            "title": "TEST raises guidance after strong demand",
            "sentiment": "bullish",
            "impact": 0.5,
            "freshness": 0.9,
            "market_alignment": 0.2,
            "sector_alignment": 0.3,
        }
    ]
    raw["backtest"]["lookback_rows"] = 70
    raw["backtest"]["evaluation_step_days"] = 20
    raw["watchlists"]["default"] = ["AAA", "BBB"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_settings(config_path)
