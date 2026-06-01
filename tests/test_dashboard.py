from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from stockpredictor.config import load_settings
from stockpredictor.ui.dashboard import (
    _BACKTEST_DEPTH_PRESETS,
    HELP_TEXT,
    _SIGNAL_BLEND_PRESETS,
    _apply_weight_overrides,
    _backtest_interpretation_messages,
    _backtest_settings_for_depth,
    _backtest_skip_reason_rows,
    _backtest_total_return,
    _backtest_trade_rows,
    _chart_levels_for_view,
    _decision_bias,
    _decision_execution_blockers,
    _decision_signal_action,
    _entry_readiness_label,
    _action_presentation,
    _indicator_rows,
    _price_chart,
    _remembered_expander,
    _result_explanation_text,
    _scanner_summary_df,
    _session_is_live,
    _normalized_signal_weights,
    _rebalance_signal_allocations,
    _signal_blend_radar,
    _signal_blend_weights,
)


def test_dashboard_initial_render_has_no_runtime_exception() -> None:
    dashboard = Path(__file__).parents[1] / "src" / "stockpredictor" / "ui" / "dashboard.py"

    app = AppTest.from_file(str(dashboard), default_timeout=30).run()

    assert not app.exception
    assert not app.tabs
    assert len(app.segmented_control) == 1
    assert app.segmented_control[0].options == ["Scanner", "Trade Plan", "News", "Backtest", "Journal", "Settings"]
    assert app.segmented_control[0].value == "Scanner"
    assert [button.label for button in app.button] == ["Scan Selected Symbols", "Clear selected symbols"]

    app.segmented_control[0].set_value("News").run()

    assert not app.exception
    assert "Get News" in [button.label for button in app.button]
    assert "Scan Selected Symbols" not in [button.label for button in app.button]


def test_price_chart_uses_configured_ma_windows_and_result_levels() -> None:
    frame = pd.DataFrame(
        {
            "Open": [10, 11, 12, 13, 14],
            "High": [11, 12, 13, 14, 15],
            "Low": [9, 10, 11, 12, 13],
            "Close": [10, 11, 12, 13, 14],
            "Volume": [100, 110, 120, 130, 140],
        },
        index=pd.date_range("2026-01-01", periods=5, freq="D"),
    )

    fig = _price_chart(frame, {"prior_high": 14.5, "session_open": 88.0}, ma_windows=[3])
    trace_names = [trace.name for trace in fig.data]
    annotation_texts = [annotation.text for annotation in fig.layout.annotations]

    assert "SMA_3" in trace_names
    assert "SMA_9" not in trace_names
    assert "prior high: $14.50" in annotation_texts
    assert all("$88.00" not in text for text in annotation_texts)


def test_price_chart_defaults_to_recent_thirty_bars() -> None:
    frame = pd.DataFrame(
        {
            "Open": range(100),
            "High": range(1, 101),
            "Low": range(100),
            "Close": range(100),
            "Volume": [100] * 100,
        },
        index=pd.date_range("2026-01-01", periods=100, freq="D"),
    )

    figure = _price_chart(frame, {}, ma_windows=[])

    assert len(figure.data[0].x) == 30


def test_chart_levels_filter_far_away_and_near_duplicate_lines() -> None:
    frame = pd.DataFrame(
        {"Low": [95, 96], "High": [105, 104]},
        index=pd.date_range("2026-01-01", periods=2, freq="D"),
    )

    levels = _chart_levels_for_view(
        {"resistance": 104.0, "prior_high": 104.2, "support": 95.0, "vwap": 70.0},
        frame,
    )

    assert levels == {"resistance": 104.0, "support": 95.0}


def test_session_is_live_requires_explicit_current_session_flag() -> None:
    assert _session_is_live(SimpleNamespace(is_live=True, live_price=101.0)) is True
    assert _session_is_live(SimpleNamespace(is_live=False, live_price=101.0)) is False
    assert _session_is_live(SimpleNamespace(live_price=101.0)) is False


def test_scanner_summary_keeps_compact_trader_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "LOW",
                "bias": "neutral",
                "signal_action": "watch",
                "action": "watch",
                "rank_score": 0.2,
                "confidence": 40.0,
                "price": 10.0,
                "change_pct": 1.2,
                "volume_anomaly": 1.1,
                "gap_pct": 0.2,
                "relative_strength_pct": -0.3,
                "top_reason": "mixed",
                "volume": 100000,
                "avg_volume": 90000,
            },
            {
                "symbol": "HIGH",
                "bias": "bullish",
                "signal_action": "long",
                "action": "long",
                "rank_score": 0.9,
                "confidence": 70.0,
                "price": 20.0,
                "change_pct": 3.0,
                "volume_anomaly": 2.5,
                "gap_pct": 1.0,
                "relative_strength_pct": 1.8,
                "top_reason": "breakout",
                "volume": 300000,
                "avg_volume": 100000,
            },
        ]
    )

    summary = _scanner_summary_df(df)

    assert summary["symbol"].tolist() == ["HIGH", "LOW"]
    assert "volume" not in summary.columns
    assert "avg_volume" not in summary.columns
    assert list(summary.columns) == [
        "symbol",
        "bias",
        "action",
        "rank_score",
        "confidence",
        "price",
        "change_pct",
        "volume_anomaly",
        "gap_pct",
        "relative_strength_pct",
        "top_reason",
    ]


def test_indicator_rows_use_readable_labels() -> None:
    rows = _indicator_rows(
        {
            "price_change_pct": 0.0123,
            "sma_3": 99.125,
            "sma_20": 101.25,
            "volume_anomaly": 2.4,
            "opening_range_status": "available",
        }
    )

    labels = {row["name"]: row["value"] for row in rows}

    assert labels["Price Change"] == "1.23%"
    assert labels["SMA 3"] == "$99.12"
    assert labels["SMA 20"] == "$101.25"
    assert labels["Relative Volume"] == "2.400"
    assert labels["Opening Range Status"] == "available"


def test_help_text_covers_decision_critical_terms() -> None:
    required = {
        "action",
        "confidence",
        "rank",
        "risk_reward",
        "entry",
        "stop",
        "target",
        "catalyst_score",
        "analysis_provider",
        "backtest_win_rate",
    }

    assert required.issubset(HELP_TEXT)
    assert all(HELP_TEXT[key] for key in required)


def test_remembered_expander_uses_non_rerunning_mode() -> None:
    captured = {}

    class Container:
        def expander(self, label, **kwargs):
            captured["label"] = label
            captured.update(kwargs)
            return "panel"

    panel = _remembered_expander("Details", "scanner_details", expanded=True, container=Container())

    assert panel == "panel"
    assert captured["label"] == "Details"
    assert captured["expanded"] is True
    assert captured["key"] == "expander_scanner_details"
    assert captured["on_change"] == "ignore"


def test_trader_blend_presets_support_balanced_and_news_stress_test() -> None:
    base = {"models": 0.35, "technicals": 0.30, "intraday": 0.10, "context": 0.20, "sentiment": 0.05}

    balanced = _signal_blend_weights(base, "Balanced confirmation")
    news_only = _signal_blend_weights(base, "News thesis stress test")

    assert balanced == pytest.approx(base)
    assert news_only == pytest.approx({"models": 0.0, "technicals": 0.0, "intraday": 0.0, "context": 0.8, "sentiment": 0.2})


def test_custom_is_the_default_trading_lens() -> None:
    assert next(iter(_SIGNAL_BLEND_PRESETS)) == "Custom"


def test_custom_signal_weights_are_normalized() -> None:
    weights = _normalized_signal_weights({"models": 20, "technicals": 30, "intraday": 0, "context": 40, "sentiment": 10})

    assert weights == pytest.approx({"models": 0.20, "technicals": 0.30, "intraday": 0.0, "context": 0.40, "sentiment": 0.10})


def test_signal_blend_radar_closes_the_shape_and_uses_percentages() -> None:
    figure = _signal_blend_radar({"models": 0.20, "technicals": 0.30, "intraday": 0.10, "context": 0.35, "sentiment": 0.05})
    trace = figure.data[0]

    assert list(trace.theta) == ["Price models", "Technicals", "Intraday tape", "News context", "News tone", "Price models"]
    assert list(trace.r) == pytest.approx([20, 30, 10, 35, 5, 20])
    assert list(figure.layout.polar.radialaxis.range) == [0, 100]


def test_signal_blend_rebalances_other_components_to_keep_exact_total() -> None:
    weights = _rebalance_signal_allocations(
        {"models": 50, "technicals": 30, "intraday": 10, "context": 20, "sentiment": 5},
        "models",
    )

    assert weights == {"models": 50, "technicals": 23, "intraday": 8, "context": 15, "sentiment": 4}
    assert sum(weights.values()) == 100


def test_signal_blend_at_one_hundred_zeros_other_components() -> None:
    weights = _rebalance_signal_allocations(
        {"models": 35, "technicals": 30, "intraday": 10, "context": 100, "sentiment": 5},
        "context",
    )

    assert weights == {"models": 0, "technicals": 0, "intraday": 0, "context": 100, "sentiment": 0}


def test_signal_blend_redistributes_evenly_after_single_axis_allocation() -> None:
    weights = _rebalance_signal_allocations(
        {"models": 0, "technicals": 0, "intraday": 0, "context": 80, "sentiment": 0},
        "context",
    )

    assert weights == {"models": 5, "technicals": 5, "intraday": 5, "context": 80, "sentiment": 5}


def test_backtest_standard_depth_expands_history_without_mutating_base_settings(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/default.example.yaml").read_text(encoding="utf-8"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    settings = load_settings(config_path)

    adjusted = _backtest_settings_for_depth(settings, "Standard review")

    assert list(_BACKTEST_DEPTH_PRESETS) == ["Quick check", "Standard review", "Deeper sample"]
    assert settings.data["period"] == "6mo"
    assert adjusted.data["period"] == "1y"
    assert adjusted.backtest["lookback_rows"] == 90
    assert adjusted.backtest["evaluation_step_days"] == 5


def test_backtest_interpretation_warns_when_trade_sample_is_too_small() -> None:
    report = SimpleNamespace(evaluations=30, trades=1, no_trade_rate=29 / 30)

    messages = _backtest_interpretation_messages(report)

    assert any(level == "warning" and "too small" in message for level, message in messages)
    assert any(level == "info" and "skipped 97%" in message for level, message in messages)


def test_backtest_helpers_separate_trades_from_skip_reasons_and_return() -> None:
    report = SimpleNamespace(
        total_return=0.05,
        trade_log=[
            {"symbol": "AAA", "exit_reason": "target_hit"},
            {"symbol": "AAA", "exit_reason": "no_trade", "skip_reasons": "risk/reward too low"},
            {"symbol": "BBB", "exit_reason": "no_trade", "skip_reasons": "risk/reward too low"},
        ],
    )

    assert _backtest_total_return(report) == 0.05
    assert _backtest_trade_rows(report) == [{"symbol": "AAA", "exit_reason": "target_hit"}]
    assert _backtest_skip_reason_rows(report) == [{"skip_reason": "risk/reward too low", "evaluations": 2}]


def test_weight_override_creates_missing_horizon_profile() -> None:
    raw = {"signal_fusion": {"weights": {}}, "horizons": {"profiles": {}}}
    weights = {"models": 0.0, "technicals": 0.0, "intraday": 0.0, "context": 0.8, "sentiment": 0.2}

    _apply_weight_overrides(raw, "swing", weights)

    assert raw["horizons"]["profiles"]["swing"]["weights"] == weights


def test_action_presentation_explains_no_trade() -> None:
    label, meaning = _action_presentation("no_trade")

    assert "NO TRADE" in label
    assert "Sitting out" in meaning


def test_old_cached_decision_gets_compatible_display_defaults() -> None:
    decision = SimpleNamespace(action="no_trade", score=0.25)

    assert _decision_bias(decision) == "bullish"
    assert _decision_signal_action(decision) == "no_trade"
    assert _decision_execution_blockers(decision) == []


def test_entry_readiness_distinguishes_market_wait_from_risk_skip() -> None:
    market_wait = SimpleNamespace(action="long", execution_blockers=["market is currently closed"])
    risk_skip = SimpleNamespace(action="no_trade", execution_blockers=["risk layer blocked trade: risk/reward too low"])

    assert _entry_readiness_label(market_wait) == "WAIT FOR MARKET OPEN"
    assert _entry_readiness_label(risk_skip) == "SKIP FRESH ENTRY"


def test_result_explanation_names_main_positive_and_negative_drivers() -> None:
    decision = SimpleNamespace(
        action="no_trade",
        signal_action="long",
        bias="bullish",
        score=0.31,
        confidence=0.44,
        score_breakdown=[
            {"component": "context", "kind": "component", "raw_score": 0.4, "weight": 0.64, "contribution": 0.25},
            {"component": "sector divergence (XLK)", "kind": "penalty", "weight": 0.9, "contribution": -0.04},
        ],
    )

    explanation = _result_explanation_text(decision)

    assert "BULLISH bias" in explanation
    assert "Context +0.250" in explanation
    assert "Sector Divergence (Xlk) -0.040" in explanation
