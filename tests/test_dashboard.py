from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from stockpredictor.ui.dashboard import (
    HELP_TEXT,
    _action_presentation,
    _indicator_rows,
    _price_chart,
    _remembered_expander,
    _scanner_summary_df,
    _trust_balanced_weights,
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

    fig = _price_chart(frame, {"prior_high": 99.0, "session_open": 88.0}, ma_windows=[3])
    trace_names = [trace.name for trace in fig.data]
    annotation_texts = [annotation.text for annotation in fig.layout.annotations]

    assert "SMA_3" in trace_names
    assert "SMA_9" not in trace_names
    assert "prior_high: $99.00" in annotation_texts
    assert "session_open: $88.00" in annotation_texts


def test_scanner_summary_keeps_compact_trader_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "LOW",
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


def test_trust_balance_scales_price_side_to_zero_at_full_ai_trust() -> None:
    base = {"models": 0.35, "technicals": 0.30, "intraday": 0.10, "context": 0.20, "sentiment": 0.05}

    original = _trust_balanced_weights(base, 25)
    ai_only = _trust_balanced_weights(base, 100)

    assert original == pytest.approx(base)
    assert ai_only == pytest.approx({"models": 0.0, "technicals": 0.0, "intraday": 0.0, "context": 0.8, "sentiment": 0.2})


def test_action_presentation_explains_no_trade() -> None:
    label, meaning = _action_presentation("no_trade")

    assert "NO TRADE" in label
    assert "Sitting out" in meaning
