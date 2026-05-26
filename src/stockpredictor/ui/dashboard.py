from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stockpredictor.backtesting import run_backtest
from stockpredictor.config import ConfigError, load_settings
from stockpredictor.data import fetch_market_data, get_market_data_provider
from stockpredictor.pipeline import analyze_symbol, scan_symbols
from stockpredictor.utils import to_serializable


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="StockPredictor", layout="wide")
    st.title("StockPredictor Trading Intelligence")
    st.caption("Research and decision support only. No brokerage execution.")

    config_path = st.sidebar.text_input("Config path", value=args.config)
    try:
        settings = load_settings(config_path)
    except ConfigError as exc:
        st.error(str(exc))
        return

    watchlists = settings.raw.get("watchlists", {})
    watchlist_name = st.sidebar.selectbox(
        "Watchlist",
        options=list(watchlists.keys()),
        index=list(watchlists.keys()).index(settings.dashboard.get("default_watchlist", "default"))
        if settings.dashboard.get("default_watchlist", "default") in watchlists
        else 0,
    )
    symbols = st.sidebar.multiselect(
        "Symbols",
        options=[symbol.upper() for symbol in watchlists.get(watchlist_name, [])],
        default=settings.watchlist(watchlist_name),
    )

    scan_tab, deep_dive_tab, backtest_tab, config_tab = st.tabs(["Scanner", "Ticker Deep Dive", "Backtest", "Config"])
    with scan_tab:
        _render_scanner(settings, symbols)
    with deep_dive_tab:
        default_symbol = settings.dashboard.get("default_symbol", "AAPL")
        selected_symbol = st.text_input("Symbol", value=str(default_symbol)).upper()
        if st.button("Analyze", type="primary"):
            _render_analysis(settings, selected_symbol)
    with backtest_tab:
        if st.button("Run Backtest"):
            report = run_backtest(settings, symbols=symbols or None)
            st.dataframe(pd.DataFrame([asdict(report)]).drop(columns=["equity_curve", "trade_log"]), use_container_width=True)
            if report.equity_curve:
                st.line_chart(pd.DataFrame(report.equity_curve).set_index("date")["equity"])
            if report.trade_log:
                st.subheader("Trade Log")
                st.dataframe(pd.DataFrame(report.trade_log), use_container_width=True)
    with config_tab:
        st.json(to_serializable(settings.raw))


def _render_scanner(settings, symbols: list[str]) -> None:
    if st.button("Run Scan", type="primary"):
        results = scan_symbols(settings, symbols=symbols or None)
        rows = [_rounded_row(result.scanner_row) for result in results]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_analysis(settings, symbol: str) -> None:
    result = analyze_symbol(symbol, settings)
    provider = get_market_data_provider(settings)
    frame = fetch_market_data(symbol, settings, provider)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Action", result.decision.action)
    col2.metric("Confidence", f"{result.decision.confidence:.2f}")
    col3.metric("Score", f"{result.decision.score:.2f}")
    col4.metric("Last Price", f"{result.snapshot.latest_close:.2f}")

    st.subheader("Price And Levels")
    st.plotly_chart(_price_chart(frame, result.features.levels), use_container_width=True)

    st.subheader("Model Predictions")
    st.dataframe(pd.DataFrame([asdict(prediction) for prediction in result.predictions]), use_container_width=True)

    st.subheader("Signal")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "action": result.decision.action,
                    "confidence": result.decision.confidence,
                    "score": result.decision.score,
                    "top_reason": result.decision.top_reason,
                    "reasons": "; ".join(result.decision.reasons),
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Risk Plan")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "entry_zone": result.risk_plan.entry_zone,
                    "entry": result.risk_plan.entry,
                    "stop_loss": result.risk_plan.stop_loss,
                    "targets": result.risk_plan.targets,
                    "risk_reward": result.risk_plan.risk_reward,
                    "position_size": result.risk_plan.position_size,
                    "liquidity_ok": result.risk_plan.liquidity_ok,
                    "setup_quality": result.risk_plan.setup_quality,
                    "invalidation": result.risk_plan.invalidation,
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Context")
    st.write(result.context.raw_summary)
    context_metrics = {
        "sentiment": result.context.sentiment,
        **result.context.features,
    }
    st.dataframe(pd.DataFrame([context_metrics]), use_container_width=True, hide_index=True)
    if result.context.catalysts:
        st.write("Catalysts")
        st.write(result.context.catalysts)
    if result.context.risks:
        st.write("Risks")
        st.write(result.context.risks)
    if result.context.reasons_to_trade:
        st.write("Reasons To Trade")
        st.write(result.context.reasons_to_trade)
    if result.context.reasons_to_skip:
        st.write("Reasons To Skip")
        st.write(result.context.reasons_to_skip)

    st.subheader("Technical Features")
    st.dataframe(
        pd.DataFrame(
            [{"name": key, "value": value} for key, value in to_serializable(result.features.indicators).items()]
        ),
        use_container_width=True,
        hide_index=True,
    )


def _price_chart(frame: pd.DataFrame, levels: dict[str, float | None]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["Close"], mode="lines", name="Close"))
    for name, value in levels.items():
        if value:
            fig.add_hline(y=float(value), annotation_text=name, line_dash="dot", opacity=0.55)
    fig.update_layout(height=420, margin={"l": 20, "r": 20, "t": 10, "b": 20})
    return fig


def _rounded_row(row: dict) -> dict:
    rounded = dict(row)
    for key in ["price", "change_pct", "volume_anomaly", "gap_pct", "atr_pct", "confidence", "score", "risk_reward", "rank_score"]:
        if rounded.get(key) is not None:
            rounded[key] = round(float(rounded[key]), 4)
    return rounded


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/default.yaml")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    main()
