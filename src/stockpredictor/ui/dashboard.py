from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd
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
            st.dataframe(pd.DataFrame([asdict(report)]).drop(columns=["equity_curve"]), use_container_width=True)
            if report.equity_curve:
                st.line_chart(pd.DataFrame(report.equity_curve).set_index("date")["equity"])
    with config_tab:
        st.json(to_serializable(settings.raw))


def _render_scanner(settings, symbols: list[str]) -> None:
    if st.button("Run Scan", type="primary"):
        results = scan_symbols(settings, symbols=symbols or None)
        rows = []
        for result in results:
            rows.append(
                {
                    "symbol": result.snapshot.symbol,
                    "action": result.decision.action,
                    "confidence": round(result.decision.confidence, 3),
                    "score": round(result.decision.score, 3),
                    "price": round(result.snapshot.latest_close, 2),
                    "regime": result.features.regime,
                    "risk_reward": result.risk_plan.risk_reward,
                    "reason": "; ".join(result.decision.reasons[:3]),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_analysis(settings, symbol: str) -> None:
    result = analyze_symbol(symbol, settings)
    provider = get_market_data_provider(settings)
    frame = fetch_market_data(symbol, settings, provider)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Action", result.decision.action)
    col2.metric("Confidence", f"{result.decision.confidence:.2f}")
    col3.metric("Score", f"{result.decision.score:.2f}")
    col4.metric("Last Price", f"{result.snapshot.latest_close:.2f}")

    st.subheader("Price")
    st.line_chart(frame[["Close"]])

    st.subheader("Model Predictions")
    st.dataframe(pd.DataFrame([asdict(prediction) for prediction in result.predictions]), use_container_width=True)

    st.subheader("Signal And Risk")
    signal_col, risk_col = st.columns(2)
    signal_col.json(to_serializable(result.decision))
    risk_col.json(to_serializable(result.risk_plan))

    st.subheader("Context")
    st.write(result.context.raw_summary)
    if result.context.catalysts:
        st.write("Catalysts")
        st.write(result.context.catalysts)
    if result.context.risks:
        st.write("Risks")
        st.write(result.context.risks)

    st.subheader("Technical Features")
    st.json(to_serializable(result.features))


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/default.yaml")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    main()
