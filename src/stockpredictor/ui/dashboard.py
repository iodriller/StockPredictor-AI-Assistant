from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from stockpredictor.backtesting import run_backtest
from stockpredictor.config import ConfigError, load_settings
from stockpredictor.data import fetch_market_data, get_market_data_provider
from stockpredictor.news import build_news_feed
from stockpredictor.pipeline import analyze_symbol, scan_symbols
from stockpredictor.symbols import normalize_symbol, search_symbols
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

    symbols = _render_symbol_sidebar(settings)

    scan_tab, deep_dive_tab, news_tab, backtest_tab, config_tab = st.tabs(["Scanner", "Ticker Deep Dive", "News Feed", "Backtest", "Config"])
    with scan_tab:
        _render_scanner(settings, symbols)
    with deep_dive_tab:
        default_symbol = symbols[0] if symbols else settings.dashboard.get("default_symbol", "AAPL")
        selected_symbol = st.text_input("Symbol", value=str(default_symbol)).upper()
        if st.button("Analyze", type="primary"):
            _render_analysis(settings, selected_symbol)
    with news_tab:
        _render_news(settings, symbols or settings.watchlist())
    with backtest_tab:
        if st.button("Run Backtest"):
            report = run_backtest(settings, symbols=symbols or None)
            report_df = pd.DataFrame([asdict(report)]).drop(columns=["equity_curve", "trade_log"])
            report_df = _percent_display(report_df, ["win_rate", "average_return", "max_drawdown", "no_trade_rate"])
            st.dataframe(
                report_df,
                use_container_width=True,
                hide_index=True,
                column_config=_backtest_column_config(),
            )
            if report.equity_curve:
                equity_df = pd.DataFrame(report.equity_curve)
                st.line_chart(equity_df.set_index("date")["equity"])
            if report.trade_log:
                st.subheader("Trade Log")
                trade_log_df = _percent_display(pd.DataFrame(report.trade_log), ["return", "confidence"])
                st.dataframe(trade_log_df, use_container_width=True, hide_index=True, column_config=_trade_log_column_config())
    with config_tab:
        st.json(to_serializable(settings.raw))


def _render_scanner(settings, symbols: list[str]) -> None:
    st.subheader("Scanner")
    st.caption("Ranks the selected symbols by movement, volume, signal confidence, catalyst/risk flags, and setup quality.")
    if st.button("Run Scan", type="primary"):
        results = scan_symbols(settings, symbols=symbols or None)
        rows = [_rounded_row(result.scanner_row) for result in results]
        if not rows:
            st.info("No symbols selected.")
            return
        df = pd.DataFrame(rows)
        actionable = int(df["action"].isin(["long", "short", "watch"]).sum()) if "action" in df else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scanned", len(df))
        c2.metric("Actionable / Watch", actionable)
        c3.metric("Avg RVOL", f"{df['volume_anomaly'].dropna().mean():.2f}" if "volume_anomaly" in df else "-")
        c4.metric("Top Rank", f"{df['rank_score'].max():.3f}" if "rank_score" in df else "-")
        st.dataframe(df, use_container_width=True, hide_index=True, column_config=_scanner_column_config())


def _render_analysis(settings, symbol: str) -> None:
    result = analyze_symbol(symbol, settings)
    provider = get_market_data_provider(settings)
    frame = fetch_market_data(symbol, settings, provider)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Action", result.decision.action)
    col2.metric("Confidence", _format_percent(result.decision.confidence))
    col3.metric("Score", f"{result.decision.score:.3f}")
    col4.metric("Last Price", _format_price(result.snapshot.latest_close), _format_percent(result.snapshot.change_pct))

    st.subheader("Price And Levels")
    st.plotly_chart(_price_chart(frame, result.features.levels), use_container_width=True)

    st.subheader("Model Predictions")
    model_df = _percent_display(pd.DataFrame([asdict(prediction) for prediction in result.predictions]), ["expected_return", "confidence"])
    st.dataframe(model_df, use_container_width=True, hide_index=True, column_config=_model_column_config())

    st.subheader("Signal")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "action": result.decision.action,
                    "confidence": result.decision.confidence * 100,
                    "score": result.decision.score,
                    "top_reason": result.decision.top_reason,
                    "reasons": "; ".join(result.decision.reasons),
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
            "score": st.column_config.NumberColumn("Score", format="%.3f"),
        },
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
        column_config=_risk_column_config(),
    )

    st.subheader("Context")
    st.write(result.context.raw_summary)
    context_metrics = {
        "sentiment": result.context.sentiment,
        **result.context.features,
    }
    context_metrics = _percent_context_metrics(context_metrics)
    st.dataframe(pd.DataFrame([context_metrics]), use_container_width=True, hide_index=True, column_config=_context_column_config())
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
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
    )
    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["Open"],
            high=frame["High"],
            low=frame["Low"],
            close=frame["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    for window in [9, 20, 50]:
        column = f"sma_{window}"
        if len(frame) >= window:
            fig.add_trace(go.Scatter(x=frame.index, y=frame["Close"].rolling(window).mean(), mode="lines", name=column.upper()), row=1, col=1)
    if "Volume" in frame:
        fig.add_trace(go.Bar(x=frame.index, y=frame["Volume"], name="Volume", marker_color="#8892a6"), row=2, col=1)
    chart_levels = dict(levels)
    if len(frame) >= 2:
        chart_levels["prior_high"] = float(frame["High"].iloc[-2])
        chart_levels["prior_low"] = float(frame["Low"].iloc[-2])
    chart_levels["session_open"] = float(frame["Open"].iloc[-1])
    for name, value in chart_levels.items():
        if value:
            fig.add_hline(y=float(value), annotation_text=f"{name}: {_format_price(float(value))}", line_dash="dot", opacity=0.55, row=1, col=1)
    fig.update_layout(height=560, margin={"l": 20, "r": 20, "t": 10, "b": 20}, xaxis_rangeslider_visible=False)
    return fig


def _render_symbol_sidebar(settings) -> list[str]:
    st.sidebar.subheader("Selected Symbols")
    if "selected_symbols_text" not in st.session_state:
        st.session_state.selected_symbols_text = ", ".join(settings.watchlist())

    watchlists = settings.raw.get("watchlists", {})
    watchlist_names = list(watchlists.keys()) or ["default"]
    watchlist_name = st.sidebar.selectbox(
        "Load watchlist",
        options=watchlist_names,
        index=watchlist_names.index(settings.dashboard.get("default_watchlist", "default"))
        if settings.dashboard.get("default_watchlist", "default") in watchlists
        else 0,
    )
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Use"):
        st.session_state.selected_symbols_text = ", ".join(settings.watchlist(watchlist_name))
    if col2.button("Clear"):
        st.session_state.selected_symbols_text = ""

    lookup_query = st.sidebar.text_input("Find ticker", placeholder="AAPL, BRK.B, Palantir, Nvidia")
    lookup_results = search_symbols(lookup_query, limit=12) if lookup_query else []
    if lookup_results:
        result_options = [f"{item['symbol']} - {item['name']}" for item in lookup_results]
        selected_lookup = st.sidebar.selectbox("Matches", options=result_options)
        if st.sidebar.button("Add match"):
            _append_symbol_to_state(selected_lookup.split(" - ", 1)[0])
    elif lookup_query:
        direct = normalize_symbol(lookup_query)
        if st.sidebar.button(f"Add {direct} directly"):
            _append_symbol_to_state(direct)

    st.sidebar.text_area(
        "Tickers to analyze",
        key="selected_symbols_text",
        height=96,
        placeholder="AAPL, NVDA, PLTR, SOFI",
    )
    symbols = _dedupe_symbols(_parse_symbols(st.session_state.selected_symbols_text))
    st.sidebar.caption(f"{len(symbols)} selected: {', '.join(symbols[:8])}{'...' if len(symbols) > 8 else ''}")
    return symbols


def _append_symbol_to_state(symbol: str) -> None:
    symbols = _dedupe_symbols(_parse_symbols(st.session_state.get("selected_symbols_text", "")) + [symbol])
    st.session_state.selected_symbols_text = ", ".join(symbols)


def _render_news(settings, symbols: list[str]) -> None:
    selected = st.multiselect("News symbols", options=symbols, default=symbols[: min(5, len(symbols))])
    headline_limit = st.slider("Headline limit", min_value=5, max_value=100, value=50, step=5)
    if st.button("Refresh News", type="primary"):
        feed = build_news_feed(selected or symbols, settings, limit=headline_limit)
        items = feed["headlines"]
        if not items:
            st.info("No recent headlines returned by the configured free provider.")
            return
        df = pd.DataFrame(items)
        bullish = int((df["sentiment"] == "bullish").sum()) if "sentiment" in df else 0
        bearish = int((df["sentiment"] == "bearish").sum()) if "sentiment" in df else 0
        neutral = len(df) - bullish - bearish
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Headlines", len(df))
        c2.metric("Bullish", bullish)
        c3.metric("Bearish", bearish)
        c4.metric("Neutral", neutral)
        c5.metric("Analysis", feed["analysis_provider"])

        st.subheader("Grand Summary By Stock")
        for summary in feed["summaries"]:
            _render_symbol_news_summary(summary)

        st.subheader("All Headlines")
        st.dataframe(
            df[["symbol", "category", "sentiment", "impact", "day_trader_relevance", "published", "provider", "title", "url"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "impact": st.column_config.NumberColumn("Impact", format="%.2f"),
                "day_trader_relevance": st.column_config.NumberColumn("Relevance", format="%.2f"),
                "url": st.column_config.LinkColumn("Link"),
            },
        )


def _render_symbol_news_summary(summary: dict) -> None:
    st.markdown(f"### {summary['symbol']}")
    st.write(summary.get("grand_summary", "No summary available."))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sources", summary.get("source_count", 0))
    c2.metric("Headlines", summary.get("headline_count", 0))
    c3.metric("Bullish", summary.get("bullish_count", 0))
    c4.metric("Bearish", summary.get("bearish_count", 0))
    c5.metric("Category", str(summary.get("dominant_category", "other")).replace("_", " "))

    focus = summary.get("day_trader_focus", {})
    st.dataframe(
        pd.DataFrame(
            [
                {"question": "Catalyst", "answer": focus.get("catalyst", "-")},
                {"question": "Risk", "answer": focus.get("risk", "-")},
                {"question": "Tradeability", "answer": focus.get("tradeability", "-")},
                {"question": "No-trade flags", "answer": "; ".join(focus.get("no_trade_flags", [])) or "-"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    sources = pd.DataFrame(summary.get("sources", []))
    with st.expander(f"Sources: {summary.get('source_count', 0)}"):
        if sources.empty:
            st.info("No linked sources were returned.")
        else:
            columns = [column for column in ["provider", "published", "category", "sentiment", "impact", "title", "url"] if column in sources.columns]
            st.dataframe(
                sources[columns],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "impact": st.column_config.NumberColumn("Impact", format="%.2f"),
                    "url": st.column_config.LinkColumn("Link"),
                },
            )


def _rounded_row(row: dict) -> dict:
    rounded = dict(row)
    percent_keys = {"change_pct", "gap_pct", "atr_pct", "confidence"}
    for key in ["price", "change_pct", "volume_anomaly", "gap_pct", "atr_pct", "confidence", "score", "risk_reward", "rank_score"]:
        if rounded.get(key) is not None:
            value = float(rounded[key])
            if key in percent_keys:
                value *= 100
            rounded[key] = round(value, 4)
    return rounded


def _parse_symbols(value: str) -> list[str]:
    return [normalize_symbol(part) for part in value.replace("\n", ",").split(",") if part.strip()]


def _dedupe_symbols(symbols: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = symbol.strip().upper()
        if clean and clean not in seen:
            seen.add(clean)
            output.append(clean)
    return output


def _format_price(value: float | None) -> str:
    return "-" if value is None else f"${float(value):,.2f}"


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{float(value) * 100:.2f}%"


def _scanner_column_config() -> dict:
    return {
        "price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "change_pct": st.column_config.NumberColumn("Change", format="%.2f%%"),
        "volume_anomaly": st.column_config.NumberColumn("RVOL", format="%.2f"),
        "gap_pct": st.column_config.NumberColumn("Gap", format="%.2f%%"),
        "atr_pct": st.column_config.NumberColumn("ATR %", format="%.2f%%"),
        "confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
        "score": st.column_config.NumberColumn("Score", format="%.3f"),
        "risk_reward": st.column_config.NumberColumn("R/R", format="%.2f"),
        "rank_score": st.column_config.NumberColumn("Rank", format="%.3f"),
    }


def _model_column_config() -> dict:
    return {
        "expected_return": st.column_config.NumberColumn("Expected Return", format="%.2f%%"),
        "confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
        "predicted_price": st.column_config.NumberColumn("Predicted", format="$%.2f"),
        "lower_bound": st.column_config.NumberColumn("Lower", format="$%.2f"),
        "upper_bound": st.column_config.NumberColumn("Upper", format="$%.2f"),
    }


def _risk_column_config() -> dict:
    return {
        "entry": st.column_config.NumberColumn("Entry", format="$%.2f"),
        "stop_loss": st.column_config.NumberColumn("Stop", format="$%.2f"),
        "risk_reward": st.column_config.NumberColumn("R/R", format="%.2f"),
        "max_position_risk": st.column_config.NumberColumn("Max Risk", format="$%.2f"),
    }


def _context_column_config() -> dict:
    return {
        "context_confidence": st.column_config.NumberColumn("Context Confidence", format="%.1f%%"),
        "catalyst_score": st.column_config.NumberColumn("Catalyst Score", format="%.2f"),
        "catalyst_freshness": st.column_config.NumberColumn("Freshness", format="%.1f%%"),
        "market_alignment": st.column_config.NumberColumn("Market Alignment", format="%.2f"),
        "sector_alignment": st.column_config.NumberColumn("Sector Alignment", format="%.2f"),
    }


def _backtest_column_config() -> dict:
    return {
        "win_rate": st.column_config.NumberColumn("Win Rate", format="%.2f%%"),
        "average_return": st.column_config.NumberColumn("Avg Return", format="%.2f%%"),
        "max_drawdown": st.column_config.NumberColumn("Max Drawdown", format="%.2f%%"),
        "sharpe_like": st.column_config.NumberColumn("Sharpe-like", format="%.2f"),
        "no_trade_rate": st.column_config.NumberColumn("No Trade", format="%.2f%%"),
    }


def _trade_log_column_config() -> dict:
    return {
        "entry": st.column_config.NumberColumn("Entry", format="$%.2f"),
        "stop_loss": st.column_config.NumberColumn("Stop", format="$%.2f"),
        "target": st.column_config.NumberColumn("Target", format="$%.2f"),
        "exit_price": st.column_config.NumberColumn("Exit", format="$%.2f"),
        "return": st.column_config.NumberColumn("Return", format="%.2f%%"),
        "confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
        "score": st.column_config.NumberColumn("Score", format="%.3f"),
        "risk_reward": st.column_config.NumberColumn("R/R", format="%.2f"),
        "equity": st.column_config.NumberColumn("Equity", format="$%.2f"),
    }


def _percent_display(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    copy = df.copy()
    for column in columns:
        if column in copy.columns:
            copy[column] = copy[column].astype(float) * 100
    return copy


def _percent_context_metrics(metrics: dict) -> dict:
    converted = dict(metrics)
    for key in ["context_confidence", "catalyst_freshness"]:
        if converted.get(key) is not None:
            converted[key] = float(converted[key]) * 100
    return converted


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/default.yaml")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    main()
