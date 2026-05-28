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
from stockpredictor.journal import (
    append_journal_entry,
    delete_journal_entry,
    load_journal_entries,
    update_journal_entry,
)
from stockpredictor.news import build_news_feed
from stockpredictor.pipeline import analyze_symbol, scan_symbols
from stockpredictor.symbols import normalize_symbol, search_symbols
from stockpredictor.utils import clean_symbol_list, to_serializable


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

    scan_tab, deep_dive_tab, news_tab, backtest_tab, journal_tab, config_tab = st.tabs(
        ["Scanner", "Ticker Deep Dive", "News Feed", "Backtest", "Journal", "Config"]
    )
    with scan_tab:
        _render_scanner(settings, symbols)
    with deep_dive_tab:
        default_symbol = symbols[0] if symbols else settings.dashboard.get("default_symbol", "AAPL")
        col_sym, col_horizon = st.columns([3, 1])
        selected_symbol = col_sym.text_input("Symbol", value=str(default_symbol)).upper()
        horizon_options = list((settings.horizons.get("profiles") or {"swing": {}}).keys()) or ["swing"]
        default_horizon = settings.horizons.get("default", "swing")
        try:
            default_index = horizon_options.index(default_horizon)
        except ValueError:
            default_index = 0
        selected_horizon = col_horizon.selectbox("Horizon", horizon_options, index=default_index)
        if st.button("Analyze", type="primary"):
            _render_analysis(settings, selected_symbol, horizon=selected_horizon)
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
                trade_log_df = _percent_display(
                    pd.DataFrame(report.trade_log),
                    ["return", "trade_return", "confidence", "max_adverse_excursion", "max_favorable_excursion"],
                )
                st.dataframe(trade_log_df, use_container_width=True, hide_index=True, column_config=_trade_log_column_config())
    with journal_tab:
        _render_journal(settings, symbols)
    with config_tab:
        st.json(to_serializable(settings.raw))


def _render_scanner(settings, symbols: list[str]) -> None:
    st.subheader("Scanner")
    st.caption("Ranks the selected symbols by movement, volume, signal confidence, catalyst/risk flags, and setup quality.")
    if settings.raw.get("scanner", {}).get("intraday_provider_note", False):
        st.caption("Provider note: premarket high/low, spread, float, halt status, and time-of-day RVOL require a dedicated intraday scanner provider.")
    if st.button("Run Scan", type="primary"):
        results = scan_symbols(
            settings,
            symbols=symbols or None,
            max_symbols=int(settings.dashboard.get("max_scan_symbols", len(symbols) if symbols else len(settings.watchlist()))),
        )
        rows = [_rounded_row(result.scanner_row) for result in results]
        if not rows:
            st.info("No symbols selected.")
            return
        df = pd.DataFrame(rows)
        df = _render_scanner_filters(df, settings)
        if df.empty:
            st.info("No symbols passed the current scanner filters.")
            return
        actionable = int(df["action"].isin(["long", "short", "watch"]).sum()) if "action" in df else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scanned", len(df))
        c2.metric("Actionable / Watch", actionable)
        c3.metric("Avg RVOL", f"{df['volume_anomaly'].dropna().mean():.2f}" if "volume_anomaly" in df else "-")
        c4.metric("Top Rank", f"{df['rank_score'].max():.3f}" if "rank_score" in df else "-")
        st.dataframe(df, use_container_width=True, hide_index=True, column_config=_scanner_column_config())


def _render_analysis(settings, symbol: str, horizon: str | None = None) -> None:
    result = analyze_symbol(symbol, settings, horizon=horizon)
    provider = get_market_data_provider(settings)
    frame = fetch_market_data(symbol, settings, provider)

    session = result.session
    live_price = session.live_price if session is not None else None
    headline_price = live_price if live_price is not None else result.snapshot.latest_close
    headline_label = "Live Price" if live_price is not None else "Last Close"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Action", result.decision.action)
    col2.metric("Confidence", _format_percent(result.decision.confidence))
    col3.metric("Score", f"{result.decision.score:.3f}")
    col4.metric(
        headline_label,
        _format_price(headline_price),
        _format_percent(result.snapshot.change_pct),
    )

    if session is not None and session.data_available:
        st.caption(
            f"Session: {session.market_session} ({session.bars_loaded} intraday bars). "
            f"Horizon: {result.horizon}."
        )
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Session VWAP", _format_price(session.session_vwap) if session.session_vwap else "-")
        sc2.metric("Session Open", _format_price(session.session_open) if session.session_open else "-")
        sc3.metric("Session High", _format_price(session.session_high) if session.session_high else "-")
        sc4.metric("Session Low", _format_price(session.session_low) if session.session_low else "-")
        sc5.metric(
            "TOD RVOL",
            f"{session.time_of_day_rvol:.2f}" if session.time_of_day_rvol is not None else "-",
        )
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Premarket High", _format_price(session.premarket_high) if session.premarket_high else "-")
        pc2.metric("Premarket Low", _format_price(session.premarket_low) if session.premarket_low else "-")
        pc3.metric("Opening Range", session.opening_range_status)
    elif session is not None:
        st.caption(f"Session: {session.market_session}. Intraday bars unavailable — analysis is daily-only.")

    market_state = result.market_state
    sector = result.sector_context
    calendar = result.calendar
    if market_state is not None or sector is not None or calendar is not None:
        st.subheader("Context Cross-Check")
        mc1, mc2, mc3, mc4 = st.columns(4)
        if market_state is not None:
            mc1.metric("SPY", _format_percent(market_state.spy_change_pct) if market_state.spy_change_pct is not None else "-")
            mc2.metric("QQQ", _format_percent(market_state.qqq_change_pct) if market_state.qqq_change_pct is not None else "-")
            mc3.metric("VIX", f"{market_state.vix_value:.2f}" if market_state.vix_value is not None else "-")
        if sector is not None and sector.sector_etf:
            mc4.metric(
                f"{sector.sector_etf} ({sector.alignment})",
                _format_percent(sector.sector_change_pct) if sector.sector_change_pct is not None else "-",
            )
        if calendar is not None and calendar.no_trade_flags:
            st.warning("Calendar flags: " + "; ".join(calendar.no_trade_flags))

    if result.previous_snapshots:
        with st.expander(f"Compared to your last {len(result.previous_snapshots)} analyses"):
            from stockpredictor.snapshots import diff_snapshots

            prev = result.previous_snapshots[-1]
            if result.snapshot_record is not None:
                diff = diff_snapshots(result.snapshot_record, prev)
                dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                dcol1.metric("Δ Score", f"{diff['score_delta']:+.3f}" if diff.get("score_delta") is not None else "-")
                dcol2.metric("Δ Confidence", f"{diff['confidence_delta']:+.3f}" if diff.get("confidence_delta") is not None else "-")
                dcol3.metric("Δ Price", _format_price(diff["live_price_delta"]) if diff.get("live_price_delta") is not None else "-")
                dcol4.metric("Action Changed?", "yes" if diff.get("action_changed") else "no")
                st.caption(f"Previous action: {diff.get('previous_action', '-')} @ {diff.get('previous_timestamp', '-')}")

    st.subheader("Price And Levels")
    st.plotly_chart(
        _price_chart(frame, result.features.levels, result.risk_plan, ma_windows=settings.features.get("ma_windows", [9, 20, 50])),
        use_container_width=True,
    )

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
        pd.DataFrame([_risk_plan_row(result.risk_plan)]),
        use_container_width=True,
        hide_index=True,
        column_config=_risk_column_config(),
    )
    if result.risk_plan.session_checks:
        st.caption(_session_check_text(result.risk_plan.session_checks))
    if result.risk_plan.no_trade_reasons:
        st.warning("No-trade reasons: " + "; ".join(result.risk_plan.no_trade_reasons))

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
            _indicator_rows(to_serializable(result.features.indicators))
        ),
        use_container_width=True,
        hide_index=True,
    )


def _price_chart(frame: pd.DataFrame, levels: dict[str, float | None], risk_plan=None, ma_windows: list[int] | None = None):
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
    for window in [int(value) for value in (ma_windows or [])]:
        column = f"sma_{window}"
        if len(frame) >= window:
            fig.add_trace(go.Scatter(x=frame.index, y=frame["Close"].rolling(window).mean(), mode="lines", name=column.upper()), row=1, col=1)
    if "Volume" in frame:
        fig.add_trace(go.Bar(x=frame.index, y=frame["Volume"], name="Volume", marker_color="#8892a6"), row=2, col=1)
    chart_levels = dict(levels)
    for name, value in chart_levels.items():
        if value:
            fig.add_hline(y=float(value), annotation_text=f"{name}: {_format_price(float(value))}", line_dash="dot", opacity=0.55, row=1, col=1)
    if risk_plan is not None:
        if risk_plan.entry_zone:
            fig.add_hrect(
                y0=float(risk_plan.entry_zone[0]),
                y1=float(risk_plan.entry_zone[1]),
                fillcolor="#2f80ed",
                opacity=0.10,
                line_width=0,
                annotation_text="entry zone",
                row=1,
                col=1,
            )
        if risk_plan.entry:
            fig.add_hline(y=float(risk_plan.entry), annotation_text=f"entry: {_format_price(risk_plan.entry)}", line_color="#2f80ed", row=1, col=1)
        if risk_plan.stop_loss:
            fig.add_hline(y=float(risk_plan.stop_loss), annotation_text=f"stop: {_format_price(risk_plan.stop_loss)}", line_color="#d64545", row=1, col=1)
        for index, target in enumerate(risk_plan.targets, start=1):
            fig.add_hline(y=float(target), annotation_text=f"target {index}: {_format_price(target)}", line_color="#1f9d55", row=1, col=1)
    fig.update_layout(height=560, margin={"l": 20, "r": 20, "t": 10, "b": 20}, xaxis_rangeslider_visible=False)
    return fig


def _render_scanner_filters(df: pd.DataFrame, settings) -> pd.DataFrame:
    defaults = settings.raw.get("scanner", {}).get("default_filters", {})
    with st.expander("Scanner Filters", expanded=False):
        st.caption("Min abs change and Max ATR are in percent. Min RVOL is the volume ratio (1.0 = average).")
        c1, c2, c3, c4 = st.columns(4)
        min_abs_change = c1.slider("Min abs change", 0.0, 20.0, float(defaults.get("min_abs_change_pct", 0.0)) * 100, 0.5, format="%.1f%%")
        min_rvol = c2.slider("Min RVOL", 0.0, 10.0, float(defaults.get("min_volume_anomaly", 0.0)), 0.25)
        max_atr = c3.slider("Max ATR", 0.0, 50.0, float(defaults.get("max_atr_pct", 0.50)) * 100, 1.0, format="%.0f%%")
        actions = sorted(df["action"].dropna().unique().tolist()) if "action" in df else []
        selected_actions = c4.multiselect("Actions", actions, default=actions)
    filtered = df.copy()
    if "change_pct" in filtered:
        filtered = filtered[filtered["change_pct"].abs() >= min_abs_change]
    if "volume_anomaly" in filtered:
        filtered = filtered[filtered["volume_anomaly"].fillna(0) >= min_rvol]
    if "atr_pct" in filtered:
        filtered = filtered[filtered["atr_pct"].fillna(0) <= max_atr]
    if selected_actions and "action" in filtered:
        filtered = filtered[filtered["action"].isin(selected_actions)]
    return filtered


def _risk_plan_row(plan) -> dict:
    return {
        "entry_zone": _format_price_range(plan.entry_zone),
        "entry": plan.entry,
        "stop_loss": plan.stop_loss,
        "stop_source": plan.stop_source or "-",
        "targets": _format_targets(plan.targets),
        "target_source": plan.target_source or "-",
        "risk_reward": plan.risk_reward,
        "risk_per_share": plan.risk_per_share,
        "max_position_risk": plan.max_position_risk,
        "planned_risk": plan.planned_risk,
        "planned_position_value": plan.planned_position_value,
        "position_size": plan.position_size,
        "liquidity_ok": plan.liquidity_ok,
        "setup_quality": plan.setup_quality,
        "invalidation": plan.invalidation,
    }


def _session_check_text(checks: dict) -> str:
    max_daily_loss = _format_price(float(checks.get("max_daily_loss", 0)))
    max_trades = checks.get("max_trades_per_day", "-")
    stop_losses = checks.get("stop_after_consecutive_losses", "-")
    pdt = "PDT warning active" if checks.get("pdt_warning") else "PDT equity check passed or disabled"
    return f"Session controls: max daily loss {max_daily_loss}, max trades {max_trades}, stop after {stop_losses} consecutive losses. {pdt}."


def _indicator_rows(indicators: dict) -> list[dict[str, str]]:
    rows = []
    percent_keys = {"price_change_pct", "range_pct", "atr_pct", "gap_pct", "volatility"}
    price_keys = {
        "vwap",
        "sma_9",
        "sma_20",
        "sma_50",
        "support",
        "resistance",
        "session_open",
        "session_high",
        "session_low",
        "prior_high",
        "prior_low",
        "prior_close",
        "opening_range_high",
        "opening_range_low",
    }
    for key, value in indicators.items():
        if value is None:
            display = "-"
        elif key in percent_keys:
            display = _format_percent(float(value))
        elif key in price_keys:
            display = _format_price(float(value))
        elif isinstance(value, float):
            display = f"{value:,.3f}"
        else:
            display = str(value)
        rows.append({"name": key, "value": display})
    return rows


def _format_price_range(value: tuple[float, float] | list[float] | None) -> str:
    if not value:
        return "-"
    return f"{_format_price(float(value[0]))} to {_format_price(float(value[1]))}"


def _format_targets(values: list[float]) -> str:
    return ", ".join(_format_price(float(value)) for value in values) if values else "-"


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
    symbols = clean_symbol_list(_parse_symbols(st.session_state.selected_symbols_text))
    st.sidebar.caption(f"{len(symbols)} selected: {', '.join(symbols[:8])}{'...' if len(symbols) > 8 else ''}")
    return symbols


def _append_symbol_to_state(symbol: str) -> None:
    symbols = clean_symbol_list(_parse_symbols(st.session_state.get("selected_symbols_text", "")) + [symbol])
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
    with st.expander(f"{summary.get('source_count', 0)} linked sources"):
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


_JOURNAL_SETUPS = ["opening_range_break", "vwap_reclaim", "vwap_loss", "trend_pullback", "gap_and_go", "reversal", "news_catalyst", "other"]
_JOURNAL_ACTIONS = ["long", "short", "watch", "no_trade", "low_confidence"]
_JOURNAL_OUTCOMES = ["open", "win", "loss", "breakeven", "skipped"]
_JOURNAL_STATES = ["calm", "neutral", "hesitant", "rushed", "revenge", "overconfident"]


def _render_journal(settings, symbols: list[str]) -> None:
    st.subheader("Trade Review Journal")
    st.caption("Local JSONL journal for reviewing setup quality, risk discipline, and outcome. This stays out of Git by default.")
    recent = load_journal_entries(settings, limit=100)
    with st.form("journal_form"):
        c1, c2, c3, c4 = st.columns(4)
        symbol = c1.text_input("Symbol", value=(symbols[0] if symbols else str(settings.dashboard.get("default_symbol", "AAPL")))).upper()
        action = c2.selectbox("Action", _JOURNAL_ACTIONS)
        setup_type = c3.selectbox("Setup", _JOURNAL_SETUPS)
        outcome = c4.selectbox("Outcome", _JOURNAL_OUTCOMES)
        c5, c6, c7, c8 = st.columns(4)
        followed_plan = c5.checkbox("Followed plan")
        risk_respected = c6.checkbox("Risk respected")
        entry_quality = c7.slider("Entry quality", 1, 5, 3)
        exit_quality = c8.slider("Exit quality", 1, 5, 3)
        emotional_state = st.selectbox("State", _JOURNAL_STATES)
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Save Review")
    if submitted:
        record = append_journal_entry(
            settings,
            {
                "symbol": symbol,
                "action": action,
                "setup_type": setup_type,
                "followed_plan": followed_plan,
                "risk_respected": risk_respected,
                "entry_quality": entry_quality,
                "exit_quality": exit_quality,
                "emotional_state": emotional_state,
                "outcome": outcome,
                "notes": notes,
            },
        )
        st.success(f"Saved journal entry for {record['symbol']}.")
        recent = load_journal_entries(settings, limit=100)
    if recent:
        df = pd.DataFrame(recent)
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
        _render_journal_edit_controls(settings, recent)
    else:
        st.info("No journal entries yet.")


def _render_journal_edit_controls(settings, recent: list[dict]) -> None:
    with st.expander("Edit or delete an entry"):
        rows_with_ids = [row for row in recent if row.get("id")]
        if not rows_with_ids:
            st.caption("Older entries do not have IDs yet. Save a new entry to start tracking edits.")
            return
        options = {f"{row['id'][:8]}  {row['timestamp']}  {row['symbol']}  ({row.get('action', '')})": row for row in rows_with_ids}
        selected_label = st.selectbox("Entry", list(options.keys()))
        selected = options[selected_label]
        with st.form(f"journal_edit_{selected['id']}"):
            updates_symbol = st.text_input("Symbol", value=selected.get("symbol", "")).upper()
            updates_action = st.selectbox(
                "Action",
                _JOURNAL_ACTIONS,
                index=_safe_index(_JOURNAL_ACTIONS, selected.get("action", "watch")),
            )
            updates_outcome = st.selectbox(
                "Outcome",
                _JOURNAL_OUTCOMES,
                index=_safe_index(_JOURNAL_OUTCOMES, selected.get("outcome", "open")),
            )
            updates_notes = st.text_area("Notes", value=str(selected.get("notes", "")))
            col_save, col_delete = st.columns(2)
            save = col_save.form_submit_button("Save edit")
            delete = col_delete.form_submit_button("Delete entry")
        if save:
            updated = update_journal_entry(
                settings,
                selected["id"],
                {
                    "symbol": updates_symbol,
                    "action": updates_action,
                    "outcome": updates_outcome,
                    "notes": updates_notes,
                },
            )
            if updated:
                st.success(f"Updated entry {selected['id'][:8]}.")
            else:
                st.error("Entry not found; reload the dashboard.")
        if delete:
            if delete_journal_entry(settings, selected["id"]):
                st.success(f"Deleted entry {selected['id'][:8]}.")
            else:
                st.error("Entry not found; reload the dashboard.")


def _safe_index(options: list[str], value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _rounded_row(row: dict) -> dict:
    rounded = dict(row)
    percent_keys = {
        "change_pct",
        "gap_pct",
        "atr_pct",
        "confidence",
        "extension_from_vwap_pct",
        "distance_to_support_pct",
        "distance_to_resistance_pct",
        "benchmark_change_pct",
        "relative_strength_pct",
    }
    for key in [
        "price",
        "change_pct",
        "volume_anomaly",
        "gap_pct",
        "atr_pct",
        "confidence",
        "score",
        "risk_reward",
        "rank_score",
        "prior_high",
        "prior_low",
        "session_open",
        "opening_range_high",
        "opening_range_low",
        "extension_from_vwap_pct",
        "distance_to_support_pct",
        "distance_to_resistance_pct",
        "benchmark_change_pct",
        "relative_strength_pct",
    ]:
        if rounded.get(key) is not None:
            value = float(rounded[key])
            if key in percent_keys:
                value *= 100
            rounded[key] = round(value, 4)
    return rounded


def _parse_symbols(value: str) -> list[str]:
    return [normalize_symbol(part) for part in value.replace("\n", ",").split(",") if part.strip()]

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
        "prior_high": st.column_config.NumberColumn("Prior High", format="$%.2f"),
        "prior_low": st.column_config.NumberColumn("Prior Low", format="$%.2f"),
        "session_open": st.column_config.NumberColumn("Open", format="$%.2f"),
        "opening_range_high": st.column_config.NumberColumn("OR High", format="$%.2f"),
        "opening_range_low": st.column_config.NumberColumn("OR Low", format="$%.2f"),
        "extension_from_vwap_pct": st.column_config.NumberColumn("VWAP Dist", format="%.2f%%"),
        "distance_to_support_pct": st.column_config.NumberColumn("Support Dist", format="%.2f%%"),
        "distance_to_resistance_pct": st.column_config.NumberColumn("Resistance Dist", format="%.2f%%"),
        "benchmark_change_pct": st.column_config.NumberColumn("Benchmark", format="%.2f%%"),
        "relative_strength_pct": st.column_config.NumberColumn("Rel Strength", format="%.2f%%"),
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
        "stop_source": st.column_config.TextColumn("Stop anchor"),
        "target_source": st.column_config.TextColumn("Target anchor"),
        "risk_reward": st.column_config.NumberColumn("R/R", format="%.2f"),
        "risk_per_share": st.column_config.NumberColumn("Risk/Share", format="$%.2f"),
        "planned_risk": st.column_config.NumberColumn("Planned Risk", format="$%.2f"),
        "planned_position_value": st.column_config.NumberColumn("Position Value", format="$%.2f"),
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
        "trade_return": st.column_config.NumberColumn("Trade Return", format="%.2f%%"),
        "r_multiple": st.column_config.NumberColumn("R", format="%.2f"),
        "max_adverse_excursion": st.column_config.NumberColumn("MAE", format="%.2f%%"),
        "max_favorable_excursion": st.column_config.NumberColumn("MFE", format="%.2f%%"),
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
