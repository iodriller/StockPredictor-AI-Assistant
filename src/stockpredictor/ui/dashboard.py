from __future__ import annotations

import argparse
import re
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from stockpredictor.backtesting import run_backtest
from copy import deepcopy

from stockpredictor import education as edu
from stockpredictor.config import ConfigError, Settings, load_settings
from stockpredictor.data import fetch_market_data, get_market_data_provider
from stockpredictor.dashboard_cache import CACHE_KEYS, load_dashboard_cache, save_dashboard_cache
from stockpredictor.journal import (
    append_journal_entry,
    delete_journal_entry,
    load_journal_entries,
    update_journal_entry,
)
from stockpredictor.models.registry import MODEL_REGISTRY
from stockpredictor.news import NewsAnalysisError, build_news_feed
from stockpredictor.pipeline import analyze_symbol, scan_symbols
from stockpredictor.symbols import normalize_symbol, search_symbols
from stockpredictor.utils import clean_symbol_list, to_float, to_serializable


_HELP_BASE = {
    "action": "Entry readiness after signal, calendar, and risk checks. This is separate from directional bias: a stock can be bullish while a fresh entry is blocked or waiting for the market to open.",
    "analysis_provider": "Shows whether news summaries came from the configured LLM, heuristic fallback, or heuristic-only mode.",
    "article_excerpts": "Short excerpts fetched from source links and passed to the LLM. This improves context but is not a full professional news feed.",
    "atr_pct": "Average True Range as a percent of price. Higher ATR means wider stops and more risk per share.",
    "avg_rvol": "Average relative volume for the scanned rows. Above 1.0 means volume is higher than recent average.",
    "backtest_average_return": "Average return per simulated trade/evaluation. This is a logic sanity check, not proof of future performance.",
    "backtest_no_trade_rate": "How often the strategy skipped instead of forcing a trade. A healthy strategy often has a meaningful no-trade rate.",
    "backtest_sharpe_like": "Simple return-to-volatility metric. Useful for comparing runs, but unstable on small samples.",
    "backtest_win_rate": "Percent of simulated trades that were profitable. Needs to be read with average win/loss and drawdown.",
    "bias": "Directional lean after evidence fusion: bullish, bearish, or neutral. Bias is separate from execution readiness.",
    "catalyst_score": "How strongly the context/news pipeline sees a tradable catalyst. Higher is better, but price and volume still need to confirm.",
    "confidence": "How much agreement/strength the fused signal has. Low confidence usually means wait or reduce attention.",
    "context_confidence": "How complete and usable the contextual evidence is. Low context confidence means news/catalyst evidence is thin.",
    "drawdown": "Largest peak-to-trough equity loss during the simulation. This is one of the main stress checks.",
    "entry": "A suggested trigger or reference price. It is not a market order instruction.",
    "entry_zone": "A price area where the setup is considered valid. Chasing outside the zone can ruin risk/reward.",
    "freshness": "How recent the catalyst evidence is. Fresh items matter more for day trading than stale headlines.",
    "gap_pct": "Open versus prior close. Gaps can signal catalysts, but large gaps can also be late or risky.",
    "headline_relevance": "A score for how relevant a headline looks to active trading based on category, impact, and freshness.",
    "impact": "Estimated directional headline impact. Positive is bullish, negative is bearish, near zero is neutral/noisy.",
    "invalidation": "The condition that says the trade idea is wrong. Define this before entry.",
    "mae": "Maximum adverse excursion: worst unrealized move against the trade during the simulation.",
    "mfe": "Maximum favorable excursion: best unrealized move in favor of the trade during the simulation.",
    "news_sources": "Number of linked source items used for this symbol summary. Click the source expander to inspect links.",
    "position": "Estimated share count from risk sizing. Blank means no actionable trade plan was generated.",
    "rank": "Scanner priority score combining movement, volume, confidence, catalysts, and setup quality.",
    "relative_strength": "Symbol performance versus the configured benchmark. Positive means it is outperforming the benchmark.",
    "risk_reward": "Potential reward divided by risk. Below the configured minimum should usually block the trade.",
    "score": "Signed fused score. Positive favors long, negative favors short, near zero means mixed/noisy.",
    "sentiment": "News/context tone. Use it as supporting evidence, not a standalone trade signal.",
    "setup": "Quality label for the current trade structure after risk, liquidity, and signal checks.",
    "stop": "Planned exit if the setup fails. The stop should be tied to structure, VWAP, ATR, or invalidation.",
    "target": "Planned reward area. Targets should be realistic relative to nearby resistance/support and volatility.",
    "top_reason": "The strongest reason behind the current signal. Read with skip reasons before acting.",
    "trade_watch": "Rows that are actionable or worth monitoring. This excludes low-confidence/no-trade rows.",
    "vwap": "Volume-weighted price reference. Current-session VWAP is the intraday fair-value line; the daily chart uses a rolling volume-weighted average because daily bars cannot reproduce true session VWAP.",
    "vwap_alignment": "Whether price is above or below VWAP. Above often supports longs; below often warns against longs.",
}


# Education-mode state. When on, every tooltip is enriched with a "how traders use
# it" line, and the workspaces show beginner guides + a glossary. This module-level
# flag is set once per render in main(), so the wrapper below needs no plumbing.
_EDU_STATE = {"on": False}


def _education_on() -> bool:
    return bool(_EDU_STATE.get("on"))


class _EducationalHelp:
    """Drop-in replacement for the HELP_TEXT dict whose lookups gain a trader-usage
    line when education mode is on — so every existing `help=HELP_TEXT[...]` call
    becomes beginner-friendly with no change at the call site."""

    def __init__(self, base: dict[str, str]) -> None:
        self._base = base

    def __getitem__(self, key: str) -> str:
        return edu.enrich_help(key, self._base[key], _education_on())

    def get(self, key: str, default: str = "") -> str:
        base = self._base.get(key, default)
        return edu.enrich_help(key, base, _education_on()) if base else base

    def __contains__(self, key: str) -> bool:
        return key in self._base

    def __iter__(self):
        return iter(self._base)

    def keys(self):
        return self._base.keys()


HELP_TEXT = _EducationalHelp(_HELP_BASE)

_ACTION_PRESENTATION = {
    "long": ("🟢 LONG", "Bullish setup passed the signal gates. Review the entry, stop, target, and invalidation before acting."),
    "short": ("🔴 SHORT", "Bearish setup passed the signal gates. Review borrow availability, entry, stop, target, and invalidation before acting."),
    "watch": ("🔵 WATCH", "There is a directional edge, but it has not cleared the trade threshold. Keep it on the screen and wait for confirmation."),
    "low_confidence": ("🟡 LOW CONFIDENCE", "The inputs are mixed or weak. Do not trade until confidence improves."),
    "no_trade": ("⚪ NO TRADE", "The current setup does not justify a new position. Sitting out is the decision."),
}

_BIAS_PRESENTATION = {
    "bullish": "BULLISH",
    "bearish": "BEARISH",
    "neutral": "NEUTRAL",
}


def _remembered_expander(label: str, state_key: str, expanded: bool = False, container=None):
    container = container or st
    expander_key = f"expander_{_slug_key(state_key)}"
    return container.expander(
        label,
        expanded=expanded,
        key=expander_key,
        on_change="ignore",
    )


def _install_expander_memory() -> None:
    st.iframe(
        """
        <script>
        (() => {
          const storageKey = "stockpredictor.expander-state.v2";
          const doc = window.parent.document;
          const storage = window.parent.localStorage;
          const loadState = () => {
            try { return JSON.parse(storage.getItem(storageKey) || "{}"); }
            catch (_) { return {}; }
          };
          const saveState = (state) => storage.setItem(storageKey, JSON.stringify(state));
          const bind = () => {
            const state = loadState();
            doc.querySelectorAll('[class*="st-key-expander_"]').forEach((wrapper) => {
              const details = wrapper.matches("details") ? wrapper : wrapper.querySelector("details");
              if (!details) return;
              // Only ever touch an expander once. Re-applying details.open on every
              // DOM mutation fights Streamlit's own re-render and causes flicker
              // (most visible on heavy panels like the glossary).
              if (details.dataset.stockpredictorMemoryBound === "true") return;
              const panelClass = [...wrapper.classList].find((name) => name.startsWith("st-key-expander_"));
              if (!panelClass) return;
              details.dataset.stockpredictorMemoryBound = "true";
              if (Object.prototype.hasOwnProperty.call(state, panelClass)) {
                details.open = Boolean(state[panelClass]);
              }
              details.addEventListener("toggle", () => {
                const latest = loadState();
                latest[panelClass] = details.open;
                saveState(latest);
              });
            });
          };
          bind();
          // Debounce so a burst of mutations only triggers one (cheap) pass that
          // binds newly added expanders, rather than thrashing on every change.
          let scheduled = false;
          const observer = new MutationObserver(() => {
            if (scheduled) return;
            scheduled = true;
            window.parent.requestAnimationFrame(() => { scheduled = false; bind(); });
          });
          observer.observe(doc.body, { childList: true, subtree: true });
        })();
        </script>
        """,
        height=1,
        width=1,
        tab_index=-1,
    )


def _slug_key(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value).lower()).strip("_") or "panel"


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="StockPredictor", layout="wide")
    st.title("StockPredictor Trading Intelligence")
    st.caption("Research and decision support only. No brokerage execution.")

    with _remembered_expander("Advanced setup", "sidebar_advanced_setup", expanded=False, container=st.sidebar):
        config_path = st.text_input("Config file", value=args.config, help="YAML config controlling providers, models, risk limits, watchlists, context, and dashboard defaults.")
    try:
        settings = load_settings(config_path)
    except ConfigError as exc:
        st.error(str(exc))
        return

    _render_education_toggle(settings)
    symbols = _render_symbol_sidebar(settings)
    _initialize_dashboard_cache(settings)
    workspace = st.segmented_control(
        "Workspace",
        options=["Scanner", "Trade Plan", "News", "Backtest", "Journal", "Settings"],
        default="Scanner",
        key="dashboard_workspace",
        help="Choose one workflow. Only the active workspace runs, which keeps interactions responsive.",
    ) or "Scanner"
    workspace_guide_keys = {
        "Scanner": "scanner",
        "Trade Plan": "trade_plan",
        "News": "news",
        "Backtest": "backtest",
        "Journal": "journal",
        "Settings": "settings",
    }
    _render_education_guide(workspace_guide_keys.get(workspace, ""))
    if workspace == "Scanner":
        _render_scanner(settings, symbols)
    elif workspace == "Trade Plan":
        _render_trade_plan_workspace(settings, symbols)
    elif workspace == "News":
        _render_news(settings, symbols or settings.watchlist())
    elif workspace == "Backtest":
        _render_backtest(settings, symbols)
    elif workspace == "Journal":
        _render_journal(settings, symbols)
    elif workspace == "Settings":
        _render_settings_tab(settings, symbols)
    _render_education_glossary()
    _install_expander_memory()


def _render_education_toggle(settings) -> None:
    """Sidebar switch for Education mode. Defaults to the config value, then the
    user can flip it live; the choice drives every tooltip and guide in the app."""
    default_on = bool(settings.dashboard.get("education_mode", True))
    if "education_mode" not in st.session_state:
        st.session_state.education_mode = default_on
    st.sidebar.toggle(
        "📚 Education mode",
        key="education_mode",
        help="Beginner-friendly mode: adds plain-language explanations, a glossary of every term, per-screen guides, and 'how traders use this' notes on every tooltip.",
    )
    _EDU_STATE["on"] = bool(st.session_state.education_mode)
    if _EDU_STATE["on"]:
        st.sidebar.caption("Learning aids are ON. Hover any ⓘ for a plain-language note.")


def _render_education_guide(guide_key: str) -> None:
    if not _education_on() or guide_key not in edu.WORKSPACE_GUIDES:
        return
    guide = edu.WORKSPACE_GUIDES[guide_key]
    with st.container(border=True):
        st.markdown(f"#### 📚 {guide['title']}")
        st.write(guide["what"])
        if guide.get("look_at"):
            st.markdown("**What to look at:** " + " · ".join(guide["look_at"]))
        if guide.get("steps"):
            st.markdown("**Steps:**")
            for index, step in enumerate(guide["steps"], start=1):
                st.markdown(f"{index}. {step}")


def _render_education_glossary() -> None:
    if not _education_on():
        return
    with _remembered_expander("📖 Glossary — every term in one place", "education_glossary", expanded=False):
        st.caption("Plain-language one-liners for every symbol, acronym, and metric in the app.")
        df = pd.DataFrame(edu.glossary_groups(), columns=["term", "meaning"])
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "term": st.column_config.TextColumn("Term", width="small"),
                "meaning": st.column_config.TextColumn("What it means (and how it's used)", width="large"),
            },
        )


def _render_trade_plan_workspace(settings, symbols: list[str]) -> None:
    default_symbol = symbols[0] if symbols else settings.dashboard.get("default_symbol", "AAPL")
    col_sym, col_horizon = st.columns([2, 3])
    selected_symbol = col_sym.text_input("Symbol", value=str(default_symbol), help="Ticker to analyze. Use the sidebar search if you do not know the symbol.").upper()
    horizon_options = list((settings.horizons.get("profiles") or {"swing": {}}).keys()) or ["swing"]
    default_horizon = settings.horizons.get("default", "swing")
    selected_horizon = _render_horizon_selector(col_horizon, horizon_options, str(default_horizon))
    if _education_on() and selected_horizon in edu.HORIZON_GUIDES:
        st.caption(f"📚 {edu.HORIZON_GUIDES[selected_horizon]}")
    default_news_limit = int(settings.context_agent.get("news_analysis", {}).get("max_headlines_per_symbol", 50))
    news_limit = st.slider(
        "Headlines to analyze",
        min_value=5,
        max_value=200,
        value=min(max(default_news_limit, 5), 200),
        step=5,
        help="How many headlines to gather and feed into this symbol's news analysis and the decision. Free providers may return fewer.",
    )
    tuned_settings = _render_tuning_controls(settings, selected_horizon)
    analyze_requested = st.button("Analyze", type="primary", help="Run the full signal, context, risk, and chart analysis for this symbol and horizon.")
    if analyze_requested:
        _render_analysis(tuned_settings, selected_symbol, horizon=selected_horizon, news_limit=news_limit, refresh=True)
    elif st.session_state.get("latest_analysis") is not None:
        _render_analysis(tuned_settings, selected_symbol, refresh=False)


def _initialize_dashboard_cache(settings) -> None:
    cache_config = str(settings.path)
    if st.session_state.get("_dashboard_cache_config") == cache_config:
        return
    for key in CACHE_KEYS:
        st.session_state.pop(key, None)
    for key, value in load_dashboard_cache(settings).items():
        st.session_state[key] = value
    st.session_state["_dashboard_cache_config"] = cache_config


def _persist_dashboard_cache(settings) -> None:
    save_dashboard_cache(settings, {key: st.session_state.get(key) for key in CACHE_KEYS if key in st.session_state})


def _render_scanner(settings, symbols: list[str]) -> None:
    st.subheader("Scanner")
    with _remembered_expander("How to read the scanner", "scanner_help", expanded=False):
        st.write(
            "Start with Rank, Action, RVOL, Gap, and Relative Strength. "
            "A good row should have movement, liquidity, a clear reason, and acceptable risk. "
            "Low-confidence and no-trade rows are useful because they tell you what to ignore."
        )
    if settings.raw.get("scanner", {}).get("intraday_provider_note", False):
        st.caption("Provider note: premarket high/low, spread, float, halt status, and time-of-day RVOL require a dedicated intraday scanner provider.")
    if st.button("Scan Selected Symbols", type="primary"):
        progress = st.progress(0, text="Starting scanner")
        status = st.empty()

        def update_progress(value: float, message: str) -> None:
            status.caption(message)
            progress.progress(float(value), text=message)

        results = scan_symbols(
            settings,
            symbols=symbols or None,
            max_symbols=int(settings.dashboard.get("max_scan_symbols", len(symbols) if symbols else len(settings.watchlist()))),
            progress_callback=update_progress,
        )
        progress.progress(1.0, text="Scanner ready")
        st.session_state.latest_scan_results = results
        st.session_state.latest_scan_symbols = [result.snapshot.symbol for result in results]
        _persist_dashboard_cache(settings)
    results = st.session_state.get("latest_scan_results")
    if results is None:
        return
    scanned_symbols = st.session_state.get("latest_scan_symbols", [])
    if scanned_symbols:
        st.caption(f"Showing last scan: {', '.join(scanned_symbols)}")
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
    c1.metric("Scanned", len(df), help="Number of rows remaining after filters.")
    c2.metric("Trade / Watch", actionable, help=HELP_TEXT["trade_watch"])
    c3.metric("Avg RVOL", f"{df['volume_anomaly'].dropna().mean():.2f}" if "volume_anomaly" in df else "-", help=HELP_TEXT["avg_rvol"])
    c4.metric("Top Rank", f"{df['rank_score'].max():.3f}" if "rank_score" in df else "-", help=HELP_TEXT["rank"])
    st.dataframe(
        _scanner_summary_df(df),
        width="stretch",
        hide_index=True,
        column_config=_scanner_column_config(compact=True),
    )
    _render_scanner_detail(df)


_MODEL_NOTES = {
    "momentum": "Feature-aware: scores moving-average alignment, mid-MA slope, and recent realized momentum. Reads what the chart shows; best at catching a fresh turn.",
    "baseline": "Recency-weighted linear trend fit. Captures the prevailing direction; weighting keeps a fresh reversal from being buried by stale history.",
    "gaussian_process": "Smoothed drift from recent log-returns with an uncertainty band. A confirmation/uncertainty estimate, not a precise price target.",
    "arima": "Univariate time-series forecast on close. Often near a random walk on liquid names, so it tends to be a low-confidence confirmation.",
}

_MODELS_DISCLAIMER = (
    "All models are price/technical only — they do not read fundamentals (valuation, "
    "earnings growth, balance sheet). Use them as one input alongside the chart, news, "
    "and risk plan, not as a standalone buy/sell call."
)

_SIGNAL_BLEND_PRESETS = {
    "Custom": {
        "description": "Shape the evidence blend directly. Use the chart to see whether your thesis is balanced or concentrated before you interpret the signal.",
        "weights": None,
    },
    "Balanced confirmation": {
        "description": "Use several independent confirmations. Best default when no single catalyst dominates.",
        "weights": None,
    },
    "Price action": {
        "description": "Prioritize chart structure, VWAP, and intraday behavior. Useful for liquid names moving without a major headline.",
        "weights": {"models": 0.15, "technicals": 0.40, "intraday": 0.30, "context": 0.12, "sentiment": 0.03},
    },
    "Catalyst-led": {
        "description": "Prioritize fresh news while still requiring price confirmation. Useful for earnings, filings, analyst actions, and event-driven moves.",
        "weights": {"models": 0.10, "technicals": 0.15, "intraday": 0.10, "context": 0.52, "sentiment": 0.13},
    },
    "Trend / swing": {
        "description": "Prioritize multi-day model and chart trend. Useful when the thesis is continuation rather than a same-day event.",
        "weights": {"models": 0.45, "technicals": 0.35, "intraday": 0.05, "context": 0.12, "sentiment": 0.03},
    },
    "News thesis stress test": {
        "description": "Use only the News/AI read. This is an explainability experiment, not the recommended live-trading default.",
        "weights": {"models": 0.0, "technicals": 0.0, "intraday": 0.0, "context": 0.80, "sentiment": 0.20},
    },
}

_SIGNAL_COMPONENT_HELP = {
    "models": "Statistical price forecasts. Traders use these as directional context, not as an entry trigger.",
    "technicals": "Trend, moving averages, RSI, MACD, VWAP, and levels. Traders use these to judge structure and timing.",
    "intraday": "Session behavior such as opening range and live VWAP alignment. Most important for day trades.",
    "context": "Catalysts, filings, macro, sector, and the AI news synthesis. Important when the move is event-driven.",
    "sentiment": "Directional tone extracted from the news. Supporting evidence; usually smaller than catalyst context.",
}

_SIGNAL_COMPONENT_LABELS = {
    "models": "Price models",
    "technicals": "Technicals",
    "intraday": "Intraday tape",
    "context": "News context",
    "sentiment": "News tone",
}

_SIGNAL_COMPONENT_KEYS = tuple(_SIGNAL_COMPONENT_LABELS)

_BACKTEST_DEPTH_PRESETS = {
    "Quick check": {
        "description": "Fast smoke test for one or two symbols. Useful after changing settings; too small for strategy conclusions.",
        "period": "6mo",
        "lookback_rows": 60,
        "holding_period_days": 5,
        "evaluation_step_days": 10,
    },
    "Standard review": {
        "description": "Recommended first review. Uses one year of daily bars and checks for a new setup once per trading week.",
        "period": "1y",
        "lookback_rows": 90,
        "holding_period_days": 5,
        "evaluation_step_days": 5,
    },
    "Deeper sample": {
        "description": "Broader two-year review. Slower, but more useful when the standard run produces too few trades.",
        "period": "2y",
        "lookback_rows": 120,
        "holding_period_days": 5,
        "evaluation_step_days": 5,
    },
}

_CHART_RANGE_BARS = {
    "30 bars": 30,
    "60 bars": 60,
    "Full history": None,
}

_CHART_LEVEL_LABELS = {
    "support": "rolling support",
    "resistance": "rolling resistance",
    "vwap": "rolling volume-weighted avg",
    "sma_20": "SMA 20",
    "sma_50": "SMA 50",
    "prior_high": "prior high",
    "prior_low": "prior low",
    "session_open": "daily open",
    "opening_range_high": "opening range high",
    "opening_range_low": "opening range low",
}


def _render_tuning_controls(settings, horizon: str) -> "Settings":
    """Expose the model/signal/LLM parameters as live controls so a trader can tune
    the analysis without editing the config file. Returns a Settings with all the
    overrides applied. Rendered as sibling expanders (never nested)."""
    new_raw = deepcopy(settings.raw)
    base_weights = _resolve_active_weights(settings, horizon)

    overrides_weights = _render_trust_and_thresholds(settings, base_weights, new_raw)
    _apply_weight_overrides(new_raw, horizon, overrides_weights)
    _render_model_controls(settings, new_raw)
    _render_llm_controls(settings, new_raw)
    return Settings(raw=new_raw, path=settings.path)


def _render_trust_and_thresholds(settings, base_weights: dict, new_raw: dict) -> dict:
    """Choose a trader-oriented signal blend and decision thresholds."""
    with _remembered_expander("Signal lens & decision thresholds", "deepdive_trust", expanded=True):
        st.caption("Choose the evidence mix for the setup you are researching. The blend changes signal interpretation; entry readiness still checks price, liquidity, levels, and risk.")
        preset = st.selectbox(
            "Trading lens",
            options=list(_SIGNAL_BLEND_PRESETS),
            index=0,
            help="Custom starts from the configured horizon weights. Named lenses are quick resets for common research styles; the news-only stress test is for inspecting the AI thesis in isolation.",
        )
        weights = _signal_blend_weights(base_weights, preset)
        st.caption(str(_SIGNAL_BLEND_PRESETS[preset]["description"]))
        controls_col, radar_col = st.columns([3, 2])
        with controls_col:
            if preset == "Custom":
                st.caption("Drag any slider to shape the blend. The other sliders rebalance automatically so the allocation always totals 100%.")
                _ensure_signal_blend_slider_state(weights)
                custom = {}
                for key in _SIGNAL_COMPONENT_KEYS:
                    custom[key] = st.slider(
                        _SIGNAL_COMPONENT_LABELS[key],
                        min_value=0,
                        max_value=100,
                        step=1,
                        format="%d%%",
                        help=_SIGNAL_COMPONENT_HELP[key],
                        key=_signal_blend_widget_key(key),
                        on_change=_rebalance_signal_blend_slider_state,
                        args=(key,),
                    ) / 100.0
                weights = custom
                st.caption(f"Evidence allocation: {sum(round(value * 100) for value in custom.values())}% / 100%.")
            else:
                st.caption("Preset applied. Switch to Custom to tune each evidence input directly.")
        with radar_col:
            st.plotly_chart(
                _signal_blend_radar(weights),
                width="stretch",
                config={"displayModeBar": False},
            )
        st.dataframe(
            pd.DataFrame(_signal_blend_rows(weights)),
            width="stretch",
            hide_index=True,
            column_config={
                "input": st.column_config.TextColumn("Evidence input"),
                "weight": st.column_config.NumberColumn("Influence", format="%.0f%%"),
                "trader_use": st.column_config.TextColumn("How traders use it"),
            },
        )

        ma_default = ",".join(str(int(value)) for value in settings.features.get("ma_windows", [9, 20, 50]))
        ma_text = st.text_input("Moving-average windows", value=ma_default, help="Comma-separated lookbacks for the trend/MA features and chart, e.g. 9,20,50 (fast, medium, slow).")

        thresholds = settings.signal_fusion.get("thresholds", {})
        tc = st.columns(4)
        long_score = tc[0].slider("Trade score", 0.10, 0.80, float(thresholds.get("long_score", 0.30)), 0.01, help="Score magnitude needed to call a directional (long/short) trade. Lower = more trades, lower selectivity.")
        watch_score = tc[1].slider("Watch score", 0.05, 0.50, float(thresholds.get("watch_score", 0.18)), 0.01, help="Score magnitude needed to flag a watch (worth keeping an eye on).")
        min_conf_trade = tc[2].slider("Min confidence to trade", 0.10, 0.80, float(settings.risk.get("min_confidence_for_trade", 0.35)), 0.01, help="Risk-layer confidence gate. Lower = more trades, lower quality.")
        min_risk_reward = tc[3].slider("Min risk/reward", 0.50, 4.00, float(settings.risk.get("min_risk_reward", 1.25)), 0.05, help="Risk-layer reward-to-risk gate. Lower values allow more setups but leave less room for losers.")

    parsed_windows = [int(part) for part in ma_text.replace(" ", "").split(",") if part.strip().isdigit()]
    new_raw.setdefault("features", {})["ma_windows"] = parsed_windows or settings.features.get("ma_windows", [9, 20, 50])
    sf = new_raw.setdefault("signal_fusion", {})
    sf.setdefault("thresholds", {}).update({"long_score": long_score, "short_score": -abs(long_score), "watch_score": watch_score})
    new_raw.setdefault("risk", {}).update({"min_confidence_for_trade": min_conf_trade, "min_risk_reward": min_risk_reward})
    return weights


def _apply_weight_overrides(new_raw: dict, horizon: str, weights: dict) -> None:
    new_raw.setdefault("signal_fusion", {}).setdefault("weights", {}).update(weights)
    # The active horizon profile's weights win over base weights in fusion, so apply
    # the override there too — otherwise the trust dial would have no effect.
    profiles = new_raw.setdefault("horizons", {}).setdefault("profiles", {})
    profile = profiles.setdefault(horizon, {})
    profile.setdefault("weights", {}).update(weights)


def _render_model_controls(settings, new_raw: dict) -> None:
    """Choose which models run and tune their hyperparameters, over the air."""
    available = list(MODEL_REGISTRY.keys())
    enabled_default = [name for name in settings.enabled_models() if name in available] or available
    with _remembered_expander("🤖 Models & hyperparameters", "deepdive_models", expanded=False):
        st.caption("Pick the models that vote on the forecast and tune each one. More models = more confirmation but slower.")
        enabled = st.multiselect("Active models", options=available, default=enabled_default, help="Which prediction models contribute to the blended forecast.")
        models_cfg = new_raw.setdefault("models", {})
        models_cfg["enabled"] = enabled or enabled_default

        if "momentum" in enabled:
            cont = float(settings.models.get("momentum", {}).get("continuation_factor", 0.6))
            value = st.slider("Momentum · continuation factor", 0.1, 1.0, cont, 0.05, help="How much of recent realized momentum is projected forward. Higher = more trend-following.")
            models_cfg.setdefault("momentum", {})["continuation_factor"] = value
        if "baseline" in enabled:
            decay = float(settings.models.get("baseline", {}).get("recency_decay", 5.0))
            value = st.slider("Baseline · recency emphasis", 0.0, 10.0, decay, 0.5, help="How strongly the trend fit weights recent bars over old history. Higher = reacts faster to reversals.")
            models_cfg.setdefault("baseline", {})["recency_decay"] = value
        if "gaussian_process" in enabled:
            gp = settings.models.get("gaussian_process", {})
            cols = st.columns(2)
            kernel = cols[0].selectbox("GP · kernel", ["matern", "rbf"], index=0 if str(gp.get("kernel", "matern")) == "matern" else 1, help="Smoothness assumption for the drift estimate. Matern is a bit rougher/more reactive than RBF.")
            train_rows = cols[1].number_input("GP · max train rows", 30, 500, int(gp.get("max_train_rows", 160)), 10, help="How many recent bars the Gaussian Process learns from.")
            models_cfg.setdefault("gaussian_process", {}).update({"kernel": kernel, "max_train_rows": int(train_rows)})
        if "arima" in enabled:
            order = list(settings.models.get("arima", {}).get("order", [1, 1, 1]))
            cols = st.columns(3)
            p = cols[0].number_input("ARIMA · p (AR)", 0, 5, int(order[0]) if len(order) > 0 else 1, 1, help="Auto-regressive terms: how many past values feed the forecast.")
            d = cols[1].number_input("ARIMA · d (diff)", 0, 2, int(order[1]) if len(order) > 1 else 1, 1, help="Differencing: how many times the series is de-trended before fitting.")
            q = cols[2].number_input("ARIMA · q (MA)", 0, 5, int(order[2]) if len(order) > 2 else 1, 1, help="Moving-average terms: how many past forecast errors feed the forecast.")
            models_cfg.setdefault("arima", {})["order"] = [int(p), int(d), int(q)]


def _render_llm_controls(settings, new_raw: dict) -> None:
    """Configure the News AI (LLM) provider in the UI — local on this machine,
    a cloud API on another — without editing the config file."""
    llm = settings.context_agent.get("news_analysis", {}).get("llm", {})
    options = ["localdeploy", "openai", "openai_compatible", "disabled"]
    current = "disabled" if not llm.get("enabled", False) else str(llm.get("provider", "localdeploy"))
    if current not in options:
        options.insert(0, current)
    with _remembered_expander("🧠 News AI (LLM) settings", "deepdive_llm", expanded=False):
        st.caption("Which AI summarizes and scores the news. Use a local model on this machine, or a cloud API key on another.")
        provider = st.selectbox("Provider", options=options, index=options.index(current), help="localdeploy = a local OpenAI-compatible server; openai = the OpenAI API; openai_compatible = any compatible endpoint; disabled = headline keywords only.")
        cfg = new_raw.setdefault("context_agent", {}).setdefault("news_analysis", {}).setdefault("llm", {})
        if provider == "disabled":
            cfg["enabled"] = False
            st.caption("News will be summarized with simple keyword heuristics only (no AI stance).")
        else:
            cfg["enabled"] = True
            cfg["provider"] = provider
            model = st.text_input("Model", value=str(llm.get("model", "qwen3vl_8b_ollama")), help="The model name to request from the provider, e.g. qwen3vl_8b_ollama (local) or gpt-5 (OpenAI).")
            cfg["model"] = model
            if provider in {"localdeploy", "openai_compatible"}:
                base_url = st.text_input("Base URL", value=str(llm.get("base_url", "http://127.0.0.1:8100/v1/chat/completions")), help="The chat-completions endpoint of your local/compatible server.")
                cfg["base_url"] = base_url
            if provider == "openai":
                api_key_env = st.text_input("API key env var", value=str(llm.get("api_key_env", "OPENAI_API_KEY")), help="Name of the environment variable that holds your OpenAI API key. The key itself is never entered here.")
                cfg["api_key_env"] = api_key_env
            cfg["fallback_to_heuristic"] = st.toggle("Fall back to keywords if the AI is unavailable", value=bool(llm.get("fallback_to_heuristic", True)), help="If on, a failed AI call quietly falls back to keyword summaries instead of erroring.")


def _resolve_active_weights(settings, horizon: str) -> dict:
    base = dict(settings.signal_fusion.get("weights", {}))
    profile = (settings.horizons.get("profiles") or {}).get(horizon, {})
    if isinstance(profile, dict) and isinstance(profile.get("weights"), dict):
        base.update(profile["weights"])
    return base


def _signal_blend_weights(base_weights: dict, preset: str) -> dict[str, float]:
    configured = _SIGNAL_BLEND_PRESETS.get(preset, _SIGNAL_BLEND_PRESETS["Balanced confirmation"])["weights"]
    return _normalized_signal_weights(dict(configured) if configured is not None else dict(base_weights))


def _normalized_signal_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(float(weights.get(key, 0.0)), 0.0) for key in _SIGNAL_COMPONENT_KEYS) or 1.0
    return {key: max(float(weights.get(key, 0.0)), 0.0) / total for key in _SIGNAL_COMPONENT_KEYS}


def _signal_blend_rows(weights: dict[str, float]) -> list[dict]:
    return [
        {"input": _SIGNAL_COMPONENT_LABELS[key], "weight": float(weights.get(key, 0.0)) * 100, "trader_use": _SIGNAL_COMPONENT_HELP[key]}
        for key in _SIGNAL_COMPONENT_KEYS
    ]


def _signal_blend_radar(weights: dict[str, float]) -> go.Figure:
    labels = [_SIGNAL_COMPONENT_LABELS[key] for key in _SIGNAL_COMPONENT_KEYS]
    values = [float(weights.get(key, 0.0)) * 100 for key in _SIGNAL_COMPONENT_KEYS]
    figure = go.Figure(
        go.Scatterpolar(
            r=[*values, values[0]],
            theta=[*labels, labels[0]],
            fill="toself",
            fillcolor="rgba(15, 118, 110, 0.22)",
            line={"color": "#0f766e", "width": 2},
            hovertemplate="%{theta}: %{r:.0f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=300,
        margin={"l": 25, "r": 25, "t": 20, "b": 20},
        polar={
            "radialaxis": {"visible": True, "range": [0, 100], "ticksuffix": "%", "dtick": 20},
            "angularaxis": {"direction": "clockwise"},
        },
        showlegend=False,
    )
    return figure


def _signal_blend_widget_key(component: str) -> str:
    return f"signal_blend_{component}"


def _ensure_signal_blend_slider_state(weights: dict[str, float]) -> None:
    current = {
        key: st.session_state.get(_signal_blend_widget_key(key))
        for key in _SIGNAL_COMPONENT_KEYS
    }
    if (
        all(value is not None for value in current.values())
        and all(0 <= int(value) <= 100 for value in current.values())
        and sum(int(value) for value in current.values()) == 100
    ):
        return
    source = current if any(value is not None for value in current.values()) else {
        key: float(weights.get(key, 0.0)) * 100
        for key in _SIGNAL_COMPONENT_KEYS
    }
    for key, value in _integer_signal_allocations(source).items():
        st.session_state[_signal_blend_widget_key(key)] = value


def _rebalance_signal_blend_slider_state(changed_component: str) -> None:
    allocations = {
        key: st.session_state.get(_signal_blend_widget_key(key), 0)
        for key in _SIGNAL_COMPONENT_KEYS
    }
    for key, value in _rebalance_signal_allocations(allocations, changed_component).items():
        st.session_state[_signal_blend_widget_key(key)] = value


def _rebalance_signal_allocations(allocations: dict[str, float], changed_component: str) -> dict[str, int]:
    if changed_component not in _SIGNAL_COMPONENT_KEYS:
        raise ValueError(f"Unknown signal component: {changed_component}")
    changed_value = min(max(int(round(float(allocations.get(changed_component, 0)))), 0), 100)
    other_components = [key for key in _SIGNAL_COMPONENT_KEYS if key != changed_component]
    rebalanced = _integer_signal_allocations(
        {key: allocations.get(key, 0) for key in other_components},
        total=100 - changed_value,
    )
    return {
        key: changed_value if key == changed_component else rebalanced[key]
        for key in _SIGNAL_COMPONENT_KEYS
    }


def _integer_signal_allocations(weights: dict[str, float | None], total: int = 100) -> dict[str, int]:
    keys = list(weights)
    if not keys:
        return {}
    safe_total = max(int(total), 0)
    positive = {key: max(float(weights.get(key) or 0.0), 0.0) for key in keys}
    weight_total = sum(positive.values())
    raw = {
        key: safe_total * (positive[key] / weight_total if weight_total else 1.0 / len(keys))
        for key in keys
    }
    result = {key: int(value) for key, value in raw.items()}
    remainder = safe_total - sum(result.values())
    for key in sorted(keys, key=lambda item: (raw[item] - result[item], -keys.index(item)), reverse=True)[:remainder]:
        result[key] += 1
    return result


def _render_horizon_selector(container, horizon_options: list[str], default_horizon: str) -> str:
    options = horizon_options or ["swing"]
    default = default_horizon if default_horizon in options else options[0]
    format_func = lambda value: str(value).replace("_", " ").title()
    help_text = "Intraday emphasizes session/VWAP behavior; swing uses multi-day signals; position uses a longer lookback and wider risk."
    if hasattr(container, "segmented_control"):
        selected = container.segmented_control("Horizon", options=options, default=default, format_func=format_func, help=help_text)
    else:
        selected = container.radio("Horizon", options=options, index=options.index(default), horizontal=True, format_func=format_func, help=help_text)
    return str(selected or default)


def _scanner_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    visible = [column for column in columns if column in df.columns]
    summary = df[visible].copy()
    if "rank_score" in summary:
        summary = summary.sort_values("rank_score", ascending=False)
    return summary


def _render_scanner_detail(df: pd.DataFrame) -> None:
    if df.empty or "symbol" not in df:
        return
    st.subheader("Symbol Detail")
    options: dict[str, dict] = {}
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        label = _scanner_detail_label(row_dict)
        options[label] = row_dict
    selected = st.selectbox("Choose symbol", list(options.keys()))
    row = options[selected]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Setup", str(row.get("setup_quality", "-")), help=HELP_TEXT["setup"])
    c2.metric("Risk/Reward", _format_scanner_value("risk_reward", row.get("risk_reward")), help=HELP_TEXT["risk_reward"])
    c3.metric("VWAP", str(row.get("vwap_alignment", "-")), help=HELP_TEXT["vwap_alignment"])
    c4.metric("Regime", str(row.get("regime", "-")), help="Market behavior label such as trending, choppy, high-volatility, or low-volatility.")

    groups = {
        "Trade Setup": [
            "action",
            "confidence",
            "score",
            "rank_score",
            "setup_quality",
            "top_reason",
            "skip_reasons",
            "risk_reward",
            "catalyst_flag",
            "risk_flag",
        ],
        "Movement And Volume": [
            "price",
            "change_pct",
            "volume",
            "avg_volume",
            "volume_anomaly",
            "gap_pct",
            "atr_pct",
            "high_relative_volume",
            "meaningful_gap",
        ],
        "Levels": [
            "session_open",
            "prior_high",
            "prior_low",
            "opening_range_high",
            "opening_range_low",
            "opening_range_status",
            "extension_from_vwap_pct",
            "distance_to_support_pct",
            "distance_to_resistance_pct",
        ],
        "Market Context": [
            "benchmark",
            "benchmark_change_pct",
            "relative_strength_pct",
            "trend",
            "regime",
            "liquidity_ok",
        ],
    }
    for group, keys in groups.items():
        rows = [
            {"field": _humanize_key(key), "value": _format_scanner_value(key, row.get(key))}
            for key in keys
            if key in row and pd.notna(row.get(key))
        ]
        if rows:
            with _remembered_expander(group, f"scanner_detail_{_slug_key(group)}", expanded=(group == "Trade Setup")):
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _scanner_detail_label(row: dict) -> str:
    symbol = row.get("symbol", "-")
    action = row.get("action", "-")
    rank = _format_scanner_value("rank_score", row.get("rank_score"))
    reason = row.get("top_reason", "-")
    return f"{symbol} | {action} | rank {rank} | {reason}"


def _render_analysis(settings, symbol: str, horizon: str | None = None, news_limit: int | None = None, refresh: bool = True) -> None:
    if refresh:
        import time

        last_durations = st.session_state.setdefault("_analyze_durations", {})
        prior = last_durations.get("last")
        start = time.monotonic()
        hint = f" (last run took ~{prior:.0f}s)" if prior else " (news AI summaries can take ~10–60s on first run)"
        progress = st.progress(0.0, text=f"Starting {symbol} analysis…{hint}")

        def on_progress(fraction: float, message: str) -> None:
            elapsed = time.monotonic() - start
            eta = (elapsed / fraction) * (1.0 - fraction) if fraction > 0.02 else None
            suffix = f" · ~{eta:.0f}s left" if eta and eta >= 1 else ""
            progress.progress(min(max(fraction, 0.0), 1.0), text=f"{message}…{suffix}")

        result = analyze_symbol(symbol, settings, horizon=horizon, news_limit=news_limit, progress_callback=on_progress)
        on_progress(0.97, f"Loading chart history for {symbol}")
        provider = get_market_data_provider(settings)
        frame = fetch_market_data(symbol, settings, provider)
        elapsed = time.monotonic() - start
        last_durations["last"] = elapsed
        progress.progress(1.0, text=f"{symbol} analysis ready in {elapsed:.0f}s")
        st.session_state.latest_analysis = {"result": result, "frame": frame}
        _persist_dashboard_cache(settings)
    else:
        latest = st.session_state.get("latest_analysis")
        if latest is None:
            return
        result = latest["result"]
        frame = latest["frame"]
    st.caption(f"Showing last completed analysis: {result.snapshot.symbol} ({result.horizon})")

    session = result.session
    live_price = session.live_price if _session_is_live(session) else None
    headline_price = live_price if live_price is not None else result.snapshot.latest_close
    headline_label = "Live Price" if live_price is not None else "Last Close"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Entry Readiness", _entry_readiness_label(result.decision), help=HELP_TEXT["action"])
    col2.metric("Confidence", _format_percent(result.decision.confidence), help=HELP_TEXT["confidence"])
    col3.metric("Score", f"{result.decision.score:.3f}", help=HELP_TEXT["score"])
    col4.metric(
        headline_label,
        _format_price(headline_price),
        _format_percent(result.snapshot.change_pct),
        help="Latest available price from intraday data when available; otherwise last daily close.",
    )

    _render_verdict_banner(result)
    _render_result_explanation(result)

    if session is not None and session.data_available:
        session_scope = "Current session" if _session_is_live(session) else f"Reference session {getattr(session, 'session_date', '') or 'latest available'}"
        st.caption(
            f"{session_scope}: {session.market_session} ({session.bars_loaded} intraday bars). "
            f"Horizon: {result.horizon}."
        )
        if not _session_is_live(session):
            st.info("Intraday bars are from the most recent completed session. They are shown for reference only and do not drive the current risk plan or intraday score.")
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        level_prefix = "Session" if _session_is_live(session) else "Prior Session"
        sc1.metric(f"{level_prefix} VWAP", _format_price(session.session_vwap) if session.session_vwap else "-", help=HELP_TEXT["vwap"])
        sc2.metric(f"{level_prefix} Open", _format_price(session.session_open) if session.session_open else "-", help="First regular-session price in the loaded intraday data.")
        sc3.metric(f"{level_prefix} High", _format_price(session.session_high) if session.session_high else "-", help="Highest price in the loaded session data.")
        sc4.metric(f"{level_prefix} Low", _format_price(session.session_low) if session.session_low else "-", help="Lowest price in the loaded session data.")
        sc5.metric(
            "TOD RVOL",
            f"{session.time_of_day_rvol:.2f}" if session.time_of_day_rvol is not None else "-",
            help="Time-of-day relative volume. Above 1.0 means current volume is heavier than expected for this part of the session.",
        )
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Premarket High", _format_price(session.premarket_high) if session.premarket_high else "-", help="High from premarket bars when the provider supplies them.")
        pc2.metric("Premarket Low", _format_price(session.premarket_low) if session.premarket_low else "-", help="Low from premarket bars when the provider supplies them.")
        pc3.metric("Opening Range", session.opening_range_status, help="Whether the first configured minutes of regular-session data are available for breakout/reclaim checks.")
    elif session is not None:
        st.caption(f"Session: {session.market_session}. Intraday bars unavailable — analysis is daily-only.")

    market_state = result.market_state
    sector = result.sector_context
    calendar = result.calendar
    if market_state is not None or sector is not None or calendar is not None:
        st.subheader("Market Cross-Check")
        mc1, mc2, mc3, mc4 = st.columns(4)
        if market_state is not None:
            mc1.metric("SPY", _format_percent(market_state.spy_change_pct) if market_state.spy_change_pct is not None else "-", help="Broad-market benchmark. Long setups are cleaner when the market supports them.")
            mc2.metric("QQQ", _format_percent(market_state.qqq_change_pct) if market_state.qqq_change_pct is not None else "-", help="Growth/tech benchmark. Especially relevant for many high-beta names.")
            mc3.metric("VIX", f"{market_state.vix_value:.2f}" if market_state.vix_value is not None else "-", help="Volatility index. Higher VIX can mean wider stops, faster reversals, and smaller position size.")
        if sector is not None and sector.sector_etf:
            mc4.metric(
                f"{sector.sector_etf} ({sector.alignment})",
                _format_percent(sector.sector_change_pct) if sector.sector_change_pct is not None else "-",
                help="Sector ETF movement and whether it aligns with the symbol. Sector confirmation helps avoid isolated weak setups.",
            )
        if calendar is not None and calendar.no_trade_flags:
            st.warning("Calendar flags: " + "; ".join(calendar.no_trade_flags))

    _render_trade_plan_summary(result)

    _render_why_not_actionable(result, settings)

    _render_news_decision_panel(result)

    if result.previous_snapshots:
        with _remembered_expander(f"Compared to your last {len(result.previous_snapshots)} analyses", f"analysis_compare_{result.snapshot.symbol}", expanded=False):
            from stockpredictor.snapshots import diff_snapshots

            prev = result.previous_snapshots[-1]
            if result.snapshot_record is not None:
                diff = diff_snapshots(result.snapshot_record, prev)
                dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                dcol1.metric("Δ Score", f"{diff['score_delta']:+.3f}" if diff.get("score_delta") is not None else "-", help="Change in fused score since the last saved analysis.")
                dcol2.metric("Δ Confidence", f"{diff['confidence_delta']:+.3f}" if diff.get("confidence_delta") is not None else "-", help="Change in signal confidence since the last saved analysis.")
                dcol3.metric("Δ Price", _format_price(diff["live_price_delta"]) if diff.get("live_price_delta") is not None else "-", help="Price change since the previous snapshot.")
                dcol4.metric("Action Changed?", "yes" if diff.get("action_changed") else "no", help="Whether the fused action changed since the previous snapshot.")
                st.caption(f"Previous action: {diff.get('previous_action', '-')} @ {diff.get('previous_timestamp', '-')}")

    st.subheader("Chart And Levels")
    chart_range = st.segmented_control(
        "Daily chart range",
        options=list(_CHART_RANGE_BARS),
        default="30 bars",
        help="Use 30 bars for the active setup, 60 for swing context, or full history for the broad trend. Candles are unadjusted daily OHLC bars from the configured provider.",
    )
    st.caption(
        f"Source: {result.snapshot.provider} daily OHLC bars through {result.snapshot.as_of}. "
        "The volume-weighted average is a rolling daily-bar reference; Session VWAP above is the true intraday session anchor when current minute bars are available."
    )
    st.plotly_chart(
        _price_chart(
            frame,
            result.features.levels,
            result.risk_plan,
            ma_windows=settings.features.get("ma_windows", [9, 20, 50]),
            visible_bars=_CHART_RANGE_BARS.get(str(chart_range), 30),
        ),
        width="stretch",
    )

    _render_context_panel(result)

    with _remembered_expander("Model Details", "analysis_model_details", expanded=False):
        model_df = _percent_display(pd.DataFrame([asdict(prediction) for prediction in result.predictions]), ["expected_return", "confidence"])
        st.dataframe(model_df, width="stretch", hide_index=True, column_config=_model_column_config())
        notes = []
        for prediction in result.predictions:
            guide = edu.MODEL_GUIDES.get(prediction.model, {})
            note = {"model": prediction.model, "what it is": guide.get("what", _MODEL_NOTES.get(prediction.model, "Model forecast."))}
            if _education_on() and guide.get("why"):
                note["why it's used"] = guide["why"]
            notes.append(note)
        if notes:
            columns = {
                "model": st.column_config.TextColumn("Model", help="Prediction model."),
                "what it is": st.column_config.TextColumn("What it is / limits", help="Plain-language description of the model and what it is good and bad at."),
                "why it's used": st.column_config.TextColumn("Why it's used", help="What this model contributes to the blended forecast."),
            }
            st.dataframe(pd.DataFrame(notes), width="stretch", hide_index=True, column_config=columns)
        st.caption(_MODELS_DISCLAIMER)

    with _remembered_expander("Full Signal / Risk Data", "analysis_full_signal_risk", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "bias": _decision_bias(result.decision),
                        "signal_action": _decision_signal_action(result.decision),
                        "action": result.decision.action,
                        "confidence": result.decision.confidence * 100,
                        "score": result.decision.score,
                        "execution_blockers": "; ".join(_decision_execution_blockers(result.decision)),
                        "top_reason": result.decision.top_reason,
                        "reasons": "; ".join(result.decision.reasons),
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
                "score": st.column_config.NumberColumn("Score", format="%.3f"),
            },
        )
        st.dataframe(
            pd.DataFrame([_risk_plan_row(result.risk_plan)]),
            width="stretch",
            hide_index=True,
            column_config=_risk_column_config(),
        )

    with _remembered_expander("Indicator Details", "analysis_indicator_details", expanded=False):
        st.dataframe(
            pd.DataFrame(
                _indicator_rows(to_serializable(result.features.indicators))
            ),
            width="stretch",
            hide_index=True,
        )


def _render_trade_plan_summary(result) -> None:
    plan = result.risk_plan
    decision = result.decision
    st.subheader("Decision And Plan")
    with _remembered_expander("Decision checklist", "decision_checklist", expanded=False):
        st.write(
            "Before acting, confirm: the setup has a clear catalyst or technical reason, "
            "price respects the key level or VWAP, risk/reward meets your minimum, "
            "position size fits max risk, and the invalidation is obvious."
        )

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Bias", _bias_label(_decision_bias(decision)), help=HELP_TEXT["bias"])
    p2.metric("Signal", _action_label(_decision_signal_action(decision)), help="Directional signal before calendar and risk execution gates.")
    p3.metric("Entry Readiness", _entry_readiness_label(decision), help=HELP_TEXT["action"])
    p4.metric("Confidence", _format_percent(decision.confidence), help=HELP_TEXT["confidence"])
    p5.metric("Risk/Reward", f"{plan.risk_reward:.2f}" if plan.risk_reward is not None else "-", help=HELP_TEXT["risk_reward"])
    p6.metric("Position", f"{plan.position_size:,}" if plan.position_size else "-", help=HELP_TEXT["position"])
    st.caption(f"Setup quality: {plan.setup_quality}")

    levels = [
        {"field": "Entry zone", "value": _format_price_range(plan.entry_zone)},
        {"field": "Entry", "value": _format_price(plan.entry) if plan.entry else "-"},
        {"field": "Stop", "value": _format_price(plan.stop_loss) if plan.stop_loss else "-"},
        {"field": "Target", "value": _format_targets(plan.targets)},
        {"field": "Invalidation", "value": plan.invalidation or "-"},
    ]
    st.dataframe(pd.DataFrame(levels), width="stretch", hide_index=True, column_config=_trade_plan_column_config())
    st.caption(f"Primary reason: {decision.top_reason or '-'}")
    if plan.no_trade_reasons:
        st.warning("No-trade reasons: " + "; ".join(plan.no_trade_reasons))
    if plan.session_checks:
        st.caption(_session_check_text(plan.session_checks))


def _news_score_contribution(decision) -> float:
    """Net amount the news-driven inputs (context + sentiment components and any news
    penalty) moved the fused score. Used for the crisp one-line news-impact readout."""
    total = 0.0
    for row in decision.score_breakdown or []:
        component = str(row.get("component", ""))
        contribution = float(row.get("contribution", 0.0) or 0.0)
        if row.get("kind") == "component" and component in {"context", "sentiment"}:
            total += contribution
        elif row.get("kind") == "penalty" and "news" in component.lower():
            total += contribution
    return total


def _news_impact_phrase(result) -> str:
    news = result.context.news_analysis or {}
    if not news:
        return ""
    stance = news.get("stance", {}) if isinstance(news.get("stance"), dict) else {}
    direction = str(stance.get("direction", "neutral"))
    conviction = float(stance.get("conviction", 0.0) or 0.0)
    contribution = _news_score_contribution(result.decision)
    return f"News: {direction} (conviction {conviction:.0%}) → {contribution:+.3f} to score"


def _render_verdict_banner(result) -> None:
    """One-line, color-coded verdict so a trader can read the decision in two seconds."""
    decision = result.decision
    action = str(decision.action)
    sentiment_word = "bullish" if decision.score > 0.05 else "bearish" if decision.score < -0.05 else "neutral"
    _, meaning = _action_presentation(action)
    signal_action = _decision_signal_action(decision)
    parts = [
        f"**{_bias_label(_decision_bias(decision))} BIAS**",
        f"signal {_action_label(signal_action)}",
        f"entry readiness {_entry_readiness_label(decision)}",
        f"score {decision.score:+.3f} ({sentiment_word})",
        f"confidence {decision.confidence:.0%}",
    ]
    news_phrase = _news_impact_phrase(result)
    if news_phrase:
        parts.append(news_phrase)
    blocker_text = ""
    execution_blockers = _decision_execution_blockers(decision)
    if execution_blockers:
        blocker_text = " Blocked by: " + "; ".join(execution_blockers) + "."
    message = "  ·  ".join(parts) + f"\n\n{meaning}{blocker_text} Primary reason: {decision.top_reason or 'No dominant reason was produced.'}"
    renderer = {"long": st.success, "short": st.error, "watch": st.info}.get(action, st.warning)
    renderer(message)


def _render_result_explanation(result) -> None:
    decision = result.decision
    rows = _score_attribution_rows(decision)
    st.subheader("Why This Result")
    st.write(_result_explanation_text(decision))
    blockers = _decision_execution_blockers(decision)
    if blockers:
        st.warning("Execution wait: " + "; ".join(blockers) + ". The setup can still be directionally valid; re-check price and risk before placing a trade.")
    elif decision.action in {"long", "short"}:
        st.success("Execution gate passed. Review the entry zone, stop, target, liquidity, and position size before acting.")
    else:
        st.info("No executable setup passed the configured signal and risk gates.")
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "input": st.column_config.TextColumn("Input", help="Evidence group or penalty used by the fusion layer."),
            "kind": st.column_config.TextColumn("Kind", help="Evidence is a weighted vote. Penalty reduces the fused score after voting."),
            "raw_score": st.column_config.NumberColumn("Raw signal", format="%+.3f", help="Directional value before weighting: positive is bullish, negative is bearish."),
            "weight": st.column_config.NumberColumn("Influence", format="%.0f%%", help="Configured influence for evidence rows. Penalties show the multiplier applied."),
            "contribution": st.column_config.NumberColumn("Score effect", format="%+.3f", help="Signed amount this row added to or removed from the final score."),
            "explanation": st.column_config.TextColumn("Interpretation", help="Plain-language explanation of how this row affected the result."),
        },
    )
    with _remembered_expander("Model vote explainability", f"model_vote_explanation_{decision.symbol}", expanded=False):
        st.caption("Models estimate direction from price history. Their votes are evidence, not entry triggers. Compare them with chart structure, volume, catalysts, and risk.")
        st.dataframe(
            pd.DataFrame(_model_vote_rows(result)),
            width="stretch",
            hide_index=True,
            column_config={
                "model": st.column_config.TextColumn("Model"),
                "direction": st.column_config.TextColumn("Forecast"),
                "expected_return": st.column_config.NumberColumn("Expected return", format="%+.2f%%"),
                "confidence": st.column_config.NumberColumn("Model confidence", format="%.1f%%"),
                "vote": st.column_config.NumberColumn("Directional vote", format="%+.3f"),
                "predicted_price": st.column_config.NumberColumn("Predicted price", format="$%.2f"),
                "explanation": st.column_config.TextColumn("What this means"),
            },
        )


def _score_attribution_rows(decision) -> list[dict]:
    rows = []
    for row in decision.score_breakdown or []:
        component = str(row.get("component", ""))
        kind = str(row.get("kind", ""))
        contribution = float(row.get("contribution", 0.0) or 0.0)
        weight = float(row.get("weight", 0.0) or 0.0)
        rows.append(
            {
                "input": _humanize_key(component),
                "kind": "evidence" if kind == "component" else kind,
                "raw_score": row.get("raw_score"),
                "weight": weight * 100,
                "contribution": contribution,
                "explanation": _attribution_explanation(component, kind, contribution),
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["contribution"])), reverse=True)


def _attribution_explanation(component: str, kind: str, contribution: float) -> str:
    direction = "supported a bullish read" if contribution > 0 else "supported a bearish read" if contribution < 0 else "was neutral"
    if kind != "component":
        return f"Reduced the prior fused score by {abs(contribution):.3f}."
    labels = {
        "models": "Statistical price forecasts",
        "technicals": "Chart structure and indicators",
        "intraday": "Current session behavior",
        "context": "Catalysts and AI news context",
        "sentiment": "News tone",
    }
    return f"{labels.get(component, _humanize_key(component))} {direction}."


def _result_explanation_text(decision) -> str:
    rows = _score_attribution_rows(decision)
    positive = [row for row in rows if float(row["contribution"]) > 0][:2]
    negative = [row for row in rows if float(row["contribution"]) < 0][:2]
    driver_text = ", ".join(f"{row['input']} {float(row['contribution']):+.3f}" for row in positive) or "no bullish driver"
    offset_text = ", ".join(f"{row['input']} {float(row['contribution']):+.3f}" for row in negative) or "no material bearish offset"
    return (
        f"{_bias_label(_decision_bias(decision))} bias with {_action_label(_decision_signal_action(decision))} signal: "
        f"the strongest positive drivers were {driver_text}. The main offsets were {offset_text}. "
        f"Final fused score: {decision.score:+.3f}; confidence: {decision.confidence:.0%}."
    )


def _model_vote_rows(result) -> list[dict]:
    rows = []
    for prediction in result.predictions:
        vote = float(result.decision.model_scores.get(prediction.model, 0.0))
        rows.append(
            {
                "model": prediction.model,
                "direction": prediction.direction,
                "expected_return": float(prediction.expected_return) * 100,
                "confidence": float(prediction.confidence) * 100,
                "vote": vote,
                "predicted_price": prediction.predicted_price,
                "explanation": _model_vote_explanation(vote, float(prediction.confidence)),
            }
        )
    return rows


def _model_vote_explanation(vote: float, confidence: float) -> str:
    direction = "bullish" if vote > 0.05 else "bearish" if vote < -0.05 else "neutral"
    strength = "strong" if abs(vote) >= 0.65 else "moderate" if abs(vote) >= 0.30 else "weak"
    return f"{strength.title()} {direction} price-history vote with {confidence:.0%} model confidence."


def _action_presentation(action: str) -> tuple[str, str]:
    key = str(action).lower()
    return _ACTION_PRESENTATION.get(key, (key.upper().replace("_", " "), "Review the score attribution and risk checks before acting."))


def _action_label(action: str) -> str:
    return _action_presentation(action)[0]


def _entry_readiness_label(decision) -> str:
    blockers = _decision_execution_blockers(decision)
    action = str(decision.action)
    if blockers:
        if action in {"long", "short"} and all(str(blocker).lower() == "market is currently closed" for blocker in blockers):
            return "WAIT FOR MARKET OPEN"
        return "SKIP FRESH ENTRY"
    if action in {"long", "short"}:
        return "READY TO REVIEW"
    if action == "watch":
        return "WATCH FOR CONFIRMATION"
    if action == "low_confidence":
        return "WAIT FOR CLEARER EDGE"
    return "SKIP FRESH ENTRY"


def _bias_label(bias: str) -> str:
    return _BIAS_PRESENTATION.get(str(bias).lower(), str(bias).upper() or "NEUTRAL")


def _decision_signal_action(decision) -> str:
    return str(getattr(decision, "signal_action", "") or decision.action)


def _decision_bias(decision) -> str:
    bias = str(getattr(decision, "bias", "") or "").lower()
    if bias:
        return bias
    score = float(getattr(decision, "score", 0.0) or 0.0)
    return "bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral"


def _decision_execution_blockers(decision) -> list[str]:
    return list(getattr(decision, "execution_blockers", []) or [])


def _render_why_not_actionable(result, settings) -> None:
    """Crisp, glanceable explanation of exactly which gate held the trade back, and
    what would flip it — directly answering 'why is everything no_trade?'."""
    decision = result.decision
    plan = result.risk_plan
    if decision.action in {"long", "short"} and plan.setup_quality == "actionable":
        st.success("Actionable: signal strength, confidence, liquidity, volatility, and risk/reward gates all passed.")
        return

    thresholds = settings.signal_fusion.get("thresholds", {})
    risk_cfg = settings.risk
    watch_score = float(thresholds.get("watch_score", 0.18))
    long_score = float(thresholds.get("long_score", 0.30))
    short_score = float(thresholds.get("short_score", -0.30))
    required_conf = max(float(thresholds.get("min_confidence", 0.30)), float(risk_cfg.get("min_confidence_for_trade", 0.35)))

    blockers: list[str] = []
    if result.calendar is not None and result.calendar.no_trade_flags:
        blockers.extend(f"Hard block — {flag}." for flag in result.calendar.no_trade_flags)
    if abs(decision.score) < watch_score:
        blockers.append(
            f"Signal too weak — score {decision.score:+.3f} is inside the no-trade zone "
            f"(needs ±{watch_score:.2f} to watch, +{long_score:.2f} long or {short_score:.2f} short to trade)."
        )
    if decision.confidence < required_conf:
        blockers.append(f"Confidence {decision.confidence:.0%} is below the required {required_conf:.0%}.")
    quality_messages = {
        "low_liquidity": "Liquidity — average volume is below the configured minimum.",
        "too_volatile": "Volatility — ATR% is above the configured maximum.",
        "extended": "Extended — price is too far from VWAP for a clean entry.",
        "poor_risk_reward": f"Risk/reward is below the configured minimum ({risk_cfg.get('min_risk_reward', '-')}).",
        "invalid_position_size": "Risk sizing produced no valid position.",
    }
    if plan.setup_quality in quality_messages:
        blockers.append(quality_messages[plan.setup_quality])
    if not blockers:
        blockers = list(plan.no_trade_reasons) or ["Setup is not actionable under configured thresholds."]

    st.warning("**Why this isn't actionable**\n\n" + "\n".join(f"- {item}" for item in blockers))

    hints: list[str] = []
    if abs(decision.score) < watch_score:
        hints.append("models, technicals, and news agreeing on a clearer direction")
    if decision.confidence < required_conf:
        hints.append("higher conviction across components")
    if plan.setup_quality == "poor_risk_reward":
        hints.append("a pullback that tightens risk, or a confirmed break above nearby resistance before recalculating the plan")
    if _decision_execution_blockers(decision) and all(
        str(blocker).lower() == "market is currently closed" for blocker in _decision_execution_blockers(decision)
    ):
        hints.append("the market reopening, followed by a fresh price and spread check")
    if hints:
        st.caption("Would become actionable with: " + "; ".join(hints) + ".")


def _render_news_decision_panel(result) -> None:
    """White-box view: shows whether news was gathered and exactly how it moved the
    fused score, tying the News-tab style summary directly to the trade decision."""
    st.subheader("How News Shaped This Decision")
    context = result.context
    news = context.news_analysis or {}
    evidence = context.evidence or []
    considered = bool(news or evidence)
    enrichment = getattr(result, "news_enrichment", {}) or {}
    enrichment_status = str(enrichment.get("status", "unknown"))

    if enrichment_status == "unavailable":
        st.warning(f"Rich news enrichment is unavailable for this analysis: {enrichment.get('error', 'unknown error')}")
    elif enrichment_status == "degraded":
        st.warning(f"Rich news enrichment is degraded: {enrichment.get('error', 'review the analyzer status below')}")

    if not considered:
        st.info(
            "No news evidence was gathered for this decision. The action below relies on "
            "models, technicals, and configured context only. Enable "
            "`context_agent.news_analysis` (and `use_in_decision`) to feed news into the decision."
        )
        return

    provider = str(news.get("analysis_provider", "heuristic"))
    freshness_values = [float(item.get("freshness", 0.0) or 0.0) for item in evidence]
    top_freshness = max(freshness_values) if freshness_values else 0.0
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("News considered", f"Yes ({provider})", help="Whether a news summary was gathered and fed into this decision, and by which analyzer.")
    b2.metric("Headlines used", len(evidence), help="Number of headlines actually passed into this decision as evidence.")
    b3.metric("Top freshness", _format_percent(top_freshness), help=HELP_TEXT["freshness"])
    b4.metric("Category", str(news.get("dominant_category", "other")).replace("_", " "), help="Most common catalyst/category among the headlines used.")

    grand_summary = str(news.get("grand_summary", "")).strip()
    if grand_summary:
        st.write(grand_summary)

    impact_phrase = _news_impact_phrase(result)
    if impact_phrase:
        st.caption(f"{impact_phrase}. Stance is an evidence summary from {provider}, blended into the catalyst score — not advice.")

    focus = news.get("day_trader_focus", {}) if isinstance(news.get("day_trader_focus"), dict) else {}
    if focus:
        st.dataframe(
            pd.DataFrame(
                [
                    {"question": "Catalyst", "answer": focus.get("catalyst", "-")},
                    {"question": "Risk", "answer": focus.get("risk", "-")},
                    {"question": "Tradeability", "answer": focus.get("tradeability", "-")},
                    {"question": "No-trade flags", "answer": "; ".join(focus.get("no_trade_flags", [])) or "-"},
                ]
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "question": st.column_config.TextColumn("Question", help="Trader question the news summary answers."),
                "answer": st.column_config.TextColumn("Answer", help="LLM or heuristic summary for this decision question."),
            },
        )

    if provider == "heuristic_fallback":
        st.warning(f"Heuristic fallback summary was used. LLM error: {news.get('llm_error', 'unknown error')}")
    elif provider == "llm_error":
        st.warning(f"LLM summary failed; headlines below are still available. Error: {news.get('llm_error', 'unknown error')}")

    if evidence:
        with _remembered_expander(f"Headlines used in this decision ({len(evidence)})", f"decision_news_evidence_{result.snapshot.symbol}", expanded=False):
            ev_df = pd.DataFrame(evidence)
            columns = [column for column in ["provider", "published", "classification_provider", "category", "sentiment", "impact", "day_trader_relevance", "freshness", "title", "url"] if column in ev_df.columns]
            st.dataframe(ev_df[columns], width="stretch", hide_index=True, column_config=_headline_column_config())


def _render_context_panel(result) -> None:
    st.subheader("Context And Catalyst")
    st.write(result.context.raw_summary)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentiment", result.context.sentiment, help=HELP_TEXT["sentiment"])
    c2.metric("Catalyst Score", f"{to_float(result.context.features.get('catalyst_score'), 0.0):.2f}", help=HELP_TEXT["catalyst_score"])
    c3.metric("Freshness", _format_percent(to_float(result.context.features.get("catalyst_freshness"), 0.0)), help=HELP_TEXT["freshness"])
    c4.metric("Context Confidence", _format_percent(to_float(result.context.features.get("context_confidence"), 0.0)), help=HELP_TEXT["context_confidence"])

    if result.context.reasons_to_trade:
        st.success("Reasons to trade: " + "; ".join(result.context.reasons_to_trade))
    if result.context.reasons_to_skip:
        st.warning("Reasons to skip: " + "; ".join(result.context.reasons_to_skip))

    with _remembered_expander("Context Details", "context_details", expanded=False):
        context_metrics = {
            "sentiment": result.context.sentiment,
            **result.context.features,
        }
        context_metrics = _percent_context_metrics(context_metrics)
        st.dataframe(pd.DataFrame([context_metrics]), width="stretch", hide_index=True, column_config=_context_column_config())
        if result.context.catalysts:
            st.write("Catalysts")
            st.write(result.context.catalysts)
        if result.context.risks:
            st.write("Risks")
            st.write(result.context.risks)


def _price_chart(
    frame: pd.DataFrame,
    levels: dict[str, float | None],
    risk_plan=None,
    ma_windows: list[int] | None = None,
    visible_bars: int | None = 30,
):
    chart_frame = frame.tail(visible_bars).copy() if visible_bars else frame.copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
    )
    fig.add_trace(
        go.Candlestick(
            x=chart_frame.index,
            open=chart_frame["Open"],
            high=chart_frame["High"],
            low=chart_frame["Low"],
            close=chart_frame["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    for window in [int(value) for value in (ma_windows or [])]:
        column = f"sma_{window}"
        if len(frame) >= window:
            moving_average = frame["Close"].rolling(window).mean().reindex(chart_frame.index)
            fig.add_trace(go.Scatter(x=chart_frame.index, y=moving_average, mode="lines", name=column.upper()), row=1, col=1)
    if "Volume" in chart_frame:
        fig.add_trace(go.Bar(x=chart_frame.index, y=chart_frame["Volume"], name="Volume", marker_color="#8892a6"), row=2, col=1)
    chart_levels = _chart_levels_for_view(levels, chart_frame)
    for name, value in chart_levels.items():
        fig.add_hline(y=float(value), annotation_text=f"{_CHART_LEVEL_LABELS.get(name, _humanize_key(name))}: {_format_price(float(value))}", line_dash="dot", opacity=0.55, row=1, col=1)
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


def _session_is_live(session) -> bool:
    return bool(session is not None and getattr(session, "is_live", False) and session.live_price is not None)


def _chart_levels_for_view(levels: dict[str, float | None], chart_frame: pd.DataFrame) -> dict[str, float]:
    if chart_frame.empty:
        return {}
    chart_low = float(chart_frame["Low"].min())
    chart_high = float(chart_frame["High"].max())
    padding = max((chart_high - chart_low) * 0.08, chart_high * 0.01)
    lower_bound = chart_low - padding
    upper_bound = chart_high + padding
    priority = [
        "resistance",
        "support",
        "prior_high",
        "prior_low",
        "session_open",
        "vwap",
        "sma_20",
        "sma_50",
        "opening_range_high",
        "opening_range_low",
    ]
    ordered_names = [*priority, *(name for name in levels if name not in priority)]
    visible: dict[str, float] = {}
    for name in ordered_names:
        value = to_float(levels.get(name), None)
        if value is None or not lower_bound <= value <= upper_bound:
            continue
        if any(abs(value / existing - 1) < 0.0075 for existing in visible.values() if existing):
            continue
        visible[name] = value
    return visible


def _render_scanner_filters(df: pd.DataFrame, settings) -> pd.DataFrame:
    defaults = settings.raw.get("scanner", {}).get("default_filters", {})
    with _remembered_expander("Filters", "scanner_filters", expanded=False):
        st.caption("Min abs change and Max ATR are in percent. Min RVOL is the volume ratio (1.0 = average).")
        c1, c2, c3, c4 = st.columns(4)
        min_abs_change = c1.slider("Min abs change", 0.0, 20.0, float(defaults.get("min_abs_change_pct", 0.0)) * 100, 0.5, format="%.1f%%", help="Minimum absolute price move. Raise this to focus only on stocks that are moving.")
        min_rvol = c2.slider("Min RVOL", 0.0, 10.0, float(defaults.get("min_volume_anomaly", 0.0)), 0.25, help=HELP_TEXT["avg_rvol"])
        max_atr = c3.slider("Max ATR", 0.0, 50.0, float(defaults.get("max_atr_pct", 0.50)) * 100, 1.0, format="%.0f%%", help=HELP_TEXT["atr_pct"])
        actions = sorted(df["action"].dropna().unique().tolist()) if "action" in df else []
        selected_actions = c4.multiselect("Actions", actions, default=actions, help=HELP_TEXT["action"])
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


def _trade_plan_column_config() -> dict:
    return {
        "field": st.column_config.TextColumn("Field", help="Part of the proposed trade plan."),
        "value": st.column_config.TextColumn("Value", help="Read each plan field together; no single value is a trade instruction."),
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
        elif key in price_keys or key.startswith("sma_"):
            display = _format_price(float(value))
        elif isinstance(value, float):
            display = f"{value:,.3f}"
        else:
            display = str(value)
        rows.append({"name": _indicator_label(key), "value": display})
    return rows


def _indicator_label(key: str) -> str:
    labels = {
        "price_change_pct": "Price Change",
        "range_pct": "Range",
        "vwap": "VWAP",
        "rsi": "RSI",
        "macd": "MACD",
        "macd_signal": "MACD Signal",
        "macd_hist": "MACD Histogram",
        "atr": "ATR",
        "atr_pct": "ATR %",
        "avg_volume": "Average Volume",
        "volume_anomaly": "Relative Volume",
        "gap_pct": "Gap",
        "session_open": "Session Open",
        "session_high": "Session High",
        "session_low": "Session Low",
        "prior_high": "Prior High",
        "prior_low": "Prior Low",
        "prior_close": "Prior Close",
        "opening_range_high": "Opening Range High",
        "opening_range_low": "Opening Range Low",
        "opening_range_status": "Opening Range Status",
        "market_regime": "Market Regime",
        "benchmark_change_pct": "Benchmark Change",
        "relative_strength_pct": "Relative Strength",
    }
    if key.startswith("sma_"):
        return key.replace("sma_", "SMA ")
    return labels.get(key, _humanize_key(key))


def _format_price_range(value: tuple[float, float] | list[float] | None) -> str:
    if not value:
        return "-"
    return f"{_format_price(float(value[0]))} to {_format_price(float(value[1]))}"


def _format_targets(values: list[float]) -> str:
    return ", ".join(_format_price(float(value)) for value in values) if values else "-"


def _render_symbol_sidebar(settings) -> list[str]:
    st.sidebar.subheader("Symbols")
    if "selected_symbols_text" not in st.session_state:
        st.session_state.selected_symbols_text = ", ".join(settings.watchlist())
    if "loaded_watchlist_name" not in st.session_state:
        st.session_state.loaded_watchlist_name = settings.dashboard.get("default_watchlist", "default")

    watchlists = settings.raw.get("watchlists", {})
    watchlist_names = list(watchlists.keys()) or ["default"]
    watchlist_name = st.sidebar.selectbox(
        "Watchlist preset",
        options=watchlist_names,
        index=watchlist_names.index(settings.dashboard.get("default_watchlist", "default"))
        if settings.dashboard.get("default_watchlist", "default") in watchlists
        else 0,
        help="Choose a configured watchlist. Changing it replaces the selected symbols box.",
    )
    if watchlist_name != st.session_state.loaded_watchlist_name:
        st.session_state.selected_symbols_text = ", ".join(settings.watchlist(watchlist_name))
        st.session_state.loaded_watchlist_name = watchlist_name

    if st.sidebar.button("Clear selected symbols", help="Remove all currently selected symbols from scan/news/backtest defaults."):
        st.session_state.selected_symbols_text = ""

    lookup_query = st.sidebar.text_input("Search symbol", placeholder="AAPL, BRK.B, Palantir, Nvidia", help="Find a ticker by symbol or company name, then add it to the selected list.")
    lookup_results = search_symbols(lookup_query, limit=12) if lookup_query else []
    if lookup_results:
        result_options = [f"{item['symbol']} - {item['name']}" for item in lookup_results]
        selected_lookup = st.sidebar.selectbox("Search results", options=result_options)
        if st.sidebar.button("Add selected symbol", help="Append the selected search result to the symbol list."):
            _append_symbol_to_state(selected_lookup.split(" - ", 1)[0])
    elif lookup_query:
        st.sidebar.caption("No matching company or valid ticker found.")

    st.sidebar.text_area(
        "Selected symbols",
        key="selected_symbols_text",
        height=96,
        placeholder="AAPL, NVDA, PLTR, SOFI",
        help="Comma- or newline-separated symbols used by scanner, news, backtest, and default trade-plan symbol.",
    )
    symbols = clean_symbol_list(_parse_symbols(st.session_state.selected_symbols_text))
    st.sidebar.caption(f"{len(symbols)} selected: {', '.join(symbols[:8])}{'...' if len(symbols) > 8 else ''}")
    return symbols


def _append_symbol_to_state(symbol: str) -> None:
    symbols = clean_symbol_list(_parse_symbols(st.session_state.get("selected_symbols_text", "")) + [symbol])
    st.session_state.selected_symbols_text = ", ".join(symbols)


def _render_news(settings, symbols: list[str]) -> None:
    _render_news_capability_note(settings)
    selected = st.multiselect("Symbols", options=symbols, default=symbols[: min(5, len(symbols))], help="Symbols to fetch and summarize news for.")
    headline_limit = st.slider("Max headlines", min_value=5, max_value=200, value=50, step=5, help="Maximum headline rows returned to the table across configured sources. Free providers can still return fewer items than requested.")
    if st.button("Get News", type="primary", help="Aggregate configured news sources, optional article excerpts, and LLM summaries."):
        progress = st.progress(0, text="Starting news refresh")
        status = st.empty()

        def update_progress(value: float, message: str) -> None:
            status.caption(message)
            progress.progress(float(value), text=message)

        try:
            feed = build_news_feed(selected or symbols, settings, limit=headline_limit, progress_callback=update_progress)
        except NewsAnalysisError as exc:
            progress.empty()
            status.empty()
            st.error(str(exc))
            st.info("Start LocalDeploy on the configured endpoint or enable llm.fallback_to_heuristic in the config.")
            return
        progress.progress(1.0, text="News feed ready")
        st.session_state.latest_news_feed = feed
        _persist_dashboard_cache(settings)
    feed = st.session_state.get("latest_news_feed")
    if feed is not None:
        _render_news_feed(feed, headline_limit)


def _render_news_feed(feed: dict, headline_limit: int) -> None:
    items = feed["headlines"]
    if not items:
        st.info("No recent headlines returned by the configured free provider.")
        return
    st.caption(f"Showing last completed news aggregation: {', '.join(feed.get('symbols', [])) or '-'}")
    df = pd.DataFrame(items)
    requested = int(feed.get("requested_headline_limit", headline_limit))
    returned = int(feed.get("returned_headline_count", len(df)))
    bullish = int((df["sentiment"] == "bullish").sum()) if "sentiment" in df else 0
    bearish = int((df["sentiment"] == "bearish").sum()) if "sentiment" in df else 0
    neutral = len(df) - bullish - bearish
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Headlines", f"{returned}/{requested}", help="Returned headlines versus the amount requested by the slider.")
    c2.metric("Bullish", bullish, help="Headlines classified as directionally positive.")
    c3.metric("Bearish", bearish, help="Headlines classified as directionally negative.")
    c4.metric("Neutral", neutral, help="Headlines without a clear directional tilt.")
    c5.metric("Analysis", feed["analysis_provider"], help=HELP_TEXT["analysis_provider"])
    _render_news_coverage(feed.get("coverage", {}))
    if returned < requested:
        st.info(
            f"Requested {requested} headlines and received {returned}. "
            "That usually means the configured free sources did not expose more recent linked items for the selected symbols."
        )
    provider_label = str(feed.get("analysis_provider", ""))
    if "heuristic_fallback" in provider_label:
        st.warning("LLM summarization failed; this feed is using heuristic fallback summaries.")
    elif "llm_error" in provider_label:
        st.warning("One or more symbol summaries failed LLM parsing. The affected symbol keeps its source links for manual review.")
    elif provider_label == "heuristic":
        st.info("LLM summarization is disabled; this feed is using heuristic summaries.")

    st.subheader("Stock News Summary")
    for summary in feed["summaries"]:
        _render_symbol_news_summary(summary)

    st.subheader("Headline Table")
    headline_columns = [
        column
        for column in ["symbol", "classification_provider", "category", "sentiment", "impact", "day_trader_relevance", "published", "provider", "title", "url"]
        if column in df.columns
    ]
    st.dataframe(
        df[headline_columns],
        width="stretch",
        hide_index=True,
        column_config=_headline_column_config(),
    )


def _render_backtest(settings, symbols: list[str]) -> None:
    st.subheader("Backtest")
    st.write("Replay the configured signal and risk rules on historical daily bars. Use this to reject weak ideas and inspect behavior, not to prove future profit.")
    st.info("Historical news and AI summaries are excluded because the free providers do not supply point-in-time news archives. This test measures the repeatable price, technical, and risk path.")
    available_symbols = clean_symbol_list(symbols or settings.watchlist())
    if not available_symbols:
        st.warning("Add at least one symbol in the sidebar before running a historical check.")
        return
    selected_symbols = st.multiselect(
        "Symbols to simulate",
        options=available_symbols,
        default=available_symbols[:1],
        help="Start with one symbol so the first run is easy to read. Add symbols when you want a broader sample.",
    )
    depth = st.selectbox(
        "Simulation depth",
        options=list(_BACKTEST_DEPTH_PRESETS),
        index=1,
        help="Quick is a smoke test. Standard is the recommended first review. Deeper fetches more history and takes longer.",
    )
    run_settings = _backtest_settings_for_depth(settings, depth)
    profile = _BACKTEST_DEPTH_PRESETS[depth]
    st.caption(str(profile["description"]))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("History", str(profile["period"]), help="How much daily price history is downloaded for each selected symbol.")
    c2.metric("Warm-up", f"{profile['lookback_rows']} bars", help="Bars gathered before the first simulated decision. The indicators and model need this history before they can vote.")
    c3.metric("Hold", f"{profile['holding_period_days']} days", help="Maximum number of trading days before an open simulated trade exits by time.")
    c4.metric("Re-check", f"Every {profile['evaluation_step_days']} days", help="How often history asks whether a fresh setup exists. Wider spacing is faster but produces fewer evaluations.")
    with _remembered_expander("Costs And Position Sizing", "backtest_simulation_settings", expanded=False):
        st.write("The simulation includes configured slippage and commission. Actionable trades use the same planned position sizing as the Trade Plan risk layer.")
        st.dataframe(
            pd.DataFrame(
                [
                    {"field": "Initial capital", "value": _format_price(float(run_settings.backtest.get("initial_capital", 0)))},
                    {"field": "Slippage", "value": f"{run_settings.backtest.get('slippage_bps', '-')} bps"},
                    {"field": "Commission", "value": _format_price(float(run_settings.backtest.get("commission_per_trade", 0)))},
                    {"field": "Model subset", "value": ", ".join(str(name) for name in run_settings.backtest.get("model_subset", [])) or "configured models"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    if st.button("Run Historical Check", type="primary", disabled=not selected_symbols, help="Replay the configured historical signal path for the selected symbols."):
        progress = st.progress(0, text="Preparing backtest")
        status = st.empty()

        def update_progress(value: float, message: str) -> None:
            status.caption(message)
            progress.progress(float(value), text=message)

        try:
            with st.spinner("Running historical simulation"):
                report = run_backtest(run_settings, symbols=selected_symbols, progress_callback=update_progress)
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"Historical check failed: {exc}")
            st.info("Try one symbol with Quick check first. If it still fails, inspect the market-data provider and network connection.")
            return
        progress.progress(1.0, text="Backtest ready")
        status.caption("Historical check complete")
        st.session_state.latest_backtest_report = report
        st.session_state.latest_backtest_symbols = selected_symbols
        st.session_state.latest_backtest_depth = depth
        _persist_dashboard_cache(settings)
    report = st.session_state.get("latest_backtest_report")
    if report is not None:
        completed_symbols = st.session_state.get("latest_backtest_symbols", [])
        completed_depth = st.session_state.get("latest_backtest_depth", "")
        if completed_symbols:
            st.caption(f"Showing last completed historical check: {', '.join(completed_symbols)}{f' · {completed_depth}' if completed_depth else ''}")
        _render_backtest_report(report)


def _render_backtest_report(report) -> None:
    evaluations = int(getattr(report, "evaluations", 0))
    trades = int(getattr(report, "trades", 0))
    no_trades = int(getattr(report, "no_trades", 0))
    st.subheader("Result Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluations", evaluations, help="Historical moments where the strategy checked whether a new setup existed.")
    c2.metric("Trades", trades, help="Evaluations that passed the configured signal and risk gates.")
    c3.metric("Skipped", no_trades, help="Evaluations where the strategy correctly chose not to open a fresh position.")
    c4.metric("No-trade rate", _format_percent(float(getattr(report, "no_trade_rate", 0.0))), help=HELP_TEXT["backtest_no_trade_rate"])
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Account return", _format_percent(_backtest_total_return(report)), help="Simulated account change after position sizing, stop/target exits, slippage, and commission.")
    c6.metric("Win rate", _format_percent(float(getattr(report, "win_rate", 0.0))), help=HELP_TEXT["backtest_win_rate"])
    c7.metric("Max drawdown", _format_percent(float(getattr(report, "max_drawdown", 0.0))), help=HELP_TEXT["drawdown"])
    c8.metric("Sharpe-like", f"{float(getattr(report, 'sharpe_like', 0.0)):.2f}", help=HELP_TEXT["backtest_sharpe_like"])
    st.caption(f"Historical range: {getattr(report, 'start', '-') or '-'} to {getattr(report, 'end', '-') or '-'}")
    for level, message in _backtest_interpretation_messages(report):
        getattr(st, level)(message)
    symbol_stats = list(getattr(report, "symbol_stats", []) or [])
    if symbol_stats:
        st.subheader("Coverage By Symbol")
        st.dataframe(pd.DataFrame(symbol_stats), width="stretch", hide_index=True)
    if report.equity_curve:
        st.subheader("Simulated Account Curve")
        st.caption("The line moves only when a simulated trade closes. Flat sections mean the strategy stayed out.")
        equity_df = pd.DataFrame(report.equity_curve)
        st.line_chart(equity_df.set_index("date")["equity"])
    trade_rows = _backtest_trade_rows(report)
    if trade_rows:
        st.subheader("Executed Trade Outcomes")
        trade_log_df = _percent_display(
            pd.DataFrame(trade_rows),
            ["return", "trade_return", "confidence", "max_adverse_excursion", "max_favorable_excursion"],
        )
        st.dataframe(trade_log_df, width="stretch", hide_index=True, column_config=_trade_log_column_config())
    skip_rows = _backtest_skip_reason_rows(report)
    if skip_rows:
        with _remembered_expander("Why Setups Were Skipped", "backtest_skip_reasons", expanded=True):
            st.write("A high skip count is not automatically a failure. It shows which gates reject entries most often.")
            st.dataframe(pd.DataFrame(skip_rows), width="stretch", hide_index=True)
    if report.trade_log:
        with _remembered_expander("Full Evaluation Log", "backtest_trade_log", expanded=False):
            st.caption("Detailed audit trail for every historical evaluation, including skipped setups.")
            trade_log_df = _percent_display(
                pd.DataFrame(report.trade_log),
                ["return", "trade_return", "confidence", "max_adverse_excursion", "max_favorable_excursion"],
            )
            st.dataframe(trade_log_df, width="stretch", hide_index=True, column_config=_trade_log_column_config())


def _backtest_settings_for_depth(settings: Settings, depth: str) -> Settings:
    profile = _BACKTEST_DEPTH_PRESETS.get(depth, _BACKTEST_DEPTH_PRESETS["Standard review"])
    raw = deepcopy(settings.raw)
    raw.setdefault("data", {})["period"] = profile["period"]
    raw.setdefault("backtest", {}).update(
        {
            "lookback_rows": profile["lookback_rows"],
            "holding_period_days": profile["holding_period_days"],
            "evaluation_step_days": profile["evaluation_step_days"],
        }
    )
    return Settings(raw=raw, path=settings.path)


def _backtest_total_return(report) -> float:
    configured = getattr(report, "total_return", None)
    if configured is not None:
        return float(configured)
    curve = list(getattr(report, "equity_curve", []) or [])
    initial = float(getattr(report, "initial_capital", 0.0) or 0.0)
    if not curve or not initial:
        return 0.0
    return (float(curve[-1]["equity"]) / initial) - 1


def _backtest_interpretation_messages(report) -> list[tuple[str, str]]:
    evaluations = int(getattr(report, "evaluations", 0))
    trades = int(getattr(report, "trades", 0))
    no_trade_rate = float(getattr(report, "no_trade_rate", 0.0))
    messages: list[tuple[str, str]] = []
    if evaluations == 0:
        return [("error", "No historical evaluations were produced. Choose a longer simulation depth or verify that the market-data provider returned enough daily bars.")]
    if trades == 0:
        messages.append(("warning", "No simulated trades passed the configured gates. This is a useful diagnostic, not a performance result. Review the skip reasons below or run a deeper sample."))
    elif trades < 10:
        messages.append(("warning", f"Only {trades} simulated trade{'s' if trades != 1 else ''} passed the gates. The sample is too small for performance conclusions; use the outcomes to inspect logic only."))
    else:
        messages.append(("info", f"{trades} simulated trades passed the gates. Read account return together with drawdown, skip reasons, and individual outcomes."))
    if no_trade_rate >= 0.90:
        messages.append(("info", f"The strategy skipped {no_trade_rate:.0%} of evaluations. That can be appropriate for a selective strategy, but inspect the dominant skip reasons to confirm the gates are not overly restrictive."))
    messages.append(("caption", "Backtests can expose weak logic and unrealistic assumptions. They cannot establish that a strategy will remain profitable in live trading."))
    return messages


def _backtest_trade_rows(report) -> list[dict]:
    return [
        row
        for row in list(getattr(report, "trade_log", []) or [])
        if str(row.get("exit_reason", "")) not in {"no_trade", "session_blocked"}
    ]


def _backtest_skip_reason_rows(report) -> list[dict[str, str | int]]:
    counts: dict[str, int] = {}
    for row in list(getattr(report, "trade_log", []) or []):
        if str(row.get("exit_reason", "")) not in {"no_trade", "session_blocked"}:
            continue
        reason = str(row.get("skip_reasons") or row.get("top_reason") or "unspecified")
        counts[reason] = counts.get(reason, 0) + 1
    return [
        {"skip_reason": reason, "evaluations": count}
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _render_settings_tab(settings, symbols: list[str]) -> None:
    st.subheader("Runtime Settings")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data", str(settings.data.get("provider", "-")), help="Configured market data provider.")
    c2.metric("Period", f"{settings.data.get('period', '-')}/{settings.data.get('interval', '-')}", help="Historical period and bar interval used by default daily analysis.")
    c3.metric("Models", ", ".join(settings.enabled_models()) or "none", help="Prediction models currently enabled in config.")
    c4.metric("Symbols", len(symbols), help="Number of selected symbols in the sidebar.")

    sections = [
        {"section": "Config file", "value": str(settings.path)},
        {"section": "Context sources", "value": ", ".join(str(source) for source in settings.context_agent.get("sources", [])) or "none"},
        {"section": "News LLM", "value": _settings_llm_summary(settings)},
        {"section": "Risk per trade", "value": _format_percent(float(settings.risk.get("max_risk_per_trade_pct", 0)))},
        {"section": "Min risk/reward", "value": str(settings.risk.get("min_risk_reward", "-"))},
        {"section": "Selected symbols", "value": ", ".join(symbols) or "-"},
    ]
    st.dataframe(pd.DataFrame(sections), width="stretch", hide_index=True)

    with _remembered_expander("Advanced Raw Config", "settings_raw_config", expanded=False):
        st.json(to_serializable(settings.raw))


def _settings_llm_summary(settings) -> str:
    llm = settings.context_agent.get("news_analysis", {}).get("llm", {})
    enabled = "enabled" if llm.get("enabled", False) else "disabled"
    return f"{enabled} ({llm.get('provider', 'heuristic')})"


def _render_news_capability_note(settings) -> None:
    llm_cfg = settings.context_agent.get("news_analysis", {}).get("llm", {})
    scrape_cfg = settings.context_agent.get("news_analysis", {}).get("article_scraping", {})
    classifier_cfg = settings.context_agent.get("news_analysis", {}).get("headline_classifier", {})
    sources = ", ".join(str(source) for source in settings.context_agent.get("sources", [])) or "none"
    llm_status = "enabled" if llm_cfg.get("enabled", False) else "disabled"
    provider = str(llm_cfg.get("provider", "heuristic"))
    fallback = "enabled" if llm_cfg.get("fallback_to_heuristic", True) else "disabled"
    scraping = "enabled" if scrape_cfg.get("enabled", False) else "disabled"
    classifier = str(classifier_cfg.get("provider", "keyword")) if classifier_cfg.get("enabled", False) else "keyword"
    classifier_fallback = "enabled" if classifier_cfg.get("fallback_to_keyword", True) else "disabled"
    with _remembered_expander("News coverage and limits", "news_coverage_limits", expanded=False):
        st.write(
            "This feed aggregates configured headline metadata and provider links. "
            "When article scraping is enabled, it fetches a short article excerpt for the LLM; it does not store full article bodies."
        )
        st.write("Headline rows use the configured classifier when it succeeds. Rows explicitly show their classifier provenance; keyword fallback is never hidden.")
        st.write(f"Configured sources: {sources}. Headline classifier: {classifier}. Keyword fallback: {classifier_fallback}. LLM summarizer: {llm_status} ({provider}). Heuristic summary fallback: {fallback}. Article excerpts: {scraping}.")
        st.write("If fallback is disabled and the configured LLM endpoint is unavailable, the News tab stops instead of producing a heuristic summary.")


def _render_news_coverage(coverage: dict) -> None:
    if not coverage:
        return
    sources = ", ".join(coverage.get("configured_sources", [])) or "none"
    headline_sources = ", ".join(coverage.get("headline_sources", [])) or str(coverage.get("headline_provider", "configured"))
    scrape_status = f"article excerpts ({coverage.get('article_excerpt_count', 0)} fetched)" if coverage.get("article_body_scraping") else "headline/link mode"
    st.caption(
        f"Coverage: {sources}; active headline sources: {headline_sources}; LLM provider: {coverage.get('llm_provider', 'heuristic')}; "
        f"headline classifiers used: {', '.join(coverage.get('classification_providers', [])) or 'keyword'}; "
        f"summary fallback: {'enabled' if coverage.get('fallback_to_heuristic') else 'disabled'}; mode: {scrape_status}."
    )
    source_counts = coverage.get("source_counts", {})
    if source_counts:
        with _remembered_expander("Source counts", "news_source_counts", expanded=False):
            st.dataframe(
                pd.DataFrame([{"source": source, "headlines": count} for source, count in sorted(source_counts.items())]),
                width="stretch",
                hide_index=True,
            )


def _render_symbol_news_summary(summary: dict) -> None:
    st.markdown(f"### {summary['symbol']}")
    st.write(summary.get("grand_summary", "No summary available."))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sources", summary.get("source_count", 0), help=HELP_TEXT["news_sources"])
    c2.metric("Headlines", summary.get("headline_count", 0), help="Number of headlines used for this symbol summary.")
    c3.metric("Bullish", summary.get("bullish_count", 0), help="Count of headlines classified as positive for this symbol.")
    c4.metric("Bearish", summary.get("bearish_count", 0), help="Count of headlines classified as negative for this symbol.")
    c5.metric("Category", str(summary.get("dominant_category", "other")).replace("_", " "), help="Most common catalyst/category in the selected headlines.")

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
        width="stretch",
        hide_index=True,
        column_config={
            "question": st.column_config.TextColumn("Question", help="Trader question the news summary is trying to answer."),
            "answer": st.column_config.TextColumn("Answer", help="LLM or heuristic summary for this decision question."),
        },
    )
    if summary.get("analysis_provider"):
        st.caption(f"Summary provider: {summary.get('analysis_provider')}")
    if summary.get("analysis_provider") == "heuristic_fallback":
        st.warning(f"Heuristic fallback summary. LLM error: {summary.get('llm_error', 'unknown error')}")
    if summary.get("analysis_provider") == "llm_error":
        st.warning(f"LLM summary failed. Source links are still available. Error: {summary.get('llm_error', 'unknown error')}")
    if summary.get("llm_notes"):
        st.write("LLM notes")
        st.write(summary.get("llm_notes"))

    sources = pd.DataFrame(summary.get("sources", []))
    with _remembered_expander(f"Open {summary.get('source_count', 0)} linked sources", f"news_sources_{summary.get('symbol', 'unknown')}", expanded=False):
        if sources.empty:
            st.info("No linked sources were returned.")
        else:
            columns = [column for column in ["provider", "published", "classification_provider", "category", "sentiment", "impact", "title", "url"] if column in sources.columns]
            st.dataframe(
                sources[columns],
                width="stretch",
                hide_index=True,
                column_config=_headline_column_config(),
            )


def _headline_column_config() -> dict:
    return {
        "symbol": st.column_config.TextColumn("Symbol", help="Ticker associated with the headline."),
        "category": st.column_config.TextColumn("Category", help="Catalyst type detected from the headline and excerpt."),
        "sentiment": st.column_config.TextColumn("Sentiment", help=HELP_TEXT["sentiment"]),
        "impact": st.column_config.NumberColumn("Impact", format="%.2f", help=HELP_TEXT["impact"]),
        "day_trader_relevance": st.column_config.NumberColumn("Relevance", format="%.2f", help=HELP_TEXT["headline_relevance"]),
        "published": st.column_config.TextColumn("Published", help="Timestamp returned by the news provider when available."),
        "provider": st.column_config.TextColumn("Provider", help="Source provider that returned the item."),
        "classification_provider": st.column_config.TextColumn("Classifier", help="Analyzer that classified this row. `keyword` means the built-in fallback; LLM-backed rows show their configured provider."),
        "title": st.column_config.TextColumn("Title", help="Headline used for catalyst and sentiment analysis."),
        "url": st.column_config.LinkColumn("Link", help="Source link for manual review."),
    }


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
        symbol = c1.text_input("Symbol", value=(symbols[0] if symbols else str(settings.dashboard.get("default_symbol", "AAPL"))), help="Ticker reviewed in this journal entry.").upper()
        action = c2.selectbox("Action", _JOURNAL_ACTIONS, help="The action you took or the system suggested.")
        setup_type = c3.selectbox("Setup", _JOURNAL_SETUPS, help="The setup pattern you were evaluating.")
        outcome = c4.selectbox("Outcome", _JOURNAL_OUTCOMES, help="Final or current result of the idea.")
        c5, c6, c7, c8 = st.columns(4)
        followed_plan = c5.checkbox("Followed plan", help="Check this only if entry/exit followed the written plan.")
        risk_respected = c6.checkbox("Risk respected", help="Check this only if position size and stop respected your risk limits.")
        entry_quality = c7.slider("Entry quality", 1, 5, 3, help="Rate whether the entry was patient, level-based, and not chased.")
        exit_quality = c8.slider("Exit quality", 1, 5, 3, help="Rate whether the exit followed stop/target/invalidation rules.")
        emotional_state = st.selectbox("State", _JOURNAL_STATES, help="Mental state can explain bad process even when the P/L looks fine.")
        notes = st.text_area("Notes", help="Write what mattered: catalyst, entry reason, stop, target, mistake, or lesson.")
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
        st.dataframe(df.sort_values("timestamp", ascending=False), width="stretch", hide_index=True)
        _render_journal_edit_controls(settings, recent)
    else:
        st.info("No journal entries yet.")


def _render_journal_edit_controls(settings, recent: list[dict]) -> None:
    with _remembered_expander("Edit or delete an entry", "journal_edit_delete", expanded=False):
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


def _humanize_key(key: str) -> str:
    return key.replace("_", " ").title()


def _format_scanner_value(key: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
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
    price_keys = {"price", "prior_high", "prior_low", "session_open", "opening_range_high", "opening_range_low"}
    volume_keys = {"volume", "avg_volume"}
    if key in percent_keys:
        return f"{float(value):.2f}%"
    if key in price_keys:
        return _format_price(float(value))
    if key in volume_keys:
        return f"{float(value):,.0f}"
    if key in {"volume_anomaly", "risk_reward"}:
        return f"{float(value):.2f}"
    if key in {"rank_score", "score"}:
        return f"{float(value):.3f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _scanner_column_config(compact: bool = False) -> dict:
    config = {
        "symbol": st.column_config.TextColumn("Symbol", width="small", help="Ticker being evaluated."),
        "bias": st.column_config.TextColumn("Bias", width="small", help=HELP_TEXT["bias"]),
        "signal_action": st.column_config.TextColumn("Signal", width="small", help="Directional signal before calendar and risk execution gates."),
        "action": st.column_config.TextColumn("Trade Action", width="small", help=HELP_TEXT["action"]),
        "price": st.column_config.NumberColumn("Price", format="$%.2f", help="Latest price used by the scanner row."),
        "change_pct": st.column_config.NumberColumn("Change", format="%.2f%%", help="Current percent move. Large moves can create opportunity, but also late-entry risk."),
        "volume_anomaly": st.column_config.NumberColumn("RVOL", format="%.2f", help="Relative volume. Above 1.0 means volume is above recent average."),
        "gap_pct": st.column_config.NumberColumn("Gap", format="%.2f%%", help=HELP_TEXT["gap_pct"]),
        "atr_pct": st.column_config.NumberColumn("ATR %", format="%.2f%%", help=HELP_TEXT["atr_pct"]),
        "prior_high": st.column_config.NumberColumn("Prior High", format="$%.2f", help="Previous bar/session high used as a reference level."),
        "prior_low": st.column_config.NumberColumn("Prior Low", format="$%.2f", help="Previous bar/session low used as a reference level."),
        "session_open": st.column_config.NumberColumn("Open", format="$%.2f", help="Current session open."),
        "opening_range_high": st.column_config.NumberColumn("OR High", format="$%.2f", help="High of the configured opening range."),
        "opening_range_low": st.column_config.NumberColumn("OR Low", format="$%.2f", help="Low of the configured opening range."),
        "extension_from_vwap_pct": st.column_config.NumberColumn("VWAP Dist", format="%.2f%%", help="Distance from VWAP. Large extension can make entries riskier."),
        "distance_to_support_pct": st.column_config.NumberColumn("Support Dist", format="%.2f%%", help="Distance to nearest support. Too far from support can widen stop risk."),
        "distance_to_resistance_pct": st.column_config.NumberColumn("Resistance Dist", format="%.2f%%", help="Distance to nearest resistance. Nearby resistance can cap upside."),
        "benchmark_change_pct": st.column_config.NumberColumn("Benchmark", format="%.2f%%", help="Move in the configured benchmark, usually SPY."),
        "relative_strength_pct": st.column_config.NumberColumn("Rel Strength", format="%.2f%%", help=HELP_TEXT["relative_strength"]),
        "confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%", help=HELP_TEXT["confidence"]),
        "score": st.column_config.NumberColumn("Score", format="%.3f", help=HELP_TEXT["score"]),
        "risk_reward": st.column_config.NumberColumn("R/R", format="%.2f", help=HELP_TEXT["risk_reward"]),
        "rank_score": st.column_config.NumberColumn("Rank", format="%.3f", help=HELP_TEXT["rank"]),
        "top_reason": st.column_config.TextColumn("Why", width="medium", help=HELP_TEXT["top_reason"]),
    }
    if compact:
        compact_keys = {
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
        }
        return {key: value for key, value in config.items() if key in compact_keys}
    return config


def _model_column_config() -> dict:
    return {
        "expected_return": st.column_config.NumberColumn("Expected Return", format="%.2f%%", help="Model-implied return over the configured horizon."),
        "confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%", help=HELP_TEXT["confidence"]),
        "predicted_price": st.column_config.NumberColumn("Predicted", format="$%.2f", help="Model-implied future price estimate."),
        "lower_bound": st.column_config.NumberColumn("Lower", format="$%.2f", help="Lower uncertainty bound when the model supplies one."),
        "upper_bound": st.column_config.NumberColumn("Upper", format="$%.2f", help="Upper uncertainty bound when the model supplies one."),
    }


def _risk_column_config() -> dict:
    return {
        "entry": st.column_config.NumberColumn("Entry", format="$%.2f", help=HELP_TEXT["entry"]),
        "stop_loss": st.column_config.NumberColumn("Stop", format="$%.2f", help=HELP_TEXT["stop"]),
        "stop_source": st.column_config.TextColumn("Stop anchor", help="Why this stop level was selected, such as support, VWAP, or ATR fallback."),
        "target_source": st.column_config.TextColumn("Target anchor", help="Why this target level was selected, such as R-multiple or structural resistance/support."),
        "risk_reward": st.column_config.NumberColumn("R/R", format="%.2f", help=HELP_TEXT["risk_reward"]),
        "risk_per_share": st.column_config.NumberColumn("Risk/Share", format="$%.2f", help="Entry minus stop for longs, or stop minus entry for shorts."),
        "planned_risk": st.column_config.NumberColumn("Planned Risk", format="$%.2f", help="Dollar risk implied by position size and stop distance."),
        "planned_position_value": st.column_config.NumberColumn("Position Value", format="$%.2f", help="Approximate notional value of the planned position."),
        "max_position_risk": st.column_config.NumberColumn("Max Risk", format="$%.2f", help="Configured maximum account risk for this trade."),
    }


def _context_column_config() -> dict:
    return {
        "context_confidence": st.column_config.NumberColumn("Context Confidence", format="%.1f%%", help=HELP_TEXT["context_confidence"]),
        "catalyst_score": st.column_config.NumberColumn("Catalyst Score", format="%.2f", help=HELP_TEXT["catalyst_score"]),
        "catalyst_freshness": st.column_config.NumberColumn("Freshness", format="%.1f%%", help=HELP_TEXT["freshness"]),
        "market_alignment": st.column_config.NumberColumn("Market Alignment", format="%.2f", help="Whether broad-market context supports the setup."),
        "sector_alignment": st.column_config.NumberColumn("Sector Alignment", format="%.2f", help="Whether sector movement supports the symbol setup."),
    }


def _backtest_column_config() -> dict:
    return {
        "win_rate": st.column_config.NumberColumn("Win Rate", format="%.2f%%", help=HELP_TEXT["backtest_win_rate"]),
        "average_return": st.column_config.NumberColumn("Avg Return", format="%.2f%%", help=HELP_TEXT["backtest_average_return"]),
        "max_drawdown": st.column_config.NumberColumn("Max Drawdown", format="%.2f%%", help=HELP_TEXT["drawdown"]),
        "sharpe_like": st.column_config.NumberColumn("Sharpe-like", format="%.2f", help=HELP_TEXT["backtest_sharpe_like"]),
        "no_trade_rate": st.column_config.NumberColumn("No Trade", format="%.2f%%", help=HELP_TEXT["backtest_no_trade_rate"]),
    }


def _trade_log_column_config() -> dict:
    return {
        "entry": st.column_config.NumberColumn("Entry", format="$%.2f", help=HELP_TEXT["entry"]),
        "stop_loss": st.column_config.NumberColumn("Stop", format="$%.2f", help=HELP_TEXT["stop"]),
        "target": st.column_config.NumberColumn("Target", format="$%.2f", help=HELP_TEXT["target"]),
        "exit_price": st.column_config.NumberColumn("Exit", format="$%.2f", help="Simulated exit price after stop, target, or time exit."),
        "return": st.column_config.NumberColumn("Return", format="%.2f%%", help="Return for the simulated evaluation period."),
        "trade_return": st.column_config.NumberColumn("Trade Return", format="%.2f%%", help="Position return after simulated exit and costs."),
        "r_multiple": st.column_config.NumberColumn("R", format="%.2f", help="Return measured in units of planned risk. 1R means reward equals initial risk."),
        "max_adverse_excursion": st.column_config.NumberColumn("MAE", format="%.2f%%", help=HELP_TEXT["mae"]),
        "max_favorable_excursion": st.column_config.NumberColumn("MFE", format="%.2f%%", help=HELP_TEXT["mfe"]),
        "confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%", help=HELP_TEXT["confidence"]),
        "score": st.column_config.NumberColumn("Score", format="%.3f", help=HELP_TEXT["score"]),
        "risk_reward": st.column_config.NumberColumn("R/R", format="%.2f", help=HELP_TEXT["risk_reward"]),
        "equity": st.column_config.NumberColumn("Equity", format="$%.2f", help="Simulated account equity after this trade/evaluation."),
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
