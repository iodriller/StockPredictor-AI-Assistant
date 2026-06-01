"""Central education content for beginner-friendly "Education mode".

One source of truth so the UI can explain *everything* consistently:
- TRADER_USE: a "how a trader actually uses this" one-liner, keyed to the same keys
  the dashboard's tooltip table (HELP_TEXT) already uses, so enabling education mode
  enriches every existing tooltip automatically.
- GLOSSARY: a plain-language one-liner for every term/acronym in the app.
- WORKSPACE_GUIDES / HORIZON_GUIDES / MODEL_GUIDES: short "what is this, what do I
  look at, what are the steps" walkthroughs for each part of the workflow.

This module is pure data + tiny formatters — no UI imports — so it stays testable
and reusable from the dashboard, API, or docs.
"""

from __future__ import annotations


# How a trader uses each metric. Keyed to the dashboard HELP_TEXT keys so tooltips
# can be enriched without touching every call site.
TRADER_USE: dict[str, str] = {
    "action": "Traders treat long/short as 'a setup worth risking money on now', and watch/no-trade as 'keep it on the screen, don't touch yet'.",
    "bias": "Decide direction first (long = expect up, short = expect down); everything else is about timing and risk.",
    "confidence": "Higher confidence = the inputs agree. Beginners should size smaller or skip when confidence is low.",
    "score": "Think of it as a vote from -1 (bearish) to +1 (bullish); near zero means 'no edge, wait'.",
    "rank": "Use it to sort a scan — start at the top, where movement, volume, and a clear reason line up.",
    "setup": "A label for trade quality. 'actionable' passed every check; the others tell you exactly what failed.",
    "top_reason": "The single biggest driver of the call — read it before anything else.",
    "trade_watch": "These are the only rows worth your attention today; everything else is noise to ignore.",
    "vwap": "The intraday 'fair price'. Buyers are in control above it, sellers below it; many traders only go long above VWAP.",
    "vwap_alignment": "A quick 'are we on the strong side?' check — above VWAP favors longs, below favors shorts.",
    "atr_pct": "How much the stock typically moves. Bigger ATR = wider stops and smaller position size to keep risk fixed.",
    "avg_rvol": "Volume vs normal. Above 1.5–2x means real interest/catalyst; low volume moves often fail.",
    "gap_pct": "An overnight jump signals news. Traders ask: is it continuing (go) or fading (fade)?",
    "rsi": "0–100 momentum gauge. >70 can be overbought (stretched), <30 oversold; trends can stay extended though.",
    "macd": "Momentum trend tool. Histogram above zero = momentum building up, below = building down.",
    "relative_strength": "Is it stronger than the market today? Traders prefer longs in names outperforming SPY.",
    "risk_reward": "Reward ÷ risk. Most traders need at least ~1.5–2 so winners pay for losers; below that, skip.",
    "entry": "A reference trigger price — not a market order. Wait for price to reach the plan, don't chase.",
    "entry_zone": "The price area where the idea is still valid. Buying above it ruins your risk/reward.",
    "stop": "Where you admit the idea is wrong and get out. Set it before entering, never widen it after.",
    "target": "Where you plan to take profit, usually near resistance/support and realistic for the volatility.",
    "invalidation": "The 'I was wrong' condition. If this happens, the trade thesis is dead — exit.",
    "position": "How many shares keep your loss within your max risk if the stop hits. Blank = no valid plan.",
    "catalyst_score": "How strong the news reason to move is. A catalyst plus price/volume confirmation is the classic setup.",
    "freshness": "How recent the news is. For day trading, a 30-minute-old catalyst matters far more than yesterday's.",
    "context_confidence": "How much usable news/context evidence there is. Low = the catalyst read is thin.",
    "sentiment": "The overall tone of the news (bullish/bearish). Supporting evidence only — confirm with price.",
    "impact": "Estimated push of a single headline: positive = bullish, negative = bearish, near zero = noise.",
    "headline_relevance": "How tradable a headline looks given its type, size, and freshness.",
    "analysis_provider": "Whether the news summary came from the AI model or a simple keyword fallback.",
    "article_excerpts": "Short article snippets fed to the AI for better summaries; not a full news feed.",
    "news_sources": "How many linked articles backed this summary — click to read them yourself.",
    "drawdown": "The worst peak-to-trough loss in the test. The main 'could I stomach this?' stress number.",
    "mae": "Worst unrealized loss during a trade — tells you if your stop was nearly hit before it worked.",
    "mfe": "Best unrealized gain during a trade — tells you if you left a lot on the table.",
    "backtest_win_rate": "Share of simulated trades that won. Read it with average win/loss size, not alone.",
    "backtest_average_return": "Average result per simulated trade. A logic sanity check, not a promise of future returns.",
    "backtest_sharpe_like": "Return vs choppiness. Higher is steadier; unstable on small samples.",
    "backtest_no_trade_rate": "How often it correctly sat out. A healthy strategy says 'no' a lot.",
}


# Plain-language one-liners for every term/acronym. (term, definition).
GLOSSARY: list[tuple[str, str]] = [
    ("Long", "Buying to profit if the price goes up."),
    ("Short", "Selling borrowed shares to profit if the price goes down."),
    ("Watch", "No trade yet — interesting enough to keep on your screen."),
    ("No-trade", "The conditions for a good trade are not met; sitting out is the call."),
    ("Low-confidence", "There's a lean, but the signals are too weak/mixed to act."),
    ("Bias", "Your expected direction: long (up) or short (down)."),
    ("Score", "A combined -1 to +1 vote across all signals; + is bullish, − is bearish."),
    ("Confidence", "How strongly the signals agree (0–100%)."),
    ("Horizon", "How long you plan to hold: intraday (hours), swing (days–weeks), position (weeks–months)."),
    ("OHLC", "Open, High, Low, Close — the four prices that define each candle/bar."),
    ("Candlestick", "A bar showing open/high/low/close for a period; green up, red down."),
    ("VWAP", "Volume-Weighted Average Price — the average price weighted by volume; intraday 'fair value'."),
    ("SMA", "Simple Moving Average — the average close over N bars; smooths the trend."),
    ("EMA", "Exponential Moving Average — like SMA but weights recent bars more."),
    ("MA alignment", "When short, medium, and long averages stack in order — a clean trend."),
    ("RSI", "Relative Strength Index (0–100) — momentum gauge; >70 overbought, <30 oversold."),
    ("MACD", "Moving Average Convergence Divergence — a momentum/trend indicator."),
    ("ATR", "Average True Range — the typical price move per bar; used to size stops."),
    ("RVOL", "Relative Volume — today's volume vs normal; >1 means heavier than usual."),
    ("Gap", "An overnight jump between yesterday's close and today's open, usually news-driven."),
    ("Support", "A price floor where buyers have stepped in before."),
    ("Resistance", "A price ceiling where sellers have stepped in before."),
    ("Opening range", "The high/low of the first minutes of trading; breakouts from it are common setups."),
    ("Premarket", "Trading before the regular session open; shows early reaction to news."),
    ("Relative strength", "How a stock performs versus the broad market (e.g. SPY)."),
    ("Benchmark", "The index you compare against, usually SPY (S&P 500)."),
    ("Regime", "The market's current behavior: trending, choppy, or high/low volatility."),
    ("Risk/Reward (R/R)", "Potential reward divided by risk; aim for ~2:1 or better."),
    ("R-multiple", "Profit/loss measured in units of your initial risk; +1R means you made what you risked."),
    ("Entry", "The price at which you plan to get in."),
    ("Stop / Stop-loss", "The price where you exit a losing trade to cap the loss."),
    ("Target", "The price where you plan to take profit."),
    ("Invalidation", "The condition that proves the trade idea wrong."),
    ("Position size", "How many shares to buy so a stop-out only loses your planned risk."),
    ("Liquidity", "How easily you can get in/out; thin (low-volume) names are risky."),
    ("Slippage", "The difference between expected and actual fill price."),
    ("Commission", "The broker's fee per trade."),
    ("PDT", "Pattern Day Trader rule — US accounts under $25k are limited to 3 day trades per 5 days."),
    ("MAE", "Maximum Adverse Excursion — the worst the trade looked before it resolved."),
    ("MFE", "Maximum Favorable Excursion — the best the trade looked before it resolved."),
    ("Win rate", "The percentage of trades that were profitable."),
    ("Drawdown", "The largest drop from a peak in account value."),
    ("Sharpe (Sharpe-like)", "A return-versus-volatility score; higher is a smoother ride."),
    ("Catalyst", "A news event (earnings, upgrade, deal) that can move the stock."),
    ("Sentiment", "The overall positive/negative tone of the news."),
    ("Stance", "The AI's net read of the news as a direction plus a conviction (0–1)."),
    ("VIX", "The market's 'fear gauge'; higher means bigger, faster swings expected."),
    ("Baseline model", "A recency-weighted straight-line trend forecast."),
    ("Gaussian Process (GP)", "A model that estimates drift and uncertainty from recent returns."),
    ("ARIMA", "A classic time-series forecast; on liquid stocks it's often near a coin-flip."),
    ("Momentum model", "Reads moving-average alignment and recent momentum — closest to what the chart shows."),
    ("AR (p)", "ARIMA's auto-regressive term: how many past prices feed the forecast."),
    ("Diff (d)", "ARIMA's differencing: how many times the series is de-trended before fitting."),
    ("MA (q)", "ARIMA's moving-average term: how many past forecast errors feed the forecast."),
    ("Kernel", "The Gaussian Process's smoothness assumption (Matern = more reactive, RBF = smoother)."),
    ("Drift", "The average per-step return a model expects — the gentle slope of the forecast."),
    ("Continuation factor", "How much of recent momentum the momentum model projects forward."),
    ("Recency emphasis", "How strongly the trend fit favors recent bars over old history."),
    ("Trust balance", "Your dial for how much the decision leans on the ML models vs the News/AI read."),
    ("Conviction", "The AI's confidence (0–1) in its bullish/bearish stance on the news."),
    ("Spread", "The gap between the best buy and sell price; wide spreads cost you on entry/exit."),
    ("Float", "The number of shares available to trade; low float can mean violent moves."),
    ("Halt", "A temporary trading pause, often after huge moves or pending news."),
    ("Time-of-day RVOL", "Volume vs what's normal for this exact time of the session."),
    ("Setup quality", "The risk layer's verdict: actionable, or why not (liquidity, volatility, R/R, etc.)."),
    ("No-trade flag", "A specific reason the tool says 'don't trade this right now'."),
    ("Hard block", "A non-negotiable stop (e.g. earnings in 24h, market closed) that forces no-trade."),
    ("Score attribution", "The white-box breakdown of how much each input moved the final score."),
    ("LLM", "Large Language Model — the AI that reads and summarizes the news."),
    ("LocalDeploy", "Running the news AI on your own machine instead of a cloud API."),
    ("ETA", "Estimated time to finish — shown on the analysis progress bar."),
]


WORKSPACE_GUIDES: dict[str, dict] = {
    "scanner": {
        "title": "Scanner — find what's worth looking at",
        "what": "A ranked list of your symbols by how tradable they look right now.",
        "look_at": ["Action (long/short/watch)", "Rank", "RVOL (volume vs normal)", "Gap", "Relative strength", "the 'Why' reason"],
        "steps": [
            "Scan your symbols and start at the top of the rank.",
            "Keep rows with a clear direction, real volume (RVOL > ~1.5), and a reason you understand.",
            "Open the strongest one in Trade Plan for the full read.",
        ],
    },
    "trade_plan": {
        "title": "Trade Plan — the full read on one stock",
        "what": "Everything for one symbol: the verdict, why, the news, the chart, and a risk plan.",
        "look_at": ["the verdict banner", "Why this isn't actionable (if shown)", "How news shaped the decision", "Entry / Stop / Target / R/R"],
        "steps": [
            "Pick a horizon (swing ≈ days–weeks is the most realistic for most beginners).",
            "Read the one-line verdict, then the 'why'.",
            "Check the news panel: is there a real, fresh catalyst?",
            "If actionable, confirm Entry/Stop/Target and that Risk/Reward is at least ~1.5.",
            "Decide yourself — the tool is decision support, not a buy/sell button.",
        ],
    },
    "news": {
        "title": "News — why a stock is moving",
        "what": "Recent headlines per symbol, summarized into catalyst, risk, and tradeability.",
        "look_at": ["the grand summary", "Catalyst vs Risk", "freshness", "the linked sources"],
        "steps": ["Read the summary.", "Check freshness — fresh catalysts matter most.", "Open a couple of sources to verify before trusting it."],
    },
    "backtest": {
        "title": "Backtest — sanity-check the logic on history",
        "what": "A simulation of the strategy on past data. A logic check, NOT proof of future profit.",
        "look_at": ["Win rate with average win/loss", "Max drawdown (can you stomach it?)", "No-trade rate"],
        "steps": ["Run it.", "Read win rate together with drawdown.", "Treat results skeptically — past ≠ future."],
    },
    "journal": {
        "title": "Journal — review your own process",
        "what": "A private log to rate whether you followed your plan and respected risk.",
        "look_at": ["Followed plan?", "Risk respected?", "Entry/exit quality", "your emotional state"],
        "steps": ["Log every trade.", "Be honest about process, not just profit/loss.", "Review weekly for repeated mistakes."],
    },
    "settings": {
        "title": "Settings — how the tool is configured",
        "what": "Providers, models, risk limits, and the education content you're reading now.",
        "look_at": ["which models are on", "risk per trade", "min risk/reward"],
        "steps": ["Glance at risk limits.", "Use Model & signal tuning on the Trade Plan to experiment safely."],
    },
}


HORIZON_GUIDES: dict[str, str] = {
    "intraday": "Intraday: in and out the same day (minutes–hours). Leans on VWAP, opening range, and volume. Fast and demanding — hardest for beginners.",
    "swing": "Swing: hold days to a couple of weeks. Often the most realistic horizon for retail — enough time for a thesis to play out without watching every tick.",
    "position": "Position: hold weeks to months. Trend-following; less noise, but ties up capital and needs patience.",
}


MODEL_GUIDES: dict[str, dict] = {
    "momentum": {
        "what": "Reads moving-average alignment, the trend's slope, and recent momentum.",
        "why": "It's the closest to what you see on the chart, so it catches a stock that's bullish *now* — even right after a turn.",
    },
    "baseline": {
        "what": "A recency-weighted straight-line trend fit over the lookback.",
        "why": "Gives the prevailing direction; weighting recent bars stops an old downtrend from masking a fresh reversal.",
    },
    "gaussian_process": {
        "what": "Estimates drift and an uncertainty band from recent returns.",
        "why": "Used as a confirmation + uncertainty check, not a precise price target.",
    },
    "arima": {
        "what": "A classic statistical time-series forecast on the closing prices.",
        "why": "Included as a low-weight confirmation; on liquid names it's often near a random walk.",
    },
}


def enrich_help(key: str, base: str, enabled: bool) -> str:
    """Append the 'how traders use it' line to a base tooltip when education is on."""
    if not enabled:
        return base
    extra = TRADER_USE.get(key)
    return f"{base}\n\n💡 {extra}" if extra else base


def glossary_groups() -> list[tuple[str, str]]:
    """The glossary as (term, definition) pairs, alphabetized by term."""
    return sorted(GLOSSARY, key=lambda item: item[0].lower())
