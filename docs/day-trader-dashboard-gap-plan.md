# Day-Trader Dashboard Gap Plan

This plan is based on the current dashboard and a quick review of day-trader
workflow sources. It is research and product planning, not financial advice.

## Sources Reviewed

- FINRA day-trading overview and PDT requirements: https://www.finra.org/investors/investing/investment-products/stocks/day-trading
- Trade Ideas volatile-market signal checklist: https://www.trade-ideas.com/2026/05/18/day-trading-signals-for-volatile-market-conditions/
- Tradewink pre-trade checklist: https://tradewink.com/learn/day-trading-checklist-guide
- Finwiz strategy overview: https://finwiz.io/day-trading/day-trading-strategies
- Reddit risk checklist discussion: https://www.reddit.com/r/Daytrading/comments/1s5mlvp/risk_checklist_for_volatile_sessions/
- Reddit personalized checklist discussion: https://www.reddit.com/r/Daytrading/comments/1hhtssb/create_a_personalized_checklist_when_daytrading/

## What Matters Most To A Day Trader

- Find what is moving now: gap, price change, relative volume, liquidity, spread,
  opening range, and clear catalyst.
- Understand why it is moving: headline, earnings, analyst action, filing,
  macro event, sector sympathy, or unusual attention.
- Know if the setup is tradeable: VWAP alignment, trend, clean structure, nearby
  support/resistance, stop distance, and realistic target.
- Define risk before entry: entry trigger, stop, target, position size, max risk,
  risk/reward, daily loss cap, and no-trade conditions.
- Avoid forced trades: conflicting signals, repeated VWAP chop, weak volume,
  no catalyst, late extension, poor liquidity, or bad mental/session state.
- Review process quality: whether the trade matched the setup, respected risk,
  and had a good exit reason.

## Current Gaps

- Search and discovery was watchlist-bound. The dashboard needs symbol lookup,
  arbitrary ticker entry, and eventually true market-wide scans.
- Scanner lacks real intraday filters such as premarket gap, opening range high
  and low, relative volume by time of day, spread, float, halts, and high-of-day
  momentum.
- News is not yet a first-class workflow. A trader needs a news feed with
  symbols, timestamps, source, sentiment, catalyst category, and summary.
- Charts need day-trader overlays: candles, volume, VWAP, 9/20/50 EMA or SMA,
  prior day high/low, premarket high/low, opening range, support/resistance, and
  planned entry/stop/target.
- Risk needs session controls: max daily loss, max trades, stop after
  consecutive losses, position sizing by stop distance, and account/PDT warning.
- Context needs better classification: catalyst type, freshness, credibility,
  sector sympathy, macro relevance, and headline importance.
- Backtesting needs intraday data, strategy-specific rules, session filters,
  slippage/spread assumptions, and review tags.

## Priority Build Plan

1. **Dashboard search and ticker entry**
   - Add all-stock symbol search by ticker/company name.
   - Add arbitrary comma-separated tickers for scan/backtest.
   - Add API endpoint for symbol lookup.
   - Status: implemented with SEC-backed symbol search, local fallback, direct
     ticker entry, and one selected-symbol workflow in the sidebar.

2. **News feed tab**
   - Show recent headlines for selected symbols.
   - Include symbol, sentiment, impact score, time, provider, title, and link.
   - Summarize bullish/bearish/neutral headline counts.
   - Classify news as earnings, analyst, filing, macro, product/business,
     legal/regulatory, M&A, market sentiment, or other.
   - Add per-stock grand summaries, day-trader focus, source counts, and
     clickable source details.
   - Use configured LLM summarization when available; fall back to deterministic
     heuristic analysis when no API key is configured.
   - Status: implemented as a first pass using `context_agent.news_analysis`.

3. **Scanner polish**
   - Show price, percent change, gap, RVOL, ATR percent, trend, regime, action,
     confidence, risk/reward, catalyst flag, risk flag, and top reason.
   - Use formatted percentages, prices, and rank score.
   - Status: first polish pass implemented with formatted table, movement/RVOL
     metrics, catalyst/risk flags, selected-symbol workflow, and rank score.
   - Later: add true intraday scans: premarket gap >2%, RVOL >2, average volume
     >1M, spread filter, opening range break, VWAP reclaim/loss.

4. **Chart polish**
   - Replace line chart with candlestick chart plus volume.
   - Overlay moving averages and key levels.
   - Status: first polish pass implemented with candlesticks, volume, moving
     averages, VWAP, prior high/low, session open, support, and resistance.
   - Later: add premarket high/low, opening range, entry, stop, targets, and
     risk box.

5. **Risk and checklist panel**
   - Show the trade plan without raw JSON.
   - Include entry zone, stop, targets, position size, max risk, risk/reward,
     invalidation, setup quality, and no-trade reason.
   - Later: add max daily loss, max trades/day, consecutive-loss stop, PDT
     warning, and journal prompt.

6. **Review and journal**
   - Add trade-review fields: setup type, followed plan, emotional state,
     entry quality, exit quality, and screenshot/link.
   - Later: use journal results to score which setups are working.

## Acceptance Criteria For The Next Increment

- A user can search for a ticker not in the watchlist and analyze it.
- A user can enter custom ticker lists for scan and backtest.
- Scanner table is readable without interpreting raw decimals.
- Ticker deep dive shows candles, volume, levels, predictions, signal, risk,
  context, and technicals in trader-facing sections.
- News tab shows recent headlines for selected tickers.
- Dashboard still runs locally through the project `.venv`.
- API includes symbol search and news endpoints.

## News Feed Target Behavior

- Each selected symbol should show:
  - Grand summary.
  - Number of linked sources.
  - Bullish, bearish, and neutral headline counts.
  - Dominant catalyst category.
  - Catalyst, risk, tradeability, and no-trade flags.
  - Expandable source table with provider, timestamp, category, sentiment,
    impact, headline, and link.
- LLM behavior is configured under `context_agent.news_analysis.llm`.
- When `OPENAI_API_KEY` is present and LLM analysis is enabled, the system uses
  OpenAI Responses API classification/summarization.
- When the key is missing, the news feed still works using the local classifier.
