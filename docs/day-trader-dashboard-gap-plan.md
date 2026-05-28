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
- Scanner now has trader-facing ranking and filters. True market-wide and
  tick-level intraday filters still need a dedicated intraday data provider for
  premarket, spread, float, halt, and high-of-day momentum coverage.
- News is now a first-class workflow with per-symbol summaries, source links,
  sentiment, impact, category, local LLM support, and optional article excerpts
  passed into the summarizer.
- Charts now show candles, volume, VWAP/moving averages, session/prior levels,
  opening-range availability, and risk-plan overlays for entry, stop, targets,
  and entry zone.
- Risk now includes session guardrails: max daily loss, max trades, consecutive
  loss stop, PDT warning, stop-distance sizing, planned risk, and no-trade
  reasons.
- Context now classifies catalyst categories, impact, sentiment, source count,
  headline relevance, and day-trader focus. Credibility scoring is still basic
  because free providers do not expose source reliability metadata.
- Backtesting now records stop/target/time exits, slippage/commission, R
  multiple, MAE/MFE, setup quality, top reason, and no-trade logs. True intraday
  strategy testing still needs intraday bars.

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
  - Use configured LLM summarization when available.
  - Make any heuristic fallback explicit in the UI and support disabling fallback
    entirely when LLM summaries are required.
  - Status: implemented using `context_agent.news_analysis`; local config now
    requires LocalDeploy for summaries, and article excerpt scraping is enabled.

3. **Scanner polish**
   - Show price, percent change, gap, RVOL, ATR percent, trend, regime, action,
     confidence, risk/reward, catalyst flag, risk flag, and top reason.
   - Use formatted percentages, prices, and rank score.
   - Status: first polish pass implemented with formatted table, movement/RVOL
     metrics, catalyst/risk flags, selected-symbol workflow, and rank score.
   - Status update: added scanner filters, VWAP distance, support/resistance
     distance, prior/session/opening-range fields, liquidity flag, high-RVOL
     flag, meaningful-gap flag, and skip reasons.
   - Provider-limited later item: add true market-wide intraday scans with
     premarket gap >2%, time-of-day RVOL, spread filter, float, halts, opening
     range break, VWAP reclaim/loss, and high-of-day momentum.

4. **Chart polish**
   - Replace line chart with candlestick chart plus volume.
   - Overlay moving averages and key levels.
   - Status: first polish pass implemented with candlesticks, volume, moving
     averages, VWAP, prior high/low, session open, support, and resistance.
   - Status update: added opening-range overlay when intraday bars are available
     and risk overlays for entry zone, entry, stop, and targets.
   - Provider-limited later item: add premarket high/low once the selected data
     provider supplies premarket bars.

5. **Risk and checklist panel**
   - Show the trade plan without raw JSON.
   - Include entry zone, stop, targets, position size, max risk, risk/reward,
     invalidation, setup quality, and no-trade reason.
   - Status: implemented with planned risk, position value, risk/share, session
     guardrails, PDT warning, and no-trade reasons.

6. **Review and journal**
   - Add trade-review fields: setup type, followed plan, emotional state,
     entry quality, exit quality, and screenshot/link.
   - Status: implemented as a local JSONL journal with dashboard and API
     create/list endpoints.
   - Later: add screenshot/link upload and use journal results to score which
     setups are working.

## Acceptance Criteria For The Next Increment

- A user can search for a ticker not in the watchlist and analyze it.
- A user can enter custom ticker lists for scan and backtest.
- Scanner table is readable without interpreting raw decimals.
- Ticker deep dive shows a decision-first trade plan, candles, volume, levels,
  context/catalyst, and advanced model/signal/indicator details in collapsed
  sections.
- News tab shows recent headlines for selected tickers.
- Dashboard still runs locally through the project `.venv`.
- API includes symbol search and news endpoints.
- API includes local journal endpoints.
- Backtest trade log includes setup quality, R multiple, MAE/MFE, and skip
  reasons.

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
- LocalDeploy is the preferred local summarizer at
  `http://127.0.0.1:8100/v1/chat/completions`.
- Heuristic fallback is controlled by `fallback_to_heuristic`; when disabled,
  the News tab stops and reports the LLM error instead of producing a fallback
  summary.
- Article excerpt fetching is controlled by
  `context_agent.news_analysis.article_scraping`.
