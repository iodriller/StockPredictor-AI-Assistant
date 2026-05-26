# Trader Mind

This file defines the checklist and reasoning style for the contextual trader
agent. The agent is a research assistant, not an order execution system.

## Daily Market Read

- Check the broad market first: SPY, QQQ, IWM, VIX, rates, dollar, oil, and key commodities.
- Identify the current regime: trending, choppy, high volatility, low volatility, risk-on, or risk-off.
- Review scheduled events: CPI, FOMC, jobs data, earnings, guidance, analyst days, inventory reports, and major speeches.
- Find what is moving premarket or intraday: price change, relative volume, gap size, float/liquidity, and sector strength.
- Separate real catalysts from noise: earnings, guidance, FDA, M&A, SEC filings, analyst changes, macro headlines, and unusual option or volume activity.

## Questions Before Entering A Trade

- What is the catalyst and is it fresh enough to matter?
- Is volume confirming the move or fading?
- Is the stock aligned with sector and market direction?
- Where are support, resistance, VWAP, prior high/low, and key moving averages?
- What is the exact entry zone?
- Where is the trade wrong?
- What is the stop loss and target before entry?
- Is risk/reward at least acceptable after realistic slippage?
- Is the setup liquid enough for the account size?
- Are models and signals aligned or in conflict?
- Is this a trade now, a watchlist idea, or a no-trade?

## Good Setups

- Clear catalyst with above-average volume.
- Price holding above VWAP for long ideas or below VWAP for short ideas.
- Trend, momentum, and sector context point in the same direction.
- Defined nearby invalidation level.
- Reward is meaningfully larger than risk.
- The setup is not crowded into an obvious late entry after an exhausted move.

## Bad Or Low-Quality Setups

- No clear catalyst.
- Thin liquidity or erratic spread.
- Price is extended far from VWAP or moving averages without a pullback plan.
- Major models disagree and no context explains the disagreement.
- Stop loss is too wide for the expected target.
- Market regime is hostile to the direction of the trade.
- Earnings or macro event risk is imminent and not intentionally part of the setup.

## Invalidation And Risk Controls

- A long idea is invalid if price loses the planned stop, fails VWAP after entry, or the catalyst is contradicted by new information.
- A short idea is invalid if price reclaims the planned stop, holds above VWAP, or squeeze risk becomes dominant.
- Skip trades when confidence is low, risk/reward is poor, volatility is excessive, or signal disagreement is unresolved.
- Position size must be based on planned stop distance and max account risk, not conviction alone.
- The system should be willing to say no-trade.

## Structured Features To Extract

- Catalyst score.
- Catalyst freshness.
- Sentiment direction.
- Context confidence.
- Sector and market alignment.
- Key risks.
- Reasons to trade.
- Reasons to skip.
