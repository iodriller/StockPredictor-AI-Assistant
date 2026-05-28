# Educative UI Plan

## Goal

Add subtle but definite decision support to the dashboard without turning it
into a tutorial page. The app should help a trader understand what each field
means, why it matters, and how to interpret it in context.

## Principles

- Prefer hover help on labels, metrics, widgets, and columns.
- Keep visible guidance short and collapsible.
- Explain terms in trader language: what it means, why it matters, and what a
  weak/strong reading usually implies.
- Avoid pretending one metric is a buy/sell rule. Tooltips should emphasize
  confirmation, risk, and no-trade conditions.
- Make heuristic/LLM/news-source limits explicit.

## Priority 1 Tooltips

- Decision fields: action/bias, confidence, score, setup quality, top reason,
  no-trade reasons.
- Risk fields: entry zone, stop, target, risk/reward, position size,
  invalidation, max risk, planned risk.
- Scanner fields: rank, change, RVOL, gap, ATR %, relative strength, VWAP
  distance, catalyst/risk flags.
- Context/news fields: catalyst score, freshness, sentiment, source count,
  impact, relevance, analysis provider, article excerpt count.
- Backtest fields: win rate, average return, drawdown, Sharpe-like, no-trade
  rate, R multiple, MAE, MFE, time exit, stop/target exit.

## Priority 2 Guidance

- Add a compact decision checklist expander on the Trade Plan tab.
- Add scanner interpretation help for filtering movers versus noisy names.
- Add news coverage help explaining LLM, heuristic fallback, article excerpts,
  and provider limitations.
- Add a backtest interpretation note that warns against trusting small samples.

## Acceptance Checks

- Core metrics show hover help icons.
- Scanner and news tables have column-level explanations.
- The Trade Plan tab explains the decision path without requiring raw JSON.
- Heuristic fallback and article excerpt limits are visible.
- Tests and dashboard compile pass.
