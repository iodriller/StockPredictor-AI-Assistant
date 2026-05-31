# News-In-Decision Plan (White-Box Catalyst Integration)

Research and decision-support planning only. No financial advice, no brokerage
execution.

## Problem

Today the project runs **two disconnected news paths**:

- **Decision path** — `context.build_context_summary` scores news with a crude
  keyword classifier (`_score_item`) and feeds `signals.fuse_signals` through the
  `context` and `sentiment` weights. News *does* affect the decision, but only via
  primitive keyword sentiment.
- **Display path** — `news.build_news_feed` produces the rich analysis (LLM grand
  summary, catalyst / risk / tradeability / no-trade flags, freshness, relevance,
  article excerpts). This is shown **only** in the News tab and is **never fed into
  the decision**.

Result: the Trade Plan shows an action but cannot show *whether* news was gathered,
*which* headlines counted, or *how much* they moved the score. The "summary" exists
but the decision is a black box with respect to it.

## Goal

1. **Bridge**: compute the rich per-symbol news analysis once and feed it into the
   decision (chosen: *bridge*, not full refactor — both paths stay, one analysis is
   shared).
2. **Soft influence** (chosen): news shapes the context/sentiment score as today,
   and LLM `no_trade_flags` apply a small confidence/score penalty and appear as
   reasons-to-skip. News never silently forces an action.
3. **White box**: the Trade Plan surfaces, next to the action, exactly what news was
   gathered and a numeric attribution of how each component (including news) moved
   the fused score.

Non-goals: changing scanner speed characteristics, full deduplication of the two
scorers, new providers, or any execution behavior.

## Design

### A. Compute the analysis once (`news.py`)
- Extract a single-symbol helper `analyze_symbol_news(symbol, settings) -> dict`
  from the existing `build_news_feed` internals (reuse `_enrich_item`,
  `_summarize_symbol_news`, freshness/relevance). Returns the enriched per-symbol
  summary **and** the list of headlines actually used.
- `build_news_feed` is refactored to call this helper per symbol (no behavior
  change to the News tab).

### B. Bridge into context (`context.py`)
- `build_context_summary(..., news_analysis: dict | None = None)`.
  - When `news_analysis` is provided (deep-dive path), derive `catalysts`, `risks`,
    `score`, `sentiment`, and freshness from the **enriched** item impacts plus the
    LLM `day_trader_focus`, instead of the crude keyword pass.
  - Attach the analysis and the exact evidence to the summary (see contracts).
  - When `None` (scanner path), keep today's lightweight behavior unchanged — this
    preserves scanner speed and avoids per-symbol LLM calls during scans.

### C. Soft decision integration (`signals.py`)
- Read `context.features["news_no_trade_flag_count"]`; if > 0, apply a single
  configurable soft penalty (`signal_fusion.thresholds.news_no_trade_penalty`,
  default ~0.15) to the score and append the flags as reasons. No hard block.
- Build a structured **score attribution** so the white box is exact:
  `score_breakdown = [{component, raw_score, weight, contribution}, ...]` for
  models / technicals / intraday / context / sentiment, plus applied penalties
  (disagreement, VIX, sector, news flags). All values already exist in
  `fuse_signals`; we just record them.

### D. Contracts (`contracts.py`)
- `ContextSummary`: add `news_analysis: dict` (grand summary, day_trader_focus,
  provider, dominant_category) and `evidence: list[dict]` (the headlines used:
  title, url, published, impact, sentiment, freshness, relevance).
- `SignalDecision`: add `score_breakdown: list[dict]`.

### E. Pipeline wiring (`pipeline.py`)
- In `analyze_symbol`, when context is enabled and config opts in
  (`context_agent.news_analysis.use_in_decision`, default true), call
  `analyze_symbol_news` once and pass it into `build_context_summary`.
- `scan_symbols` path passes nothing → stays light/fast.

### F. White-box UX (`ui/dashboard.py`)
Add a **"How news shaped this decision"** white-box block in the Trade Plan deep
dive (within Decision And Plan / Context panel):
- **Status badges**: `News considered: Yes (LLM｜heuristic)` · headlines-used count
  · freshness · dominant category. If nothing was gathered, an explicit
  "No news evidence was gathered for this decision."
- **Grand summary** text.
- **Catalyst / Risk / Tradeability / No-trade flags** — reuse the same 4-row table
  the News tab already renders, so both tabs look consistent.
- **Score attribution table**: component · raw · weight · contribution, with the
  context/sentiment (news) rows highlighted, plus a one-line readout like
  "News/context contributed +0.12 of the 0.34 fused score." This is the literal
  white-box answer to "was the summary taken into account?".
- **Expander "Headlines used in this decision"**: the exact `evidence` rows with
  links, using the existing headline column config.
- Keep all current "research only" framing and `HELP_TEXT` patterns.

### G. Config (`configs/default.example.yaml`)
- `context_agent.news_analysis.use_in_decision: true`
- `signal_fusion.thresholds.news_no_trade_penalty: 0.15`
Both conservative; setting `use_in_decision: false` restores today's behavior.

## Files To Change
- `src/stockpredictor/news.py` — extract `analyze_symbol_news`.
- `src/stockpredictor/context.py` — accept/consume `news_analysis`, attach evidence.
- `src/stockpredictor/signals.py` — score_breakdown + soft news penalty.
- `src/stockpredictor/contracts.py` — new fields.
- `src/stockpredictor/pipeline.py` — compute once, wire into deep dive only.
- `src/stockpredictor/ui/dashboard.py` — white-box decision-news panel.
- `configs/default.example.yaml` — new knobs.
- `tests/test_core.py`, `tests/test_news.py` — coverage (below).
- This doc + a status note in `docs/day-trader-dashboard-gap-plan.md`.

## Tests / Success Criteria
- `analyze_symbol_news` returns a summary + the used headlines for a symbol.
- With `news_analysis` provided, `ContextSummary.evidence` is non-empty and
  `score`/`sentiment` derive from enriched impacts.
- `SignalDecision.score_breakdown` contributions sum (within tolerance) to
  `decision.score` before penalties; penalties are itemized.
- LLM `no_trade_flags` produce a measurable soft score reduction and a
  reasons-to-skip entry — never a forced `no_trade`.
- Scanner path (`news_analysis=None`) behavior and speed unchanged.
- Deep-dive with news disabled (`use_in_decision: false`) matches prior output.
- `/analyze/{symbol}` JSON now includes `news_analysis`, `evidence`, and
  `score_breakdown` (serialization is automatic via `to_serializable`).

## Verification
- `pytest` (full suite) green.
- Launch via `scripts/start-local.ps1`; analyze a symbol with live headlines and
  confirm the white box shows the summary + attribution; confirm the scanner still
  returns promptly.

## Reporting Back (per CLAUDE.md)
On completion: what changed, what was verified (tests + manual dashboard run), what
was not verified (e.g. real LLM endpoint availability), remaining risk, and any
larger refactor (full unification of the two scorers) worth doing separately.
