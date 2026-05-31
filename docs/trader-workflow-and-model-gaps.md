# Trader Workflow, Model Gaps, and Fix Plan

An evidence-based evaluation of why the tool returns `no_trade`/`watch` for
clearly bullish stocks, why the ML outputs look negative, what a real trader
needs, and a concrete plan. Research/decision-support only — not financial advice.

## TL;DR — what is actually wrong

Reproduced locally (synthetic but deterministic; see "Evidence" below):

1. **The fused score is mathematically capped below the "trade" line.** On a clean
   **+60% uptrend**, the score breakdown is: `models +0.027`, `technicals +0.243`,
   everything else 0 → **total 0.27**. `long` requires `0.35`. So even a perfect
   uptrend cannot reach `long` from price signals alone — it lands on `watch`. With
   neutral news (the common case) the system is *structurally* stuck below
   actionable. This is the #1 cause of "everything is no_trade/watch".
2. **The models barely move the needle.** A healthy +1–2% / 5-day forecast is
   divided by a flat `0.05` scale and then multiplied *again* by confidence, so a
   strong, agreeing forecast contributes ~`0.03–0.10`. Double-discounting crushes
   the model signal.
3. **The baseline model is dominated by stale history.** It fits one straight line
   over the whole lookback (180 days for swing). A stock that bottomed and turned
   sharply up in the last 30 days returns **expected_return = −0.233** — the model
   says "−23%" about a chart a trader reads as a clean bullish reversal. This is the
   "ML shows negative for a bullish stock" symptom.
4. **The technical score saturates.** It is a sum of boolean flags capped at ~0.81,
   so a mild uptrend and a screaming breakout look identical. No resolution.
5. **The Gaussian Process reverts to the mean.** It regresses price on the time
   index and then *extrapolates past the training range*, where a GP collapses
   toward its prior mean — i.e. it predicts a pullback exactly when price makes new
   highs.
6. **Models ignore the indicators the trader sees.** Baseline/GP both regress
   price on a time index; ARIMA(1,1,1) is ~a random walk on close. None use RSI,
   MACD, moving-average alignment, volume, or momentum — the things shown elsewhere
   in the UI. The user's instinct ("it can't just predict from the last value") is
   essentially correct.
7. **No trader-tunable parameters in the UI.** MA windows, model horizon/lookback,
   RSI bands, fusion weights, and score thresholds are all config-file-only. The UI
   exposes only the horizon control (and now headline count).

### Evidence (reproducible)
| Scenario (swing) | last px | baseline | gp | arima | action | score |
|---|---|---|---|---|---|---|
| recent bullish reversal (down 170d, up 30d) | 120 | **−0.233** | +0.035 | +0.041 | watch | +0.229 |
| strong uptrend (+60% / 200d, at highs) | 160 | +0.009 | +0.009 | +0.009 | watch | +0.27 |
| uptrend + small pullback | 143 | +0.078 | −0.012 | −0.013 | watch | +0.229 |

Score breakdown for the strong uptrend: `models 0.077×0.35=+0.027`,
`technicals 0.81×0.30=+0.243`, `context/sentiment/intraday = 0`. Ceiling ≈ 0.27.

## How a real trader should be able to use this tool (and where it breaks)

### A) Day trader
1. **Pre-market scan** for what is moving: gap %, relative volume, catalyst.
   → Scanner tab. *Gap/RVOL/float/halt need an intraday provider (known gap).*
2. **Understand the catalyst** — news/earnings/analyst. → News tab + the new
   "How News Shaped This Decision" panel. *Works.*
3. **Mark levels** — VWAP, opening range, prior high/low, support/resistance.
   → Chart. *Works for daily; premarket levels need an intraday provider.*
4. **Pick a bias** long/short. → Decision panel. **Breaks: bias is almost always
   `watch`/`no_trade` because of the score cap (#1).**
5. **Plan the risk** — entry trigger, stop, target, size, R/R, daily-loss cap.
   → Trade plan / risk layer. *Works, but only renders when bias is actionable.*
6. **Execute elsewhere, then journal.** → Journal tab. *Works.*

### B) Investor / swing
1. **Find candidates** (search/watchlist). *Works.*
2. **Assess the multi-week/month trend and a forecast.** → position horizon + ML.
   **Breaks: models are short-horizon trend extrapolators on price only; no
   fundamentals (valuation, earnings growth, balance sheet) at all.**
3. **Decide an entry zone, stop, and size.** *Works when actionable.*
4. **Monitor and re-evaluate over time.** → snapshots/journal. *Works.*

**The core gap for both:** step 4/2 (the actual decision) is where the math
prevents a clean read from ever becoming actionable, and the models don't reflect
what the trader sees on the chart.

## The models we have — and how to fix each

- **BaselineTrendModel** — single linear `polyfit` of close vs index over the full
  lookback. *Fix:* align the regression window to the horizon (e.g. swing ≈ 20–40
  bars, position ≈ 60–120) and/or weight recent bars (exponential) so a recent
  reversal isn't buried by months of old data. Expose the window in the UI.
- **GaussianProcessPriceModel** — GP of price vs time index, extrapolated beyond
  the training range (mean-reversion artifact). *Fix:* model **log-returns with
  drift** instead of raw price, or restrict to interpolation + an explicit drift
  term; otherwise it will keep fading new highs. Honestly label it a smoother, not
  a forecaster.
- **ArimaPriceModel** — ARIMA(1,1,1) on close ≈ random walk, tiny drift. *Fix:*
  allow a trend term / auto-order, or treat it as a low-weight confirmation only.
- **Cross-cutting fix (biggest leverage):** add a **feature-aware momentum/trend
  model** that scores multi-timeframe slope + MA alignment (e.g. 9/20/50) +
  MACD/RSI + RVOL — i.e. the same evidence the chart shows — so the model and the
  trader agree. This is what makes "bullish stock → bullish model" true.

## Fix plan (prioritized)

**P1 — Make a clean signal reachable (fixes "always no_trade/watch").**
- Rescale `model_component`: normalize `expected_return` by a volatility/horizon-aware
  expected move (ATR- or horizon-based), not a flat `0.05`; stop double-discounting
  by confidence (use confidence to weight, not to multiply down).
- De-saturate `technical_score`: make it a continuous trend-strength score, not a
  capped flag sum.
- Re-balance fusion so price-based evidence can reach `long` on its own; verify with
  the backtest that `no_trade_rate` drops to a sane band without becoming reckless.

**P2 — Fix the model correctness bugs.**
- Baseline: horizon-aligned, recency-weighted window.
- GP: model returns/drift, or downweight; remove mean-reversion-on-highs artifact.
- Add the feature-aware momentum model and enable it by default.

**P3 — Expose trader parameters in the UI.**
- A "Model & signal settings" panel: MA windows, model horizon/lookback, RSI bands,
  fusion weights, and score thresholds — live, per analysis, with sensible presets
  for day-trading vs investing.

**P4 — Honesty + investing support.**
- Label what each model is and its limits in the UI.
- (Later) optional fundamentals for the investor workflow (valuation/growth), since
  the current stack is purely technical.

## Implemented (P1 + P2, investing/swing calibration)

- **Model score rescaled** (`signals._model_component`): horizon-aware reference
  move (`model_reference_move_per_day_pct`/`_floor_pct`) + confidence floor
  (`model_confidence_floor`) instead of flat `/0.05 × confidence`. A clean uptrend's
  model vote went from ~0.08 to ~0.5–0.9.
- **Baseline** now uses a recency-weighted fit (`models.baseline.recency_decay`,
  default 5): a fresh reversal reads ~−0.04 instead of the old −0.23.
- **Gaussian Process** now models log-returns/drift, removing the
  mean-reversion-at-new-highs artifact.
- **New `momentum` model** (enabled first) scores MA alignment + mid-MA slope +
  recent realized momentum, so "bullish now" reads bullish (+0.94 on the reversal).
- **Horizon-aware VWAP-extension gate** (`risk.py`): swing/position no longer get
  blocked for sitting far from a long-window VWAP (`max_entry_distance_from_vwap_pct`
  moved into the horizon profiles; intraday stays tight at 0.08).

Verified end-to-end (synthetic fixtures): UPTREND score +0.45 (swing) / +0.58
(position), DOWN −0.43 / −0.56, model_scores swing ±0.9 — i.e. direction is now
correct and clean trends reach the trade band. Remaining blocks on these pure-line
fixtures are `poor_risk_reward` (degenerate synthetic support/resistance), a real
trade-quality gate, not the old broken score ceiling. Full `pytest` green (82).

### P3 (done) — trader parameters exposed in the UI
A "Model & signal tuning" panel on the Trade Plan exposes MA windows, the five
signal weights (applied to both base and the active horizon profile so they take
effect), the trade/watch score thresholds, and the trade confidence gate. Overrides
apply to that analysis run via an in-memory `Settings`, no config edit required.

### P4 (done) — model honesty labels
Model Details now shows a plain-language "what it is / limits" note per model and a
disclaimer that the models are price/technical only (no fundamentals).

### Live-data validation (done)
Ran the pipeline against real yfinance data. Findings:
- A bug my GP rewrite introduced (reading drift off the last index and compounding
  it → +55% forecasts on PLTR/MSFT) was fixed by averaging + clamping the per-step
  drift. GP now returns sane ±≤0.1 moves.
- Post-fix, the system **differentiates real names**: AAPL +0.34 (swing) / +0.41
  (position, reaches the trade band), MSFT ~+0.18, PLTR ~+0.10 (baseline bearish vs
  momentum bullish → mixed). Binding gates are now legitimate (confidence on noisy
  names, R/R geometry) — not a broken ceiling.

**Still open (separate work):** P4 fundamentals (valuation/growth) for the investor
workflow; and the risk-geometry gate (min R/R + ATR stop sizing) is the next lever
if more fills are wanted — but loosening it trades quality for quantity, so it needs
the user's risk appetite, not a silent default change. ARIMA contributes ~0 on most
liquid names (near random walk) and could be dropped or replaced later.

## Verification approach
Every P1/P2 change is checked against: (a) the three scenarios above must produce
`long`/`watch`/`no_trade` that match the chart, (b) the synthetic backtest
`no_trade_rate` and `win_rate` must stay sane, (c) full `pytest` green. No change is
reported as working without those numbers.
