# Priority 1 Increment Plan

This increment turns the current MVP from a working scaffold into a more
trustworthy trader research tool. Priority 1 means the smallest set of changes
that improves trust, configurability, trader usefulness, and verification
without adding broker execution, auth, paid data feeds, or a database.

## Priority 1 Gaps

- Runtime hygiene: use a project `.venv` and avoid relying on global packages.
- Configuration truth: make enabled features, models, context sources, risk, and
  backtest settings control behavior end to end.
- Trader context: turn `traders.mind.md` into a checklist-backed structured
  context output with catalysts, risks, alignment, and skip reasons.
- Scanner usefulness: rank symbols by trader-relevant movement, volume, gap,
  regime, confidence, risk/reward, catalyst/risk flags, and top reason.
- Risk planning: use support, resistance, VWAP, ATR, liquidity, and configured
  limits to produce entry zones, stops, targets, invalidation, and skip reasons.
- Backtest realism: simulate stops, targets, slippage, commissions, time exits,
  no-trades, and per-trade logs.
- Dashboard usefulness: show trader-facing tables, metrics, levels, context, and
  backtest evidence instead of making JSON the main interface.
- Verification discipline: tests must cover each P1 behavior and documented run
  commands must work from the project environment.

## Success Criteria

- `python -m pytest` passes from the project `.venv`.
- Disabling a feature or model in config prevents that feature/model from
  appearing in outputs.
- Context output includes structured features: catalyst score, freshness,
  market alignment, sector alignment, reasons to trade, and reasons to skip.
- Scanner returns a ranked table with movement, volume anomaly, gap, regime,
  action, confidence, risk/reward, catalyst/risk flags, and top reason.
- Risk output includes entry zone, stop, targets, risk/reward, liquidity status,
  position size, setup quality, and invalidation.
- Backtest output includes trade logs and exit reasons for stop hit, target hit,
  time exit, and no-trade paths.
- Dashboard can be manually opened and used without reading raw JSON as the
  primary workflow.

## Implementation Phases

1. Stabilize runtime: document `.venv` setup and keep run commands in module
   form for Windows reliability.
2. Enforce config: wire enabled feature/model/context/backtest/risk settings
   into runtime decisions.
3. Upgrade trader outputs: add structured context, scanner rows, and risk plans.
4. Upgrade backtests: simulate exits and expose trade-level evidence.
5. Improve dashboard: show tables, metrics, charts, and explanation panels.
6. Verify: add focused tests and run CLI/API/dashboard smoke checks.

## Acceptance Checks

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m stockpredictor.cli analyze AAPL --config configs/default.yaml
.\.venv\Scripts\python -m stockpredictor.cli scan --config configs/default.yaml
.\.venv\Scripts\python -m stockpredictor.cli backtest --config configs/default.yaml
```

Manual dashboard check:

```powershell
.\.venv\Scripts\python -m stockpredictor.cli dashboard --config configs/default.yaml
```

Open `http://127.0.0.1:8501` and verify scanner, ticker deep dive, risk plan,
context, and backtest panels render as trader-facing views.
