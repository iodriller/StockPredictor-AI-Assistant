StockPredictor Trading Intelligence
===================================

This repository is now a configuration-first research MVP for stock prediction,
signal fusion, trade-plan generation, contextual trader reasoning, backtesting,
an API service, and a Streamlit dashboard. The original Gaussian-process
notebook is preserved only as a legacy reference.

This is research and decision-support software. It is not financial advice and
does not place brokerage orders.

Quick Start
-----------

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item configs/default.example.yaml configs/default.yaml
```

Run the API:

```powershell
.\.venv\Scripts\python -m stockpredictor.cli api --config configs/default.yaml
```

Run the dashboard:

```powershell
.\.venv\Scripts\python -m stockpredictor.cli dashboard --config configs/default.yaml
```

Run the full local stack:

```powershell
.\scripts\start-local.ps1
```

This starts LocalDeploy on `8100` when the sibling `..\LocalDeploy` project is
available, the FastAPI service on `8000`, and the dashboard on `8501`, then opens
the dashboard. Logs and pid files are written to `.logs/`.

By default the launcher refreshes StockPredictor API/dashboard processes so the
dashboard cannot keep serving stale imports after code changes. Use
`-ReuseExisting` only when you explicitly want to keep already-running
StockPredictor processes.

Stop the StockPredictor services:

```powershell
.\scripts\stop-local.ps1
```

Add `-IncludeLocalDeploy` if you also want to stop the LocalDeploy process
started by the launcher.

Analyze one symbol:

```powershell
.\.venv\Scripts\python -m stockpredictor.cli analyze AAPL --config configs/default.yaml
```

Scan the configured watchlist:

```powershell
.\.venv\Scripts\python -m stockpredictor.cli scan --config configs/default.yaml
```

Run tests:

```powershell
.\.venv\Scripts\python -m pytest
```

The module form is preferred on Windows because it does not depend on the user
Python Scripts directory being on `PATH`.

API Additions
-------------

The local API includes:

- `GET /health`
- `GET /config`
- `GET /symbols/search?q=palantir`
- `GET /news?symbols=AAPL,PLTR`
- `GET /scan?symbols=AAPL,PLTR`
- `POST /scan`
- `GET /analyze/{symbol}`
- `POST /analyze/{symbol}`
- `POST /backtest`
- `GET /signals/latest?session_id=default`
- `GET /journal`
- `POST /journal`
- `PATCH /journal/{entry_id}`
- `DELETE /journal/{entry_id}`
- `GET /snapshots/{symbol}` — recent persisted analysis snapshots for delta comparison.

`/analyze/{symbol}` and `/scan` accept a `horizon` parameter (`intraday`, `swing`,
or `position`) that flexes ATR multiples, model lookback, target R, and signal
weights. Each analyze call also returns:

- `session` — today's intraday context (live price, session VWAP, premarket high/low, opening range, time-of-day RVOL).
- `market_state` — SPY / QQQ / IWM / VIX state.
- `sector_context` — sector ETF for the symbol and its alignment with the symbol's move.
- `calendar` — current market session, next earnings, configured macro events, and resulting no-trade flags.
- `snapshot_record` and `previous_snapshots` — current analysis persisted to JSONL with the prior N records loaded so the dashboard can show "what changed since I last looked."

The GET scan/analyze routes are read-only. The POST routes are kept for
compatibility and store latest results under a caller-supplied `session_id`
rather than one process-global latest result.

The journal stores local review notes in `data/trade_journal.local.jsonl` by
default. That file is ignored by Git.

Configuration
-------------

The example runtime configuration is `configs/default.example.yaml`. Copy it to
`configs/default.yaml` for local use, then edit the local file for keys, local
endpoints, account settings, and watchlists. `configs/default.yaml` is ignored
by Git and is intentionally not a committed source of truth.

The main runtime configuration is `configs/default.yaml`. It controls data
providers, watchlists, indicators, enabled models, signal-fusion weights, risk
limits, context-agent sources, backtest settings, and dashboard defaults.

The default data provider is `yfinance`. If market data fails and
`allow_synthetic_fallback` is enabled, the app uses deterministic synthetic data
so the local UI and tests can still run.

Day-trader overlays such as premarket high/low, spreads, halt status, float, and
true time-of-day relative volume require an intraday/scanner data provider. The
default free provider exposes enough data for local research, but not every
professional scanner field.

### Backtest Cadence

With the default `backtest.evaluation_step_days: 5` and `holding_period_days: 5`
on a 6-month daily dataset, each symbol generates roughly 10–25 evaluation
points. That is not enough to make `sharpe_like` or `win_rate` statistically
meaningful; treat the backtest report as a sanity check on logic, not as a
performance estimate. Lower `evaluation_step_days` and raise `period` (e.g.
`2y`) before drawing conclusions.

The simulator now enforces `risk.max_trades_per_day`,
`risk.stop_after_consecutive_losses`, and `risk.max_daily_loss_pct` and reports
the position size used (`exposure_basis: planned` or `fraction`) per trade in
the trade log.

### Notes On Risk And Scanner Defaults

- `risk.pdt_warning_enabled` only fires when `risk.account_size < risk.pdt_min_equity`.
  With the default `account_size: 100000`, the warning is silent unless you lower
  `account_size` to reflect a smaller real account.
- Scanner filter sliders show units in the same scale as the displayed columns:
  - `Min abs change` and `Max ATR` are in **percent** (`5.0` means 5%).
  - `Min RVOL` is the **ratio** of current volume to average (e.g. `1.5`).
- `data.cache_ttl_seconds` controls the in-memory market-data cache. Set to `0`
  to disable caching (every analyze/scan refetches).
- Backtest `use_planned_position_size` controls whether the simulator sizes by
  `RiskPlan.position_size` (true) or by a flat `risk.max_position_fraction` of
  equity (false). The planned size is conservative; the fraction mode is for
  smoke tests.

Local News LLM
--------------

The news feed can use a local LLM through the sibling `LocalDeploy` project.
The default config points to:

```text
http://127.0.0.1:8100/v1/chat/completions
```

with model/profile:

```text
qwen3vl_8b_ollama
```

The local config can require LLM summaries by setting:

```yaml
context_agent:
  news_analysis:
    llm:
      fallback_to_heuristic: false
```

When fallback is disabled and LocalDeploy is unavailable, the News tab reports
the LLM error instead of silently producing a heuristic summary. When fallback
is enabled, the UI labels the result as `heuristic_fallback`.

Optional article excerpt fetching is configured under
`context_agent.news_analysis.article_scraping`. The app fetches short article
excerpts for summarization; it does not store full article bodies.

Trader Agent
------------

`traders.mind.md` defines the trader-style reasoning checklist used by the
context pipeline: catalyst, volume, trend, levels, entry, stop, target,
invalidation, risk/reward, and no-trade reasons.

Legacy Notebook
---------------

`finance_app.ipynb` is preserved as the original Gaussian-process reference.
Runtime code now lives under `src/stockpredictor`.
