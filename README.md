StockPredictor Trading Intelligence
===================================

This repository started as a Gaussian-process stock-price notebook. It now
contains a configuration-first research MVP for stock prediction, signal fusion,
trade-plan generation, contextual trader reasoning, backtesting, an API service,
and a Streamlit dashboard.

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

- `GET /symbols/search?q=palantir`
- `GET /news?symbols=AAPL,PLTR`
- `GET /journal`
- `POST /journal`

The journal stores local review notes in `data/trade_journal.local.jsonl` by
default. That file is ignored by Git.

Configuration
-------------

The example runtime configuration is `configs/default.example.yaml`. Copy it to
`configs/default.yaml` for local use. The local file is ignored by Git so keys,
local endpoints, account settings, and watchlists do not get pushed.

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

If LocalDeploy is not running, the app falls back to the built-in heuristic news
classifier and summarizer.

Trader Agent
------------

`traders.mind.md` defines the trader-style reasoning checklist used by the
context pipeline: catalyst, volume, trend, levels, entry, stop, target,
invalidation, risk/reward, and no-trade reasons.

Legacy Notebook
---------------

`finance_app.ipynb` is preserved as the original Gaussian-process reference.
Runtime code now lives under `src/stockpredictor`.
