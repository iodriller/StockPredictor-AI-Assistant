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

Configuration
-------------

The main runtime configuration is `configs/default.yaml`. It controls data
providers, watchlists, indicators, enabled models, signal-fusion weights, risk
limits, context-agent sources, backtest settings, and dashboard defaults.

The default data provider is `yfinance`. If market data fails and
`allow_synthetic_fallback` is enabled, the app uses deterministic synthetic data
so the local UI and tests can still run.

Trader Agent
------------

`traders.mind.md` defines the trader-style reasoning checklist used by the
context pipeline: catalyst, volume, trend, levels, entry, stop, target,
invalidation, risk/reward, and no-trade reasons.

Legacy Notebook
---------------

`finance_app.ipynb` is preserved as the original Gaussian-process reference.
Runtime code now lives under `src/stockpredictor`.
