# AGENTS.md

## Project

StockPredictor is configuration-first market-research and decision-support
software. It combines market data, indicators, models, contextual reasoning,
signal fusion, trade-plan generation, backtesting, a FastAPI service, and a
Streamlit dashboard. It does not place brokerage orders and is not financial
advice.

Use `README.md` for operator behavior, `configs/default.example.yaml` for the
committed configuration contract, and `traders.mind.md` for the trader reasoning
checklist. The notebook is legacy reference, not the application architecture.

## Commands

Windows development:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item configs/default.example.yaml configs/default.yaml
.\scripts\start-local.ps1
.\scripts\stop-local.ps1
.\.venv\Scripts\python -m stockpredictor.cli analyze AAPL --config configs/default.yaml
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
```

The API and dashboard can also be launched through
`python -m stockpredictor.cli api` and `dashboard`. Prefer module execution on
Windows so the Scripts directory does not need to be on `PATH`.

## Architecture and Data Rules

- `src/stockpredictor/` is the application package; keep CLI, API, dashboard, data
  providers, models, signals, risk, persistence, and context responsibilities
  separated.
- Treat `configs/default.example.yaml` as the public schema and
  `configs/default.yaml` as private local state.
- Preserve explicit provenance for live, cached, synthetic, heuristic, and LLM
  outputs. Never present a fallback as live market evidence.
- Keep GET routes read-only and caller/session state isolated.
- Backtest output is a logic sanity check unless the sample size and cadence
  support stronger conclusions. Do not describe it as expected performance.
- Do not add order execution, broker credentials, or investment claims as an
  incidental change.
- Do not commit local configuration, API keys, journals, snapshots, dashboard
  caches, logs, or generated reports.

## Change Style

- Inspect configuration, provider behavior, and existing tests before editing.
- Make the smallest coherent change and reuse existing pipeline patterns.
- Fix shared calculations or schemas at their source rather than patching one
  symbol, screen, headline, or prompt example.
- Keep market-data failures observable; diagnose readiness, ports, logs, and
  provider responses before changing behavior.

## Verification

- Documentation or guidance only: verify referenced paths and run
  `git diff --check`; application tests are not required.
- Python behavior: run the focused test and `python -m pytest`.
- Python quality: run `python -m ruff check .`.
- Dashboard/API integration: start only the necessary services and record
  provider/profile/fallback state. Live market, browser, and LLM checks must be
  reported separately from deterministic tests.

Never claim a check ran unless its output was observed.

## Git and Safety

- Preserve unrelated changes and keep commits focused.
- Use the configured repository-owner identity.
- Do not add assistant names, co-author trailers, session links, or tool
  attribution to Git artifacts.
- Never expose credentials, local configuration, journal content, or user data in
  logs, screenshots, commits, or pull requests.
