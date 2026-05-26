from __future__ import annotations

import os
from typing import Any

from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

from .backtesting import run_backtest
from .config import load_settings
from .pipeline import analyze_symbol, scan_symbols
from .utils import to_serializable


class ScanRequest(BaseModel):
    symbols: list[str] | None = Field(default=None)
    config_path: str | None = Field(default=None)


class BacktestRequest(BaseModel):
    symbols: list[str] | None = Field(default=None)
    config_path: str | None = Field(default=None)


class AnalyzeRequest(BaseModel):
    config_path: str | None = Field(default=None)


def create_app(config_path: str | None = None) -> FastAPI:
    settings = load_settings(config_path or os.environ.get("STOCKPREDICTOR_CONFIG"))
    app = FastAPI(title=settings.app.get("name", "StockPredictor"), version="0.1.0")
    app.state.settings = settings
    app.state.latest_signals = []

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "config_path": str(app.state.settings.path)}

    @app.get("/config")
    def config() -> dict[str, Any]:
        return to_serializable(app.state.settings.raw)

    @app.post("/scan")
    def scan(request: ScanRequest | None = Body(default=None)) -> list[dict[str, Any]]:
        active_settings = _settings_for_request(app, request.config_path if request else None)
        results = scan_symbols(active_settings, symbols=request.symbols if request else None)
        app.state.latest_signals = results
        return to_serializable(results)

    @app.post("/analyze/{symbol}")
    def analyze(symbol: str, request: AnalyzeRequest | None = Body(default=None)) -> dict[str, Any]:
        active_settings = _settings_for_request(app, request.config_path if request else None)
        result = analyze_symbol(symbol, active_settings)
        app.state.latest_signals = [result]
        return to_serializable(result)

    @app.post("/backtest")
    def backtest(request: BacktestRequest | None = Body(default=None)) -> dict[str, Any]:
        active_settings = _settings_for_request(app, request.config_path if request else None)
        return to_serializable(run_backtest(active_settings, symbols=request.symbols if request else None))

    @app.get("/signals/latest")
    def latest_signals() -> list[dict[str, Any]]:
        return to_serializable(app.state.latest_signals)

    return app


def _settings_for_request(app: FastAPI, config_path: str | None):
    if not config_path:
        return app.state.settings
    return load_settings(config_path)


app = create_app()

