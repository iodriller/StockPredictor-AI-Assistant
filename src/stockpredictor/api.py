from __future__ import annotations

import os
from typing import Any

from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

from fastapi import HTTPException

from .backtesting import run_backtest
from .config import load_settings
from .journal import (
    append_journal_entry,
    delete_journal_entry,
    load_journal_entries,
    update_journal_entry,
)
from .news import build_news_feed
from .pipeline import analyze_symbol, scan_symbols
from .symbols import search_symbols
from .utils import to_serializable


class ScanRequest(BaseModel):
    symbols: list[str] | None = Field(default=None)
    config_path: str | None = Field(default=None)
    session_id: str = Field(default="default")
    max_symbols: int | None = Field(default=None)


class BacktestRequest(BaseModel):
    symbols: list[str] | None = Field(default=None)
    config_path: str | None = Field(default=None)


class AnalyzeRequest(BaseModel):
    config_path: str | None = Field(default=None)
    session_id: str = Field(default="default")


class JournalRequest(BaseModel):
    symbol: str
    action: str = Field(default="watch")
    setup_type: str = Field(default="unclassified")
    followed_plan: bool = Field(default=False)
    emotional_state: str = Field(default="neutral")
    entry_quality: int = Field(default=3, ge=1, le=5)
    exit_quality: int = Field(default=3, ge=1, le=5)
    risk_respected: bool = Field(default=False)
    outcome: str = Field(default="open")
    notes: str = Field(default="")
    decision_score: float | None = Field(default=None)
    confidence: float | None = Field(default=None)
    risk_reward: float | None = Field(default=None)
    config_path: str | None = Field(default=None)


class JournalUpdateRequest(BaseModel):
    symbol: str | None = Field(default=None)
    action: str | None = Field(default=None)
    setup_type: str | None = Field(default=None)
    followed_plan: bool | None = Field(default=None)
    emotional_state: str | None = Field(default=None)
    entry_quality: int | None = Field(default=None, ge=1, le=5)
    exit_quality: int | None = Field(default=None, ge=1, le=5)
    risk_respected: bool | None = Field(default=None)
    outcome: str | None = Field(default=None)
    notes: str | None = Field(default=None)
    decision_score: float | None = Field(default=None)
    confidence: float | None = Field(default=None)
    risk_reward: float | None = Field(default=None)
    config_path: str | None = Field(default=None)


def create_app(config_path: str | None = None) -> FastAPI:
    settings = load_settings(config_path or os.environ.get("STOCKPREDICTOR_CONFIG"))
    app = FastAPI(title=settings.app.get("name", "StockPredictor"), version="0.1.0")
    app.state.settings = settings
    app.state.latest_signals_by_session = {}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "config_path": str(app.state.settings.path)}

    @app.get("/config")
    def config() -> dict[str, Any]:
        return to_serializable(app.state.settings.raw)

    @app.get("/symbols/search")
    def symbol_search(q: str, limit: int = 25) -> list[dict[str, Any]]:
        return to_serializable(search_symbols(q, limit=limit))

    @app.get("/news")
    def news(symbols: str = "", limit: int = 25) -> dict[str, Any]:
        requested_symbols = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]
        if not requested_symbols:
            requested_symbols = app.state.settings.watchlist()[:5]
        return to_serializable(build_news_feed(requested_symbols, app.state.settings, limit=limit))

    @app.get("/scan")
    def scan_get(symbols: str = "", config_path: str | None = None, max_symbols: int | None = None) -> list[dict[str, Any]]:
        active_settings = _settings_for_request(app, config_path)
        requested_symbols = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()] or None
        return to_serializable(scan_symbols(active_settings, symbols=requested_symbols, max_symbols=max_symbols))

    @app.post("/scan")
    def scan(request: ScanRequest | None = Body(default=None)) -> list[dict[str, Any]]:
        active_settings = _settings_for_request(app, request.config_path if request else None)
        results = scan_symbols(
            active_settings,
            symbols=request.symbols if request else None,
            max_symbols=request.max_symbols if request else None,
        )
        _set_latest_signals(app, request.session_id if request else "default", results)
        return to_serializable(results)

    @app.get("/analyze/{symbol}")
    def analyze_get(symbol: str, config_path: str | None = None) -> dict[str, Any]:
        active_settings = _settings_for_request(app, config_path)
        return to_serializable(analyze_symbol(symbol, active_settings))

    @app.post("/analyze/{symbol}")
    def analyze(symbol: str, request: AnalyzeRequest | None = Body(default=None)) -> dict[str, Any]:
        active_settings = _settings_for_request(app, request.config_path if request else None)
        result = analyze_symbol(symbol, active_settings)
        _set_latest_signals(app, request.session_id if request else "default", [result])
        return to_serializable(result)

    @app.post("/backtest")
    def backtest(request: BacktestRequest | None = Body(default=None)) -> dict[str, Any]:
        active_settings = _settings_for_request(app, request.config_path if request else None)
        return to_serializable(run_backtest(active_settings, symbols=request.symbols if request else None))

    @app.get("/signals/latest")
    def latest_signals(session_id: str = "default") -> list[dict[str, Any]]:
        return to_serializable(app.state.latest_signals_by_session.get(session_id, []))

    @app.get("/journal")
    def journal(limit: int = 100) -> list[dict[str, Any]]:
        return to_serializable(load_journal_entries(app.state.settings, limit=limit))

    @app.post("/journal")
    def create_journal_entry(request: JournalRequest) -> dict[str, Any]:
        active_settings = _settings_for_request(app, request.config_path)
        payload = request.model_dump(exclude={"config_path"})
        return to_serializable(append_journal_entry(active_settings, payload))

    @app.patch("/journal/{entry_id}")
    def patch_journal_entry(entry_id: str, request: JournalUpdateRequest) -> dict[str, Any]:
        active_settings = _settings_for_request(app, request.config_path)
        updates = request.model_dump(exclude={"config_path"}, exclude_none=True)
        result = update_journal_entry(active_settings, entry_id, updates)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Journal entry {entry_id} not found")
        return to_serializable(result)

    @app.delete("/journal/{entry_id}")
    def remove_journal_entry(entry_id: str, config_path: str | None = None) -> dict[str, Any]:
        active_settings = _settings_for_request(app, config_path)
        if not delete_journal_entry(active_settings, entry_id):
            raise HTTPException(status_code=404, detail=f"Journal entry {entry_id} not found")
        return {"deleted": entry_id}

    return app


def _settings_for_request(app: FastAPI, config_path: str | None):
    if not config_path:
        return app.state.settings
    return load_settings(config_path)


def _set_latest_signals(app: FastAPI, session_id: str, results: list[Any]) -> None:
    app.state.latest_signals_by_session[str(session_id or "default")] = results
