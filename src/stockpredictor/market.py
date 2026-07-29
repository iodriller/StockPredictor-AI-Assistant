"""Broad-market and sector context.

Fetches SPY/QQQ/IWM/VIX and the symbol's sector ETF so a single-symbol
decision can be cross-checked against where the market is right now.
Degrades to neutral state if data is unavailable (synthetic, off-hours,
provider hiccup).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from .config import Settings
from .contracts import MarketState, SectorContext
from .data import MarketDataProvider, fetch_market_data


LOGGER = logging.getLogger(__name__)

# Static sector-to-ETF map. Extend via `market_context.sector_etf_overrides` in YAML.
_DEFAULT_SECTOR_ETFS = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
}

# Hand-tuned overrides for common day-trading tickers where the "right" ETF
# isn't the official sector ETF (e.g. NVDA traders watch SMH/SOXX, not XLK).
_SYMBOL_OVERRIDES = {
    "NVDA": "SOXX",
    "AMD": "SOXX",
    "SMCI": "SOXX",
    "MU": "SOXX",
    "ASML": "SOXX",
    "TSLA": "XLY",
    "RIVN": "XLY",
    "MARA": "BITQ",
    "RIOT": "BITQ",
    "COIN": "BITQ",
    "SOFI": "XLF",
    "PLTR": "XLK",
}


def build_market_state(settings: Settings, provider: MarketDataProvider | None = None) -> MarketState:
    cfg = settings.raw.get("market_context", {}) or {}
    if not cfg.get("enabled", True):
        return MarketState(as_of=datetime.now(timezone.utc).isoformat(), notes=["market context disabled by config"])
    state = MarketState(as_of=datetime.now(timezone.utc).isoformat())
    symbols = cfg.get("market_symbols", {"spy": "SPY", "qqq": "QQQ", "iwm": "IWM", "vix": "^VIX"})

    spy_change = _safe_change_pct(settings, symbols.get("spy", "SPY"), provider)
    qqq_change = _safe_change_pct(settings, symbols.get("qqq", "QQQ"), provider)
    iwm_change = _safe_change_pct(settings, symbols.get("iwm", "IWM"), provider)
    vix_value, vix_change = _safe_value_and_change(settings, symbols.get("vix", "^VIX"), provider)

    state.spy_change_pct = spy_change
    state.qqq_change_pct = qqq_change
    state.iwm_change_pct = iwm_change
    state.vix_value = vix_value
    state.vix_change_pct = vix_change
    state.market_trend = _market_trend(spy_change, qqq_change)
    state.risk_environment = _risk_environment(vix_value, vix_change)
    return state


def build_sector_context(
    symbol: str,
    settings: Settings,
    provider: MarketDataProvider | None = None,
    include_live_lookup: bool = True,
) -> SectorContext:
    cfg = settings.raw.get("market_context", {}) or {}
    if not cfg.get("sector_enabled", True):
        return SectorContext(symbol=symbol.upper(), notes=["sector context disabled by config"])
    overrides = cfg.get("sector_etf_overrides", {}) or {}
    sector_etfs = {**_DEFAULT_SECTOR_ETFS, **(cfg.get("sector_etfs", {}) or {})}
    symbol_overrides = {**_SYMBOL_OVERRIDES, **{str(k).upper(): str(v).upper() for k, v in overrides.items()}}

    sector_name = "unknown"
    etf = symbol_overrides.get(symbol.upper(), "")
    if not etf and include_live_lookup:
        sector_name = _yfinance_sector(symbol) or "unknown"
        etf = sector_etfs.get(sector_name, "")

    if not etf:
        return SectorContext(symbol=symbol.upper(), sector_name=sector_name, notes=["no sector ETF mapping for this symbol"])

    change = _safe_change_pct(settings, etf, provider)
    trend = _trend_label(change)
    alignment = "unknown"
    symbol_change = _safe_change_pct(settings, symbol, provider)
    if change is not None and symbol_change is not None:
        if (symbol_change > 0 and change > 0) or (symbol_change < 0 and change < 0):
            alignment = "aligned"
        elif abs(change) < 0.001:
            alignment = "neutral"
        else:
            alignment = "divergent"
    return SectorContext(
        symbol=symbol.upper(),
        sector_name=sector_name,
        sector_etf=etf,
        sector_change_pct=change,
        sector_trend=trend,
        alignment=alignment,
    )


def market_no_trade_flags(state: MarketState | None, sector: SectorContext | None) -> Iterable[str]:
    if state is None:
        return
    if state.risk_environment == "elevated":
        yield "VIX is elevated — broad-market risk is higher than normal"
    if sector and sector.alignment == "divergent":
        yield f"symbol is moving against its sector ({sector.sector_etf})"


def _safe_change_pct(settings: Settings, symbol: str, provider: MarketDataProvider | None) -> float | None:
    if not symbol:
        return None
    try:
        frame = fetch_market_data(symbol, settings, provider)
    except Exception as exc:
        LOGGER.info("Market context fetch for %s failed: %s", symbol, exc)
        return None
    if frame is None or len(frame) < 2:
        return None
    last = float(frame["Close"].iloc[-1])
    prev = float(frame["Close"].iloc[-2])
    if prev == 0:
        return None
    return (last / prev) - 1


def _safe_value_and_change(settings: Settings, symbol: str, provider: MarketDataProvider | None) -> tuple[float | None, float | None]:
    if not symbol:
        return None, None
    try:
        frame = fetch_market_data(symbol, settings, provider)
    except Exception as exc:
        LOGGER.info("VIX fetch for %s failed: %s", symbol, exc)
        return None, None
    if frame is None or len(frame) < 2:
        return None, None
    last = float(frame["Close"].iloc[-1])
    prev = float(frame["Close"].iloc[-2])
    change = ((last / prev) - 1) if prev else None
    return last, change


def _yfinance_sector(symbol: str) -> str | None:
    try:
        import yfinance as yf

        info = getattr(yf.Ticker(symbol), "info", {}) or {}
        sector = info.get("sector")
        return str(sector) if sector else None
    except Exception:
        return None


def _market_trend(spy_change: float | None, qqq_change: float | None) -> str:
    if spy_change is None and qqq_change is None:
        return "unknown"
    avg = sum(value for value in [spy_change, qqq_change] if value is not None) / max(
        1, sum(1 for value in [spy_change, qqq_change] if value is not None)
    )
    if avg > 0.003:
        return "risk_on"
    if avg < -0.003:
        return "risk_off"
    return "mixed"


def _risk_environment(vix_value: float | None, vix_change: float | None) -> str:
    if vix_value is None:
        return "unknown"
    if vix_value >= 25:
        return "elevated"
    if vix_value >= 18:
        return "normal_high"
    if vix_value >= 13:
        return "normal"
    return "complacent"


def _trend_label(change_pct: float | None) -> str:
    if change_pct is None:
        return "unknown"
    if change_pct > 0.003:
        return "up"
    if change_pct < -0.003:
        return "down"
    return "flat"


def safe_pct_series(series: pd.Series) -> pd.Series:
    return series.pct_change().fillna(0.0)
