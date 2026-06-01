from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Protocol

import numpy as np
import pandas as pd

from .config import Settings
from .contracts import MarketSnapshot
from .utils import TTLCache, to_float


LOGGER = logging.getLogger(__name__)
_MARKET_DATA_CACHE = TTLCache(ttl_seconds=300)
_INTRADAY_CACHE = TTLCache(ttl_seconds=60)


class MarketDataProvider(Protocol):
    name: str

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        ...


@dataclass
class YFinanceProvider:
    name: str = "yfinance"

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        import yfinance as yf

        frame = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return normalize_ohlcv(frame, symbol)

    def fetch_intraday(self, symbol: str, period: str, interval: str, prepost: bool = True) -> pd.DataFrame:
        import yfinance as yf

        frame = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            prepost=prepost,
            progress=False,
            threads=False,
        )
        return normalize_ohlcv(frame, symbol)


@dataclass
class SyntheticProvider:
    seed: int = 7
    name: str = "synthetic"

    def fetch_intraday(self, symbol: str, period: str, interval: str, prepost: bool = True) -> pd.DataFrame:
        # Produces a deterministic intraday session so tests and offline use still
        # exercise the session/intraday pipeline. We synthesize one trading day.
        stable_symbol_seed = sum((idx + 1) * ord(char) for idx, char in enumerate(symbol.upper()))
        rng = np.random.default_rng(self.seed + stable_symbol_seed + 991)
        interval_minutes = max(1, _interval_to_minutes(interval.lower()) or 1)
        bars_per_session = max(60, 390 // interval_minutes)
        market_day = pd.Timestamp.now(tz="America/New_York").normalize()
        start = market_day.replace(hour=9, minute=30).tz_convert("UTC")
        index = pd.date_range(start=start, periods=bars_per_session, freq=f"{interval_minutes}min")
        base_price = 100 + (stable_symbol_seed % 250)
        increments = rng.normal(0.0002, 0.0015, bars_per_session)
        closes = base_price * np.exp(np.cumsum(increments))
        spread = np.maximum(closes * 0.0008, 0.05)
        opens = closes * (1 + rng.normal(0.0, 0.0005, bars_per_session))
        highs = np.maximum(opens, closes) + spread
        lows = np.minimum(opens, closes) - spread
        volume = rng.integers(15_000, 250_000, bars_per_session).astype(float)
        return pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume},
            index=index,
        )

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        rows = _period_to_rows(period, interval)
        stable_symbol_seed = sum((idx + 1) * ord(char) for idx, char in enumerate(symbol.upper()))
        rng = np.random.default_rng(self.seed + stable_symbol_seed)
        interval_minutes = _interval_to_minutes(interval.lower())
        freq = f"{interval_minutes}min" if interval_minutes is not None else "B"
        end = pd.Timestamp.now(tz="UTC").floor(freq) if interval_minutes is not None else pd.Timestamp.now(tz="UTC").normalize()
        index = pd.date_range(end=end, periods=rows, freq=freq)
        drift = 0.0008 + (stable_symbol_seed % 9) / 10000
        shocks = rng.normal(drift, 0.018, rows)
        close = 100 * np.exp(np.cumsum(shocks))
        spread = np.maximum(close * rng.normal(0.012, 0.003, rows), close * 0.004)
        open_ = close * (1 + rng.normal(0.0, 0.006, rows))
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = rng.integers(750_000, 6_000_000, rows)
        return pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume.astype(float),
            },
            index=index,
        )


@dataclass
class FallbackProvider:
    primary: MarketDataProvider
    fallback: MarketDataProvider
    min_rows: int
    name: str = "fallback"

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        try:
            frame = self.primary.fetch(symbol, period, interval)
            if len(frame) >= self.min_rows:
                frame.attrs["provider"] = self.primary.name
                return frame
            LOGGER.warning(
                "Primary market data provider %s returned %s rows for %s; minimum is %s. Falling back to %s.",
                self.primary.name,
                len(frame),
                symbol.upper(),
                self.min_rows,
                self.fallback.name,
            )
        except Exception as exc:
            LOGGER.warning(
                "Primary market data provider %s failed for %s. Falling back to %s: %s",
                self.primary.name,
                symbol.upper(),
                self.fallback.name,
                exc,
            )
        frame = self.fallback.fetch(symbol, period, interval)
        frame.attrs["provider"] = self.fallback.name
        return frame

    def fetch_intraday(self, symbol: str, period: str, interval: str, prepost: bool = True) -> pd.DataFrame:
        primary_fetcher = getattr(self.primary, "fetch_intraday", None)
        if primary_fetcher is not None:
            try:
                frame = primary_fetcher(symbol, period=period, interval=interval, prepost=prepost)
                if frame is not None and not frame.empty:
                    frame.attrs["provider"] = self.primary.name
                    return frame
            except Exception as exc:
                LOGGER.info(
                    "Primary intraday provider %s failed for %s. Falling back to %s: %s",
                    self.primary.name,
                    symbol.upper(),
                    self.fallback.name,
                    exc,
                )
        fallback_fetcher = getattr(self.fallback, "fetch_intraday", None)
        if fallback_fetcher is None:
            raise RuntimeError(f"Fallback provider {self.fallback.name} does not support intraday data")
        frame = fallback_fetcher(symbol, period=period, interval=interval, prepost=prepost)
        frame.attrs["provider"] = self.fallback.name
        return frame


def get_market_data_provider(settings: Settings) -> MarketDataProvider:
    provider_name = str(settings.data.get("provider", "yfinance")).lower()
    synthetic = SyntheticProvider(seed=int(settings.data.get("synthetic_seed", 7)))
    if provider_name == "synthetic":
        return synthetic
    if provider_name == "yfinance":
        provider: MarketDataProvider = YFinanceProvider()
    else:
        raise ValueError(f"Unsupported market data provider: {provider_name}")

    if settings.data.get("allow_synthetic_fallback", False):
        return FallbackProvider(provider, synthetic, min_rows=int(settings.data.get("min_rows", 80)))
    return provider


def fetch_intraday_data(symbol: str, settings: Settings, provider: MarketDataProvider | None = None) -> pd.DataFrame | None:
    """Fetch today's intraday bars (yfinance only). Returns None when unavailable so callers degrade gracefully."""
    provider = provider or get_market_data_provider(settings)
    intraday_cfg = settings.data.get("intraday", {}) or {}
    if not intraday_cfg.get("enabled", True):
        return None
    period = str(intraday_cfg.get("period", "1d"))
    interval = str(intraday_cfg.get("interval", "1m"))
    prepost = bool(intraday_cfg.get("include_premarket", True))
    cache_ttl = float(intraday_cfg.get("cache_ttl_seconds", 60) or 0)
    cache_key = (getattr(provider, "name", "provider"), "intraday", symbol.upper(), period, interval, prepost)
    if cache_ttl > 0:
        _INTRADAY_CACHE.ttl_seconds = cache_ttl
        cached = _INTRADAY_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()
    fetcher = getattr(provider, "fetch_intraday", None)
    if fetcher is None:
        return None
    try:
        frame = fetcher(symbol.upper(), period=period, interval=interval, prepost=prepost)
    except Exception as exc:
        LOGGER.info("Intraday fetch for %s failed: %s", symbol.upper(), exc)
        return None
    if frame is None or frame.empty:
        return None
    if cache_ttl > 0:
        _INTRADAY_CACHE.set(cache_key, frame.copy())
    return frame


def fetch_market_data(symbol: str, settings: Settings, provider: MarketDataProvider | None = None) -> pd.DataFrame:
    provider = provider or get_market_data_provider(settings)
    period = str(settings.data.get("period", "6mo"))
    interval = str(settings.data.get("interval", "1d"))
    cache_ttl = float(settings.data.get("cache_ttl_seconds", 0) or 0)
    cache_key = (getattr(provider, "name", "provider"), symbol.upper(), period, interval)
    if cache_ttl > 0:
        _MARKET_DATA_CACHE.ttl_seconds = cache_ttl
        cached = _MARKET_DATA_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()
    frame = provider.fetch(symbol.upper(), period=period, interval=interval)
    min_rows = int(settings.data.get("min_rows", 80))
    if len(frame) < min_rows:
        raise ValueError(f"{symbol.upper()} returned {len(frame)} rows; minimum is {min_rows}")
    if cache_ttl > 0:
        _MARKET_DATA_CACHE.set(cache_key, frame.copy())
    return frame


def build_snapshot(symbol: str, frame: pd.DataFrame, settings: Settings, provider_name: str) -> MarketSnapshot:
    latest = frame.iloc[-1]
    previous_close = frame["Close"].iloc[-2] if len(frame) > 1 else latest["Close"]
    avg_volume = frame["Volume"].tail(int(settings.features.get("volume_window", 20))).mean()
    change_pct = ((latest["Close"] / previous_close) - 1) if previous_close else 0.0
    as_of = frame.index[-1].isoformat() if hasattr(frame.index[-1], "isoformat") else datetime.now(timezone.utc).isoformat()
    return MarketSnapshot(
        symbol=symbol.upper(),
        as_of=as_of,
        timeframe=str(settings.data.get("interval", "1d")),
        provider=str(frame.attrs.get("provider", provider_name)),
        rows=int(len(frame)),
        latest_close=to_float(latest["Close"]),
        latest_volume=to_float(latest.get("Volume")),
        change_pct=to_float(change_pct),
        avg_volume=to_float(avg_volume),
    )


def normalize_ohlcv(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"No market data returned for {symbol.upper()}")
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [column[0] for column in normalized.columns]
    rename_map = {column: str(column).title() for column in normalized.columns}
    normalized = normalized.rename(columns=rename_map)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Market data for {symbol.upper()} missing column(s): {', '.join(missing)}")
    normalized = normalized[required].apply(pd.to_numeric, errors="coerce").dropna()
    normalized = normalized.sort_index()
    normalized.index = pd.to_datetime(normalized.index)
    return normalized


def _period_to_rows(period: str, interval: str) -> int:
    period = period.lower()
    interval = interval.lower()
    trading_days = _period_to_trading_days(period)
    minutes = _interval_to_minutes(interval)
    if minutes is not None:
        return max(30, trading_days * max(1, 390 // minutes))
    if interval.endswith("wk"):
        interval_weeks = int(interval[:-2] or 1)
        return max(4, trading_days // max(1, interval_weeks * 5))
    if interval.endswith("d"):
        interval_days = int(interval[:-1] or 1)
        return max(30, trading_days // max(1, interval_days))
    return max(60, trading_days)


def _period_to_trading_days(period: str) -> int:
    if period.endswith("y"):
        return max(1, int(period[:-1] or 1) * 252)
    if period.endswith("mo"):
        return max(1, int(period[:-2] or 6) * 21)
    if period.endswith("d"):
        return max(1, int(period[:-1] or 30))
    return 180


def _interval_to_minutes(interval: str) -> int | None:
    if interval.endswith("m"):
        return max(1, int(interval[:-1] or 1))
    if interval.endswith("h"):
        return max(1, int(interval[:-1] or 1) * 60)
    return None
