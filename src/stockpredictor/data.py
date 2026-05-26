from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
import pandas as pd

from .config import Settings
from .contracts import MarketSnapshot
from .utils import to_float


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


@dataclass
class SyntheticProvider:
    seed: int = 7
    name: str = "synthetic"

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        rows = _period_to_rows(period, interval)
        stable_symbol_seed = sum((idx + 1) * ord(char) for idx, char in enumerate(symbol.upper()))
        rng = np.random.default_rng(self.seed + stable_symbol_seed)
        index = pd.date_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=rows, freq="B")
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
        except Exception:
            pass
        frame = self.fallback.fetch(symbol, period, interval)
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


def fetch_market_data(symbol: str, settings: Settings, provider: MarketDataProvider | None = None) -> pd.DataFrame:
    provider = provider or get_market_data_provider(settings)
    frame = provider.fetch(
        symbol.upper(),
        period=str(settings.data.get("period", "6mo")),
        interval=str(settings.data.get("interval", "1d")),
    )
    min_rows = int(settings.data.get("min_rows", 80))
    if len(frame) < min_rows:
        raise ValueError(f"{symbol.upper()} returned {len(frame)} rows; minimum is {min_rows}")
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
    if interval.endswith("m") or interval.endswith("h"):
        return 390
    if period.endswith("y"):
        return max(120, int(period[:-1] or 1) * 252)
    if period.endswith("mo"):
        return max(60, int(period[:-2] or 6) * 21)
    if period.endswith("d"):
        return max(30, int(period[:-1] or 30))
    return 180
