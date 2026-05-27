from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from collections.abc import Iterable
from typing import Any, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd


T = TypeVar("T")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_in_timezone_iso(timezone_name: str | None) -> str:
    if not timezone_name:
        return now_utc_iso()
    try:
        return datetime.now(ZoneInfo(timezone_name)).isoformat()
    except ZoneInfoNotFoundError:
        return now_utc_iso()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def dedupe_preserve_order(values: Iterable[T]) -> list[T]:
    output: list[T] = []
    seen: set[T] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def clean_symbol_list(symbols: Iterable[str]) -> list[str]:
    return dedupe_preserve_order(symbol.strip().upper() for symbol in symbols if symbol and symbol.strip())


class TTLCache:
    """Tiny TTL cache for in-process use. Not thread-safe; fine for the dashboard/API single-process model."""

    def __init__(self, ttl_seconds: float) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self._store: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any) -> Any | None:
        record = self._store.get(key)
        if record is None:
            return None
        expires_at, value = record
        if expires_at < datetime.now(timezone.utc).timestamp():
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: Any, value: Any) -> None:
        expires_at = datetime.now(timezone.utc).timestamp() + self.ttl_seconds
        self._store[key] = (expires_at, value)

    def clear(self) -> None:
        self._store.clear()


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return to_serializable(value.reset_index().to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return to_serializable(value.to_dict())
    if isinstance(value, np.ndarray):
        return to_serializable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
