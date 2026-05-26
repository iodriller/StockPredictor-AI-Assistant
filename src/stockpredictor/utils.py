from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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

