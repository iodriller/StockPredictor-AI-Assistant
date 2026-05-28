from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from stockpredictor.config import Settings
from stockpredictor.contracts import ModelPrediction


class PredictionModel(ABC):
    name: str

    @abstractmethod
    def predict(self, symbol: str, frame: pd.DataFrame, settings: Settings) -> ModelPrediction:
        raise NotImplementedError


def direction_from_return(expected_return: float, threshold: float = 0.0025) -> str:
    if expected_return > threshold:
        return "up"
    if expected_return < -threshold:
        return "down"
    return "flat"


def direction_threshold(settings: Settings, horizon_days: int) -> float:
    models_cfg = settings.models
    explicit = models_cfg.get("direction_threshold_pct")
    if explicit is not None:
        return float(explicit)
    per_day = float(models_cfg.get("direction_threshold_per_day_pct", 0.0005))
    return per_day * max(1, int(horizon_days))


def horizon_model_params(settings: Settings, horizon: str | None) -> tuple[int, int]:
    """Resolve (horizon_days, lookback_rows) for the requested analysis horizon."""
    profile = settings.horizon_profile(horizon)
    horizon_days = int(profile.get("horizon_days", settings.models.get("horizon_days", 5)))
    lookback_rows = int(profile.get("lookback_rows", settings.models.get("lookback_rows", 180)))
    return horizon_days, lookback_rows

