from __future__ import annotations

from copy import deepcopy

import pandas as pd

from stockpredictor.config import Settings
from stockpredictor.contracts import ModelPrediction
from stockpredictor.models.arima import ArimaPriceModel
from stockpredictor.models.base import horizon_model_params
from stockpredictor.models.baseline import BaselineTrendModel
from stockpredictor.models.gaussian_process import GaussianProcessPriceModel


MODEL_REGISTRY = {
    "baseline": BaselineTrendModel,
    "gaussian_process": GaussianProcessPriceModel,
    "arima": ArimaPriceModel,
}


def run_models(
    symbol: str,
    frame: pd.DataFrame,
    settings: Settings,
    model_names: list[str] | None = None,
    horizon: str | None = None,
) -> list[ModelPrediction]:
    names = model_names or settings.enabled_models()
    settings_for_models = _settings_with_horizon(settings, horizon)
    predictions: list[ModelPrediction] = []
    for name in names:
        model_class = MODEL_REGISTRY.get(name)
        if not model_class:
            continue
        try:
            predictions.append(model_class().predict(symbol, frame, settings_for_models))
        except Exception as exc:
            predictions.append(
                ModelPrediction(
                    model=name,
                    symbol=symbol.upper(),
                    horizon_days=int(settings_for_models.models.get("horizon_days", 5)),
                    direction="flat",
                    expected_return=0.0,
                    confidence=0.0,
                    predicted_price=float(frame["Close"].iloc[-1]),
                    metadata={"error": str(exc)},
                )
            )
    return predictions


def _settings_with_horizon(settings: Settings, horizon: str | None) -> Settings:
    if horizon is None:
        return settings
    horizon_days, lookback_rows = horizon_model_params(settings, horizon)
    if horizon_days == int(settings.models.get("horizon_days", 5)) and lookback_rows == int(settings.models.get("lookback_rows", 180)):
        return settings
    raw = deepcopy(settings.raw)
    raw.setdefault("models", {})
    raw["models"]["horizon_days"] = horizon_days
    raw["models"]["lookback_rows"] = lookback_rows
    return Settings(raw=raw, path=settings.path)

