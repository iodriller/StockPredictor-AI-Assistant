from __future__ import annotations

import pandas as pd

from stockpredictor.config import Settings
from stockpredictor.contracts import ModelPrediction
from stockpredictor.models.arima import ArimaPriceModel
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
) -> list[ModelPrediction]:
    names = model_names or settings.enabled_models()
    predictions: list[ModelPrediction] = []
    for name in names:
        model_class = MODEL_REGISTRY.get(name)
        if not model_class:
            continue
        try:
            predictions.append(model_class().predict(symbol, frame, settings))
        except Exception as exc:
            predictions.append(
                ModelPrediction(
                    model=name,
                    symbol=symbol.upper(),
                    horizon_days=int(settings.models.get("horizon_days", 5)),
                    direction="flat",
                    expected_return=0.0,
                    confidence=0.0,
                    predicted_price=float(frame["Close"].iloc[-1]),
                    metadata={"error": str(exc)},
                )
            )
    return predictions

