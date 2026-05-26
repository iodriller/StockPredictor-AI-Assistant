from __future__ import annotations

import warnings

import pandas as pd

from stockpredictor.config import Settings
from stockpredictor.contracts import ModelPrediction
from stockpredictor.models.base import PredictionModel, direction_from_return
from stockpredictor.utils import clamp


class ArimaPriceModel(PredictionModel):
    name = "arima"

    def predict(self, symbol: str, frame: pd.DataFrame, settings: Settings) -> ModelPrediction:
        from statsmodels.tsa.arima.model import ARIMA

        model_cfg = settings.models.get(self.name, {})
        order = tuple(int(value) for value in model_cfg.get("order", [1, 1, 1]))
        horizon = int(settings.models.get("horizon_days", 5))
        max_rows = min(int(model_cfg.get("max_train_rows", 220)), len(frame))
        close = frame["Close"].tail(max_rows).astype(float)
        current = float(close.iloc[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(close, order=order).fit()
            forecast = fitted.forecast(steps=horizon)
        predicted = float(forecast.iloc[-1])
        expected_return = (predicted / current) - 1
        residual_std = float(getattr(fitted, "resid", pd.Series(dtype=float)).std() or 0.0)
        confidence = clamp(min(abs(expected_return) * 12, 0.45) + 0.25 - min(residual_std / current, 0.3), 0.05, 0.75)
        return ModelPrediction(
            model=self.name,
            symbol=symbol.upper(),
            horizon_days=horizon,
            direction=direction_from_return(expected_return),
            expected_return=float(expected_return),
            confidence=float(confidence),
            predicted_price=predicted,
            lower_bound=predicted - residual_std,
            upper_bound=predicted + residual_std,
            metadata={"order": order, "train_rows": max_rows},
        )

