from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.config import Settings
from stockpredictor.contracts import ModelPrediction
from stockpredictor.models.base import PredictionModel, direction_from_return
from stockpredictor.utils import clamp


class BaselineTrendModel(PredictionModel):
    name = "baseline"

    def predict(self, symbol: str, frame: pd.DataFrame, settings: Settings) -> ModelPrediction:
        horizon = int(settings.models.get("horizon_days", 5))
        lookback = min(int(settings.models.get("lookback_rows", 180)), len(frame))
        close = frame["Close"].tail(lookback).to_numpy(dtype=float)
        current = float(close[-1])
        x = np.arange(len(close), dtype=float)
        slope, intercept = np.polyfit(x, close, 1)
        predicted = float(intercept + slope * (len(close) - 1 + horizon))
        expected_return = (predicted / current) - 1
        corr = np.corrcoef(x, close)[0, 1] if len(close) > 2 else 0.0
        if np.isnan(corr):
            corr = 0.0
        confidence = clamp(abs(corr) * 0.45 + min(abs(expected_return) * 12, 0.35), 0.05, 0.80)
        return ModelPrediction(
            model=self.name,
            symbol=symbol.upper(),
            horizon_days=horizon,
            direction=direction_from_return(expected_return),
            expected_return=float(expected_return),
            confidence=float(confidence),
            predicted_price=predicted,
            metadata={"lookback_rows": lookback, "slope": float(slope)},
        )

