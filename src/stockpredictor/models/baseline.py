from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.config import Settings
from stockpredictor.contracts import ModelPrediction
from stockpredictor.models.base import PredictionModel, direction_from_return, direction_threshold
from stockpredictor.utils import clamp


class BaselineTrendModel(PredictionModel):
    name = "baseline"

    def predict(self, symbol: str, frame: pd.DataFrame, settings: Settings) -> ModelPrediction:
        horizon = int(settings.models.get("horizon_days", 5))
        lookback = min(int(settings.models.get("lookback_rows", 180)), len(frame))
        close = frame["Close"].tail(lookback).to_numpy(dtype=float)
        current = float(close[-1])
        x = np.arange(len(close), dtype=float)
        # Recency-weighted linear fit: a single unweighted line over a long lookback
        # is dominated by stale history and reports a falling trend for a stock that
        # just reversed up. Exponential weights emphasize the recent bars a trader
        # actually reads, while still using the full window for context.
        decay = float(settings.models.get(self.name, {}).get("recency_decay", 3.0))
        weights = np.exp(np.linspace(-abs(decay), 0.0, len(close))) if len(close) > 1 else None
        slope, intercept = np.polyfit(x, close, 1, w=weights)
        predicted = float(intercept + slope * (len(close) - 1 + horizon))
        expected_return = (predicted / current) - 1
        # Weighted correlation so the fit-quality signal also reflects recent bars.
        corr = _weighted_corr(x, close, weights) if len(close) > 2 else 0.0
        if np.isnan(corr):
            corr = 0.0
        confidence = clamp(abs(corr) * 0.45 + min(abs(expected_return) * 12, 0.35), 0.05, 0.80)
        return ModelPrediction(
            model=self.name,
            symbol=symbol.upper(),
            horizon_days=horizon,
            direction=direction_from_return(expected_return, direction_threshold(settings, horizon)),
            expected_return=float(expected_return),
            confidence=float(confidence),
            predicted_price=predicted,
            metadata={"lookback_rows": lookback, "slope": float(slope)},
        )


def _weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.corrcoef(x, y)[0, 1])
    w = weights / weights.sum()
    mx = float((w * x).sum())
    my = float((w * y).sum())
    cov = float((w * (x - mx) * (y - my)).sum())
    vx = float((w * (x - mx) ** 2).sum())
    vy = float((w * (y - my) ** 2).sum())
    denom = (vx * vy) ** 0.5
    return cov / denom if denom > 0 else 0.0

