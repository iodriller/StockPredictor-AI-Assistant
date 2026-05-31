from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.config import Settings
from stockpredictor.contracts import ModelPrediction
from stockpredictor.models.base import PredictionModel, direction_from_return, direction_threshold
from stockpredictor.utils import clamp


class MomentumModel(PredictionModel):
    """Feature-aware trend/momentum forecast.

    Unlike the price-vs-index extrapolators, this reads the same evidence a trader
    reads on the chart — moving-average alignment, the slope of the mid MA, and
    recent realized momentum — so a stock that is bullish *now* reads bullish here,
    even if a long unweighted fit still leans on stale history. This is the model
    that keeps the system from saying "down" about a clean reversal up.
    """

    name = "momentum"

    def predict(self, symbol: str, frame: pd.DataFrame, settings: Settings) -> ModelPrediction:
        model_cfg = settings.models.get(self.name, {})
        continuation = float(model_cfg.get("continuation_factor", 0.6))
        horizon = int(settings.models.get("horizon_days", 5))
        lookback = min(int(settings.models.get("lookback_rows", 180)), len(frame))
        close = frame["Close"].tail(lookback).astype(float).reset_index(drop=True)
        current = float(close.iloc[-1])
        n = len(close)
        if n < 5 or current <= 0:
            return ModelPrediction(
                model=self.name,
                symbol=symbol.upper(),
                horizon_days=horizon,
                direction="flat",
                expected_return=0.0,
                confidence=0.05,
                predicted_price=current,
                metadata={"note": "insufficient history"},
            )

        k = max(1, min(horizon, n - 1))
        recent_ret = current / float(close.iloc[-1 - k]) - 1.0

        sma_fast = float(close.rolling(min(9, n)).mean().iloc[-1])
        sma_mid_series = close.rolling(min(20, n)).mean()
        sma_mid = float(sma_mid_series.iloc[-1])
        sma_slow = float(close.rolling(min(50, n)).mean().iloc[-1])

        # Mid-MA slope over the recent horizon — captures whether the trend itself is
        # turning, which a single recent return can miss.
        mid_prev = float(sma_mid_series.iloc[-1 - k]) if not np.isnan(sma_mid_series.iloc[-1 - k]) else sma_mid
        ma_slope_ret = (sma_mid / mid_prev - 1.0) if mid_prev else 0.0

        expected_return = continuation * (0.6 * recent_ret + 0.4 * ma_slope_ret)

        # Moving-average alignment in [-1, 1]: stacked bullish vs bearish.
        align = np.mean([
            np.sign(current - sma_fast),
            np.sign(sma_fast - sma_mid),
            np.sign(sma_mid - sma_slow),
        ])
        # Consistency: share of up-days over the recent window mapped to [-1, 1].
        recent_changes = np.sign(np.diff(close.iloc[-1 - k:].to_numpy()))
        consistency = float(np.mean(recent_changes)) if recent_changes.size else 0.0

        confidence = clamp(
            0.20 + 0.35 * abs(float(align)) + 0.25 * abs(consistency) + min(abs(expected_return) * 10, 0.20),
            0.05,
            0.90,
        )
        predicted = current * (1.0 + expected_return)
        return ModelPrediction(
            model=self.name,
            symbol=symbol.upper(),
            horizon_days=horizon,
            direction=direction_from_return(expected_return, direction_threshold(settings, horizon)),
            expected_return=float(expected_return),
            confidence=float(confidence),
            predicted_price=float(predicted),
            metadata={
                "recent_return": float(recent_ret),
                "ma_slope_return": float(ma_slope_ret),
                "ma_alignment": float(align),
                "consistency": consistency,
            },
        )
