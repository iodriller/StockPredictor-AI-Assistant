from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from stockpredictor.config import Settings
from stockpredictor.contracts import ModelPrediction
from stockpredictor.models.base import PredictionModel, direction_from_return, direction_threshold
from stockpredictor.utils import clamp


class GaussianProcessPriceModel(PredictionModel):
    name = "gaussian_process"

    def predict(self, symbol: str, frame: pd.DataFrame, settings: Settings) -> ModelPrediction:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
        from sklearn.preprocessing import MinMaxScaler

        model_cfg = settings.models.get(self.name, {})
        horizon = int(settings.models.get("horizon_days", 5))
        configured_rows = min(
            int(settings.models.get("lookback_rows", model_cfg.get("max_train_rows", 160))),
            int(model_cfg.get("max_train_rows", settings.models.get("lookback_rows", 160))),
        )
        max_rows = min(configured_rows, len(frame))
        close = frame["Close"].tail(max_rows).to_numpy(dtype=float).reshape(-1, 1)
        current = float(close[-1, 0])
        x_raw = np.arange(max_rows, dtype=float).reshape(-1, 1)
        y_scaler = MinMaxScaler()
        x_scaler = MinMaxScaler()
        x_train = x_scaler.fit_transform(x_raw)
        y_train = y_scaler.fit_transform(close).ravel()
        kernel_name = str(model_cfg.get("kernel", "matern")).lower()
        kernel = Matern(length_scale=1.0, nu=1.5) if kernel_name == "matern" else RBF(length_scale=1.0)
        kernel = kernel + WhiteKernel(noise_level=0.01)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=int(model_cfg.get("n_restarts_optimizer", 0)),
            random_state=42,
        )
        x_future = x_scaler.transform(np.array([[max_rows - 1 + horizon]], dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(x_train, y_train)
            y_pred, sigma = gp.predict(x_future, return_std=True)
        predicted = float(y_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))[0, 0])
        uncertainty = float(sigma[0]) * max(float(close.max() - close.min()), 1.0)
        expected_return = (predicted / current) - 1
        uncertainty_penalty = clamp(uncertainty / max(current, 1.0), 0.0, 0.8)
        confidence = clamp(min(abs(expected_return) * 14, 0.55) + 0.25 - uncertainty_penalty, 0.05, 0.85)
        return ModelPrediction(
            model=self.name,
            symbol=symbol.upper(),
            horizon_days=horizon,
            direction=direction_from_return(expected_return, direction_threshold(settings, horizon)),
            expected_return=float(expected_return),
            confidence=float(confidence),
            predicted_price=predicted,
            lower_bound=predicted - uncertainty,
            upper_bound=predicted + uncertainty,
            metadata={"train_rows": max_rows, "kernel": str(gp.kernel_)},
        )
