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
        close = frame["Close"].tail(max_rows).to_numpy(dtype=float)
        current = float(close[-1])
        # Model log-RETURNS, not the price level. Regressing price on a time index and
        # extrapolating past the training range makes a GP revert to its mean — i.e.
        # predict a pullback exactly when price is making new highs. Fitting the return
        # series and reading the smoothed current drift avoids that artifact.
        log_returns = np.diff(np.log(np.clip(close, 1e-9, None)))
        if len(log_returns) < 5:
            drift = float(np.mean(log_returns)) if len(log_returns) else 0.0
            predicted = current * float(np.exp(drift * horizon))
            expected_return = (predicted / current) - 1
            return ModelPrediction(
                model=self.name,
                symbol=symbol.upper(),
                horizon_days=horizon,
                direction=direction_from_return(expected_return, direction_threshold(settings, horizon)),
                expected_return=float(expected_return),
                confidence=0.1,
                predicted_price=predicted,
                metadata={"train_rows": max_rows, "note": "insufficient history for GP"},
            )
        x_raw = np.arange(len(log_returns), dtype=float).reshape(-1, 1)
        x_scaler = MinMaxScaler()
        x_train = x_scaler.fit_transform(x_raw)
        kernel_name = str(model_cfg.get("kernel", "matern")).lower()
        kernel = Matern(length_scale=1.0, nu=1.5) if kernel_name == "matern" else RBF(length_scale=1.0)
        kernel = kernel + WhiteKernel(noise_level=0.01)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=int(model_cfg.get("n_restarts_optimizer", 0)),
            random_state=42,
        )
        # Use the GP's smoothed AVERAGE drift over the window, not the last point — a
        # single recent outlier return read off the endpoint and compounded over the
        # horizon produces absurd forecasts (e.g. +55%). Clamp the per-step drift to a
        # sane band and the compounded move to a realistic range.
        max_daily_drift = float(model_cfg.get("max_daily_drift_pct", 0.02))
        max_expected_move = float(model_cfg.get("max_expected_move_pct", 0.40))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(x_train, log_returns)
            drift_all, sigma_all = gp.predict(x_train, return_std=True)
        drift = clamp(float(np.mean(drift_all)), -max_daily_drift, max_daily_drift)
        sigma = float(np.mean(sigma_all))
        expected_return = clamp(float(np.exp(drift * horizon) - 1.0), -max_expected_move, max_expected_move)
        predicted = current * (1.0 + expected_return)
        uncertainty = current * sigma * (horizon ** 0.5)
        uncertainty_penalty = clamp(sigma * (horizon ** 0.5) * 12, 0.0, 0.8)
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
