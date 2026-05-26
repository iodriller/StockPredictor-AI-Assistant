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

