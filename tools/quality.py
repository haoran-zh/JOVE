from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class ServiceLabel:
    feature: np.ndarray
    label: float
    weight: float = 1.0


class WeightedLinearQualityModel:
    """Online weighted linear model with LinUCB-style uncertainty.

    The model estimates residual quality around a neutral prior, so the cold-start
    prediction is prior_mean for every role/API/context feature.
    """

    def __init__(self, dimension: int, lambda_reg: float = 1.0, prior_mean: float = 0.0) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if lambda_reg <= 0:
            raise ValueError("lambda_reg must be positive")
        self.dimension = dimension
        self.lambda_reg = float(lambda_reg)
        self.prior_mean = float(prior_mean)
        self.g_inv = np.eye(dimension, dtype=float) / self.lambda_reg
        self.r = np.zeros(dimension, dtype=float)
        self.theta = np.zeros(dimension, dtype=float)
        self.num_updates = 0

    @staticmethod
    def _as_feature(feature: np.ndarray) -> np.ndarray:
        vector = np.asarray(feature, dtype=float).reshape(-1)
        if not np.all(np.isfinite(vector)):
            raise ValueError("feature contains non-finite values")
        return vector

    @staticmethod
    def _clip01(value: float) -> float:
        return min(max(float(value), 0.0), 1.0)

    def predict(self, feature: np.ndarray) -> float:
        z = self._as_feature(feature)
        if z.shape[0] != self.dimension:
            raise ValueError(f"feature dimension {z.shape[0]} != model dimension {self.dimension}")
        return self._clip01(self.prior_mean + float(z @ self.theta))

    def uncertainty(self, feature: np.ndarray) -> float:
        z = self._as_feature(feature)
        if z.shape[0] != self.dimension:
            raise ValueError(f"feature dimension {z.shape[0]} != model dimension {self.dimension}")
        value = float(z @ self.g_inv @ z)
        return float(np.sqrt(max(value, 0.0)))

    def ucb(self, feature: np.ndarray, beta: float) -> float:
        z = self._as_feature(feature)
        if z.shape[0] != self.dimension:
            raise ValueError(f"feature dimension {z.shape[0]} != model dimension {self.dimension}")
        return self._clip01(self.prior_mean + float(z @ self.theta) + float(beta) * self.uncertainty(z))

    def update(self, feature: np.ndarray, label: float, weight: float = 1.0) -> None:
        z = self._as_feature(feature)
        if z.shape[0] != self.dimension:
            raise ValueError(f"feature dimension {z.shape[0]} != model dimension {self.dimension}")
        y = self._clip01(label)
        alpha = max(float(weight), 0.0)
        if alpha == 0.0:
            return
        gz = self.g_inv @ z
        denom = 1.0 + alpha * float(z @ gz)
        self.g_inv = self.g_inv - (alpha / denom) * np.outer(gz, gz)
        self.r = self.r + alpha * z * (y - self.prior_mean)
        self.theta = self.g_inv @ self.r
        self.num_updates += 1

    def update_many(self, labels: List[ServiceLabel]) -> None:
        for item in labels:
            self.update(item.feature, item.label, item.weight)
