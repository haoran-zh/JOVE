"""Feature-weighted cost/latency estimators for JOVE.

JOVE uses paper eqs. (11)–(12): softmax-weighted MiniLM cost mean and latency
quantile, shrinking toward catalog priors. Empty histories fall back to
model-specific priors; latency/cost prior weight is rho_n = n0 / (n0 + n).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Mapping, Optional, Sequence

import numpy as np

from .cost_accounting import DEFAULT_COST_USD_SCALE, extract_usage_usd, usd_to_budget_cost, usage_to_budget_cost
from .latency import validate_latency_tolerance

ROLE_EXEC = "exec"
ROLE_VER = "ver"


def _as_feature(feature: object) -> np.ndarray:
    vector = np.asarray(feature, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError("feature must be non-empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError("feature contains non-finite values")
    return vector


def softmax_kernel_weights(query: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Softmax of inner products; numerically stable and sums to one."""
    if features.ndim != 2:
        raise ValueError("features must have shape (n, d)")
    logits = features @ query
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(features.shape[0], 1.0 / features.shape[0], dtype=float)
    return weights / total


def weighted_empirical_quantile(values: Sequence[float], weights: Sequence[float], quantile: float) -> float:
    """Smallest u such that the weighted mass of {d <= u} is at least quantile."""
    q = float(quantile)
    if not 0.0 < q <= 1.0:
        raise ValueError(f"quantile must be in (0, 1], got {q}")
    samples = np.asarray(values, dtype=float).reshape(-1)
    mass = np.asarray(weights, dtype=float).reshape(-1)
    if samples.size == 0:
        raise ValueError("values must be non-empty")
    if samples.shape != mass.shape:
        raise ValueError("values and weights must have the same length")
    if np.any(mass < 0.0) or not np.all(np.isfinite(mass)):
        raise ValueError("weights must be finite and non-negative")
    total = float(mass.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    mass = mass / total
    order = np.argsort(samples, kind="mergesort")
    cume = np.cumsum(mass[order])
    index = int(np.searchsorted(cume, q, side="left"))
    index = min(max(index, 0), samples.size - 1)
    return float(samples[order][index])


@dataclass
class ResourceRecord:
    feature: np.ndarray
    cost: float
    latency: float


class FeatureWeightedResourceModel:
    """Softmax-weighted cost and latency estimates over MiniLM features."""

    def __init__(
        self,
        prior_count: float = 1.0,
        max_samples: int = 200,
        margin_seconds: float = 0.25,
    ) -> None:
        if prior_count < 0.0:
            raise ValueError("prior_count must be non-negative")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.prior_count = float(prior_count)
        self.margin_seconds = float(margin_seconds)
        self.exec_history: Dict[str, Deque[ResourceRecord]] = defaultdict(lambda: deque(maxlen=max_samples))
        self.ver_history: Dict[str, Deque[ResourceRecord]] = defaultdict(lambda: deque(maxlen=max_samples))

    def _history(self, role: str) -> Dict[str, Deque[ResourceRecord]]:
        if role == ROLE_VER:
            return self.ver_history
        if role == ROLE_EXEC:
            return self.exec_history
        raise ValueError(f"unknown role {role!r}")

    def _append(self, history: Deque[ResourceRecord], feature: object, cost: float, latency: float) -> None:
        if not np.isfinite(cost):
            return
        if latency < 0.0 or not np.isfinite(latency):
            return
        history.append(
            ResourceRecord(
                feature=_as_feature(feature).copy(),
                cost=float(cost),
                latency=float(latency),
            )
        )

    def update_execution(self, api_id: str, feature: object, cost: float, latency: float) -> None:
        name = str(api_id or "").strip()
        if not name:
            return
        self._append(self.exec_history[name], feature, cost, latency)

    def update_verification(self, api_id: str, feature: object, cost: float, latency: float = 0.0) -> None:
        name = str(api_id or "").strip()
        if not name:
            return
        self._append(self.ver_history[name], feature, cost, latency)

    def _records(self, api_id: str, role: str) -> Sequence[ResourceRecord]:
        name = str(api_id or "").strip()
        if not name:
            return ()
        return tuple(self._history(role).get(name, ()))

    def _shrink(self, estimate: float, prior: float, n: int) -> float:
        if n <= 0:
            return float(prior)
        rho = self.prior_count / (self.prior_count + float(n))
        return rho * float(prior) + (1.0 - rho) * float(estimate)

    def estimate_cost(self, api_id: str, feature: object, prior: float, *, role: str = ROLE_EXEC) -> float:
        records = self._records(api_id, role)
        if not records:
            return float(prior)
        query = _as_feature(feature)
        stacked = np.stack([item.feature for item in records], axis=0)
        if stacked.shape[1] != query.shape[0]:
            raise ValueError(f"feature dimension {query.shape[0]} != history dimension {stacked.shape[1]}")
        weights = softmax_kernel_weights(query, stacked)
        costs = np.asarray([item.cost for item in records], dtype=float)
        weighted = float(weights @ costs)
        return self._shrink(weighted, prior, len(records))

    def estimate_realized_mean_cost(self, api_id: str, prior: float, *, role: str = ROLE_EXEC) -> float:
        """Per-LLM expected cost from billed USD×scale observations.

        Catalog ``prior`` is used only when this API has no realized history.
        Feature kernel / catalog shrinkage are not applied. JOVE expected costs
        use ``estimate_cost`` instead.
        """
        records = self._records(api_id, role)
        if not records:
            return float(prior)
        return float(sum(item.cost for item in records) / len(records))

    def estimate_latency(
        self,
        api_id: str,
        feature: object,
        prior: float,
        tolerance: Optional[float] = None,
        quantile: float = 0.9,
    ) -> float:
        records = self._records(api_id, ROLE_EXEC)
        if not records:
            return float(prior)
        if tolerance is not None:
            quantile = 1.0 - validate_latency_tolerance(tolerance, "per-node latency tolerance")
        query = _as_feature(feature)
        stacked = np.stack([item.feature for item in records], axis=0)
        if stacked.shape[1] != query.shape[0]:
            raise ValueError(f"feature dimension {query.shape[0]} != history dimension {stacked.shape[1]}")
        weights = softmax_kernel_weights(query, stacked)
        latencies = [item.latency for item in records]
        weighted = weighted_empirical_quantile(latencies, weights, quantile)
        return self._shrink(weighted, prior, len(records)) + self.margin_seconds


def observe_executor_call(
    model: FeatureWeightedResourceModel,
    api_id: str,
    feature: object,
    usage: Optional[Mapping[str, object]],
    fallback_cost: float,
    latency_seconds: float,
    scale: float = DEFAULT_COST_USD_SCALE,
) -> None:
    cost = usage_to_budget_cost(usage, scale=scale, fallback_budget=fallback_cost)
    model.update_execution(api_id, feature, cost, latency_seconds)


def observe_verifier_call(
    model: FeatureWeightedResourceModel,
    api_id: str,
    feature: object,
    usage: Optional[Mapping[str, object]],
    fallback_cost: float,
    latency_seconds: float = 0.0,
    scale: float = DEFAULT_COST_USD_SCALE,
    cost_factor: float = 1.0,
) -> None:
    """Record a verifier observation for JOVE expected ``C_ver``.

    When billed ``usage.cost`` is present, multiply by ``cost_factor`` (default
    catalog ``verifier_cost_factor``, e.g. 0.01) so the cheap-verifier assumption
    applies under OpenRouter USD accounting. Catalog ``fallback_cost`` is already
    ``executor_cost_unit * verifier_cost_factor`` and is left unchanged.
    """
    usd = extract_usage_usd(usage)
    if usd is None:
        cost = float(fallback_cost)
    else:
        cost = usd_to_budget_cost(usd, scale=scale) * float(cost_factor)
    model.update_verification(api_id, feature, cost, latency_seconds)


def run_self_checks() -> None:
    """Unit checks for softmax cost, weighted quantiles, and prior shrinkage."""
    query = np.ones(2, dtype=float)
    query = query / np.linalg.norm(query)
    empty = FeatureWeightedResourceModel(prior_count=1.0, margin_seconds=0.0)
    assert empty.estimate_cost("api", query, 9.0) == 9.0
    assert empty.estimate_latency("api", query, 9.0, tolerance=0.05) == 9.0

    identical = FeatureWeightedResourceModel(prior_count=0.0, margin_seconds=0.0)
    for latency, cost in ((0.1, 1.0), (0.2, 1.0), (0.22, 1.0), (0.7, 1.0)):
        identical.update_execution("api", query, cost, latency)
    assert identical.estimate_latency("api", query, 9.0, tolerance=0.05) == 0.7

    axis_x = np.array([1.0, 0.0], dtype=float)
    axis_y = np.array([0.0, 1.0], dtype=float)
    mixed = FeatureWeightedResourceModel(prior_count=0.0, margin_seconds=0.0)
    mixed.update_execution("api", axis_x, 1.0, 0.1)
    mixed.update_execution("api", axis_y, 10.0, 1.0)
    cheap_cost = mixed.estimate_cost("api", axis_x, 5.0)
    expensive_cost = mixed.estimate_cost("api", axis_y, 5.0)
    assert cheap_cost < expensive_cost
    cheap_latency = mixed.estimate_latency("api", axis_x, 5.0, tolerance=0.5)
    expensive_latency = mixed.estimate_latency("api", axis_y, 5.0, tolerance=0.5)
    assert cheap_latency < expensive_latency

    shrink = FeatureWeightedResourceModel(prior_count=1.0, margin_seconds=0.0)
    shrink.update_execution("api", query, 4.0, 2.0)
    assert abs(shrink.estimate_cost("api", query, 8.0) - 6.0) < 1e-12
    assert abs(shrink.estimate_latency("api", query, 8.0, tolerance=0.05) - 5.0) < 1e-12

    usd = FeatureWeightedResourceModel(prior_count=1.0, margin_seconds=0.0)
    assert usd.estimate_realized_mean_cost("api", 9.0) == 9.0
    usd.update_execution("api", axis_x, 40.0, 0.1)
    usd.update_execution("api", axis_y, 60.0, 1.0)
    assert abs(usd.estimate_realized_mean_cost("api", 9.0) - 50.0) < 1e-12
    usd.update_verification("ver", axis_x, 12.0, 0.0)
    usd.update_verification("ver", axis_y, 18.0, 0.0)
    assert abs(usd.estimate_realized_mean_cost("ver", 1.0, role=ROLE_VER) - 15.0) < 1e-12

    # Cheap-verifier assumption: billed USD observations are scaled by cost_factor.
    cheap = FeatureWeightedResourceModel(prior_count=0.0, margin_seconds=0.0)
    observe_verifier_call(
        cheap,
        "ver",
        axis_x,
        {"cost": 1.0},
        fallback_cost=2.0,
        scale=100.0,
        cost_factor=0.01,
    )
    assert abs(cheap.estimate_cost("ver", axis_x, 2.0, role=ROLE_VER) - 1.0) < 1e-12


if __name__ == "__main__":
    run_self_checks()
    print("resource_estimation self-checks ok")
