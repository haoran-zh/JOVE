from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Deque, Dict, Optional, Sequence, Tuple


def validate_latency_tolerance(tolerance: float, name: str = "latency tolerance") -> float:
    value = float(tolerance)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value}")
    return value


def split_latency_tolerance(global_tolerance: float, num_tasks: int) -> float:
    task_count = int(num_tasks)
    if task_count <= 0:
        raise ValueError("num_tasks must be positive")
    return validate_latency_tolerance(global_tolerance, "global latency tolerance") / task_count


def conservative_empirical_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    q = float(quantile)
    if not 0.0 < q <= 1.0:
        raise ValueError(f"quantile must be in (0, 1], got {q}")
    sorted_values = sorted(float(value) for value in values)
    index = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


class RollingQuantileLatencyModel:
    """API/task-type latency model for conservative MILP bounds.

    If a per-node violation tolerance epsilon is supplied to safe_bound, the
    bound uses the conservative empirical (1 - epsilon)-quantile. Otherwise it
    falls back to the constructor quantile for backward-compatible callers.
    """

    def __init__(self, quantile: float = 0.9, margin_seconds: float = 0.25, max_samples: int = 100) -> None:
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")
        self.quantile = float(quantile)
        self.margin_seconds = float(margin_seconds)
        self.samples: Dict[Tuple[str, str], Deque[float]] = defaultdict(lambda: deque(maxlen=max_samples))

    def update(self, api_id: str, task_type: str, latency_seconds: float) -> None:
        if latency_seconds >= 0:
            self.samples[(api_id, task_type)].append(float(latency_seconds))

    def safe_bound(
        self,
        api_id: str,
        task_type: str,
        fallback_seconds: float,
        tolerance: Optional[float] = None,
    ) -> float:
        values = list(self.samples.get((api_id, task_type), []))
        if not values:
            return float(fallback_seconds)
        quantile = self.quantile
        if tolerance is not None:
            quantile = 1.0 - validate_latency_tolerance(tolerance, "per-node latency tolerance")
        return float(conservative_empirical_quantile(values, quantile) + self.margin_seconds)
