"""Budget cost accounting in OpenRouter-USD units scaled to match gamma.

Paper/operating point: gamma=300 means $0.0003 per prompt on average, via
``budget_cost = usd_cost * COST_USD_SCALE`` with ``COST_USD_SCALE = 1e6``.

JOVE expected node costs use MiniLM softmax histories in
``tools.resource_estimation``. This tracker keeps realized USD bookkeeping
(per-model EMA of ``usage.cost * COST_USD_SCALE``) for logging and as a
fallback when a call has no USD usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

# $0.0003 USD/prompt <-> gamma=300
DEFAULT_COST_USD_SCALE = 1_000_000.0


def extract_usage_usd(usage: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Return OpenRouter ``usage.cost`` in USD, or None if unavailable."""
    if not isinstance(usage, Mapping):
        return None
    raw = usage.get("cost")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def usd_to_budget_cost(usd: float, scale: float = DEFAULT_COST_USD_SCALE) -> float:
    return float(usd) * float(scale)


def usage_to_budget_cost(
    usage: Optional[Mapping[str, Any]],
    *,
    scale: float = DEFAULT_COST_USD_SCALE,
    fallback_budget: float = 0.0,
) -> float:
    """Map one API ``usage`` blob to budget units (USD * scale), else fallback."""
    usd = extract_usage_usd(usage)
    if usd is None:
        return float(fallback_budget)
    return usd_to_budget_cost(usd, scale=scale)


@dataclass
class ModelBudgetCostTracker:
    """Online mean of per-call budget cost (USD * scale) keyed by model id."""

    scale: float = DEFAULT_COST_USD_SCALE
    ema_alpha: float = 0.2
    counts: Dict[str, int] = field(default_factory=dict)
    means: Dict[str, float] = field(default_factory=dict)
    last_usd: Dict[str, float] = field(default_factory=dict)

    def update(self, model: str, usage: Optional[Mapping[str, Any]]) -> Optional[float]:
        """Incorporate one call. Returns budget cost if USD was present."""
        name = str(model or "").strip()
        if not name:
            return None
        usd = extract_usage_usd(usage)
        if usd is None:
            return None
        budget = usd_to_budget_cost(usd, scale=self.scale)
        self.last_usd[name] = float(usd)
        n = self.counts.get(name, 0)
        if n <= 0:
            self.means[name] = budget
        else:
            alpha = min(max(float(self.ema_alpha), 0.0), 1.0)
            prev = float(self.means.get(name, budget))
            self.means[name] = (1.0 - alpha) * prev + alpha * budget
        self.counts[name] = n + 1
        return budget

    def estimate(self, model: str, fallback_budget: float) -> float:
        name = str(model or "").strip()
        if name and self.counts.get(name, 0) > 0:
            return float(self.means[name])
        return float(fallback_budget)

    def snapshot(self) -> Dict[str, object]:
        return {
            "scale": float(self.scale),
            "ema_alpha": float(self.ema_alpha),
            "counts": dict(self.counts),
            "means": dict(self.means),
            "last_usd": dict(self.last_usd),
        }


def observe_call_usages(
    tracker: ModelBudgetCostTracker,
    calls: Iterable[Tuple[str, Optional[Mapping[str, Any]]]],
) -> Tuple[float, float]:
    """Update tracker from (model, usage) pairs.

    Returns ``(total_usd, total_budget_cost)`` using USD when present, else 0 for
    that call (tracker is not updated when USD is missing).
    """
    total_usd = 0.0
    total_budget = 0.0
    for model, usage in calls:
        usd = extract_usage_usd(usage)
        budget = tracker.update(model, usage)
        if usd is not None and budget is not None:
            total_usd += usd
            total_budget += budget
    return total_usd, total_budget


def realized_prompt_costs_from_usages(
    calls: Iterable[Tuple[str, Optional[Mapping[str, Any]], float]],
    *,
    scale: float = DEFAULT_COST_USD_SCALE,
) -> Tuple[float, float]:
    """Sum USD / budget cost for calls.

    Each item is ``(model, usage, fallback_budget)``. When ``usage.cost`` is
    missing (e.g. NIM), ``fallback_budget`` is used for the budget total and USD
    contribution is 0.
    """
    total_usd = 0.0
    total_budget = 0.0
    for _model, usage, fallback in calls:
        usd = extract_usage_usd(usage)
        if usd is None:
            total_budget += float(fallback)
        else:
            total_usd += float(usd)
            total_budget += usd_to_budget_cost(usd, scale=scale)
    return total_usd, total_budget


def queue_budget_with_verifier_factor(
    non_verifier_calls: Iterable[Tuple[str, Optional[Mapping[str, Any]], float]],
    verifier_calls: Iterable[Tuple[str, Optional[Mapping[str, Any]], float]],
    *,
    verifier_cost_factor: float,
    scale: float = DEFAULT_COST_USD_SCALE,
) -> Tuple[float, float, float]:
    """Budget cost for the virtual queue with discounted verifier spend.

    Returns ``(total_usd, full_budget, queue_budget)`` where ``total_usd`` /
    ``full_budget`` keep true billed verifier cost for logging, and
    ``queue_budget`` charges ``verifier_cost_factor ×`` verifier budget only
    (paper-style cheap-verifier assumption).
    """
    non_usd, non_budget = realized_prompt_costs_from_usages(non_verifier_calls, scale=scale)
    ver_usd, ver_budget = realized_prompt_costs_from_usages(verifier_calls, scale=scale)
    factor = float(verifier_cost_factor)
    queue_budget = float(non_budget) + factor * float(ver_budget)
    return float(non_usd + ver_usd), float(non_budget + ver_budget), float(queue_budget)
