from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .planning import Plan, TaskNode

NO_VERIFIER = "__none__"


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text)


@dataclass
class JoveSelection:
    executor_by_task: Dict[str, str]
    verifier_by_task: Dict[str, Optional[str]]
    finish_time: Dict[str, float]
    objective_value: float
    expected_cost: float

    def as_pairs(self) -> Dict[str, Tuple[str, Optional[str]]]:
        return {
            task_id: (self.executor_by_task[task_id], self.verifier_by_task[task_id])
            for task_id in self.executor_by_task
        }


@dataclass
class JoveInputs:
    exec_quality_ucb: Mapping[Tuple[str, str], float]
    exec_uncertainty: Mapping[Tuple[str, str], float]
    exec_cost: Mapping[Tuple[str, str], float]
    verifier_cost: Mapping[str, float]
    safe_latency: Mapping[Tuple[str, str], float]


def verification_information_gain(uncertainty: float) -> float:
    """D-optimal verification gain I(u) = (1/2) log(1 + u^2) from paper eq. (14)."""
    u = max(float(uncertainty), 0.0)
    return 0.5 * math.log(1.0 + u * u)


def uncertainty_reduction(uncertainty: float) -> float:
    """Alias for verification_information_gain (kept for call-site compatibility)."""
    return verification_information_gain(uncertainty)


def _choices_for_task(
    task: TaskNode,
    api_candidates: List[str],
    fixed_verifier_model: str,
    allow_self_verification: bool,
) -> List[Tuple[str, str]]:
    del task  # Choice set is task-indexed by callers; verifier model is fixed at run level.
    choices: List[Tuple[str, str]] = []
    for executor in api_candidates:
        choices.append((executor, NO_VERIFIER))
        if allow_self_verification or executor != fixed_verifier_model:
            choices.append((executor, fixed_verifier_model))
    return choices


def _objective_coeff(
    task_id: str,
    executor: str,
    verifier: str,
    coeffs: JoveInputs,
    q_t: float,
    k_c: float,
    k_v: float,
) -> float:
    value = (
        float(coeffs.exec_quality_ucb[(task_id, executor)])
        - k_c * q_t * float(coeffs.exec_cost[(task_id, executor)])
    )
    if verifier != NO_VERIFIER:
        value += (
            k_v * verification_information_gain(coeffs.exec_uncertainty[(task_id, executor)])
            - k_c * q_t * float(coeffs.verifier_cost[task_id])
        )
    return value


def _selected_cost(task_id: str, executor: str, verifier: str, coeffs: JoveInputs) -> float:
    cost = float(coeffs.exec_cost[(task_id, executor)])
    if verifier != NO_VERIFIER:
        cost += float(coeffs.verifier_cost[task_id])
    return cost


def _finish_times(plan: Plan, selected: Mapping[str, Tuple[str, str]], coeffs: JoveInputs) -> Dict[str, float]:
    finish: Dict[str, float] = {}
    tasks_by_id = plan.task_by_id()
    for task_id in plan.topological_order():
        executor, _verifier = selected[task_id]
        duration = float(coeffs.safe_latency[(task_id, executor)])
        task = tasks_by_id[task_id]
        pred_finish = max((finish[pred] for pred in task.predecessors), default=0.0)
        finish[task_id] = pred_finish + duration
    return finish


def _is_deadline_feasible(plan: Plan, finish: Mapping[str, float], mu_t: float) -> bool:
    return all(float(finish[sink]) <= mu_t + 1e-8 for sink in plan.sinks())


def is_jove_infeasible_error(exc: BaseException) -> bool:
    """True when JOVE has no deadline-feasible assignment (PuLP or enumeration)."""
    message = str(exc)
    return "Infeasible" in message or "No deadline-feasible" in message


def solve_jove_selection(
    plan: Plan,
    api_candidates: List[str],
    coeffs: JoveInputs,
    mu_t: float,
    q_t: float,
    k_c: float = 1.0,
    k_v: float = 1.0,
    fixed_verifier_model: str = "",
    allow_self_verification: bool = False,
    enum_limit: int = 200000,
) -> JoveSelection:
    try:
        import pulp  # type: ignore
    except ImportError:
        return _solve_by_enumeration(
            plan=plan,
            api_candidates=api_candidates,
            coeffs=coeffs,
            mu_t=mu_t,
            q_t=q_t,
            k_c=k_c,
            k_v=k_v,
            fixed_verifier_model=fixed_verifier_model,
            allow_self_verification=allow_self_verification,
            enum_limit=enum_limit,
        )
    return _solve_with_pulp(
        pulp=pulp,
        plan=plan,
        api_candidates=api_candidates,
        coeffs=coeffs,
        mu_t=mu_t,
        q_t=q_t,
        k_c=k_c,
        k_v=k_v,
        fixed_verifier_model=fixed_verifier_model,
        allow_self_verification=allow_self_verification,
    )


def _solve_with_pulp(
    pulp: object,
    plan: Plan,
    api_candidates: List[str],
    coeffs: JoveInputs,
    mu_t: float,
    q_t: float,
    k_c: float,
    k_v: float,
    fixed_verifier_model: str,
    allow_self_verification: bool,
) -> JoveSelection:
    problem = pulp.LpProblem("jove_api_allocation", pulp.LpMaximize)
    z = {}
    choices_by_task = {
        task.id: _choices_for_task(task, api_candidates, fixed_verifier_model, allow_self_verification)
        for task in plan.tasks
    }
    for task in plan.tasks:
        for executor, verifier in choices_by_task[task.id]:
            z[(task.id, executor, verifier)] = pulp.LpVariable(
                f"z_{task.id}_{sanitize_name(executor)}_{sanitize_name(verifier)}",
                lowBound=0,
                upBound=1,
                cat="Binary",
            )
    F = {task.id: pulp.LpVariable(f"F_{task.id}", lowBound=0, cat="Continuous") for task in plan.tasks}

    problem += pulp.lpSum(
        _objective_coeff(task.id, executor, verifier, coeffs, q_t, k_c, k_v)
        * z[(task.id, executor, verifier)]
        for task in plan.tasks
        for executor, verifier in choices_by_task[task.id]
    )

    for task in plan.tasks:
        problem += (
            pulp.lpSum(z[(task.id, executor, verifier)] for executor, verifier in choices_by_task[task.id]) == 1,
            f"one_exec_verify_pair_{task.id}",
        )

    sources = set(plan.sources())
    for task in plan.tasks:
        duration = pulp.lpSum(
            float(coeffs.safe_latency[(task.id, executor)]) * z[(task.id, executor, verifier)]
            for executor, verifier in choices_by_task[task.id]
        )
        if task.id in sources:
            problem += F[task.id] >= duration, f"source_finish_{task.id}"
        for pred in task.predecessors:
            problem += F[task.id] >= F[pred] + duration, f"pred_{pred}_to_{task.id}"

    for sink in plan.sinks():
        problem += F[sink] <= mu_t, f"deadline_{sink}"

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status_name = pulp.LpStatus[status]
    if status_name != "Optimal":
        raise RuntimeError(f"JOVE allocation did not solve to optimality. Status: {status_name}")

    selected: Dict[str, Tuple[str, str]] = {}
    for task in plan.tasks:
        active_pairs = [
            (executor, verifier)
            for executor, verifier in choices_by_task[task.id]
            if pulp.value(z[(task.id, executor, verifier)]) > 0.5
        ]
        if len(active_pairs) != 1:
            raise RuntimeError(f"Expected one selected executor/verifier pair for {task.id}, got {active_pairs}")
        selected[task.id] = active_pairs[0]
    finish = _finish_times(plan, selected, coeffs)
    objective = float(pulp.value(problem.objective))
    return _selection_from_pairs(plan, selected, coeffs, finish, objective)


def _solve_by_enumeration(
    plan: Plan,
    api_candidates: List[str],
    coeffs: JoveInputs,
    mu_t: float,
    q_t: float,
    k_c: float,
    k_v: float,
    fixed_verifier_model: str,
    allow_self_verification: bool,
    enum_limit: int,
) -> JoveSelection:
    choices_by_task = [
        _choices_for_task(task, api_candidates, fixed_verifier_model, allow_self_verification)
        for task in plan.tasks
    ]
    total = math.prod(len(choices) for choices in choices_by_task)
    if total > enum_limit:
        raise RuntimeError(
            "PuLP is not installed and exhaustive fallback would enumerate "
            f"{total} assignments. Install pulp or reduce the smoke-test size."
        )
    best_pairs: Optional[Dict[str, Tuple[str, str]]] = None
    best_finish: Optional[Dict[str, float]] = None
    best_objective = -math.inf
    for combo in itertools.product(*choices_by_task):
        selected = {task.id: pair for task, pair in zip(plan.tasks, combo)}
        finish = _finish_times(plan, selected, coeffs)
        if not _is_deadline_feasible(plan, finish, mu_t):
            continue
        objective = sum(
            _objective_coeff(task.id, selected[task.id][0], selected[task.id][1], coeffs, q_t, k_c, k_v)
            for task in plan.tasks
        )
        if objective > best_objective:
            best_objective = objective
            best_pairs = selected
            best_finish = finish
    if best_pairs is None or best_finish is None:
        raise RuntimeError("No deadline-feasible JOVE assignment found.")
    return _selection_from_pairs(plan, best_pairs, coeffs, best_finish, best_objective)


def _selection_from_pairs(
    plan: Plan,
    selected: Mapping[str, Tuple[str, str]],
    coeffs: JoveInputs,
    finish: Mapping[str, float],
    objective: float,
) -> JoveSelection:
    executor_by_task = {task_id: executor for task_id, (executor, _verifier) in selected.items()}
    verifier_by_task = {
        task_id: (None if verifier == NO_VERIFIER else verifier) for task_id, (_executor, verifier) in selected.items()
    }
    expected_cost = sum(
        _selected_cost(task_id, selected[task_id][0], selected[task_id][1], coeffs) for task_id in selected
    )
    return JoveSelection(
        executor_by_task=executor_by_task,
        verifier_by_task=verifier_by_task,
        finish_time=dict(finish),
        objective_value=float(objective),
        expected_cost=float(expected_cost),
    )


def update_virtual_queue(q_t: float, realized_cost: float, gamma: float) -> float:
    return max(float(q_t) + float(realized_cost) - float(gamma), 0.0)
