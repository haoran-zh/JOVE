"""JOVE: joint executor and verification allocation for LLM task graphs."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from fractions import Fraction
import json
import math
import os
import pickle
import re
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from tools.api_config import (
    APIConfig,
    DEFAULT_VERIFIER_MODEL,
    apply_provider_defaults,
    build_api_configs,
    parse_api_candidates,
    parse_verifier_candidates,
)
from tools.client import (
    FAILURE_FATAL_CONFIG,
    FAILURE_RATE_LIMIT,
    FAILURE_UNSTABLE_SERVICE,
    ApiCallFailure,
    DEFAULT_API_KEYS_FILE,
    DEFAULT_NVIDIA_NIM_API_KEY_NAME,
    NVIDIA_NIM_BASE_URL,
    OPENROUTER_BASE_URL,
    PROVIDER_OPENROUTER,
    OpenAICompatibleClient,
    classify_api_failure,
)
from tools.cost_accounting import (
    DEFAULT_COST_USD_SCALE,
    ModelBudgetCostTracker,
    realized_prompt_costs_from_usages,
    queue_budget_with_verifier_factor,
)
from tools.datasets import (
    DATASET_CHOICES,
    expand_examples_by_node_count,
    graph_variant_metric_fields,
    load_online_examples,
    parse_graph_node_counts,
    shuffle_gpqa_choices,
)
from tools.evaluation import (
    ANSWER_TYPE_CHOICES,
    expected_livebench_answer_items,
    extract_choice_letter,
    normalize_livebench_solution_text,
    score_answer,
)
from tools.execution import (
    DEFAULT_API_FAILURE_OUTPUT,
    VerifierResult,
    build_task_prompt,
    build_verifier_prompt,
    execute_plan,
    run_verifiers,
)
from tools.feature_map import ROLE_EXEC, ROLE_VER, MiniLMFeatureMap
from tools.latency import split_latency_tolerance, validate_latency_tolerance
from tools.resource_estimation import (
    FeatureWeightedResourceModel,
    observe_executor_call,
    observe_verifier_call,
    run_self_checks as run_resource_estimation_self_checks,
)
from tools.optimizer import (
    JoveInputs,
    is_jove_infeasible_error,
    solve_jove_selection,
    uncertainty_reduction,
    update_virtual_queue,
)
from tools.planning import (
    Plan,
    PromptExample,
    TaskNode,
    answer_instruction_for_type,
    build_execution_context,
    build_verification_context,
    parse_plan_from_chat_result,
    planner_response_source,
    planner_response_to_text,
    planner_task_count_phrase,
    request_plan_with_fixed_api,
)
from tools.quality import ServiceLabel, WeightedLinearQualityModel

DEFAULT_LOG_DIR = "logs"
RESULTS_PKL_FILENAME = "metrics.pkl"
SUMMARY_JSON_FILENAME = "summary.json"


def _artifact_slug(value: object) -> str:
    text = str(value or "unknown").strip().lower()
    chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in text]
    slug = "".join(chars).strip("_")
    return slug or "unknown"


@dataclass
class TrialArtifacts:
    trial_id: str
    trial_dir: Path
    log_path: Optional[Path]

    @property
    def pickle_path(self) -> Path:
        return self.trial_dir / RESULTS_PKL_FILENAME

    @property
    def summary_path(self) -> Path:
        return self.trial_dir / SUMMARY_JSON_FILENAME


REASONING_PROFILE_DEFAULT = "default"
REASONING_PROFILE_HARD = "hard_reasoning"
REASONING_PROFILE_MMLU = "multi_choice_reasoning"
REASONING_PROFILE_LIVEBENCH = "livebench_reasoning"
HARD_REASONING_DATASETS = {"aime24"}
MMLU_REASONING_DATASETS = {"mmlu_pro", "gpqa"}
LIVEBENCH_REASONING_DATASETS = {"livebench_reasoning"}


@dataclass(frozen=True)
class ReasoningProfileConfig:
    name: str = REASONING_PROFILE_DEFAULT
    planner_max_tasks: Optional[int] = None
    executor_max_tokens: Optional[int] = None
    use_final_aggregation: bool = False
    final_aggregation_max_tokens: int = 512
    use_full_context: bool = False
    use_answer_normalization: bool = False
    use_calibrated_verifier: bool = False


@dataclass
class FinalAnswerCallResult:
    output: str
    model: str
    latency_seconds: float = 0.0
    usage: Optional[Dict[str, object]] = None
    raw_output: str = ""
    failed: bool = False
    failure_category: Optional[str] = None
    failure_message: Optional[str] = None
    status_code: Optional[int] = None
    retry_count: int = 0
    final_action: Optional[str] = None


@dataclass
class LLMJudgeResult:
    correct: Optional[bool]
    raw_output: str
    model: str
    latency_seconds: float = 0.0
    usage: Optional[Dict[str, object]] = None
    failed: bool = False
    failure_category: Optional[str] = None
    failure_message: Optional[str] = None
    status_code: Optional[int] = None
    retry_count: int = 0


@dataclass(frozen=True)
class AnswerNormalizationResult:
    formatted_answer: str = ""
    choice_letter: Optional[str] = None
    source: str = "disabled"
    failed: bool = False
    ambiguous: bool = False
    raw_candidate: str = ""
    reason: str = ""


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_values(metrics: List[Mapping[str, object]], key: str) -> List[float]:
    values: List[float] = []
    for item in metrics:
        value = _as_float(item.get(key))
        if value is not None:
            values.append(value)
    return values


def _mean_or_none(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _dataset_name_for_profile(args: argparse.Namespace, example: PromptExample) -> str:
    return str(example.dataset or getattr(args, "dataset", "") or "").strip().lower()


def _resolve_mode_bool(value: object, auto_default: bool) -> bool:
    mode = str(value or "auto").strip().lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    return bool(auto_default)


def resolve_reasoning_profile(args: argparse.Namespace, example: PromptExample) -> ReasoningProfileConfig:
    mode = str(getattr(args, "reasoning_profile", "auto") or "auto").strip().lower()
    dataset_name = _dataset_name_for_profile(args, example)
    answer_type = str(example.answer_type or "").strip().lower()

    profile_name = REASONING_PROFILE_DEFAULT
    if mode != "off":
        if dataset_name in LIVEBENCH_REASONING_DATASETS or (mode == "on" and answer_type == "livebench_solution"):
            profile_name = REASONING_PROFILE_LIVEBENCH
        elif dataset_name in HARD_REASONING_DATASETS or (mode == "on" and answer_type == "numeric"):
            profile_name = REASONING_PROFILE_HARD
        elif dataset_name in MMLU_REASONING_DATASETS or (mode == "on" and answer_type == "multiple_choice"):
            profile_name = REASONING_PROFILE_MMLU

    aggregation_mode = str(getattr(args, "enable_final_aggregation", "auto") or "auto").strip().lower()
    if aggregation_mode == "off":
        use_final_aggregation = False
    elif aggregation_mode == "on":
        use_final_aggregation = True
    else:
        use_final_aggregation = profile_name in {REASONING_PROFILE_HARD, REASONING_PROFILE_LIVEBENCH}

    use_full_context = _resolve_mode_bool(
        getattr(args, "reasoning_full_context", "auto"),
        profile_name in {REASONING_PROFILE_MMLU, REASONING_PROFILE_LIVEBENCH},
    )
    use_answer_normalization = _resolve_mode_bool(
        getattr(args, "enable_answer_normalization", "auto"),
        profile_name in {REASONING_PROFILE_MMLU, REASONING_PROFILE_LIVEBENCH},
    )
    use_calibrated_verifier = _resolve_mode_bool(
        getattr(args, "mmlu_calibrated_verifier", "auto"),
        profile_name in {REASONING_PROFILE_MMLU, REASONING_PROFILE_LIVEBENCH},
    )
    final_tokens = max(1, int(getattr(args, "final_aggregation_max_tokens", 512)))
    if profile_name == REASONING_PROFILE_HARD:
        return ReasoningProfileConfig(
            name=profile_name,
            planner_max_tasks=max(1, int(getattr(args, "hard_reasoning_max_tasks", 5))),
            executor_max_tokens=max(1, int(getattr(args, "hard_reasoning_executor_max_tokens", 1024))),
            use_final_aggregation=use_final_aggregation,
            final_aggregation_max_tokens=final_tokens,
            use_full_context=use_full_context,
            use_answer_normalization=use_answer_normalization,
            use_calibrated_verifier=use_calibrated_verifier,
        )
    if profile_name == REASONING_PROFILE_MMLU:
        return ReasoningProfileConfig(
            name=profile_name,
            planner_max_tasks=max(1, int(getattr(args, "mmlu_reasoning_max_tasks", 5))),
            executor_max_tokens=max(1, int(getattr(args, "mmlu_executor_max_tokens", 768))),
            use_final_aggregation=use_final_aggregation,
            final_aggregation_max_tokens=final_tokens,
            use_full_context=use_full_context,
            use_answer_normalization=use_answer_normalization,
            use_calibrated_verifier=use_calibrated_verifier,
        )
    if profile_name == REASONING_PROFILE_LIVEBENCH:
        return ReasoningProfileConfig(
            name=profile_name,
            planner_max_tasks=max(1, int(getattr(args, "livebench_reasoning_max_tasks", 3))),
            executor_max_tokens=max(1, int(getattr(args, "livebench_executor_max_tokens", 1024))),
            use_final_aggregation=use_final_aggregation,
            final_aggregation_max_tokens=final_tokens,
            use_full_context=use_full_context,
            use_answer_normalization=use_answer_normalization,
            use_calibrated_verifier=use_calibrated_verifier,
        )
    return ReasoningProfileConfig(
        name=REASONING_PROFILE_DEFAULT,
        planner_max_tasks=None,
        executor_max_tokens=None,
        use_final_aggregation=use_final_aggregation,
        final_aggregation_max_tokens=final_tokens,
        use_full_context=use_full_context,
        use_answer_normalization=use_answer_normalization,
        use_calibrated_verifier=use_calibrated_verifier,
    )


def _profile_dict(profile: ReasoningProfileConfig) -> Dict[str, object]:
    return {
        "dataset_profile": profile.name,
        "profile_planner_max_tasks": profile.planner_max_tasks,
        "profile_executor_max_tokens": profile.executor_max_tokens,
        "use_final_aggregation": profile.use_final_aggregation,
        "final_aggregation_max_tokens": profile.final_aggregation_max_tokens,
        "use_full_context": profile.use_full_context,
        "use_answer_normalization": profile.use_answer_normalization,
        "use_calibrated_verifier": profile.use_calibrated_verifier,
    }


def _args_snapshot(args: argparse.Namespace) -> Dict[str, object]:
    return dict(vars(args))


def _api_config_snapshot(api_configs: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    snapshot: Dict[str, Dict[str, object]] = {}
    for model, config in api_configs.items():
        if is_dataclass(config):
            data = dict(asdict(config))
        else:
            data = {
                "model": getattr(config, "model", model),
                "executor_cost_unit": getattr(config, "executor_cost_unit", None),
                "safe_latency_prior": getattr(config, "safe_latency_prior", None),
                "verifier_cost_factor": getattr(config, "verifier_cost_factor", None),
            }
        if hasattr(config, "verifier_cost_unit"):
            data["verifier_cost_unit"] = getattr(config, "verifier_cost_unit")
        snapshot[str(model)] = data
    return snapshot


def _example_snapshot(examples: List[PromptExample]) -> List[Dict[str, object]]:
    return [
        {
            "prompt_index": index,
            "question": example.question,
            "answer": example.answer,
            "dataset": example.dataset,
            "answer_type": example.answer_type,
            "answer_instruction": example.answer_instruction,
            "metadata": dict(example.metadata),
        }
        for index, example in enumerate(examples, start=1)
    ]


def build_metric_trajectories(metrics: List[Mapping[str, object]]) -> Dict[str, object]:
    per_prompt: List[Dict[str, object]] = []
    for item in metrics:
        per_prompt.append(
            {
                "prompt_index": item.get("prompt_index"),
                "skipped": bool(item.get("skipped", False)),
                "is_correct": item.get("is_correct"),
                "running_average_correctness": item.get("running_average_correctness"),
                "running_answered_prompts": item.get("running_answered_prompts"),
                "running_correct_prompts": item.get("running_correct_prompts"),
                "q_before": item.get("q_before"),
                "q_after": item.get("q_after"),
                "cost": item.get("cost"),
                "realized_total_latency": item.get("realized_total_latency"),
                "realized_sink_latency": item.get("realized_sink_latency"),
                "realized_executor_dag_latency": item.get("realized_executor_dag_latency"),
                "constraint_realized_latency": item.get("constraint_realized_latency"),
                "constraint_safe_latency": item.get("constraint_safe_latency"),
                "latency_tolerance_delta": item.get("latency_tolerance_delta"),
                "latency_node_tolerance": item.get("latency_node_tolerance"),
                "latency_quantile_level": item.get("latency_quantile_level"),
                "realized_verifier_stage_latency": item.get("realized_verifier_stage_latency"),
                "budget_feasible": item.get("budget_feasible"),
                "latency_feasible": item.get("latency_feasible"),
                "planned_call_count": item.get("planned_call_count"),
                "verifier_calls": item.get("verifier_calls"),
                "selected_baseline_model": item.get("selected_baseline_model"),
            }
        )
    return {
        "per_prompt": per_prompt,
        "running_average_correctness": [item.get("running_average_correctness") for item in per_prompt],
        "virtual_queue_after": [item.get("q_after") for item in per_prompt],
        "cost": [item.get("cost") for item in per_prompt],
        "realized_total_latency": [item.get("realized_total_latency") for item in per_prompt],
        "realized_sink_latency": [item.get("realized_sink_latency") for item in per_prompt],
    }


def build_run_summary(
    args: argparse.Namespace,
    metrics: List[Mapping[str, object]],
    q_t: float,
    run_status: str,
    quality_model_updates: int = 0,
    total_verifier_calls: Optional[int] = None,
    planner_skips: int = 0,
    budget_infeasible_skips: int = 0,
    latency_infeasible_skips: int = 0,
    executor_service_failures_by_model: Optional[Mapping[str, int]] = None,
    verifier_service_failures_by_model: Optional[Mapping[str, int]] = None,
    rate_limit_retries_by_model_stage: Optional[Mapping[str, int]] = None,
) -> Dict[str, object]:
    answered = [item for item in metrics if item.get("is_correct") is not None]
    completed = [item for item in metrics if not item.get("skipped")]
    correct_count = sum(1 for item in answered if bool(item.get("is_correct")))
    llm_judged = [item for item in metrics if item.get("llm_judge_correct") is not None]
    llm_judge_correct_count = sum(1 for item in llm_judged if bool(item.get("llm_judge_correct")))
    costs = _numeric_values(metrics, "cost")
    realized_usds = _numeric_values(metrics, "realized_usd")
    q_values = _numeric_values(metrics, "q_after")
    total_latencies = _numeric_values(metrics, "realized_total_latency")
    completed_total_latencies = _numeric_values(completed, "realized_total_latency")
    sink_latencies = _numeric_values(completed, "realized_sink_latency")
    executor_dag_latencies = _numeric_values(completed, "realized_executor_dag_latency")
    verifier_stage_latencies = _numeric_values(completed, "realized_verifier_stage_latency")
    safe_constraint_latencies = _numeric_values(completed, "constraint_safe_latency")
    realized_constraint_latencies = _numeric_values(completed, "constraint_realized_latency")
    latency_node_tolerances = _numeric_values(completed, "latency_node_tolerance")
    latency_quantile_levels = _numeric_values(completed, "latency_quantile_level")
    return {
        "run_status": run_status,
        "dataset": args.dataset,
        "selection_policy": "jove",
                "sample_size": getattr(args, "sample_size", None),
        "seed": getattr(args, "seed", None),
        "fixed_verifier_model": getattr(args, "verifier_model", None),
        "num_prompts": len(metrics),
        "num_completed_prompts": len(completed),
        "num_with_gold": len(answered),
        "correct_prompts": correct_count,
        "accuracy": (correct_count / len(answered)) if answered else None,
        "llm_judge_enabled": bool(getattr(args, "enable_llm_judge", False)),
        "llm_judge_model": getattr(args, "llm_judge_model", None),
        "llm_judge_num_evaluated": len(llm_judged),
        "llm_judge_correct_prompts": llm_judge_correct_count,
        "llm_judge_accuracy": (llm_judge_correct_count / len(llm_judged)) if llm_judged else None,
        "reasoning_profile_mode": getattr(args, "reasoning_profile", None),
        "final_aggregation_mode": getattr(args, "enable_final_aggregation", None),
        "running_average_correctness": metrics[-1].get("running_average_correctness") if metrics else None,
        "final_q_t": q_t,
        "max_q_t": max(q_values) if q_values else q_t,
        "gamma": args.gamma,
        "mu": args.mu,
        "latency_tolerance_delta": getattr(args, "latency_tolerance_delta", None),
        "mean_latency_node_tolerance": _mean_or_none(latency_node_tolerances),
        "mean_latency_quantile_level": _mean_or_none(latency_quantile_levels),
        "total_cost": sum(costs),
        "mean_cost_per_prompt": _mean_or_none(costs),
        "mean_cost_per_completed_prompt": _mean_or_none(_numeric_values(completed, "cost")),
        "total_realized_usd": sum(realized_usds) if realized_usds else None,
        "mean_realized_usd_per_prompt": _mean_or_none(realized_usds),
        "mean_realized_usd_per_completed_prompt": _mean_or_none(_numeric_values(completed, "realized_usd")),
        "cost_usd_scale": getattr(args, "cost_usd_scale", DEFAULT_COST_USD_SCALE),
        "k_c": getattr(args, "k_c", None),
        "k_v": getattr(args, "k_v", None),
                "total_realized_total_latency": sum(total_latencies),
        "mean_realized_total_latency": _mean_or_none(total_latencies),
        "mean_completed_realized_total_latency": _mean_or_none(completed_total_latencies),
        "mean_realized_sink_latency": _mean_or_none(sink_latencies),
        "mean_realized_executor_dag_latency": _mean_or_none(executor_dag_latencies),
        "mean_constraint_realized_latency": _mean_or_none(realized_constraint_latencies or executor_dag_latencies),
        "mean_constraint_safe_latency": _mean_or_none(safe_constraint_latencies),
        "mean_realized_verifier_stage_latency": _mean_or_none(verifier_stage_latencies),
        "quality_model_updates": quality_model_updates,
        "total_verifier_calls": (
            total_verifier_calls
            if total_verifier_calls is not None
            else sum(int(item.get("verifier_calls", 0) or 0) for item in metrics)
        ),
        "planner_skips": planner_skips,
        "budget_infeasible_skips": budget_infeasible_skips,
        "latency_infeasible_skips": latency_infeasible_skips,
        "executor_service_failures_by_model": dict(executor_service_failures_by_model or {}),
        "verifier_service_failures_by_model": dict(verifier_service_failures_by_model or {}),
        "rate_limit_retries_by_model_stage": dict(rate_limit_retries_by_model_stage or {}),
    }


def save_experiment_results(
    artifacts: Optional[TrialArtifacts],
    args: argparse.Namespace,
    examples: List[PromptExample],
    api_candidates: List[str],
    verifier_candidates: List[str],
    api_configs: Mapping[str, object],
    metrics: List[Dict[str, object]],
    summary: Mapping[str, object],
) -> None:
    if artifacts is None:
        return
    artifacts.trial_dir.mkdir(parents=True, exist_ok=True)
    enriched_summary = dict(summary)
    enriched_summary.update(
        {
            "trial_id": artifacts.trial_id,
            "trial_dir": str(artifacts.trial_dir),
            "log_path": str(artifacts.log_path) if artifacts.log_path else None,
            "pickle_path": str(artifacts.pickle_path),
            "summary_path": str(artifacts.summary_path),
        }
    )
    payload = {
        "trial_id": artifacts.trial_id,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "args": _args_snapshot(args),
        "api_candidates": list(api_candidates),
        "fixed_verifier_model": verifier_candidates[0] if verifier_candidates else None,
        "verifier_candidates": list(verifier_candidates),
        "api_configs": _api_config_snapshot(api_configs),
        "examples": _example_snapshot(examples),
        "prompt_metrics": metrics,
        "trajectories": build_metric_trajectories(metrics),
        "summary": enriched_summary,
    }

    tmp_pickle = artifacts.pickle_path.with_suffix(artifacts.pickle_path.suffix + ".tmp")
    with tmp_pickle.open("wb") as handle:
        pickle.dump(payload, handle)
    tmp_pickle.replace(artifacts.pickle_path)

    tmp_summary = artifacts.summary_path.with_suffix(artifacts.summary_path.suffix + ".tmp")
    with tmp_summary.open("w", encoding="utf-8") as handle:
        json.dump(enriched_summary, handle, indent=2, default=str)
    tmp_summary.replace(artifacts.summary_path)



class TeeStream:
    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)  # type: ignore[attr-defined]
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()  # type: ignore[attr-defined]

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@dataclass
class FeatureTables:
    exec_features: Dict[Tuple[str, str], object]
    ver_features: Dict[Tuple[str, str], object]
    exec_pred: Dict[Tuple[str, str], float]
    exec_ucb: Dict[Tuple[str, str], float]
    exec_uncertainty: Dict[Tuple[str, str], float]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bi-role verifier-aware online API allocation example.")
    parser.add_argument("--base-url", default=os.environ.get("API_RECRUITER_BASE_URL", NVIDIA_NIM_BASE_URL), help=f"Chat completions URL. NIM default, or OpenRouter: {OPENROUTER_BASE_URL}")
    parser.add_argument("--api-keys-file", default=os.environ.get("API_RECRUITER_API_KEYS_FILE", DEFAULT_API_KEYS_FILE), help="Path to a KEY=VALUE file (NVIDIA: API_key1=...; OpenRouter: OPENROUTER_API_KEY=...).")
    parser.add_argument("--api-key-name", default=os.environ.get("API_RECRUITER_API_KEY_NAME", os.environ.get("NVIDIA_NIM_API_KEY_NAME", DEFAULT_NVIDIA_NIM_API_KEY_NAME)), help="Named key from --api-keys-file. NIM: API_key1/API_key2. OpenRouter defaults to OPENROUTER_API_KEY when --base-url is OpenRouter.")
    parser.add_argument("--planner-model", default=os.environ.get("PLANNER_MODEL", "meta/llama-3.3-70b-instruct"))
    parser.add_argument("--planner-max-tokens", type=int, default=int(os.environ.get("PLANNER_MAX_TOKENS", "1536")), help="Maximum tokens for planner DAG JSON responses.")
    parser.add_argument("--planner-max-retries", type=int, default=int(os.environ.get("API_RECRUITER_PLANNER_MAX_RETRIES", "3")), help="Maximum same-model retries for planner unstable-service failures.")
    parser.add_argument("--planner-retry-delay", type=float, default=float(os.environ.get("API_RECRUITER_PLANNER_RETRY_DELAY", "5.0")), help="Seconds to wait before retrying a planner unstable-service failure; the shared request limiter still applies.")
    parser.add_argument("--api-candidates", default=os.environ.get("API_CANDIDATES"))
    parser.add_argument("--verifier-model", default=os.environ.get("API_RECRUITER_VERIFIER_MODEL", os.environ.get("VERIFIER_MODEL", DEFAULT_VERIFIER_MODEL)), help="Fixed verifier model used for every selected verifier call.")
    parser.add_argument("--verifier-max-retries", type=int, default=int(os.environ.get("API_RECRUITER_VERIFIER_MAX_RETRIES", "5")), help="Maximum same-model verifier-call retries after non-fatal verifier API failures.")
    parser.add_argument("--verifier-retry-delay", type=float, default=float(os.environ.get("API_RECRUITER_VERIFIER_RETRY_DELAY", "3.0")), help="Seconds to wait before retrying a failed verifier call; the shared request limiter still applies.")
    parser.add_argument("--verifier-candidates", default=os.environ.get("VERIFIER_CANDIDATES"), help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=os.environ.get("API_RECRUITER_DATASET", "bamboogle"), help="Online benchmark stream to process.")
    parser.add_argument("--dataset-split", default=os.environ.get("API_RECRUITER_DATASET_SPLIT"), help="Optional dataset source split override. The framework still processes it as one online stream.")
    parser.add_argument("--prompt", default=None, help="Run one explicit prompt instead of loading a dataset stream.")
    parser.add_argument("--gold-answer", default="", help="Optional gold answer for exact-match reporting with --prompt.")
    parser.add_argument("--answer-type", choices=ANSWER_TYPE_CHOICES, default="freeform", help="Answer scorer for --prompt runs.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=int(os.environ.get("ONLINE_SAMPLE_SIZE", os.environ.get("BAMBOOGLE_SAMPLE_SIZE", "0"))),
        help="After seeded shuffle, keep this many examples (e.g. 500 for MMLU-Pro). Use 0 or negative for the full shuffled stream.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("ONLINE_SAMPLE_SEED", os.environ.get("BAMBOOGLE_SAMPLE_SEED", "0"))),
        help="RNG seed for dataset shuffle (and subsample). Experiment reporting uses 0, 4, 5.",
    )
    parser.add_argument(
        "--graph-node-counts",
        default=os.environ.get("API_RECRUITER_GRAPH_NODE_COUNTS", "2,3,4,5"),
        help="Comma-separated planner node counts. Each query is expanded into one DAG per count, then shuffled. Empty disables.",
    )
    parser.add_argument("--mu", type=float, default=12.0, help="Prompt-level user-facing safe latency deadline.")
    parser.add_argument(
        "--latency-tolerance-delta",
        type=float,
        default=float(os.environ.get("API_RECRUITER_LATENCY_TOLERANCE_DELTA", "0.1")),
        help="Global per-prompt latency violation tolerance delta; split uniformly across task nodes.",
    )
    parser.add_argument("--gamma", type=float, default=150.0, help="Long-term per-prompt budget target (USD * --cost-usd-scale; default scale 1e6 maps gamma=300 to $0.0003).")
    parser.add_argument("--V", type=float, default=None, help=argparse.SUPPRESS)  # Deprecated; ignored.
    parser.add_argument("--k-c", type=float, default=float(os.environ.get("API_RECRUITER_K_C", "1e-5")), help="Virtual-queue cost coefficient k_c.")
    parser.add_argument("--k-v", type=float, default=float(os.environ.get("API_RECRUITER_K_V", "0.2")), help="Verifier D-optimal information-gain coefficient k_v (I(u)=0.5*log(1+u^2))")
    parser.add_argument(
        "--cost-usd-scale",
        type=float,
        default=float(os.environ.get("API_RECRUITER_COST_USD_SCALE", str(DEFAULT_COST_USD_SCALE))),
        help="Multiply OpenRouter usage.cost (USD) by this factor for budget/queue units. Default 1e6 so gamma=300 ≡ $0.0003/prompt.",
    )
    parser.add_argument("--beta", type=float, default=0.25, help="LinUCB optimism multiplier.")
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--prior-mean", type=float, default=0.5)
    parser.add_argument("--verifier-threshold", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--verifier-prior-scale", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--verifier-prior-half-life", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-parallel-tasks", type=int, default=int(os.environ.get("API_RECRUITER_MAX_PARALLEL_TASKS", "1")), help="Max ready DAG tasks (and verifiers) to run concurrently. Use >1 for normal/high-RPM parallel mode. Forced to 1 when --slow is set.")
    parser.add_argument("--request-interval", type=float, default=float(os.environ.get("API_RECRUITER_REQUEST_INTERVAL", "2.0")), help="Global minimum seconds between planner/executor/verifier HTTP request starts. Default 2.0s = 30 RPM. Use 0 with high-RPM paid APIs for true overlap.")
    parser.add_argument("--slow", action="store_true", help="Compatibility slow mode: enforce at least --slow-interval between request starts and force --max-parallel-tasks 1.")
    parser.add_argument("--slow-interval", type=float, default=float(os.environ.get("API_RECRUITER_SLOW_INTERVAL", "1.5")), help="Compatibility interval floor used when --slow is set.")
    parser.add_argument("--rate-limit-max-retries", type=int, default=int(os.environ.get("API_RECRUITER_RATE_LIMIT_MAX_RETRIES", "5")), help="Maximum retries for rate-limit failures on the same model/API.")
    parser.add_argument("--rate-limit-base-delay", type=float, default=float(os.environ.get("API_RECRUITER_RATE_LIMIT_BASE_DELAY", "2.0")), help="Base exponential-backoff delay for rate-limit retries when Retry-After is absent.")
    parser.add_argument("--rate-limit-max-delay", type=float, default=float(os.environ.get("API_RECRUITER_RATE_LIMIT_MAX_DELAY", "60.0")), help="Maximum delay for one rate-limit retry sleep.")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--run-label", default=os.environ.get("API_RECRUITER_RUN_LABEL", ""), help="Optional short label appended to the trial folder and log file names.")
    parser.add_argument("--no-log", action="store_true", help="Do not tee stdout/stderr to logs/.")
    parser.add_argument("--reasoning-profile", choices=("auto", "off", "on"), default=os.environ.get("API_RECRUITER_REASONING_PROFILE", "auto"), help="Dataset-aware reasoning profile for the isolated reasoning runner.")
    parser.add_argument("--hard-reasoning-max-tasks", type=int, default=int(os.environ.get("API_RECRUITER_HARD_REASONING_MAX_TASKS", "5")))
    parser.add_argument("--mmlu-reasoning-max-tasks", type=int, default=int(os.environ.get("API_RECRUITER_MMLU_REASONING_MAX_TASKS", "5")))
    parser.add_argument("--livebench-reasoning-max-tasks", type=int, default=int(os.environ.get("API_RECRUITER_LIVEBENCH_REASONING_MAX_TASKS", "3")))
    parser.add_argument("--hard-reasoning-executor-max-tokens", type=int, default=int(os.environ.get("API_RECRUITER_HARD_REASONING_EXECUTOR_MAX_TOKENS", "1024")))
    parser.add_argument("--mmlu-executor-max-tokens", type=int, default=int(os.environ.get("API_RECRUITER_MMLU_EXECUTOR_MAX_TOKENS", "768")))
    parser.add_argument("--livebench-executor-max-tokens", type=int, default=int(os.environ.get("API_RECRUITER_LIVEBENCH_EXECUTOR_MAX_TOKENS", "1024")))
    parser.add_argument("--final-aggregation-max-tokens", type=int, default=int(os.environ.get("API_RECRUITER_FINAL_AGGREGATION_MAX_TOKENS", "512")))
    parser.add_argument("--enable-final-aggregation", choices=("auto", "off", "on"), default=os.environ.get("API_RECRUITER_ENABLE_FINAL_AGGREGATION", "auto"))
    parser.add_argument("--reasoning-full-context", choices=("auto", "off", "on"), default=os.environ.get("API_RECRUITER_REASONING_FULL_CONTEXT", "auto"))
    parser.add_argument("--enable-answer-normalization", choices=("auto", "off", "on"), default=os.environ.get("API_RECRUITER_ENABLE_ANSWER_NORMALIZATION", "auto"))
    parser.add_argument("--mmlu-calibrated-verifier", choices=("auto", "off", "on"), default=os.environ.get("API_RECRUITER_MMLU_CALIBRATED_VERIFIER", "auto"))
    parser.add_argument("--enable-llm-judge", action="store_true", help="Run an optional NVIDIA NIM LLM judge for diagnostic evaluation only.")
    parser.add_argument("--llm-judge-model", default=os.environ.get("API_RECRUITER_LLM_JUDGE_MODEL"), help="NVIDIA NIM model for optional diagnostic LLM judging. Defaults to --verifier-model.")
    parser.add_argument("--llm-judge-max-tokens", type=int, default=int(os.environ.get("API_RECRUITER_LLM_JUDGE_MAX_TOKENS", "16")))
    parser.add_argument("--smoke-test", action="store_true", help="Run a small non-network optimizer smoke test and exit.")
    return parser


def _expand_graph_examples(args: argparse.Namespace, examples: List[PromptExample]) -> List[PromptExample]:
    if args.prompt:
        return examples
    node_counts = parse_graph_node_counts(getattr(args, "graph_node_counts", ""))
    if not node_counts:
        return examples
    return expand_examples_by_node_count(examples, node_counts=node_counts, seed=args.seed)


def build_examples(args: argparse.Namespace) -> List[PromptExample]:
    if args.prompt:
        return [
            PromptExample(
                question=args.prompt,
                answer=args.gold_answer,
                dataset="custom",
                answer_type=args.answer_type,
                answer_instruction=answer_instruction_for_type(args.answer_type),
            )
        ]
    examples = load_online_examples(
        dataset_name=args.dataset,
        sample_size=args.sample_size,
        seed=args.seed,
        split=args.dataset_split,
    )
    return _expand_graph_examples(args, examples)


def planner_max_tasks_for_example(
    example: PromptExample,
    default_max_tasks: int = 5,
    profile_config: Optional[ReasoningProfileConfig] = None,
) -> int:
    if isinstance(example.metadata, dict) and example.metadata.get("planner_node_count") is not None:
        try:
            return min(max(int(example.metadata["planner_node_count"]), 1), 5)
        except (TypeError, ValueError):
            pass
    if profile_config is not None and profile_config.planner_max_tasks is not None:
        return min(max(int(profile_config.planner_max_tasks), 1), 5)
    value = None
    if isinstance(example.metadata, dict):
        value = example.metadata.get("planner_max_tasks")
    try:
        max_tasks = int(value) if value is not None else int(default_max_tasks)
    except (TypeError, ValueError):
        max_tasks = int(default_max_tasks)
    return min(max(max_tasks, 1), 5)


def planner_min_tasks_for_example(
    example: PromptExample,
    max_tasks: int,
    profile_config: Optional[ReasoningProfileConfig] = None,
) -> int:
    task_limit = max(1, int(max_tasks))
    if isinstance(example.metadata, dict) and example.metadata.get("planner_node_count") is not None:
        try:
            return min(max(int(example.metadata["planner_node_count"]), 1), task_limit)
        except (TypeError, ValueError):
            pass
    if profile_config is not None:
        if profile_config.name == REASONING_PROFILE_MMLU:
            return 3 if task_limit >= 3 else task_limit
        if profile_config.name == REASONING_PROFILE_LIVEBENCH:
            return 2 if task_limit >= 2 else 1
        if profile_config.name == REASONING_PROFILE_HARD:
            return 3 if task_limit >= 3 else 1
    return 1 if task_limit <= 1 else 2


def _graph_metric_fields(example: PromptExample, planner_min_tasks: int, planner_max_tasks: int) -> Dict[str, object]:
    return {
        **graph_variant_metric_fields(example),
        "planner_min_tasks": int(planner_min_tasks),
        "planner_max_tasks": int(planner_max_tasks),
    }


def build_profiled_planner_system_prompt(
    profile_config: ReasoningProfileConfig,
    max_tasks: int,
    min_tasks: Optional[int] = None,
) -> str:
    task_limit = max(1, int(max_tasks))
    if min_tasks is None:
        min_tasks = planner_min_tasks_for_example(
            PromptExample(question="", metadata={}),
            task_limit,
            profile_config=profile_config,
        )
    else:
        min_tasks = min(max(int(min_tasks), 1), task_limit)
    count_phrase = planner_task_count_phrase(min_tasks, task_limit)
    if profile_config.name == REASONING_PROFILE_MMLU:
        profile_instruction = (
            "This is a multiple-choice reasoning problem. Create substantive executable tasks for solving intermediate "
            "reasoning subproblems, comparing plausible choices, and a final option-selection sink task. Do not create "
            "read-only, restatement, question-parsing, option-listing, formatting, tagging, XML output, or answer-extraction "
            "tasks. Every non-sink task must compute a useful fact, formula, constraint, intermediate value, or option "
            "comparison needed for the answer. The sink task should synthesize the reasoning and identify the selected "
            "option letter with concise support; final <answer>LETTER</answer> formatting is handled outside the DAG."
        )
    elif profile_config.name == REASONING_PROFILE_LIVEBENCH:
        profile_instruction = (
            f"This is a LiveBench-Reasoning problem. Create {count_phrase} substantive executable tasks and exactly one final "
            "answer-generation sink. For zebra_puzzle, formalize the constraints, solve the assignment, then answer all "
            "requested questions in order. For web_of_lies_v2, extract truth/lie statements, propagate truth values, then "
            "answer the requested yes/no questions in order. For spatial, solve the geometry/counting question and output "
            "the final integer. Do not create read-only, formatting, tagging, answer-extraction, verifier, checker, repair, "
            "or validation tasks. The sink must produce only the ordered final answer requested by the dataset."
        )
    else:
        profile_instruction = (
            "This is a hard reasoning problem. Create executable tasks for understanding the problem, carrying out "
            "the main reasoning steps, and a final answer-generation sink task. The sink task must aggregate earlier "
            "outputs into the requested dataset-specific final format."
        )
    task_groups_list = ', '.join(f'"{name}"' for name in (
        "Anchor Identification",
        "Fact Retrieval",
        "Reasoning Over Intermediate Results",
        "General Analysis",
    ))
    return (
        f"You are a planner for benchmark questions. Decompose the user prompt into {count_phrase} "
        "small executable tasks arranged as a DAG. Return JSON only; no markdown or extra text. "
        "Do not solve the question, compute the final answer, or include reasoning in the plan. "
        f"{profile_instruction} "
        "Do not create verifier, checker, validator, auditor, repair, or formatting-only tasks. Verification and final-format cleanup are handled outside the DAG. "
        "Do not include verifier_id, verifier model names, executor APIs, or API/model selection metadata. "
        "Each task needs lowercase fields id, description, predecessors, type, output_format, input_template. "
        f"type must be one of: {task_groups_list}. "
        "Use {t1}, {t2}, ... only for predecessor outputs in input_template. "
        "The last topological task must be the final-answer sink and must depend on all reasoning outputs needed for the answer. "
        "Keep dependencies minimal and acyclic. "
        "Schema: {\"tasks\":[{\"id\":\"t1\",\"description\":\"...\",\"predecessors\":[],"
        "\"type\":\"General Analysis\",\"output_format\":\"concise result\",\"input_template\":\"...\"}]}."
    )


def request_profiled_plan_with_fixed_api(
    client: OpenAICompatibleClient,
    example: PromptExample,
    planner_model: str,
    max_tokens: int,
    max_tasks: int,
    profile_config: ReasoningProfileConfig,
    min_tasks: Optional[int] = None,
) -> object:
    final_instruction = str(example.answer_instruction or answer_instruction_for_type(example.answer_type)).strip()
    metadata_lines = []
    if str(example.dataset or "").strip().lower() == "livebench_reasoning":
        metadata_lines.append("LiveBench task: " + str(example.metadata.get("task", "")))
        expected_items = expected_livebench_answer_items(example)
        if expected_items is not None:
            metadata_lines.append(f"Expected final answer items: {expected_items}")
    metadata_text = ("\n" + "\n".join(metadata_lines)) if metadata_lines else ""
    user_prompt = (
        f"Dataset: {example.dataset}{metadata_text}\n"
        f"Final answer instruction: {final_instruction}\n\n"
        f"Question:\n{example.question}"
    )
    messages = [
        {
            "role": "system",
            "content": build_profiled_planner_system_prompt(profile_config, max_tasks, min_tasks=min_tasks),
        },
        {"role": "user", "content": user_prompt},
    ]
    return client.chat(
        model=planner_model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        call_label="planner",
    )


def build_final_aggregation_prompt(example: PromptExample, plan: Plan, outputs: Mapping[str, str]) -> str:
    final_instruction = str(example.answer_instruction or answer_instruction_for_type(example.answer_type)).strip()
    lines = [
        "Original problem:",
        example.question.strip(),
    ]
    options = example.metadata.get("options") if isinstance(example.metadata, dict) else None
    if isinstance(options, list) and options:
        lines.append("\nAnswer choices:")
        for index, option in enumerate(options):
            lines.append(f"{chr(ord('A') + index)}. {option}")
    lines.append("\nTask outputs:")
    tasks_by_id = plan.task_by_id()
    for task_id in plan.topological_order():
        task = tasks_by_id[task_id]
        lines.append(f"- {task_id} ({task.description}): {outputs.get(task_id, '').strip()}")
    lines.extend([
        "\nUse the task outputs to produce the final answer only.",
        final_instruction,
    ])
    if example.answer_type == "multiple_choice":
        lines.append("Return exactly one option letter inside the requested final-answer tag.")
    elif example.answer_type == "numeric":
        lines.append("Return the final integer only inside the requested final-answer tag.")
    elif example.answer_type == "livebench_solution":
        task_name = str(example.metadata.get("task", "") if isinstance(example.metadata, dict) else "").strip()
        expected_items = expected_livebench_answer_items(example)
        lines.append(f"LiveBench task: {task_name or 'unknown'}.")
        if expected_items is not None:
            lines.append(f"Return exactly {expected_items} comma-separated answer item(s), in the same order as the questions.")
        if task_name == "spatial":
            lines.append("Return one integer inside <solution>...</solution>.")
        elif task_name == "web_of_lies_v2":
            lines.append("Return only yes/no words inside <solution>...</solution>.")
        else:
            lines.append("Return only the ordered comma-separated final answer inside <solution>...</solution>.")
    return "\n".join(lines)


def run_final_aggregation(
    client: OpenAICompatibleClient,
    example: PromptExample,
    plan: Plan,
    outputs: Mapping[str, str],
    model: str,
    max_tokens: int,
) -> FinalAnswerCallResult:
    messages = [
        {"role": "system", "content": "Aggregate task outputs into the exact requested final-answer format. No explanation."},
        {"role": "user", "content": build_final_aggregation_prompt(example, plan, outputs)},
    ]
    try:
        response = client.chat(model, messages, temperature=0.0, max_tokens=max_tokens, call_label="final_aggregation")
    except ApiCallFailure as exc:
        if exc.category == FAILURE_FATAL_CONFIG:
            raise
        return FinalAnswerCallResult(
            output="",
            model=model,
            latency_seconds=exc.latency_seconds,
            usage={},
            raw_output="",
            failed=True,
            failure_category=exc.category,
            failure_message=exc.message,
            status_code=exc.status_code,
            retry_count=exc.retry_count,
            final_action="sink_output_fallback",
        )
    output = (response.content or "").strip()
    return FinalAnswerCallResult(
        output=output,
        model=model,
        latency_seconds=response.latency_seconds,
        usage=response.usage,
        raw_output=output,
        status_code=response.status_code,
        retry_count=response.retry_count,
        failed=not bool(output),
        final_action=None if output else "sink_output_fallback",
    )


def _valid_choice_letters_for_example(example: PromptExample) -> str:
    options = example.metadata.get("options") if isinstance(example.metadata, dict) else None
    count = len(options) if isinstance(options, list) and options else 10
    return "".join(chr(ord("A") + idx) for idx in range(min(max(count, 1), 26)))


def _normalize_option_text(text: str) -> str:
    lowered = str(text or "").lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(normalized.split())


def _find_choice_letter_in_text(text: str, valid_letters: str) -> Tuple[Optional[str], str, bool]:
    if not str(text or "").strip():
        return None, "empty", False
    direct = extract_choice_letter(text, valid_letters)
    if direct:
        return direct, "choice_letter", False

    letters = re.escape(valid_letters)
    patterns = [
        (rf"<ANSWER>\s*([{letters}])\s*</ANSWER>", "answer_tag"),
        (rf">\s*([{letters}])\s*</[^>]+>", "malformed_tag_letter"),
        (rf"(?:FINAL\s+)?(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*[\(\[]?([{letters}])[\)\].:]?", "answer_phrase"),
        (rf"(?:SELECT|SELECTED|CHOOSE|CHOSE|PICK|PICKED)\s+(?:THE\s+)?(?:OPTION|CHOICE)?\s*[\(\[]?([{letters}])[\)\].:]?", "selection_phrase"),
        (rf"\b([{letters}])\s+(?:IS|SEEMS|APPEARS)\s+(?:THE\s+)?(?:CORRECT|BEST|ANSWER|CHOICE|OPTION)\b", "letter_is_best"),
        (rf"(?:THEREFORE|THUS|SO),?\s+(?:THE\s+)?(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*[\(\[]?([{letters}])[\)\].:]?", "conclusion_phrase"),
        (rf"^\s*[\(\[]?([{letters}])[\)\].:]?\s*$", "bare_letter"),
    ]
    hits: List[Tuple[str, str]] = []
    upper = str(text or "").upper()
    for pattern, source in patterns:
        for match in re.finditer(pattern, upper, flags=re.IGNORECASE | re.DOTALL):
            hits.append((match.group(1).upper(), source))
    distinct = sorted({letter for letter, _source in hits})
    if len(distinct) == 1:
        source = next(source for letter, source in hits if letter == distinct[0])
        return distinct[0], source, False
    if len(distinct) > 1:
        return None, "ambiguous_choice_letters", True
    return None, "no_choice_letter", False


def _find_choice_by_option_text(text: str, example: PromptExample) -> Tuple[Optional[str], str, bool]:
    options = example.metadata.get("options") if isinstance(example.metadata, dict) else None
    if not isinstance(options, list) or not options:
        return None, "no_options", False
    text_norm = _normalize_option_text(text)
    if not text_norm:
        return None, "empty", False
    hits: List[str] = []
    for index, option in enumerate(options):
        option_norm = _normalize_option_text(str(option))
        if len(option_norm) < 2:
            continue
        pattern = rf"(?<![a-z0-9]){re.escape(option_norm)}(?![a-z0-9])"
        if re.search(pattern, text_norm):
            hits.append(chr(ord("A") + index))
    distinct = sorted(set(hits))
    if len(distinct) == 1:
        return distinct[0], "option_text_match", False
    if len(distinct) > 1:
        return None, "ambiguous_option_text", True
    return None, "no_option_text_match", False


def _fraction_from_decimal_text(value: str) -> Optional[Fraction]:
    cleaned = str(value or "").strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def _numeric_option_value(option_text: str) -> Optional[Fraction]:
    text = str(option_text or "").strip().lower()
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None

    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(?:in|out of|of)\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        numerator = _fraction_from_decimal_text(match.group(1))
        denominator = _fraction_from_decimal_text(match.group(2))
        if numerator is not None and denominator not in {None, Fraction(0, 1)}:
            return numerator / denominator

    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        numerator = _fraction_from_decimal_text(match.group(1))
        denominator = _fraction_from_decimal_text(match.group(2))
        if numerator is not None and denominator not in {None, Fraction(0, 1)}:
            return numerator / denominator

    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if match:
        value = _fraction_from_decimal_text(match.group(1))
        if value is not None:
            return value / 100

    match = re.fullmatch(r"-?\d+(?:\.\d+)?", text)
    if match:
        return _fraction_from_decimal_text(text)
    return None


def _canonicalize_equivalent_choice_letter(letter: str, example: PromptExample) -> Tuple[str, str]:
    options = example.metadata.get("options") if isinstance(example.metadata, dict) else None
    if not isinstance(options, list) or not options:
        return letter, ""
    index = ord(str(letter).upper()) - ord("A")
    if index < 0 or index >= len(options):
        return letter, ""
    selected_value = _numeric_option_value(str(options[index]))
    if selected_value is None:
        return letter, ""
    for option_index, option in enumerate(options):
        if _numeric_option_value(str(option)) == selected_value:
            canonical = chr(ord("A") + option_index)
            if canonical != letter:
                return canonical, f"equivalent_numeric_option:{letter}_to_{canonical}"
            return letter, ""
    return letter, ""


def _numeric_values_in_text(text: str) -> List[Fraction]:
    body = str(text or "").lower().replace(",", "")
    values: List[Tuple[int, int, Fraction]] = []
    occupied: List[Tuple[int, int]] = []

    def add_value(start: int, end: int, value: Optional[Fraction]) -> None:
        if value is None:
            return
        values.append((start, end, value))
        occupied.append((start, end))

    for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(?:in|out of|of)\s*(-?\d+(?:\.\d+)?)", body):
        numerator = _fraction_from_decimal_text(match.group(1))
        denominator = _fraction_from_decimal_text(match.group(2))
        if numerator is not None and denominator not in {None, Fraction(0, 1)}:
            add_value(match.start(), match.end(), numerator / denominator)

    for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", body):
        if any(match.start() < end and start < match.end() for start, end in occupied):
            continue
        numerator = _fraction_from_decimal_text(match.group(1))
        denominator = _fraction_from_decimal_text(match.group(2))
        if numerator is not None and denominator not in {None, Fraction(0, 1)}:
            add_value(match.start(), match.end(), numerator / denominator)

    for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*%", body):
        if any(match.start() < end and start < match.end() for start, end in occupied):
            continue
        value = _fraction_from_decimal_text(match.group(1))
        if value is not None:
            add_value(match.start(), match.end(), value / 100)

    for match in re.finditer(r"(?<![a-z0-9./-])-?\d+(?:\.\d+)?(?![a-z0-9./-])", body):
        if any(match.start() < end and start < match.end() for start, end in occupied):
            continue
        add_value(match.start(), match.end(), _fraction_from_decimal_text(match.group(0)))

    return [value for _start, _end, value in values]


def _find_choice_by_numeric_value_text(text: str, example: PromptExample) -> Tuple[Optional[str], str, bool]:
    options = example.metadata.get("options") if isinstance(example.metadata, dict) else None
    if not isinstance(options, list) or not options:
        return None, "no_options", False
    value_to_letter: Dict[Fraction, str] = {}
    for index, option in enumerate(options):
        value = _numeric_option_value(str(option))
        if value is not None and value not in value_to_letter:
            value_to_letter[value] = chr(ord("A") + index)
    if not value_to_letter:
        return None, "no_numeric_options", False
    hits = [value_to_letter[value] for value in _numeric_values_in_text(text) if value in value_to_letter]
    distinct = sorted(set(hits))
    if len(distinct) == 1:
        return distinct[0], "numeric_value_match", False
    if len(distinct) > 1:
        return None, "ambiguous_numeric_value", True
    return None, "no_numeric_value_match", False


def build_mmlu_executor_guidance(example: PromptExample) -> str:
    if str(example.answer_type or "").strip().lower() != "multiple_choice":
        return ""
    return (
        "Multiple-choice guidance: use the full original question and all answer choices. "
        "Resolved dependency values are hints, not authority; if a predecessor output conflicts with the original problem, options, or basic domain facts, correct it. "
        "For intermediate tasks, solve the requested subproblem and keep relevant option letters/text; do not guess a final option unless the task asks for final selection. "
        "For source tasks, compute the requested substantive fact or intermediate value directly from the original prompt rather than merely restating or selecting an option. "
        "For comparison or final-selection tasks, compare against every listed option and return the exact option letter with its option text. "
        "If two or more options are semantically or numerically equivalent, choose the earliest equivalent option letter in the listed order."
    )


def normalize_mmlu_final_answer(
    example: PromptExample,
    current_final_answer: str,
    sink_final_answer: str,
    plan: Plan,
    outputs: Mapping[str, str],
) -> AnswerNormalizationResult:
    if str(example.answer_type or "").strip().lower() != "multiple_choice":
        return AnswerNormalizationResult(source="disabled", failed=True, reason="not_multiple_choice")

    valid_letters = _valid_choice_letters_for_example(example)
    candidates: List[Tuple[str, str]] = [
        ("current_final_answer", current_final_answer),
        ("sink_task", sink_final_answer),
    ]
    for task_id in reversed(plan.topological_order()):
        candidates.append((f"task_output:{task_id}", outputs.get(task_id, "")))

    seen_texts = set()
    unique_candidates: List[Tuple[str, str]] = []
    for source, text in candidates:
        cleaned = str(text or "").strip()
        if not cleaned or cleaned in seen_texts:
            continue
        seen_texts.add(cleaned)
        unique_candidates.append((source, cleaned))

    ambiguous_sources: List[str] = []
    for source, text in unique_candidates:
        letter, numeric_source, ambiguous = _find_choice_by_numeric_value_text(text, example)
        if letter:
            return AnswerNormalizationResult(
                formatted_answer=f"<answer>{letter}</answer>",
                choice_letter=letter,
                source=f"{source}:{numeric_source}",
                raw_candidate=text,
            )
        if ambiguous:
            ambiguous_sources.append(f"{source}:{numeric_source}")

        letter, letter_source, ambiguous = _find_choice_letter_in_text(text, valid_letters)
        if letter:
            canonical_letter, canonical_source = _canonicalize_equivalent_choice_letter(letter, example)
            source_detail = f"{source}:{letter_source}"
            if canonical_source:
                source_detail = f"{source_detail}:{canonical_source}"
            return AnswerNormalizationResult(
                formatted_answer=f"<answer>{canonical_letter}</answer>",
                choice_letter=canonical_letter,
                source=source_detail,
                raw_candidate=text,
            )
        if ambiguous:
            ambiguous_sources.append(f"{source}:{letter_source}")

        letter, option_source, ambiguous = _find_choice_by_option_text(text, example)
        if letter:
            canonical_letter, canonical_source = _canonicalize_equivalent_choice_letter(letter, example)
            source_detail = f"{source}:{option_source}"
            if canonical_source:
                source_detail = f"{source_detail}:{canonical_source}"
            return AnswerNormalizationResult(
                formatted_answer=f"<answer>{canonical_letter}</answer>",
                choice_letter=canonical_letter,
                source=source_detail,
                raw_candidate=text,
            )
        if ambiguous:
            ambiguous_sources.append(f"{source}:{option_source}")

    if ambiguous_sources:
        return AnswerNormalizationResult(
            source=",".join(ambiguous_sources[:3]),
            failed=True,
            ambiguous=True,
            reason="ambiguous_choice",
        )
    return AnswerNormalizationResult(source="no_match", failed=True, reason="no_unique_choice")


def _livebench_task_name(example: PromptExample) -> str:
    if not isinstance(example.metadata, dict):
        return ""
    return str(example.metadata.get("task") or "").strip().lower()


def build_livebench_executor_guidance(example: PromptExample) -> str:
    if str(example.answer_type or "").strip().lower() != "livebench_solution":
        return ""
    task_name = _livebench_task_name(example)
    expected_items = expected_livebench_answer_items(example)
    count_text = f" exactly {expected_items}" if expected_items is not None else " the requested number of"
    base = (
        "LiveBench guidance: use the full original problem. Resolved dependency values are hints, not authority; "
        "if they conflict with the original constraints, correct them. The final sink must output only"
        f"{count_text} answer item(s) in the same order as the questions. "
    )
    if task_name == "zebra_puzzle":
        return base + "For zebra puzzles, maintain a one-to-one assignment table: each person has exactly one value per attribute and each value is used once. Answer each requested attribute/position query in order."
    if task_name == "web_of_lies_v2":
        return base + "For web-of-lies, propagate truth values from explicit truth/lie statements and answer each requested person/location with yes or no."
    if task_name == "spatial":
        return base + "For spatial problems, reason about the geometry/counting setup. If the question asks how many pieces or objects result, count all resulting requested pieces, not just the number of source objects affected. If a plane through a sphere center cuts a solid sphere into two equal halves, count two hemisphere pieces from that sphere. Return one integer for the final answer."
    return base


def build_profile_executor_guidance(example: PromptExample, profile_config: ReasoningProfileConfig) -> str:
    if profile_config.name == REASONING_PROFILE_MMLU and profile_config.use_answer_normalization:
        return build_mmlu_executor_guidance(example)
    if profile_config.name == REASONING_PROFILE_LIVEBENCH:
        return build_livebench_executor_guidance(example)
    return ""


def normalize_livebench_final_answer(
    example: PromptExample,
    current_final_answer: str,
    sink_final_answer: str,
    plan: Plan,
    outputs: Mapping[str, str],
) -> AnswerNormalizationResult:
    if str(example.answer_type or "").strip().lower() != "livebench_solution":
        return AnswerNormalizationResult(source="disabled", failed=True, reason="not_livebench_solution")

    expected_items = expected_livebench_answer_items(example)
    candidates: List[Tuple[str, str]] = [
        ("current_final_answer", current_final_answer),
        ("sink_task", sink_final_answer),
    ]
    for task_id in reversed(plan.topological_order()):
        candidates.append((f"task_output:{task_id}", outputs.get(task_id, "")))

    seen_texts = set()
    for source, text in candidates:
        cleaned = str(text or "").strip()
        if not cleaned or cleaned in seen_texts:
            continue
        seen_texts.add(cleaned)
        normalized, extract_source, failed = normalize_livebench_solution_text(cleaned, expected_items=expected_items)
        if not failed and normalized:
            return AnswerNormalizationResult(
                formatted_answer=f"<solution>{normalized}</solution>",
                source=f"{source}:{extract_source}",
                raw_candidate=cleaned,
            )

    return AnswerNormalizationResult(source="no_match", failed=True, reason="no_valid_livebench_answer")


def normalize_livebench_sink_answer(
    example: PromptExample,
    sink_final_answer: str,
) -> Optional[AnswerNormalizationResult]:
    if str(example.answer_type or "").strip().lower() != "livebench_solution":
        return None
    cleaned = str(sink_final_answer or "").strip()
    if not cleaned:
        return None
    expected_items = expected_livebench_answer_items(example)
    normalized, extract_source, failed = normalize_livebench_solution_text(cleaned, expected_items=expected_items)
    if failed or not normalized:
        return None
    return AnswerNormalizationResult(
        formatted_answer=f"<solution>{normalized}</solution>",
        source=f"sink_task:{extract_source}",
        raw_candidate=cleaned,
    )


def normalize_profile_final_answer(
    profile_config: ReasoningProfileConfig,
    example: PromptExample,
    current_final_answer: str,
    sink_final_answer: str,
    plan: Plan,
    outputs: Mapping[str, str],
) -> AnswerNormalizationResult:
    if profile_config.name == REASONING_PROFILE_MMLU:
        return normalize_mmlu_final_answer(
            example=example,
            current_final_answer=current_final_answer,
            sink_final_answer=sink_final_answer,
            plan=plan,
            outputs=outputs,
        )
    if profile_config.name == REASONING_PROFILE_LIVEBENCH:
        return normalize_livebench_final_answer(
            example=example,
            current_final_answer=current_final_answer,
            sink_final_answer=sink_final_answer,
            plan=plan,
            outputs=outputs,
        )
    return AnswerNormalizationResult(source="disabled", failed=True, reason="unsupported_profile")


def _parse_llm_judge_output(raw_output: str) -> Optional[bool]:
    lowered = str(raw_output or "").strip().lower()
    has_true = "true" in lowered
    has_false = "false" in lowered
    if has_true and not has_false:
        return True
    if has_false and not has_true:
        return False
    return None


def run_optional_llm_judge(
    args: argparse.Namespace,
    client: OpenAICompatibleClient,
    example: PromptExample,
    predicted_answer: str,
) -> Optional[LLMJudgeResult]:
    if not bool(getattr(args, "enable_llm_judge", False)) or not example.answer:
        return None
    base_url = str(getattr(client, "base_url", "") or "").lower()
    if "integrate.api.nvidia.com" not in base_url and "build.nvidia.com" not in base_url:
        raise ValueError("--enable-llm-judge is restricted to NVIDIA NIM-compatible base URLs.")
    model = str(getattr(args, "llm_judge_model", None) or getattr(args, "verifier_model", ""))
    max_tokens = max(1, int(getattr(args, "llm_judge_max_tokens", 16)))
    livebench_instruction = ""
    if str(example.answer_type or "").strip().lower() == "livebench_solution":
        expected_items = expected_livebench_answer_items(example)
        count_text = f" exactly {expected_items}" if expected_items is not None else " the same number of"
        livebench_instruction = (
            "For LiveBench-Reasoning, compare the ordered final answer list only. "
            f"Require{count_text} answer item(s), ignore tags, bold markup, and extra reasoning, "
            "and normalize case, punctuation, hyphens, and articles. "
        )
    prompt = (
        "Here is a benchmark problem with a standard answer and a predicted answer. "
        "Judge whether the predicted final answer is equivalent to the standard answer. "
        "For numeric answers, the same numerical value is correct. For multiple-choice answers, the same option letter is correct. "
        f"{livebench_instruction}"
        "Ignore extra reasoning or formatting if the final answer is equivalent. Output only True or False.\n\n"
        f"Problem:\n{example.question}\n\n"
        f"Standard answer:\n{example.answer}\n\n"
        f"Predicted answer:\n{predicted_answer}"
    )
    messages = [
        {"role": "system", "content": "You are a strict answer-equivalence judge. Output only True or False."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat(model, messages, temperature=0.0, max_tokens=max_tokens, call_label="llm_judge")
    except ApiCallFailure as exc:
        if exc.category == FAILURE_FATAL_CONFIG:
            raise
        return LLMJudgeResult(
            correct=None,
            raw_output="",
            model=model,
            latency_seconds=exc.latency_seconds,
            usage={},
            failed=True,
            failure_category=exc.category,
            failure_message=exc.message,
            status_code=exc.status_code,
            retry_count=exc.retry_count,
        )
    raw_output = (response.content or "").strip()
    return LLMJudgeResult(
        correct=_parse_llm_judge_output(raw_output),
        raw_output=raw_output,
        model=model,
        latency_seconds=response.latency_seconds,
        usage=response.usage,
        failed=False,
        status_code=response.status_code,
        retry_count=response.retry_count,
    )


def request_plan_with_retries(
    args: argparse.Namespace,
    client: OpenAICompatibleClient,
    example: PromptExample,
    max_tasks: int,
    profile_config: Optional[ReasoningProfileConfig] = None,
    min_tasks: Optional[int] = None,
) -> Tuple[object, int]:
    max_retries = max(int(args.planner_max_retries), 0)
    retry_delay = max(float(args.planner_retry_delay), 0.0)
    service_retries = 0
    for attempt in range(max_retries + 1):
        try:
            if profile_config is not None and profile_config.name != REASONING_PROFILE_DEFAULT:
                result = request_profiled_plan_with_fixed_api(
                    client=client,
                    example=example,
                    planner_model=args.planner_model,
                    max_tokens=args.planner_max_tokens,
                    max_tasks=max_tasks,
                    profile_config=profile_config,
                    min_tasks=min_tasks,
                )
            else:
                result = request_plan_with_fixed_api(
                    client,
                    example.question,
                    args.planner_model,
                    max_tokens=args.planner_max_tokens,
                    answer_instruction=example.answer_instruction,
                    max_tasks=max_tasks,
                    min_tasks=min_tasks,
                )
            return result, service_retries
        except ApiCallFailure as exc:
            if exc.category != FAILURE_UNSTABLE_SERVICE or attempt >= max_retries:
                setattr(exc, "planner_service_retries", service_retries)
                raise
            service_retries += 1
            print(
                f"[planner_retry] prompt_dataset={example.dataset} model={exc.model} "
                f"category={exc.category} retry={service_retries}/{max_retries} "
                f"delay={retry_delay:.2f}s message={exc.message}",
                flush=True,
            )
            if retry_delay > 0.0:
                time.sleep(retry_delay)
    raise RuntimeError("unreachable planner retry state")


def planned_selection_call_count(plan: Plan, selection: object) -> int:
    verifier_by_task = getattr(selection, "verifier_by_task", {})
    return len(plan.tasks) + sum(1 for verifier in verifier_by_task.values() if verifier is not None)


def init_quality_model(feature_map: MiniLMFeatureMap, api_candidates: List[str], lambda_reg: float, prior_mean: float) -> WeightedLinearQualityModel:
    probe = feature_map.encode({"probe": "initialize quality model dimension"}, ROLE_EXEC, api_candidates[0])
    return WeightedLinearQualityModel(dimension=len(probe), lambda_reg=lambda_reg, prior_mean=prior_mean)


def build_feature_tables(
    prompt: str,
    plan: Plan,
    api_candidates: List[str],
    feature_map: MiniLMFeatureMap,
    quality_model: WeightedLinearQualityModel,
    beta: float,
    fixed_verifier_model: str,
) -> FeatureTables:
    exec_features: Dict[Tuple[str, str], object] = {}
    ver_features: Dict[Tuple[str, str], object] = {}
    exec_pred: Dict[Tuple[str, str], float] = {}
    exec_ucb: Dict[Tuple[str, str], float] = {}
    exec_uncertainty: Dict[Tuple[str, str], float] = {}

    for task in plan.tasks:
        exec_context = build_execution_context(prompt, plan, task)
        for executor in api_candidates:
            feature = feature_map.encode(exec_context, ROLE_EXEC, executor)
            key = (task.id, executor)
            exec_features[key] = feature
            exec_pred[key] = quality_model.predict(feature)
            exec_ucb[key] = quality_model.ucb(feature, beta=beta)
            exec_uncertainty[key] = quality_model.uncertainty(feature)

        verifier = fixed_verifier_model
        ver_context = build_verification_context(prompt, plan, task, executor_api="selected by optimizer")
        ver_feature = feature_map.encode(ver_context, ROLE_VER, verifier)
        ver_features[(task.id, verifier)] = ver_feature

    return FeatureTables(
        exec_features=exec_features,
        ver_features=ver_features,
        exec_pred=exec_pred,
        exec_ucb=exec_ucb,
        exec_uncertainty=exec_uncertainty,
    )


def build_jove_inputs(
    plan: Plan,
    api_candidates: List[str],
    api_configs: Mapping[str, object],
    resource_model: FeatureWeightedResourceModel,
    tables: FeatureTables,
    fixed_verifier_model: str,
    latency_tolerance_delta: float,
) -> JoveInputs:
    exec_cost: Dict[Tuple[str, str], float] = {}
    verifier_cost: Dict[str, float] = {}
    safe_latency: Dict[Tuple[str, str], float] = {}
    node_latency_tolerance = split_latency_tolerance(latency_tolerance_delta, len(plan.tasks))
    for task in plan.tasks:
        verifier_config = api_configs[fixed_verifier_model]
        legacy_verifier = float(verifier_config.verifier_cost_unit)  # type: ignore[attr-defined]
        verifier_feature = tables.ver_features[(task.id, fixed_verifier_model)]
        verifier_cost[task.id] = resource_model.estimate_cost(
            fixed_verifier_model,
            verifier_feature,
            legacy_verifier,
            role=ROLE_VER,
        )
        for executor in api_candidates:
            config = api_configs[executor]
            legacy_exec = float(config.executor_cost_unit)  # type: ignore[attr-defined]
            exec_feature = tables.exec_features[(task.id, executor)]
            exec_cost[(task.id, executor)] = resource_model.estimate_cost(
                executor,
                exec_feature,
                legacy_exec,
                role=ROLE_EXEC,
            )
            safe_latency[(task.id, executor)] = resource_model.estimate_latency(
                executor,
                exec_feature,
                float(config.safe_latency_prior),  # type: ignore[attr-defined]
                tolerance=node_latency_tolerance,
            )
    return JoveInputs(
        exec_quality_ucb=tables.exec_ucb,
        exec_uncertainty=tables.exec_uncertainty,
        exec_cost=exec_cost,
        verifier_cost=verifier_cost,
        safe_latency=safe_latency,
    )


def realized_finish_times(plan: Plan, execution_latency: Mapping[str, float]) -> Dict[str, float]:
    finish: Dict[str, float] = {}
    tasks = plan.task_by_id()
    for task_id in plan.topological_order():
        pred_finish = max((finish[pred] for pred in tasks[task_id].predecessors), default=0.0)
        finish[task_id] = pred_finish + float(execution_latency.get(task_id, 0.0))
    return finish


def running_correctness_stats(metrics: List[Mapping[str, object]]) -> Dict[str, object]:
    answered = [item for item in metrics if item.get("is_correct") is not None]
    correct = sum(1 for item in answered if bool(item.get("is_correct")))
    average = (correct / len(answered)) if answered else None
    return {
        "answered_prompts": len(answered),
        "correct_prompts": correct,
        "average_correctness": average,
    }


def print_running_correctness(metrics: List[Dict[str, object]]) -> None:
    stats = running_correctness_stats(metrics)
    if metrics:
        metrics[-1].update(
            {
                "running_answered_prompts": stats["answered_prompts"],
                "running_correct_prompts": stats["correct_prompts"],
                "running_average_correctness": stats["average_correctness"],
            }
        )
    average = stats["average_correctness"]
    average_text = "none" if average is None else f"{float(average):.4f}"
    print(
        f"[running_correctness] answered={stats['answered_prompts']} "
        f"correct={stats['correct_prompts']} average_correctness={average_text}",
        flush=True,
    )


def update_quality_from_verifiers(
    quality_model: WeightedLinearQualityModel,
    selection_pairs: Mapping[str, Tuple[str, Optional[str]]],
    verifier_results: Mapping[str, VerifierResult],
    tables: FeatureTables,
) -> int:
    labels: List[ServiceLabel] = []
    for task_id, result in verifier_results.items():
        executor, verifier = selection_pairs[task_id]
        if verifier is None:
            continue

        executor_failure_observed = (
            getattr(result, "failed", False)
            and getattr(result, "final_action", None) == "skipped_executor_failure"
        )
        if getattr(result, "failed", False) and not executor_failure_observed:
            continue

        labels.append(
            ServiceLabel(
                feature=tables.exec_features[(task_id, executor)],
                label=0.0 if executor_failure_observed else (1.0 if result.correct else 0.0),
                weight=1.0,
            )
        )
    quality_model.update_many(labels)
    return len(labels)


def run_prompt_loop(args: argparse.Namespace, artifacts: Optional[TrialArtifacts] = None) -> None:
    args.selection_policy = "jove"
    provider = apply_provider_defaults(args)
    api_candidates = parse_api_candidates(args.api_candidates, provider=provider)
    fixed_verifier_source = args.verifier_model or args.verifier_candidates
    verifier_model = parse_verifier_candidates(fixed_verifier_source, provider=provider)[0]
    args.verifier_model = verifier_model
    if not getattr(args, "llm_judge_model", None):
        args.llm_judge_model = verifier_model
    verifier_candidates = [verifier_model]
    config_models = [*api_candidates, verifier_model]
    api_configs = build_api_configs(dict.fromkeys(config_models).keys())
    examples = build_examples(args)
    args.latency_tolerance_delta = validate_latency_tolerance(
        args.latency_tolerance_delta, "latency_tolerance_delta"
    )

    request_interval = max(float(args.request_interval), float(args.slow_interval) if args.slow else 0.0)
    # --slow means one-by-one execution; force sequential DAG workers.
    effective_parallelism = 1 if args.slow else max(1, int(args.max_parallel_tasks))
    client = OpenAICompatibleClient(
        base_url=args.base_url,
        app_name="jove",
        min_request_interval_seconds=request_interval,
        rate_limit_max_retries=args.rate_limit_max_retries,
        rate_limit_base_delay_seconds=args.rate_limit_base_delay,
        rate_limit_max_delay_seconds=args.rate_limit_max_delay,
        api_key_name=args.api_key_name,
        api_keys_file=args.api_keys_file,
    )
    print(f"[provider] {provider} base_url={args.base_url}", flush=True)

    feature_map = MiniLMFeatureMap(model_name=args.embedding_model)
    quality_model = init_quality_model(feature_map, api_candidates, args.lambda_reg, args.prior_mean)
    resource_model = FeatureWeightedResourceModel()
    q_t = 0.0
    cost_tracker = ModelBudgetCostTracker(scale=float(getattr(args, "cost_usd_scale", DEFAULT_COST_USD_SCALE)))
    cost_usd_scale = float(cost_tracker.scale)
    planner_skip_count = 0
    budget_infeasible_count = 0
    latency_infeasible_count = 0
    executor_failures_by_model: Counter[str] = Counter()
    verifier_failures_by_model: Counter[str] = Counter()
    rate_limit_retries_by_model_stage: Counter[str] = Counter()
    metrics: List[Dict[str, object]] = []
    print(
        f"[setup] prompts={len(examples)} dataset={args.dataset} executor_apis={len(api_candidates)} "
        f"sample_size={args.sample_size} seed={args.seed} "
        f"planner_model={args.planner_model} planner_max_retries={args.planner_max_retries} "
        f"fixed_verifier={verifier_model} verifier_max_retries={args.verifier_max_retries} "
        f"verifier_retry_delay={args.verifier_retry_delay:.2f}s embedding_model={args.embedding_model} "
        f"policy=jove k_c={args.k_c:.3g} k_v={args.k_v:.3g} "
        f"latency_delta={args.latency_tolerance_delta:.3g} "
        f"graph_node_counts={getattr(args, 'graph_node_counts', '') or 'disabled'} "
        f"api_key_name={client.api_key_name or 'env_or_code'} "
        f"request_interval={request_interval:.2f}s max_parallel_tasks={effective_parallelism} slow_compat={args.slow}",
        flush=True,
    )

    for prompt_index, example in enumerate(examples, start=1):
        prompt_start = time.perf_counter()
        planner_stage_wall_time = 0.0
        planner_api_latency = None
        print(f"\n[prompt {prompt_index}/{len(examples)}] {example.question}", flush=True)
        if example.answer:
            print(f"[gold_answer] {example.answer}", flush=True)
        graph_fields = graph_variant_metric_fields(example)
        if graph_fields.get("planner_node_count") is not None:
            print(
                f"[graph_variant] source_example_index={graph_fields.get('source_example_index')} "
                f"node_count={graph_fields.get('planner_node_count')} "
                f"pseudo_sample_index={graph_fields.get('pseudo_sample_index')} "
                f"graph_variant_index={graph_fields.get('graph_variant_index')}",
                flush=True,
            )
        q_before = q_t
        profile_config = resolve_reasoning_profile(args, example)
        planner_max_tasks = planner_max_tasks_for_example(example, profile_config=profile_config)
        planner_min_tasks = planner_min_tasks_for_example(example, planner_max_tasks, profile_config=profile_config)
        planner_service_retries = 0
        print(
            f"[dataset_profile] profile={profile_config.name} final_aggregation={profile_config.use_final_aggregation} "
            f"full_context={profile_config.use_full_context} answer_normalization={profile_config.use_answer_normalization} "
            f"calibrated_verifier={profile_config.use_calibrated_verifier} "
            f"planner_min_tasks={planner_min_tasks} planner_max_tasks={planner_max_tasks} "
            f"executor_max_tokens={profile_config.executor_max_tokens or 'default'}",
            flush=True,
        )

        print(
            f"[1/8] Planning executable task DAG ... min_tasks={planner_min_tasks} max_tasks={planner_max_tasks} "
            f"planner_retries={args.planner_max_retries}",
            flush=True,
        )
        planner_start = time.perf_counter()
        try:
            planner_chat_result, planner_service_retries = request_plan_with_retries(
                args=args,
                client=client,
                example=example,
                max_tasks=planner_max_tasks,
                profile_config=profile_config,
                min_tasks=planner_min_tasks,
            )
        except ApiCallFailure as exc:
            planner_stage_wall_time = time.perf_counter() - planner_start
            prompt_elapsed = time.perf_counter() - prompt_start
            if exc.category == FAILURE_FATAL_CONFIG:
                raise
            planner_service_retries = int(getattr(exc, "planner_service_retries", planner_service_retries))
            planner_skip_count += 1
            if exc.retry_count:
                rate_limit_retries_by_model_stage[f"planner:{exc.model}"] += exc.retry_count
            print(
                f"[planner_skip] prompt_index={prompt_index} category={exc.category} model={exc.model} "
                f"status_code={exc.status_code} retry_count={exc.retry_count} "
                f"planner_service_retries={planner_service_retries} action=skip_prompt message={exc.message}",
                flush=True,
            )
            metrics.append(
                {
                    "prompt_index": prompt_index,
                    "dataset": example.dataset,
                    "answer_type": example.answer_type,
                    **_profile_dict(profile_config),
                    "selection_policy": "jove",
                    "skipped": True,
                    "skip_stage": "planner",
                    "skip_reason": exc.category,
                    "failure_metadata": {**exc.to_dict(), "planner_service_retries": planner_service_retries},
                    **_graph_metric_fields(example, planner_min_tasks, planner_max_tasks),
                    "planner_service_retries": planner_service_retries,
                    "q_before": q_before,
                    "q_after": q_t,
                    "q_delta": q_t - q_before,
                    "virtual_queue_before": q_before,
                    "virtual_queue_after": q_t,
                    "queue_budget_gamma": args.gamma,
                    "budget_feasible": None,
                    "planned_call_count": 0,
                    "cost": 0.0,
                    "expected_cost": 0.0,
                    "realized_cost": 0.0,
                    "planner_stage_wall_time": planner_stage_wall_time,
                    "planner_api_latency_seconds": exc.latency_seconds,
                    "realized_total_latency": prompt_elapsed,
                    "realized_prompt_wall_time_seconds": prompt_elapsed,
                    "verifier_calls": 0,
                    "is_correct": None,
                }
            )
            print_running_correctness(metrics)
            save_experiment_results(
                artifacts,
                args,
                examples,
                api_candidates,
                verifier_candidates,
                api_configs,
                metrics,
                build_run_summary(
                    args,
                    metrics,
                    q_t,
                    run_status="running",
                    planner_skips=planner_skip_count,
                    budget_infeasible_skips=budget_infeasible_count,
                    latency_infeasible_skips=latency_infeasible_count,
                    executor_service_failures_by_model=executor_failures_by_model,
                    verifier_service_failures_by_model=verifier_failures_by_model,
                    rate_limit_retries_by_model_stage=rate_limit_retries_by_model_stage,
                ),
            )
            continue
        planner_stage_wall_time = time.perf_counter() - planner_start
        planner_api_latency = planner_chat_result.latency_seconds
        if planner_chat_result.retry_count:
            rate_limit_retries_by_model_stage[f"planner:{args.planner_model}"] += planner_chat_result.retry_count
        planner_raw_response = planner_response_to_text(planner_chat_result)
        planner_source = planner_response_source(planner_chat_result)
        print(f"[planner_response_source] {planner_source}", flush=True)
        try:
            plan = parse_plan_from_chat_result(
                planner_chat_result,
                max_tasks=planner_max_tasks,
                min_tasks=planner_min_tasks,
            )
        except ValueError as exc:
            prompt_elapsed = time.perf_counter() - prompt_start
            planner_skip_count += 1
            print(
                f"[planner_skip] prompt_index={prompt_index} category=planner_parse_failure model={args.planner_model} "
                f"status_code={planner_chat_result.status_code} retry_count={planner_chat_result.retry_count} "
                f"action=skip_prompt message={exc}",
                flush=True,
            )
            metrics.append(
                {
                    "prompt_index": prompt_index,
                    "dataset": example.dataset,
                    "answer_type": example.answer_type,
                    **_profile_dict(profile_config),
                    "selection_policy": "jove",
                    "skipped": True,
                    "skip_stage": "planner",
                    "skip_reason": "planner_parse_failure",
                    "planner_response_source": planner_source,
                    "planner_raw_response": planner_raw_response,
                    **_graph_metric_fields(example, planner_min_tasks, planner_max_tasks),
                    "planner_service_retries": planner_service_retries,
                    "failure_metadata": {
                        "stage": "planner",
                        "model": args.planner_model,
                        "failure_category": "planner_parse_failure",
                        "status_code": planner_chat_result.status_code,
                        "retry_count": planner_chat_result.retry_count,
                        "planner_service_retries": planner_service_retries,
                        "planner_max_tasks": planner_max_tasks,
                        "final_action": "skip_prompt",
                        "message": str(exc),
                    },
                    "q_before": q_before,
                    "q_after": q_t,
                    "q_delta": q_t - q_before,
                    "virtual_queue_before": q_before,
                    "virtual_queue_after": q_t,
                    "queue_budget_gamma": args.gamma,
                    "budget_feasible": None,
                    "planned_call_count": 0,
                    "cost": 0.0,
                    "expected_cost": 0.0,
                    "realized_cost": 0.0,
                    "planner_stage_wall_time": planner_stage_wall_time,
                    "planner_api_latency_seconds": planner_api_latency,
                    "realized_total_latency": prompt_elapsed,
                    "realized_prompt_wall_time_seconds": prompt_elapsed,
                    "verifier_calls": 0,
                    "is_correct": None,
                }
            )
            print_running_correctness(metrics)
            save_experiment_results(
                artifacts,
                args,
                examples,
                api_candidates,
                verifier_candidates,
                api_configs,
                metrics,
                build_run_summary(
                    args,
                    metrics,
                    q_t,
                    run_status="running",
                    planner_skips=planner_skip_count,
                    budget_infeasible_skips=budget_infeasible_count,
                    latency_infeasible_skips=latency_infeasible_count,
                    executor_service_failures_by_model=executor_failures_by_model,
                    verifier_service_failures_by_model=verifier_failures_by_model,
                    rate_limit_retries_by_model_stage=rate_limit_retries_by_model_stage,
                ),
            )
            continue
        print("[planner_parsed_plan]", flush=True)
        print(json.dumps({"tasks": [task.to_dict() for task in plan.tasks]}, indent=2), flush=True)
        latency_node_tolerance = split_latency_tolerance(args.latency_tolerance_delta, len(plan.tasks))
        latency_quantile_level = 1.0 - latency_node_tolerance
        print(
            f"[latency_tolerance] delta={args.latency_tolerance_delta:.6g} nodes={len(plan.tasks)} "
            f"node_tolerance={latency_node_tolerance:.6g} quantile_level={latency_quantile_level:.6g}",
            flush=True,
        )

        print("[2/8] Computing role-conditioned MiniLM features, quality, and uncertainty coefficients ...", flush=True)
        feature_stage_start = time.perf_counter()
        tables = build_feature_tables(
            prompt=example.question,
            plan=plan,
            api_candidates=api_candidates,
            feature_map=feature_map,
            quality_model=quality_model,
            beta=args.beta,
            fixed_verifier_model=verifier_model,
        )
        coeffs = build_jove_inputs(
            plan,
            api_candidates,
            api_configs,
            resource_model,
            tables,
            fixed_verifier_model=verifier_model,
            latency_tolerance_delta=args.latency_tolerance_delta,
        )
        feature_stage_wall_time = time.perf_counter() - feature_stage_start

        optimization_stage_start = time.perf_counter()
        optimization_stage_wall_time: Optional[float] = None
        constraint_skip: Optional[BaseException] = None
        try:
            print(
                f"[3/8] Solving JOVE allocation (executor + verifier-call decisions) ... "
                f"k_c={args.k_c:.3g} k_v={args.k_v:.3g}",
                flush=True,
            )
            selection = solve_jove_selection(
                plan=plan,
                api_candidates=api_candidates,
                coeffs=coeffs,
                mu_t=args.mu,
                q_t=q_t,
                k_c=args.k_c,
                k_v=args.k_v,
                fixed_verifier_model=verifier_model,
                allow_self_verification=False,
            )
        except RuntimeError as exc:
            if not is_jove_infeasible_error(exc):
                raise
            constraint_skip = exc
        if constraint_skip is not None:
            is_latency_infeasible = True
            if is_latency_infeasible:
                latency_infeasible_count += 1
            else:
                budget_infeasible_count += 1
            optimization_stage_wall_time = time.perf_counter() - optimization_stage_start
            prompt_elapsed = time.perf_counter() - prompt_start
            skip_reason = "latency_infeasible" if is_latency_infeasible else "budget_infeasible"
            skip_stage = "latency" if is_latency_infeasible else "budget"
            print(
                f"[{skip_reason}] prompt_index={prompt_index} policy=jove "
                f"message={constraint_skip}",
                flush=True,
            )
            metrics.append(
                {
                    "prompt_index": prompt_index,
                    "dataset": example.dataset,
                    "answer_type": example.answer_type,
                    **_profile_dict(profile_config),
                    "selection_policy": "jove",
                    "skipped": True,
                    "skip_stage": skip_stage,
                    "skip_reason": skip_reason,
                    "planner_response_source": planner_source,
                    "planner_raw_response": planner_raw_response,
                    **_graph_metric_fields(example, planner_min_tasks, planner_max_tasks),
                    "planner_service_retries": planner_service_retries,
                    "q_before": q_before,
                    "q_after": q_t,
                    "q_delta": q_t - q_before,
                    "virtual_queue_before": q_before,
                    "virtual_queue_after": q_t,
                    "queue_budget_gamma": args.gamma,
                    "budget_feasible": True if is_latency_infeasible else False,
                    "latency_feasible": False if is_latency_infeasible else None,
                    "latency_tolerance_delta": args.latency_tolerance_delta,
                    "latency_node_tolerance": latency_node_tolerance,
                    "latency_quantile_level": latency_quantile_level,
                    "planned_call_count": len(plan.tasks),
                    "cost": 0.0,
                    "expected_cost": 0.0,
                    "realized_cost": 0.0,
                    "planner_stage_wall_time": planner_stage_wall_time,
                    "planner_api_latency_seconds": planner_api_latency,
                    "feature_stage_wall_time": feature_stage_wall_time,
                    "optimization_stage_wall_time": optimization_stage_wall_time,
                    "realized_total_latency": prompt_elapsed,
                    "realized_prompt_wall_time_seconds": prompt_elapsed,
                    "verifier_calls": 0,
                    "is_correct": None,
                }
            )
            print_running_correctness(metrics)
            save_experiment_results(
                artifacts,
                args,
                examples,
                api_candidates,
                verifier_candidates,
                api_configs,
                metrics,
                build_run_summary(
                    args,
                    metrics,
                    q_t,
                    run_status="running",
                    planner_skips=planner_skip_count,
                    budget_infeasible_skips=budget_infeasible_count,
                    latency_infeasible_skips=latency_infeasible_count,
                    executor_service_failures_by_model=executor_failures_by_model,
                    verifier_service_failures_by_model=verifier_failures_by_model,
                    rate_limit_retries_by_model_stage=rate_limit_retries_by_model_stage,
                ),
            )
            continue
        optimization_stage_wall_time = time.perf_counter() - optimization_stage_start
        tasks_by_id = plan.task_by_id()
        for task_id in plan.topological_order():
            selected_executor = selection.executor_by_task[task_id]
            selected_verifier = selection.verifier_by_task[task_id]
            verifier_call = 1 if selected_verifier else 0
            exec_uncertainty = coeffs.exec_uncertainty[(task_id, selected_executor)]
            verifier_gain = uncertainty_reduction(exec_uncertainty)
            selected_cost = float(coeffs.exec_cost[(task_id, selected_executor)])
            if selected_verifier is not None:
                selected_cost += float(coeffs.verifier_cost[task_id])
            queue_weighted_cost = args.k_c * q_t * selected_cost
            print(
                f"  {task_id}: exec={selected_executor} fixed_verifier={verifier_model} "
                f"verifier_call={verifier_call} selected_verifier={selected_verifier or 'none'} "
                f"uncertainty={exec_uncertainty:.4f} verification_gain={verifier_gain:.4f} "
                f"queue_weighted_expected_cost={queue_weighted_cost:.2f} "
                f"safe_finish={selection.finish_time[task_id]:.2f}s"
            )

        print("[4/8] Executing selected APIs on the task DAG ...", flush=True)
        execution_stage_start = time.perf_counter()
        outputs, execution_results = execute_plan(
            client=client,
            prompt=example.question,
            plan=plan,
            executor_by_task=selection.executor_by_task,
            max_parallel_tasks=effective_parallelism,
            answer_instruction=example.answer_instruction,
            max_tokens=profile_config.executor_max_tokens,
            include_full_prompt=profile_config.use_full_context,
            task_guidance=build_profile_executor_guidance(example, profile_config),
        )
        execution_stage_wall_time = time.perf_counter() - execution_stage_start

        for result in execution_results.values():
            if result.retry_count:
                rate_limit_retries_by_model_stage[f"execute:{result.model}"] += result.retry_count
            if result.failed:
                executor_failures_by_model[result.model] += 1

        print("[executor_outputs]", flush=True)
        for task_id in plan.topological_order():
            result = execution_results[task_id]
            failure = ""
            if result.failed:
                failure = (
                    f" | failed=true category={result.failure_category} "
                    f"status_code={result.status_code} retry_count={result.retry_count} action={result.final_action}"
                )
            print(f"--- {task_id} | executor={result.model} | latency={result.latency_seconds:.2f}s{failure} ---", flush=True)
            print(result.output, flush=True)

        print("[5/8] Running selected fixed-verifier calls ...", flush=True)
        verification_stage_start = time.perf_counter()
        verifier_results = run_verifiers(
            client=client,
            prompt=example.question,
            plan=plan,
            executor_by_task=selection.executor_by_task,
            verifier_by_task=selection.verifier_by_task,
            outputs=outputs,
            max_parallel_tasks=effective_parallelism,
            execution_results=execution_results,
            answer_instruction=example.answer_instruction,
            verifier_max_retries=args.verifier_max_retries,
            verifier_retry_delay_seconds=args.verifier_retry_delay,
            include_full_prompt=profile_config.use_full_context,
            calibrated_for_reasoning=profile_config.use_calibrated_verifier,
        )
        verification_stage_wall_time = time.perf_counter() - verification_stage_start
        for result in verifier_results.values():
            if result.retry_count and result.final_action != "skipped_executor_failure":
                rate_limit_retries_by_model_stage[f"verify:{result.verifier}"] += result.retry_count
            if result.failed and result.final_action not in {"skipped_executor_failure", "skip_quality_update"}:
                verifier_failures_by_model[result.verifier] += 1
        for task_id, result in verifier_results.items():
            failure = ""
            if result.failed:
                failure = (
                    f" failed=true category={result.failure_category} status_code={result.status_code} "
                    f"retry_count={result.retry_count} action={result.final_action}"
                )
            print(f"  {task_id}: verifier_call=1 verifier={result.verifier} correct={result.correct}{failure} reason={result.reason[:120]}", flush=True)

        print("[6/8] Updating unified service-quality model and resource estimates ...", flush=True)
        update_count = update_quality_from_verifiers(
            quality_model=quality_model,
            selection_pairs=selection.as_pairs(),
            verifier_results=verifier_results,
            tables=tables,
        )
        execution_latency = {task_id: result.latency_seconds for task_id, result in execution_results.items()}
        latency_update_count = 0
        latency_update_skipped_failed = 0
        for task_id, result in execution_results.items():
            if getattr(result, "failed", False):
                latency_update_skipped_failed += 1
                continue
            legacy = float(api_configs[result.model].executor_cost_unit)  # type: ignore[attr-defined]
            observe_executor_call(
                resource_model,
                result.model,
                tables.exec_features[(task_id, result.model)],
                result.usage,
                legacy,
                result.latency_seconds,
                scale=cost_usd_scale,
            )
            latency_update_count += 1
        for task_id, result in verifier_results.items():
            if result.final_action == "skipped_executor_failure":
                continue
            legacy = float(api_configs[result.verifier].verifier_cost_unit) if result.verifier in api_configs else 0.0  # type: ignore[attr-defined]
            ver_factor = float(api_configs[result.verifier].verifier_cost_factor) if result.verifier in api_configs else 0.01  # type: ignore[attr-defined]
            observe_verifier_call(
                resource_model,
                result.verifier,
                tables.ver_features[(task_id, result.verifier)],
                result.usage,
                legacy,
                result.latency_seconds,
                scale=cost_usd_scale,
                cost_factor=ver_factor,
            )
        print(
            f"  quality_updates={update_count} total_model_updates={quality_model.num_updates} "
            f"latency_updates={latency_update_count} skipped_failed_latency_updates={latency_update_skipped_failed}",
            flush=True,
        )

        print("[7/8] Building final answer and updating budget queue ...", flush=True)
        final_task_id = plan.topological_order()[-1]
        sink_final_answer = outputs[final_task_id].strip()
        final_answer = sink_final_answer
        final_answer_source = "sink_task"
        final_aggregation_result: Optional[FinalAnswerCallResult] = None
        final_aggregation_stage_wall_time = 0.0
        final_aggregation_expected_cost = 0.0
        pre_aggregation_livebench_normalization = None
        if profile_config.name == REASONING_PROFILE_LIVEBENCH and profile_config.use_answer_normalization:
            pre_aggregation_livebench_normalization = normalize_livebench_sink_answer(example, sink_final_answer)
        if profile_config.use_final_aggregation and pre_aggregation_livebench_normalization is not None:
            final_answer = pre_aggregation_livebench_normalization.formatted_answer or sink_final_answer
            final_answer_source = "answer_normalization_pre_aggregation"
            print(
                f"[final_aggregation] skipped=true action=valid_livebench_sink_answer "
                f"source={pre_aggregation_livebench_normalization.source}",
                flush=True,
            )
        elif profile_config.use_final_aggregation:
            aggregation_model = execution_results[final_task_id].model
            final_aggregation_expected_cost = cost_tracker.estimate(
                aggregation_model,
                float(api_configs[aggregation_model].executor_cost_unit),  # type: ignore[attr-defined]
            )
            aggregation_start = time.perf_counter()
            final_aggregation_result = run_final_aggregation(
                client=client,
                example=example,
                plan=plan,
                outputs=outputs,
                model=aggregation_model,
                max_tokens=profile_config.final_aggregation_max_tokens,
            )
            final_aggregation_stage_wall_time = time.perf_counter() - aggregation_start
            if final_aggregation_result.retry_count:
                rate_limit_retries_by_model_stage[f"final_aggregation:{final_aggregation_result.model}"] += final_aggregation_result.retry_count
            if not final_aggregation_result.failed and final_aggregation_result.output.strip():
                final_answer = final_aggregation_result.output.strip()
                final_answer_source = "final_aggregation"
            else:
                final_answer_source = "sink_task_fallback_after_aggregation_failure"
            print(
                f"[final_aggregation] model={final_aggregation_result.model} failed={final_aggregation_result.failed} "
                f"latency={final_aggregation_result.latency_seconds:.2f}s action={final_aggregation_result.final_action or 'used'}",
                flush=True,
            )
            if final_aggregation_result.raw_output:
                print(f"[final_aggregation_output] {final_aggregation_result.raw_output}", flush=True)
        pre_normalization_final_answer = final_answer
        formatted_final_answer = ""
        answer_normalization_result: Optional[AnswerNormalizationResult] = None
        if profile_config.use_answer_normalization:
            answer_normalization_result = normalize_profile_final_answer(
                profile_config=profile_config,
                example=example,
                current_final_answer=final_answer,
                sink_final_answer=sink_final_answer,
                plan=plan,
                outputs=outputs,
            )
            if not answer_normalization_result.failed and answer_normalization_result.formatted_answer:
                formatted_final_answer = answer_normalization_result.formatted_answer
                final_answer = formatted_final_answer
                final_answer_source = "answer_normalization"
            print(
                f"[answer_normalization] failed={answer_normalization_result.failed} "
                f"ambiguous={answer_normalization_result.ambiguous} source={answer_normalization_result.source} "
                f"choice={answer_normalization_result.choice_letter or ''} reason={answer_normalization_result.reason}",
                flush=True,
            )
            if formatted_final_answer:
                print(f"[formatted_final_answer] {formatted_final_answer}", flush=True)
        print(f"[sink_final_answer_source] task_id={final_task_id} executor={execution_results[final_task_id].model}", flush=True)
        print(f"[sink_final_answer] {sink_final_answer}", flush=True)
        print(f"[final_answer_source] {final_answer_source}", flush=True)
        print(f"[final_answer] {final_answer}", flush=True)
        is_correct = score_answer(final_answer, example)
        llm_judge_result = run_optional_llm_judge(args, client, example, final_answer)
        if llm_judge_result is not None:
            if llm_judge_result.retry_count:
                rate_limit_retries_by_model_stage[f"llm_judge:{llm_judge_result.model}"] += llm_judge_result.retry_count
            print(
                f"[llm_judge] correct={llm_judge_result.correct} failed={llm_judge_result.failed} "
                f"model={llm_judge_result.model} raw={llm_judge_result.raw_output}",
                flush=True,
            )
        expected_cost = float(selection.expected_cost) + float(final_aggregation_expected_cost)
        non_verifier_usages = []
        verifier_usages = []
        if planner_chat_result is not None:
            planner_fallback = float(api_configs[args.planner_model].executor_cost_unit) if args.planner_model in api_configs else 0.0  # type: ignore[attr-defined]
            non_verifier_usages.append((args.planner_model, getattr(planner_chat_result, "usage", None), planner_fallback))
            cost_tracker.update(args.planner_model, getattr(planner_chat_result, "usage", None))
        for task_id, exec_result in execution_results.items():
            legacy = float(api_configs[exec_result.model].executor_cost_unit)  # type: ignore[attr-defined]
            non_verifier_usages.append((exec_result.model, exec_result.usage, legacy))
            cost_tracker.update(exec_result.model, exec_result.usage)
        for task_id, ver_result in verifier_results.items():
            legacy = float(api_configs[ver_result.verifier].verifier_cost_unit) if ver_result.verifier in api_configs else 0.0  # type: ignore[attr-defined]
            verifier_usages.append((ver_result.verifier, ver_result.usage, legacy))
            cost_tracker.update(ver_result.verifier, ver_result.usage)
        if final_aggregation_result is not None:
            legacy = float(api_configs[final_aggregation_result.model].executor_cost_unit)  # type: ignore[attr-defined]
            non_verifier_usages.append((final_aggregation_result.model, final_aggregation_result.usage, legacy))
            cost_tracker.update(final_aggregation_result.model, final_aggregation_result.usage)
        if llm_judge_result is not None:
            judge_fallback = float(api_configs[llm_judge_result.model].executor_cost_unit) if llm_judge_result.model in api_configs else 0.0  # type: ignore[attr-defined]
            non_verifier_usages.append((llm_judge_result.model, llm_judge_result.usage, judge_fallback))
            cost_tracker.update(llm_judge_result.model, llm_judge_result.usage)
        verifier_cost_factor = (
            float(api_configs[verifier_model].verifier_cost_factor)  # type: ignore[attr-defined]
            if verifier_model in api_configs
            else 0.01
        )
        realized_usd, realized_cost, queue_realized_cost = queue_budget_with_verifier_factor(
            non_verifier_usages,
            verifier_usages,
            verifier_cost_factor=verifier_cost_factor,
            scale=cost_usd_scale,
        )
        planned_call_count = planned_selection_call_count(plan, selection)
        budget_feasible = realized_cost <= args.gamma + 1e-8
        q_t = update_virtual_queue(q_t, queue_realized_cost, args.gamma)
        realized_finish = realized_finish_times(plan, execution_latency)
        safe_sink_latency = max(selection.finish_time[sink] for sink in plan.sinks())
        latency_feasible = safe_sink_latency <= args.mu + 1e-8
        realized_sink_latency = max(realized_finish[sink] for sink in plan.sinks())
        executor_api_latency_sum = sum(result.latency_seconds for result in execution_results.values())
        verifier_api_latency_sum = sum(result.latency_seconds for result in verifier_results.values())
        final_aggregation_api_latency = final_aggregation_result.latency_seconds if final_aggregation_result is not None else 0.0
        prompt_elapsed = time.perf_counter() - prompt_start

        print("[8/8] Correctness:", flush=True)
        metric = {
            "prompt_index": prompt_index,
            "dataset": example.dataset,
            "answer_type": example.answer_type,
            **_profile_dict(profile_config),
            "planner_response_source": planner_source,
            "planner_raw_response": planner_raw_response,
            "planner_content": (planner_chat_result.message.get("content") if planner_chat_result is not None and planner_chat_result.message else None),
            **_graph_metric_fields(example, planner_min_tasks, planner_max_tasks),
            "planner_service_retries": planner_service_retries,
            "executor_outputs": {
                task_id: {
                    "executor": result.model,
                    "output": result.output,
                    "latency_seconds": result.latency_seconds,
                    "usage": result.usage,
                    "failed": result.failed,
                    "failure_category": result.failure_category,
                    "failure_message": result.failure_message,
                    "status_code": result.status_code,
                    "retry_count": result.retry_count,
                    "final_action": result.final_action,
                }
                for task_id, result in execution_results.items()
            },
            "verifier_results": {
                task_id: {
                    "executor": result.executor,
                    "verifier": result.verifier,
                    "correct": result.correct,
                    "reason": result.reason,
                    "raw_output": result.raw_output,
                    "latency_seconds": result.latency_seconds,
                    "usage": result.usage,
                    "failed": result.failed,
                    "failure_category": result.failure_category,
                    "failure_message": result.failure_message,
                    "status_code": result.status_code,
                    "retry_count": result.retry_count,
                    "final_action": result.final_action,
                }
                for task_id, result in verifier_results.items()
            },
            "final_answer_source_task": final_task_id,
            "final_answer_source": final_answer_source,
            "sink_final_answer": sink_final_answer,
            "pre_normalization_final_answer": pre_normalization_final_answer,
            "formatted_final_answer": formatted_final_answer,
            "answer_normalization_source": answer_normalization_result.source if answer_normalization_result is not None else None,
            "answer_normalization_choice": answer_normalization_result.choice_letter if answer_normalization_result is not None else None,
            "answer_normalization_failed": answer_normalization_result.failed if answer_normalization_result is not None else None,
            "answer_normalization_ambiguous": answer_normalization_result.ambiguous if answer_normalization_result is not None else None,
            "answer_normalization_reason": answer_normalization_result.reason if answer_normalization_result is not None else None,
            "answer_normalization_raw_candidate": answer_normalization_result.raw_candidate if answer_normalization_result is not None else None,
            "final_answer": final_answer,
            "final_aggregation": (
                {
                    "model": final_aggregation_result.model,
                    "output": final_aggregation_result.output,
                    "raw_output": final_aggregation_result.raw_output,
                    "latency_seconds": final_aggregation_result.latency_seconds,
                    "usage": final_aggregation_result.usage,
                    "failed": final_aggregation_result.failed,
                    "failure_category": final_aggregation_result.failure_category,
                    "failure_message": final_aggregation_result.failure_message,
                    "status_code": final_aggregation_result.status_code,
                    "retry_count": final_aggregation_result.retry_count,
                    "final_action": final_aggregation_result.final_action,
                }
                if final_aggregation_result is not None
                else None
            ),
            "is_correct": is_correct,
            "llm_judge_correct": llm_judge_result.correct if llm_judge_result is not None else None,
            "llm_judge_raw": llm_judge_result.raw_output if llm_judge_result is not None else None,
            "llm_judge_model": llm_judge_result.model if llm_judge_result is not None else None,
            "llm_judge_failed": llm_judge_result.failed if llm_judge_result is not None else None,
            "llm_judge_latency_seconds": llm_judge_result.latency_seconds if llm_judge_result is not None else 0.0,
            "llm_judge_usage": llm_judge_result.usage if llm_judge_result is not None else None,
            "q_before": q_before,
            "q_after": q_t,
            "q_delta": q_t - q_before,
            "virtual_queue_before": q_before,
            "virtual_queue_after": q_t,
            "queue_budget_gamma": args.gamma,
            "k_c": args.k_c,
            "k_v": args.k_v,
            "selection_policy": "jove",
            "selected_baseline_model": None,
            "planned_call_count": planned_call_count,
            "budget_feasible": budget_feasible,
            "latency_feasible": latency_feasible,
            "latency_tolerance_delta": args.latency_tolerance_delta,
            "latency_node_tolerance": latency_node_tolerance,
            "latency_quantile_level": latency_quantile_level,
            "cost": realized_cost,
            "expected_cost": expected_cost,
            "realized_cost": realized_cost,
            "realized_usd": realized_usd,
            "queue_realized_cost": queue_realized_cost,
            "verifier_cost_factor": verifier_cost_factor,
            "cost_usd_scale": cost_usd_scale,
            "final_aggregation_expected_cost": final_aggregation_expected_cost,
            "cost_including_final_aggregation": realized_cost,
            "safe_sink_latency": safe_sink_latency,
            "realized_sink_latency": realized_sink_latency,
            "realized_executor_dag_latency": realized_sink_latency,
            "constraint_safe_latency": safe_sink_latency,
            "constraint_realized_latency": realized_sink_latency,
            "realized_verifier_stage_latency": verification_stage_wall_time,
            "realized_total_latency": prompt_elapsed,
            "realized_prompt_wall_time_seconds": prompt_elapsed,
            "realized_api_latency_seconds": executor_api_latency_sum + verifier_api_latency_sum + final_aggregation_api_latency,
            "executor_api_latency_sum": executor_api_latency_sum,
            "verifier_api_latency_sum": verifier_api_latency_sum,
            "final_aggregation_api_latency_seconds": final_aggregation_api_latency,
            "planner_stage_wall_time": planner_stage_wall_time,
            "planner_api_latency_seconds": planner_api_latency,
            "feature_stage_wall_time": feature_stage_wall_time,
            "optimization_stage_wall_time": optimization_stage_wall_time,
            "execution_stage_wall_time": execution_stage_wall_time,
            "verification_stage_wall_time": verification_stage_wall_time,
            "final_aggregation_stage_wall_time": final_aggregation_stage_wall_time,
            "verifier_calls": len(verifier_results),
            "fixed_verifier_model": verifier_model,
            "verifier_max_retries": args.verifier_max_retries,
            "verifier_retry_delay": args.verifier_retry_delay,
            "selected_verifier_calls": {
                task_id: {
                    "executor": selection.executor_by_task[task_id],
                    "fixed_verifier": verifier_model,
                    "verifier_call": selection.verifier_by_task[task_id] is not None,
                    "selected_verifier": selection.verifier_by_task[task_id],
                    "uncertainty": coeffs.exec_uncertainty[(task_id, selection.executor_by_task[task_id])],
                    "uncertainty_reduction": uncertainty_reduction(
                        coeffs.exec_uncertainty[(task_id, selection.executor_by_task[task_id])]
                    ),
                    "queue_weighted_expected_cost": args.k_c
                    * q_before
                    * (
                        float(coeffs.exec_cost[(task_id, selection.executor_by_task[task_id])])
                        + (
                            float(coeffs.verifier_cost[task_id])
                            if selection.verifier_by_task[task_id] is not None
                            else 0.0
                        )
                    ),
                }
                for task_id in plan.topological_order()
            },
            "selected_pairs": selection.as_pairs(),
                                            }
        metrics.append(metric)
        print(f"[is_correct] {str(is_correct).lower()}", flush=True)
        print_running_correctness(metrics)
        save_experiment_results(
            artifacts,
            args,
            examples,
            api_candidates,
            verifier_candidates,
            api_configs,
            metrics,
            build_run_summary(
                args,
                metrics,
                q_t,
                run_status="running",
                quality_model_updates=quality_model.num_updates,
                planner_skips=planner_skip_count,
                budget_infeasible_skips=budget_infeasible_count,
                latency_infeasible_skips=latency_infeasible_count,
                executor_service_failures_by_model=executor_failures_by_model,
                verifier_service_failures_by_model=verifier_failures_by_model,
                rate_limit_retries_by_model_stage=rate_limit_retries_by_model_stage,
            ),
        )
    summary = build_run_summary(
        args,
        metrics,
        q_t,
        run_status="complete",
        quality_model_updates=quality_model.num_updates,
        total_verifier_calls=sum(int(item.get("verifier_calls", 0) or 0) for item in metrics),
        planner_skips=planner_skip_count,
        budget_infeasible_skips=budget_infeasible_count,
        latency_infeasible_skips=latency_infeasible_count,
        executor_service_failures_by_model=executor_failures_by_model,
        verifier_service_failures_by_model=verifier_failures_by_model,
        rate_limit_retries_by_model_stage=rate_limit_retries_by_model_stage,
    )
    save_experiment_results(
        artifacts,
        args,
        examples,
        api_candidates,
        verifier_candidates,
        api_configs,
        metrics,
        summary,
    )
    print("\n[summary]", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if artifacts is not None:
        print(f"[results] metrics_pickle={artifacts.pickle_path} summary_json={artifacts.summary_path}", flush=True)


def run_smoke_test() -> None:
    plan = Plan(
        tasks=[
            TaskNode(id="t1", description="Find the anchor entity", task_type="Anchor Identification"),
            TaskNode(
                id="t2",
                description="Reason over the anchor",
                predecessors=["t1"],
                task_type="Reasoning Over Intermediate Results",
            ),
        ]
    )
    apis = ["cheap", "strong"]
    exec_ucb = {
        ("t1", "cheap"): 0.60,
        ("t1", "strong"): 0.61,
        ("t2", "cheap"): 0.60,
        ("t2", "strong"): 0.61,
    }
    exec_uncertainty = {
        ("t1", "cheap"): 10.0,
        ("t1", "strong"): 1.0,
        ("t2", "cheap"): 10.0,
        ("t2", "strong"): 1.0,
    }
    exec_cost = {(task.id, "cheap"): 1.0 for task in plan.tasks}
    exec_cost.update({(task.id, "strong"): 3.0 for task in plan.tasks})
    verifier_cost = {task.id: 0.75 for task in plan.tasks}
    safe_latency = {(task.id, "cheap"): 1.0 for task in plan.tasks}
    safe_latency.update({(task.id, "strong"): 1.2 for task in plan.tasks})
    coeffs = JoveInputs(
        exec_quality_ucb=exec_ucb,
        exec_uncertainty=exec_uncertainty,
        exec_cost=exec_cost,
        verifier_cost=verifier_cost,
        safe_latency=safe_latency,
    )
    assert abs(split_latency_tolerance(0.1, 2) - 0.05) < 1e-12
    run_resource_estimation_self_checks()

    usd_resource = FeatureWeightedResourceModel(prior_count=1.0, margin_seconds=0.0)
    usd_resource.update_execution("cheap", [1.0, 0.0], 40.0, 0.2)
    usd_resource.update_execution("cheap", [0.0, 1.0], 60.0, 0.8)
    usd_resource.update_verification("strong", [1.0, 0.0], 12.0, 0.0)
    usd_resource.update_verification("strong", [0.0, 1.0], 18.0, 0.0)
    usd_tables = FeatureTables(
        exec_features={
            ("t1", "cheap"): [1.0, 0.0],
            ("t1", "strong"): [1.0, 0.0],
            ("t2", "cheap"): [0.0, 1.0],
            ("t2", "strong"): [0.0, 1.0],
        },
        ver_features={("t1", "strong"): [1.0, 0.0], ("t2", "strong"): [0.0, 1.0]},
        exec_pred={},
        exec_ucb={
            ("t1", "cheap"): 0.5,
            ("t1", "strong"): 0.5,
            ("t2", "cheap"): 0.5,
            ("t2", "strong"): 0.5,
        },
        exec_uncertainty={
            ("t1", "cheap"): 1.0,
            ("t1", "strong"): 1.0,
            ("t2", "cheap"): 1.0,
            ("t2", "strong"): 1.0,
        },
    )
    usd_configs = {
        "cheap": APIConfig(model="cheap", executor_cost_unit=9.0, safe_latency_prior=1.0),
        "strong": APIConfig(model="strong", executor_cost_unit=180.0, safe_latency_prior=2.0),
    }
    usd_coeffs = build_jove_inputs(
        plan, apis, usd_configs, usd_resource, usd_tables, "strong", 0.1
    )
    assert usd_coeffs.exec_cost[("t1", "cheap")] < usd_coeffs.exec_cost[("t2", "cheap")]
    assert usd_coeffs.exec_cost[("t1", "strong")] == 180.0
    low_kv_selection = solve_jove_selection(
        plan=plan,
        api_candidates=apis,
        coeffs=coeffs,
        mu_t=2.0,
        q_t=0.1,
        k_c=1.0,
        k_v=0.0,
        fixed_verifier_model="strong",
        allow_self_verification=False,
    )
    assert all(verifier is None for _executor, verifier in low_kv_selection.as_pairs().values())

    verification_selection = solve_jove_selection(
        plan=plan,
        api_candidates=apis,
        coeffs=coeffs,
        mu_t=2.0,
        q_t=0.1,
        k_c=1.0,
        k_v=2.0,
        fixed_verifier_model="strong",
        allow_self_verification=False,
    )
    assert set(verification_selection.executor_by_task) == {"t1", "t2"}
    assert verification_selection.executor_by_task == {"t1": "cheap", "t2": "cheap"}
    assert all(verifier == "strong" for _executor, verifier in verification_selection.as_pairs().values())
    for task_id, (executor, verifier) in verification_selection.as_pairs().items():
        assert verifier is None or verifier == "strong", task_id
        assert verifier is None or verifier != executor, task_id
    assert max(verification_selection.finish_time[sink] for sink in plan.sinks()) <= 2.0
    assert abs(verification_selection.expected_cost - 3.5) < 1e-8

    high_pressure_selection = solve_jove_selection(
        plan=plan,
        api_candidates=apis,
        coeffs=coeffs,
        mu_t=2.0,
        q_t=10.0,
        k_c=10.0,
        k_v=2.0,
        fixed_verifier_model="strong",
        allow_self_verification=False,
    )
    assert all(verifier is None for _executor, verifier in high_pressure_selection.as_pairs().values())

    from tools.datasets import _subsample
    from tools.planning import PromptExample as _PE

    _ex = [_PE(question=f"q{i}", answer=str(i)) for i in range(10)]
    _s0 = _subsample(_ex, sample_size=0, seed=0)
    _s1 = _subsample(_ex, sample_size=0, seed=1)
    assert len(_s0) == 10 and len(_s1) == 10
    assert [e.question for e in _s0] != [e.question for e in _s1]
    assert [e.question for e in _subsample(_ex, 0, 0)] == [e.question for e in _s0]
    _sub5 = _subsample(_ex, sample_size=5, seed=0)
    assert len(_sub5) == 5
    assert [e.question for e in _sub5] == [e.question for e in _s0[:5]]

    from tools.datasets import expand_examples_by_node_count

    _ex_expand = [_PE(question=f"q{i}", answer=str(i)) for i in range(3)]
    _expanded = expand_examples_by_node_count(_ex_expand, node_counts=(2, 3, 4, 5), seed=0)
    assert len(_expanded) == 12
    assert {e.metadata["pseudo_sample_index"] for e in _expanded} == set(range(12))
    by_question = {}
    for example in _expanded:
        by_question.setdefault(example.question, set()).add(example.metadata["planner_node_count"])
    assert all(counts == {2, 3, 4, 5} for counts in by_question.values())
    assert [e.metadata["source_example_index"] for e in _expanded[:4]] != [0, 0, 0, 0]
    node_example = _expanded[0]
    assert planner_max_tasks_for_example(node_example) == int(node_example.metadata["planner_node_count"])
    assert planner_min_tasks_for_example(node_example, planner_max_tasks_for_example(node_example)) == int(
        node_example.metadata["planner_node_count"]
    )
    assert planner_task_count_phrase(4, 4) == "exactly 4"

    from tools.api_config import (
        DEFAULT_OPENROUTER_MODELS,
        DEFAULT_OPENROUTER_PLANNER_MODEL,
        DEFAULT_OPENROUTER_VERIFIER_MODEL,
        DEFAULT_OPENROUTER_CONFIGS,
        apply_provider_defaults,
        parse_api_candidates,
    )
    from tools.client import DEFAULT_OPENROUTER_API_KEY_NAME, OPENROUTER_BASE_URL, PROVIDER_OPENROUTER

    assert len(DEFAULT_OPENROUTER_MODELS) == 6
    assert "meta-llama/llama-3.2-3b-instruct" not in DEFAULT_OPENROUTER_MODELS
    assert "google/gemma-3-4b-it" not in DEFAULT_OPENROUTER_MODELS
    assert "qwen/qwen3-235b-a22b-2507" not in DEFAULT_OPENROUTER_MODELS
    assert "qwen/qwen3.8-max-0902" not in DEFAULT_OPENROUTER_MODELS
    assert "openai/gpt-oss-120b" in DEFAULT_OPENROUTER_MODELS
    assert "qwen/qwen3-vl-235b-a22b-thinking" in DEFAULT_OPENROUTER_MODELS
    assert "mistralai/mistral-large" not in DEFAULT_OPENROUTER_MODELS
    assert DEFAULT_OPENROUTER_PLANNER_MODEL == "google/gemini-2.5-flash-lite"
    assert DEFAULT_OPENROUTER_PLANNER_MODEL in DEFAULT_OPENROUTER_CONFIGS
    for model_id in DEFAULT_OPENROUTER_MODELS:
        assert model_id in DEFAULT_OPENROUTER_CONFIGS, model_id
    assert parse_api_candidates(None, provider=PROVIDER_OPENROUTER) == list(DEFAULT_OPENROUTER_MODELS)
    openrouter_args = argparse.Namespace(
        base_url=OPENROUTER_BASE_URL,
        planner_model="meta/llama-3.3-70b-instruct",
        verifier_model="meta/llama-3.2-3b-instruct",
        api_key_name="API_key1",
    )
    assert apply_provider_defaults(openrouter_args) == PROVIDER_OPENROUTER
    assert openrouter_args.planner_model == DEFAULT_OPENROUTER_PLANNER_MODEL
    assert openrouter_args.verifier_model == DEFAULT_OPENROUTER_VERIFIER_MODEL
    assert openrouter_args.api_key_name == DEFAULT_OPENROUTER_API_KEY_NAME
    from tools.api_config import lookup_api_config

    verifier_cfg = lookup_api_config("qwen/qwen3.8-max-0902")
    assert abs(verifier_cfg.verifier_cost_factor - 0.01) < 1e-12
    assert abs(verifier_cfg.verifier_cost_unit - 2.0) < 1e-12
    assert abs(lookup_api_config("openai/gpt-oss-120b").executor_cost_unit - 120.0) < 1e-12
    assert abs(lookup_api_config("qwen/qwen3-vl-235b-a22b-thinking").executor_cost_unit - 235.0) < 1e-12
    assert abs(lookup_api_config("mistralai/mistral-large").executor_cost_unit - 130.0) < 1e-12
    assert lookup_api_config("mistralai/mistral-large-2512").model == "mistralai/mistral-large"
    assert lookup_api_config("mistralai/mistral-large-2512:batch").model == "mistralai/mistral-large"

    assert score_answer("<answer>C</answer>", PromptExample("q", "C", answer_type="multiple_choice", metadata={"options": ["a", "b", "c"]}))
    assert score_answer("The answer is \boxed{033}", PromptExample("q", "33", answer_type="numeric"))
    assert score_answer("<solution>1, filmmaking, police-officer</solution>", PromptExample("q", "1, filmmaking, police-officer", answer_type="livebench_solution"))
    assert score_answer(
        "Reasoning omitted.\n<solution>1, filmmaking, police officer, journalist</solution>",
        PromptExample(
            "q",
            "1, filmmaking, police-officer, journalist",
            answer_type="livebench_solution",
            metadata={"expected_answer_items": 4, "task": "zebra_puzzle"},
        ),
    )
    assert score_answer(
        "After propagation, the answers are **no, yes, yes**",
        PromptExample(
            "q",
            "no, yes, yes",
            answer_type="livebench_solution",
            metadata={"expected_answer_items": 3, "task": "web_of_lies_v2"},
        ),
    )
    assert score_answer(
        "The final answer is **03**",
        PromptExample("q", "3", answer_type="livebench_solution", metadata={"expected_answer_items": 1, "task": "spatial"}),
    )
    assert not score_answer(
        "<solution>no, yes</solution>",
        PromptExample("q", "no, yes, yes", answer_type="livebench_solution", metadata={"expected_answer_items": 3}),
    )

    assert classify_api_failure(status_code=429, detail="too many requests") == FAILURE_RATE_LIMIT
    assert classify_api_failure(status_code=503, detail="service unavailable") == FAILURE_UNSTABLE_SERVICE

    class _FakeResponse:
        def __init__(self, status_code: int, payload: object, headers: Optional[Mapping[str, str]] = None) -> None:
            self.status_code = status_code
            self._payload = payload
            self.headers = dict(headers or {})
            self.text = json.dumps(payload)

        @property
        def ok(self) -> bool:
            return 200 <= self.status_code < 300

        def json(self) -> object:
            return self._payload

    class _RetryClient(OpenAICompatibleClient):
        def __init__(self) -> None:
            super().__init__(
                api_key="dummy",
                base_url=NVIDIA_NIM_BASE_URL,
                min_request_interval_seconds=0.0,
                rate_limit_max_retries=1,
                rate_limit_base_delay_seconds=0.0,
                rate_limit_max_delay_seconds=0.0,
            )
            self.models: List[str] = []

        def _post_payload(self, payload: Dict[str, object]) -> _FakeResponse:  # type: ignore[override]
            self.models.append(str(payload["model"]))
            if len(self.models) == 1:
                return _FakeResponse(429, {"error": {"message": "rate limit"}})
            return _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}})

    retry_client = _RetryClient()
    retry_result = retry_client.chat("same-model", [{"role": "user", "content": "hello"}], call_label="smoke")
    assert retry_client.models == ["same-model", "same-model"]
    assert retry_result.retry_count == 1

    class _FailingClient:
        def chat(self, model: str, messages: List[Dict[str, str]], *args: object, call_label: Optional[str] = None, **kwargs: object) -> object:
            raise ApiCallFailure(
                category=FAILURE_UNSTABLE_SERVICE,
                model=model,
                stage=call_label or "api_call",
                message="simulated timeout",
                latency_seconds=0.01,
            )

    failure_plan = Plan(tasks=[TaskNode(id="t1", description="Answer directly")])
    failed_outputs, failed_results = execute_plan(
        client=_FailingClient(),  # type: ignore[arg-type]
        prompt="question",
        plan=failure_plan,
        executor_by_task={"t1": "executor"},
        max_parallel_tasks=1,
    )
    assert failed_outputs["t1"] == DEFAULT_API_FAILURE_OUTPUT
    assert failed_results["t1"].failed
    assert failed_results["t1"].final_action == "default_output"

    class _SlowHttpClient(OpenAICompatibleClient):
        def __init__(self, sleep_seconds: float = 0.25) -> None:
            super().__init__(
                api_key="dummy",
                base_url=NVIDIA_NIM_BASE_URL,
                min_request_interval_seconds=0.0,
                rate_limit_max_retries=0,
            )
            self.sleep_seconds = float(sleep_seconds)

        def _post_payload(self, payload: Dict[str, object]) -> _FakeResponse:  # type: ignore[override]
            time.sleep(self.sleep_seconds)
            return _FakeResponse(
                200,
                {
                    "choices": [{"message": {"content": f"ok-{payload.get('model')}"}}],
                    "usage": {"total_tokens": 1},
                },
            )

    parallel_plan = Plan(
        tasks=[
            TaskNode(id="a", description="Independent branch A"),
            TaskNode(id="b", description="Independent branch B"),
        ]
    )
    parallel_client = _SlowHttpClient(0.25)
    parallel_wall_start = time.perf_counter()
    _, parallel_results = execute_plan(
        client=parallel_client,
        prompt="question",
        plan=parallel_plan,
        executor_by_task={"a": "m1", "b": "m2"},
        max_parallel_tasks=2,
    )
    parallel_wall = time.perf_counter() - parallel_wall_start
    assert set(parallel_results) == {"a", "b"}
    assert parallel_wall < 0.40, f"expected overlapping HTTP, wall={parallel_wall:.3f}s"
    assert abs(parallel_results["a"].latency_seconds - 0.25) < 0.08
    assert abs(parallel_results["b"].latency_seconds - 0.25) < 0.08
    assert abs(uncertainty_reduction(10.0) - 0.5 * math.log(1.0 + 100.0)) < 1e-9

    class _DummyQualityModel:
        def __init__(self) -> None:
            self.labels: List[ServiceLabel] = []

        def update_many(self, labels: List[ServiceLabel]) -> None:
            self.labels = list(labels)

    failed_verifier = VerifierResult(
        task_id="t1",
        executor="executor",
        verifier="verifier",
        correct=False,
        reason=DEFAULT_API_FAILURE_OUTPUT,
        raw_output=DEFAULT_API_FAILURE_OUTPUT,
        latency_seconds=0.0,
        usage={},
        failed=True,
        failure_category=FAILURE_UNSTABLE_SERVICE,
        final_action="skip_quality_update",
    )
    dummy_model = _DummyQualityModel()
    skipped_updates = update_quality_from_verifiers(
        quality_model=dummy_model,  # type: ignore[arg-type]
        selection_pairs={"t1": ("executor", "verifier")},
        verifier_results={"t1": failed_verifier},
        tables=FeatureTables({}, {}, {}, {}, {}),
    )
    assert skipped_updates == 0
    assert dummy_model.labels == []

    executor_failed_verifier_result = VerifierResult(
        task_id="t1",
        executor="executor",
        verifier="verifier",
        correct=False,
        reason="Skipped verifier because executor API call failed due to unstable service",
        raw_output="Skipped verifier because executor API call failed due to unstable service",
        latency_seconds=0.0,
        usage={},
        failed=True,
        failure_category=FAILURE_UNSTABLE_SERVICE,
        final_action="skipped_executor_failure",
    )
    executor_failure_tables = FeatureTables(
        exec_features={("t1", "executor"): [1.0, 0.0]},
        ver_features={},
        exec_pred={},
        exec_ucb={},
        exec_uncertainty={},
    )
    executor_failure_model = _DummyQualityModel()
    executor_failure_updates = update_quality_from_verifiers(
        quality_model=executor_failure_model,  # type: ignore[arg-type]
        selection_pairs={"t1": ("executor", "verifier")},
        verifier_results={"t1": executor_failed_verifier_result},
        tables=executor_failure_tables,
    )
    assert executor_failure_updates == 1
    assert executor_failure_model.labels[0].label == 0.0
    assert executor_failure_model.labels[0].weight == 1.0

    reliable_verifier_result = VerifierResult(
        task_id="t1",
        executor="executor",
        verifier="verifier",
        correct=True,
        reason="verifier accepted the executor output",
        raw_output="correct",
        latency_seconds=0.0,
        usage={},
    )
    reliable_model = _DummyQualityModel()
    reliable_updates = update_quality_from_verifiers(
        quality_model=reliable_model,  # type: ignore[arg-type]
        selection_pairs={"t1": ("executor", "verifier")},
        verifier_results={"t1": reliable_verifier_result},
        tables=executor_failure_tables,
    )
    assert reliable_updates == 1
    assert reliable_model.labels[0].label == 1.0
    assert reliable_model.labels[0].weight == 1.0

    try:
        solve_jove_selection(
            plan=plan,
            api_candidates=apis,
            coeffs=coeffs,
            mu_t=0.5,
            q_t=0.0,
            k_c=1.0,
            k_v=2.0,
            fixed_verifier_model="strong",
            allow_self_verification=False,
        )
        raise AssertionError("expected MILP to be deadline-infeasible")
    except RuntimeError as exc:
        assert is_jove_infeasible_error(exc)

    parser = build_arg_parser()
    bamboogle_args = parser.parse_args(["--dataset", "bamboogle"])
    bamboogle_profile = resolve_reasoning_profile(
        bamboogle_args,
        PromptExample("q", "a", dataset="bamboogle", answer_type="freeform"),
    )
    assert bamboogle_profile.name == REASONING_PROFILE_DEFAULT
    assert not bamboogle_profile.use_final_aggregation
    assert not bamboogle_profile.use_full_context
    assert not bamboogle_profile.use_answer_normalization
    assert not bamboogle_profile.use_calibrated_verifier
    assert not bamboogle_args.enable_llm_judge

    aime_args = parser.parse_args(["--dataset", "aime24"])
    aime_profile = resolve_reasoning_profile(
        aime_args,
        PromptExample("q", "1", dataset="aime24", answer_type="numeric"),
    )
    assert aime_profile.name == REASONING_PROFILE_HARD
    assert aime_profile.planner_max_tasks == 5
    assert aime_profile.executor_max_tokens == 1024
    assert aime_profile.use_final_aggregation
    assert not aime_profile.use_full_context
    assert not aime_profile.use_answer_normalization
    assert not aime_profile.use_calibrated_verifier

    livebench_args = parser.parse_args(["--dataset", "livebench_reasoning"])
    livebench_profile = resolve_reasoning_profile(
        livebench_args,
        PromptExample(
            "q",
            "3",
            dataset="livebench_reasoning",
            answer_type="livebench_solution",
            metadata={"task": "spatial", "expected_answer_items": 1},
        ),
    )
    assert livebench_profile.name == REASONING_PROFILE_LIVEBENCH
    assert livebench_profile.planner_max_tasks == 3
    assert livebench_profile.executor_max_tokens == 1024
    assert livebench_profile.use_final_aggregation
    assert livebench_profile.use_full_context
    assert livebench_profile.use_answer_normalization
    assert livebench_profile.use_calibrated_verifier
    assert "LiveBench-Reasoning" in build_profiled_planner_system_prompt(livebench_profile, 3)

    mmlu_args = parser.parse_args(["--dataset", "mmlu_pro"])
    mmlu_profile = resolve_reasoning_profile(
        mmlu_args,
        PromptExample("q", "A", dataset="mmlu_pro", answer_type="multiple_choice"),
    )
    assert mmlu_profile.name == REASONING_PROFILE_MMLU
    assert mmlu_profile.planner_max_tasks == 5
    assert mmlu_profile.executor_max_tokens == 768
    assert not mmlu_profile.use_final_aggregation
    assert mmlu_profile.use_full_context
    assert mmlu_profile.use_answer_normalization
    assert mmlu_profile.use_calibrated_verifier

    gpqa_args = parser.parse_args(["--dataset", "gpqa"])
    gpqa_profile = resolve_reasoning_profile(
        gpqa_args,
        PromptExample("q", "B", dataset="gpqa", answer_type="multiple_choice"),
    )
    assert gpqa_profile.name == REASONING_PROFILE_MMLU
    assert gpqa_profile.use_answer_normalization
    gpqa_options, gpqa_gold, _ = shuffle_gpqa_choices(
        "correct",
        ["wrong1", "wrong2", "wrong3"],
        record_id="rec-1",
        question="GPQA question?",
        seed=0,
    )
    assert gpqa_options[ord(gpqa_gold) - ord("A")] == "correct"
    assert score_answer(
        f"<answer>{gpqa_gold}</answer>",
        PromptExample(
            "q",
            gpqa_gold,
            dataset="gpqa",
            answer_type="multiple_choice",
            metadata={"options": gpqa_options},
        ),
    )

    mmlu_system_prompt = build_profiled_planner_system_prompt(mmlu_profile, 5)
    assert "Do not create read-only" in mmlu_system_prompt
    assert "formatting is handled outside the DAG" in mmlu_system_prompt
    assert "exactly 4" in build_profiled_planner_system_prompt(mmlu_profile, 4, min_tasks=4)

    downstream_task = TaskNode(
        id="t2",
        description="Compare the options using the intermediate result",
        predecessors=["t1"],
        output_format="selected option letter and concise support",
        input_template="Use {t1} to compare the choices.",
    )
    downstream_prompt = build_task_prompt(
        "Question text\n\nOptions:\nA. alpha\nB. beta",
        downstream_task,
        {"t1": "alpha is supported"},
        include_full_prompt=mmlu_profile.use_full_context,
        task_guidance=build_mmlu_executor_guidance(PromptExample("q", "A", dataset="mmlu_pro", answer_type="multiple_choice")),
    )
    assert "Original user prompt" in downstream_prompt
    assert "Options:" in downstream_prompt
    assert "earliest equivalent option letter" in downstream_prompt
    assert "Resolved dependency values are hints" in downstream_prompt
    default_downstream_prompt = build_task_prompt(
        "Question text\n\nOptions:\nA. alpha\nB. beta",
        downstream_task,
        {"t1": "alpha is supported"},
    )
    assert "Original user prompt" not in default_downstream_prompt

    calibrated_verifier_prompt = build_verifier_prompt(
        "Question text\n\nOptions:\nA. alpha\nB. beta",
        downstream_task,
        {"t1": "alpha is supported"},
        "A",
        include_full_prompt=True,
        calibrated_for_reasoning=True,
    )
    assert "prefer correct=false" in calibrated_verifier_prompt
    assert "Original user prompt" in calibrated_verifier_prompt
    assert "reasonable partial result" not in calibrated_verifier_prompt
    assert "use your best judgment" not in calibrated_verifier_prompt
    strict_verifier_prompt = build_verifier_prompt(
        "Question text",
        downstream_task,
        {"t1": "alpha is supported"},
        "A",
    )
    assert "prefer correct=false" in strict_verifier_prompt
    assert "Original user prompt" in strict_verifier_prompt
    assert "reasonable partial result" not in strict_verifier_prompt
    from tools.execution import _parse_verifier_json

    assert _parse_verifier_json('{"correct": true, "reason": "ok"}') == (True, "ok")
    assert _parse_verifier_json('{"correct": false}')[0] is False
    assert _parse_verifier_json("thinking about the answer...")[0] is None

    norm_example = PromptExample(
        "Who designed the clothing?\n\nOptions:\nA. Donna Karan\nB. Calvin Klein\nC. Vera Wang",
        "A",
        dataset="mmlu_pro",
        answer_type="multiple_choice",
        metadata={"options": ["Donna Karan", "Calvin Klein", "Vera Wang"]},
    )
    norm_plan = Plan(tasks=[
        TaskNode(id="t1", description="Solve", output_format="concise result"),
        TaskNode(id="t2", description="Select option", predecessors=["t1"], output_format="selected option"),
    ])
    assert normalize_mmlu_final_answer(norm_example, "B", "B", norm_plan, {"t1": "", "t2": "B"}).choice_letter == "B"
    assert normalize_mmlu_final_answer(norm_example, "<answer>C</answer>", "", norm_plan, {"t1": "", "t2": ""}).choice_letter == "C"
    malformed = normalize_mmlu_final_answer(norm_example, "<Donna Karan>A</Donna Karan>", "", norm_plan, {"t1": "", "t2": ""})
    assert malformed.formatted_answer == "<answer>A</answer>"
    option_text = normalize_mmlu_final_answer(norm_example, "Donna Karan", "", norm_plan, {"t1": "", "t2": ""})
    assert option_text.choice_letter == "A"
    ambiguous = normalize_mmlu_final_answer(norm_example, "Donna Karan and Calvin Klein", "", norm_plan, {"t1": "", "t2": ""})
    assert ambiguous.failed and ambiguous.ambiguous

    livebench_norm_example = PromptExample(
        "q",
        "3",
        dataset="livebench_reasoning",
        answer_type="livebench_solution",
        metadata={"expected_answer_items": 1, "task": "spatial"},
    )
    livebench_norm = normalize_livebench_final_answer(
        livebench_norm_example,
        "The final answer is **03**",
        "",
        Plan(tasks=[TaskNode(id="t1", description="Solve")]),
        {"t1": ""},
    )
    assert livebench_norm.formatted_answer == "<solution>3</solution>"
    assert livebench_norm.source == "current_final_answer:bold"

    equivalent_options = ["3 in 4", "1 in 3", "2 in 3", "2 in 4", "1 in 10", "1", "3 in 3", "4 in 4", "1 in 4", "1 in 2"]
    equivalent_example = PromptExample("q", "F", dataset="mmlu_pro", answer_type="multiple_choice", metadata={"options": equivalent_options})
    equivalent = normalize_mmlu_final_answer(equivalent_example, "<answer>H</answer>", "", norm_plan, {"t1": "", "t2": ""})
    assert equivalent.choice_letter == "F"
    assert "equivalent_numeric_option:H_to_F" in equivalent.source
    contradictory = normalize_mmlu_final_answer(
        equivalent_example,
        "The probability is 100%, i.e. 4 out of 4, so option H is equivalent. <answer>J</answer>",
        "",
        norm_plan,
        {"t1": "", "t2": ""},
    )
    assert contradictory.choice_letter == "F"
    assert contradictory.source.endswith("numeric_value_match")

    aggregation_off_args = parser.parse_args(["--dataset", "aime24", "--enable-final-aggregation", "off"])
    aggregation_off_profile = resolve_reasoning_profile(
        aggregation_off_args,
        PromptExample("q", "1", dataset="aime24", answer_type="numeric"),
    )
    assert aggregation_off_profile.name == REASONING_PROFILE_HARD
    assert not aggregation_off_profile.use_final_aggregation

    q_next = update_virtual_queue(0.0, verification_selection.expected_cost, gamma=2.0)
    assert q_next >= 0.0
    print(
        json.dumps(
            {
                "smoke_test": "ok",
                "low_kv_selection": low_kv_selection.as_pairs(),
                "verification_selection": verification_selection.as_pairs(),
                "high_pressure_selection": high_pressure_selection.as_pairs(),
                "uncertainty_reduction_cheap": uncertainty_reduction(exec_uncertainty[("t1", "cheap")]),
                "verification_information_gain_u10": uncertainty_reduction(10.0),
                "expected_cost": verification_selection.expected_cost,
            },
            indent=2,
        ),
        flush=True,
    )

def build_trial_artifacts(
    log_dir: str,
    *,
    dataset: str,
    selection_policy: str,
    run_label: str = "",
    enable_log: bool = True,
) -> TrialArtifacts:
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset_slug = _artifact_slug(dataset)
    policy_slug = _artifact_slug(selection_policy)
    label_slug = _artifact_slug(run_label) if str(run_label or "").strip() else ""
    label_part = f"_{label_slug}" if label_slug else ""
    base_id = f"jove_{dataset_slug}_{policy_slug}{label_part}_{time.strftime('%Y%m%d_%H%M%S')}"
    for suffix in range(1000):
        trial_id = base_id if suffix == 0 else f"{base_id}_{suffix:03d}"
        trial_dir = root / trial_id
        try:
            trial_dir.mkdir()
        except FileExistsError:
            continue
        log_path = trial_dir / f"{trial_id}.log" if enable_log else None
        return TrialArtifacts(trial_id=trial_id, trial_dir=trial_dir, log_path=log_path)
    raise RuntimeError(f"Could not create a unique trial directory under {root}")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.smoke_test:
        run_smoke_test()
        return

    args.selection_policy = "jove"
    artifact_dataset = "custom" if args.prompt else args.dataset
    artifacts = build_trial_artifacts(
        args.log_dir,
        dataset=artifact_dataset,
        selection_policy="jove",
        run_label=args.run_label,
        enable_log=not args.no_log,
    )
    if args.no_log:
        print(f"[trial_dir] {artifacts.trial_dir}", flush=True)
        print(f"[results] metrics_pickle={artifacts.pickle_path} summary_json={artifacts.summary_path}", flush=True)
        run_prompt_loop(args, artifacts=artifacts)
        return

    assert artifacts.log_path is not None
    with artifacts.log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = TeeStream(sys.stdout, log_file)
        tee_stderr = TeeStream(sys.stderr, log_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print(f"[trial_dir] {artifacts.trial_dir}", flush=True)
            print(f"[log] Writing run output to {artifacts.log_path}", flush=True)
            print(f"[results] metrics_pickle={artifacts.pickle_path} summary_json={artifacts.summary_path}", flush=True)
            run_prompt_loop(args, artifacts=artifacts)


if __name__ == "__main__":
    main()
