from __future__ import annotations

import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from .client import (
    FAILURE_FATAL_CONFIG,
    FAILURE_UNSTABLE_SERVICE,
    ApiCallFailure,
    ChatResult,
    OpenAICompatibleClient,
)
from .planning import Plan, TaskNode


DEFAULT_API_FAILURE_OUTPUT = "API call failed due to unstable service"
_SKIPPED_VERIFIER_AFTER_EXECUTOR_FAILURE = "Skipped verifier because executor API call failed due to unstable service"
# Qwen3.8 Max (and other reasoning verifiers) spend the token budget on thinking.
# 32 used to consume the whole cap, so JSON never arrived and the parser defaulted to true.
VERIFIER_MAX_TOKENS = 512


@dataclass
class ExecutionResult:
    task_id: str
    model: str
    output: str
    latency_seconds: float
    usage: Dict[str, object]
    failed: bool = False
    failure_category: Optional[str] = None
    failure_message: Optional[str] = None
    status_code: Optional[int] = None
    retry_count: int = 0
    final_action: Optional[str] = None


@dataclass
class VerifierResult:
    task_id: str
    executor: str
    verifier: str
    correct: bool
    reason: str
    raw_output: str
    latency_seconds: float
    usage: Dict[str, object]
    failed: bool = False
    failure_category: Optional[str] = None
    failure_message: Optional[str] = None
    status_code: Optional[int] = None
    retry_count: int = 0
    final_action: Optional[str] = None


_PLACEHOLDER_RE = re.compile(r"\{\{?\s*(t\d+)\s*\}?\}", flags=re.IGNORECASE)


def compact_dependency_value(value: str) -> str:
    return " ".join(extract_answer_span(value).split())


def render_task_instruction(task: TaskNode, predecessor_outputs: Mapping[str, str]) -> str:
    template = task.resolved_template()

    def replace(match: re.Match[str]) -> str:
        task_id = match.group(1).lower()
        return compact_dependency_value(predecessor_outputs.get(task_id, match.group(0)))

    return " ".join(_PLACEHOLDER_RE.sub(replace, template).split())


def build_task_prompt(
    prompt: str,
    task: TaskNode,
    predecessor_outputs: Mapping[str, str],
    answer_instruction: str = "",
    is_sink: bool = False,
    include_full_prompt: bool = False,
    task_guidance: str = "",
) -> str:
    instruction = render_task_instruction(task, predecessor_outputs)
    lines = []
    if include_full_prompt or not task.predecessors:
        lines.append(f"Original user prompt:\n{prompt}")
    lines.append(f"Current task ({task.id}): {instruction}")
    lines.append(f"Output: {task.output_format}")
    guidance = str(task_guidance or "").strip()
    if guidance:
        lines.append(guidance)
    if is_sink and answer_instruction:
        lines.append(answer_instruction.strip())
    if predecessor_outputs:
        lines.append("\nResolved dependency values:")
        for pred in task.predecessors:
            lines.append(f"- {pred}: {compact_dependency_value(predecessor_outputs.get(pred, ''))}")
    lines.append("Return only this task's concise result.")
    return "\n".join(lines)


def build_verifier_prompt(
    prompt: str,
    task: TaskNode,
    predecessor_outputs: Mapping[str, str],
    executor_output: str,
    answer_instruction: str = "",
    is_sink: bool = False,
    include_full_prompt: bool = False,
    calibrated_for_reasoning: bool = False,
) -> str:
    instruction = render_task_instruction(task, predecessor_outputs)
    # Always show the original question. Verifying a subtask without it
    # produced fluent false-positives (format-ok, fact-wrong).
    lines = [f"Original user prompt:\n{prompt}"]
    lines.extend(
        [
            f"Task to verify ({task.id}): {instruction}",
            f"Output: {task.output_format}",
        ]
    )
    if is_sink and answer_instruction:
        lines.append(answer_instruction.strip())
    if predecessor_outputs:
        lines.append("\nDependent information:")
        for pred in task.predecessors:
            lines.append(f"- {pred}: {predecessor_outputs.get(pred, '')}")
    lines.extend(
        [
            "\nExecutor output:",
            executor_output,
        ]
    )
    if calibrated_for_reasoning:
        lines.extend(
            [
                "Judge factual correctness of this subtask output against the original prompt.",
                "For intermediate tasks, correct=true only if stated facts/clues are actually supported; terse or extra explanation is fine when the content is true.",
                "Do not mark true just because the text is on-topic, well written, or a plausible guess.",
                "For final option-selection tasks, accept a bare option letter, tagged option letter, option text, or concise selected choice only when that choice is supported.",
                "Mark correct=false if any name, date, number, place, or option is wrong, invented, or unverifiable, or if the output is empty, a refusal, off-task, or restates the question.",
                "If you are unsure, prefer correct=false.",
                'Return JSON only: {"correct": true/false, "reason": "short reason"}.',
            ]
        )
    else:
        lines.extend(
            [
                "Is the executor output factually correct for this task?",
                "correct=true only if the required result is right given the original prompt and dependencies.",
                "correct=false if a name/date/number/place/option is wrong or invented, the answer is a fluent guess you cannot verify, the output is empty/refusal/off-task, or only the format is right.",
                "If you are unsure, prefer correct=false.",
                'Return JSON only: {"correct": true/false, "reason": "short reason"}.',
            ]
        )
    return "\n".join(lines)


def extract_answer_span(text: str) -> str:
    stripped = text.strip()
    tagged_match = re.search(r"<answer>\s*(.*?)\s*</answer>", stripped, flags=re.IGNORECASE | re.DOTALL)
    if tagged_match:
        stripped = tagged_match.group(1).strip()
    stripped = re.sub(r"^answer\s*:\s*", "", stripped, flags=re.IGNORECASE)
    return stripped.strip().strip("`").strip()


_MONTH_ALIASES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_PATTERN = "|".join(sorted(_MONTH_ALIASES, key=len, reverse=True))


def _ascii_fold(text: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _clean_answer_text(text: str) -> str:
    return _ascii_fold(extract_answer_span(text)).lower()


def _valid_iso_date(year: int, month: int, day: int) -> str | None:
    from datetime import date

    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def canonical_full_dates(text: str) -> set[str]:
    cleaned = _clean_answer_text(text)
    cleaned = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", cleaned)
    dates: set[str] = set()

    for match in re.finditer(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", cleaned):
        iso = _valid_iso_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if iso:
            dates.add(iso)

    for match in re.finditer(rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", cleaned):
        iso = _valid_iso_date(int(match.group(3)), _MONTH_ALIASES[match.group(1)], int(match.group(2)))
        if iso:
            dates.add(iso)

    for match in re.finditer(rf"\b(\d{{1,2}})\s+({_MONTH_PATTERN})\.?[,]?\s+(\d{{4}})\b", cleaned):
        iso = _valid_iso_date(int(match.group(3)), _MONTH_ALIASES[match.group(2)], int(match.group(1)))
        if iso:
            dates.add(iso)

    return dates


def canonical_years(text: str) -> set[str]:
    return set(re.findall(r"(?<!\d)(?:1[0-9]{3}|2[0-9]{3})(?!\d)", _clean_answer_text(text)))


def _is_single_year_answer(normalized_answer: str) -> bool:
    return re.fullmatch(r"(?:1[0-9]{3}|2[0-9]{3})", normalized_answer) is not None


def _is_compatible_date_or_year(predicted_answer: str, gold_answer: str, pred: str, gold: str) -> bool:
    pred_dates = canonical_full_dates(predicted_answer)
    gold_dates = canonical_full_dates(gold_answer)
    if len(pred_dates) == 1 and pred_dates == gold_dates:
        return True

    pred_years = canonical_years(predicted_answer)
    gold_years = canonical_years(gold_answer)
    if _is_single_year_answer(pred) and len(gold_years) == 1 and pred in gold_years:
        return True
    if _is_single_year_answer(gold) and len(pred_years) == 1 and gold in pred_years:
        return True
    return False


def _name_tokens(normalized_answer: str) -> List[str]:
    tokens = normalized_answer.split()
    if len(tokens) < 2 or any(any(ch.isdigit() for ch in token) for token in tokens):
        return []
    return tokens


def _is_compatible_person_name(pred: str, gold: str) -> bool:
    pred_tokens = _name_tokens(pred)
    gold_tokens = _name_tokens(gold)
    if len(pred_tokens) < 2 or len(gold_tokens) < 2:
        return False
    if pred_tokens[:2] == gold_tokens[:2]:
        return True
    return False


def normalize_for_exact_match(text: str) -> str:
    normalized = _clean_answer_text(text)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return " ".join(normalized.split())


def is_exact_match(predicted_answer: str, gold_answer: str) -> bool:
    pred = normalize_for_exact_match(predicted_answer)
    gold = normalize_for_exact_match(gold_answer)
    if bool(pred) and pred == gold:
        return True
    if _is_compatible_date_or_year(predicted_answer, gold_answer, pred, gold):
        return True
    if _is_compatible_person_name(pred, gold):
        return True
    return False


def _parse_verifier_json(raw: str) -> tuple[Optional[bool], str]:
    text = raw.strip()
    if not text:
        return None, ""
    lowered_text = text.lower()
    if lowered_text in {"true", "false"}:
        return lowered_text == "true", ""
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE))
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        if isinstance(data, dict) and "correct" in data:
            value = data["correct"]
            reason = str(data.get("reason") or data.get("explanation") or "").strip()
            if isinstance(value, bool):
                return value, reason
            if isinstance(value, str):
                return value.strip().lower() in {"true", "yes", "correct", "1"}, reason
        if isinstance(data, bool):
            return data, ""
    lowered = text.lower()
    if re.search(r"\bincorrect\b|\bnot correct\b", lowered):
        return False, ""
    if re.search(r'"correct"\s*:\s*false|\bcorrect\s*=\s*false\b', lowered):
        return False, ""
    if re.search(r'"correct"\s*:\s*true|\bcorrect\s*=\s*true\b', lowered):
        return True, ""
    # Unparsed output must not default to true: that poisoned LinUCB with false positives.
    return None, ""


def _execute_one_task(
    client: OpenAICompatibleClient,
    prompt: str,
    task: TaskNode,
    model: str,
    predecessor_outputs: Mapping[str, str],
    answer_instruction: str = "",
    is_sink: bool = False,
    max_tokens: int = 256,
    include_full_prompt: bool = False,
    task_guidance: str = "",
) -> ExecutionResult:
    messages = [
        {"role": "system", "content": "Solve only the given subtask. Be concise."},
        {
            "role": "user",
            "content": build_task_prompt(
                prompt,
                task,
                predecessor_outputs,
                answer_instruction,
                is_sink,
                include_full_prompt,
                task_guidance,
            ),
        },
    ]
    try:
        response = client.chat(model, messages, 0.1, max_tokens, call_label=f"execute:{task.id}")
    except ApiCallFailure as exc:
        if exc.category == FAILURE_FATAL_CONFIG:
            raise
        return ExecutionResult(
            task_id=task.id,
            model=model,
            output=DEFAULT_API_FAILURE_OUTPUT,
            latency_seconds=exc.latency_seconds,
            usage={},
            failed=True,
            failure_category=exc.category,
            failure_message=exc.message,
            status_code=exc.status_code,
            retry_count=exc.retry_count,
            final_action="default_output",
        )
    output = (response.content or "").strip()
    return ExecutionResult(
        task_id=task.id,
        model=model,
        output=output,
        latency_seconds=response.latency_seconds,
        usage=response.usage,
        status_code=response.status_code,
        retry_count=response.retry_count,
    )


def execute_plan(
    client: OpenAICompatibleClient,
    prompt: str,
    plan: Plan,
    executor_by_task: Mapping[str, str],
    max_parallel_tasks: int = 1,
    answer_instruction: str = "",
    max_tokens: Optional[int] = None,
    include_full_prompt: bool = False,
    task_guidance: str = "",
) -> tuple[Dict[str, str], Dict[str, ExecutionResult]]:
    remaining_preds = {task.id: set(task.predecessors) for task in plan.tasks}
    tasks_by_id = plan.task_by_id()
    outputs: Dict[str, str] = {}
    results: Dict[str, ExecutionResult] = {}
    done: set[str] = set()
    parallelism = max(1, int(max_parallel_tasks))
    sinks = set(plan.sinks())
    task_max_tokens = 256 if max_tokens is None else max(1, int(max_tokens))

    while len(done) < len(plan.tasks):
        ready = sorted([task_id for task_id, preds in remaining_preds.items() if not preds and task_id not in done])
        if not ready:
            raise RuntimeError("No ready tasks found; the graph may contain a cycle.")
        for start in range(0, len(ready), parallelism):
            batch = ready[start:start + parallelism]
            if parallelism == 1:
                for task_id in batch:
                    task = tasks_by_id[task_id]
                    model = executor_by_task[task_id]
                    predecessor_outputs = {pred: outputs[pred] for pred in task.predecessors}
                    result = _execute_one_task(
                        client,
                        prompt,
                        task,
                        model,
                        predecessor_outputs,
                        answer_instruction=answer_instruction,
                        is_sink=task_id in sinks,
                        max_tokens=task_max_tokens,
                        include_full_prompt=include_full_prompt,
                        task_guidance=task_guidance,
                    )
                    outputs[task_id] = result.output
                    results[task_id] = result
                    done.add(task_id)
                    for other in remaining_preds:
                        remaining_preds[other].discard(task_id)
                continue

            futures = []
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                for task_id in batch:
                    task = tasks_by_id[task_id]
                    model = executor_by_task[task_id]
                    predecessor_outputs = {pred: outputs[pred] for pred in task.predecessors}
                    futures.append((
                        task_id,
                        executor.submit(
                            _execute_one_task,
                            client,
                            prompt,
                            task,
                            model,
                            predecessor_outputs,
                            answer_instruction,
                            task_id in sinks,
                            task_max_tokens,
                            include_full_prompt,
                            task_guidance,
                        ),
                    ))
                for task_id, future in futures:
                    result = future.result()
                    outputs[task_id] = result.output
                    results[task_id] = result
                    done.add(task_id)
                    for other in remaining_preds:
                        remaining_preds[other].discard(task_id)
    return outputs, results


def _run_one_verifier(
    client: OpenAICompatibleClient,
    prompt: str,
    task: TaskNode,
    executor_model: str,
    verifier_model: str,
    predecessor_outputs: Mapping[str, str],
    executor_output: str,
    answer_instruction: str = "",
    is_sink: bool = False,
    verifier_max_retries: int = 0,
    verifier_retry_delay_seconds: float = 0.0,
    include_full_prompt: bool = False,
    calibrated_for_reasoning: bool = False,
) -> VerifierResult:
    system_content = (
        "Factual verifier. If unsure, return correct=false. Return JSON only."
        if calibrated_for_reasoning
        else "Strict factual verifier. If unsure, return correct=false. Return JSON only."
    )
    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": build_verifier_prompt(
                prompt,
                task,
                predecessor_outputs,
                executor_output,
                answer_instruction,
                is_sink,
                include_full_prompt,
                calibrated_for_reasoning,
            ),
        },
    ]
    max_retries = max(int(verifier_max_retries), 0)
    retry_delay = max(float(verifier_retry_delay_seconds), 0.0)
    verifier_retries = 0
    total_http_retries = 0
    accumulated_latency = 0.0
    while True:
        try:
            response = client.chat(
                verifier_model,
                messages,
                0.0,
                VERIFIER_MAX_TOKENS,
                call_label=f"verify:{task.id}:executor={executor_model}",
            )
            response.retry_count += total_http_retries + verifier_retries
            response.latency_seconds += accumulated_latency
            break
        except ApiCallFailure as exc:
            if exc.category == FAILURE_FATAL_CONFIG:
                raise
            total_http_retries += int(exc.retry_count or 0)
            accumulated_latency += float(exc.latency_seconds or 0.0)
            if verifier_retries >= max_retries:
                return VerifierResult(
                    task_id=task.id,
                    executor=executor_model,
                    verifier=verifier_model,
                    correct=False,
                    reason=DEFAULT_API_FAILURE_OUTPUT,
                    raw_output=DEFAULT_API_FAILURE_OUTPUT,
                    latency_seconds=accumulated_latency,
                    usage={},
                    failed=True,
                    failure_category=exc.category,
                    failure_message=(
                        f"{exc.message}; verifier_retries={verifier_retries}/{max_retries}"
                    ),
                    status_code=exc.status_code,
                    retry_count=total_http_retries + verifier_retries,
                    final_action="skip_quality_update",
                )
            verifier_retries += 1
            print(
                f"[verifier_retry] task={task.id} executor={executor_model} verifier={verifier_model} "
                f"category={exc.category} retry={verifier_retries}/{max_retries} "
                f"delay={retry_delay:.2f}s status_code={exc.status_code} message={exc.message}",
                flush=True,
            )
            if retry_delay > 0.0:
                time.sleep(retry_delay)
    raw_output = response.content or ""
    parsed, reason = _parse_verifier_json(raw_output)
    if parsed is None:
        print(
            f"[verifier_unparsed] task={task.id} executor={executor_model} verifier={verifier_model} "
            f"raw={raw_output[:180]!r}",
            flush=True,
        )
        return VerifierResult(
            task_id=task.id,
            executor=executor_model,
            verifier=verifier_model,
            correct=False,
            reason="unparsed verifier json; skipped quality update",
            raw_output=raw_output,
            latency_seconds=response.latency_seconds,
            usage=response.usage,
            failed=True,
            failure_category="parse_error",
            failure_message="verifier output did not contain a parseable correct boolean",
            status_code=response.status_code,
            retry_count=response.retry_count,
            final_action="skip_quality_update",
        )
    return VerifierResult(
        task_id=task.id,
        executor=executor_model,
        verifier=verifier_model,
        correct=parsed,
        reason=reason,
        raw_output=raw_output,
        latency_seconds=response.latency_seconds,
        usage=response.usage,
        status_code=response.status_code,
        retry_count=response.retry_count,
    )


def _skipped_verifier_result(
    task: TaskNode,
    executor_model: str,
    verifier_model: str,
    executor_result: ExecutionResult,
) -> VerifierResult:
    return VerifierResult(
        task_id=task.id,
        executor=executor_model,
        verifier=verifier_model,
        correct=False,
        reason=_SKIPPED_VERIFIER_AFTER_EXECUTOR_FAILURE,
        raw_output=_SKIPPED_VERIFIER_AFTER_EXECUTOR_FAILURE,
        latency_seconds=0.0,
        usage={},
        failed=True,
        failure_category=executor_result.failure_category or FAILURE_UNSTABLE_SERVICE,
        failure_message=executor_result.failure_message or _SKIPPED_VERIFIER_AFTER_EXECUTOR_FAILURE,
        status_code=executor_result.status_code,
        retry_count=executor_result.retry_count,
        final_action="skipped_executor_failure",
    )


def run_verifiers(
    client: OpenAICompatibleClient,
    prompt: str,
    plan: Plan,
    executor_by_task: Mapping[str, str],
    verifier_by_task: Mapping[str, Optional[str]],
    outputs: Mapping[str, str],
    max_parallel_tasks: int = 1,
    execution_results: Optional[Mapping[str, ExecutionResult]] = None,
    answer_instruction: str = "",
    verifier_max_retries: int = 0,
    verifier_retry_delay_seconds: float = 0.0,
    include_full_prompt: bool = False,
    calibrated_for_reasoning: bool = False,
) -> Dict[str, VerifierResult]:
    tasks_by_id = plan.task_by_id()
    selected = [task_id for task_id in plan.topological_order() if verifier_by_task.get(task_id)]
    sinks = set(plan.sinks())
    results: Dict[str, VerifierResult] = {}
    parallelism = max(1, int(max_parallel_tasks))
    for start in range(0, len(selected), parallelism):
        batch = selected[start:start + parallelism]
        if parallelism == 1:
            for task_id in batch:
                task = tasks_by_id[task_id]
                verifier = verifier_by_task[task_id]
                assert verifier is not None
                executor_result = execution_results.get(task_id) if execution_results is not None else None
                if executor_result is not None and executor_result.failed:
                    results[task_id] = _skipped_verifier_result(
                        task,
                        executor_by_task[task_id],
                        verifier,
                        executor_result,
                    )
                    continue
                predecessor_outputs = {pred: outputs[pred] for pred in task.predecessors}
                results[task_id] = _run_one_verifier(
                    client,
                    prompt,
                    task,
                    executor_by_task[task_id],
                    verifier,
                    predecessor_outputs,
                    outputs[task_id],
                    answer_instruction=answer_instruction,
                    is_sink=task_id in sinks,
                    verifier_max_retries=verifier_max_retries,
                    verifier_retry_delay_seconds=verifier_retry_delay_seconds,
                    include_full_prompt=include_full_prompt,
                    calibrated_for_reasoning=calibrated_for_reasoning,
                )
            continue

        futures = []
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            for task_id in batch:
                task = tasks_by_id[task_id]
                verifier = verifier_by_task[task_id]
                assert verifier is not None
                executor_result = execution_results.get(task_id) if execution_results is not None else None
                if executor_result is not None and executor_result.failed:
                    results[task_id] = _skipped_verifier_result(
                        task,
                        executor_by_task[task_id],
                        verifier,
                        executor_result,
                    )
                    continue
                predecessor_outputs = {pred: outputs[pred] for pred in task.predecessors}
                futures.append(
                    (
                        task_id,
                        executor.submit(
                            _run_one_verifier,
                            client,
                            prompt,
                            task,
                            executor_by_task[task_id],
                            verifier,
                            predecessor_outputs,
                            outputs[task_id],
                            answer_instruction,
                            task_id in sinks,
                            verifier_max_retries,
                            verifier_retry_delay_seconds,
                            include_full_prompt,
                            calibrated_for_reasoning,
                        ),
                    )
                )
            for task_id, future in futures:
                results[task_id] = future.result()
    return results
