from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from .client import ApiCallFailure, ChatResult, OpenAICompatibleClient, PROVIDER_OPENROUTER

TASK_GROUPS = (
    "Anchor Identification",
    "Fact Retrieval",
    "Reasoning Over Intermediate Results",
    "General Analysis",
)
DEFAULT_TASK_GROUP = "General Analysis"
BAMBOOGLE_DATASET_NAME = "chiayewken/bamboogle"
BAMBOOGLE_SPLIT = "test"
BAMBOOGLE_QUESTION_COLUMNS = ("Question", "question")
BAMBOOGLE_ANSWER_COLUMNS = ("Answer", "answer")
DEFAULT_PLANNER_MAX_TOKENS = int(os.environ.get("PLANNER_MAX_TOKENS", "1024"))
DEFAULT_PLANNER_MAX_TASKS = 5


@dataclass(frozen=True)
class PromptExample:
    question: str
    answer: str = ""
    dataset: str = "bamboogle"
    answer_type: str = "freeform"
    answer_instruction: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def answer_instruction_for_type(answer_type: str) -> str:
    kind = str(answer_type or "freeform").strip().lower()
    if kind == "multiple_choice":
        return "Final format: <answer>LETTER</answer>."
    if kind == "numeric":
        return "Final format: <answer>INTEGER</answer>."
    if kind == "livebench_solution":
        return "Final format: <solution>comma-separated final answer</solution>."
    return "Final format: <answer>short answer</answer>."


def prompt_with_answer_instruction(prompt: str, answer_instruction: str = "") -> str:
    instruction = str(answer_instruction or "").strip()
    base = str(prompt or "").strip()
    return f"{base}\n{instruction}" if instruction else base


@dataclass
class TaskNode:
    id: str
    description: str
    predecessors: List[str] = field(default_factory=list)
    task_type: str = DEFAULT_TASK_GROUP
    output_format: str = "concise answer"
    input_template: Optional[str] = None

    def resolved_template(self) -> str:
        return self.input_template or self.description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "predecessors": list(self.predecessors),
            "type": self.task_type,
            "output_format": self.output_format,
            "input_template": self.resolved_template(),
        }


@dataclass
class Plan:
    tasks: List[TaskNode]

    @property
    def task_ids(self) -> List[str]:
        return [task.id for task in self.tasks]

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return [(pred, task.id) for task in self.tasks for pred in task.predecessors]

    def task_by_id(self) -> Dict[str, TaskNode]:
        return {task.id: task for task in self.tasks}

    def sources(self) -> List[str]:
        return [task.id for task in self.tasks if not task.predecessors]

    def sinks(self) -> List[str]:
        has_child = {task.id: False for task in self.tasks}
        for pred, _child in self.edges:
            has_child[pred] = True
        return [task_id for task_id, child in has_child.items() if not child]

    def children(self) -> Dict[str, List[str]]:
        children = {task.id: [] for task in self.tasks}
        for pred, child in self.edges:
            children[pred].append(child)
        for values in children.values():
            values.sort()
        return children

    def topological_order(self) -> List[str]:
        preds = {task.id: set(task.predecessors) for task in self.tasks}
        order: List[str] = []
        ready = sorted([task_id for task_id, values in preds.items() if not values])
        while ready:
            task_id = ready.pop(0)
            order.append(task_id)
            for other in self.tasks:
                if task_id in preds[other.id]:
                    preds[other.id].remove(task_id)
                    if not preds[other.id]:
                        ready.append(other.id)
                        ready.sort()
        if len(order) != len(self.tasks):
            raise ValueError("Planner returned a graph with a cycle.")
        return order

    def graph_features(self) -> Dict[str, Dict[str, Any]]:
        order = self.topological_order()
        children = self.children()
        depth = {task_id: 0 for task_id in self.task_ids}
        for task_id in order:
            for child in children[task_id]:
                depth[child] = max(depth[child], depth[task_id] + 1)

        reverse_order = list(reversed(order))
        height = {task_id: 0 for task_id in self.task_ids}
        descendants: Dict[str, set[str]] = {task_id: set() for task_id in self.task_ids}
        for task_id in reverse_order:
            for child in children[task_id]:
                height[task_id] = max(height[task_id], 1 + height[child])
                descendants[task_id].add(child)
                descendants[task_id].update(descendants[child])
        max_path_len = max((depth[task_id] + height[task_id] for task_id in self.task_ids), default=0)
        features: Dict[str, Dict[str, Any]] = {}
        sinks = set(self.sinks())
        sources = set(self.sources())
        for task in self.tasks:
            features[task.id] = {
                "depth": depth[task.id],
                "height": height[task.id],
                "in_degree": len(task.predecessors),
                "out_degree": len(children[task.id]),
                "descendant_count": len(descendants[task.id]),
                "is_source": task.id in sources,
                "is_sink": task.id in sinks,
                "is_critical_by_topology": depth[task.id] + height[task.id] == max_path_len,
            }
        return features


@dataclass
class PlannerResult:
    plan: Plan
    raw_response: str


def _resolve_dataset_column_name(available_columns: List[str], candidate_names: Tuple[str, ...]) -> str:
    lower_name_map = {name.lower(): name for name in available_columns}
    for candidate in candidate_names:
        if candidate in available_columns:
            return candidate
        resolved = lower_name_map.get(candidate.lower())
        if resolved is not None:
            return resolved
    raise ValueError(f"Could not find columns {candidate_names} in {available_columns}.")


def load_bamboogle_examples(sample_size: Optional[int], seed: Optional[int] = None) -> List[PromptExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets to load Bamboogle: pip install datasets") from exc
    dataset = load_dataset(BAMBOOGLE_DATASET_NAME, split=BAMBOOGLE_SPLIT)
    question_col = _resolve_dataset_column_name(list(dataset.column_names), BAMBOOGLE_QUESTION_COLUMNS)
    answer_col = _resolve_dataset_column_name(list(dataset.column_names), BAMBOOGLE_ANSWER_COLUMNS)
    examples = [
        PromptExample(
            question=str(question).strip(),
            answer=str(answer).strip(),
            dataset="bamboogle",
            answer_type="freeform",
        )
        for question, answer in zip(dataset[question_col], dataset[answer_col])
        if str(question).strip()
    ]
    if not examples:
        raise ValueError("No usable Bamboogle examples found.")
    import random

    # Always shuffle with seed (full stream or subsample) so multi-seed runs differ in order.
    rng = random.Random(seed)
    rng.shuffle(examples)
    if sample_size is None or int(sample_size) <= 0 or int(sample_size) >= len(examples):
        return examples
    return examples[: max(1, int(sample_size))]


def _parse_json_like(candidate: str) -> Optional[Any]:
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return None


def _extract_balanced_brace_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    start: Optional[int] = None
    depth = 0
    in_string = False
    string_char = ""
    escape = False
    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == string_char:
                in_string = False
            continue
        if char in {"'", '"'}:
            in_string = True
            string_char = char
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:idx + 1])
                start = None
    return candidates


def _extract_plan_from_plaintext(text: str) -> Optional[Dict[str, Any]]:
    task_pattern = re.compile(r"^(?:\d+\.\s*)?(?:[-*]\s*)?(t\d+)\s*[:\-]\s*(.+)$", flags=re.IGNORECASE)
    tasks: List[Dict[str, Any]] = []
    current_id: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_id, current_lines
        if current_id is None:
            return
        tasks.append(
            {
                "id": current_id,
                "description": " ".join(line.strip() for line in current_lines if line.strip()),
                "predecessors": [],
                "type": DEFAULT_TASK_GROUP,
            }
        )
        current_id = None
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = task_pattern.match(line)
        if match:
            flush()
            current_id = match.group(1).lower()
            current_lines = [match.group(2).strip()]
        elif current_id is not None:
            current_lines.append(line)
    flush()
    return {"tasks": tasks} if tasks else None


def _json_repair_candidates(candidate: str) -> List[str]:
    raw = candidate.strip()
    if not raw:
        return []
    candidates = [raw]
    repaired = raw
    # Common model typo: a top-level tasks array closes as ...]]} instead of ...]}.
    for _ in range(3):
        updated = re.sub(r"\]\s*\]\s*\}\s*$", "]}", repaired)
        if updated == repaired:
            break
        repaired = updated
        if repaired not in candidates:
            candidates.append(repaired)
    repaired_commas = re.sub(r",\s*([}\]])", r"\1", repaired)
    if repaired_commas not in candidates:
        candidates.append(repaired_commas)
    return candidates


def _coerce_plan_data(data: Any) -> Optional[Dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        return data
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        return {"tasks": data}
    if isinstance(data, dict) and data.get("id") and data.get("description"):
        item = dict(data)
        item["id"] = "t1"
        item["predecessors"] = []
        item.setdefault("type", DEFAULT_TASK_GROUP)
        item.setdefault("output_format", "concise answer")
        template = str(item.get("input_template") or "").strip()
        if not template or re.search(r"\{\{?\s*t\d+\s*\}?\}", template, flags=re.IGNORECASE):
            item["input_template"] = str(item.get("description", "")).strip()
        return {"tasks": [item]}
    return None


def _parse_plan_candidate(candidate: str) -> Optional[Dict[str, Any]]:
    for variant in _json_repair_candidates(candidate):
        parsed = _parse_json_like(variant)
        coerced = _coerce_plan_data(parsed)
        if coerced is not None:
            return coerced
    return None


def extract_json_object(text: str) -> Dict[str, Any]:
    direct = _parse_plan_candidate(text)
    if direct is not None:
        return direct
    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        parsed = _parse_plan_candidate(block)
        if parsed is not None:
            return parsed
    plan_candidates: List[Dict[str, Any]] = []
    task_candidates: List[Dict[str, Any]] = []
    for candidate in _extract_balanced_brace_candidates(text):
        parsed = _parse_plan_candidate(candidate)
        if parsed is None:
            continue
        if isinstance(parsed.get("tasks"), list) and len(parsed["tasks"]) > 1:
            plan_candidates.append(parsed)
        else:
            task_candidates.append(parsed)
    if plan_candidates:
        return plan_candidates[0]
    if task_candidates:
        return task_candidates[0]
    fallback = _extract_plan_from_plaintext(text)
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not find a JSON object in planner output:\n{text}")


def normalize_task_group(value: Any) -> str:
    normalized = " ".join(str(value or "").split()).strip().lower()
    if not normalized:
        return DEFAULT_TASK_GROUP
    for task_group in TASK_GROUPS:
        if normalized == task_group.lower():
            return task_group
    return DEFAULT_TASK_GROUP


def planner_task_bounds(
    max_tasks: Optional[int] = None,
    min_tasks: Optional[int] = None,
) -> Tuple[int, int]:
    task_limit = max(1, int(max_tasks if max_tasks is not None else DEFAULT_PLANNER_MAX_TASKS))
    if min_tasks is None:
        min_t = 1 if task_limit <= 1 else 2
    else:
        min_t = min(max(int(min_tasks), 1), task_limit)
    return min_t, task_limit


def planner_task_count_phrase(min_tasks: int, max_tasks: int) -> str:
    if int(min_tasks) == int(max_tasks):
        return f"exactly {int(max_tasks)}"
    return f"{int(min_tasks)} to {int(max_tasks)}"


def planner_task_count_constraint(min_tasks: int, max_tasks: int) -> str:
    """Emphatic task-count instruction for system/user planner prompts."""
    min_t = int(min_tasks)
    max_t = int(max_tasks)
    if min_t == max_t:
        ids = ", ".join(f"t{i}" for i in range(1, max_t + 1))
        return (
            f"HARD CONSTRAINT: the JSON \"tasks\" array MUST contain exactly {max_t} task objects "
            f"Do not add extra setup, cleanup, summary, or verification tasks. "
            f"If the question seems to need more or fewer steps, merge or split work so the "
            f"final task count is still exactly {max_t}."
        )
    return (
        f"HARD CONSTRAINT: the JSON \"tasks\" array MUST contain between {min_t} and {max_t} "
        f"task objects (inclusive)."
    )


def build_plan_from_data(
    data: Dict[str, Any],
    verifier_candidates: Optional[Iterable[str]] = None,
    max_tasks: Optional[int] = None,
    min_tasks: Optional[int] = None,
) -> Plan:
    raw_items = data.get("tasks")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"Planner output is missing a non-empty tasks list: {data}")
    # Prefer the requested count in the prompt/schema, but still execute mismatched
    # DAGs rather than skipping the prompt (common on hard datasets).
    n_tasks = len(raw_items)
    if min_tasks is not None and n_tasks < int(min_tasks):
        print(
            f"[planner_count_mismatch] got={n_tasks} below min_tasks={int(min_tasks)}; proceeding with returned DAG",
            flush=True,
        )
    if max_tasks is not None and n_tasks > int(max_tasks):
        print(
            f"[planner_count_mismatch] got={n_tasks} above max_tasks={int(max_tasks)}; proceeding with returned DAG",
            flush=True,
        )
    known_ids = [str(item.get("id", "")).lower() for item in raw_items]
    tasks: List[TaskNode] = []
    for item in raw_items:
        task_id = str(item.get("id", "")).lower().strip()
        if not task_id:
            raise ValueError(f"Task is missing id: {item}")
        description = " ".join(str(item.get("description", "")).split())
        input_template = " ".join(str(item.get("input_template", description)).split())
        predecessors = [str(value).lower() for value in item.get("predecessors", [])]
        referenced_ids = [
            ref_id.lower()
            for ref_id in re.findall(r"\b(?:task\s+)?(t\d+)\b", description, flags=re.IGNORECASE)
        ]
        template_ids = [
            ref_id.lower()
            for ref_id in re.findall(r"\{\{?\s*(t\d+)\s*\}?\}", input_template, flags=re.IGNORECASE)
        ]
        for ref_id in referenced_ids + template_ids:
            if ref_id != task_id and ref_id in known_ids and ref_id not in predecessors:
                predecessors.append(ref_id)
        unique_predecessors: List[str] = []
        for pred in predecessors:
            if pred != task_id and pred in known_ids and pred not in unique_predecessors:
                unique_predecessors.append(pred)
        raw_task_type = " ".join(str(item.get("type", "")).split())
        raw_task_type_lower = raw_task_type.lower()
        description_lower = description.lower()
        if "verification" in raw_task_type_lower or re.match(
            r"^(verify|check|validate|confirm|audit)\b", description_lower
        ):
            raise ValueError(
                "Planner returned a verification task node. "
                "Use the fixed run-level verifier instead of DAG verification tasks."
            )
        tasks.append(
            TaskNode(
                id=task_id,
                description=description,
                predecessors=unique_predecessors,
                task_type=normalize_task_group(raw_task_type),
                output_format=" ".join(str(item.get("output_format", "concise answer")).split()),
                input_template=input_template,
            )
        )
    plan = Plan(tasks=tasks)
    plan.topological_order()
    return plan


def build_planner_system_prompt(
    verifier_candidates: Optional[Iterable[str]] = None,
    max_tasks: int = DEFAULT_PLANNER_MAX_TASKS,
    min_tasks: Optional[int] = None,
) -> str:
    del verifier_candidates  # Kept for backward-compatible callers; verifier is fixed at run level.
    min_t, task_limit = planner_task_bounds(max_tasks, min_tasks)
    count_phrase = planner_task_count_phrase(min_t, task_limit)
    count_constraint = planner_task_count_constraint(min_t, task_limit)
    task_groups_list = ", ".join(f'"{task_group}"' for task_group in TASK_GROUPS)
    system = (
        f"You are a planner for benchmark questions. Decompose the user prompt into {count_phrase} "
        "small executable tasks arranged as a DAG. "
        f"{count_constraint} "
        "Return JSON only; no markdown fences or extra text. "
        "Do not solve the question, compute the final answer, or include reasoning. "
        "For multi-hop factual QA, split anchor/entity lookup from the final attribute lookup. "
        "For math, science, multiple-choice, or reasoning questions, split solution reasoning from final answer extraction. "
        'Each task needs lowercase fields "id", "description", "predecessors", "type", "output_format", "input_template". '
        f'"type" must be one of: {task_groups_list}. '
        'Use "Anchor Identification" for the main entity/event/place/object; "Fact Retrieval" for a direct property; '
        '"Reasoning Over Intermediate Results" for ranking, filtering, arithmetic, conditional checks, or combining earlier outputs; '
        '"General Analysis" otherwise. Do not create task nodes whose purpose is to verify, check, validate, confirm, or audit another task output. '
        'Verification is handled outside the DAG by a fixed verifier model and a later binary verifier-call decision. '
        'Do not include verifier_id, verifier model names, executor APIs, or any API/model selection metadata in tasks. '
        'Use precise output_format values such as "full name only", "date only", "year only", "number only", '
        '"integer only", "option letter only", or "short phrase only". Avoid generic "text", "JSON", "dict", or "boolean" unless required. '
        'The sink task output_format must satisfy any final-format instruction in the user message. '
        'Each input_template must be a concrete executable prompt; use {t1}, {t2}, ... only for predecessor outputs. '
        'The sink input_template must ask for the original final answer using predecessors, not just repeat an intermediate lookup. '
        'Keep dependencies minimal and acyclic. '
        'Schema: {"tasks":[{"id":"t1","description":"...","predecessors":[],"type":"Anchor Identification",'
        '"output_format":"full name only","input_template":"..."}]}. '
        f"Remember: {count_constraint}"
    )
    return system


def build_planner_user_message(
    prompt: str,
    *,
    answer_instruction: str = "",
    max_tasks: int = DEFAULT_PLANNER_MAX_TASKS,
    min_tasks: Optional[int] = None,
) -> str:
    """User-turn planner message with an explicit task-count reminder."""
    min_t, task_limit = planner_task_bounds(max_tasks, min_tasks)
    count_phrase = planner_task_count_phrase(min_t, task_limit)
    count_constraint = planner_task_count_constraint(min_t, task_limit)
    user_prompt = prompt_with_answer_instruction(prompt, answer_instruction)
    return (
        f"Plan requirement: produce {count_phrase} executable DAG tasks.\n"
        f"{count_constraint}\n\n"
        f"Question:\n{user_prompt}"
    )


def build_planner_payload(
    verifier_candidates: Optional[Iterable[str]] = None,
    max_tasks: int = DEFAULT_PLANNER_MAX_TASKS,
    min_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    del verifier_candidates  # Kept for backward-compatible callers; verifier is fixed at run level.
    min_t, task_limit = planner_task_bounds(max_tasks, min_tasks)
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "task_plan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            # Keep schema permissive so a mismatched count still parses;
                            # the prompt still requests the target count.
                            "minItems": 1,
                            "maxItems": max(int(task_limit), 10),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "description": {"type": "string"},
                                    "predecessors": {"type": "array", "items": {"type": "string"}},
                                    "type": {"type": "string", "enum": list(TASK_GROUPS)},
                                    "output_format": {"type": "string"},
                                    "input_template": {"type": "string"},
                                },
                                "required": ["id", "description", "predecessors", "type", "output_format", "input_template"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["tasks"],
                    "additionalProperties": False,
                },
            },
        },
        "plugins": [{"id": "response-healing"}],
    }


def request_plan_with_fixed_api(
    client: OpenAICompatibleClient,
    prompt: str,
    planner_model: str,
    verifier_candidates: Optional[Iterable[str]] = None,
    max_tokens: int = DEFAULT_PLANNER_MAX_TOKENS,
    answer_instruction: str = "",
    max_tasks: int = DEFAULT_PLANNER_MAX_TASKS,
    min_tasks: Optional[int] = None,
) -> ChatResult:
    messages = [
        {
            "role": "system",
            "content": build_planner_system_prompt(
                verifier_candidates, max_tasks=max_tasks, min_tasks=min_tasks
            ),
        },
        {
            "role": "user",
            "content": build_planner_user_message(
                prompt,
                answer_instruction=answer_instruction,
                max_tasks=max_tasks,
                min_tasks=min_tasks,
            ),
        },
    ]
    if client.provider == PROVIDER_OPENROUTER:
        try:
            return client.chat(
                model=planner_model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                extra_payload=build_planner_payload(verifier_candidates, max_tasks=max_tasks, min_tasks=min_tasks),
                call_label="planner",
            )
        except (requests.HTTPError, ApiCallFailure) as exc:
            error_text = str(exc).lower()
            if isinstance(exc, ApiCallFailure):
                error_text = f"{error_text} {exc.response_detail or ''}".lower()
            if not any(key in error_text for key in ("response_format", "json_schema", "structured", "plugin")):
                raise
            return client.chat(model=planner_model, messages=messages, temperature=0.0, max_tokens=max_tokens, call_label="planner")
    return client.chat(model=planner_model, messages=messages, temperature=0.0, max_tokens=max_tokens, call_label="planner")


def planner_response_to_text(result: ChatResult) -> str:
    if isinstance(result.content, str) and result.content.strip():
        return result.content
    if isinstance(result.reasoning_content, str) and result.reasoning_content.strip():
        return result.reasoning_content
    return json.dumps(result.raw, ensure_ascii=False, indent=2)


def planner_response_source(result: ChatResult) -> str:
    message = result.message or {}
    direct_content = message.get("content")
    if isinstance(direct_content, str) and direct_content.strip():
        return "content"
    if isinstance(result.reasoning_content, str) and result.reasoning_content.strip():
        return "reasoning_content"
    return "raw_json"


def parse_plan_from_chat_result(
    result: ChatResult,
    verifier_candidates: Optional[Iterable[str]] = None,
    max_tasks: Optional[int] = None,
    min_tasks: Optional[int] = None,
) -> Plan:
    text = planner_response_to_text(result)
    if not text.strip():
        raw_response = json.dumps(result.raw, ensure_ascii=False, indent=2)
        raise ValueError(
            "Planner API response did not contain textual content or reasoning_content. "
            "The raw provider response was printed before parsing; raw response follows:\n"
            f"{raw_response}"
        )
    return build_plan_from_data(extract_json_object(text), max_tasks=max_tasks, min_tasks=min_tasks)


def plan_with_fixed_api_with_raw_response(
    client: OpenAICompatibleClient,
    prompt: str,
    planner_model: str,
    verifier_candidates: Optional[Iterable[str]] = None,
    max_tokens: int = DEFAULT_PLANNER_MAX_TOKENS,
    answer_instruction: str = "",
    max_tasks: int = DEFAULT_PLANNER_MAX_TASKS,
    min_tasks: Optional[int] = None,
) -> PlannerResult:
    result = request_plan_with_fixed_api(
        client=client,
        prompt=prompt,
        planner_model=planner_model,
        verifier_candidates=verifier_candidates,
        max_tokens=max_tokens,
        answer_instruction=answer_instruction,
        max_tasks=max_tasks,
        min_tasks=min_tasks,
    )
    raw_response = planner_response_to_text(result)
    plan = parse_plan_from_chat_result(
        result,
        verifier_candidates=verifier_candidates,
        max_tasks=max_tasks,
        min_tasks=min_tasks,
    )
    return PlannerResult(plan=plan, raw_response=raw_response)


def plan_with_fixed_api(
    client: OpenAICompatibleClient,
    prompt: str,
    planner_model: str,
    verifier_candidates: Optional[Iterable[str]] = None,
    max_tokens: int = DEFAULT_PLANNER_MAX_TOKENS,
    answer_instruction: str = "",
    max_tasks: int = DEFAULT_PLANNER_MAX_TASKS,
    min_tasks: Optional[int] = None,
) -> Plan:
    return plan_with_fixed_api_with_raw_response(
        client=client,
        prompt=prompt,
        planner_model=planner_model,
        verifier_candidates=verifier_candidates,
        max_tokens=max_tokens,
        answer_instruction=answer_instruction,
        max_tasks=max_tasks,
        min_tasks=min_tasks,
    ).plan


def build_execution_context(prompt: str, plan: Plan, task: TaskNode) -> Dict[str, Any]:
    graph_features = plan.graph_features()[task.id]
    return {
        "original_prompt": prompt,
        "task_id": task.id,
        "task_description": task.description,
        "task_type": task.task_type,
        "output_format": task.output_format,
        "input_template": task.resolved_template(),
        "predecessors": task.predecessors,
        "graph_features": graph_features,
    }


def build_verification_context(prompt: str, plan: Plan, task: TaskNode, executor_api: Optional[str] = None) -> Dict[str, Any]:
    graph_features = plan.graph_features()[task.id]
    return {
        "original_prompt": prompt,
        "task_id": task.id,
        "task_description": task.description,
        "task_type": task.task_type,
        "output_format": task.output_format,
        "input_template": task.resolved_template(),
        "predecessors": task.predecessors,
        "graph_features": graph_features,
        "executor_api": executor_api or "selected at execution time",
        "binary_question": "Does the executor output correctly solve this task given the original prompt and predecessor outputs?",
    }


def node_importance_weights(plan: Plan, rho_desc: float = 0.15, rho_crit: float = 0.5) -> Dict[str, float]:
    features = plan.graph_features()
    return {
        task_id: 1.0
        + rho_desc * float(values["descendant_count"])
        + rho_crit * float(bool(values["is_critical_by_topology"]))
        for task_id, values in features.items()
    }
