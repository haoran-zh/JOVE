from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .planning import PromptExample, answer_instruction_for_type, load_bamboogle_examples

DATASET_BAMBOOGLE = "bamboogle"
DATASET_MMLU_PRO = "mmlu_pro"
DATASET_AIME24 = "aime24"
DATASET_LIVEBENCH_REASONING = "livebench_reasoning"
DATASET_GPQA = "gpqa"
DATASET_CHOICES = (
    DATASET_BAMBOOGLE,
    DATASET_MMLU_PRO,
    DATASET_AIME24,
    DATASET_LIVEBENCH_REASONING,
    DATASET_GPQA,
)

DEFAULT_DATASET_SPLITS = {
    DATASET_BAMBOOGLE: "test",
    DATASET_MMLU_PRO: "test",
    DATASET_AIME24: "train",
    DATASET_LIVEBENCH_REASONING: "test",
    DATASET_GPQA: "train",
}

GPQA_HF_DATASET = "Idavidrein/gpqa"
GPQA_DEFAULT_CONFIG = "gpqa_diamond"
GPQA_CHOICE_LETTERS = ("A", "B", "C", "D")


def _subsample(examples: List[PromptExample], sample_size: Optional[int], seed: Optional[int]) -> List[PromptExample]:
    """Shuffle the stream with ``seed``, then optionally keep the first ``sample_size`` examples.

    Always shuffles (even when using the full stream) so different seeds yield different
    online order for mean/std reporting. ``sample_size <= 0`` means keep all examples
    after shuffling.
    """
    if not examples:
        raise ValueError("No usable examples found.")
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    if sample_size is None or int(sample_size) <= 0 or int(sample_size) >= len(shuffled):
        return shuffled
    return shuffled[: max(1, int(sample_size))]


DEFAULT_GRAPH_NODE_COUNTS = (2, 3, 4, 5)
GRAPH_NODE_EXPAND_SEED_OFFSET = 0x9E3779B9


def parse_graph_node_counts(raw: object) -> Tuple[int, ...]:
    text = str(raw or "").strip()
    if not text:
        return ()
    counts: List[int] = []
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 1:
            raise ValueError(f"graph node counts must be positive, got {value}")
        counts.append(min(value, 5))
    return tuple(counts)


def graph_variant_metric_fields(example: PromptExample) -> Dict[str, object]:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    return {
        "source_example_index": metadata.get("source_example_index"),
        "planner_node_count": metadata.get("planner_node_count"),
        "pseudo_sample_index": metadata.get("pseudo_sample_index"),
        "graph_variant_index": metadata.get("graph_variant_index"),
    }


def expand_examples_by_node_count(
    examples: Sequence[PromptExample],
    node_counts: Sequence[int] = DEFAULT_GRAPH_NODE_COUNTS,
    seed: Optional[int] = None,
) -> List[PromptExample]:
    """Replicate each query for each planner node count, then shuffle the expanded stream.

    Pseudo-sample index is ``source_index * |node_counts| + variant`` before shuffle, so a
    125-query stream becomes 500 graph samples. Shuffling mixes graphs of the same query
    among other queries instead of presenting the four variants consecutively.
    """
    counts = [int(count) for count in node_counts]
    if not counts:
        return list(examples)
    expanded: List[PromptExample] = []
    for source_index, example in enumerate(examples):
        for variant_index, node_count in enumerate(counts):
            metadata = dict(example.metadata or {})
            metadata.update(
                {
                    "source_example_index": source_index,
                    "planner_node_count": int(node_count),
                    "planner_min_tasks": int(node_count),
                    "planner_max_tasks": int(node_count),
                    "graph_variant_index": variant_index,
                    "pseudo_sample_index": source_index * len(counts) + variant_index,
                }
            )
            expanded.append(replace(example, metadata=metadata))
    rng = random.Random((int(seed or 0) ^ GRAPH_NODE_EXPAND_SEED_OFFSET) & 0xFFFFFFFF)
    rng.shuffle(expanded)
    return expanded


def _load_dataset_module() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets to load benchmark streams: pip install datasets") from exc
    return load_dataset


def _option_letter(index: int) -> str:
    return chr(ord("A") + int(index))


def _gpqa_shuffle_seed(record_id: object, question: str, seed: Optional[int]) -> int:
    """Build a stable per-example RNG seed so option order is reproducible across runs."""
    base = int(seed or 0)
    identity = str(record_id or "").strip() or str(question or "").strip()
    digest = hashlib.sha256(f"{base}:{identity}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def shuffle_gpqa_choices(
    correct_answer: str,
    incorrect_answers: Sequence[str],
    *,
    record_id: object = None,
    question: str = "",
    seed: Optional[int] = None,
) -> Tuple[List[str], str, List[int]]:
    """Shuffle GPQA correct+incorrect answers into A-D and return (options, gold_letter, permutation).

    ``permutation[i]`` is the original index placed at display position ``i``, where original
    index 0 is always the correct answer.
    """
    incorrect = [str(item).strip() for item in incorrect_answers]
    if len(incorrect) != 3:
        raise ValueError(f"GPQA expects exactly 3 incorrect answers, got {len(incorrect)}")
    correct = str(correct_answer).strip()
    if not correct:
        raise ValueError("GPQA correct answer is empty")
    raw_choices = [correct, *incorrect]
    order = list(range(4))
    rng = random.Random(_gpqa_shuffle_seed(record_id, question, seed))
    rng.shuffle(order)
    ordered = [raw_choices[idx] for idx in order]
    gold_letter = GPQA_CHOICE_LETTERS[order.index(0)]
    return ordered, gold_letter, order


def load_mmlu_pro_examples(sample_size: Optional[int], seed: Optional[int], split: str) -> List[PromptExample]:
    load_dataset = _load_dataset_module()
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    examples: List[PromptExample] = []
    for row in dataset:
        question = str(row.get("question", "")).strip()
        options = [str(item).strip() for item in row.get("options", [])]
        if not question or not options:
            continue
        option_lines = [f"{_option_letter(idx)}. {option}" for idx, option in enumerate(options)]
        prompt = f"{question}\n\nOptions:\n" + "\n".join(option_lines)
        raw_answer = row.get("answer")
        answer = str(raw_answer).strip().upper()
        if not answer and row.get("answer_index") is not None:
            answer = _option_letter(int(row["answer_index"]))
        metadata: Dict[str, Any] = {
            "question_id": row.get("question_id"),
            "category": row.get("category"),
            "src": row.get("src"),
            "options": options,
            "source_split": split,
            "planner_max_tasks": 2,
        }
        examples.append(
            PromptExample(
                question=prompt,
                answer=answer,
                dataset=DATASET_MMLU_PRO,
                answer_type="multiple_choice",
                answer_instruction=answer_instruction_for_type("multiple_choice"),
                metadata=metadata,
            )
        )
    return _subsample(examples, sample_size, seed)


def _livebench_answer_item_count(answer: str) -> int:
    items = [item.strip() for item in str(answer or "").split(",") if item.strip()]
    return max(1, len(items))


def _livebench_answer_instruction(task: object, expected_items: int) -> str:
    task_name = str(task or "").strip().lower()
    if task_name == "spatial":
        return (
            "Final format: <solution>INTEGER</solution>. Return exactly one integer and no explanation. "
            "For piece/count questions, count the resulting requested objects or pieces, not just source objects. If a plane through a sphere center cuts a solid sphere into two equal halves, that one sphere contributes two hemisphere pieces."
        )
    if task_name == "web_of_lies_v2":
        return (
            f"Final format: <solution>answer1, answer2, answer3</solution>. "
            f"Return exactly {expected_items} yes/no answers in the same order as the questions and no explanation. "
            "Propagate truth values consistently from explicit says/tells-truth/lies statements."
        )
    if task_name == "zebra_puzzle":
        return (
            f"Final format: <solution>answer1, answer2, ...</solution>. "
            f"Return exactly {expected_items} answers in the same order as the questions and no explanation. "
            "Use a one-to-one assignment table: each person has exactly one value per attribute and each value is used once."
        )
    return answer_instruction_for_type("livebench_solution")


def load_aime24_examples(sample_size: Optional[int], seed: Optional[int], split: str) -> List[PromptExample]:
    load_dataset = _load_dataset_module()
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split=split)
    examples: List[PromptExample] = []
    for row in dataset:
        problem = str(row.get("Problem", "")).strip()
        if not problem:
            continue
        answer = str(row.get("Answer", "")).strip()
        prompt = problem
        metadata = {
            "id": row.get("ID"),
            "source_split": split,
            "planner_max_tasks": 2,
        }
        examples.append(
            PromptExample(
                question=prompt,
                answer=answer,
                dataset=DATASET_AIME24,
                answer_type="numeric",
                answer_instruction=answer_instruction_for_type("numeric"),
                metadata=metadata,
            )
        )
    return _subsample(examples, sample_size, seed)


def load_livebench_reasoning_examples(sample_size: Optional[int], seed: Optional[int], split: str) -> List[PromptExample]:
    load_dataset = _load_dataset_module()
    dataset = load_dataset("livebench/reasoning", split=split)
    examples: List[PromptExample] = []
    for row in dataset:
        turns = row.get("turns") or []
        question = str(turns[0] if turns else "").strip()
        if not question:
            continue
        answer = str(row.get("ground_truth", "")).strip()
        expected_items = _livebench_answer_item_count(answer)
        metadata = {
            "question_id": row.get("question_id"),
            "category": row.get("category"),
            "task": row.get("task"),
            "level": row.get("level"),
            "source_split": split,
            "expected_answer_items": expected_items,
            "planner_max_tasks": 3,
        }
        examples.append(
            PromptExample(
                question=question,
                answer=answer,
                dataset=DATASET_LIVEBENCH_REASONING,
                answer_type="livebench_solution",
                answer_instruction=_livebench_answer_instruction(row.get("task"), expected_items),
                metadata=metadata,
            )
        )
    return _subsample(examples, sample_size, seed)


def load_gpqa_examples(
    sample_size: Optional[int],
    seed: Optional[int],
    split: str,
    config_name: str = GPQA_DEFAULT_CONFIG,
) -> List[PromptExample]:
    """Load GPQA as a multiple-choice stream (default: diamond, HF split train).

    Requires HuggingFace access to the gated ``Idavidrein/gpqa`` dataset.
    Options are deterministically shuffled into A-D using ``seed`` + record id.
    """
    load_dataset = _load_dataset_module()
    config = str(config_name or GPQA_DEFAULT_CONFIG).strip() or GPQA_DEFAULT_CONFIG
    source_split = str(split or "train").strip() or "train"
    try:
        dataset = load_dataset(GPQA_HF_DATASET, config, split=source_split)
    except Exception as exc:  # noqa: BLE001 - surface gated-access / network failures clearly
        message = str(exc)
        hint = ""
        lowered = message.lower()
        if "gated" in lowered or "401" in lowered or "403" in lowered or "authentication" in lowered:
            hint = (
                " GPQA is gated on HuggingFace: accept terms for Idavidrein/gpqa and run "
                "`hf auth login` (or set HF_TOKEN) on this machine."
            )
        raise SystemExit(f"Failed to load {GPQA_HF_DATASET} ({config}/{source_split}): {exc}.{hint}") from exc

    examples: List[PromptExample] = []
    for row in dataset:
        question = str(row.get("Question", "")).strip()
        correct = str(row.get("Correct Answer", "")).strip()
        incorrect = [
            str(row.get("Incorrect Answer 1", "")).strip(),
            str(row.get("Incorrect Answer 2", "")).strip(),
            str(row.get("Incorrect Answer 3", "")).strip(),
        ]
        if not question or not correct or any(not item for item in incorrect):
            continue
        record_id = row.get("Record ID")
        try:
            options, answer, permutation = shuffle_gpqa_choices(
                correct,
                incorrect,
                record_id=record_id,
                question=question,
                seed=seed,
            )
        except ValueError:
            continue
        option_lines = [f"{_option_letter(idx)}. {option}" for idx, option in enumerate(options)]
        prompt = f"{question}\n\nOptions:\n" + "\n".join(option_lines)
        metadata: Dict[str, Any] = {
            "record_id": record_id,
            "subdomain": row.get("Subdomain"),
            "high_level_domain": row.get("High-level domain"),
            "options": options,
            "correct_answer_text": correct,
            "option_permutation": permutation,
            "gpqa_config": config,
            "source_split": source_split,
            "canary_string": row.get("Canary String"),
            "planner_max_tasks": 2,
        }
        examples.append(
            PromptExample(
                question=prompt,
                answer=answer,
                dataset=DATASET_GPQA,
                answer_type="multiple_choice",
                answer_instruction=answer_instruction_for_type("multiple_choice"),
                metadata=metadata,
            )
        )
    return _subsample(examples, sample_size, seed)


def load_online_examples(
    dataset_name: str,
    sample_size: Optional[int],
    seed: Optional[int],
    split: Optional[str] = None,
) -> List[PromptExample]:
    """Load a benchmark stream, shuffle with ``seed``, then keep ``sample_size`` (0 = all)."""
    name = str(dataset_name or DATASET_BAMBOOGLE).strip().lower()
    source_split = split or DEFAULT_DATASET_SPLITS.get(name)
    if name == DATASET_BAMBOOGLE:
        return load_bamboogle_examples(sample_size=sample_size, seed=seed)
    if name == DATASET_MMLU_PRO:
        return load_mmlu_pro_examples(sample_size=sample_size, seed=seed, split=source_split or "test")
    if name == DATASET_AIME24:
        return load_aime24_examples(sample_size=sample_size, seed=seed, split=source_split or "train")
    if name == DATASET_LIVEBENCH_REASONING:
        return load_livebench_reasoning_examples(sample_size=sample_size, seed=seed, split=source_split or "test")
    if name == DATASET_GPQA:
        return load_gpqa_examples(sample_size=sample_size, seed=seed, split=source_split or "train")
    raise ValueError(f"Unsupported dataset {dataset_name!r}. Expected one of: {', '.join(DATASET_CHOICES)}")
