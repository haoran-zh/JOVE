from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from .execution import extract_answer_span, is_exact_match, normalize_for_exact_match
from .planning import PromptExample

ANSWER_FREEFORM = "freeform"
ANSWER_MULTIPLE_CHOICE = "multiple_choice"
ANSWER_NUMERIC = "numeric"
ANSWER_LIVEBENCH_SOLUTION = "livebench_solution"
ANSWER_TYPE_CHOICES = (
    ANSWER_FREEFORM,
    ANSWER_MULTIPLE_CHOICE,
    ANSWER_NUMERIC,
    ANSWER_LIVEBENCH_SOLUTION,
)

LIVEBENCH_FAILURE_MARKERS = (
    "api call failed due to unstable service",
    "math tool failed",
    "wikipedia search failed",
    "tool_failure_output",
)


def _valid_choice_letters(example: PromptExample) -> str:
    options = example.metadata.get("options") if isinstance(example.metadata, dict) else None
    count = len(options) if isinstance(options, list) and options else 10
    return "".join(chr(ord("A") + idx) for idx in range(min(max(count, 1), 26)))


def extract_choice_letter(text: str, valid_letters: Iterable[str]) -> Optional[str]:
    letters = "".join(sorted({str(letter).upper() for letter in valid_letters if str(letter).strip()}))
    if not letters:
        return None
    span = extract_answer_span(text).strip()
    upper = span.upper()
    patterns = [
        rf"(?:FINAL\s+)?(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*[\(\[]?([{letters}])[\)\].:]?",
        rf"^\s*[\(\[]?([{letters}])[\)\].:]?\s*$",
        rf"^\s*[\(\[]?([{letters}])[\)\].:]\s+",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    short = re.sub(r"[^A-Z]", " ", upper).split()
    if len(short) <= 3:
        for token in reversed(short):
            if len(token) == 1 and token in letters:
                return token
    return None


def extract_last_integer(text: str) -> Optional[int]:
    span = extract_answer_span(text)
    boxed = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", span)
    if boxed:
        return int(boxed[-1])
    matches = re.findall(r"(?<![A-Za-z0-9])-?\d+(?![A-Za-z0-9])", span.replace(",", ""))
    if not matches:
        return None
    return int(matches[-1])


def expected_livebench_answer_items(example: PromptExample) -> Optional[int]:
    value = example.metadata.get("expected_answer_items") if isinstance(example.metadata, dict) else None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def extract_livebench_solution_text(text: str) -> Tuple[str, str]:
    value = str(text or "").strip()
    if not value:
        return "", "empty"

    solution_matches = re.findall(r"<solution>\s*(.*?)\s*</solution>", value, flags=re.IGNORECASE | re.DOTALL)
    if solution_matches:
        return solution_matches[-1].strip(), "solution_tag"

    answer_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", value, flags=re.IGNORECASE | re.DOTALL)
    if answer_matches:
        return answer_matches[-1].strip(), "answer_tag"

    bold_matches = re.findall(r"\*\*\s*(.*?)\s*\*\*", value, flags=re.DOTALL)
    bold_matches = [match.strip() for match in bold_matches if match.strip()]
    if bold_matches:
        return bold_matches[-1], "bold"

    nonempty_lines = [line.strip() for line in value.splitlines() if line.strip()]
    phrase_patterns = [
        r"(?:final\s+answer|final\s+answers|answer|answers)\s*(?:is|are|:)\s*(.+)$",
        r"(?:therefore|thus|so),?\s*(?:the\s+)?(?:answer|answers)\s*(?:is|are|:)\s*(.+)$",
    ]
    for line in reversed(nonempty_lines):
        cleaned = line.strip().strip("`")
        for pattern in phrase_patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match and match.group(1).strip():
                return match.group(1).strip(), "final_phrase"

    if nonempty_lines:
        return nonempty_lines[-1].strip().strip("`"), "last_line"
    return value, "full_text"


def _split_livebench_items(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def normalize_livebench_item(text: str) -> str:
    normalized = normalize_for_exact_match(text)
    compact = normalized.replace(" ", "")
    if re.fullmatch(r"-?\d+", compact):
        return str(int(compact))
    return normalized


def normalize_livebench_solution_text(
    text: str,
    expected_items: Optional[int] = None,
) -> Tuple[str, str, bool]:
    value, source = extract_livebench_solution_text(text)
    lowered_value = value.lower()
    if any(marker in lowered_value for marker in LIVEBENCH_FAILURE_MARKERS):
        return "", "failure_output", True
    raw_items = _split_livebench_items(value)
    if expected_items is not None and len(raw_items) != int(expected_items):
        return "", source, True
    normalized_items = [normalize_livebench_item(item) for item in raw_items]
    normalized_items = [item for item in normalized_items if item]
    if expected_items is not None and len(normalized_items) != int(expected_items):
        return "", source, True
    if not normalized_items:
        return "", source, True
    return ", ".join(normalized_items), source, False


def _normalize_solution_list(text: str, expected_items: Optional[int] = None) -> str:
    normalized, _source, failed = normalize_livebench_solution_text(text, expected_items=expected_items)
    if failed:
        return ""
    return ",".join(item.strip() for item in normalized.split(","))


def score_answer(predicted_answer: str, example: PromptExample) -> Optional[bool]:
    if not example.answer:
        return None
    answer_type = str(example.answer_type or ANSWER_FREEFORM)
    if answer_type == ANSWER_MULTIPLE_CHOICE:
        gold = extract_choice_letter(example.answer, _valid_choice_letters(example)) or example.answer.strip().upper()[:1]
        pred = extract_choice_letter(predicted_answer, _valid_choice_letters(example))
        return pred == gold
    if answer_type == ANSWER_NUMERIC:
        gold = extract_last_integer(example.answer)
        pred = extract_last_integer(predicted_answer)
        return pred is not None and gold is not None and pred == gold
    if answer_type == ANSWER_LIVEBENCH_SOLUTION:
        expected_items = expected_livebench_answer_items(example)
        pred = _normalize_solution_list(predicted_answer, expected_items=expected_items)
        gold = _normalize_solution_list(example.answer, expected_items=expected_items)
        return bool(pred) and bool(gold) and pred == gold
    return is_exact_match(predicted_answer, example.answer)
