from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class APIConfig:
    model: str
    executor_cost_unit: float
    safe_latency_prior: float
    verifier_cost_factor: float = 1.0
    @property
    def verifier_cost_unit(self) -> float:
        return self.executor_cost_unit * self.verifier_cost_factor

DEFAULT_OPENROUTER_MODELS = [
    "meta-llama/llama-3.2-3b-instruct",
    "google/gemma-3-4b-it",
    "openai/gpt-oss-20b",
    "mistralai/mistral-small-3.2-24b-instruct",
    "qwen/qwen3-32b",
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-oss-120b",
    "mistralai/mistral-large-2512",
]
DEFAULT_OPENROUTER_PLANNER_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_OPENROUTER_VERIFIER_MODEL = "qwen/qwen3-235b-a22b-2507"
DEFAULT_OPENROUTER_INFERENCE_MODEL = "qwen/qwen3-32b"
DEFAULT_OPENROUTER_VERIFIER_MODELS = [
    DEFAULT_OPENROUTER_VERIFIER_MODEL,
]
DEFAULT_VERIFIER_MODEL = DEFAULT_OPENROUTER_VERIFIER_MODEL
DEFAULT_VERIFIER_MODELS = DEFAULT_OPENROUTER_VERIFIER_MODELS
_NIM_PLANNER_DEFAULT = "meta/llama-3.3-70b-instruct"
_NIM_VERIFIER_DEFAULT = "meta/llama-3.2-3b-instruct"


# Decide priors by calling each API at least once. 
DEFAULT_OPENROUTER_CONFIGS: Dict[str, APIConfig] = {
    "meta-llama/llama-3.2-3b-instruct": APIConfig(
        model="meta-llama/llama-3.2-3b-instruct",
        executor_cost_unit=0.0,
        safe_latency_prior=0.0,
    ),
    "google/gemma-3-4b-it": APIConfig(
        model="google/gemma-3-4b-it",
        executor_cost_unit=0.0,
        safe_latency_prior=0.0,
    ),
    "openai/gpt-oss-20b": APIConfig(
        model="openai/gpt-oss-20b",
        executor_cost_unit=0.0,
        safe_latency_prior=0.0,
    ),
    "mistralai/mistral-small-3.2-24b-instruct": APIConfig(
        model="mistralai/mistral-small-3.2-24b-instruct",
        executor_cost_unit=0.0,
        safe_latency_prior=0.0,
    ),
    "qwen/qwen3-32b": APIConfig(
        model="qwen/qwen3-32b",
        executor_cost_unit=0.0,
        safe_latency_prior=0.0,
    ),
    "google/gemini-2.5-flash-lite": APIConfig(
        model="google/gemini-2.5-flash-lite",
        executor_cost_unit=0.0,
        safe_latency_prior=0.0,
    ),
    "meta-llama/llama-3.3-70b-instruct": APIConfig(
        model="meta-llama/llama-3.3-70b-instruct",
        executor_cost_unit=0.0,
        safe_latency_prior=0.0,
    ),
    "openai/gpt-oss-120b": APIConfig(
        model="openai/gpt-oss-120b",
        executor_cost_unit=0.0,
        safe_latency_prior=0.0,
    ),
    "qwen/qwen3-vl-235b-a22b-thinking": APIConfig(
        model="qwen/qwen3-vl-235b-a22b-thinking",
        executor_cost_unit=0.0,
        safe_latency_prior=0.0,
    ),
    "mistralai/mistral-large-2512": APIConfig(
        model="mistralai/mistral-large",
        executor_cost_unit=0.0,
        safe_latency_prior=0.0,
    ),
    "qwen/qwen3-235b-a22b-2507": APIConfig(
        model="qwen/qwen3-235b-a22b-2507",
        executor_cost_unit=0.0,
        safe_latency_prior=0.0,
    ),
    # Verifier only. 2.4T-total MoE flagship; keep catalog unit above 235b.
    "qwen/qwen3.8-max-0902": APIConfig(
        model="qwen/qwen3.8-max-0902",
        executor_cost_unit=0.0,
        safe_latency_prior=0.0,
    ),
}
# Kept so non-OpenRouter callers resolve to the same paper pool.
DEFAULT_NVIDIA_NIM_MODELS = DEFAULT_OPENROUTER_MODELS
DEFAULT_NVIDIA_NIM_CONFIGS = DEFAULT_OPENROUTER_CONFIGS



def _model_aliases(candidates: Iterable[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for model in candidates:
        canonical = str(model).strip()
        if not canonical:
            continue
        aliases[canonical.lower()] = canonical
        aliases[canonical.split("/")[-1].lower()] = canonical
    return aliases


def canonicalize_model_id(model: str, candidates: Iterable[str]) -> str:
    value = str(model or "").strip()
    if not value:
        raise ValueError("model id is empty")
    aliases = _model_aliases(candidates)
    resolved = aliases.get(value.lower())
    if resolved is None:
        options = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unknown model id {value!r}. Expected one of: {options}")
    return resolved


def default_models_for_provider(provider: Optional[str] = None) -> List[str]:
    if str(provider or "").strip().lower() == "openrouter":
        return list(DEFAULT_OPENROUTER_MODELS)
    return list(DEFAULT_NVIDIA_NIM_MODELS)


def parse_api_candidates(value: Optional[str], *, provider: Optional[str] = None) -> List[str]:
    if not value:
        return default_models_for_provider(provider)
    candidates = [item.strip() for item in value.split(",") if item.strip()]
    if not candidates:
        raise ValueError("api candidate list is empty")
    return candidates


def parse_verifier_candidates(value: Optional[str], *, provider: Optional[str] = None) -> List[str]:
    if str(provider or "").strip().lower() == "openrouter":
        default_pool = DEFAULT_OPENROUTER_VERIFIER_MODELS
    else:
        default_pool = DEFAULT_VERIFIER_MODELS
    raw_candidates = default_pool if not value else [item.strip() for item in value.split(",") if item.strip()]
    if not raw_candidates:
        raise ValueError("verifier candidate list is empty")
    default_aliases = _model_aliases(default_pool)
    candidates: List[str] = []
    for item in raw_candidates:
        canonical = default_aliases.get(item.lower(), item)
        if canonical not in candidates:
            candidates.append(canonical)
    return candidates


def normalize_verifier_id(value: object, candidates: Iterable[str] = DEFAULT_VERIFIER_MODELS) -> str:
    return canonicalize_model_id(str(value or ""), candidates)


def lookup_api_config(model: str) -> APIConfig:
    """Resolve cost/latency priors from OpenRouter or NIM catalogs."""
    if model in DEFAULT_OPENROUTER_CONFIGS:
        return DEFAULT_OPENROUTER_CONFIGS[model]
    if model in DEFAULT_NVIDIA_NIM_CONFIGS:
        return DEFAULT_NVIDIA_NIM_CONFIGS[model]
    return APIConfig(model=model, executor_cost_unit=50.0, safe_latency_prior=5.0)


def build_api_configs(candidates: Iterable[str]) -> Dict[str, APIConfig]:
    configs: Dict[str, APIConfig] = {}
    for model in candidates:
        configs[model] = lookup_api_config(model)
    return configs


def apply_provider_defaults(args: object, *, provider: Optional[str] = None) -> str:
    """Remap NIM CLI defaults when running against OpenRouter.

    Mutates ``args`` in place for planner/verifier/api-key-name when the
    current values still match the NVIDIA NIM argparse defaults.
    """
    from .client import (
        DEFAULT_NVIDIA_NIM_API_KEY_NAME,
        DEFAULT_OPENROUTER_API_KEY_NAME,
        PROVIDER_OPENROUTER,
        detect_api_provider,
    )

    base_url = str(getattr(args, "base_url", "") or "")
    resolved = provider or detect_api_provider(base_url)
    if resolved != PROVIDER_OPENROUTER:
        return resolved

    planner = str(getattr(args, "planner_model", "") or "")
    if planner in {"", _NIM_PLANNER_DEFAULT}:
        setattr(args, "planner_model", DEFAULT_OPENROUTER_PLANNER_MODEL)

    verifier = str(getattr(args, "verifier_model", "") or "")
    if verifier in {"", _NIM_VERIFIER_DEFAULT, DEFAULT_VERIFIER_MODEL}:
        setattr(args, "verifier_model", DEFAULT_OPENROUTER_VERIFIER_MODEL)

    key_name = str(getattr(args, "api_key_name", "") or "")
    if key_name in {"", DEFAULT_NVIDIA_NIM_API_KEY_NAME, "API_key1"}:
        setattr(args, "api_key_name", DEFAULT_OPENROUTER_API_KEY_NAME)
    return resolved
