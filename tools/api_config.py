from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class APIConfig:
    model: str
    executor_cost_unit: float
    safe_latency_prior: float
    # Paper-style assumption: verification is cheaper than executing the same model.
    # Catalog prior and (for OpenRouter) discounted verifier USD both use this factor.
    verifier_cost_factor: float = 0.01
    @property
    def verifier_cost_unit(self) -> float:
        return self.executor_cost_unit * self.verifier_cost_factor


DEFAULT_NVIDIA_NIM_MODELS = [
    # Small
    "meta/llama-3.2-3b-instruct",
    "nvidia/nemotron-mini-4b-instruct",
    # Medium
    "openai/gpt-oss-20b",
    # Large
    "meta/llama-3.1-70b-instruct",
    "nvidia/nemotron-3-ultra-550b-a55b",
    # Older / previously used pool members (kept; some may be flaky or EOL on NIM).
    "meta/llama-3.3-70b-instruct",
    "openai/gpt-oss-120b",
    "google/gemma-3n-e2b-it",
    "mistralai/mistral-medium-3.5-128b",
    "mistralai/mistral-large-3-675b-instruct-2512",
    "meta/llama-4-maverick-17b-128e-instruct",
    "deepseek-ai/deepseek-v4-flash",
]

DEFAULT_VERIFIER_MODELS = [
    "meta/llama-3.2-3b-instruct",
    "nvidia/nemotron-mini-4b-instruct",
    "openai/gpt-oss-20b",
    "meta/llama-3.1-70b-instruct",
    "nvidia/nemotron-3-ultra-550b-a55b",
    # Kept for alias/backward compatibility.
    "meta/llama-3.3-70b-instruct",
    "openai/gpt-oss-120b",
    "deepseek-ai/deepseek-v4-flash",
    "meta/llama-4-maverick-17b-128e-instruct",
    "mistralai/mistral-large-3-675b-instruct-2512",
    "mistralai/mistral-medium-3.5-128b",
]
DEFAULT_VERIFIER_MODEL = DEFAULT_VERIFIER_MODELS[0]

# OpenRouter chat/completions pool (IDs from https://openrouter.ai/models).
# 3 midsize + 3 large. Llama-3.2-3B and Gemma-3-4B were dropped after JOVE
# full-stream runs locked on 3B (optimistic verifier labels + cheap cost).
# Those two IDs remain in DEFAULT_OPENROUTER_CONFIGS for old-run lookup.
# qwen/qwen3.8-max-0902 is the fixed verifier only (not an executor arm).
# qwen/qwen3-235b-a22b-2507 remains in the catalog for old-run lookup.
DEFAULT_OPENROUTER_MODELS = [
    # Midsize
    "openai/gpt-oss-20b",
    "mistralai/mistral-small-3.2-24b-instruct",
    "qwen/qwen3-32b",
    # Large
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-oss-120b",
    "qwen/qwen3-vl-235b-a22b-thinking",
]
DEFAULT_OPENROUTER_PLANNER_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_OPENROUTER_VERIFIER_MODEL = "qwen/qwen3.8-max-0902"
DEFAULT_OPENROUTER_INFERENCE_MODEL = "qwen/qwen3-32b"
DEFAULT_OPENROUTER_VERIFIER_MODELS = [
    DEFAULT_OPENROUTER_VERIFIER_MODEL,
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen3-32b",
    "mistralai/mistral-small-3.2-24b-instruct",
    "openai/gpt-oss-20b",
    "meta-llama/llama-3.2-3b-instruct",
    "google/gemma-3-4b-it",
    DEFAULT_OPENROUTER_PLANNER_MODEL,
]

DEFAULT_NVIDIA_NIM_CONFIGS: Dict[str, APIConfig] = {
    "meta/llama-3.2-3b-instruct": APIConfig(
        model="meta/llama-3.2-3b-instruct",
        executor_cost_unit=3.0,
        safe_latency_prior=1.0,
    ),
    "nvidia/nemotron-mini-4b-instruct": APIConfig(
        model="nvidia/nemotron-mini-4b-instruct",
        executor_cost_unit=4.0,
        safe_latency_prior=0.8,
    ),
    "meta/llama-3.1-8b-instruct": APIConfig(
        model="meta/llama-3.1-8b-instruct",
        executor_cost_unit=8.0,
        safe_latency_prior=1.2,
    ),
    "openai/gpt-oss-20b": APIConfig(
        model="openai/gpt-oss-20b",
        executor_cost_unit=20.0,
        safe_latency_prior=1.5,
    ),
    "nvidia/nemotron-3-nano-30b-a3b": APIConfig(
        model="nvidia/nemotron-3-nano-30b-a3b",
        executor_cost_unit=30.0,
        safe_latency_prior=2.0,
    ),
    "nvidia/llama-3.3-nemotron-super-49b-v1": APIConfig(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        executor_cost_unit=49.0,
        safe_latency_prior=2.5,
    ),
    "meta/llama-3.1-70b-instruct": APIConfig(
        model="meta/llama-3.1-70b-instruct",
        executor_cost_unit=70.0,
        safe_latency_prior=2.5,
    ),
    "mistralai/mistral-nemotron": APIConfig(
        model="mistralai/mistral-nemotron",
        executor_cost_unit=40.0,
        safe_latency_prior=2.0,
    ),
    "meta/llama-3.3-70b-instruct": APIConfig(
        model="meta/llama-3.3-70b-instruct",
        executor_cost_unit=70.0,
        safe_latency_prior=2.5,
    ),
    "openai/gpt-oss-120b": APIConfig(
        model="openai/gpt-oss-120b",
        executor_cost_unit=120.0,
        safe_latency_prior=4.0,
    ),
    "google/gemma-3n-e2b-it": APIConfig(
        model="google/gemma-3n-e2b-it",
        executor_cost_unit=2.0,
        safe_latency_prior=1.5,
    ),
    "nvidia/nemotron-3-ultra-550b-a55b": APIConfig(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        executor_cost_unit=550.0,
        safe_latency_prior=8.0,
    ),
    "mistralai/mistral-medium-3.5-128b": APIConfig(
        model="mistralai/mistral-medium-3.5-128b",
        executor_cost_unit=128.0,
        safe_latency_prior=5.0,
    ),
    "deepseek-ai/deepseek-v4-flash": APIConfig(
        model="deepseek-ai/deepseek-v4-flash",
        executor_cost_unit=150.0,
        safe_latency_prior=1.0,
    ),
    "meta/llama-4-maverick-17b-128e-instruct": APIConfig(
        model="meta/llama-4-maverick-17b-128e-instruct",
        executor_cost_unit=17.0,
        safe_latency_prior=2.0,
    ),
    "mistralai/mistral-large-3-675b-instruct-2512": APIConfig(
        model="mistralai/mistral-large-3-675b-instruct-2512",
        executor_cost_unit=256.0,
        safe_latency_prior=4.0,
    ),
}

# Cost units follow nominal parameter scale (3, 4, 20, 24, 32, 70, 120).
# qwen3-235b remains in the catalog for verifier-cost lookup only.
DEFAULT_OPENROUTER_CONFIGS: Dict[str, APIConfig] = {
    "meta-llama/llama-3.2-3b-instruct": APIConfig(
        model="meta-llama/llama-3.2-3b-instruct",
        executor_cost_unit=3.0,
        safe_latency_prior=1.0,
    ),
    "google/gemma-3-4b-it": APIConfig(
        model="google/gemma-3-4b-it",
        executor_cost_unit=4.0,
        safe_latency_prior=0.9,
    ),
    "openai/gpt-oss-20b": APIConfig(
        model="openai/gpt-oss-20b",
        executor_cost_unit=20.0,
        safe_latency_prior=1.4,
    ),
    "mistralai/mistral-small-3.2-24b-instruct": APIConfig(
        model="mistralai/mistral-small-3.2-24b-instruct",
        executor_cost_unit=24.0,
        safe_latency_prior=1.6,
    ),
    "qwen/qwen3-32b": APIConfig(
        model="qwen/qwen3-32b",
        executor_cost_unit=32.0,
        safe_latency_prior=1.8,
    ),
    "google/gemini-2.5-flash-lite": APIConfig(
        model="google/gemini-2.5-flash-lite",
        executor_cost_unit=28.0,
        safe_latency_prior=1.2,
    ),
    "meta-llama/llama-3.3-70b-instruct": APIConfig(
        model="meta-llama/llama-3.3-70b-instruct",
        executor_cost_unit=70.0,
        safe_latency_prior=2.8,
    ),
    "openai/gpt-oss-120b": APIConfig(
        model="openai/gpt-oss-120b",
        executor_cost_unit=120.0,
        safe_latency_prior=4.0,
    ),
    # Live large arm (replaced mistral-large in the mid+large OpenRouter pool).
    "qwen/qwen3-vl-235b-a22b-thinking": APIConfig(
        model="qwen/qwen3-vl-235b-a22b-thinking",
        executor_cost_unit=235.0,
        safe_latency_prior=6.0,
    ),
    # Old-run / catalog-lookup: prior live large arm + Batch-API-only aliases.
    "mistralai/mistral-large": APIConfig(
        model="mistralai/mistral-large",
        executor_cost_unit=130.0,
        safe_latency_prior=4.5,
    ),
    "mistralai/mistral-large-2512:batch": APIConfig(
        model="mistralai/mistral-large",
        executor_cost_unit=130.0,
        safe_latency_prior=4.5,
    ),
    "mistralai/mistral-large-2512": APIConfig(
        model="mistralai/mistral-large",
        executor_cost_unit=130.0,
        safe_latency_prior=4.5,
    ),
    "qwen/qwen3-235b-a22b-2507": APIConfig(
        model="qwen/qwen3-235b-a22b-2507",
        executor_cost_unit=120.0,
        safe_latency_prior=3.5,
    ),
    # Verifier only. 2.4T-total MoE flagship; keep catalog unit above 235b.
    "qwen/qwen3.8-max-0902": APIConfig(
        model="qwen/qwen3.8-max-0902",
        executor_cost_unit=200.0,
        safe_latency_prior=5.0,
    ),
}

_NIM_PLANNER_DEFAULT = "meta/llama-3.3-70b-instruct"
_NIM_VERIFIER_DEFAULT = "meta/llama-3.2-3b-instruct"


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
