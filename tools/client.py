from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import requests

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_API_KEYS_FILE = "API_keys.txt"
DEFAULT_NVIDIA_NIM_API_KEY_NAME = "API_key1"
DEFAULT_OPENROUTER_API_KEY_NAME = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_HTTP_TIMEOUT = 60
PROVIDER_NVIDIA_NIM = "nvidia_nim"
PROVIDER_OPENROUTER = "openrouter"

FAILURE_RATE_LIMIT = "rate_limit"
FAILURE_UNSTABLE_SERVICE = "unstable_service"
FAILURE_FATAL_CONFIG = "fatal_config"
DEFAULT_RATE_LIMIT_MAX_RETRIES = 5
DEFAULT_RATE_LIMIT_BASE_DELAY_SECONDS = 2.0
DEFAULT_RATE_LIMIT_MAX_DELAY_SECONDS = 60.0

_RATE_LIMIT_TERMS = (
    "rate limit",
    "rate-limit",
    "ratelimit",
    "too many requests",
    "quota",
    "rpm",
    "tpm",
    "requests per minute",
    "tokens per minute",
)
_FATAL_CONFIG_TERMS = (
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "api key",
    "invalid model",
    "unknown model",
    "model not found",
    "does not exist",
    "not a valid model",
    "malformed",
    "invalid request",
    "bad request",
    "validation",
    "schema",
)
_TRANSIENT_TERMS = (
    "temporarily unavailable",
    "try again",
    "transient",
    "overloaded",
    "upstream",
    "timeout",
    "timed out",
    "connection",
    "service unavailable",
    "gateway",
)

# Optional code-level API keys for local experiments. Keep these empty in
# versioned code; prefer API_keys.txt or environment variables.
NVIDIA_NIM_API_KEY = ""
OPENROUTER_API_KEY = ""
OPENAI_API_KEY = ""


def detect_api_provider(base_url: str) -> str:
    normalized = (base_url or "").strip().lower()
    if "integrate.api.nvidia.com" in normalized or "build.nvidia.com" in normalized:
        return PROVIDER_NVIDIA_NIM
    if "openrouter.ai" in normalized:
        return PROVIDER_OPENROUTER
    return "openai_compatible"


def load_named_api_keys(path: str = DEFAULT_API_KEYS_FILE) -> Dict[str, str]:
    key_path = Path(path).expanduser()
    if not key_path.is_absolute():
        key_path = Path.cwd() / key_path
    if not key_path.exists():
        raise ValueError(f"API key file not found: {key_path}")

    keys: Dict[str, str] = {}
    for raw_line in key_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and value:
            keys[name] = value
    return keys


def _normalize_api_key_name(value: str) -> str:
    selected = str(value or "").strip()
    if not selected:
        return selected
    lowered = selected.lower()
    if lowered.isdigit():
        return f"API_key{lowered}"
    if lowered.startswith("key") and lowered[3:].isdigit():
        return f"API_key{lowered[3:]}"
    if lowered.startswith("api_key") and lowered[7:].isdigit():
        return f"API_key{lowered[7:]}"
    return selected


def resolve_named_api_key(api_key_name: str, api_keys_file: str = DEFAULT_API_KEYS_FILE) -> str:
    selected = _normalize_api_key_name(api_key_name)
    keys = load_named_api_keys(api_keys_file)
    if selected in keys:
        return keys[selected]
    available = ", ".join(sorted(keys)) or "<none>"
    raise ValueError(
        f"API key name {api_key_name!r} was not found in {api_keys_file!r}. "
        f"Available key names: {available}"
    )


def resolve_api_key(
    base_url: str,
    api_key_name: Optional[str] = None,
    api_keys_file: str = DEFAULT_API_KEYS_FILE,
) -> str:
    provider = detect_api_provider(base_url)
    file_keys: Dict[str, str] = {}
    try:
        file_keys = load_named_api_keys(api_keys_file)
    except ValueError:
        file_keys = {}

    if provider == PROVIDER_NVIDIA_NIM:
        selected_name = api_key_name or os.environ.get("NVIDIA_NIM_API_KEY_NAME")
        if selected_name:
            return resolve_named_api_key(selected_name, api_keys_file)
        key_name, code_value = "NVIDIA_NIM_API_KEY", NVIDIA_NIM_API_KEY
    elif provider == PROVIDER_OPENROUTER:
        selected_name = _normalize_api_key_name(
            api_key_name or os.environ.get("OPENROUTER_API_KEY_NAME") or ""
        )
        if selected_name and selected_name in file_keys:
            return file_keys[selected_name]
        if DEFAULT_OPENROUTER_API_KEY_NAME in file_keys:
            return file_keys[DEFAULT_OPENROUTER_API_KEY_NAME]
        if selected_name:
            # Named key was requested but missing from the file.
            return resolve_named_api_key(selected_name, api_keys_file)
        key_name, code_value = DEFAULT_OPENROUTER_API_KEY_NAME, OPENROUTER_API_KEY
    else:
        key_name, code_value = "OPENAI_API_KEY", OPENAI_API_KEY

    if code_value:
        return code_value
    env_value = os.environ.get(key_name)
    if env_value:
        return env_value
    if key_name in file_keys:
        return file_keys[key_name]
    raise ValueError(
        f"Define {key_name!r} in {api_keys_file!r}, as an environment variable, "
        f"or via --api-key-name for provider {provider!r}."
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def parse_retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return max(float(raw), 0.0)
    except ValueError:
        pass
    try:
        retry_time = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_time.tzinfo is None:
        retry_time = retry_time.replace(tzinfo=timezone.utc)
    return max((retry_time - datetime.now(timezone.utc)).total_seconds(), 0.0)


def classify_api_failure(
    status_code: Optional[int] = None,
    detail: str = "",
    retry_after: Optional[str] = None,
    exception: Optional[BaseException] = None,
) -> str:
    """Classify provider/API failures for retry and experiment accounting."""
    if exception is not None:
        if isinstance(exception, (requests.Timeout, requests.ConnectionError)):
            return FAILURE_UNSTABLE_SERVICE
        if isinstance(exception, requests.RequestException):
            return FAILURE_UNSTABLE_SERVICE

    lowered = str(detail or "").lower()
    if status_code == 429 or retry_after or _contains_any(lowered, _RATE_LIMIT_TERMS):
        return FAILURE_RATE_LIMIT
    if status_code in {401, 403}:
        return FAILURE_FATAL_CONFIG
    if status_code is not None and status_code >= 500:
        return FAILURE_UNSTABLE_SERVICE
    if status_code in {408, 409, 425}:
        return FAILURE_UNSTABLE_SERVICE
    if status_code is not None and 400 <= status_code < 500:
        return FAILURE_FATAL_CONFIG
    if _contains_any(lowered, _TRANSIENT_TERMS):
        return FAILURE_UNSTABLE_SERVICE
    if _contains_any(lowered, _FATAL_CONFIG_TERMS):
        return FAILURE_FATAL_CONFIG
    return FAILURE_UNSTABLE_SERVICE


@dataclass
class ApiCallFailure(Exception):
    category: str
    model: str
    stage: str
    message: str
    status_code: Optional[int] = None
    retry_count: int = 0
    latency_seconds: float = 0.0
    response_detail: Optional[str] = None
    final_action: str = "raise"

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "model": self.model,
            "failure_category": self.category,
            "status_code": self.status_code,
            "retry_count": self.retry_count,
            "final_action": self.final_action,
            "message": self.message,
        }


@dataclass
class ChatResult:
    content: Optional[str]
    latency_seconds: float
    usage: Dict[str, Any]
    raw: Dict[str, Any]
    reasoning_content: Optional[str] = None
    message: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None
    retry_count: int = 0
    failure_category: Optional[str] = None
    failure_message: Optional[str] = None

    @property
    def total_tokens(self) -> Optional[float]:
        value = self.usage.get("total_tokens") if isinstance(self.usage, dict) else None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class OpenAICompatibleClient:
    """Small HTTP client for OpenAI-compatible chat completion providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = NVIDIA_NIM_BASE_URL,
        app_name: str = "bi-role-api-allocation",
        default_timeout: int = DEFAULT_HTTP_TIMEOUT,
        min_request_interval_seconds: float = 0.0,
        rate_limit_max_retries: int = DEFAULT_RATE_LIMIT_MAX_RETRIES,
        rate_limit_base_delay_seconds: float = DEFAULT_RATE_LIMIT_BASE_DELAY_SECONDS,
        rate_limit_max_delay_seconds: float = DEFAULT_RATE_LIMIT_MAX_DELAY_SECONDS,
        api_key_name: Optional[str] = None,
        api_keys_file: str = DEFAULT_API_KEYS_FILE,
    ) -> None:
        self.base_url = base_url
        self.provider = detect_api_provider(base_url)
        self.api_key_name = _normalize_api_key_name(api_key_name or "") or None
        self.api_keys_file = api_keys_file
        self.api_key = api_key or resolve_api_key(base_url, api_key_name=api_key_name, api_keys_file=api_keys_file)
        self.default_timeout = float(os.environ.get("API_RECRUITER_HTTP_TIMEOUT", str(default_timeout)))
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == PROVIDER_OPENROUTER:
            self.headers["HTTP-Referer"] = "https://localhost"
            self.headers["X-Title"] = app_name
        self._thread_local = threading.local()
        self.min_request_interval_seconds = max(float(min_request_interval_seconds), 0.0)
        self.rate_limit_max_retries = max(int(rate_limit_max_retries), 0)
        self.rate_limit_base_delay_seconds = max(float(rate_limit_base_delay_seconds), 0.0)
        self.rate_limit_max_delay_seconds = max(float(rate_limit_max_delay_seconds), 0.0)
        self._rate_lock = threading.Lock()
        self._next_request_time = 0.0

    def _session_for_thread(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _wait_for_rate_limit(self) -> float:
        """Reserve a start slot under lock; sleep outside so threads can overlap HTTP."""
        if self.min_request_interval_seconds <= 0.0:
            return 0.0
        with self._rate_lock:
            now = time.monotonic()
            scheduled = max(now, self._next_request_time)
            self._next_request_time = scheduled + self.min_request_interval_seconds
            waited = scheduled - now
        if waited > 0.0:
            time.sleep(waited)
        return waited

    def _post_payload(self, payload: Dict[str, Any]) -> requests.Response:
        return self._session_for_thread().post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=self.default_timeout,
        )

    @staticmethod
    def _error_detail(resp: requests.Response) -> str:
        try:
            data = resp.json()
        except ValueError:
            return resp.text.strip() or "<empty response body>"
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            metadata = error.get("metadata")
            raw = metadata.get("raw") if isinstance(metadata, dict) else None
            return str(raw or error.get("message") or data)
        return str(data)

    @staticmethod
    def _message_field_to_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts) if parts else None
        return str(value)

    @classmethod
    def _extract_chat_result(
        cls,
        data: Dict[str, Any],
        latency_seconds: float,
        status_code: Optional[int] = None,
        retry_count: int = 0,
    ) -> ChatResult:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"Chat completion response has no choices: {data}")
        first_choice = choices[0]
        message = first_choice.get("message") if isinstance(first_choice, dict) else None
        if not isinstance(message, dict):
            message = {}
        content = cls._message_field_to_text(message.get("content"))
        reasoning_content = cls._message_field_to_text(
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("thinking")
        )
        if content is None and reasoning_content is not None:
            content = reasoning_content
        usage = data.get("usage") or {}
        return ChatResult(
            content=content,
            reasoning_content=reasoning_content,
            message=message,
            latency_seconds=latency_seconds,
            usage=usage,
            raw=data,
            status_code=status_code,
            retry_count=retry_count,
        )

    def _rate_limit_delay(self, resp: requests.Response, retry_count: int) -> float:
        retry_after = parse_retry_after_seconds(resp.headers.get("Retry-After"))
        if retry_after is not None:
            return min(retry_after, self.rate_limit_max_delay_seconds)
        delay = self.rate_limit_base_delay_seconds * (2 ** max(retry_count - 1, 0))
        if self.rate_limit_max_delay_seconds > 0.0:
            delay = min(delay, self.rate_limit_max_delay_seconds)
        return max(delay, 0.0)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        call_label: Optional[str] = None,
    ) -> ChatResult:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_payload:
            payload.update(extra_payload)

        label = call_label or "api_call"
        retry_count = 0
        while True:
            waited = self._wait_for_rate_limit()
            if waited > 0.0:
                print(
                    f"[api_call:wait] stage={label} model={model} waited={waited:.2f}s "
                    f"interval={self.min_request_interval_seconds:.2f}s",
                    flush=True,
                )
            print(f"[api_call:start] stage={label} model={model} retry={retry_count}", flush=True)
            start = time.perf_counter()
            try:
                resp = self._post_payload(payload)
            except requests.RequestException as exc:
                latency = time.perf_counter() - start
                category = classify_api_failure(exception=exc)
                print(
                    f"[api_call:error] stage={label} model={model} category={category} "
                    f"retry_count={retry_count} latency={latency:.2f}s error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                raise ApiCallFailure(
                    category=category,
                    model=model,
                    stage=label,
                    message=f"{type(exc).__name__}: {exc}",
                    retry_count=retry_count,
                    latency_seconds=latency,
                    final_action="raise",
                ) from exc

            latency = time.perf_counter() - start
            print(
                f"[api_call:end] stage={label} model={model} status={resp.status_code} "
                f"retry={retry_count} latency={latency:.2f}s",
                flush=True,
            )
            if resp.ok:
                try:
                    data = resp.json()
                    result = self._extract_chat_result(data, latency, status_code=resp.status_code, retry_count=retry_count)
                except ValueError as exc:
                    detail = resp.text.strip()[:1000]
                    message = f"Malformed provider response: {exc}"
                    print(
                        f"[api_call:error] stage={label} model={model} category={FAILURE_UNSTABLE_SERVICE} "
                        f"status={resp.status_code} retry_count={retry_count} detail={message}",
                        flush=True,
                    )
                    raise ApiCallFailure(
                        category=FAILURE_UNSTABLE_SERVICE,
                        model=model,
                        stage=label,
                        message=message,
                        status_code=resp.status_code,
                        retry_count=retry_count,
                        latency_seconds=latency,
                        response_detail=detail,
                        final_action="raise",
                    ) from exc
                if not ((result.content or "").strip() or (result.reasoning_content or "").strip()):
                    message = "Provider response had no textual content."
                    print(
                        f"[api_call:error] stage={label} model={model} category={FAILURE_UNSTABLE_SERVICE} "
                        f"status={resp.status_code} retry_count={retry_count} detail={message}",
                        flush=True,
                    )
                    raise ApiCallFailure(
                        category=FAILURE_UNSTABLE_SERVICE,
                        model=model,
                        stage=label,
                        message=message,
                        status_code=resp.status_code,
                        retry_count=retry_count,
                        latency_seconds=latency,
                        response_detail=str(result.raw)[:1000],
                        final_action="raise",
                    )
                return result

            detail = self._error_detail(resp)
            retry_after = resp.headers.get("Retry-After")
            category = classify_api_failure(status_code=resp.status_code, detail=detail, retry_after=retry_after)
            print(
                f"[api_call:error] stage={label} model={model} category={category} "
                f"status={resp.status_code} retry_count={retry_count} detail={detail}",
                flush=True,
            )
            if category == FAILURE_RATE_LIMIT and retry_count < self.rate_limit_max_retries:
                retry_count += 1
                delay = self._rate_limit_delay(resp, retry_count)
                print(
                    f"[api_call:retry] stage={label} model={model} category={category} "
                    f"retry={retry_count}/{self.rate_limit_max_retries} delay={delay:.2f}s",
                    flush=True,
                )
                if delay > 0.0:
                    time.sleep(delay)
                continue
            raise ApiCallFailure(
                category=category,
                model=model,
                stage=label,
                message=f"{resp.status_code} error for model {model!r}: {detail}",
                status_code=resp.status_code,
                retry_count=retry_count,
                latency_seconds=latency,
                response_detail=detail,
                final_action="raise",
            )
