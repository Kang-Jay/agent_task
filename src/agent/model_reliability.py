from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError


# openai==2.21.0 defaults to two SDK-managed retries. Keeping the value
# explicit makes the existing behavior auditable and lets tests detect drift.
SDK_MAX_RETRIES = 2

THINKING_TIMEOUT_SECONDS = 90.0
STANDARD_PLANNER_TIMEOUT_SECONDS = 15.0
THINKING_TEMPERATURE = 1.0
STANDARD_TEMPERATURE = 0.1
THINKING_MAX_TOKENS = 2048
STANDARD_PLANNER_MAX_TOKENS = 300
STANDARD_COMPLETE_MAX_TOKENS = 220


@dataclass(frozen=True)
class RequestProfile:
    timeout_seconds: float | None
    temperature: float
    max_tokens: int
    sdk_max_retries: int = SDK_MAX_RETRIES

    @property
    def attempt_budget(self) -> int:
        return self.sdk_max_retries + 1

    @property
    def estimated_max_wall_time_seconds(self) -> float | None:
        if self.timeout_seconds is None:
            return None
        return self.timeout_seconds * self.attempt_budget


def request_profile(operation: str, *, thinking_model: bool) -> RequestProfile:
    """Return the pre-existing request settings without changing their values."""
    if thinking_model:
        return RequestProfile(
            timeout_seconds=THINKING_TIMEOUT_SECONDS,
            temperature=THINKING_TEMPERATURE,
            max_tokens=THINKING_MAX_TOKENS,
        )
    if operation in {"plan_task", "plan_action"}:
        return RequestProfile(
            timeout_seconds=STANDARD_PLANNER_TIMEOUT_SECONDS,
            temperature=STANDARD_TEMPERATURE,
            max_tokens=STANDARD_PLANNER_MAX_TOKENS,
        )
    if operation == "complete_json":
        return RequestProfile(
            timeout_seconds=None,
            temperature=STANDARD_TEMPERATURE,
            max_tokens=STANDARD_COMPLETE_MAX_TOKENS,
        )
    raise ValueError(f"unsupported model operation: {operation}")


@dataclass(frozen=True)
class ModelCallContext:
    client_request_id: str
    operation: str
    provider: str
    model: str
    profile: RequestProfile
    started_at: str
    started_monotonic: float

    @classmethod
    def start(
        cls,
        *,
        operation: str,
        provider: str,
        model: str,
        profile: RequestProfile,
    ) -> "ModelCallContext":
        return cls(
            client_request_id=uuid4().hex,
            operation=operation,
            provider=provider,
            model=model,
            profile=profile,
            started_at=datetime.now(timezone.utc).isoformat(),
            started_monotonic=perf_counter(),
        )


def build_success_audit(
    context: ModelCallContext,
    response: Any,
) -> dict[str, Any]:
    return {
        **_base_audit(context),
        "status": "succeeded",
        "provider_request_id": _provider_request_id(response),
        "usage": _usage_payload(response),
    }


def request_headers(context: ModelCallContext) -> dict[str, str]:
    return {"X-Client-Request-ID": context.client_request_id}


def build_provider_error(
    context: ModelCallContext,
    exc: Exception,
) -> dict[str, Any]:
    status_code = _status_code(exc)
    error_kind, retryable = _classify_error(exc, status_code)
    provider_request_id = (
        getattr(exc, "request_id", None)
        or _provider_request_id(getattr(exc, "response", None))
    )
    error_code = _provider_error_code(exc)
    provider_error = _provider_error_payload(exc)
    return {
        **_base_audit(context),
        "status": "failed",
        "provider_request_id": provider_request_id,
        "usage": None,
        "error": {
            "kind": error_kind,
            "type": type(exc).__name__,
            "message": _redact_message(str(exc)),
            "status_code": status_code,
            "provider_code": error_code,
            "provider_error": provider_error,
            "retryable": retryable,
        },
    }


def build_validation_error(
    context: ModelCallContext,
    *,
    message: str,
    response: Any | None = None,
    kind: str = "invalid_response",
) -> dict[str, Any]:
    return {
        **_base_audit(context),
        "status": "failed",
        "provider_request_id": _provider_request_id(response),
        "usage": _usage_payload(response),
        "error": {
            "kind": kind,
            "type": "ModelResponseValidationError",
            "message": _redact_message(message),
            "status_code": None,
            "provider_code": None,
            "provider_error": None,
            "retryable": False,
        },
    }


def build_skipped_provider_error(
    *,
    operation: str,
    provider: str,
    model: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "client_request_id": None,
        "provider_request_id": None,
        "operation": operation,
        "provider": provider,
        "model": model,
        "started_at": None,
        "latency_ms": 0.0,
        "request_deadline_seconds": None,
        "retry_owner": "openai_sdk",
        "sdk_max_retries": SDK_MAX_RETRIES,
        "attempt_budget": SDK_MAX_RETRIES + 1,
        "usage": None,
        "error": {
            "kind": "unsupported_capability",
            "type": "ProviderSkipped",
            "message": reason,
            "status_code": None,
            "provider_code": None,
            "provider_error": None,
            "retryable": False,
        },
    }


def build_no_credentials_error(operation: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "client_request_id": None,
        "provider_request_id": None,
        "operation": operation,
        "provider": "none",
        "model": None,
        "started_at": None,
        "latency_ms": 0.0,
        "request_deadline_seconds": None,
        "per_attempt_timeout_seconds": None,
        "estimated_max_wall_time_seconds": None,
        "retry_owner": "openai_sdk",
        "sdk_max_retries": SDK_MAX_RETRIES,
        "attempt_budget": SDK_MAX_RETRIES + 1,
        "usage": None,
        "error": {
            "kind": "no_credentials",
            "type": "NoModelCredentials",
            "message": "API key not available",
            "status_code": None,
            "provider_code": None,
            "provider_error": None,
            "retryable": False,
        },
    }


def legacy_error_message(audit: dict[str, Any]) -> str:
    error = audit.get("error") or {}
    provider = audit.get("provider") or "unknown_provider"
    kind = error.get("kind") or "provider_error"
    message = error.get("message") or "model call failed"
    return f"{provider}: {kind}: {message}"


def _base_audit(context: ModelCallContext) -> dict[str, Any]:
    return {
        "client_request_id": context.client_request_id,
        "operation": context.operation,
        "provider": context.provider,
        "model": context.model,
        "started_at": context.started_at,
        "latency_ms": round(
            max(0.0, perf_counter() - context.started_monotonic) * 1000.0,
            3,
        ),
        "request_deadline_seconds": context.profile.timeout_seconds,
        "per_attempt_timeout_seconds": context.profile.timeout_seconds,
        "estimated_max_wall_time_seconds": (
            context.profile.estimated_max_wall_time_seconds
        ),
        "retry_owner": "openai_sdk",
        "sdk_max_retries": context.profile.sdk_max_retries,
        "attempt_budget": context.profile.attempt_budget,
    }


def _provider_request_id(response: Any) -> str | None:
    if response is None:
        return None
    for attribute in ("_request_id", "request_id"):
        value = getattr(response, attribute, None)
        if value:
            return str(value)
    headers = getattr(response, "headers", None)
    if headers:
        for key in ("x-request-id", "request-id", "x-trace-id"):
            value = headers.get(key)
            if value:
                return str(value)
    return None


def _usage_payload(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None) if response is not None else None
    if usage is None:
        return None
    if isinstance(usage, dict):
        return dict(usage)
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(exclude_none=True)
        return dict(payload) if isinstance(payload, dict) else None
    payload: dict[str, Any] = {}
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        value = getattr(usage, name, None)
        if value is not None:
            payload[name] = value
    return payload or None


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _provider_error_code(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("code") is not None:
            return str(error["code"])
    return None


def _provider_error_payload(exc: Exception) -> dict[str, Any] | None:
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if not isinstance(error, dict):
        return None
    payload: dict[str, Any] = {}
    for key in ("type", "code", "param", "message"):
        value = error.get(key)
        if value is not None:
            payload[key] = _redact_message(str(value))
    return payload or None


def _classify_error(
    exc: Exception,
    status_code: int | None,
) -> tuple[str, bool]:
    if isinstance(exc, APITimeoutError) or isinstance(exc, TimeoutError):
        return "timeout", True
    if isinstance(exc, RateLimitError) or status_code == 429:
        return "rate_limit", True
    if isinstance(exc, APIConnectionError):
        return "connection_error", True
    if isinstance(exc, APIStatusError):
        if status_code in {408, 409} or (status_code is not None and status_code >= 500):
            return "provider_status_error", True
        if status_code in {401, 403}:
            return "authentication_error", False
        return "provider_status_error", False
    if status_code in {408, 409, 429} or (
        status_code is not None and status_code >= 500
    ):
        return "provider_status_error", True
    return "provider_error", False


def _redact_message(message: str) -> str:
    redacted = re.sub(
        r"\bsk-[A-Za-z0-9_-]{8,}\b",
        "[redacted-api-key]",
        message,
    )
    redacted = re.sub(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*\b",
        "Bearer [redacted]",
        redacted,
    )
    return redacted[:240]
