from __future__ import annotations

import unittest
from unittest.mock import Mock

from openai._constants import DEFAULT_MAX_RETRIES

from src.agent.model_reliability import (
    SDK_MAX_RETRIES,
    ModelCallContext,
    build_no_credentials_error,
    build_provider_error,
    build_success_audit,
    build_validation_error,
    legacy_error_message,
    request_headers,
    request_profile,
)


class ModelReliabilityTests(unittest.TestCase):
    def test_request_profiles_preserve_frozen_settings(self) -> None:
        thinking = request_profile("plan_action", thinking_model=True)
        standard_plan = request_profile(
            "plan_action",
            thinking_model=False,
        )
        standard_complete = request_profile(
            "complete_json",
            thinking_model=False,
        )

        self.assertEqual(thinking.timeout_seconds, 90.0)
        self.assertEqual(thinking.temperature, 1.0)
        self.assertEqual(thinking.max_tokens, 2048)
        self.assertEqual(standard_plan.timeout_seconds, 15.0)
        self.assertEqual(standard_plan.temperature, 0.1)
        self.assertEqual(standard_plan.max_tokens, 300)
        self.assertIsNone(standard_complete.timeout_seconds)
        self.assertEqual(standard_complete.temperature, 0.1)
        self.assertEqual(standard_complete.max_tokens, 220)
        self.assertEqual(SDK_MAX_RETRIES, DEFAULT_MAX_RETRIES)

    def test_success_audit_extracts_request_id_and_usage(self) -> None:
        profile = request_profile("plan_action", thinking_model=True)
        context = ModelCallContext.start(
            operation="plan_action",
            provider="kimi",
            model="kimi-k2.6",
            profile=profile,
        )
        usage = Mock()
        usage.model_dump.return_value = {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        }
        response = Mock(
            _request_id="provider-request-1",
            usage=usage,
        )

        audit = build_success_audit(context, response)

        self.assertEqual(audit["status"], "succeeded")
        self.assertEqual(
            audit["provider_request_id"],
            "provider-request-1",
        )
        self.assertEqual(audit["usage"]["total_tokens"], 14)
        self.assertEqual(audit["request_deadline_seconds"], 90.0)
        self.assertEqual(audit["per_attempt_timeout_seconds"], 90.0)
        self.assertEqual(audit["estimated_max_wall_time_seconds"], 270.0)
        self.assertEqual(audit["sdk_max_retries"], 2)
        self.assertEqual(audit["attempt_budget"], 3)
        self.assertEqual(len(audit["client_request_id"]), 32)
        self.assertEqual(
            request_headers(context),
            {"X-Client-Request-ID": audit["client_request_id"]},
        )

    def test_timeout_error_is_retryable_and_redacts_secret(self) -> None:
        profile = request_profile("plan_action", thinking_model=False)
        context = ModelCallContext.start(
            operation="plan_action",
            provider="openai_compatible",
            model="gpt-4o-mini",
            profile=profile,
        )

        audit = build_provider_error(
            context,
            TimeoutError("request failed for sk-secret123456789"),
        )

        self.assertEqual(audit["error"]["kind"], "timeout")
        self.assertTrue(audit["error"]["retryable"])
        self.assertNotIn(
            "sk-secret123456789",
            audit["error"]["message"],
        )
        self.assertIn("[redacted-api-key]", audit["error"]["message"])

    def test_status_error_like_exception_is_classified(self) -> None:
        profile = request_profile("plan_task", thinking_model=False)
        context = ModelCallContext.start(
            operation="plan_task",
            provider="test",
            model="test-model",
            profile=profile,
        )

        class StatusError(Exception):
            status_code = 503
            request_id = "provider-request-503"
            body = {
                "error": {
                    "type": "server_error",
                    "code": "overloaded",
                    "param": "messages",
                    "message": "temporarily unavailable sk-secret123456789",
                }
            }

        audit = build_provider_error(
            context,
            StatusError("temporarily unavailable"),
        )

        self.assertEqual(
            audit["error"]["kind"],
            "provider_status_error",
        )
        self.assertTrue(audit["error"]["retryable"])
        self.assertEqual(audit["error"]["status_code"], 503)
        self.assertEqual(audit["error"]["provider_code"], "overloaded")
        self.assertEqual(audit["error"]["provider_error"]["type"], "server_error")
        self.assertEqual(audit["error"]["provider_error"]["param"], "messages")
        self.assertNotIn(
            "sk-secret123456789",
            audit["error"]["provider_error"]["message"],
        )
        self.assertEqual(
            audit["provider_request_id"],
            "provider-request-503",
        )

    def test_validation_error_retains_usage_and_legacy_message(self) -> None:
        profile = request_profile("complete_json", thinking_model=False)
        context = ModelCallContext.start(
            operation="complete_json",
            provider="test",
            model="test-model",
            profile=profile,
        )
        response = Mock(
            _request_id="provider-validation",
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "total_tokens": 6,
            },
        )

        audit = build_validation_error(
            context,
            response=response,
            message="JSON response could not be decoded",
        )

        self.assertEqual(audit["status"], "failed")
        self.assertFalse(audit["error"]["retryable"])
        self.assertEqual(audit["usage"]["total_tokens"], 6)
        self.assertIn("invalid_response", legacy_error_message(audit))

    def test_no_credentials_error_uses_uniform_audit_shape(self) -> None:
        audit = build_no_credentials_error("plan_action")

        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["operation"], "plan_action")
        self.assertEqual(audit["error"]["kind"], "no_credentials")
        self.assertFalse(audit["error"]["retryable"])
        self.assertIn("no_credentials", legacy_error_message(audit))


if __name__ == "__main__":
    unittest.main()
