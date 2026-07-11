from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.agent.model_reliability import (
    ModelCallContext,
    build_no_credentials_error,
    build_provider_error,
    build_skipped_provider_error,
    build_success_audit,
    build_validation_error,
    legacy_error_message,
    request_headers,
    request_profile,
)


ROOT = Path(__file__).resolve().parents[2]
API_KEY_PATH = ROOT / "apikey.txt"


@dataclass(frozen=True)
class ApiCredential:
    provider: str
    api_key: str
    base_url: str | None = None
    model: str | None = None


def load_credentials(path: Path = API_KEY_PATH) -> list[ApiCredential]:
    """Load credentials from file or environment variables.

    Priority: environment variables > apikey.txt file
    """
    credentials: list[ApiCredential] = []

    # Try environment variables first
    env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MODEL_API_KEY")
    env_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("MODEL_BASE_URL")
    env_model = os.environ.get("MODEL_NAME")

    if env_key:
        provider = "env_openai" if not env_base_url else "env_custom"
        credentials.append(ApiCredential(
            provider=provider,
            api_key=env_key,
            base_url=env_base_url,
            model=env_model or "gpt-4o-mini"
        ))

    # Then try apikey.txt if exists
    if path.exists():
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        pending_label: str | None = None
        for line in lines:
            if line.endswith(":") and not line.lower().startswith("sk-"):
                pending_label = line[:-1].strip().lower()
                continue
            if "=" in line:
                name, value = line.split("=", 1)
                provider = name.strip().lower()
                api_key = value.strip()
            elif ":" in line and not line.lower().startswith("sk-"):
                name, value = line.split(":", 1)
                provider = name.strip().lower()
                api_key = value.strip()
            else:
                provider = (pending_label or "openai").lower()
                api_key = line
            if api_key.lower().startswith("sk-"):
                credentials.append(_credential_for(provider, api_key))
                pending_label = None

    return credentials


def _credential_for(provider: str, api_key: str) -> ApiCredential:
    normalized = provider.replace(" ", "").replace("_", "").lower()
    if "kimi" in normalized or "moonshot" in normalized:
        return ApiCredential(
            provider="kimi",
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            model="kimi-k2.6",
        )
    if "deepseek" in normalized:
        return ApiCredential(provider="deepseek", api_key=api_key, base_url="https://api.deepseek.com/v1", model="deepseek-chat")
    return ApiCredential(provider="openai_compatible", api_key=api_key, model="gpt-4o-mini")


class ModelAdapter:
    def __init__(self, credentials: list[ApiCredential] | None = None):
        self.credentials = credentials if credentials is not None else load_credentials()

    def available(self) -> bool:
        return bool(self.credentials)

    def audit(self) -> dict[str, Any]:
        return {
            "available": self.available(),
            "providers": [
                {
                    "provider": credential.provider,
                    "base_url": credential.base_url or "default",
                    "model": credential.model,
                    "key_length": len(credential.api_key),
                }
                for credential in self.credentials
            ],
        }

    def plan_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create one global task plan before step-level action planning."""
        if not self.available():
            audit = build_no_credentials_error("plan_task")
            return {
                "error": "no_credentials",
                "provider_errors": [audit],
                "fallback_reason": "API key not available",
            }

        system_prompt = """You are an embodied AI2-THOR global task planner.
Return ONLY valid JSON with these fields:
- task_summary: one concise sentence
- ordered_subgoal_ids: every supplied semantic subgoal id exactly once
- failure_policy: one concise factual recovery policy

Do not add or remove subgoals.
Do not claim an action already succeeded.
Respect limitations and approximate completion modes.
Do not output hidden reasoning."""

        task_contract = payload.get("task_contract", {})
        layered_memories = payload.get("layered_memories", {})
        prompt = f"""Instruction: {payload.get('instruction', '')}
Observation Summary: {payload.get('observation_summary', '')}
Task Contract:
{json.dumps(task_contract, ensure_ascii=False)}
Relevant Layered Memory With Evidence:
{json.dumps(layered_memories, ensure_ascii=False)}
AI2-THOR Environment Context:
{json.dumps(payload.get('environment_context', {}), ensure_ascii=False)}

Order all supplied subgoal ids into a complete executable global plan."""
        errors: list[str] = []
        provider_errors: list[dict[str, Any]] = []
        require_vision = bool(payload.get("require_vision", False))
        for credential in self.credentials:
            supports_vision = self._supports_vision(credential)
            if require_vision and not supports_vision:
                audit = build_skipped_provider_error(
                    operation="plan_task",
                    provider=credential.provider,
                    model=credential.model or "gpt-4o-mini",
                    reason="skipped because visual input is required",
                )
                provider_errors.append(audit)
                errors.append(legacy_error_message(audit))
                continue
            model_name = credential.model or "gpt-4o-mini"
            profile = request_profile(
                "plan_task",
                thinking_model=self._is_thinking_model(model_name),
            )
            context = ModelCallContext.start(
                operation="plan_task",
                provider=credential.provider,
                model=model_name,
                profile=profile,
            )
            response: Any | None = None
            try:
                client = OpenAI(
                    api_key=credential.api_key,
                    base_url=credential.base_url,
                    timeout=profile.timeout_seconds,
                    max_retries=profile.sdk_max_retries,
                )
                user_content: str | list[dict[str, Any]] = prompt
                observation_image = payload.get("observation_image")
                target_crop = payload.get("target_crop")
                if supports_vision and observation_image:
                    user_content = [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": observation_image,
                                "detail": "low",
                            },
                        },
                    ]
                    if target_crop:
                        user_content.extend(
                            [
                                {
                                    "type": "text",
                                    "text": "The next image is the user-selected target reference.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": target_crop,
                                        "detail": "low",
                                    },
                                },
                            ]
                        )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=profile.temperature,
                    max_tokens=profile.max_tokens,
                    extra_headers=request_headers(context),
                    response_format=(
                        {"type": "json_object"}
                        if credential.provider != "deepseek"
                        else None
                    ),
                )
                result = json.loads(response.choices[0].message.content or "{}")
                ordered_ids = result.get("ordered_subgoal_ids")
                if not isinstance(ordered_ids, list) or not all(
                    isinstance(item, str) and item for item in ordered_ids
                ):
                    audit = build_validation_error(
                        context,
                        response=response,
                        message="invalid ordered_subgoal_ids",
                    )
                    provider_errors.append(audit)
                    errors.append(legacy_error_message(audit))
                    continue
                result["provider_used"] = credential.provider
                result["model_used"] = model_name
                result["vision_input_used"] = isinstance(user_content, list)
                result["model_call"] = build_success_audit(context, response)
                return result
            except json.JSONDecodeError as exc:
                audit = build_validation_error(
                    context,
                    response=response,
                    message=f"JSON decode error: {str(exc)[:100]}",
                )
                provider_errors.append(audit)
                errors.append(legacy_error_message(audit))
            except Exception as exc:
                audit = build_provider_error(context, exc)
                provider_errors.append(audit)
                errors.append(legacy_error_message(audit))
        return {
            "error": "all_model_calls_failed",
            "errors": errors,
            "provider_errors": provider_errors,
            "fallback_reason": (
                "No configured vision model completed global task planning"
                if require_vision
                else "Global task planning failed"
            ),
        }

    def plan_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Plan next action using model.

        Input payload should contain:
        - instruction: str
        - observation_summary: str
        - candidates: list
        - confidence: float
        - memory_summary: str
        - layered_memories: object, spatial, task, failure, skill and episode
        - negative_memory: list
        - explored_regions: dict
        - retrieved_hints: list
        - allowed_actions: list
        - terminal_actions: list
        - current_step: int
        - max_steps: int
        - observation_image: current robot RGB as a data URL
        - target_crop: optional clicked target crop as a data URL

        Output:
        {
            "thought_summary": "short explainable summary",
            "action": {"type": "TURN_RIGHT", "args": {}},
            "skill_call": {...},
            "confidence": 0.42,
            "stop_reason": null
        }
        """
        if not self.available():
            audit = build_no_credentials_error("plan_action")
            return {
                "error": "no_credentials",
                "provider_errors": [audit],
                "fallback_reason": "API key not available"
            }

        # Build prompt
        system_prompt = """You are an embodied AI2-THOR task planner.
Return ONLY valid JSON with these fields:
- thought_summary: brief explanation (1 sentence)
- action: {type: ACTION_NAME, args: {}}
- confidence: float 0-1
- stop_reason: null or string
- task_progress: brief factual progress summary

ACTION_NAME must be from the allowed list.
Use the supplied action parameter schema exactly.
Respect the task plan and unsupported-capability warnings.
Do not choose STOP or Done unless completion_status.complete is true.
Seeing the target is not task completion for navigation or interaction tasks.
Never claim a physical action succeeded before environment feedback confirms it.
Do not output hidden reasoning, only the final short summary."""

        user_prompt = self._build_planner_prompt(payload)

        errors: list[str] = []
        provider_errors: list[dict[str, Any]] = []
        require_vision = bool(payload.get("require_vision", False))
        for credential in self.credentials:
            supports_vision = self._supports_vision(credential)
            if require_vision and not supports_vision:
                audit = build_skipped_provider_error(
                    operation="plan_action",
                    provider=credential.provider,
                    model=credential.model or "gpt-4o-mini",
                    reason="skipped because visual input is required",
                )
                provider_errors.append(audit)
                errors.append(legacy_error_message(audit))
                continue
            model_name = credential.model or "gpt-4o-mini"
            profile = request_profile(
                "plan_action",
                thinking_model=self._is_thinking_model(model_name),
            )
            context = ModelCallContext.start(
                operation="plan_action",
                provider=credential.provider,
                model=model_name,
                profile=profile,
            )
            response: Any | None = None
            try:
                client = OpenAI(
                    api_key=credential.api_key,
                    base_url=credential.base_url,
                    timeout=profile.timeout_seconds,
                    max_retries=profile.sdk_max_retries,
                )
                user_content = self._build_user_content(
                    payload,
                    include_images=supports_vision,
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=profile.temperature,
                    max_tokens=profile.max_tokens,
                    extra_headers=request_headers(context),
                    response_format={"type": "json_object"} if credential.provider != "deepseek" else None,
                )
                content = response.choices[0].message.content or "{}"
                result = json.loads(content)

                # Validate output
                if "action" not in result or "type" not in result.get("action", {}):
                    audit = build_validation_error(
                        context,
                        response=response,
                        message="missing action.type in response",
                    )
                    provider_errors.append(audit)
                    errors.append(legacy_error_message(audit))
                    continue

                # Add skill_call if not present
                if "skill_call" not in result:
                    action_type = result["action"]["type"]
                    result["skill_call"] = {
                        "name": action_type,
                        "args": result["action"].get("args", {}),
                        "preconditions": [],
                        "expected_observation": f"Execute {action_type}"
                    }

                result["provider_used"] = credential.provider
                result["model_used"] = credential.model or "gpt-4o-mini"
                result["vision_input_used"] = isinstance(user_content, list)
                result["model_call"] = build_success_audit(context, response)
                return result

            except json.JSONDecodeError as exc:
                audit = build_validation_error(
                    context,
                    response=response,
                    message=f"JSON decode error: {str(exc)[:100]}",
                )
                provider_errors.append(audit)
                errors.append(legacy_error_message(audit))
            except Exception as exc:
                audit = build_provider_error(context, exc)
                provider_errors.append(audit)
                errors.append(legacy_error_message(audit))

        return {
            "error": "all_model_calls_failed",
            "errors": errors,
            "provider_errors": provider_errors,
            "fallback_reason": (
                "No configured vision model completed the request"
                if require_vision
                else "Model API call failed"
            ),
        }

    def _build_planner_prompt(self, payload: dict[str, Any]) -> str:
        """Build standardized prompt from payload."""
        instruction = payload.get("instruction", "")
        observation_summary = payload.get("observation_summary", "")
        confidence = payload.get("confidence", 0.0)
        memory_summary = payload.get("memory_summary", "")
        retrieved_hints = payload.get("retrieved_hints", [])
        allowed_actions = payload.get("allowed_actions", [])
        terminal_actions = payload.get("terminal_actions", [])
        current_step = payload.get("current_step", 0)
        max_steps = payload.get("max_steps", 20)
        negative_memory = payload.get("negative_memory", [])
        action_specs = payload.get("action_specs", [])
        environment_context = payload.get("environment_context", {})
        task_plan = payload.get("task_plan", {})
        execution_plan = payload.get("execution_plan", {})
        completion_status = payload.get("completion_status", {})
        episodic_memories = payload.get("episodic_memories", [])
        layered_memories = payload.get("layered_memories", {})
        episodic_text = "\n".join(
            (
                f"- action={memory.get('action', 'UNKNOWN')}; "
                f"success={memory.get('action_success', False)}; "
                f"region={memory.get('region') or 'unknown'}; "
                f"lesson={memory.get('lesson', '')}"
            )
            for memory in episodic_memories
        ) or "none"

        prompt = f"""Task: {instruction}

Current Observation: {observation_summary}
Current Confidence: {confidence:.3f}
Step: {current_step}/{max_steps}

Memory: {memory_summary}
Hints: {', '.join(retrieved_hints) if retrieved_hints else 'none'}
Negative Memory: {', '.join(negative_memory[-3:]) if negative_memory else 'none'}
Relevant Executed Episodes:
{episodic_text}
Layered Memory With Evidence:
{json.dumps(layered_memories, ensure_ascii=False)}

Allowed Actions: {', '.join(allowed_actions)}
Action Schemas:
{json.dumps(action_specs, ensure_ascii=False)}
AI2-THOR Environment Context:
{json.dumps(environment_context, ensure_ascii=False)}
Terminal Actions: {', '.join(terminal_actions)}
Task Plan:
{json.dumps(task_plan, ensure_ascii=False)}
Persistent Execution Plan:
{json.dumps(execution_plan, ensure_ascii=False)}
Completion Status:
{json.dumps(completion_status, ensure_ascii=False)}

For object interactions, use an exact objectId from AI2-THOR Environment Context.
Never invent an objectId or use an action whose affordance is false.
If the required object is not visible, reachable, or in the required state, choose a navigation
or observation action instead of attempting the interaction.
Act only on the current_subgoal_id in Persistent Execution Plan.
Plan exactly one next action. Return JSON with thought_summary, task_progress, action, confidence, and stop_reason."""

        return prompt

    def _build_user_content(
        self,
        payload: dict[str, Any],
        *,
        include_images: bool,
    ) -> str | list[dict[str, Any]]:
        prompt = self._build_planner_prompt(payload)
        observation_image = payload.get("observation_image")
        target_crop = payload.get("target_crop")
        if not include_images or not observation_image:
            return prompt

        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": observation_image,
                    "detail": "low",
                },
            },
        ]
        if target_crop:
            content.append(
                {
                    "type": "text",
                    "text": "The next image is the user-selected target reference crop.",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": target_crop,
                        "detail": "low",
                    },
                }
            )
        return content

    @staticmethod
    def _supports_vision(credential: ApiCredential) -> bool:
        return credential.provider in {
            "kimi",
            "openai",
            "openai_compatible",
            "env_openai",
            "env_custom",
        }

    @staticmethod
    def _is_thinking_model(model_name: str) -> bool:
        """Kimi K2 native-multimodal models emit hidden reasoning and only
        accept temperature=1, so they need special request parameters."""
        normalized = (model_name or "").lower()
        return normalized.startswith("kimi-k2")

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if not self.available():
            audit = build_no_credentials_error("complete_json")
            return {
                "error": "no_credentials",
                "errors": [legacy_error_message(audit)],
                "provider_errors": [audit],
            }
        errors: list[str] = []
        provider_errors: list[dict[str, Any]] = []
        for credential in self.credentials:
            model_name = credential.model or "gpt-4o-mini"
            profile = request_profile(
                "complete_json",
                thinking_model=self._is_thinking_model(model_name),
            )
            context = ModelCallContext.start(
                operation="complete_json",
                provider=credential.provider,
                model=model_name,
                profile=profile,
            )
            response: Any | None = None
            try:
                client = OpenAI(
                    api_key=credential.api_key,
                    base_url=credential.base_url,
                    timeout=profile.timeout_seconds,
                    max_retries=profile.sdk_max_retries,
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=profile.temperature,
                    max_tokens=profile.max_tokens,
                    extra_headers=request_headers(context),
                    response_format={"type": "json_object"} if credential.provider != "deepseek" else None,
                )
                content = response.choices[0].message.content or "{}"
                result = json.loads(content)
                result["model_call"] = build_success_audit(context, response)
                return result
            except json.JSONDecodeError as exc:
                audit = build_validation_error(
                    context,
                    response=response,
                    message=f"JSON decode error: {str(exc)[:100]}",
                )
                provider_errors.append(audit)
                errors.append(legacy_error_message(audit))
            except Exception as exc:
                audit = build_provider_error(context, exc)
                provider_errors.append(audit)
                errors.append(legacy_error_message(audit))
        return {
            "error": "all_model_calls_failed",
            "errors": errors,
            "provider_errors": provider_errors,
        }


def smoke_test() -> dict[str, Any]:
    adapter = ModelAdapter()
    if not adapter.available():
        return {"ok": False, "reason": "no credentials"}
    result = adapter.complete_json(
        "Return only compact JSON.",
        'Return {"ok": true, "action": "TURN_RIGHT"} and no extra text.',
    )
    return {"ok": bool(result.get("ok") or result.get("action")), "result": result, "audit": adapter.audit()}


if __name__ == "__main__":
    print(json.dumps(smoke_test(), ensure_ascii=False, indent=2))
