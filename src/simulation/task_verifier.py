from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.agent.task_semantics import TaskPlan


TaskOutcome = Literal[
    "in_progress",
    "exact_success",
    "approximate_success",
    "failed",
    "unsupported",
    "terminated",
]


@dataclass(frozen=True)
class TaskVerification:
    outcome: TaskOutcome
    complete: bool
    reason: str
    evidence_ledger: list[dict[str, Any]]
    completion_status: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.completion_status)
        payload["outcome"] = self.outcome
        payload["complete"] = self.complete
        payload["reason"] = self.reason
        payload["evidence_ledger"] = list(self.evidence_ledger)
        return payload


class TaskVerifier:
    """Final task-outcome authority independent of model output and UI."""

    def verify(
        self,
        task_plan: TaskPlan,
        *,
        steps: list[dict[str, Any]],
        target_visible: bool,
        confidence: float,
        stop_confidence_threshold: float,
        environment_context: dict[str, Any] | None = None,
        failed: bool = False,
        failure_reason: str | None = None,
        terminated: bool = False,
        termination_reason: str | None = None,
    ) -> TaskVerification:
        completion_status = task_plan.completion_status(
            steps=steps,
            target_visible=target_visible,
            confidence=confidence,
            stop_confidence_threshold=stop_confidence_threshold,
            environment_context=environment_context,
        )

        if completion_status["complete"]:
            outcome: TaskOutcome = (
                "approximate_success"
                if task_plan.completion_mode == "approximate_sit"
                else "exact_success"
            )
            reason = str(completion_status["reason"])
        elif not task_plan.supported:
            outcome = "unsupported"
            reason = str(
                task_plan.clarification
                or completion_status["reason"]
                or "task capability is unsupported"
            )
        elif failed:
            outcome = "failed"
            reason = failure_reason or "task execution failed"
        elif terminated:
            outcome = "terminated"
            reason = termination_reason or "episode terminated before task completion"
        else:
            outcome = "in_progress"
            reason = str(completion_status["reason"])

        evidence_ledger = [
            {
                "predicate": str(item.get("id")),
                "passed": bool(item.get("complete")),
                "evidence": item.get("evidence"),
                "source": "deterministic_task_predicate",
            }
            for item in completion_status.get("subgoal_progress", [])
        ]
        if not evidence_ledger:
            evidence_ledger.append(
                {
                    "predicate": "task_completion",
                    "passed": bool(completion_status["complete"]),
                    "evidence": reason,
                    "source": "deterministic_task_predicate",
                }
            )

        return TaskVerification(
            outcome=outcome,
            complete=bool(completion_status["complete"]),
            reason=reason,
            evidence_ledger=evidence_ledger,
            completion_status=completion_status,
        )
