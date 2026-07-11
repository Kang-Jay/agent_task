from __future__ import annotations

import unittest
from pathlib import Path


class UIStreamContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ui"
            / "static"
            / "index.html"
        ).read_text(encoding="utf-8")

    def test_environment_feedback_switches_to_post_action_pov(
        self,
    ) -> None:
        self.assertIn(
            'payload.observation_phase === "after_action"',
            self.html,
        )
        self.assertIn(
            'asset(payload.observation_path)',
            self.html,
        )

    def test_post_action_phase_is_visible_in_ui_state(self) -> None:
        self.assertIn(
            'setRunStage("feedback", 85, stepDetail)',
            self.html,
        )

    def test_each_stream_run_has_an_isolated_abort_context(self) -> None:
        self.assertIn("let activeStreamRun = null", self.html)
        self.assertIn("let streamGeneration = 0", self.html)
        self.assertIn("function beginStreamRun()", self.html)
        self.assertIn("generation: ++streamGeneration", self.html)
        self.assertIn("context.controller?.abort()", self.html)
        self.assertIn("activeStreamRun === context", self.html)
        self.assertIn("function invalidateActiveStreamRun()", self.html)

    def test_run_and_episode_identity_gate_old_stream_events(self) -> None:
        self.assertIn("function streamIdentity(message, payload)", self.html)
        self.assertIn(
            "if (runId && context.runId && runId !== context.runId) return false",
            self.html,
        )
        self.assertIn(
            "if (episodeId && context.episodeId && episodeId !== context.episodeId) return false",
            self.html,
        )
        self.assertIn("if (runId && !context.runId) context.runId = runId", self.html)
        self.assertIn(
            "if (episodeId && !context.episodeId) context.episodeId = episodeId",
            self.html,
        )
        self.assertIn('response.headers.get("X-Run-Id")', self.html)
        self.assertIn('response.headers.get("X-Episode-Id")', self.html)

    def test_event_sequence_drops_duplicates_and_out_of_order_events(self) -> None:
        self.assertIn("lastEventSeq: -1", self.html)
        self.assertIn("seenEventIds: new Set()", self.html)
        self.assertIn(
            "const {runId, episodeId, eventSeq, eventId}",
            self.html,
        )
        self.assertIn("context.seenEventIds.has(eventKey)", self.html)
        self.assertIn("context.seenEventIds.add(eventKey)", self.html)
        self.assertIn(
            "if (numericSeq <= context.lastEventSeq) return false",
            self.html,
        )
        self.assertIn("context.lastEventSeq = numericSeq", self.html)

    def test_legacy_events_without_protocol_metadata_remain_supported(self) -> None:
        self.assertIn(
            "message.run_id ?? metadata.run_id ?? payload.run_id ?? null",
            self.html,
        )
        self.assertIn(
            "message.event_seq ?? metadata.event_seq ?? payload.event_seq ?? null",
            self.html,
        )
        self.assertIn(
            'if (eventSeq !== null && eventSeq !== undefined && eventSeq !== "")',
            self.html,
        )

    def test_terminal_event_is_consumed_once(self) -> None:
        self.assertIn("const TERMINAL_STREAM_EVENTS = new Set", self.html)
        self.assertIn("context.terminalConsumed = true", self.html)
        self.assertIn(
            "if (!isCurrentStreamRun(context) || context.terminalConsumed) return false",
            self.html,
        )
        self.assertIn("payload.task_success ?? message.task_success", self.html)
        self.assertIn("payload.terminal_reason", self.html)

    def test_all_required_runtime_stages_are_visible(self) -> None:
        for stage in (
            'initialization: "初始化"',
            'observation: "观察"',
            'planning: "规划"',
            'execution: "执行"',
            'feedback: "反馈"',
            'success: "成功"',
            'failure: "失败"',
            'cancelled: "取消"',
        ):
            with self.subTest(stage=stage):
                self.assertIn(stage, self.html)
        self.assertIn('role="progressbar"', self.html)
        self.assertIn('id="runProgressLabel"', self.html)

    def test_stale_stream_errors_cannot_overwrite_current_run(self) -> None:
        self.assertIn(
            "if (runContext && !isCurrentStreamRun(runContext)) return",
            self.html,
        )
        self.assertIn("invalidateActiveStreamRun()", self.html)

    def test_cancellation_notifies_new_backend_and_aborts_transport(self) -> None:
        self.assertIn(
            "/api/demo/ai2thor/stream/${encodeURIComponent(context.runId)}/cancel",
            self.html,
        )
        self.assertIn('{method: "POST", keepalive: true}', self.html)
        self.assertIn("context.controller?.abort()", self.html)


if __name__ == "__main__":
    unittest.main()
