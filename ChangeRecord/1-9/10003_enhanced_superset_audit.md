# 10003 Enhanced Superset Audit

## Superset Rule

The enhanced version keeps every normal-version behavior and only adds extra output fields and UI panels. The normal `/api/agent/step` call still accepts language instruction plus first-person visual observation and still returns `thought`, `action`, `confidence`, and `done`.

## Enhanced Features Added

- Multimodal target binding through `clicked_point` or `target_crop`.
- Target crop signature matching in the vision module.
- Session replay with recent thought/action/confidence records.
- Search map with visited region counts, unexplored regions, region confidence, and negative memory.
- Confidence trace for per-step confidence visualization.
- Trace export endpoint at `/api/agent/export/{session_id}`.
- Config audit endpoint at `/api/agent/audit`.
- Static dataset serving for the web demo.
- Research memory downloader and index under `research/`.

## Basic Requirement Coverage

| Requirement | Implementation | Check |
| --- | --- | --- |
| Task instruction input | `AgentRequest.instruction` | Covered |
| Visual observation input | `AgentRequest.observation_image` path/base64/data URL | Covered |
| Thought output | `AgentResponse.thought` | Covered |
| Action output | `AgentResponse.action` | Covered |
| Multi-round interaction | `SessionMemory.steps`, replay, trace export | Covered |
| Target search | `HeuristicVision` candidates + confidence stop guard | Covered |
| Language instruction basic mode | `target_binding.mode = language_only` | Covered |
| Interaction memory advanced mode | `SessionMemory`, negative memory, search map | Covered |
| Click target crop enhancement | `clicked_point` / `target_crop` path | Covered |
| Web presentation | FastAPI + native HTML UI | Covered |

## Hyperparameter And Pipeline Audit

- Action space is loaded from `configs/agent_config.json`.
- Pipeline stages are validated exactly in `src/task/config.py`.
- `agent.max_steps` equals `evaluation.max_episode_steps`.
- `agent.stop_confidence_threshold` equals `evaluation.min_success_confidence`.
- Vision weights sum to 1.0.
- `STOP` is required and validated as a legal action.
- Tests verify enhanced fields without changing normal behavior.

## Final Validation Command

```powershell
python -m src.data.generate_demo_dataset
python -m unittest discover -s tests -v
python -m src.evaluation.evaluator
python research\download_research_assets.py
```

