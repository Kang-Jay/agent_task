# 10024 Plan2 Engineering Reliability And Cinematic Tool

## Scope

This record covers the Plan2 engineering reliability and demo generation workstream only.
The implementation intentionally does not change mapping, memory, stream/UI, evaluation metrics, or the AI2-THOR interaction executor in this pass.

## Frozen Configuration

The following model/runtime request values remain frozen by tests and were not changed:

- Kimi thinking model path: `kimi-k2.6`
- thinking temperature: `1.0`
- thinking max tokens: `2048`
- thinking per-attempt timeout: `90.0` seconds
- standard planner temperature: `0.1`
- standard planner max tokens: `300`
- standard planner per-attempt timeout: `15.0` seconds
- complete_json standard max tokens: `220`
- OpenAI SDK retry owner: `openai_sdk`
- SDK max retries: `2`

`request_deadline_seconds` is retained for backward compatibility. New audit fields clarify that this is a per-attempt SDK timeout, not a total wall-clock deadline.

## Implemented Changes

### 1. Model API reliability audit

Files:

- `src/agent/model_reliability.py`
- `src/agent/model_adapter.py`
- `tests/test_model_reliability.py`
- `tests/test_model_adapter_reliability.py`

Changes:

- Added `request_headers(context)` and passed `X-Client-Request-ID` through all three model API calls.
- Added `per_attempt_timeout_seconds` and `estimated_max_wall_time_seconds` to model call audit payloads.
- Added structured, redacted `provider_error` details for provider `type`, `code`, `param`, and `message`.
- Added uniform no-credential audit payloads for `plan_task`, `plan_action`, and `complete_json`.
- Preserved existing model, timeout, max token, temperature, and SDK retry values.

### 2. Dependency lock

File:

- `requirements.txt`

The dependency file is now pinned to the remote 3090GPU2 runtime that executes real AI2-THOR:

- `fastapi==0.139.0`
- `uvicorn==0.50.2`
- `pillow==12.3.0`
- `numpy==2.2.6`
- `pyyaml==6.0.3`
- `opencv-python==5.0.0.93`
- `openai==2.44.0`
- `ai2thor==5.0.0`

Local environment versions were inspected but not used as the lock source because the real Unity validation runs on 3090GPU2.

### 3. Evaluation manifest validation

Files:

- `configs/evaluation/plan2_multiscene_v1.json`
- `src/evaluation/manifest.py`
- `tests/test_evaluation_manifest.py`

Changes verified in this pass:

- The Plan2 manifest is inference-only.
- Oracle and non-oracle groups are paired.
- At least three scenes are represented.
- Interaction and visual-search task types are present.
- Result paths are relative JSON paths without traversal.
- Scene split leakage and pair mismatches are rejected.

### 4. Cinematic demo tool

Files:

- `tools/make_cinematic_demo.py`
- `tests/test_cinematic_demo_tool.py`

Changes verified in this pass:

- The tool is CLI-driven and has no import-time file side effects.
- Post-action `DemoStep.observation_path` semantics are required by default.
- Legacy pre-action summaries are rejected unless explicitly allowed.
- Temporary frame directories are cleaned by default.
- `--keep-frames` remains available for explicit debugging.
- Generated MP4 output is browser-compatible H.264/yuv420p.
- CLI end-to-end generation writes verification JSON and leaves no retained frame directory by default.

## Subagent Review

Six subagent restarts were attempted after quota failures. Five restarts were blocked by thread limits. One read-only reviewer completed successfully and identified the model reliability findings implemented here:

- client request IDs were not forwarded to provider APIs;
- timeout audit naming was ambiguous;
- provider errors lacked structured details;
- no-credential paths lacked uniform audit shape;
- repeated API call code needs continued consolidation in a later pass.

The repeated API-call consolidation was intentionally not done in this pass to avoid unnecessary churn after the reliability fixes were covered by tests.

## Verification

Focused tests:

```powershell
python -B -m unittest discover -s tests -p 'test_model_reliability.py' -v
python -B -m unittest discover -s tests -p 'test_model_adapter_reliability.py' -v
python -B -m unittest discover -s tests -p 'test_cinematic_demo_tool.py' -v
python -B -m unittest discover -s tests -p 'test_evaluation_manifest.py' -v
```

Result:

- model reliability: 6 tests OK
- model adapter reliability: 1 test OK
- cinematic demo tool: 6 tests OK
- evaluation manifest: 9 tests OK
- focused total: 22 tests OK

Full regression:

```powershell
python -B -m unittest discover -s tests -v
python -B -m compileall -q src tools tests
git diff --check
```

Result:

- 312 tests OK
- 2 live-model tests skipped by explicit gate
- compileall passed
- git diff whitespace check passed

Repository hygiene:

- `__pycache__` directories under `src`, `tools`, and `tests` were removed after compile verification.
- Generated videos, raw frames, caches, and API keys remain ignored.
- No temporary frame directory is kept by the cinematic tool unless explicitly requested.

## Remaining Work

This workstream does not complete the whole Plan2 project. Remaining Plan2 work includes:

- real non-oracle RGB-D map integration into live planning;
- real Unity execution of `????????????`;
- real Unity execution of `????????`;
- final remote pull, remote regression, live episode generation, video generation, and decoded video inspection.
