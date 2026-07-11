# 1007 Closed-Loop VLM And Placement Verification

## Objective
Make the final AI2-THOR demos run as a real closed loop instead of a hardcoded sequence:

1. Natural-language instruction only.
2. Real VLM planner receives current robot RGB and environment context.
3. AI2-THOR executor performs the selected action.
4. Environment metadata/postconditions feed back into memory and verifier.
5. Verifier decides whether the task is complete.

Final required demos:

- Find the television in the room.
- Put the vase into the box.
- Find the right-side door and walk out.
- Find the sofa and sit down by the supported Crouch approximation.

## Changes Made

### 1. Prevent approach guidance from bypassing the VLM

File: `src/agent/controller.py`

- Added `_has_successful_real_vlm_step(state)`.
- `APPROACH_TARGET` simulator guidance is now allowed only after the session already contains at least one successful real VLM vision step.
- This prevents the first executable step from being produced only by AI2-THOR oracle approach metadata.
- Approach guidance remains available later as environment-grounded navigation assistance.

Why this is needed:

- The previous flow could complete or advance interaction tasks through `simulator_oracle` before the VLM saw the image.
- Final demos require `EmbodiedSearchAgent -> VLM planner -> AI2-THOR executor -> verifier`, not a scripted oracle-only path.

### 2. Generalize PutObject completion evidence

Files:

- `src/simulation/ai2thor_adapter.py`
- `src/agent/task_semantics.py`
- `tools/run_final_agent_demos.py`

Changes:

- Added generic `final_state.placement` evidence:
  - `movedObjectId`
  - `movedObjectType`
  - `receptacleObjectId`
  - `receptacleObjectType`
  - `parentReceptacles`
  - `receptacleObjectIds`
  - `inventoryObjects`
- Task completion now validates generic placement evidence first.
- Legacy `vaseObjectId` / `boxObjectId` fields remain as compatibility fallback for existing vase-box summaries.
- The final demo runner now accepts the generic placement structure and still rejects missing pickup/put postconditions.

Why this is needed:

- `vase_into_box` should be one instance of a general put-object-in-receptacle predicate.
- The verifier must not be a demo-specific hardcoded vase/box checker.

### 3. Added regression coverage

Files:

- `tests/test_model_planner.py`
- `tests/test_task_semantics.py`

Added tests:

- First-step approach metadata does not bypass VLM planning.
- Approach guidance can still be used after a successful real VLM vision step.
- Generic placement evidence supports `put mug in bowl`.
- Wrong generic receptacle evidence is rejected.
- Right-door exit rejects selected/crossed door evidence when `requested_relation=right` but `relation_verified=False`.

## Verification Completed Locally

Commands run:

```powershell
python -B -m py_compile src\agent\controller.py src\agent\task_semantics.py src\simulation\ai2thor_adapter.py tools\run_final_agent_demos.py tests\test_model_planner.py tests\test_task_semantics.py tests\test_final_agent_demo_runner.py
python -B -m unittest discover -s tests -p test_task_semantics.py -v
python -B -m unittest discover -s tests -p test_final_agent_demo_runner.py -v
python -B -m unittest discover -s tests -p test_model_planner.py -v
python -B -m unittest discover -s tests -p test_*.py -v
python -B -m unittest discover -s tests -p test_execution_commit.py -v
python -B -m unittest discover -s tests -p test_ai2thor_sync.py -v
git diff --check
```

Results:

- Full local test suite: 364 tests passed, 2 skipped.
- Focused planner/semantics/final-runner tests passed.
- `git diff --check` passed.
- No temporary videos, API keys, venvs, caches, or generated demo outputs are staged.

## Remaining Required Remote Verification

Run on `3090GPU2` after push and pull:

```bash
cd /home/scale/kangjay/kaohe
git pull --ff-only origin main
PYTHONPATH=. .mamba-env/bin/python -B -m unittest discover -s tests -p test_model_planner.py -v
PYTHONPATH=. .mamba-env/bin/python -B -m unittest discover -s tests -p test_task_semantics.py -v
PYTHONPATH=. .mamba-env/bin/python -B -m unittest discover -s tests -p test_final_agent_demo_runner.py -v
rm -rf docs/ai2thor_outputs/final-agent-demos-* docs/ai2thor_outputs/final_agent_demos
PYTHONPATH=. xvfb-run -a .mamba-env/bin/python tools/run_final_agent_demos.py --output-dir docs/ai2thor_outputs/final_agent_demos
```

Acceptance checklist for the four final videos:

- Every task has a readable `demo.mp4` and `ai2thor_demo_summary.json`.
- Every task has at least one real VLM vision call with provider/model metadata and `vision_input_used=True`.
- Every action is executed through AI2-THOR executor feedback, not only a synthetic script.
- Television: final evidence proves target visible/located.
- Vase into box: final placement evidence and pickup/put postconditions agree.
- Right-door exit: selected door is the right-side door and threshold crossing is verified.
- Sofa sit: sofa is approached, `Crouch` succeeds, and final state is marked as approximate sit.

## Notes

This change intentionally does not remove simulator guidance. It changes simulator guidance from a first-step decision maker into a post-VLM, environment-verified execution aid. That matches the intended agent pattern: VLM decides from observation and task context, while AI2-THOR metadata verifies feasibility and completion.
