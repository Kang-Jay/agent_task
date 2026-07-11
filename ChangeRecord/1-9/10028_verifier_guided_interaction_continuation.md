# 10028 Verifier-Guided Interaction Continuation

## Objective

Fix the remaining interaction-task stall where the final demo could spend repeated turns in approach guidance and then block on another slow VLM request before executing the next missing embodied predicate.

The target closed-loop contract remains:

1. User provides only a natural-language instruction.
2. A real VLM vision step observes the robot RGB and task context.
3. AI2-THOR executes navigation and interaction actions.
4. Environment metadata and postconditions update memory.
5. The verifier decides which predicate is still missing and whether the task is complete.

## Problem

`vase_into_box` could reach the target through verified AI2-THOR approach guidance but then continue waiting for another VLM call before issuing `PickupObject` or `PutObject`. Since the Kimi vision request can take several minutes including retries, this caused final demo generation to hang.

The old continuation action also contained a demo-specific fallback:

- `PickupObject` always used `Vase`.
- `PutObject` always used `Vase -> Box`.

That was not acceptable for a general embodied agent harness.

## Changes Made

### 1. Let approach guidance yield to verifier-guided interaction continuation

File: `src/agent/controller.py`

- Added `guidance_yielded_for_interaction` in `EmbodiedSearchAgent.step()`.
- This branch activates only after the session already has a successful real VLM vision step.
- Once repeated verified approach navigation has yielded, the agent directly calls `_continue_supported_task()` from the current verifier `missing_actions`.
- The trace marks this as `fallback_reason = verifier_guided_interaction_continuation`.

This preserves VLM participation while preventing repeated slow model calls from blocking deterministic next predicates such as pickup and put after the VLM has already grounded the task context.

### 2. Generalize PickupObject and PutObject target selection

File: `src/agent/controller.py`

- Replaced fixed `Vase` / `Box` continuation args with metadata-driven selection.
- `PickupObject` now selects a visible pickupable object matching the task instruction and current AI2-THOR context.
- `PutObject` now uses the current `inventoryObjects[0]` as the held object and selects a visible receptacle matching the instruction.
- `OpenObject` now also carries the selected object id when available.

The selection uses:

- `TaskPlan.matching_target_object_ids()`
- object affordances such as `pickupable`, `receptacle`, `openable`
- visibility and distance
- instruction-level type aliases such as mug, bowl, vase, box, door, sofa, and television

### 3. Added regression tests

File: `tests/test_model_planner.py`

Added tests that prove:

- `put the mug in the bowl` chooses `Mug|1`, not a nearer `Vase|1`.
- `PutObject` uses the held inventory object and chooses `Bowl|1`, not a nearer `Box|1`.
- After a real VLM vision step and repeated approach guidance, the agent skips a new slow model call and emits the next verifier-guided interaction action.

## Verification Completed Locally

Commands run:

```powershell
python -B -m py_compile src\agent\controller.py tests\test_model_planner.py
python -B -m unittest discover -s tests -p test_model_planner.py -v
python -B -m unittest discover -s tests -p test_task_semantics.py -v
python -B -m unittest discover -s tests -p test_final_agent_demo_runner.py -v
git diff --check
```

Results:

- `test_model_planner.py`: 30 tests OK.
- `test_task_semantics.py`: 13 tests OK.
- `test_final_agent_demo_runner.py`: 12 tests OK.
- `git diff --check`: no whitespace errors.

## Next Required Verification

1. Commit and push the local changes to `main`.
2. Pull on `3090GPU2:/home/scale/kangjay/kaohe`.
3. Run the focused remote test suite.
4. Run `vase_into_box` first to verify the stall is resolved.
5. Run all four final demos:
   - `television`
   - `vase_into_box`
   - `right_door_exit`
   - `sofa_sit`
6. Inspect the generated summaries and videos before final acceptance.
