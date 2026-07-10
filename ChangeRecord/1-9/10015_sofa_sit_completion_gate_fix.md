# 10015 Sofa Sit Completion Gate Fix

## Problem

The instruction `找到房间里的沙发并坐下` could terminate after the first observation when the sofa was already visible. The UI then presented the episode as completed even though AI2-THOR had not executed or verified a sitting action.

## Root Causes

1. A model or rule planner could propose `STOP` as soon as visual confidence crossed the search threshold.
2. The generic `done` explanation always stated that the target had been confirmed, including `ASK_CLARIFY` and other unsuccessful terminal outcomes.
3. `TaskPlan.is_visual_search` did not exclude unsupported compound tasks.
4. The AI2-THOR adapter independently classified instructions and could apply the segmentation search oracle without using the Agent task plan.
5. The web UI treated every `episode_completed` event as successful, regardless of `completion_status.complete`.

## Completion Authority

The VLM proposes actions and may propose `STOP` or `Done`, but it is not the final completion authority.

Completion requires:

- task-semantic validation of all required subgoals;
- AI2-THOR environment evidence;
- action-specific postcondition verification;
- `completion_status.complete == true`.

AI2-THOR 5.0.0 exposes `Crouch` and `Stand`, but it does not expose a verified `SitOnObject` action or a sitting-on-furniture state. The system must therefore not report `坐在沙发上` as successful. An explicit instruction such as `走到沙发旁并蹲下` may use `Crouch` as a documented approximation, but crouching must not be labeled as real sitting.

## Changes

- Unsupported tasks now take precedence over generic illegal-action handling.
- `ASK_CLARIFY` carries the actual capability limitation instead of `illegal action blocked`.
- Human-readable and structured reasoning distinguish successful completion from unsupported or unverified termination.
- Unsupported compound tasks are no longer classified as pure visual search.
- The segmentation oracle is enabled only when the Agent task plan is supported and explicitly marked as pure visual search.
- The UI displays `needs-input` or `incomplete` when the final step has not passed completion verification.
- Added regression tests for both:
  - `走到沙发上并坐下`
  - `找到房间里的沙发并坐下`

## Expected Result

For `找到房间里的沙发并坐下`:

- the system must not claim that seeing the sofa completes the instruction;
- the validated action is `ASK_CLARIFY`;
- `completion_status.complete` remains `false`;
- the explanation states that AI2-THOR cannot verify sitting on furniture;
- the UI marks the episode as incomplete rather than successful.

For pure visual search instructions such as `找到房间里的电视`:

- segmentation-confirmed `INSPECT -> STOP` remains available;
- successful STOP still requires the configured confidence threshold.

## Verification

Focused regression suite:

```text
test_task_semantics.py: 5 passed
test_model_planner.py: 12 passed
test_ai2thor_sync.py: 10 passed
```

The full suite and remote streamed replay must pass before deployment is considered complete.
