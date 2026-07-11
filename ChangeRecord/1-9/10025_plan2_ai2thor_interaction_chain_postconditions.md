# Plan2 AI2-THOR Interaction Chain And Strict Postconditions

## Scope
- Workspace: `D:\cache\SummerCap\kaohe\zju`
- Task: continue Plan2 AI2-THOR interaction action chain and postcondition validation.
- Write scope respected: interaction/executor/postcondition/runtime related tests only for this continuation.
- Frozen settings: no model, hyperparameter, pipeline, `configs/agent_config.json`, or `configs/ai2thor_actions_v5.json` changes.
- Explicitly not touched in this continuation: UI, memory, stream, mapping, evaluation implementation.

## Current Implementation Status
- `OpenObject`, `PickupObject`, and `PutObject` already have native AI2-THOR action forwarding through the action executor/runtime boundary.
- `AI2ThorInteractionResolver` grounds semantic arguments into real `objectId` values before Unity calls.
- `PutObject` treats `objectId`/`receptacleObjectId` as the destination receptacle, not the held object.
- `PutObject` rejects execution unless exactly one inventory object is held.
- `PickupObject` rejects execution when inventory is non-empty.
- Closed openable receptacles are rejected before `PutObject` unless they are opened first.
- Runtime snapshots preserve `inventoryObjects` before/after each Unity action.
- Session execution commits only when Unity reports success and strict postcondition verification passes.
- Strict postconditions validate concrete state transitions instead of trusting `lastActionSuccess` alone.

## Added Coverage In This Pass
- Added resolver coverage for the final-task shape: `把花瓶放到纸箱里`.
- Added `PickupObject(Vase)` grounding to the concrete vase object.
- Added `PutObject(Vase -> CardboardBox)` grounding where the destination is the box receptacle.
- Added continuous simulated chain coverage for `PickupObject(Vase) -> PutObject(Box)`.
- Verified postcondition evidence includes:
  - `receptacleObjectId = Box|1`
  - `releasedObjectIds = [Vase|1]`
  - `placedObjectIds = [Vase|1]`
  - empty inventory after placement

## Test Evidence
Commands run locally:

```powershell
python -B -m unittest discover -s tests -p "test_ai2thor_interactions.py" -v
python -B -m unittest discover -s tests -p "test_ai2thor_interaction_chain.py" -v
python -B -m unittest discover -s tests -p "test_ai2thor_postconditions.py" -v
python -B -m unittest discover -s tests -p "test_ai2thor_runtime.py" -v
python -B -m unittest discover -s tests -p "test_ai2thor_session.py" -v
python -B -m unittest discover -s tests -p "test_task_semantics.py" -v
python -B -m compileall -q src tests
git diff --check
git diff -- configs/agent_config.json configs/ai2thor_actions_v5.json
```

Observed result:
- `test_ai2thor_interactions.py`: 21 tests OK.
- `test_ai2thor_interaction_chain.py`: 4 tests OK.
- `test_ai2thor_postconditions.py`: 23 tests OK.
- `test_ai2thor_runtime.py`: 9 tests OK.
- `test_ai2thor_session.py`: 10 tests OK.
- `test_task_semantics.py`: 10 tests OK.
- `compileall`: OK.
- `git diff --check`: OK.
- Config diff check: no changes to `configs/agent_config.json` or `configs/ai2thor_actions_v5.json`.

## Real Unity Acceptance Command
Run this on `3090GPU2` after pulling the branch:

```bash
cd /home/scale/kangjay/kaohe
.mamba-env/bin/python - <<'PY'
import json
from pathlib import Path
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from src.simulation.ai2thor_interactions import AI2ThorInteractionResolver
from src.simulation.ai2thor_runtime import execute_controller_action
from src.simulation.ai2thor_postconditions import AI2ThorPostconditionVerifier

out_dir = Path('docs/ai2thor_outputs/interaction_chain_validation_plan2_vase_box')
out_dir.mkdir(parents=True, exist_ok=True)
controller = Controller(
    scene='FloorPlan225',
    platform=CloudRendering,
    agentMode='default',
    width=960,
    height=540,
    quality='Low',
    gridSize=0.25,
    rotateStepDegrees=90,
    snapToGrid=True,
    renderInstanceSegmentation=True,
)
resolver = AI2ThorInteractionResolver()
verifier = AI2ThorPostconditionVerifier()
try:
    instruction = '把花瓶放到纸箱里'
    results = []
    for action, args in [
        ('PickupObject', {'objectType': 'Vase'}),
        ('PutObject', {'object': 'Vase', 'receptacleType': 'Box'}),
    ]:
        binding = resolver.resolve(
            action=action,
            args=args,
            instruction=instruction,
            metadata=controller.last_event.metadata,
        )
        if not binding.valid:
            raise SystemExit(json.dumps({'status': 'binding_failed', 'action': action, 'errors': binding.errors}, ensure_ascii=False, indent=2))
        execution = execute_controller_action(controller, action=action, args=binding.args)
        postcondition = verifier.verify(
            action=execution.action,
            args=execution.args,
            before=execution.before_metadata,
            after=execution.after_metadata,
            runtime_success=execution.success,
        )
        results.append({
            'action': action,
            'bound_args': binding.args,
            'runtime_success': execution.success,
            'postcondition': postcondition.to_dict(),
            'inventory_after': execution.inventory_after,
        })
        if not execution.success or not postcondition.passed:
            raise SystemExit(json.dumps({'status': 'failed', 'results': results}, ensure_ascii=False, indent=2))
    payload = {'status': 'passed', 'scene': 'FloorPlan225', 'instruction': instruction, 'results': results}
    (out_dir / 'validation.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
finally:
    controller.stop()
PY
```

## Remaining Caveat
- The local tests validate the resolver/runtime/postcondition chain with deterministic simulated Unity metadata.
- The real Unity command above is the required acceptance check for scene-specific object availability and reachability. If `FloorPlan225` does not expose a visible/interactable vase and box from the initial pose, the acceptance script should be rerun after navigating to an interactable pose or with the manifest initial pose used by the evaluation episode.
