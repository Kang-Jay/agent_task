# 10017 Plan 2 Strict Execution Record

## 0. Purpose

This record is the execution authority for implementing
`Plan_2_hierarchical_embodied_agent_upgrade.md`.

The implementation must proceed in dependency order. A later phase may start
only after the current phase has:

1. production code;
2. focused tests;
3. adjacent integration tests;
4. full local regression;
5. configuration audit;
6. recorded failures and fixes;
7. evidence that no unrelated collaborator changes were overwritten.

The project uses an inference-only multimodal model API. This execution does
not train or fine-tune a model and does not create checkpoints.

## 1. Baseline Snapshot

Baseline commit before this execution:

```text
309e752 fix: gate completion for unsupported compound tasks
branch: main
```

The working tree already contains collaborator changes. They are treated as
authoritative in-progress work and must be reviewed and integrated, not
reverted.

Modified tracked areas observed at baseline:

- `configs/agent_config.json`
- `src/agent/controller.py`
- `src/agent/model_adapter.py`
- `src/agent/task_semantics.py`
- `src/simulation/ai2thor_adapter.py`
- `src/types/schema.py`
- `src/ui/app.py`
- `src/ui/static/index.html`
- `tests/test_schema.py`

Relevant untracked implementation work observed at baseline:

- `src/simulation/object_closeup.py`
- `tests/test_object_closeup.py`
- `tests/test_ai2thor_closeup_integration.py`
- `ChangeRecord/1-9/10016_object_click_closeup_render.md`

These files belong to the object-click and close-up feature. Plan 2 changes
must preserve them unless a verified interface correction is required.

## 2. Baseline Test Result

Command:

```powershell
python -B -m unittest discover -s tests -v
```

Result:

```text
145 tests run
3 failures
1 live-model test skipped by environment gate
```

Failing tests:

1. `test_agent_rejects_unverifiable_sit_task`
2. `test_find_sofa_and_sit_is_not_reduced_to_visual_search`
3. `test_sit_on_sofa_is_not_falsely_supported`

The failures are not to be hidden. They expose a deliberate strategy change:

- ChangeRecord 10015 required unsupported sit instructions to return
  `ASK_CLARIFY`.
- Plan 2 now permits a clearly labelled approximation:
  locate furniture -> approach -> execute `Crouch` -> verify
  `agent.isStanding=false` -> report `approximate_success`.

The old tests therefore require replacement, but the new implementation is
not yet acceptable because it currently treats target visibility as approach
proximity and does not pass `environment_context` into every completion check.

## 3. Frozen Existing Configuration

The following existing values remain unchanged:

- `agent.max_steps = 20`
- `agent.history_window = 6`
- `agent.repeated_action_penalty = 0.12`
- `agent.stop_confidence_threshold = 0.78`
- `agent.target_visible_threshold = 0.58`
- `agent.exploration_confidence_floor = 0.18`
- `agent.default_turn_angle_degrees = 30`
- `vision.image_size = [448, 448]`
- `vision.grid_rows = 3`
- `vision.grid_cols = 3`
- `vision.candidate_patch_size = 96`
- `memory.long_term_capacity = 200`
- `memory.negative_memory_capacity = 80`
- `memory.retrieval_top_k = 3`
- `evaluation.min_success_iou = 0.3`

No threshold may be added merely to make a test pass. Simulator-defined
limits must come from AI2-THOR runtime configuration or metadata and must be
recorded with their source.

The close-up parameters introduced by ChangeRecord 10016 are outside the
current task-semantics phase. They must remain unchanged until their own
remote Unity verification is complete.

## 4. Execution Phases

### Phase 0: Baseline and Collaboration Freeze

Actions:

1. Record branch, commit, tracked modifications and untracked files.
2. Run the complete local suite and preserve the exact failures.
3. Read all diffs in files touched by collaborators.
4. Verify `apikey.txt`, videos, caches and downloaded research repositories
   remain excluded from Git.
5. Confirm the next ChangeRecord number before creating new records.

Exit criteria:

- baseline facts recorded;
- no collaborator change overwritten;
- existing failures classified by root cause;
- configuration values frozen.

### Phase 1: Task State and Completion Semantics

Scope:

- `src/agent/task_semantics.py`
- `src/agent/controller.py`
- related schema only when required;
- focused tests.

Actions:

1. Keep the sofa instruction supported only as
   `completion_mode=approximate_sit`.
2. Keep the limitation
   `native_sit_on_furniture_state_unavailable`.
3. Require separate evidence for:
   - target located;
   - target approached;
   - `Crouch` executed successfully;
   - simulator posture verified.
4. Pass `request.environment_context` into every completion calculation.
5. Do not equate object visibility with approach proximity.
6. Prevent `STOP` or `Done` before every required predicate passes.
7. Return `approximate_success`, never exact sitting success.
8. Replace old unsupported-task tests with exact predicate tests.

Focused test order:

```powershell
python -B -m unittest tests.test_task_semantics -v
python -B -m unittest tests.test_model_planner -v
python -B -m unittest tests.test_ai2thor_sync -v
python -B -m unittest discover -s tests -v
```

Exit criteria:

- seeing a sofa cannot finish the task;
- crouching far from the sofa cannot finish the task;
- approaching without crouching cannot finish the task;
- crouching near the matching sofa with verified posture completes only as an
  approximation;
- full local regression passes.

### Phase 2: Persistent Global Task Plan

Scope:

- task plan schema;
- model adapter planning interface;
- controller session state;
- structured progress events.

Actions:

1. Add a structured global plan with task type, target binding, ordered
   subgoals, completion predicates and failure policy.
2. Persist the plan for the episode instead of regenerating unrelated goals
   every step.
3. Let the model choose one next action for the current subgoal.
4. Record model `Done` as a proposal only.
5. Advance a subgoal only after deterministic evidence passes.
6. Replan after execution failure, invalid binding or stale target evidence.
7. Expose plan progress without exposing hidden chain-of-thought.

Required tests:

- plan serialization;
- plan persistence across steps;
- instruction immutability;
- current-subgoal advancement;
- rejected premature `Done`;
- replan after failed action;
- illegal model action rejection;
- visual input audit.

Exit criteria:

- every episode has one auditable global plan;
- current subgoal is deterministic and persisted;
- the model cannot directly declare task success.

### Phase 3: Action Catalog, Binding and Postconditions

Actions:

1. Keep the official AI2-THOR action catalog as the only native action source.
2. Map legacy abstract actions to native actions through existing aliases.
3. Bind interactions to a real visible `objectId`.
4. Validate actor mode, action exposure, overload and parameter types.
5. Verify action-specific state changes.
6. Preserve the already implemented Open/Pickup/Put resolver and
   postcondition tests.
7. Add only task-relevant actions to a model prompt; do not give the model
   unrestricted system actions.

Required real-Unity chain:

```text
OpenObject -> PickupObject -> PutObject
```

Exit criteria:

- each advertised action has preconditions, execution evidence and
  postconditions;
- no invented object ID executes;
- continuous interaction chain passes in real Unity.

### Phase 4: Independent Task Verifier

Actions:

1. Separate task predicates from model output.
2. Produce one of:
   - `exact_success`
   - `approximate_success`
   - `failed`
   - `unsupported`
   - `terminated`
3. Store evidence for every completed predicate.
4. Reject missing, stale or contradictory evidence.
5. Keep action API success separate from task success.

Exit criteria:

- every successful episode has a complete evidence ledger;
- false success is covered by regression tests;
- UI status comes from verifier output.

### Phase 5: Sofa Approximation End-to-End

Execution sequence:

```text
locate sofa
-> bind matching sofa instance
-> approach using environment geometry
-> align/inspect
-> execute Crouch
-> verify distance and isStanding=false
-> approximate_success
```

Negative cases:

- sofa visible but far away;
- wrong furniture instance;
- failed movement;
- `Crouch` rejected;
- posture unchanged;
- target lost;
- max steps reached.

Exit criteria:

- focused local tests pass;
- remote AI2-THOR episode passes;
- browser stream, episode JSON and video agree.

### Phase 6: Geometric Search and Open-Vocabulary Evidence

Actions:

1. Replace the UI-region search map as a planning source with pose and
   reachable-space geometry.
2. Reuse VLFM concepts selectively:
   value map, frontier selection and target evidence fusion.
3. Keep simulator segmentation as an oracle evaluation mode, not the only
   production perception path.
4. Add open-vocabulary evidence with explicit confidence and object tracking.
5. Prevent hidden target coordinates from influencing non-oracle planning.

Exit criteria:

- frontier selection is reproducible from recorded geometry;
- target evidence is attributable to observation frames;
- oracle and non-oracle results are reported separately.

### Phase 7: Hierarchical Memory and Visualization

Memory layers:

1. live simulator state;
2. working step memory;
3. task/subgoal memory;
4. spatial/object memory;
5. episodic success/failure memory;
6. reusable skill/failure patterns.

Actions:

1. Keep `SessionMemory` focused on the current episode.
2. Store object instances with object ID, location, state, confidence and
   last-seen step.
3. Store spatial exploration and frontier state.
4. Store completed episodes with failure and recovery evidence.
5. Retrieve relevant memory and record exactly which entries influenced the
   action payload.
6. Add invalidation for stale or contradictory memory.
7. Visualize:
   - current subgoal;
   - object memory table;
   - spatial trajectory/frontiers;
   - retrieved episodic memories;
   - memory influence on the current decision.

Exit criteria:

- memory changes a controlled repeated-task decision;
- stale memory cannot silently complete a task;
- memory views match persisted structures.

### Phase 8: Streaming Web Experience

Actions:

1. Stream plan creation, perception, retrieval, action proposal, validation,
   execution, postcondition and verifier events.
2. Show current progress and termination reason.
3. Preserve manual control and target-click close-up interaction.
4. Keep all labels synchronized with actual action timing.
5. Clearly identify real model, rule fallback, replay and live Unity modes.

Exit criteria:

- no long blank interval during execution;
- browser state matches backend event order;
- no replay is presented as live execution.

### Phase 9: Evaluation Task Set and Regression

This phase creates no training data and performs no model training.

Required coverage:

- pure visual search;
- language plus clicked-object target;
- navigation;
- sofa approximation;
- Open/Pickup/Put;
- unsupported capability;
- invalid target;
- execution failure;
- API failure;
- memory reuse;
- non-oracle perception.

Required reports:

- task success and approximate success;
- false-stop and missed-stop rates;
- SPL/navigation error;
- illegal action count;
- postcondition pass rate;
- subgoal completion and replan count;
- API/fallback rate;
- memory retrieval quality;
- failed episode analysis.

Exit criteria:

- every result is traceable to a frozen task-set version and configuration;
- failed episodes remain in the report;
- test tasks are not used to tune prompts or thresholds.

### Phase 10: Remote Deployment and Demonstration

Actions:

1. Run local compile and full regression.
2. Review every staged file explicitly.
3. Push reviewed local commits to GitHub `main`.
4. Pull on `/home/scale/kangjay/kaohe` using `git pull --ff-only`.
5. Verify local, GitHub and remote SHA equality.
6. Run live multimodal API tests.
7. Run real AI2-THOR episodes.
8. Record videos and inspect action/observation alignment frame by frame.

Exit criteria:

- visual search, clicked target, sofa approximation and Open/Pickup/Put demos
  all have auditable evidence;
- no secret, video, cache or downloaded repository is committed;
- remote service runs the reviewed SHA.

## 5. Per-Module Gate

For every code module:

```text
read current implementation and collaborator diff
-> write focused test or update obsolete test
-> make the smallest production change
-> run focused test
-> run adjacent integration tests
-> run full suite
-> reread modified code and configuration
-> update this ChangeRecord with evidence
-> advance to the next module
```

A failing test must be classified as one of:

- production regression;
- obsolete expectation;
- environment dependency;
- flaky external service;
- unsupported platform.

No failure may be ignored or converted into a skip merely to make the suite
green.

## 6. Repository Cleanliness

- Do not use `git add .`.
- Do not commit `apikey.txt`, videos, frames, logs, caches, databases, model
  weights or downloaded research repositories.
- Do not create temporary top-level scripts.
- New files require a single clear owner and import/use path.
- Duplicate reports must be consolidated before deletion.
- Dead code removal requires reference search and regression testing.
- Preserve collaborator changes unless the same line must change for a
  verified Plan 2 fix.

## 7. Current Active Phase

Active phase: **Phase 5 - Sofa Approximation End-to-End**

Immediate work:

1. synchronize the reviewed implementation to `3090GPU2`;
2. run the permanent real-Unity sofa validation tool;
3. run the actual Agent instruction `找到房间里的沙发并坐下`;
4. verify the Agent does not finish from visual detection alone;
5. compare browser stream, episode JSON, verifier evidence and recording.

Execution evidence will be appended below after each gate passes.

## 8. Execution Evidence

### Phase 0

Status: completed.

Evidence:

- baseline commit and dirty files recorded;
- complete test suite executed;
- three failing sit-semantic tests preserved and classified;
- existing configuration read and frozen;
- ChangeRecords 10015 and 10016 reviewed;
- no production code modified during baseline audit.

### Phase 1

Status: completed locally.

Production changes:

- `src/agent/task_semantics.py`
  - validates finite AI2-THOR target-distance evidence;
  - separates located, approached, crouched and posture-verified predicates;
  - reports `approximate_success` only after every predicate passes;
  - does not introduce a new distance threshold.
- `src/agent/controller.py`
  - passes `environment_context` into both completion calculations;
  - rejects premature `STOP`/`Done` for approximate sitting;
  - continues with approach, `Crouch` or evidence refresh;
  - exposes explicit incomplete-task reasoning;
  - adds Chinese labels for `Crouch` and `Stand`.
- `tests/test_task_semantics.py`
  - replaces obsolete unsupported-sit expectations;
  - covers missing distance, missing crouch and verified approximation.
- `tests/test_model_planner.py`
  - verifies a model-proposed premature `Done` becomes `Crouch`;
  - verifies the episode remains incomplete and non-terminal.

No configuration or hyperparameter was changed.

Test evidence:

```text
python -B -m unittest discover -s tests -p test_task_semantics.py -v
8 tests passed

python -B -m unittest discover -s tests -p test_model_planner.py -v
12 tests passed

python -B -m unittest discover -s tests -p test_ai2thor_sync.py -v
10 tests passed

python -B -m compileall -q src tests tools
passed

python -B -m unittest discover -s tests -v
148 tests passed, 1 live-model test skipped by its explicit environment gate
```

One failed test invocation was classified as a command-entry error:
`python -m unittest tests.test_task_semantics` cannot work because `tests`
is not a Python package. It was immediately replaced with the repository's
supported `unittest discover` form; no test was skipped.

Remote real-Unity validation remains pending and is not claimed by this local
phase result.

### Phase 2

Status: completed locally.

Production changes:

- added typed `ExecutionSubgoal` and `TaskExecutionPlan` schemas;
- persisted one execution plan per `SessionState`;
- added evidence-driven subgoal status advancement;
- added `ModelAdapter.plan_task()` for first-step multimodal global planning;
- constrained model plans to contain every deterministic semantic subgoal
  exactly once;
- added explicit `semantic_fallback` for missing or invalid model plans;
- included the persistent execution plan and current subgoal in every
  step-level model prompt;
- exported the execution plan through responses, commit results and trace
  persistence.

No configuration, threshold or model parameter was changed. The task planner
uses the same provider-specific timeout, temperature and token settings as
the existing action planner.

Test evidence:

```text
test_task_planner.py: 6 passed
test_model_planner.py: 12 passed
test_execution_commit.py: 4 passed
compileall: passed
full suite: 154 passed, 1 explicitly gated live-model test skipped
```

Verified properties:

- global task planning is called only once per session;
- the same plan ID is reused on later steps;
- invalid or incomplete model subgoal lists cannot replace the semantic
  contract;
- current-subgoal progress advances only from deterministic evidence;
- a complete evidence set closes the plan.

Live paid-model plan generation remains pending until the reviewed local
changes are synchronized.

### Phase 3

Status: completed for catalog, binding, postconditions and real Unity chain.

Reused implementation:

- `configs/ai2thor_actions_v5.json`
- `src/simulation/ai2thor_actions.py`
- `src/simulation/ai2thor_interactions.py`
- `src/simulation/ai2thor_postconditions.py`
- `tools/validate_ai2thor_interaction_chain.py`

Audit results:

- catalog modes: `arm`, `default`, `drone`, `locobot`, `stretch`;
- default mode exposes 38 planner actions, 194 manual actions and 294 system
  actions;
- all 17 interaction actions referenced by `TaskSemantics` exist in the
  catalog and are planner-exposed for default mode;
- local catalog, resolver and postcondition suites passed:
  9 + 7 + 6 tests;
- the local Python environment is AI2-THOR 2.7.4 and is therefore not used as
  real-runtime evidence;
- remote `3090GPU2` is AI2-THOR 5.0.0 with Unity commit
  `f0825767cd50d69f666c7f282e54abfe58f1e917`, exactly matching the catalog.

Remote real-Unity command:

```bash
cd /home/scale/kangjay/kaohe
PYTHONPATH=. .mamba-env/bin/python tools/validate_ai2thor_interaction_chain.py
```

Remote result:

```text
status=passed
scene=FloorPlan1
OpenObject: runtime and postcondition passed
PickupObject: runtime and inventory postcondition passed
PutObject: runtime and requested-receptacle postcondition passed
forceAction=false for all three interactions
```

Evidence:

- `docs/ai2thor_outputs/interaction_chain_validation/validation.json`
- `docs/ai2thor_outputs/interaction_chain_validation/01_open_fridge.png`
- `docs/ai2thor_outputs/interaction_chain_validation/02_pickup_egg.png`
- `docs/ai2thor_outputs/interaction_chain_validation/03_put_egg_in_bowl.png`

Two audit command errors were recorded rather than hidden:

1. an initial audit listed an unsupported `stretchab` mode;
2. a second audit tried to import a non-existent mode constant.

The final audit reads mode names directly from the action catalog metadata.
Neither command modified production code or runtime state.

### Phase 4

Status: completed locally.

Production changes:

- added `src/simulation/task_verifier.py`;
- introduced typed `TaskVerification` results;
- separated task outcome classification from model output;
- retained deterministic `TaskPlan.completion_status()` predicates as the
  factual source;
- classified outcomes as `in_progress`, `exact_success`,
  `approximate_success`, `failed`, `unsupported` or `terminated`;
- exposed a predicate evidence ledger;
- routed controller completion decisions through `TaskVerifier`;
- kept simulator action success distinct from overall task success.

Focused test evidence:

```text
python -B -m unittest discover -s tests -p test_task_verifier.py -v
4 tests passed
```

Verified negative cases:

- a model-proposed `Done` with missing predicates remains `in_progress`;
- termination does not imply success;
- unsupported capability is not reported as success;
- sofa completion is classified only as `approximate_success`.

Full regression at the Phase 4 gate:

```text
158 tests passed
2 live-model tests skipped by their explicit environment gate
```

No configuration, model parameter or task threshold was changed.

### Live Multimodal API Gate

Status: completed.

The paid API was intentionally invoked using the existing local
`apikey.txt` configuration. No key value was printed or committed.

Command:

```powershell
$env:RUN_LIVE_MODEL_TESTS='1'
python -B -m unittest discover -s tests -p test_live_model_integration.py -v
```

Result:

```text
2 tests passed
task planner used visual input
step planner used visual input
```

The tests exercise both `ModelAdapter.plan_task()` and
`ModelAdapter.plan_action()`. They do not expose hidden chain-of-thought;
they validate structured plan/action responses and vision usage.

### Provider Request Parameter Audit

The baseline working tree already contained a collaborator modification that
maps a Kimi credential to `kimi-k2.6` and applies the following
provider-specific request settings:

- `temperature=1.0`;
- `max_tokens=2048`;
- `timeout=90.0` seconds.

This Plan 2 execution did not select or tune those values to improve an
evaluation result. They were preserved as pre-existing collaborator work and
audited before commit.

Runtime credential audit, without printing any key:

```text
kimi: https://api.moonshot.cn/v1, model=kimi-k2.6
deepseek: https://api.deepseek.com/v1, model=deepseek-chat
```

Validation:

- the paid live multimodal task and action tests both passed with the
  configured Kimi model;
- `tests/test_model_planner.py` freezes the Kimi request model, timeout,
  temperature and token budget with a mocked transport;
- non-thinking providers retain their existing `temperature=0.1`,
  `max_tokens=300` action-planning behavior;
- no Agent threshold, evaluation threshold or task semantic was changed.

The close-up feature in ChangeRecord 10016 does not claim ownership of this
provider mapping. It only preserves the mapping that already existed in its
baseline working tree.

### Phase 5

Status: local implementation and tests completed; remote Unity execution
pending synchronization.

Production test tool:

- `tools/validate_ai2thor_sofa_approximation.py`

The tool:

1. launches real FloorPlan211;
2. selects a real `Sofa` object from simulator metadata;
3. uses `GetInteractablePoses` only to establish a reproducible validation
   start pose;
4. executes `TeleportFull` as validation setup, not as an Agent skill;
5. verifies the sofa is visible and has finite distance evidence;
6. executes native `Crouch`;
7. validates the action postcondition;
8. submits the world-state evidence to `TaskVerifier`;
9. requires `approximate_success`;
10. writes auditable JSON and image evidence under
    `docs/ai2thor_outputs/sofa_approximation_validation/`.

Added regression coverage:

- `tests/test_sofa_approximate_sit.py`
- `tests/test_task_verifier.py`
- existing AI2-THOR postcondition tests.

Latest local gate:

```text
python -B -m compileall -q src tests tools
passed

python -B -m unittest discover -s tests -v
162 tests passed
2 live-model tests skipped by their explicit environment gate

git diff --check
passed
```

Local Windows AI2-THOR is version 2.7.4 and is not accepted as runtime
evidence for this project. The next gate must run the tool and the full Agent
episode on remote AI2-THOR 5.0.0.
