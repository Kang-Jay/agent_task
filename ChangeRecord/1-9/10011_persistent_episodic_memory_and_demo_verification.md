# ChangeRecord 10011: Persistent Episodic Memory and Demo Verification

## Date

2026-07-10

## Objective

Upgrade the existing visual-search Agent from process-local session history to a
persistent, execution-grounded episodic memory pipeline. The implementation must:

1. reuse the memory principles found in the downloaded research code;
2. record only actions that were actually executed by the simulator;
3. retrieve relevant prior episodes before planning;
4. expose recalled memory to the model, API response, web UI, and recorded demo;
5. preserve every existing configured threshold and pipeline stage;
6. pass module, integration, live-model, evaluation, and video checks.

## Research Reuse Decision

The implementation uses project-native code rather than importing a complete
external framework.

### Reflexion

Reference:
`research/papers/code/reflexion/alfworld_runs/generate_reflections.py`

Reused principle:
- convert completed or failed execution into a concise reusable lesson;
- retrieve only a small relevant memory set for the next attempt.

### Voyager

Reference:
`research/papers/code/voyager/voyager/agents/skill.py`

Reused principle:
- persist reusable experience outside the current process;
- retrieve top-k experience by task relevance;
- allow memory to survive Agent restart.

### LangGraph Store

Reference:
`research/codebases/langgraph/source/libs/checkpoint/langgraph/store/memory/__init__.py`

Reused principle:
- namespace memory records;
- use an explicit add/search interface;
- separate transient session state from persistent storage.

### Mem0

Reference:
`research/codebases/mem0/source/mem0/memory/main.py`

Reused principle:
- attach session, Agent run, and environment metadata to memories;
- support scoped search rather than injecting all history.

### Rejected Direct Reuse

The project does not directly depend on Mem0, Chroma, or a remote embedding
service. Those dependencies would introduce embedding dimensions, similarity
thresholds, model choices, and deployment services that are not defined by the
project requirements or current configuration.

The first implementation therefore uses deterministic lexical similarity and
the existing configured values:

- `memory.long_term_capacity = 200`
- `memory.retrieval_top_k = 3`
- `memory.negative_memory_capacity = 80`

No new model, training, fine-tuning, embedding, or navigation hyperparameter was
invented.

## Implemented Changes

### 1. Persistent Episodic Store

New file:
- `src/memory/episodic_store.py`

Features:
- standard-library SQLite persistence;
- namespace, session, instruction, executed action, success, confidence, region,
  lesson, environment metadata, and timestamp;
- deterministic English/Chinese task tokenization;
- relevance-first ranking, then executed success, confidence, and recency;
- current-session exclusion;
- configured capacity pruning;
- explicit connection commit, rollback, and close for Windows compatibility.

Runtime database:
- `datasets/embodied_search_v1/memory/episodic_memory.sqlite3`
- ignored by Git as runtime state.

### 2. Execution-Grounded Memory Commit

Modified:
- `src/memory/session_memory.py`
- `src/agent/controller.py`
- `src/simulation/room_simulator.py`
- `src/simulation/ai2thor_adapter.py`

Rules:
- a proposed action is not persisted as reusable experience;
- memory is written only after `commit_execution`;
- simulator-overridden actions replace proposals before storage;
- action success, robot state, backend, scene, planner source, and skill call are
  stored as metadata;
- repeated commit of the same step does not duplicate the episode;
- failed executions generate an explicit alternative-action lesson;
- successful terminal execution preserves final visual evidence and STOP context.

### 3. Retrieval Before Planning

Modified:
- `src/agent/controller.py`
- `src/agent/model_adapter.py`

Flow:

1. validate request;
2. load/create session;
3. retrieve up to configured top-k prior executed episodes;
4. analyze the current image;
5. retrieve task hints;
6. inject episodic lessons into the multimodal planner payload;
7. validate and execute the selected action;
8. persist the confirmed result.

The documented pipeline order remains unchanged:

`validate_request -> decode_observation -> load_memory -> analyze_vision ->
retrieve_hints -> plan_action -> validate_action -> update_memory ->
emit_response`

### 4. Response and UI Visibility

Modified:
- `src/types/schema.py`
- `src/ui/static/index.html`
- `src/simulation/room_simulator.py`

Added response field:
- `recalled_memories`

The web memory panel now shows:
- executed action;
- success or failure;
- observed region;
- reusable lesson.

The recorded video now contains:
- current action and confidence;
- task thought;
- retrieved task hints;
- recalled executed episode.

### 5. Video Rendering Quality

Modified:
- `src/simulation/room_simulator.py`

Improvements:
- cross-platform Chinese font search with Microsoft YaHei, SimHei, Noto CJK,
  WenQuanYi, and DejaVu fallbacks;
- pixel-width text wrapping;
- mixed Chinese/English token-aware wrapping;
- maximum line limits and ellipsis;
- no missing-glyph boxes in the verified final frame;
- no English word splitting in the verified final frame.

### 6. Configuration and Repository Safety

Modified:
- `src/task/config.py`
- `.gitignore`
- `src/evaluation/evaluator.py`

Validation added:
- long-term memory capacity must be positive;
- negative-memory capacity must be positive;
- retrieval top-k must be positive and no greater than long-term capacity.

Repository safety:
- SQLite runtime files are ignored;
- generated web/demo trajectory JSON files are ignored;
- curated tracked dataset trajectories remain tracked;
- evaluator uses a temporary trajectory directory and an offline model adapter;
- two consecutive evaluator runs no longer delete or modify formal trajectories.

## Tests

### Episodic Memory Tests

New file:
- `tests/test_episodic_memory.py`

Covered:
- persistence across store instances;
- relevant memory ranking;
- unrelated memory rejection;
- current-session exclusion;
- configured capacity pruning;
- failed execution lesson;
- duplicate commit prevention;
- Agent restart recovery;
- model payload injection;
- API response/search-map visibility;
- Chinese structured-thought integrity.

Result:
- 5/5 passed.

### Full Unit and Integration Suite

Command:

```powershell
python -B -m unittest discover -s tests -v
```

Result:
- 70 tests passed;
- 1 live test skipped by its default environment guard;
- 0 failures;
- 0 errors.

### Explicit Live Multimodal Test

Command:

```powershell
$env:RUN_LIVE_MODEL_TESTS='1'
python -B -m unittest discover -s tests -p "test_live_model_integration.py" -v
```

Result:
- 1/1 passed;
- provider: `kimi`;
- model: `moonshot-v1-8k-vision-preview`;
- visual input used: `true`;
- returned action was inside the configured action space.

### Evaluation

Two consecutive runs:

```powershell
python -m src.evaluation.evaluator
python -m src.evaluation.evaluator
```

Both results:
- episodes: 3;
- successes: 3;
- success rate: 1.0;
- illegal actions: 0;
- average confidence: 0.862;
- average IoU: 0.8213351783801345;
- SPL: unavailable because the static dataset has no measured path lengths;
- SPL coverage: 0.0.

Formal trajectory files remained unchanged.

### Runtime Audit

Endpoint:
- `GET http://127.0.0.1:18001/api/agent/audit`

Verified:
- status: `ok`;
- stop confidence threshold: `0.78`;
- memory backend: `sqlite`;
- memory capacity: `200`;
- retrieval top-k: `3`;
- model credentials available: `true`;
- pipeline exactly matches `configs/agent_config.json`.

## Final Demo Verification

URL:
- `http://127.0.0.1:18001`

Video:
- `docs/demo_outputs/embodied_visual_search_demo.mp4`

Summary:
- `docs/demo_outputs/demo_summary.json`

Final run:
- instruction: `Locate the red cup`;
- steps: 4;
- first-step recalled memories: 3;
- final action: `STOP`;
- final done state: `true`;
- final confidence: `0.928`;
- model provider: `kimi`;
- model: `moonshot-v1-8k-vision-preview`;
- visual input used: `true`;
- video resolution: 1600 x 900;
- encoded frames: 8;
- readable frames: 8;
- frame rate: 2 FPS;
- duration: 4.0 seconds;
- file size: 242,507 bytes.

Frames inspected:
- initial exploration frame;
- intermediate rotation/search frame;
- final red-cup confirmation and STOP frame.

Visual inspection confirmed:
- robot heading changes match TURN_RIGHT;
- target is absent during exploration;
- red cup becomes visible in the final robot view;
- global map and robot heading remain consistent;
- final STOP and confidence are visible;
- recalled episode is visible;
- no missing Chinese glyphs;
- no text overflow or split English words in the final frame.

The in-app browser runtime had no available browser instance, so the page itself
could not be screenshot-tested in that browser. HTTP endpoints, generated frames,
video decoding, and local image inspection were completed successfully.

## Hyperparameter and Structure Audit

Unchanged:
- `agent.max_steps = 20`
- `agent.history_window = 6`
- `agent.stop_confidence_threshold = 0.78`
- `agent.target_visible_threshold = 0.58`
- `agent.default_turn_angle_degrees = 30`
- vision input size and all vision weights;
- allowed and terminal actions;
- evaluation confidence and IoU thresholds;
- documented pipeline stage order.

The memory implementation consumes existing configuration values and does not
silently add training, fine-tuning, embedding, navigation, or model parameters.

## Remaining Limitations

This phase improves the Agent substantially but does not justify a claim of a
complete research-grade embodied-search system.

1. The local `RoomSimulator` remains a deterministic compatibility/demo backend,
   not AI2-THOR.
2. The AI2-THOR demo still uses segmentation grounding and search overrides to
   guarantee presentation success; an unbiased Agent-only benchmark remains
   necessary.
3. The formal evaluation dataset still has only three static positive episodes.
4. SPL coverage is zero because measured optimal and executed path lengths are
   not present in the static dataset.
5. Episodic retrieval is lexical, not learned semantic retrieval.
6. The project remains inference-only; no training or fine-tuning pipeline has
   been executed or claimed.

## Conclusion

The project now has a real persistent memory loop:

`visual observation + instruction -> prior executed-episode recall -> multimodal
planning -> simulator execution -> confirmed outcome -> persistent lesson`

The implementation is tested, configuration-bound, restart-persistent,
model-visible, UI-visible, video-visible, and does not introduce undocumented
hyperparameters.
