# 10001 Goal Execution Plan - Embodied Visual Search Agent

## Project Goal

Build a web-callable embodied visual search Agent for unfamiliar scenes.

Required basic behavior:

- Input: task instruction and first-person visual observation.
- Output: explainable thought summary plus executable action.
- Interaction: multi-step observation-action loop until the target object is found.
- Delivery: Agent or skill style system with a web interface.

Enhanced behavior must be a strict superset of the basic version:

- Long-term interaction memory.
- Click-selected target crop plus language instruction as multimodal task input.
- Trajectory replay and visual explanation panels.
- Retrieval-enhanced planning, confidence scoring, and exploration status.

## Stage 1 - Research And Memory Library

### 1. Requirement Extraction

Inputs inspected:

- `题目.txt`
- `视觉搜索Agent示例.pptx`

Extracted hard requirements:

- The system is a multimodal embodied visual search Agent.
- The target is searched in an unfamiliar scene.
- The input contains a task instruction and visual observation.
- The output contains thought and action.
- The search is completed through multi-round interaction.
- The basic function accepts language instruction.
- The advanced function considers interaction memory.
- The enhancement accepts a clicked object crop plus language instruction.

### 2. Paper References

Primary theoretical references:

- Embodied-Reasoner: observation-thought-action trajectories for embodied interactive search.
- ReAct: interleaved reasoning and action loop.
- SayCan: action feasibility filtering with language planning.
- ObjectNav Revisited: object-goal navigation task and success metrics.
- REVERIE: visual referring expression plus embodied target localization.

Enhancement references:

- Voyager: skill memory and retrieval.
- Reflexion: failure reflection memory.
- SIMA and SIMA2: generalist embodied agents in 3D virtual worlds.
- HAMT and DUET: history-aware and topology-aware navigation.
- VIMA: multimodal prompts with language and visual target references.
- CoW and NavProg: zero-shot object navigation and modular visual programs.

### 3. Codebase References

Recommended open-source references:

- `allenai/ai2thor`: indoor embodied simulation and object interaction.
- `facebookresearch/habitat-lab` and `habitat-sim`: benchmark-grade embodied navigation.
- `langchain-ai/langgraph`: stateful agent graph loop.
- `mem0ai/mem0`: long-term memory design.
- `run-llama/llama_index`: retrieval orchestration.
- `chroma-core/chroma`: local vector retrieval.
- `gradio-app/gradio` or `Chainlit/chainlit`: agent UI inspiration.

Local reference:

- `D:\GitProjects\claudecode`: useful ideas include event-stream agent loops, typed tools, concurrency-safe execution, memory relevance selection, and clean session isolation. This project should not copy its compiled source style or heavy global state model.

### 4. Current Implementation Decision

To keep the project reproducible and clean, the first implementation uses a dependency-light local stack:

- Python backend with FastAPI.
- Native browser UI using HTML/CSS/JavaScript.
- Pillow and NumPy for deterministic vision heuristics.
- JSON configuration as the single source of truth for actions, thresholds, pipeline settings, and evaluation settings.
- No external model dependency is required for the default demo path, but model adapters are separated so a VLM/LLM can be added later without changing the API.

## Stage 2 - Basic Version Implementation

### 1. Data Preparation And Processing

Required files:

- `datasets/embodied_search_v1/images/*.png`
- `datasets/embodied_search_v1/annotations/episodes.jsonl`
- `datasets/embodied_search_v1/trajectories/*.json`

Checks:

- Every image path exists.
- Every target bounding box is inside the image.
- Every action is declared in `configs/agent_config.json`.
- Every episode has a stop condition.
- Dataset generation is deterministic.

### 2. Module Construction

Core modules:

- `src/types/schema.py`: typed request, response, action, observation, and trace records.
- `src/task/config.py`: config loader and consistency validator.
- `src/vision/heuristic_vision.py`: deterministic first-person observation analyzer and target crop matcher.
- `src/memory/session_memory.py`: per-session interaction memory and long-term memory store.
- `src/rag/retriever.py`: retrieval over prior traces and object-location priors.
- `src/agent/controller.py`: ReAct-style observation-thought-action loop.
- `src/evaluation/evaluator.py`: dataset and trajectory evaluator.
- `src/ui/app.py`: FastAPI application and static UI.

### 3. Basic Agent Pipeline

For each `/api/agent/step` call:

1. Validate request and load the shared config.
2. Decode the observation image or load a dataset image.
3. Read session memory.
4. Run visual analysis.
5. Retrieve relevant target-location hints.
6. Build a concise thought summary.
7. Select one valid action from the configured action space.
8. Update session memory and replay trace.
9. Return thought, action, confidence, candidate objects, memory summary, and done flag.

### 4. Validation

Basic validation must pass before enhancement work:

- Config consistency check.
- Unit tests for image processing, memory, action legality, and target crop matching.
- Dataset validation.
- Offline evaluation over sample episodes.
- API smoke test through FastAPI test client or direct controller call.

## Stage 3 - Strict Superset Enhanced Version

Enhancement modules must not remove or weaken basic requirements.

### 1. Required Superset Features

- Multimodal goal binding from language plus clicked crop.
- Long-term interaction memory and negative memory.
- Retrieval-enhanced search priors.
- Confidence-driven stop/continue decision.
- Trajectory replay with every observation, thought, action, confidence, and memory update.
- Visual UI panels for observation, target crop, candidates, memory, confidence, and replay.

### 2. Additional Innovation Points

- Target crop color-signature matching for zero-shot local demos.
- Exploration heat and repeated-action penalty.
- Search-zone recommendation using object-location priors.
- Stop decision guardrail: only stop when confidence crosses the configured threshold.
- Exportable trace JSON for later model fine-tuning.
- Config audit endpoint to prove hyperparameters, action space, and pipeline structure match the written configuration.

### 3. Final Audit

Before final delivery:

- Re-read `题目.txt` and PPTX-derived requirements.
- Verify every basic requirement has an implemented feature and test evidence.
- Verify enhanced features are additive and do not replace the normal version.
- Verify no hardcoded action list conflicts with config.
- Verify no temporary, dead, or unused files remain.
- Run the complete test suite and evaluator.
- Start the web server and provide the local URL.

