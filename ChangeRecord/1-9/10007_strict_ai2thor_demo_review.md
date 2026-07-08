# 10007 Strict AI2-THOR Demo Review and Recording

## 1. Goal

This record closes the current audit-and-demo iteration for the embodied visual search Agent project.

The requested outcome was:

1. Re-check the PPT/problem requirements in `D:\cache\SummerCap\kaohe\zju\题目.txt` and `D:\cache\SummerCap\kaohe\zju\视觉搜索Agent示例.pptx`.
2. Use five high-reasoning subagents for independent review, with the main process summarizing and deciding.
3. Verify the existing pipeline for standardized data preparation, module construction, training/validation, hyperparameter/structure consistency, and codebase cleanliness.
4. Fix the rough/fallback simulator problem by running a real AI2-THOR-style demo on the remote 3090GPU2 server.
5. Record a local browser demo video and inspect the result visually.

## 2. Requirements Re-Checked

### 2.1 Problem Text

The core task is an embodied visual search Agent:

- Input: task instruction plus visual observation.
- Output: thought plus action.
- Behavior: search a target object in an unfamiliar environment through multi-round interaction.
- Deliverable form: Agent or skill.
- Reference direction: Embodied-Reasoner / SIMA-style multimodal embodied decision loop.

### 2.2 PPT-Derived Checklist

The PPT evidence extracted earlier in `docs/audit/ppt_problem_extracted_text.json` and `docs/audit/ppt_problem_extracted_text.md` indicates the demo should include:

- Language instruction.
- Robot first-person observation.
- Intermediate thought/action steps.
- Trajectory or video replay.
- Advanced memory/interaction support.
- Enhanced multimodal input: click object in scene, generate target crop/screenshot, combine with language instruction.
- A formal simulator-style scene, not a rough hand-drawn fallback.

## 3. Five-Subagent Review Summary

Five independent review agents completed read-only audits before this fix pass.

| Reviewer | Focus | Main Finding |
|---|---|---|
| Heisenberg | PPT / requirements checklist | Core Agent contract was partially covered, but click-to-generate-target-crop and strict AI2-THOR consistency were incomplete. |
| Curie | Data and evaluation pipeline | Dataset/evaluation were too small and weak: 3 one-step visible-target examples, no real split, no negative/multistep cases, no bbox IoU validation. |
| Franklin | Agent planning / vision | Thought was post-hoc, STOP was too aggressive, metadata was not driving planning, and RAG/memory did not materially affect decisions. |
| Plato | Simulator/UI/demo | Real AI2-THOR and fallback were mixed together, so the webpage could look like an AI2-THOR demo while silently returning fallback output. |
| Aristotle | Config/cleanliness | Several config values were not consistently consumed; `ASK_CLARIFY` was configured terminal but not treated as terminal; defaults drifted from config. |

## 4. Code Changes Made

### 4.1 Strict AI2-THOR Mode

Changed `/api/demo/ai2thor/run` so strict AI2-THOR mode no longer silently falls back to the local PPT-style simulator.

- If AI2-THOR is unavailable and `allow_fallback=false`, the endpoint now returns an HTTP error instead of a fake fallback demo.
- Fallback remains available only when explicitly requested.
- UI text now labels the backend as `Real AI2-THOR FloorPlan211 (strict, no fallback)`.

Files changed:

- `src/ui/app.py`
- `src/ui/static/index.html`

### 4.2 AI2-THOR Runtime Consistency

Updated the local AI2-THOR adapter to match the remote runtime requirements:

- Added `snapToGrid=False` to avoid `rotateStepDegrees=30` conflict.
- Added `renderInstanceSegmentation=True`.
- Capped route `max_steps` to configured `agent.max_steps`.
- Each AI2-THOR step now carries `backend="ai2thor"` and `scene="FloorPlan211"`.

Files changed:

- `src/simulation/ai2thor_adapter.py`

### 4.3 Simulator-Grounded Target Confirmation

Added AI2-THOR instance segmentation grounding for target objects.

Instead of trusting the color heuristic and stopping early, the AI2-THOR route now:

1. Reads `event.instance_masks`.
2. Matches target aliases such as `television`, `pillow`, `sofa`, `floor lamp`, `remote`.
3. Builds a real object bbox from the simulator mask.
4. Requires one `INSPECT` confirmation step before `STOP`.
5. Keeps searching if the heuristic says STOP but no AI2-THOR target mask is grounded.

This directly fixed the earlier failure where the Agent stopped after one step on a false "red region" while the visible object list did not include the requested object.

Files changed:

- `src/simulation/ai2thor_adapter.py`

### 4.4 Configured Terminal Action Handling

Fixed controller terminal handling:

- Before: `done = action.type == "STOP"`.
- After: `done = action.type in config.terminal_actions`.

This makes `ASK_CLARIFY` consistent with `configs/agent_config.json`.

Files changed:

- `src/agent/controller.py`

### 4.5 Tests Added

Added tests for:

- AI2-THOR segmentation target grounding from a fake mask.
- Configured terminal action behavior for `ASK_CLARIFY`.

Files changed:

- `tests/test_agent.py`

## 5. Tests Run

### 5.1 Local Tests

Command:

```powershell
python -B -m unittest discover -s tests -v
```

Result:

- 8 tests run.
- 8 passed.
- No failures.

### 5.2 Remote Tests on 3090GPU2

Command executed under:

```bash
/home/scale/kangjay/kaohe/.mamba-env/bin/python -B -m unittest discover -s tests -v
```

Result:

- 8 tests run.
- 8 passed.
- No failures.

## 6. Remote Deployment Status

Remote server:

- Host alias: `3090GPU2`
- Project path: `/home/scale/kangjay/kaohe`
- Python env: `/home/scale/kangjay/kaohe/.mamba-env`
- AI2-THOR version: `5.0.0`
- Remote service process observed: `.mamba-env/bin/python -m src.ui.app`
- Remote listen address: `127.0.0.1:8000`

Local tunnel used:

```powershell
ssh -N -L 18000:127.0.0.1:8000 3090GPU2
```

Local browser URL:

```text
http://127.0.0.1:18000
```

## 7. Strict AI2-THOR Demo Result

Strict route request:

```json
{
  "instruction": "Find the television in the room",
  "scene": "FloorPlan211",
  "max_steps": 20,
  "allow_fallback": false
}
```

Strict route output:

- Steps: 7
- Backend for every step: `ai2thor`
- Scene for every step: `FloorPlan211`
- Final action: `STOP`
- Final confidence: `0.943`
- Final target evidence: `Television (segmented)`

Step trajectory:

| Step | Backend | Scene | Action | Confidence | Visible Evidence |
|---:|---|---|---|---:|---|
| 0 | ai2thor | FloorPlan211 | TURN_RIGHT | 0.570 | Floor, ShelvingUnit |
| 1 | ai2thor | FloorPlan211 | TURN_RIGHT | 0.570 | Floor |
| 2 | ai2thor | FloorPlan211 | TURN_RIGHT | 0.570 | Floor |
| 3 | ai2thor | FloorPlan211 | MOVE_FORWARD | 0.570 | Floor, GarbageCan |
| 4 | ai2thor | FloorPlan211 | TURN_RIGHT | 0.570 | Floor |
| 5 | ai2thor | FloorPlan211 | INSPECT | 0.943 | Floor, GarbageCan, Television, Television (segmented) |
| 6 | ai2thor | FloorPlan211 | STOP | 0.943 | Floor, GarbageCan, Television, Television (segmented) |

Evidence files:

- `docs/remote_ai2thor_checks/strict_route_response.json`
- `docs/ai2thor_outputs/ai2thor_demo_summary.json`
- `docs/ai2thor_outputs/ai2thor_visual_search_demo.mp4`
- `docs/ai2thor_outputs/frames/ai2thor_frame_06.png`

## 8. Browser Demo Recording

The browser page was opened through the local tunnel at:

```text
http://127.0.0.1:18000
```

The page triggered the strict AI2-THOR route and then replayed the trajectory.

Recorded browser video:

- `docs/browser_recordings/remote_ai2thor_strict_replay_demo.mp4`

Video verification:

- `docs/browser_recordings/strict_replay_video_verification.json`

Verification metrics:

- Exists: true
- Size: 1,008,997 bytes
- Frames: 90
- FPS: 10.0
- Resolution: 1264 x 712
- Source frames: 90
- Nonblank sample frame statistics:
  - First frame mean/std: `79.0 / 73.72`
  - Middle frame mean/std: `81.13 / 73.94`
  - Last frame mean/std: `81.13 / 73.94`

Manually inspected frames:

- `docs/browser_recordings/frames_ai2thor_replay/browser_ai2thor_replay_000.png`
- `docs/browser_recordings/frames_ai2thor_replay/browser_ai2thor_replay_045.png`
- `docs/browser_recordings/frames_ai2thor_replay/browser_ai2thor_replay_089.png`
- `docs/ai2thor_outputs/frames/ai2thor_frame_06.png`

Visual inspection result:

- The scene is a real AI2-THOR Unity-rendered room, not the old hand-drawn fallback.
- The browser UI shows backend `ai2thor` and scene `FloorPlan211`.
- The replay includes search actions and ends at `STOP`.
- The final thought states that AI2-THOR segmentation grounds the requested target as Television.

## 9. Pipeline Review: Completed vs Remaining Gaps

### 9.1 Now Satisfied for Demo Purposes

- Real remote AI2-THOR simulator can run on 3090GPU2.
- Strict route no longer hides simulator failures with fallback.
- Browser page can call the Agent demo route through the local tunnel.
- Demo video is recorded locally and visually inspected.
- Agent output includes thought, action, confidence, and done.
- Multi-step trajectory is present.
- Final target confirmation is grounded by AI2-THOR instance segmentation.
- Tests pass locally and remotely after changes.

### 9.2 Still Not Fully Satisfied for a Research-Grade Standardized Pipeline

The project is now a much more credible demo, but it is not yet a complete standardized training/evaluation pipeline.

Remaining gaps:

1. Dataset is still too small:
   - Current generated dataset has only a few synthetic examples.
   - It lacks train/validation/test splits.
   - It lacks negative cases, invisible-target cases, distractor cases, and long-horizon cases.

2. Evaluation remains shallow:
   - Existing evaluator does not yet enforce bbox IoU.
   - It does not measure path efficiency, success weighted by path length, collision rate, SPL, or final target category correctness.

3. Training/fine-tuning is not implemented:
   - Current Agent is still largely rule/heuristic-based.
   - `apikey.txt` and model adapter exist, but model-backed planning is not fully integrated into the Agent decision loop.

4. Memory is not yet strong enough:
   - Session memory and hints exist.
   - Long-term reusable memory does not yet materially drive planning across tasks.

5. Enhanced multimodal click flow is only partially covered:
   - Backend schema supports `clicked_point` and `target_crop`.
   - The frontend still does not fully implement the PPT-style interaction where the user clicks an object in the scene, auto-generates the crop, and launches a full multimodal demo video from that selected target.

6. UI can be more polished:
   - Current UI is functional and now uses real AI2-THOR frames.
   - It is not yet a "wow" version with richer cinematic replay, object overlays, mask highlighting, camera frustum visualization, and side-by-side multimodal target binding.

## 10. Codebase Cleanliness

Cleanliness scan checked for common temporary/cache files:

- `__pycache__`
- `.pyc`
- `.pytest_cache`
- `.tmp`
- `.bak`
- `.log`

Local scan did not report matching temporary artifacts.

Intentional evidence outputs are retained under:

- `docs/browser_recordings/`
- `docs/ai2thor_outputs/`
- `docs/remote_ai2thor_checks/`

These are not dead files for this iteration because they are the demo evidence requested by the user.

## 11. Final Assessment

The earlier criticism was valid: the old webpage/demo could be rough because fallback and real AI2-THOR were mixed, and the Agent could stop on unreliable heuristic evidence.

This iteration fixed the core demo credibility problem:

- Real AI2-THOR is running on 3090GPU2.
- The webpage can call it through `http://127.0.0.1:18000`.
- The strict demo no longer silently falls back.
- The Agent now uses simulator instance segmentation to confirm the target.
- A local browser replay video has been recorded and inspected.

However, this should not be claimed as a fully complete research-grade project yet. The next phase should build the missing standardized pipeline: larger AI2-THOR episode generation, formal splits, richer metrics, model-backed planning, click-to-target frontend flow, and a more polished advanced demo layer.
