# 10006 AI2-THOR Demo Integration Verified

## Goal

Continue the project toward the PPT-style embodied visual search demo:

1. Identify the simulator shown in the reference material.
2. Try to run the same kind of simulator locally.
3. Connect the Agent to the simulator path.
4. Produce a complete web-callable demo with recorded video.
5. Verify the demo output and keep the codebase clean.

## Simulator Identification

The reference screenshot corresponds to an AI2-THOR-style embodied simulator demo, specifically a `FloorPlan211` scene layout with:

- robot egocentric camera view,
- global/top-down environment view,
- language task instruction,
- Agent thought/action output,
- step-wise search trajectory.

## AI2-THOR Runtime Findings

Native Windows is not a viable runtime for the real simulator in this workspace:

- Current Windows Python can import `ai2thor`, but it is version `2.7.4`.
- The API route correctly reports native Windows as unavailable for real AI2-THOR simulation.
- The web route now avoids trying to launch the unsupported Windows Unity build and returns a clean fallback demo.

WSL2 was also investigated:

- WSL Ubuntu is available.
- NVIDIA tooling is visible through `nvidia-smi`.
- `ai2thor==5.0.0` was installed in WSL and the CloudRendering Unity build downloaded successfully.
- Unity crashed before rendering because Vulkan selected `llvmpipe (LLVM 17.0.6, 256 bits)` instead of an NVIDIA Vulkan physical device.
- This is an environment/GPU Vulkan configuration blocker, not an Agent pipeline blocker.

Required WSL environment remediation before a real AI2-THOR video can be produced:

```bash
sudo apt update
sudo apt install -y vulkan-tools libvulkan1 mesa-vulkan-drivers
vulkaninfo --summary
```

The required success condition is that `vulkaninfo --summary` shows an NVIDIA device, not only `llvmpipe`.

## Implemented Integration

Added `src/simulation/ai2thor_adapter.py`:

- exposes `AI2ThorVisualSearchDemo`;
- reports simulator status and diagnostics;
- maps project actions to AI2-THOR actions:
  - `MOVE_FORWARD` -> `MoveAhead`,
  - `TURN_LEFT` -> `RotateLeft`,
  - `TURN_RIGHT` -> `RotateRight`,
  - `LOOK_UP` -> `LookUp`,
  - `LOOK_DOWN` -> `LookDown`,
  - `INSPECT`/`STOP` -> `Pass`;
- captures robot POV frames;
- renders a top-down map from simulator metadata;
- writes an MP4 and JSON summary when AI2-THOR is available.

Updated `src/ui/app.py`:

- added `GET /api/simulator/status`;
- added `GET /api/simulator/diagnostics`;
- added `POST /api/demo/ai2thor/run`;
- the AI2-THOR route now checks runtime availability first;
- when AI2-THOR is unavailable, it returns the complete local demo and includes the fallback reason.

Updated `src/simulation/room_simulator.py`:

- upgraded the local demo to a FloorPlan211-compatible presentation style;
- enlarged output video to `1600x900`;
- added robot POV, global map, field-of-view arc, decision panel, hints, confidence, and backend metadata;
- preserved the configured Agent thresholds and pipeline.

Updated `src/ui/static/index.html`:

- added backend selector:
  - AI2-THOR first with automatic fallback,
  - local FloorPlan211-compatible demo;
- added scene input defaulting to `FloorPlan211`;
- added simulator status panel;
- added side-by-side robot POV, global map, Agent action/confidence/thought, and replay timeline.

Updated `requirements.txt`:

- added `opencv-python` for MP4 writing and verification;
- added `openai` for the existing model adapter.

## Verification

Commands run:

```powershell
python -B -m unittest discover -s tests -v
python -B -m src.simulation.room_simulator
```

Result:

- 6/6 unit tests passed.
- Demo generated 4 Agent steps.
- Final action: `STOP`.
- Final confidence: `0.959`.
- Target: `red cup`.
- Output video: `docs/demo_outputs/embodied_visual_search_demo.mp4`.
- Video dimensions: `1600x900`.
- Video frame count: `8`.

API-level verification:

- started FastAPI on a temporary port;
- called `GET /api/simulator/status`;
- called `GET /api/agent/audit`;
- called `POST /api/demo/ai2thor/run`;
- verified that the AI2-THOR-first route returns the local demo fallback cleanly on this machine;
- verified that the model adapter smoke test succeeds using the configured `apikey.txt`;
- extracted and checked key video frames:
  - `docs/demo_outputs/video_checks/video_check_00.png`,
  - `docs/demo_outputs/video_checks/video_check_04.png`,
  - `docs/demo_outputs/video_checks/video_check_07.png`.

Verification evidence is stored in:

```text
docs/demo_outputs/api_video_verification.json
```

The latest verification report records:

- `audit_ok: true`;
- `model_adapter_ok: true`;
- `demo_backend: local_ppt_style_fallback`;
- `requested_backend: ai2thor`;
- `step_count: 4`;
- `last_action: STOP`;
- `last_confidence: 0.959`;
- readable video dimensions and nonblank key-frame statistics.

## Codebase Cleanliness

Removed temporary files created during environment diagnosis:

- `docs/ai2thor_diag.sh`;
- `docs/ai2thor_retry.sh`;
- `get-pip.py`;
- `.wsl_ai2thor_venv`.

The remaining generated demo artifacts are intentional evidence outputs under `docs/demo_outputs/`.

## Current Deliverable Status

The project now has a complete web-callable demo:

```powershell
python -m src.ui.app
```

Open:

```text
http://127.0.0.1:8000
```

The UI can:

- audit the Agent/model adapter;
- request the AI2-THOR-first simulator path;
- fall back automatically when the host cannot run AI2-THOR;
- show robot POV, global environment map, thought, action, confidence, and trajectory;
- replay the steps;
- load the generated MP4.

## Remaining Blocker For Real AI2-THOR Visuals

The only missing item relative to the exact PPT simulator is host-level Vulkan/GPU configuration for AI2-THOR in WSL/Linux. Once WSL Vulkan exposes an NVIDIA physical device, the existing `AI2ThorVisualSearchDemo` adapter can be used to produce a real `FloorPlan211` video through the same web route.
