# Embodied Visual Search Agent

This project implements a web-callable embodied visual search Agent.

The normal version satisfies the required behavior:

- Language instruction plus first-person visual observation as input.
- Thought summary plus discrete action as output.
- Multi-step interaction loop for target search.
- Config-driven action space and pipeline.

The enhanced version is a strict superset:

- Click-selected target crop plus language instruction.
- Long-term and negative interaction memory.
- Retrieval-enhanced target-location hints.
- Confidence-driven stopping.
- Trajectory replay and visual panels.

## Run

```powershell
python -m pip install -r requirements.txt
python -m src.data.generate_demo_dataset
python -m src.evaluation.evaluator
python -m src.simulation.room_simulator
python -m src.ui.app
```

Open:

```text
http://127.0.0.1:8000
```

## Test

```powershell
python -m unittest discover -s tests
```

## Configuration

The project uses `configs/agent_config.json` as the single source of truth for:

- Pipeline stages.
- Action space.
- Agent thresholds.
- Vision settings.
- Memory settings.
- Dataset and evaluation paths.

## Demo Recording

```powershell
python -B -m src.simulation.room_simulator
```

Outputs:

- `docs/demo_outputs/embodied_visual_search_demo.mp4`
- `docs/demo_outputs/demo_summary.json`
- `docs/demo_outputs/frames/*.png`

## AI2-THOR Demo Mode

The web UI exposes an AI2-THOR-first route at `/api/demo/ai2thor/run`.
It attempts the real `FloorPlan211` simulator, then falls back to the
local FloorPlan211-compatible demo when the host cannot launch AI2-THOR.

Current verified behavior on this machine:

- Native Windows imports an old AI2-THOR package, but the PyPI Unity build
  cannot run the target simulator reliably here.
- WSL2 can install AI2-THOR 5.0.0 and download the CloudRendering build, but
  Unity only sees the `llvmpipe` Vulkan device and crashes before rendering.
- The fallback still produces a complete replayable demo with robot POV,
  global map, thought/action trace, confidence, API-key model audit, and MP4
  recording.

Generated verification evidence:

- `docs/demo_outputs/api_video_verification.json`
- `docs/demo_outputs/video_checks/video_check_00.png`
- `docs/demo_outputs/video_checks/video_check_04.png`
- `docs/demo_outputs/video_checks/video_check_07.png`
