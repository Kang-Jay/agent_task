# 10005 Final Demo Verified

## Current Status

The project now has a complete demonstrable embodied visual search demo rather than only a form-style Agent panel.

## API Configuration

- `apikey.txt` was parsed without printing secrets.
- The model adapter detected two configured providers:
  - OpenAI-compatible key, default model `gpt-4o-mini`.
  - DeepSeek key, base URL `https://api.deepseek.com/v1`, model `deepseek-chat`.
- API smoke test passed with a valid JSON response.

## Demo Behavior

The recorded demo shows a multi-step embodied search:

1. Step 0: robot view has no reliable target; Agent rotates right with low confidence.
2. Step 1: sofa is visible; Agent keeps scanning with low confidence.
3. Step 2: blue book and sofa are visible; Agent keeps scanning with low confidence.
4. Step 3: red cup, table, blue book, and sofa are visible; Agent stops with high confidence.

Final semantic check:

- Step count: 4.
- Pre-target confidence: 0.31.
- Final action: `STOP`.
- Final target visible: `red cup`.
- Final confidence: 0.948.

## Video Output

Generated video:

- `docs/demo_outputs/embodied_visual_search_demo.mp4`

Generated supporting files:

- `docs/demo_outputs/demo_summary.json`
- `docs/demo_outputs/frames/demo_frame_00.png`
- `docs/demo_outputs/frames/demo_frame_01.png`
- `docs/demo_outputs/frames/demo_frame_02.png`
- `docs/demo_outputs/frames/demo_frame_03.png`
- robot POV frames and top-down map frames for every step.

Automated video checks passed:

- Video resolution: 1280 x 720.
- Frame count: 8.
- Non-blank frame standard deviation checks passed.
- Output size is non-trivial.

## Web Demo

The running web app was verified at:

- `http://127.0.0.1:8000`

Verified endpoints:

- `/api/agent/audit`: HTTP 200.
- `/api/demo/run`: HTTP 200.
- `/`: contains the new `Robot POV + simulated room` demo UI and `Run Full Demo` control.

## Quality Notes

The earlier issue where the demo stopped immediately at step 0 has been fixed. The demo now starts away from the target, performs a search sweep, keeps low confidence while the target is not visible, and stops only after the red cup enters the robot view.

