# Plan2 Remaining Task F: Demo Recording and Video Verification

## Scope

This record defines the final recording and video verification procedure for the
Plan2 demos. It does not change model configuration, action configuration,
hyperparameters, prompt budget, training settings, or simulator settings.

The task is split into two evidence layers:

1. real AI2-THOR episode evidence from the live agent/runtime path;
2. optional cinematic rendering from the already generated episode summary.

The cinematic video is acceptable only as a presentation layer. It is not a
substitute for simulator metadata, postconditions, event stream logs, or task
success checks.

## Current Recording Chain

The live AI2-THOR demo path writes runtime artifacts under:

```text
docs/ai2thor_outputs/<session_id>/<episode_id>/
```

Expected files for each completed run:

```text
ai2thor_demo_summary.json
ai2thor_visual_search_demo.mp4
frames/
```

The live adapter currently creates the raw browser/demo video from
`DemoStep.frame_path` frames using `src.simulation.video_encoding.write_browser_compatible_mp4`.
That encoder first writes a temporary OpenCV `mp4v` source video, then transcodes
with ffmpeg to:

```text
codec: h264
pixel_format: yuv420p
movflags: +faststart
```

The cinematic renderer is:

```text
tools/make_cinematic_demo.py
```

It consumes an existing `ai2thor_demo_summary.json`, requires post-action
observations by default, renders a 1920x1080 HUD video, probes the generated MP4
with OpenCV, and writes a verification JSON containing:

- frame count;
- fps;
- width and height;
- codec and pixel format;
- source step count;
- action list;
- final action;
- post-action semantic status;
- sha256 hash;
- sampled frame statistics.

## Dependency Check

Required Python packages are pinned in `requirements.txt`:

```text
opencv-python==5.0.0.93
pillow==12.3.0
numpy==2.2.6
ai2thor==5.0.0
```

The encoder requires either system ffmpeg or `imageio-ffmpeg`.

Current local dependency observation:

```text
cv2: available
numpy: available
PIL: available
ai2thor: available
imageio_ffmpeg: available
system ffmpeg: not found
ffmpeg fallback: imageio-ffmpeg binary is available
```

Important local gap:

```text
requirements.txt pins ai2thor==5.0.0, but the current local Python import reports ai2thor 2.7.4.
```

This must be corrected in the active runtime environment before local live
AI2-THOR evidence is treated as authoritative. The remote 3090GPU2 environment
must also print and record the AI2-THOR version before final acceptance.

## Artifact Hygiene

The current `.gitignore` excludes generated video and bulky runtime artifacts:

```text
*.mp4
*.mov
*.avi
*.mkv
docs/**/frames/
docs/browser_recordings/frames*/
docs/cinematic_demo/frames/
docs/ai2thor_outputs/frames/
docs/demo_outputs/frames/
apikey.txt
.env
```

The final commit may include small JSON/Markdown evidence summaries when useful,
but must not include API keys, raw frames, generated videos, Unity caches, or
temporary ffmpeg/OpenCV files.

Before commit:

```powershell
git status --short
git diff --check
git hash-object configs/agent_config.json
Get-FileHash configs/agent_config.json -Algorithm SHA256
```

Expected `configs/agent_config.json` identity:

```text
git blob: e9311e26ec93dab9b28941b611d1324bd3cabdf5
sha256:   AD6E2EAC4BA087EB8188FD5FBC7EB4B0CD7ECA745A07220CE9447223DE2DE780
```

## Required Final Demo Tasks

The final two demos must be run through the actual agent/demo path, not only
through isolated scripted probes:

1. `找到右边的门，然后走出去`
2. `把花瓶放到纸箱里`

The first task is a navigation/cross-door task. It is not complete when a door is
only detected visually. Acceptance requires simulator evidence that the agent
crossed the selected right-side doorway.

The second task is an interaction chain task. It is not complete when the vase or
box is only detected visually. Acceptance requires ordered Unity interaction:

```text
PickupObject(vase) -> optional OpenObject(box) -> PutObject(box)
```

and postcondition evidence that the vase is no longer in inventory and is now in
the target receptacle.

## Live Episode Commands

First verify the runtime environment:

```powershell
python -B -m unittest discover -s tests -p "test_video_encoding.py" -v
python -B -m unittest discover -s tests -p "test_cinematic_demo_tool.py" -v
python -B -m unittest discover -s tests -q
```

Use this PowerShell-safe runtime probe instead:

```powershell
@'
import cv2
import ai2thor
from src.simulation.video_encoding import find_ffmpeg
print("cv2", cv2.__version__)
print("ai2thor", getattr(ai2thor, "__version__", "unknown"))
print("ffmpeg", find_ffmpeg())
'@ | python -
```

Run the live web service if it is not already running:

```powershell
python -B -m src.ui.app
```

Then run the two tasks through the same API/stream path used by the web demo.
The exact command depends on whether the service is launched locally or on
3090GPU2, but the evidence must include the returned `summary_path`,
`video_path`, `episode_id`, terminal event, and task success fields.

Minimum API evidence to save from each run:

```text
episode_id
instruction
scene
summary_path
video_path
task_success
terminal_reason
steps[*].action
steps[*].observation_phase
steps[*].robot_before / robot_after
steps[*].action_success
steps[*].completion_status
postcondition evidence for interaction actions
```

## Cinematic Rendering Commands

After each live episode has produced a valid `ai2thor_demo_summary.json`, render
the presentation video:

```powershell
python -B tools/make_cinematic_demo.py `
  --summary docs/ai2thor_outputs/<session_id>/<episode_id>/ai2thor_demo_summary.json `
  --output docs/cinematic_demo/<task_slug>_<episode_id>.mp4 `
  --verification docs/cinematic_demo/<task_slug>_<episode_id>.verification.json `
  --fps 24 `
  --hold-frames 28 `
  --intro-frames 48 `
  --outro-frames 48
```

Do not use `--allow-non-ai2thor` for final evidence.

Do not use `--allow-legacy-pre-action` for final evidence.

Use `--keep-frames` only for debugging a failed inspection. Kept frames must not
be committed.

## Video Verification Criteria

For every final demo video, inspect both the raw live video and the cinematic
video.

Hard pass criteria:

1. MP4 file exists and has non-zero size.
2. Video opens with OpenCV.
3. Video decodes all frames reported by `CAP_PROP_FRAME_COUNT`.
4. Width and height match the expected renderer:
   - live video: must match the adapter frame size;
   - cinematic video: `1920x1080`.
5. Cinematic verification reports:
   - `codec == "h264"`;
   - `pixel_format == "yuv420p"`;
   - `all_steps_ai2thor == true`;
   - `post_action_semantics == true`;
   - `frame_count == rendered source frame count`.
6. First, middle, and final decoded frames are visually non-blank.
7. The displayed action label matches the state change in the next/post-action
   observation.
8. The final frame agrees with simulator metadata and task-specific
   postconditions.
9. No misleading direction labels are present: `TURN_RIGHT` must correspond to
   the simulator's right rotation in the post-action frame sequence.
10. The video path, summary path, verification path, sha256, and final
    acceptance result are recorded in a ChangeRecord entry.

PowerShell-safe video probe:

```powershell
@'
import json
from pathlib import Path
import cv2

video = Path(r"docs/cinematic_demo/<task_slug>_<episode_id>.mp4")
cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise SystemExit(f"cannot open {video}")
reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
decoded = 0
means = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    decoded += 1
    if decoded in {1, max(1, reported // 2), reported}:
        means.append(float(frame.mean()))
cap.release()
print(json.dumps({
    "video": str(video),
    "reported_frames": reported,
    "decoded_frames": decoded,
    "width": width,
    "height": height,
    "sample_means": means,
}, indent=2))
if reported <= 0 or decoded != reported:
    raise SystemExit("frame decode mismatch")
if width <= 0 or height <= 0:
    raise SystemExit("invalid dimensions")
if any(mean <= 1.0 for mean in means):
    raise SystemExit("blank sampled frame")
'@ | python -
```

## Remote 3090GPU2 Acceptance Commands

After local tests pass and the local commit is pushed:

```powershell
ssh 3090GPU2 "cd /home/scale/kangjay/kaohe && git pull --ff-only"
ssh 3090GPU2 "cd /home/scale/kangjay/kaohe && git status --short --branch && git rev-parse HEAD"
ssh 3090GPU2 "cd /home/scale/kangjay/kaohe && .mamba-env/bin/python -B -m unittest discover -s tests -q"
ssh 3090GPU2 "cd /home/scale/kangjay/kaohe && .mamba-env/bin/python -B -m unittest discover -s tests -p 'test_video_encoding.py' -v"
ssh 3090GPU2 "cd /home/scale/kangjay/kaohe && .mamba-env/bin/python -B -m unittest discover -s tests -p 'test_cinematic_demo_tool.py' -v"
```

Record dependency versions on 3090GPU2:

```powershell
ssh 3090GPU2 "cd /home/scale/kangjay/kaohe && .mamba-env/bin/python -c \"import cv2, ai2thor; from src.simulation.video_encoding import find_ffmpeg; print('cv2', cv2.__version__); print('ai2thor', getattr(ai2thor, '__version__', 'unknown')); print('ffmpeg', find_ffmpeg())\""
```

If the service is managed manually:

```powershell
ssh 3090GPU2 "cd /home/scale/kangjay/kaohe && curl -fsS --max-time 10 http://127.0.0.1:8000/api/simulator/status"
```

Then run both final demos on the remote runtime, generate cinematic videos from
their summaries, and run the video probe above against the remote output paths.

## Mainline Gaps To Close

These gaps block claiming Plan2 final completion:

1. Local active AI2-THOR version currently appears inconsistent with
   `requirements.txt`; final runtime must use the pinned environment or record a
   justified environment correction.
2. The cinematic renderer is well tested, but it depends on a completed summary;
   final acceptance still needs live agent episodes for the two Chinese tasks.
3. The live raw video currently records `step.frame_path`; final review must
   confirm those frames are post-action composite frames, not stale pre-action
   frames.
4. The right-door and vase-to-box task semantics must be validated through the
   actual agent path, not only through deterministic object/action probes.
5. Final video inspection must include decoded-frame checks and visual review of
   first/middle/final frames.
6. The final ChangeRecord must include local and remote test outputs, episode
   paths, video hashes, and explicit pass/fail conclusions.

## Definition Of Done

Task F is complete only when:

1. local video encoding and cinematic tests pass;
2. local full regression passes;
3. remote 3090GPU2 full regression passes;
4. both final tasks complete in the live AI2-THOR agent/demo path;
5. raw and cinematic videos are generated for both tasks;
6. each video passes decode, dimensions, frame count, non-blank, action
   alignment, and final-state checks;
7. generated media remains uncommitted;
8. `configs/agent_config.json` hash remains unchanged;
9. GitHub, local, and 3090GPU2 point to the same final commit.
