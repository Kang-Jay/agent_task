# ChangeRecord 10012: Browser Video H.264 Fix

## Date

2026-07-10

## Problem

The AI2-THOR timeline and generated frames loaded correctly, but the left-side
HTML5 video player remained black and displayed `0:00`.

## Root Cause

The remote video was valid and the static route returned HTTP 200 with
`Content-Type: video/mp4`, but the video stream used:

- codec: MPEG-4 Part 2 (`mp4v`);
- pixel format: `yuv420p`;
- resolution: 1600 x 900.

Chrome and Edge do not reliably support MPEG-4 Part 2 inside an HTML5 MP4 video
element. The browser therefore could not decode the stream even though the file
was downloadable.

## Immediate Remote Repair

The existing remote video was transcoded with:

- codec: H.264 (`libx264`);
- profile: High;
- pixel format: `yuv420p`;
- fast-start metadata: enabled.

Verified remote output:

- duration: 7.0 seconds;
- size: 349,431 bytes;
- HTTP content type: `video/mp4`;
- byte-range response: `206 Partial Content`;
- `Accept-Ranges: bytes`;
- `Content-Range: bytes 0-1023/349431`.

Remote service:

- path: `/home/scale/kangjay/kaohe`;
- process: `./.mamba-env/bin/python -m src.ui.app`;
- PID recorded in `ui_server.pid`;
- audit endpoint: healthy.

## Permanent Code Fix

New module:

- `src/simulation/video_encoding.py`

The module:

1. writes a temporary frame sequence with OpenCV;
2. locates system `ffmpeg` or bundled `imageio-ffmpeg`;
3. transcodes to H.264 with `yuv420p`;
4. adds MP4 fast-start metadata;
5. removes the temporary source file;
6. fails explicitly if H.264 cannot be produced.

Updated generators:

- `src/simulation/ai2thor_adapter.py`;
- `src/simulation/room_simulator.py`;
- `tools/make_cinematic_demo.py`.

The remote files were updated minimally rather than replacing the older remote
simulation files wholesale.

## Tests

New:

- `tests/test_video_encoding.py`

Coverage:

- H.264 output metadata;
- four expected encoded frames;
- full OpenCV decode;
- ffmpeg reports `Video: h264`;
- temporary source file cleanup;
- empty frame-list rejection.

Results:

- video encoding tests: 2/2 passed;
- AI2-THOR synchronization tests: 4/4 passed;
- multimodal simulator tests: 11/11 passed;
- complete local suite: 72 passed, 1 guarded live test skipped;
- `git diff --check`: no patch errors.

## User Verification

Open:

`http://127.0.0.1:8000`

Use `Ctrl+F5` once to clear the previously failed cached video response. The
left-side player should then show a 7-second video and allow normal playback and
seeking.
