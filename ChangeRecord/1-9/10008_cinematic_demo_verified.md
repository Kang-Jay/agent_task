# 10008 Cinematic AI2-THOR Demo Verified

## 1. Goal

Create a cooler demo video on top of the verified strict AI2-THOR trajectory, while preserving factual correctness:

- Use real AI2-THOR frames, not the local fallback.
- Keep the original 7-step embodied search trajectory.
- Add a higher-impact presentation layer: title sequence, HUD, scan beam, confidence bar, trajectory timeline, target lock box, and completion card.
- Verify the video programmatically and manually inspect representative frames.

## 2. Source Evidence

The cinematic video is generated from the already verified strict AI2-THOR run:

- Summary: `docs/ai2thor_outputs/ai2thor_demo_summary.json`
- Source simulator video: `docs/ai2thor_outputs/ai2thor_visual_search_demo.mp4`
- Source frames: `docs/ai2thor_outputs/frames/`

The source trajectory remains:

| Step | Backend | Scene | Action | Confidence | Evidence |
|---:|---|---|---|---:|---|
| 0 | ai2thor | FloorPlan211 | TURN_RIGHT | 0.570 | Floor, ShelvingUnit |
| 1 | ai2thor | FloorPlan211 | TURN_RIGHT | 0.570 | Floor |
| 2 | ai2thor | FloorPlan211 | TURN_RIGHT | 0.570 | Floor |
| 3 | ai2thor | FloorPlan211 | MOVE_FORWARD | 0.570 | Floor, GarbageCan |
| 4 | ai2thor | FloorPlan211 | TURN_RIGHT | 0.570 | Floor |
| 5 | ai2thor | FloorPlan211 | INSPECT | 0.943 | Television segmented |
| 6 | ai2thor | FloorPlan211 | STOP | 0.943 | Television segmented |

## 3. New Output

Cinematic video:

- `docs/cinematic_demo/ai2thor_cinematic_visual_search_demo.mp4`

Generation script:

- `tools/make_cinematic_demo.py`

Verification report:

- `docs/cinematic_demo/cinematic_demo_verification.json`

Representative inspected frames:

- `docs/cinematic_demo/frames/cinematic_0000.png`
- `docs/cinematic_demo/frames/cinematic_0146.png`
- `docs/cinematic_demo/frames/cinematic_0291.png`

## 4. Visual Enhancements

The cinematic layer adds:

- 1080p title sequence.
- Real AI2-THOR POV as the main visual surface.
- Top-down map panel.
- Animated scan beam over the POV image.
- Action HUD with current step, action, thought, and confidence.
- Confidence progress bar.
- Full trajectory timeline.
- Target lock overlay on final simulator-grounded Television bbox.
- Completion card with final status.
- Badges for `REAL AI2-THOR`, `INSTANCE SEGMENTATION`, and `NO FALLBACK`.

## 5. Automated Verification

Command:

```powershell
python -B -m py_compile tools/make_cinematic_demo.py
python -B tools/make_cinematic_demo.py
```

Result:

- Script compilation passed.
- Video generation passed.
- All script assertions passed.

Verification metrics:

```json
{
  "video_path": "docs\\cinematic_demo\\ai2thor_cinematic_visual_search_demo.mp4",
  "exists": true,
  "bytes": 8046001,
  "frame_count": 292,
  "fps": 24.0,
  "width": 1920,
  "height": 1080,
  "duration_seconds": 12.17,
  "source_steps": 7,
  "all_steps_ai2thor": true,
  "final_action": "STOP",
  "final_confidence": 0.943,
  "final_best_candidate": {
    "label": "Television",
    "bbox": [780, 315, 960, 540],
    "confidence": 0.943,
    "color_name": "segmentation",
    "region": "bottom right",
    "reason": "AI2-THOR instance segmentation matched the requested target object"
  }
}
```

Assertions checked:

- Video file exists.
- Video size is nontrivial.
- Frame count matches generated frame list.
- Resolution is exactly 1920 x 1080.
- Every source step is `ai2thor`.
- Final action is `STOP`.
- Final target label is `Television`.

## 6. Manual Visual Inspection

Inspected frames:

1. `cinematic_0000.png`
   - Title card is readable.
   - Goal and demo type are clear.

2. `cinematic_0146.png`
   - Main AI2-THOR scene is visible.
   - HUD shows the active embodied action.
   - Timeline and confidence bar are readable.

3. `cinematic_0291.png`
   - Final completion overlay no longer collides with the header.
   - Final STOP state is visible.
   - Target lock is visible over the Television bbox.
   - The frame still clearly shows real AI2-THOR content.

## 7. Final Status

The cinematic demo is complete and verified.

This output is a presentation-quality superset of the strict AI2-THOR demo. It does not fabricate capabilities: all simulator frames and trajectory decisions are sourced from the verified AI2-THOR run, while the cinematic layer only improves visual communication.
