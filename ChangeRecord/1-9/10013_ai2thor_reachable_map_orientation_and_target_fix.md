# ChangeRecord 10013: AI2-THOR Reachable Map Orientation and Target Fix

## Date

2026-07-10

## User-Visible Problem

The right-side visualization in the AI2-THOR demo appeared to disagree with
`TURN_LEFT` / `TURN_RIGHT`, and its object labels did not look like a coherent
navigation map.

The user observation was correct. Two visualization problems and one timing
ambiguity existed:

1. AI2-THOR world `+Z` was projected downward on the image.
2. The robot triangle used the same inverted screen-space sign.
3. The frame showed the observation before execution while the panel called
   the proposed command simply `Action`, which could be interpreted as already
   executed.

The original map also plotted AI2-THOR object centers as gray points and
labeled every visible object. Those points were not reachable navigation
positions, so labels such as `Floor`, `GarbageCan`, and duplicated
`Television` markers were misleading.

## Correct Semantics

The right-side image is now an audit-oriented reachable-space map:

- gray dots: positions returned by AI2-THOR `GetReachablePositions`;
- teal line / points: robot trajectory accumulated in the current episode;
- blue triangle: current robot position and heading;
- red target: the exact simulator object instance confirmed by instance
  segmentation;
- `Best candidate`: the confirmed target label and confidence.

It is not the robot camera input, a textured floor-plan reconstruction, or a
complete learned semantic map. The Agent receives the robot RGB observation;
the right-side map exists to make execution auditable.

## Coordinate Convention

AI2-THOR uses yaw with:

- `0 deg`: facing world `+Z`;
- positive yaw: clockwise rotation when viewed from above;
- `90 deg`: right;
- `180 deg`: down;
- `270 deg`: left.

The corrected screen projection maps world `+Z` upward:

```text
screen_y = top + (max_z - world_z) / z_range * map_height
```

The robot marker is locked by tests to:

```text
0 deg -> up
90 deg -> right
180 deg -> down
270 deg -> left
```

## Code Changes

Updated `src/simulation/ai2thor_adapter.py`:

1. Query `GetReachablePositions` once after controller initialization.
2. Record the pre-action robot state for each trajectory step.
3. Replace object-center dots with reachable navigation positions.
4. Invert the screen-space Z projection to match a conventional top-down map.
5. Add `_heading_triangle()` as the single heading conversion implementation.
6. Draw the accumulated robot path separately from reachable positions.
7. Carry the simulator `object_id` into the grounded target candidate.
8. Draw only the exact visible target instance after segmentation confirms it.
9. Do not display an unseen target from simulator metadata before detection.
10. Label the state as `Heading ... before <PLANNED_ACTION>`.
11. Rename the panel field from `Action` to `Planned action`.

No configured hyperparameters, stop thresholds, turn angles, action-space
settings, model settings, or pipeline stage definitions were changed.

## Tests

Updated `tests/test_ai2thor_sync.py`:

- AI2-THOR yaw-to-screen direction test;
- reachable-space, path, and target rendering test;
- unseen-target leakage prevention test;
- existing grounded-target and structured-thought synchronization tests.

Verification results:

- syntax compilation: passed;
- AI2-THOR map/synchronization tests: 7/7 passed;
- complete local test suite: 75 passed;
- guarded live paid-model test: 1 skipped by its explicit environment guard;
- `git diff --check`: passed.

## Real AI2-THOR Verification

Environment:

- server: `3090GPU2`;
- project: `/home/scale/kangjay/kaohe`;
- simulator: AI2-THOR 5.0.0;
- scene: `FloorPlan211`;
- service PID after deployment: `3323490`;
- local access through the existing SSH tunnel:
  `http://127.0.0.1:8000`.

The real task `Find the television in the room` completed in seven steps:

| Step | Planned action | Heading before action | Target candidate |
| --- | --- | ---: | --- |
| 0 | TURN_RIGHT | 90 | none |
| 1 | TURN_RIGHT | 120 | none |
| 2 | TURN_RIGHT | 150 | none |
| 3 | MOVE_FORWARD | 180 | none |
| 4 | TURN_RIGHT | 180 | none |
| 5 | INSPECT | 210 | Television |
| 6 | STOP | 210 | Television |

The sequence rotates clockwise on the corrected map. The repeated `180 deg`
between steps 3 and 4 is expected because step 3 is a forward movement, not a
rotation.

The confirmed object instance was:

```text
Television|-02.03|+01.08|+00.56
```

The first frame contains no target marker. The red target appears only after
AI2-THOR instance segmentation confirms that exact visible object.

## Browser Resource Verification

- root page: HTTP 200;
- final top-down PNG: HTTP 200, `image/png`;
- video range request: HTTP 206;
- video content type: `video/mp4`;
- byte range: `0-1023/371411`;
- codec: H.264;
- pixel format: `yuv420p`;
- resolution: 1600 x 900;
- duration: 7 seconds.

No controllable local browser window was available for an automated webpage
screenshot in this run. Generated first/final composite frames were inspected
directly, and all webpage resources were validated over the same HTTP endpoint.

## Remaining Limitation

The reachable-space map is a waypoint visualization, not a learned occupancy
or semantic map. A future research-grade version can add an Agent-owned
egocentric occupancy map and frontier exploration, but it must not use hidden
simulator object metadata before perception confirms an object.
