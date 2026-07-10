# 10014 AI2-THOR interaction, startup, and Unity map validation

Date: 2026-07-10

## Scope

This change completes three linked runtime requirements:

1. Remove the AI2-THOR initialization timeout caused by incompatible launch settings.
2. Validate a real state-dependent `OpenObject -> PickupObject -> PutObject` chain.
3. Replace the procedural PIL overview with a real Unity-rendered global camera view.

## Initialization timeout root cause

The application configured:

- `rotateStepDegrees=30`
- `snapToGrid=True`

AI2-THOR 5.0.0 rejects this combination. Grid snapping only supports rotations
aligned to 0, 90, 180, 270, or 360 degrees. Unity logged the invalid parameter
combination, then raised a `NullReferenceException` while emitting metadata.
Python waited for the FIFO response until the 100 second server timeout.

The failure was reproduced with both 640x360 and 960x540 rendering. With the
same parameters except `snapToGrid=False`, both resolutions initialized in
approximately 2.6 seconds and returned 163 reachable positions.

## Runtime fix

- Added `src/simulation/ai2thor_runtime.py`.
- Preserved the configured 30 degree turn angle.
- Compute `snapToGrid` from the actual agent mode and rotation compatibility.
- Use the same launch rule in the streamed demo and the live session manager.
- Retain a partially initialized Controller and call `stop()` on initialization
  failure to avoid leaving a Unity child process behind.

No model, training, vision, or task hyperparameter was changed.

## Real interaction validation

Added `tools/validate_ai2thor_interaction_chain.py`.

Fixture:

- Scene: `FloorPlan1`
- Initial state: Egg inside a closed Fridge
- Final target: Egg inside a Bowl on the CounterTop

Validated chain:

1. `OpenObject(Fridge)` and verify `Fridge.isOpen == true`.
2. `PickupObject(Egg)` and verify the Egg enters `inventoryObjects`.
3. `PutObject(Bowl)` and verify:
   - the Egg leaves inventory;
   - the Egg lists the Bowl in `parentReceptacles`;
   - the Bowl lists the Egg in `receptacleObjectIds`.

All three interaction actions ran with `forceAction=false`.
`GetInteractablePoses` and `TeleportFull` are used only as deterministic test
setup actions and are not exposed as autonomous Agent skills.

`PutObject` postcondition validation was strengthened so inventory release alone
is insufficient. The released object must be registered in the requested
receptacle in both directions.

Remote evidence:

- `docs/ai2thor_outputs/interaction_chain_validation/validation.json`
- `docs/ai2thor_outputs/interaction_chain_validation/00_initial.png`
- `docs/ai2thor_outputs/interaction_chain_validation/01_open_fridge.png`
- `docs/ai2thor_outputs/interaction_chain_validation/02_pickup_egg.png`
- `docs/ai2thor_outputs/interaction_chain_validation/03_put_egg_in_bowl.png`

## Unity 3D global map

The demo now performs this once after scene initialization:

1. `GetMapViewCameraProperties`
2. `AddThirdPartyCamera` with the returned position, rotation,
   orthographic projection, and orthographic size.

Every later AI2-THOR Event provides the live third-party camera frame. The demo:

- center-crops the 16:9 orthographic frame to a square;
- preserves real Unity geometry, materials, lighting, and object states;
- projects the Agent trajectory using the camera center and orthographic size;
- overlays the Agent heading using AI2-THOR clockwise yaw semantics;
- overlays only visually grounded targets;
- records `map_view_source=unity_third_party_camera`.

The original PIL map remains only as an explicit
`procedural_2d_fallback`. A `map_camera_fallback` stream event includes the
failure reason if Unity camera creation is unavailable.

## Tests

Local:

- 39 AI2-THOR focused tests passed.
- 118 full tests passed.
- 1 paid live-model test remained intentionally skipped.

Remote 3090GPU2:

- 118 full tests passed.
- Real interaction validation passed.
- Stream validation passed:
  - first event: 114.6 ms;
  - 19 ordered events;
  - `map_camera_ready` present;
  - no `map_camera_fallback`;
  - two completed steps;
  - both steps used `unity_third_party_camera`;
  - terminal event: `episode_completed`;
  - total duration: 10.34 seconds.

Remote stream evidence:

- `docs/ai2thor_outputs/stream_startup_validation.json`
- `docs/ai2thor_outputs/stream-map-validation/<episode_id>/`

## Runtime endpoint

The service runs on the remote host at `127.0.0.1:8000`.
With the existing SSH tunnel, the browser endpoint is:

`http://127.0.0.1:18000`
