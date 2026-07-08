# 10004 Simulated Demo Upgrade Plan

## Problem Found

The earlier web page was an Agent control panel, not a PPT-style embodied demo. It had no visible robot, scene, first-person trajectory, or recorded interaction, so it was not sufficient as a final presentation demo.

## Corrective Goal

Build a complete demonstrable embodied visual search scene:

- Simulated indoor room.
- Robot first-person view.
- Top-down robot map.
- Multi-round thought/action/search loop.
- API key backed model adapter.
- Replay and generated video.
- Careful video inspection before final delivery.

## Implementation Choice

Use a lightweight deterministic simulator first. Heavy simulators such as AI2-THOR or Habitat are useful research references, but installing them can dominate the delivery and introduce GPU/Unity/data dependencies. The local simulator gives a reliable demo now while keeping the architecture ready for heavier simulator adapters later.

