# 10021 Plan2 Remaining Six Workstreams

## 1. Purpose

This record tracks the implementation and verification of the six remaining
Plan2 workstreams. It is intentionally created before implementation so that
planned work, completed work, failed experiments, and accepted evidence remain
separate.

The workstreams are:

1. non-oracle RGB-D occupancy mapping, frontier extraction, and semantic value
   based exploration;
2. parameterized AI2-THOR interaction execution and strict postconditions,
   including an OpenObject -> PickupObject -> PutObject chain;
3. hierarchical embodied memory with auditable retrieval evidence;
4. versioned streaming events, run isolation, cancellation, and terminal error
   semantics;
5. a frozen multi-scene inference-only evaluation manifest and complete metrics;
6. API reliability, dependency/research manifests, and a post-action-compatible
   cinematic demo generator.

After those gates pass, two real AI2-THOR instructions must be executed:

- `找到右边的门，然后走出去`
- `把花瓶放到纸箱里`

## 2. Frozen Baseline

- Local branch: `main`
- Baseline commit: `768c17329c92a46b7d50e03906721fda65b1cdd5`
- Remote project: `3090GPU2:/home/scale/kangjay/kaohe`
- Remote baseline commit: `768c17329c92a46b7d50e03906721fda65b1cdd5`
- Agent config Git object:
  `e9311e26ec93dab9b28941b611d1324bd3cabdf5`
- Agent config SHA-256:
  `AD6E2EAC4BA087EB8188FD5FBC7EB4B0CD7ECA745A07220CE9447223DE2DE780`
- Baseline local regression: 207 tests passed, 2 live-model tests skipped.
- Formal route: inference-only. No training, fine-tuning, checkpoint generation,
  or evaluation-data leakage is permitted.

The model, provider, prompt contract, temperature, max tokens, API timeout,
navigation step size, turn angle, stop threshold, visibility threshold, and
memory capacities remain frozen unless a separately recorded experiment proves
that a change is necessary and compatible with Plan2.

## 3. Dependency Order

The six workstreams may be implemented in parallel only inside disjoint file
boundaries. Integration and acceptance remain sequential:

1. freeze the baseline and inspect concurrent changes;
2. implement and unit-test each isolated module;
3. integrate mapping and memory into planning without exposing hidden simulator
   truth to the non-oracle decision path;
4. integrate parameterized interaction execution and postcondition verification;
5. integrate versioned streaming only after backend result semantics are stable;
6. run the fixed evaluation manifest and metric validation;
7. run compile, whitespace, secret, artifact, and full regression gates;
8. push one reviewed commit series to GitHub;
9. pull the exact commits on 3090GPU2 and repeat the regression suite;
10. execute the two requested real Unity tasks;
11. inspect episode JSON, Unity metadata, stream events, screenshots, and decoded
    video before claiming success.

No later gate may be marked complete when an earlier dependency is failing.

## 4. Module Test Gates

Every workstream must pass the following sequence:

1. schema and input validation tests;
2. deterministic pure unit tests;
3. subsystem tests with controlled simulator/model doubles;
4. integration tests across the real repository interfaces;
5. negative and failure-path tests;
6. full local regression;
7. real Unity validation on 3090GPU2 where the capability depends on AI2-THOR;
8. artifact and trace inspection.

Generated videos, raw frames, API keys, caches, model responses, and Unity caches
must remain ignored and uncommitted.

## 5. Task-Specific Acceptance: Exit Through Right Door

The instruction must not complete when a door or doorway is merely detected.
The accepted execution must contain:

1. visual grounding of the right-side door or doorway;
2. a structured plan that separates locating, approaching, crossing, and
   verifying exit;
3. navigation actions that move the agent through the selected right-side
   threshold;
4. post-action pose evidence showing the threshold was crossed;
5. a completion predicate that remains incomplete before crossing;
6. no STOP/Done decision based only on target visibility;
7. a real Unity trajectory and decoded video whose turn directions and final
   position agree with metadata.

If the selected AI2-THOR scene has no traversable right-side doorway, the run
must fail explicitly and a scene satisfying the fixed task manifest must be
selected. The system must not silently reinterpret the task as visual search.

## 6. Task-Specific Acceptance: Put Vase Into Cardboard Box

The instruction must not complete when the vase or box is merely visible. The
accepted execution must contain:

1. separate grounding of the source object and destination receptacle;
2. ordered subgoals for approach, pickup, receptacle preparation when required,
   placement, and verification;
3. `PickupObject` with the bound vase objectId;
4. `OpenObject` only if the selected receptacle is openable and currently closed;
5. `PutObject` with the bound box receptacleObjectId;
6. inventory transition proving the vase was held and then released;
7. bidirectional receptacle evidence proving that the same vase is contained by
   the same box;
8. strict failure if the vase is released into a different receptacle;
9. real Unity evidence and a decoded video for the complete chain.

## 7. Status

- Baseline freeze: complete.
- Six implementation workstreams: in progress.
- Local integration: not started.
- Remote integration: not started.
- Right-door task: not started.
- Vase-to-box task: not started.
- Plan2 completion: not claimed.

This file must be updated with exact changed files, commands, test counts,
commits, episode identifiers, artifact hashes, failures, and remaining
limitations as work progresses.
