# Quick Start

## 1. Verify the Reviewed Local Tree

```powershell
cd D:\cache\SummerCap\kaohe\zju
git status --short
git diff --check
python -B -m compileall -q src tests tools
python -B -m unittest discover -s tests -v
```

Do not proceed to deployment while the suite fails. Live model tests are
separate because they make paid network requests:

```powershell
$env:RUN_LIVE_MODEL_TESTS='1'
python -B -m unittest discover -s tests -p test_live_model_integration.py -v
```

## 2. Run Locally

```powershell
python -m src.ui.app
```

Open `http://127.0.0.1:8000`.

The local host is suitable for API/UI work and deterministic unit tests. It
is not accepted as real Unity evidence when it loads AI2-THOR 2.7.4.

## 3. Synchronize Through Git

Review and stage explicit files. Do not use `git add .`.

```powershell
git diff --name-only
git diff --cached --check
git commit -m "feat: add hierarchical task planning and task verification"
git push origin main
```

On `3090GPU2`, preserve any dirty worktree before pulling:

```bash
cd /home/scale/kangjay/kaohe
git status --short
git stash push -u -m pre-plan2-sync
git pull --ff-only origin main
git rev-parse HEAD
```

Keep the stash until the pulled commit and tests prove that all collaborator
changes are present. Do not use reset or checkout to discard remote work.

## 4. Validate Real AI2-THOR

```bash
cd /home/scale/kangjay/kaohe
PYTHONPATH=. .mamba-env/bin/python -m unittest discover -s tests -v
PYTHONPATH=. .mamba-env/bin/python \
  tools/validate_ai2thor_interaction_chain.py
PYTHONPATH=. .mamba-env/bin/python \
  tools/validate_ai2thor_sofa_approximation.py
```

Required facts:

- `OpenObject -> PickupObject -> PutObject` changes real Unity state.
- Sofa detection alone cannot finish the task.
- Sofa approximation requires proximity, successful `Crouch` and
  `isStanding=false`.
- Completion is labelled `approximate_success`, not exact sitting.

## 5. Start the Remote Service

```bash
cd /home/scale/kangjay/kaohe
.mamba-env/bin/python -m src.ui.app
```

The service listens on remote `127.0.0.1:8000`. From Windows:

```powershell
ssh -N -L 18000:127.0.0.1:8000 3090GPU2
```

Open `http://127.0.0.1:18000`.

## 6. Runtime Rules

- Use `/api/demo/ai2thor/stream` for the streamed real-Unity demo.
- Strict AI2-THOR is the default; do not set `allow_fallback=true` when
  collecting acceptance evidence.
- `apikey.txt`, videos, frames, logs, caches and downloaded research source
  trees must remain ignored.
- The Agent exposes structured decision summaries and verifier evidence, not
  hidden chain-of-thought.
- Configuration and model parameters must come from the existing config and
  provider adapter; do not introduce unrecorded values.

Detailed phase gates and evidence are in
`ChangeRecord/1-9/10017_plan2_strict_execution.md`.
