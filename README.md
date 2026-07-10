# Embodied Visual Search Agent

An inference-only multimodal embodied Agent for AI2-THOR. The system accepts
language, robot RGB observations and an optional clicked-object close-up,
then executes an auditable task plan through validated simulator actions.

## Core Pipeline

```text
language + RGB + optional clicked target
-> persistent task plan and ordered subgoals
-> multimodal action proposal
-> action catalog and object-binding validation
-> AI2-THOR execution
-> action postcondition verification
-> independent task predicate verification
-> continue, recover, replan or verified completion
```

The visible reasoning panel contains a decision summary and evidence, not
hidden model chain-of-thought.

## Current Semantics

- Model `Done` is only a completion proposal.
- Simulator action success is not task success.
- `OpenObject`, `PickupObject` and `PutObject` use real object IDs and
  postcondition checks.
- iTHOR has no native `SitOnObject`. “找到房间里的沙发并坐下” is therefore
  reported only as `approximate_success` after the same sofa is located,
  approached, `Crouch` succeeds and `agent.isStanding=false` is verified.
- The real AI2-THOR endpoint is strict by default. Local fallback is used only
  when the caller explicitly sends `allow_fallback=true`, and its backend is
  labelled `local_ppt_style_fallback`.

## Install and Test

```powershell
python -m pip install -r requirements.txt
python -B -m compileall -q src tests tools
python -B -m unittest discover -s tests -v
```

Live paid-model tests are opt-in:

```powershell
$env:RUN_LIVE_MODEL_TESTS='1'
python -B -m unittest discover -s tests -p test_live_model_integration.py -v
```

Credentials are read from environment variables or ignored `apikey.txt`:

- `OPENAI_API_KEY` or `MODEL_API_KEY`
- `OPENAI_BASE_URL` or `MODEL_BASE_URL`
- `MODEL_NAME`

## Run the Web App

```powershell
python -m src.ui.app
```

Open `http://127.0.0.1:8000`.

Important endpoints:

- `POST /api/demo/ai2thor/stream`: streamed real AI2-THOR execution
- `POST /api/demo/ai2thor/run`: non-streamed real AI2-THOR execution
- `POST /api/agent/step`: one Agent step
- `GET /api/agent/audit`: model and configuration audit
- `GET /api/simulator/actions`: action catalog
- `GET /api/simulator/status`: runtime diagnostics

## Real Unity Validation

The Windows environment may import an older AI2-THOR build. Project runtime
evidence must be produced on the Linux AI2-THOR 5.0.0 deployment:

```bash
cd /home/scale/kangjay/kaohe
PYTHONPATH=. .mamba-env/bin/python \
  tools/validate_ai2thor_interaction_chain.py
PYTHONPATH=. .mamba-env/bin/python \
  tools/validate_ai2thor_sofa_approximation.py
```

## Configuration Authority

`configs/agent_config.json` is the source of truth for pipeline stages, Agent
thresholds, vision, memory, evaluation and close-up rendering. The official
AI2-THOR action catalog is `configs/ai2thor_actions_v5.json`.

Do not tune prompts or thresholds against evaluation episodes. This project
does not train or fine-tune a model.

## Plans and Evidence

- `Plan_2_hierarchical_embodied_agent_upgrade.md`
- `ChangeRecord/1-9/10016_object_click_closeup_render.md`
- `ChangeRecord/1-9/10017_plan2_strict_execution.md`
- `research/references/embodied_agent_codebase_manifest.md`
