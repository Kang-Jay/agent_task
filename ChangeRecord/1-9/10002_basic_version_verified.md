# 10002 Basic Version Verified

## Implemented Basic Requirements

- Language instruction input is accepted by `AgentRequest.instruction`.
- First-person visual observation is accepted as a path, base64 image, or data URL.
- The Agent outputs a thought summary and a discrete action.
- Multi-round state is maintained by `SessionMemory`.
- Valid actions are loaded only from `configs/agent_config.json`.
- The normal web version is exposed by `src/ui/app.py`.

## Validation Evidence

Commands run:

```powershell
python -m src.data.generate_demo_dataset
python -m unittest discover -s tests -v
python -m src.evaluation.evaluator
```

Observed result:

- Demo dataset generated: 3 episodes.
- Unit tests: 5 passed.
- Offline evaluation: 3 / 3 successes.
- Illegal actions: 0.
- Average confidence: 0.9403.

## Fix Made During Validation

The first validation attempt failed because 3x3 grid mean color diluted small target objects into the background. The vision module was corrected to use local color-region connected components and crop-signature matching. After this fix, the normal version passed all tests.

