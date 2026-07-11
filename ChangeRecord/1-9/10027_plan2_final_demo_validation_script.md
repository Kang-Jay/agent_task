# Plan2 Final Demo Validation Script

## Scope
- Added a formal AI2-THOR acceptance script for the two final user-facing tasks:
  - `找到右边的门，然后走出去`
  - `把花瓶放到纸箱里`
- The script is inference-only and does not change model, training, fine-tuning, hyperparameters, or config files.
- Generated frames and validation JSON are runtime artifacts and must stay out of commits.

## Validation Contract
- Right-door exit succeeds only when the agent crosses the door threshold plane in `FloorPlan402`; seeing a door is not enough.
- Vase-to-box succeeds only when `PickupObject` and `PutObject` pass strict AI2-THOR postconditions and final metadata proves the vase is inside the box with empty inventory.

## Files
- `tools/validate_final_demo_tasks.py`
- `tests/test_final_demo_validation.py`

## Local Test Command
```powershell
python -B -m unittest discover -s tests -p test_final_demo_validation.py -v
```

## Remote Acceptance Command
```bash
cd /home/scale/kangjay/kaohe
.mamba-env/bin/python -B tools/validate_final_demo_tasks.py \
  --output-dir docs/ai2thor_outputs/final_demo_validation
```
