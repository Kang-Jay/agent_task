# 10029 Approach Micro-Alignment Stall Guard

## 背景

远端真实 `vase_into_box` run 在 `43481e4` 后仍失败，失败模式从“对不可见花瓶重复 `PickupObject`”变成了在 verified approach 阶段持续执行小角度 `TURN_LEFT` / `TURN_RIGHT`，最终耗尽 20 步且没有执行 `PickupObject` / `PutObject`。这说明 approach guidance 已经到达姿态微调阶段，但缺少停滞检测和退出条件。

## 修复

1. 在 `EmbodiedSearchAgent._verified_approach_action()` 中增加 approach 微调停滞检测。
2. 当最近连续多步都是 successful `verified_approach_navigation`，且动作只是在 5 度以内左右/上下反复微调时，不再继续执行相同 approach 建议。
3. 停滞后让主链路回到既有闭环：
   - 如果交互目标已经可执行，进入 verifier-guided interaction continuation。
   - 如果交互目标仍不可见或不可执行，回到 VLM planner，让真实视觉输入重新决策。

## 标准化验证

已新增 `test_stalled_micro_alignment_yields_back_to_model`，复现 `vase_into_box` 中小角度左右对齐循环后 `_verified_approach_action()` 必须返回 `None`，避免无限重复 approach action。

本地已通过：

```text
PYTHONPATH=. python -B -m unittest discover -s tests -p test_model_planner.py -v
32 tests OK

PYTHONPATH=. python -B -m unittest discover -s tests -p test_task_semantics.py -v
13 tests OK

PYTHONPATH=. python -B -m unittest discover -s tests -p test_final_agent_demo_runner.py -v
12 tests OK
```

## 后续验证

1. 只提交本次修改：`src/agent/controller.py`、`tests/test_model_planner.py`、本 ChangeRecord。
2. 推送后在 `3090GPU2:/home/scale/kangjay/kaohe` 拉取。
3. 重新运行真实 `vase_into_box` demo，确认是否进入 `PickupObject -> PutObject -> terminal`。
4. 若通过，再统一生成四个最终 demo：
   - 找到房间里的电视
   - 把花瓶放到箱子里
   - 找到右边的门，然后走出去
   - 找到房间里的沙发并坐下

