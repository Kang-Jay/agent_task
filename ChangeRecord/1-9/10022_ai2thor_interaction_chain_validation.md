# 10022 AI2-THOR 交互动作链与严格后置条件验收

## 目标

继续 Plan2 中 AI2-THOR 交互动作链部分，确认并验收 `OpenObject -> PickupObject -> PutObject` 可以在现有项目链路中完成参数化执行、对象解析、手持状态跟踪、`receptacleObjectId` 绑定和严格后置条件校验。

本轮只允许核查和使用 interaction / executor / postcondition / runtime 相关链路；不修改 UI、memory、stream、mapping、evaluation，也不修改模型、超参或 action catalog。

## 关键结论

- `AI2ThorInteractionResolver` 已支持交互动作对象解析：
  - `OpenObject` 按 `openable=True` 和 `isOpen=False` 绑定可打开对象。
  - `PickupObject` 要求空手，并绑定 `pickupable=True` 的目标物体。
  - `PutObject` 将 `objectId` 标准化为目标 receptacle，而不是手持物体。
  - `PutObject` 支持 `receptacleObjectId`、`receptacleType`、`target`、`object`、`heldObject` 等语义字段，并会在执行前验证手持物体与目标容器不混淆。
- `execute_controller_action` 已按原生 AI2-THOR action 名称和参数转发到 Unity，不重写 `OpenObject` / `PickupObject` / `PutObject`，并记录 before/after metadata、inventory before/after、runtime success 和 error message。
- `AI2ThorPostconditionVerifier` 已对三类动作做严格后置条件：
  - `OpenObject`：目标对象必须从未打开变为打开，且不能由其他对象变化冒充成功。
  - `PickupObject`：指定目标必须干净进入 inventory，不能错拿、重复拿或凭空替换。
  - `PutObject`：唯一手持物体必须离开 inventory，并同时出现在目标 receptacle 的 `receptacleObjectIds` 和该物体的 `parentReceptacles` 中；只满足一侧登记会失败。
- `AI2ThorSessionManager.execute` 已将 runtime execution 与 semantic postcondition 串联，只有二者同时通过才算 committed。

## 配置冻结

已检查以下配置未被本轮修改：

- `configs/agent_config.json`
- `configs/ai2thor_actions_v5.json`

因此本轮没有自定义或篡改模型、超参、允许动作、AI2-THOR action catalog、微调设置或训练结构。

## 本地专项测试

已执行：

```powershell
python -B -m unittest discover -s tests -p 'test_ai2thor_interactions.py' -v
python -B -m unittest discover -s tests -p 'test_ai2thor_postconditions.py' -v
python -B -m unittest discover -s tests -p 'test_ai2thor_runtime.py' -v
python -B -m unittest discover -s tests -p 'test_ai2thor_session.py' -v
python -B -m unittest discover -s tests -p 'test_ai2thor_interaction_chain.py' -v
```

结果：

- `test_ai2thor_interactions.py`：20 tests OK
- `test_ai2thor_postconditions.py`：23 tests OK
- `test_ai2thor_runtime.py`：9 tests OK
- `test_ai2thor_session.py`：10 tests OK
- `test_ai2thor_interaction_chain.py`：3 tests OK

覆盖点包括：

- Open/Pickup/Put 连续链路对象绑定。
- PutObject 中 `objectId` 代表 receptacle，而不是 held object。
- `receptacleObjectId` alias 归一化。
- 非空 inventory 禁止 PickupObject。
- 无 inventory 禁止 PutObject。
- 多 inventory 禁止 PutObject。
- 手持物体与请求物体不一致时拒绝执行。
- 关闭容器不能 PutObject。
- runtime success 但放入错误容器时后置条件失败。
- 只更新 object parent 或只更新 receptacle child 任何一侧都不能通过。

## 本地全量回归

已执行：

```powershell
python -B -m unittest discover -s tests -v
python -B -m compileall -q src tests tools
git diff --check
git diff -- configs/agent_config.json configs/ai2thor_actions_v5.json
```

结果：

- `Ran 302 tests in 31.683s`
- `OK (skipped=2)`
- `compileall` 通过
- `git diff --check` 通过
- config freeze diff 为空

跳过的 2 个测试为现有 live model gate，不属于本轮交互动作链阻塞项。

## 真实 Unity 验收命令

在 3090GPU2 的项目目录 `/home/scale/kangjay/kaohe` 中执行：

```bash
cd /home/scale/kangjay/kaohe
PYTHONPATH=. .mamba-env/bin/python -B tools/validate_ai2thor_interaction_chain.py \
  --output-dir docs/ai2thor_outputs/interaction_chain_validation_plan2
```

验收预期：

- scene: `FloorPlan1`
- setup actions: `GetInteractablePoses`, `TeleportFull`
- chain: `OpenObject -> PickupObject -> PutObject`
- runtime configuration:
  - width `960`
  - height `540`
  - rotateStepDegrees `30.0`
  - snapToGrid `False`
  - renderInstanceSegmentation `True`
- interaction actions must all use `forceAction=False`
- output includes:
  - `00_initial.png`
  - `01_open_fridge.png`
  - `02_pickup_egg.png`
  - `03_put_egg_in_bowl.png`
  - `validation.json`
- `validation.json.status` must be `passed`
- final state must show:
  - Egg released from inventory
  - Egg parent includes Bowl
  - Bowl `receptacleObjectIds` includes Egg

## 当前边界

本轮完成的是 AI2-THOR 原生交互动作链的执行和校验底座，不等于完成全部 Plan2。以下仍是后续工作：

- 让高层任务规划稳定地产生多步交互 plan，例如“把花瓶放到纸箱里”自动拆解为 approach / pickup / approach / put。
- 将交互链接入网页流式 demo 的用户指令执行，而不仅是 session/runtime 和真实 Unity validation tool。
- 对“找到右边的门并走出去”补齐门语义、房间出口判断、跨房间或开放区域终止判据。
- 增加真实多场景评估，而不是只验证固定 fixture。

## 代码库清洁度

本轮没有创建临时文件、缓存文件或视频文件。新增记录仅为本 ChangeRecord。

当前仓库仍存在其他 Plan2 工作流留下的未提交改动和新增目录，包括 model reliability、memory、mapping、stream、UI、evaluation 等；它们不属于本轮交互动作链写集，未在本轮修改或回滚。
