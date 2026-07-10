# 10019 Plan2 几何接近技能、严格执行验证与远端验收计划

## 1. 记录目的

本记录对应 `Plan_2_hierarchical_embodied_agent_upgrade.md` 中“找到沙发并坐下”任务的接近、执行与完成判定修复。

本轮目标不是宣称 Plan2 已完成，而是建立一个可审计、可重复验证的 AI2-THOR 几何导航上界基线，解决此前真实 Agent 运行中的核心故障：

1. VLM 能正确理解“找到沙发并坐下”的总体任务。
2. Agent 不再因为看见沙发就 `INSPECT -> STOP`。
3. Agent 必须先到达与同一 Sofa objectId 对齐的 AI2-THOR interactable pose。
4. 到达后执行 `Crouch`，以 AI2-THOR 支持的能力近似“坐下”。
5. `Crouch` 后必须在同一步验证 `agent.isStanding=False`。
6. 只有“接近证据 + Crouch 成功 + 姿态证据”同时成立时，任务才返回 `approximate_success`。

## 2. 基线与失败事实

### 2.1 代码基线

- 修改前本地、`origin/main` 和 3090 已提交代码均为：
  - `fb7d8b89988b757ce9584ff862ebbf7f1585d620`
- 修改前远端服务和独立 Sofa 验证已经证明：
  - AI2-THOR FloorPlan211 可正常启动。
  - Sofa 可通过 instance segmentation 绑定到具体 objectId。
  - `GetInteractablePoses` 可返回合法站立位姿。
  - 在该位姿执行 `Crouch` 后，AI2-THOR 返回 `isStanding=False`。
  - 同一步 TaskVerifier 可返回 `approximate_success`。

### 2.2 真实 Agent 失败事实

此前远端真实多模态 Agent 轨迹没有完成任务：

- 第 0 步：
  - Kimi `kimi-k2.6` 使用了视觉输入。
  - 模型正确判断需要先接近沙发。
  - 输出 `MOVE_FORWARD`。
  - Unity 执行失败，机器人位置没有改变。
- 第 1 步：
  - 模型响应缺少 `action.type`。
  - 系统进入规则 fallback，再次输出 `MOVE_FORWARD distance=1`。
  - Unity 再次执行失败。
- Agent 没有到达 interactable pose，也没有执行 `Crouch`。

因此当前根因已经从“提前 STOP”转为：

1. 缺少可执行的场景几何接近技能。
2. 单次前进距离过长，容易被小物体或障碍阻挡。
3. 不完整路径和畸形路径缺少严格门控。
4. 动作后置条件只能证明状态发生变化，不能证明方向和幅度正确。
5. 失败动作可能被重复执行。

## 3. 本轮结构设计

### 3.1 VLM 与几何技能职责

本轮保持分层 Agent 结构：

- VLM：
  - 理解自然语言总体任务。
  - 生成全局任务计划。
  - 判断任务包含 `locate -> approach -> crouch -> verify` 子目标。
- AI2-THOR 几何技能：
  - 只承担低层、可验证的 interactable-pose 接近。
  - 不替代总体任务理解。
  - 每一步重新读取真实 Unity 状态。
- TaskVerifier：
  - 不相信模型文字中的“任务完成”声明。
  - 只根据执行后的 simulator evidence 判定完成。

### 3.2 Oracle 边界

`GetInteractablePoses` 和 `GetShortestPathToPoint` 使用模拟器真值，因此本轮技能必须明确标记：

- `planner_source="simulator_oracle"`
- `fallback_reason="verified_approach_navigation"`
- `skill_call.name="APPROACH_TARGET"`

该实现属于：

- 可执行性验证基线。
- Oracle upper-bound。
- Verifier/执行闭环测试工具。

该实现不属于：

- 最终非 oracle zero-shot planner。
- RGB-D occupancy map。
- frontier exploration。
- value map。
- 论文级非 oracle 视觉导航结果。

不能将该基线的成功率与未来非 oracle Agent 成功率混合统计。

## 4. 低层接近技能实现

### 4.1 接近证据

`AI2ThorApproachVerifier` 继续使用官方 `GetInteractablePoses`。

接近成功要求当前 Agent 与同一目标 objectId 的候选位姿同时匹配：

- `x/y/z`
- yaw
- camera horizon
- standing 状态

可见目标或有限距离不能替代 interactable-pose 证据。

### 4.2 候选位姿处理

本轮不再盲目使用 `poses[0]`：

1. 过滤缺字段、非数值、`NaN`、`Infinity` 和无 standing 布尔值的候选。
2. 按 Agent 到候选位姿的平面直线距离稳定排序。
3. 从最近候选开始请求路径。
4. 若最近候选不可达或只返回 `PathPartial`，继续尝试下一个候选。
5. 选择第一个返回完整可执行路径的最近候选。

这是一种确定性的“最近可达候选”策略，不宣称全局最短或全局最优。

### 4.3 路径门控

只有以下两种状态可以生成低层动作：

- `PathComplete`
- `PoseAlignment`

以下状态不得驱动动作：

- `PathPartial`
- `PathInvalid`
- 空 status
- 空 corners
- 非列表 corners
- 缺失 x/z 的 corner
- `NaN`/`Infinity` corner
- 路径查询失败

### 4.4 短步执行

项目已有 AI2-THOR `gridSize=0.25`。

本轮没有修改该值，只将字面量统一为：

```python
DEFAULT_GRID_SIZE_METERS = 0.25
```

几何技能输出：

```text
MOVE_FORWARD distance=min(剩余路径段长度, 0.25)
```

这样每执行一个短步后都会重新：

1. 读取机器人新位姿。
2. 获取 interactable poses。
3. 查询路径。
4. 生成下一动作。

避免一次执行 1 米路径段导致碰撞后完全无法恢复。

### 4.5 姿态对齐

到达目标位置但 yaw 或 horizon 未对齐时，技能生成：

- `TURN_LEFT`
- `TURN_RIGHT`
- `LOOK_UP`
- `LOOK_DOWN`

方向使用 AI2-THOR 坐标约定：

- yaw `0°` 面向 `+Z`
- yaw `90°` 面向 `+X`
- yaw 正方向对应右转

`350° -> 20°` 的跨零度右转已经加入回归测试。

## 5. Controller 准入规则

Controller 只接受满足全部条件的 approach recommendation：

1. 当前任务 `completion_mode == "approximate_sit"`。
2. 当前尚未验证 approach 完成。
3. `source == "ai2thor_interactable_pose"`。
4. `path_status` 为 `PathComplete` 或 `PoseAlignment`。
5. approach objectId 非空。
6. approach objectId 必须属于自然语言任务对应的目标对象。
7. 推荐动作必须属于当前 task-conditioned action candidates。
8. `PathComplete` 只允许 Move/Turn。
9. `PoseAlignment` 只允许 Turn/Look。
10. 参数结构必须严格匹配动作：
    - Move：仅 `distance`
    - Turn/Look：仅 `angle`
11. 参数必须：
    - 为数值
    - 非 bool
    - finite
    - 大于 0

任何条件不满足时，系统回到既有 VLM/规则链路，不能伪装成 simulator-oracle 技能。

## 6. Oracle 信息隔离

传给模型的 `environment_context` 会移除：

- `matched_pose`
- `target_pose`
- `recommended_action`

模型仍可看到：

- 是否已验证 approach。
- objectId。
- evidence source。
- path status。

这样可以让模型知道任务阶段，但不能读取精确 oracle 位姿或复制 oracle 推荐动作并将其伪装为 `model_planner` 输出。

## 7. 失败循环保护

几何动作执行失败后，系统不会立即或隔着 `INSPECT` 原样重复相同动作。

最近窗口内：

- 若发现同类型、同参数的失败动作，拒绝相同 oracle recommendation。
- `INSPECT` 不视为导航状态变化。
- 只有成功的 Move/Turn/Look 导航动作证明状态改变后，才允许重新尝试同类动作。

该设计用于阻止：

```text
MOVE_FORWARD 失败
INSPECT
MOVE_FORWARD 失败
INSPECT
...
```

## 8. 严格动作后置条件

### 8.1 显式移动参数

当动作包含 `moveMagnitude` 时，必须同时满足：

1. AI2-THOR `lastActionSuccess=true`。
2. 实际平面方向与执行前 yaw 对应的动作方向一致。
3. lateral error 在既有位置容差内。
4. 实际移动距离与请求 `moveMagnitude` 一致。
5. before/after Agent pose 元数据完整且 finite。

因此以下情况会失败：

- `MoveAhead` 实际向后移动。
- `MoveAhead` 实际横向移动。
- 请求 0.25m 但只移动 0.10m。
- 只有 Y 轴抖动。
- after-agent 元数据缺失。

### 8.2 显式旋转参数

当动作包含 `degrees` 时：

- `RotateRight` 的 after yaw 必须等于 `before + degrees`。
- `RotateLeft` 的 after yaw 必须等于 `before - degrees`。
- 使用环形角差处理 `0°/360°`。

因此以下情况会失败：

- `TURN_RIGHT` 实际左转。
- 请求 30° 但只转 10°。
- yaw 元数据缺失或非有限值。

### 8.3 显式 Look 参数

- `LookDown` 要求 horizon 增加指定角度。
- `LookUp` 要求 horizon 减少指定角度。
- 错方向、部分角度和缺失元数据均失败。

### 8.4 AI2-THOR 默认参数兼容

AI2-THOR 允许部分导航动作省略参数，并使用 controller 默认值。

为兼容既有 Session API：

- 参数显式存在：执行严格方向和幅度验证。
- 参数省略：只验证状态确实发生变化，并在 evidence 中记录：
  - `used_controller_default=True`

几何 approach 技能始终提供显式参数，因此其验证仍然是严格的。

## 9. 配置与超参一致性

本轮未修改 `configs/agent_config.json`。

配置文件 Git object hash：

```text
e9311e26ec93dab9b28941b611d1324bd3cabdf5
```

实际配置保持：

- `default_turn_angle_degrees = 30`
- `max_steps = 20`
- `stop_confidence_threshold = 0.78`
- `target_visible_threshold = 0.58`
- AI2-THOR `gridSize = 0.25`

本轮未修改：

- 模型选择。
- temperature。
- max_tokens。
- API provider 优先级。
- 微调参数。
- 训练 epoch。
- batch size。
- optimizer。
- checkpoint。
- 数据集切分。
- TaskVerifier 成功定义。

本轮仍为 inference-only，不包含训练或微调步骤。

## 10. 测试过程

### 10.1 初始命令错误

首次使用：

```text
python -m unittest tests.test_xxx
```

由于本仓库 `tests` 目录不是 Python package，出现 3 个 `ModuleNotFoundError`。

该问题属于测试命令错误，不是功能失败。随后统一改用：

```text
python -B -m unittest discover -s tests -p "test_xxx.py" -v
```

### 10.2 中间真实回归

严格后置条件首次实现后，全量测试出现：

```text
test_execute_preserves_session_state: FAIL
```

根因：

- AI2-THOR 允许省略 `moveMagnitude/degrees`，使用 controller 默认值。
- 第一版严格 verifier 错误地要求参数必须显式存在。

修复：

- 显式参数继续严格验证。
- 参数省略时兼容 controller default，并验证状态发生变化。

修复后该会话测试重新通过。

### 10.3 最终分模块结果

- `test_ai2thor_approach.py`
  - 9/9 通过。
- `test_ai2thor_postconditions.py`
  - 12/12 通过。
- `test_model_planner.py`
  - 20/20 通过。
- `test_ai2thor_session.py`
  - 7/7 通过。
- `test_execution_commit.py`
  - 7/7 通过。
- `test_task_semantics.py`
  - 10/10 通过。

### 10.4 最终全量结果

```text
Ran 194 tests
OK (skipped=2)
```

两项跳过测试均为显式环境开关控制的真实付费模型 API 测试：

- `test_live_global_task_plan_uses_visual_input`
- `test_live_planner_returns_allowed_action`

默认测试流程不会静默调用付费 API。

### 10.5 工程检查

- `python -B -m compileall -q src tests`：通过。
- `git diff --check`：通过。
- 无 conflict marker。
- 无临时 Python 文件。
- 无新增未跟踪代码文件。
- `apikey.txt`、视频、frames、缓存和日志继续由 `.gitignore` 排除。
- 当前仅存在 Windows `core.autocrlf` 的 LF/CRLF 提示，不是 diff 空白错误。

## 11. 十路并行审查汇总

本轮有效汇总了十路独立只读审查，覆盖：

1. 路径坐标系与 yaw。
2. Controller oracle 准入。
3. approach 测试覆盖。
4. 真实 FloorPlan211 Sofa E2E。
5. Plan2 与 ChangeRecord 一致性。
6. 动作参数归一化。
7. approximate-sit 完成判定。
8. streaming 与模型调用证据。
9. Git/远端同步和敏感文件。
10. 标准化评估与动作后置条件。

审查发现并已处理：

- `PathPartial` 可驱动动作。
- 畸形 corner 未拒绝。
- 固定使用 `poses[0]`。
- 单步距离过长。
- 错误 objectId 可伪造 approach recommendation。
- 非有限参数。
- 失败动作循环。
- TURN 方向未验证。
- Move 方向和幅度未验证。
- Look 方向未验证。
- 缺失 Agent metadata 默认零值。
- controller default 参数兼容回归。
- 精确 oracle navigation payload 泄漏给 VLM。

审查仍确认的未完成事项见第 13 节。

## 12. 远端真实 Unity 验收步骤

本地代码提交后，3090 只允许执行：

```bash
cd /home/scale/kangjay/kaohe
git pull --ff-only
```

不得：

- `git reset --hard`
- `stash pop`
- `stash drop`
- 强制覆盖远端验证输出

远端已有 stash 和运行证据必须保留。

### 12.1 三端一致性

依次确认：

1. 本地 `git rev-parse HEAD`
2. GitHub `origin/main`
3. 3090 `/home/scale/kangjay/kaohe` HEAD

三者必须完全一致。

### 12.2 远端全量测试

```bash
PYTHONPATH=. .mamba-env/bin/python -B -m unittest discover -s tests -v
```

要求：

- 194 项或更多测试通过。
- 除显式 live API gate 外无跳过。
- 无失败。
- 无异常退出。

### 12.3 确定性真实 Unity 物理链路

先运行不依赖付费模型延迟的确定性 Agent/技能验证：

1. 场景：FloorPlan211。
2. 禁止 `TeleportFull`。
3. 禁止 `forceAction`。
4. 必须通过：
   - adapter
   - Agent
   - action executor
   - postcondition verifier
   - commit_execution
   - TaskVerifier
5. 至少一个 Move/Turn 动作真实成功并产生正确位姿变化。
6. 每一步重新查询路径。
7. 到达同一 Sofa objectId 的 interactable pose。
8. 执行 `Crouch`。
9. `isStanding=False`。
10. 同一步返回 `approximate_success`。
11. execution plan 状态为 `completed`。
12. 轨迹中不得出现 `TeleportFull`。

### 12.4 真实多模态模型 Agent

确定性物理链路通过后，再启用 `apikey.txt` 中配置的多模态模型。

要求：

1. 全局任务计划使用真实视觉输入。
2. 模型理解“找到沙发并坐下”不是纯目标检测。
3. 几何技能步骤明确显示 `simulator_oracle`。
4. 不将几何技能步骤显示成 VLM 隐藏思维链。
5. 动作实际成功。
6. 最终完成由 simulator evidence 判定。
7. 保存：
   - 每步 RGB。
   - before/after pose。
   - action 与 normalized args。
   - path status。
   - Sofa objectId。
   - runtime result。
   - postcondition result。
   - completion status。
   - execution plan。
   - 完整 stream event。

### 12.5 浏览器和视频检查

完成真实 episode 后：

1. 启动 UI 服务。
2. 使用本地 SSH 转发访问 `http://127.0.0.1:18000`。
3. 从浏览器真实运行任务。
4. 录制完整视频。
5. 逐帧检查：
   - POV 是动作前还是动作后。
   - TURN 标签与画面方向一致。
   - 地图位置与机器人位姿一致。
   - objectId 和目标标注一致。
   - `Crouch` 后画面和状态一致。
   - 最终完成条件与轨迹 JSON 一致。

## 13. 尚未完成的 Plan2 项目

本轮完成的是 oracle 几何执行基线，不代表 Plan2 完成。

仍需继续：

1. 非 oracle RGB-D occupancy map。
2. frontier exploration。
3. value map 和目标语义价值传播。
4. 非 oracle 接近与 oracle upper-bound 分开评估。
5. OpenObject -> PickupObject -> PutObject 的真实 Unity 连续闭环。
6. 更多交互动作的严格语义后置条件。
7. object/spatial/task/failure/skill/episode 分层 memory。
8. memory 可视化和引用证据。
9. 固定 scene/seed/初始 pose 的标准评估集。
10. oracle 与非 oracle 独立的 SPL、Success、Collision、False Success 指标。
11. streaming 的动作前后画面对齐。
12. streaming 心跳、模型耗时、request ID 和 token usage。
13. 前端 run_id 隔离和取消安全。
14. API deadline/retry/input/auth 加固。
15. 依赖锁定、第三方源码 manifest 和研究代码清理。
16. 浏览器真实演示和逐帧视频验收。

## 14. 当前退出判断

### 本地代码阶段

已满足：

- 低层几何 approach 实现完成。
- 严格路径门控完成。
- 多候选可达选择完成。
- 短步执行完成。
- Controller 准入完成。
- oracle payload 隔离完成。
- 失败循环保护完成。
- 严格方向/幅度后置条件完成。
- 配置和超参未擅自修改。
- 194 项全量测试通过，2 项 live API gate 跳过。
- compile 和 diff 检查通过。

### 整体项目阶段

尚未满足：

- 新代码尚未完成远端真实 Unity 自主 episode 验收。
- 尚未完成浏览器流式闭环。
- 尚未录制并逐帧检查新 demo。
- 尚未完成非 oracle Plan2 导航。

因此当前准确状态为：

> 本地几何 approach、严格执行验证和回归测试完成；可以进入提交、远端同步和真实 Unity 验收。不能宣称 Plan2 或最终视觉搜索 Agent 已完成。
