# 10018 Plan 2：真实接近证据与同一步完成判定

## 1. 本轮目标

本轮只处理会直接导致“找到沙发后立即蹲下/停止”以及“规划了移动但 Unity 不移动”的根因，不扩大到未经验证的空间建图、长期对象记忆或训练流程。

执行原则：

1. 每个模块独立修改、独立测试，通过后才进入下一模块。
2. 不新增模型超参数、训练超参数或任务阈值。
3. AI2-THOR 参数必须来自当前固定动作目录或仓库内已经通过真实 Unity 验证的脚本。
4. VLM 只能提出动作；任务是否完成由确定性 verifier 决定。
5. 保留第三方研究仓库为只读参考，不从 `research/codebases` 直接 import 生产代码。
6. 不覆盖其他作者的修改，不清理未确认用途的研究资产或答辩证据。

## 2. 十路并行审查结论

本轮恢复并完成十路只读审查，主代理负责交叉核对和最终实现。

| 审查方向 | 主要结论 | 本轮处理 |
|---|---|---|
| 几何搜索 | 当前九宫格 `search_map` 不是 occupancy/frontier 地图；非 oracle RGB-D 链路尚未实现 | 记录为后续 Phase 6 阻断项 |
| 开放词汇感知 | heuristic vision 会把未知目标误判为大色块；visual-search demo 仍依赖 segmentation oracle | 记录为 production/non-oracle 分轨阻断项 |
| 分层 memory | 当前仅为 step log + SQLite 检索，不是对象/空间/失败/技能分层记忆 | 本轮仅修复重复 commit 和 event 对齐 |
| 流式 UI | worker 异常可能产生空流，存在事件乱序、XSS、取消确认和 run 隔离问题 | 记录为 UI 流协议阻断项 |
| 固定评估 | evaluator 仍是单步合成评估，成功口径与 `TaskVerifier` 不统一 | 记录为 inference-only 评估阻断项 |
| 全动作空间 | 动作目录覆盖不等于安全闭环；`forceAction`、actor 权限、对象绑定、未注册 postcondition 仍需治理 | 本轮修复抽象导航参数和 Crouch postcondition |
| 模型 API | 视觉 provider 路由、总体 deadline、SDK retry、输入边界和公开 smoke-test 存在风险 | 记录为 API 稳定性与安全阻断项 |
| 仓库清洁 | 依赖未锁定、文档重号、敏感文件历史和远端目录需复核 | 提交前执行专门审计 |
| 研究代码复用 | 六个核心仓库 commit 已固定；生产代码不得直接 import；VLFM/VLMaps/AriGraph 只能按协议重写 | 保持现有隔离边界 |
| 完成判定 | approach 没有生产者、动作后不复验、重复 commit、历史 Crouch 可能错误满足任务 | 本轮重点完成 |

## 3. 修复一：抽象动作参数映射

### 3.1 根因

Agent 规划器输出：

- `MOVE_FORWARD {"distance": ...}`
- `TURN_RIGHT {"angle": ...}`

AI2-THOR 5.0 官方动作目录要求：

- `MoveAhead {"moveMagnitude": ...}`
- `RotateRight {"degrees": ...}`

旧实现只映射动作名，不映射参数名，因此动作会在目录校验阶段失败，机器人实际不会移动。

### 3.2 实现

在 `src/simulation/ai2thor_actions.py` 增加统一的原生参数归一化：

- `distance -> moveMagnitude`
- `angle -> degrees`

映射由 action catalog 在校验前执行，适用于所有调用者，不在 controller 或 UI 中重复实现。

### 3.3 测试门禁

- 抽象 Move 参数能够匹配 `MoveAhead` overload。
- 抽象 Turn 参数能够匹配 `RotateRight` overload。
- executor 最终传给 controller 的参数只能是 AI2-THOR 原生参数。

定向结果：`tests/test_ai2thor_actions.py` 11/11 通过。

## 4. 修复二：严格 approach 证据

### 4.1 根因

旧完成规则将“目标可见且 metadata 有有限 distance”直接视为 `approach_verified=true`。这会导致机器人在房间远处看到沙发后立刻执行 Crouch。

### 4.2 新契约

`approach_target` 只接受结构化证据：

```json
{
  "verified": true,
  "objectId": "Sofa|...",
  "source": "ai2thor_interactable_pose"
}
```

同时必须满足：

1. `objectId` 属于当前任务指令目标类型。
2. 证据来自 AI2-THOR `GetInteractablePoses`。
3. 当前 agent position、rotation、camera horizon、standing 状态与返回候选位姿一致。
4. 单纯 `distance` 不再具有完成证明力。

### 4.3 实现

新增 `src/simulation/ai2thor_approach.py`：

- 使用官方 `GetInteractablePoses`。
- 请求 `standings=[当前 isStanding]`。
- 保留既有真实 Unity 验证脚本使用的 `maxPoses=64`，未新增自定义数值。
- 数值匹配复用 `AI2ThorPostconditionVerifier` 已有 position/angle epsilon。
- 查询失败、无候选、姿态字段缺失或位姿不匹配均返回 `verified=false`。

AI2-THOR 主循环在以下两个时间点生成证据：

1. 调用 Agent 规划前。
2. 动作执行并取得 post-action metadata 后。

目标 ID 只在当前观测已绑定目标后保存；Crouch 完成判定可携带该动作执行前、同一目标的 approach witness，避免蹲下改变相机高度后丢失目标证据。

### 4.4 测试门禁

- 匹配交互位姿通过。
- 错位姿拒绝。
- 缺少 `isStanding` 拒绝。
- `GetInteractablePoses` 失败拒绝。
- approach objectId 与指令目标不一致拒绝。
- 只有 distance、没有 approach witness 拒绝。

定向结果：

- `tests/test_ai2thor_approach.py` 4/4 通过。
- `tests/test_task_semantics.py` 10/10 通过。
- `tests/test_task_verifier.py` 4/4 通过。

## 5. 修复三：阻止提前 Crouch/Done/STOP

控制器现在对 `approximate_sit` 使用明确门禁：

1. 未定位目标：继续搜索。
2. 已定位但 `approach_verified=false`：禁止 Crouch、Done、STOP，继续移动或探索。
3. approach 已验证但 Crouch 未执行：允许执行 Crouch。
4. Crouch postcondition 未通过：保持任务进行中。
5. verifier 全部谓词通过：报告 `approximate_success`。

新增回归：

- VLM 看到沙发后提出 Done，重规划为 MOVE_FORWARD。
- VLM 未接近时直接提出 Crouch，重规划为 MOVE_FORWARD。

定向结果：`tests/test_model_planner.py` 14/14 通过。

## 6. 修复四：动作后同一步复验

### 6.1 根因

旧流程在动作执行前计算 `completion_status`，动作执行后只写 memory，不重新运行 verifier。最后一步即使成功 Crouch，UI 和 execution plan 也至少滞后一轮。

### 6.2 新流程

```text
Agent proposal
-> action catalog validation
-> Unity execution
-> action postcondition
-> commit actual execution
-> verify with post-action environment context
-> update current step completion_status
-> update execution plan
-> decide done
```

`EmbodiedSearchAgent.commit_execution()` 现在返回：

- post-action `completion_status`
- post-action `execution_plan`
- 最终 `done`

AI2-THOR adapter 使用该最终结果构建视频帧、流式事件和 episode 终止状态。

### 6.3 memory 一致性修复

- `(session_id, step_id)` 提交保持幂等。
- 未提供 step_id 时，重复提交最新 step 会显式报错。
- `long_term_events` 保存 step_id，并按 session + step 精确回写，避免旧 step 提交污染新 event。
- 当前 step 保存最终 completion status 和 post-action environment context。

### 6.4 测试门禁

- 成功 Crouch 在同一步成为 `approximate_success`。
- 失败 Crouch 同一步保持 incomplete。
- execution plan 在同一步变为 completed。
- 重复提交拒绝且 SQLite 不产生重复记录。
- 提交旧 step 不修改更新的 pending step。

定向结果：`tests/test_execution_commit.py` 7/7 通过。

## 7. 修复五：Crouch postcondition

旧实现把缺失的 `agent.isStanding` 通过 `bool(None)` 转成 false，可能错误判定 Crouch 成功。

新实现要求：

- `isStanding` 必须存在。
- 类型必须为 bool。
- Crouch 后必须为 false。
- Stand 后必须为 true。

定向结果：`tests/test_ai2thor_postconditions.py` 7/7 通过。

## 8. 标准化流程检查

### 8.1 数据准备

本轮属于 inference-time 控制、执行和验证修复，不涉及训练或微调。没有新增训练集、checkpoint、训练步数、学习率或未经来源验证的超参数。

### 8.2 模块构建

模块边界保持清晰：

- action catalog：动作名和参数归一化。
- approach verifier：AI2-THOR 交互位姿证据。
- task semantics/verifier：完成谓词。
- controller：VLM 动作门禁。
- adapter：Unity 执行闭环。
- session memory：幂等执行事实和计划状态。

### 8.3 验证

执行顺序：

1. 每个模块定向测试。
2. AI2-THOR 相关跨模块测试。
3. task/model/memory 回归。
4. 完整测试套件。

最终本地结果：

```text
Ran 178 tests
OK (skipped=2)
```

两个跳过项均为显式环境开关控制的真实付费模型 API 测试，不是失败或静默跳过。

## 9. 未修改的超参数与结构

本轮没有修改：

- `stop_confidence_threshold`
- `target_visible_threshold`
- `max_steps`
- 模型 temperature
- 模型 max_tokens
- 模型名称或 provider URL
- gridSize
- rotateStepDegrees
- 训练/微调设置
- pipeline stage 顺序

新增的 `maxPoses=64` 不是新设参数，来自仓库中已经在真实 AI2-THOR Unity 环境通过的 `validate_ai2thor_sofa_approximation.py`。

## 10. 仍未完成的 Plan 2 门禁

本轮不能宣称 Plan 2 全部完成。以下仍是明确阻断项：

1. 非 oracle RGB-D occupancy/frontier/value map。
2. 对象、空间、任务、失败、技能和完整 episode 分层 memory。
3. VLM 交互链的有序对象谓词，例如 Open Fridge -> Pickup Egg -> Put Egg in Bowl。
4. agent 禁止 `forceAction/manualInteract` 和服务端 actor 权限边界。
5. 所有 planner 动作的已注册 postcondition；unchecked 不得计入成功。
6. production non-oracle 与 simulator oracle 的评估隔离。
7. inference-only 固定回归集、false-stop、SPL、API/fallback 和 memory A/B 指标。
8. 流式协议的唯一终态、错误传播、顺序验证、取消确认、run 隔离和 DOM XSS 修复。
9. 模型 API 的总体 deadline、retry、provider 能力路由、输入边界和鉴权。
10. 依赖锁定、敏感文件历史治理、远端目录和 stash 最终审计。

## 11. 远端执行步骤

必须按顺序执行：

1. 本地 `git diff --check`。
2. 敏感文件和大文件审计。
3. 本地提交并 push `main`。
4. 远端 `/home/scale/kangjay/kaohe` 保存 status、HEAD、stash 清单。
5. 远端只做 `git pull --ff-only origin main`，不自动 pop 旧 stash。
6. 远端运行完整单测。
7. 远端运行 `validate_ai2thor_sofa_approximation.py`。
8. 远端运行真实 Agent 流式任务“找到房间里的沙发并坐下”。
9. 检查 NDJSON 必须包含多步动作、approach evidence、Crouch postcondition 和同一步 `approximate_success`。
10. 只有全部通过后才重启网页服务并记录远端 HEAD、进程和访问 URL。

## 12. 本轮退出标准

- [x] 抽象导航动作真实传入 AI2-THOR 原生参数。
- [x] distance 不再等于 approach。
- [x] 真实交互位姿可产生 approach witness。
- [x] 未接近时 Crouch/Done/STOP 被阻止。
- [x] Crouch postcondition 缺字段时失败。
- [x] 动作后同一步更新任务完成状态和 execution plan。
- [x] 重复 execution commit 被拒绝。
- [x] 本地完整测试 178/178 通过。
- [ ] 敏感文件当前版本和历史治理完成。
- [ ] GitHub 与远端 HEAD 同步。
- [ ] 远端真实 Unity 沙发验证通过。
- [ ] 远端真实 Agent 流式沙发任务通过。
