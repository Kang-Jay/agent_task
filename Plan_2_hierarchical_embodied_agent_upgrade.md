# Plan 2：分层具身视觉搜索 Agent 升级与严格验收计划

更新时间：2026-07-10
状态：待执行
适用项目：`D:\cache\SummerCap\kaohe\zju`

## 0. 计划目标

本计划在 `Plan_1_agent_demo_repair.md` 已完成的工程基础上，解决当前系统仍然存在的核心问题：

1. 从“单帧图像输入后直接生成单步动作”升级为“总体任务计划、子目标执行、反馈验证、失败恢复和动态重规划”。
2. 从启发式旋转/前进升级为具有几何依据的零样本视觉搜索，包括占据地图、frontier、目标证据地图和可达性检查。
3. 从“看见目标即可停止”升级为独立的任务谓词验证；VLM 只能提出完成建议，不能自行裁定任务成功。
4. 从步骤日志和文本相似度检索升级为分层具身记忆，包括工作记忆、任务记忆、空间记忆、对象记忆、失败记忆、技能记忆和情景记忆。
5. 完整实现 PPT 要求的语言输入、多模态点选输入、机器人第一视角、中间过程、交互轨迹、长程记忆和视频演示。
6. 保持现有配置、动作目录、pipeline 和后置条件逻辑可追溯；任何模型、推理参数或结构设置均不得凭经验随意更改。

最终交付必须是可以在真实 AI2-THOR Unity 环境中演示的网页系统，而不是只播放预录轨迹或使用静态 fallback 的页面。

## 1. 事实来源与优先级

发生定义冲突时，按以下顺序裁决：

1. AI2-THOR 官方动作、metadata 和运行时实际返回结果。
2. PPT 中明确写出的功能要求。
3. `configs/agent_config.json`、schema、现有 pipeline 和经过测试的项目代码。
4. 官方论文、官方代码仓库、许可证和固定 commit。
5. 本计划提出但尚未验证的设计建议。

不得使用以下内容作为事实来源：

- 未经核验的博客或论坛二手总结。
- 第三方复刻仓库对原论文能力的声明。
- VLM 自己生成但没有环境证据的“任务已完成”文本。
- 只通过源码字符串扫描得到的“功能已实现”结论。
- 演示视频中的视觉效果代替真实执行日志和 simulator metadata。

当前根目录中的 `题目.txt` 实际包含 API 凭据，不是任务要求文档。它不得进入 prompt、日志、报告或 Git；项目要求以 PPT、配置和代码为准。

## 2. PPT 基本要求与当前差距

| PPT 要求 | 当前基础 | 主要差距 |
|---|---|---|
| 语言指令下达视觉搜索任务 | 已有 instruction 输入 | 复杂任务尚未生成总体子目标计划 |
| 机器人第一视角图像输入 | 已有 AI2-THOR RGB 画面 | 非 oracle 感知仍主要是启发式视觉 |
| 中间步骤与交互轨迹 | 已有 step/timeline | 缺少可审计的子目标进度、动作结果和重规划事件 |
| 交互轨迹视频 | 已有录制与编码 | 必须确认视频来自真实实时 Agent，而不是 fallback 或 replay 冒充 |
| 点选目标物体形成图像加语言输入 | 已有 clicked point/target crop 基础 | 需要与任务计划、目标绑定和后续多帧跟踪完全贯通 |
| 长程交互记忆 | 已有 SQLite episodic store | 当前主要按 instruction token Jaccard 检索，不是具身空间/状态记忆 |

## 3. 调研结论与代码复用边界

### 3.1 已下载并固定版本

详细来源记录见 `research/references/embodied_agent_codebase_manifest.md`。

| 仓库 | 固定提交 | 许可证 | 本项目用途 |
|---|---|---|---|
| VLFM | `584ed56008754fde7997d904983607def8328322` | MIT | obstacle map、frontier、value map 的轻量实现依据 |
| VLMaps | `58060f97239074338ab419a2090d43fa752d724d` | MIT | 开放词汇特征落图和空间语言查询思路 |
| ConceptGraphs | `93277a02bd89171f8121e84203121cf7af9ebb5d` | MIT | 对象节点、观测合并和关系结构 |
| AriGraph | `e884b76d7fa5185a3a8a55e5a67393b5a43f5ef2` | MIT | episodic 与 semantic graph 分层 |
| ProgPrompt | `56e65510747dff809c1b0bac9318508da9d9a2d4` | NVIDIA License | 仅参考程序化计划、断言和恢复结构 |
| L3MVN | `204250c26060f32e3fb4a3dbba196d2e97fcfc82` | 未发现根许可证 | 仅参考论文中的 object-room prior 和 frontier ranking |

`SG-Nav` 和 `ReMEmbR` 的官方仓库已经核验，但 GitHub 网络连续超时，当前标记为待下载。下载成功前不得声称其代码已经进入本地工具库。

### 3.2 推荐采用的论文逻辑

1. **LLM-Planner**
   - 采用总体任务分解、已完成计划反馈和动态重规划。
   - 不直接复制 ALFRED 环境封装。

2. **ProgPrompt**
   - 采用“前置条件、动作、断言、恢复分支”的结构。
   - 因许可证和 VirtualHome 耦合，只重写协议，不复制源码。

3. **SayCan**
   - 采用“语言相关性 × 当前技能可执行性”的动作门控。
   - 可执行性必须来自 AI2-THOR 对象状态和动作前置条件。

4. **VLFM**
   - 第一优先级空间搜索方案。
   - 采用 RGB-D、相机位姿、障碍地图、frontier 和目标价值图。
   - 不引入完整 Habitat policy、PointNav 模型或旧 CUDA 环境。

5. **VLMaps / ConceptGraphs**
   - 采用开放词汇特征聚合、对象证据表和对象关系结构。
   - 不引入完整 3D SLAM、Grounded-SAM、LLaVA、PyTorch3D 和 Habitat 数据管线。

6. **AriGraph / Voyager / Reflexion**
   - AriGraph：采用 episodic 与 semantic/object graph 分层。
   - Voyager：仅在 critic/verifier 确认成功后写入技能。
   - Reflexion：保存结构化失败摘要和恢复建议。
   - 旧记忆不能覆盖当前 simulator state。

7. **ReAct / Inner Monologue**
   - 用于定义可展示的执行轨迹：
     `Observation -> Decision Summary -> Action -> Result -> Replan`。
   - 不展示或声称展示模型隐藏思维链。

### 3.3 明确排除

- 不复制 L3MVN 源码、权重和日志。
- 不复制 ProgPrompt 源码到主项目。
- 不引入完整 ConceptGraphs、VLMaps 或 Habitat 运行栈。
- 不将 PaLM-E、RT-2、SIMA 的第三方复刻作为正式依赖。
- 不将 AI2-THOR 隐藏目标位置或全量 object metadata 输入 planner 后仍声称纯 RGB zero-shot。
- 不让 simulator oracle、rule fallback 或 replay 冒充真实 VLM 决策。

## 4. 目标总体架构

```text
语言指令 + RGB/Depth + 可选点选目标
                  |
                  v
        Multimodal Task Interpreter
                  |
                  v
        VLM Task Planner（总体计划）
                  |
                  v
        Task/Progress Memory（子目标状态）
                  |
                  v
 Spatial/Object/Episodic Memory Retrieval
                  |
                  v
       Action Planner / Skill Selector
                  |
                  v
       Action Schema + Affordance Gate
                  |
                  v
           AI2-THOR Executor
                  |
                  v
        Action Postcondition Verifier
                  |
                  v
          Task Predicate Verifier
          /          |           \
    下一子目标     局部恢复       全局重规划
                  |
                  v
        Verified Success / Failure
```

### 4.1 VLM 职责

VLM 负责：

- 理解语言和点选图像目标。
- 生成总体目标摘要和有序子目标。
- 在当前子目标内提出下一动作。
- 根据动作结果解释失败并提出恢复或重规划。
- 提出 `completion_proposal`。

VLM 不负责：

- 直接确认动作已成功。
- 覆盖 AI2-THOR metadata。
- 仅凭“看到物体”结束复合任务。
- 输出任意不在动作目录中的动作。
- 自行修改配置阈值或任务结构。

### 4.2 Simulator 与 Verifier 职责

Simulator/Verifier 负责：

- 动作是否被 Unity 接受。
- 对象是否存在、可见、可交互和可达。
- `isOpen`、`isPickedUp`、inventory、receptacle 关系。
- Agent pose、旋转、horizon、`isStanding`。
- 子目标和总任务谓词是否成立。
- 精确成功、近似成功、失败和等待澄清的最终分类。

### 4.3 任务计划结构

任务计划必须是结构化数据，不只是一段自然语言：

```json
{
  "goal_summary": "找到沙发并执行可验证的坐下近似动作",
  "completion_mode": "approximate_sit",
  "limitations": ["native_sit_on_furniture_state_unavailable"],
  "subgoals": [
    {
      "id": "locate_sofa",
      "description": "定位并持续跟踪沙发",
      "allowed_action_families": ["navigation", "camera"],
      "success_predicates": ["target_observed", "target_track_stable"],
      "failure_predicates": ["search_exhausted"],
      "recovery": ["select_new_frontier"]
    },
    {
      "id": "approach_sofa",
      "description": "移动到满足任务定义的目标邻域",
      "success_predicates": ["target_bound", "target_reachable", "distance_verified"]
    },
    {
      "id": "crouch_near_sofa",
      "description": "在沙发附近执行 Crouch",
      "success_predicates": ["crouch_action_succeeded", "agent_is_not_standing"]
    }
  ]
}
```

具体距离、重试次数、地图分辨率等不得在代码中临时硬编码。未在当前配置或官方实现中确定的参数必须先进入配置审查和实验记录。

## 5. 成功、终止和澄清语义

必须拆分以下概念：

- `terminated`：本 episode 是否停止执行。
- `success`：任务是否通过全部 verifier。
- `needs_clarification`：是否等待用户补充信息。
- `failure_reason`：能力不支持、超步数、目标不存在、动作失败等。
- `completion_mode`：`exact` 或明确标记的近似模式。

`done=True` 不再等于成功。

严格终止协议：

```text
VLM proposes Done/STOP
-> Task Verifier 检查全部 required predicates
-> 全部成立：VERIFIED_SUCCESS
-> 部分成立：IN_PROGRESS，选择未完成子目标
-> 证据冲突：UNCERTAIN，补充观察
-> 动作失败：RECOVERABLE_FAILURE 或 FAILED
-> 能力不支持：UNSUPPORTED，不得返回成功
```

### 5.1 “找到沙发并坐下”

AI2-THOR iTHOR 没有原生 `SitOnObject` 或 `isSittingOnObjectId`。

不修改 Unity 时，只允许以下近似：

```text
LocateSofa
-> BindTargetSofaId
-> NavigateNearSofa
-> Face/InspectSofa
-> Crouch
-> Verify same target + proximity + Crouch success + isStanding=false
-> APPROXIMATE_SUCCESS
```

必须返回 `completion_mode=approximate_sit`，不得显示“真实坐在沙发上”。

如果要实现精确坐下，需要单独扩展 Unity：

- 座位锚点。
- 可坐对象属性。
- `SitOnObject` 动作。
- 坐姿动画和碰撞约束。
- `isSitting` 和 `sittingOnObjectId` metadata。
- 对应 postcondition 和任务谓词。

### 5.2 Open -> Pickup -> Put

最终成功按世界状态判断，不按历史中出现过动作名称判断：

- `OpenObject`：指定容器对象 `isOpen=true`。
- `PickupObject`：指定目标进入 inventory 且 `isPickedUp=true`。
- `PutObject`：指定目标离开 inventory，并进入指定 receptacle 的双向关系。
- 目标放入后又被拿出，最终任务必须判失败。

## 6. 配置与超参治理

当前 `configs/agent_config.json` 中已有值全部冻结，除非通过独立实验和 ChangeRecord 批准：

- `max_steps=20`
- `history_window=6`
- `repeated_action_penalty=0.12`
- `stop_confidence_threshold=0.78`
- `target_visible_threshold=0.58`
- `exploration_confidence_floor=0.18`
- `default_turn_angle_degrees=30`
- `image_size=[448,448]`
- `grid_rows=3`
- `grid_cols=3`
- `candidate_patch_size=96`
- `long_term_capacity=200`
- `negative_memory_capacity=80`
- `retrieval_top_k=3`
- `min_success_iou=0.3`

治理规则：

1. 不为适配论文代码而直接覆盖这些值。
2. 新增参数必须写入正式 config/schema，不得散落硬编码。
3. 每个新增参数必须记录来源：现有配置、AI2-THOR 官方属性、论文官方配置或本项目消融实验。
4. 只有通过固定评估任务集和相同 seed 的对照实验，才能修改阈值。
5. smoke test 可以缩小运行规模，但其结果不能作为性能结论。
6. 本项目采用 inference-only 路线，不设置模型训练或微调步骤；如未来出现新的训练需求，必须另立计划并重新审查数据、模型许可证和全部超参。
7. 所有模型、prompt、配置和动作目录必须记录版本或哈希。

## 7. 实施阶段与依赖顺序

所有阶段必须顺序执行。前一阶段未满足退出条件时，不得进入后一阶段。

### 阶段 0：工作区、秘密和基线冻结

#### 目标

确认当前其他作者的修改、Git 状态、测试基线、远端部署版本和秘密文件边界。

#### 操作

1. 检查：

```powershell
git status --short --branch
git diff --check
git rev-parse HEAD
git rev-parse origin/main
```

2. 将修改分类为：
   - 本任务修改。
   - 其他作者正在修改。
   - 未跟踪临时报告。
   - 运行生成物。
   - 第三方研究源码。

3. 不覆盖其他作者对以下文件的并行修改：
   - `src/agent/model_adapter.py`
   - `src/agent/task_semantics.py`
   - 其他 `git status` 中已修改文件。

4. 确认秘密和大文件被忽略：
   - `apikey.txt`
   - `题目.txt`
   - `.env*`
   - 视频、frames、日志、PID、数据库运行副本。
   - `research/codebases/*/source/`

5. 清理策略：
   - 不立即删除未跟踪报告。
   - 先核对其中是否有唯一信息。
   - 唯一信息合并进 Plan、ChangeRecord 或 README 后再删除重复文件。

#### 测试

```powershell
python -B -m compileall -q src tests tools
python -B -m unittest discover -s tests -v
```

#### 退出条件

- 测试基线有明确记录。
- 不存在被意外跟踪的密钥和第三方源码。
- 其他作者的修改已识别，后续改动不会覆盖它们。

### 阶段 1：Schema 和状态语义重构

#### 目标

建立任务计划、子目标、终止状态、证据和执行结果的统一契约。

#### 修改范围

- `src/types/schema.py`
- `src/agent/task_semantics.py`
- `src/memory/session_memory.py`
- API 序列化和 UI 消费接口。

#### 必须新增或明确的结构

1. `TaskPlan`
   - `goal_summary`
   - `task_types`
   - `completion_mode`
   - `limitations`
   - `subgoals`
   - `planner_source`
   - `plan_version`

2. `Subgoal`
   - `id`
   - `description`
   - `status`
   - `target_binding`
   - `allowed_action_families`
   - `preconditions`
   - `success_predicates`
   - `failure_predicates`
   - `recovery_options`
   - `evidence`

3. `CompletionStatus`
   - `terminated`
   - `success`
   - `needs_clarification`
   - `failure_reason`
   - `completion_mode`
   - `verified_predicates`
   - `missing_predicates`
   - `evidence_refs`

4. `ActionExecution`
   - proposed action
   - validated action
   - bound object ID
   - simulator result
   - postcondition result
   - commit status

#### 规则

- `done` 仅作为兼容字段，不能继续承担成功语义。
- `ASK_CLARIFY` 表示等待输入，不得记录为成功。
- 未注册 postcondition 的动作不得默认 `passed=true`。
- `skill_call`、实际 action 和执行日志必须一致。

#### 测试

- 更新 `tests/test_schema.py`。
- 更新 `tests/test_task_semantics.py`。
- 新增以下失败优先测试：
  - `done=True` 但 `success=False`。
  - `ASK_CLARIFY` 不算成功。
  - 非法 action 被拒绝。
  - 缺少 predicate 时拒绝 STOP。
  - `skill_call` 与 action 不一致时失败。
  - 未注册 postcondition 不得成为成功证据。

```powershell
python -B -m unittest discover -s tests -p "test_schema.py" -v
python -B -m unittest discover -s tests -p "test_task_semantics.py" -v
python -B -m unittest discover -s tests -p "test_ai2thor_postconditions.py" -v
```

#### 退出条件

- API、trace、memory、UI 使用同一套状态语义。
- 测试能够区分成功、失败、等待澄清和继续执行。

### 阶段 2：VLM 总体任务规划

#### 目标

让 VLM 在 episode 开始时生成总体计划，而不是每帧只输出一个动作。

#### 修改范围

- `src/agent/model_adapter.py`
- `src/agent/controller.py`
- `src/memory/session_memory.py`
- `tests/test_model_planner.py`

#### 实现步骤

1. 在 `ModelAdapter` 增加 `plan_task(payload)`。
2. 输入只包含：
   - 用户 instruction。
   - 初始 RGB 或目标 crop。
   - 允许的技能族。
   - AI2-THOR 能力限制。
   - 当前可公开给 planner 的观察。
3. 输出严格 JSON：
   - goal summary。
   - ordered subgoals。
   - predicate templates。
   - completion proposal policy。
   - limitations。
4. 首次成功生成后，将计划持久化到 session。
5. 后续步骤只更新进度，不重复生成全部计划。
6. 只有以下情况允许 replan：
   - 当前计划不可执行。
   - 目标绑定失效。
   - 连续动作失败。
   - 新观察推翻计划假设。
   - verifier 发现谓词冲突。
7. VLM 调用失败时：
   - 明确记录 fallback 原因。
   - 严格演示模式不得静默切换规则规划。
   - 非严格模式可使用确定性模板计划，但必须标注来源。

#### Prompt 约束

- 不请求隐藏思维链。
- 只请求可审计的计划摘要、当前依据、缺失条件和下一动作。
- 明确说明 `STOP` 是建议，最终由 verifier 裁决。
- 明确说明 AI2-THOR 没有原生 SitOnObject。

#### Token 控制

当前生产主链路曾因完整动作 schema 和大量对象 metadata 造成 prompt 过长。必须：

- 只发送当前子目标相关动作。
- 只发送动作必需参数和安全参数。
- 只发送可见或最近相关对象。
- 不发送整个动作目录和全部 40+ 对象。
- 记录输入 token、输出 token 和截断情况。

#### 测试

- `plan_task()` JSON 校验。
- 视觉输入真正发送。
- 无图像结果在严格视觉模式被拒绝。
- 超长环境被压缩但关键对象不丢失。
- VLM 过早输出 STOP 时继续执行未完成子目标。
- 模型异常时来源字段正确。

```powershell
python -B -m unittest discover -s tests -p "test_model_planner.py" -v
python -B -m unittest discover -s tests -p "test_live_model_integration.py" -v
```

真实 API 测试只使用现有 provider/model 配置，不新增或猜测模型名称、温度和 token 参数。

#### 退出条件

- session 中存在持久化总体计划。
- 每步能指出当前子目标。
- 主链路可证明使用了视觉输入和 model planner。

### 阶段 3：动作白名单、参数绑定与可执行性门控

#### 目标

完整支持项目声明的 AI2-THOR 动作，同时阻止缺参数、错误对象和危险参数。

#### 修改范围

- `configs/ai2thor_actions_v5.json`
- `src/simulation/ai2thor_action_catalog.py`
- `src/simulation/ai2thor_interactions.py`
- `src/agent/controller.py`
- `tests/test_ai2thor_actions.py`
- `tests/test_ai2thor_interactions.py`

#### 实现步骤

1. 动作定义只从 AI2-THOR 5.0.0 动作目录生成。
2. 为每个动作建立：
   - required params。
   - optional params。
   - 参数类型。
   - agent mode。
   - 对象 affordance。
   - 是否允许 VLM 自动调用。
3. 将抽象技能映射到真实动作：
   - navigation。
   - camera。
   - object interaction。
   - posture。
   - arm/manipulation。
4. 对 `objectId`：
   - 只能绑定当前环境真实存在的对象。
   - 必须满足 affordance。
   - 必须满足可见、可交互或动作文档要求。
5. 禁止 VLM 自由设置 `forceAction=true`。
6. 动作执行前运行 precondition gate，执行后运行 postcondition verifier。
7. UI 手动动作和 Agent 动作使用同一验证路径。

#### 测试

- 每个自动动作都有 schema。
- 缺必需参数被拒绝。
- 参数类型错误被拒绝。
- 虚构 object ID 被拒绝。
- affordance 不符被拒绝。
- `forceAction` 被拒绝。
- action/skill_call 冲突被拒绝。

#### 退出条件

- 所有 Agent 可调用动作均经过 schema、对象绑定和前置条件检查。
- “支持完整动作空间”不再仅表示配置中列出动作名称。

### 阶段 4：独立任务谓词与 Evidence Ledger

#### 目标

建立与 VLM 解耦的任务完成判定。

#### 新增模块建议

- `src/simulation/task_predicates.py`
- `src/simulation/task_verifier.py`

新增文件必须职责单一，不再创建额外嵌套目录。

#### Evidence Ledger

每个谓词记录：

- predicate ID。
- 当前值。
- 证据来源。
- 观察或执行时间。
- 对象 ID。
- 相关 pose。
- 是否仍有效。
- 是否被新证据推翻。

#### 谓词类型

- `observed(target)`
- `track_stable(target)`
- `reachable(target_region)`
- `near(agent, target)`
- `facing(agent, target)`
- `open(container)`
- `holding(object)`
- `inside(object, receptacle)`
- `posture(agent, crouched)`
- `action_succeeded(action)`

#### 规则

- VLM 证据只能是候选证据。
- 当前传感器和 simulator metadata 优先。
- 旧证据必须有有效期或重新验证。
- 最终 success 必须满足所有 required predicates。
- 缺少证据与证据为 false 必须区分。

#### 测试

- 目标看见但未接近：不成功。
- 动作历史存在但最终状态被撤销：不成功。
- metadata 与旧 memory 冲突：当前 metadata 胜出。
- VLM 提议 Done 但缺 predicate：继续执行。
- 所有 predicate 成立：成功。

#### 退出条件

- 任意成功结果均能列出完整证据。
- 错误 STOP 测试为零容忍。

### 阶段 5：“找到沙发并坐下”闭环

#### 目标

修复当前只执行一步、看到沙发后 Inspect/Stop 的错误。

#### 实现步骤

1. 生成并保存总体计划：
   - locate sofa。
   - bind sofa object。
   - approach sofa。
   - align/inspect sofa。
   - crouch。
   - verify posture and proximity。
2. 同一个任务必须持续绑定同一个 Sofa object ID。
3. “接近”不能等同于 metadata `visible=true`。
4. 接近条件必须由正式配置或 AI2-THOR 可交互/可达定义产生，不得临时填写距离数字。
5. `Crouch` 成功后，在下一状态验证 `agent.isStanding=false`。
6. 返回 `approximate_sit`。
7. UI 显示限制说明。

#### 测试

- 看见沙发后不会 STOP。
- 沙发可见但距离不满足时继续移动。
- 远离沙发 Crouch 不算完成。
- Crouch 失败不算完成。
- `isStanding=true` 不算完成。
- 目标 ID 发生变化时重新绑定或重规划。
- 成功结果明确标记近似。

#### 真实 Unity 验证

在远端执行完整 episode，保存：

- 每步 RGB。
- 当前子目标。
- proposed/validated/executed action。
- Sofa object ID。
- Agent pose。
- 目标距离证据。
- Crouch 返回值。
- `isStanding`。
- 最终 verifier。

#### 退出条件

- 不再出现一步后错误终止。
- 完成证据可重放。
- 系统不宣称真实 SitOnObject。

### 阶段 6：VLFM-lite 几何搜索

#### 目标

替换固定左右转 fallback，建立真正的 frontier exploration。

#### 实现原则

- 使用 AI2-THOR 精确相机位姿，不引入 SLAM。
- 启用并读取 depth。
- 构建轻量：
  - occupancy map。
  - visited map。
  - collision/blocked map。
  - frontier set。
  - target-evidence map。
  - agent trajectory。
- 参考 VLFM 数学逻辑，重写 AI2-THOR 投影和坐标接口。

#### 禁止

- 不复制 Habitat policy。
- 不安装完整 VLFM 环境覆盖当前环境。
- 不直接沿用 VLFM 未核验的阈值。
- 不使用隐藏目标 metadata 更新 target map。

#### 测试

- pose -> grid -> pose 回投一致。
- 障碍与自由空间更新正确。
- frontier 位于已知自由空间和未知区域边界。
- 不可达 frontier 被过滤。
- 碰撞边不会重复选择。
- 隐藏目标移动不会在首次观察前影响地图。
- 固定轨迹 replay 产生确定性地图。

#### 退出条件

- Agent 在无目标证据时依据 frontier，而不是固定动作模板探索。
- 地图只使用允许给 planner 的观察。

### 阶段 7：开放词汇感知和对象证据表

#### 目标

建立非 oracle 的目标检测、跨帧跟踪和对象记忆。

#### 设计

对象证据表至少包含：

- internal track ID。
- 候选类别分布。
- 目标匹配分数。
- 图像区域或 mask。
- 估计世界位置。
- 首次/最后观察时间。
- 观察次数。
- 正证据和负证据。
- 与 simulator object ID 的 verifier-only 对齐。

#### 分层模式

1. `heuristic_test_stub`
   - 只用于单元测试。
2. `open_vocabulary`
   - VLM 或开放词汇检测器用于真实 Agent。
3. `simulator_oracle`
   - 只用于上界评估和 verifier，不进入 zero-shot planner。

#### 模型选型规则

- 不在本阶段随意决定 GroundingDINO、CLIP、SAM 或其他模型超参。
- 先记录候选模型的许可证、权重来源、显存、延迟和输入尺寸。
- 使用真实场景标注集进行比较后再冻结模型。
- 权重不得进入 Git。

#### 测试

- 语言目标和点选 crop 均能产生目标查询。
- 多帧相同目标保持 track。
- 误检不会立即触发 STOP。
- oracle 与非 oracle 结果分开记录。
- 目标不存在时不得成功。

#### 退出条件

- “零样本视觉搜索”不再依赖颜色启发式或隐藏 metadata。
- 感知结果具有可审计证据。

### 阶段 8：分层具身 Memory

#### 目标

替换“每动作写入 + instruction Jaccard + FIFO”的单一记忆策略。

#### 分层

1. `L0 LiveState`
   - 当前传感器、Agent pose、inventory、动作结果。
   - 最高优先级，不属于长期记忆。

2. `L1 WorkingMemory`
   - 当前 episode 的近期步骤、当前子目标、局部地图状态。

3. `L2 TaskMemory`
   - 总体计划、子目标状态、对象绑定和 Evidence Ledger。

4. `L3 SpatialObjectMemory`
   - 对象轨迹、位置、关系、last seen、有效期和冲突版本。

5. `L4 EpisodicFailureMemory`
   - 关键成功、确认失败、恢复过程和人工纠正。

6. `L5 SkillMemory`
   - 技能前置条件、后置条件、适用对象和验证结果。

#### 写入门控

以下情况才写长期记忆：

- 子目标完成。
- 动作确认失败。
- 首次发现重要对象。
- 关键状态变化。
- 恢复成功。
- 人工纠正。
- 最终任务成功或失败。

普通无信息步骤只保留在 working memory。

#### 冲突优先级

```text
安全约束
> 当前传感器与 simulator state
> 当前已验证世界模型
> 当前 episode
> 长期记忆
> 通用先验
```

#### 必须新增的字段

- `observed_at`
- `last_verified`
- `valid_from`
- `valid_to`
- `state_fingerprint`
- `source_refs`
- `supersedes`
- `contradicted_by`
- `failure_class`
- `recovery_action`

#### 检索

先进行结构化过滤：

- environment/scene。
- task type。
- target type。
- object/state。
- failure class。
- 时间有效性。

然后再做语义、空间和状态匹配；最终使用 LiveState 重排和验证。

#### 测试

- 移动物体后旧位置失效。
- 门状态变化后旧经验不覆盖当前状态。
- 失败条件消失后不再错误抑制动作。
- 同指令不同场景不串记忆。
- 并发 session 隔离。
- 成功技能只有 verifier 通过后写入。
- memory ablation 可关闭并复现实验。

#### 退出条件

- 长期记忆不会直接覆盖实时状态。
- 检索结果能够解释来源、有效期和适用条件。

### 阶段 9：网页流式任务执行与可审计展示

#### 目标

让用户看到实时任务计划、当前进度、动作执行和验证，而不是等待结束后只显示一条结果。

#### 必备流式事件

- `request_validated`
- `simulator_starting`
- `simulator_ready`
- `task_plan_started`
- `task_plan_generated`
- `memory_retrieved`
- `subgoal_started`
- `observation_received`
- `perception_updated`
- `action_proposed`
- `action_validated`
- `action_executed`
- `postcondition_checked`
- `subgoal_completed`
- `replan_started`
- `replan_completed`
- `completion_proposed`
- `completion_verified`
- `episode_completed`
- `episode_failed`
- `error`

#### UI 必备区域

1. 任务输入：
   - 语言 instruction。
   - 点选目标图像。
   - scene 和 backend。

2. 机器人视角：
   - 当前 RGB。
   - 可选目标框/mask。
   - 当前动作和执行结果。

3. 地图：
   - Unity 第三方相机全局视图。
   - occupancy/frontier/trajectory overlay。
   - oracle 信息与 Agent 信息采用不同图层和颜色。

4. 任务计划：
   - 总体 goal。
   - 子目标列表。
   - 当前子目标。
   - 已完成/缺失谓词。

5. 决策摘要：
   - 当前观察证据。
   - 候选技能。
   - 可执行性检查。
   - 选择动作。
   - 动作后验证。

6. Memory：
   - 当前工作记忆。
   - 检索到的情景经验。
   - 对象 last seen。
   - 失败恢复建议。

7. 运行状态：
   - model planner / fallback / oracle 来源。
   - token、延迟和错误。
   - exact / approximate / failed。

#### 安全要求

- 所有文本必须安全转义，禁止将模型文本直接作为 HTML。
- 后端异常通过明确 `error` 事件发送，不得吞掉后正常 EOF。
- fallback 和 replay 必须有明显标签。
- 页面不能声称展示隐藏思维链，只展示结构化决策摘要。

#### 测试

- stream 事件顺序测试。
- worker 异常传播测试。
- 页面 DOM XSS 测试。
- 断线重连测试。
- 模型 API 超时测试。
- Unity 初始化超时测试。
- 播放、暂停、回放和实时模式区分测试。

#### 退出条件

- 页面可以实时看到总体计划、当前子目标和 verifier 进度。
- 任意错误都能在 UI 明确呈现。

### 阶段 10：标准化评估任务集与回归基准

#### 目标

建立可复现的评估与回归基础，而不是训练模型，也不是只保存少量成功演示。

#### 数据类型

1. 原始 episode：
   - instruction。
   - scene。
   - seed。
   - 初始 pose。
   - 目标描述或目标 crop。
   - 可用动作 profile。

2. 每步 observation：
   - RGB。
   - depth。
   - camera pose。
   - Agent pose。
   - 允许保存的感知标注。
   - simulator metadata 证据副本。

3. 任务计划：
   - 总体计划。
   - replan 历史。
   - 子目标状态。

4. 执行轨迹：
   - proposed action。
   - validated action。
   - executed action。
   - simulator result。
   - postcondition。

5. 完成标注：
   - required predicates。
   - 最终 predicate values。
   - success/failed/unsupported/approximate。

6. Memory 标注：
   - 写入候选。
   - 是否通过写入门控。
   - 检索 query。
   - 命中记录。
   - 是否有帮助或造成负迁移。

#### Split 原则

- 按 scene 划分，防止相同房间泄漏。
- 对象类别、任务类型和动作族在各 split 中有明确覆盖报告。
- 成功、失败、目标不存在、不可达和能力不支持均需覆盖。
- 语言模板不能在 train/test 中只做表面改写。
- 点选图像任务和纯语言任务分别报告。

不在本计划中凭空指定 episode 数量。评估规模由覆盖矩阵、场景数、任务族和统计稳定性决定，并写入 evaluation dataset card。该任务集只用于回归、验证和最终评估，不用于训练或微调模型。

#### 数据质量检查

- 文件存在和可解码。
- 时间顺序正确。
- action 与 observation 对齐。
- object ID 引用有效。
- split 无 scene 泄漏。
- success 与 verifier 一致。
- 近似任务标签正确。
- oracle 数据没有混入非 oracle 输入。
- 密钥和个人路径不在数据中。

#### 输出

- dataset schema。
- dataset card。
- split manifest。
- coverage report。
- validation report。
- 数据版本和校验值。

#### 测试

- schema validation。
- split leakage test。
- trajectory replay test。
- verifier label consistency test。
- missing file test。
- corrupt image test。

#### 退出条件

- 任一评估结果都能追溯到固定数据版本。
- 数据覆盖不依赖少量成功样本。

### 阶段 11：Inference-only 配置冻结与真实模型验收

#### 原则

本项目正式路线固定为 inference-only：使用现有多模态 API 负责总体规划和逐步决策，使用确定性 executor、AI2-THOR metadata 和 verifier 负责执行与验收。本阶段不训练、不微调、不生成 checkpoint。

#### 必须冻结的内容

- provider、base URL、API 协议和模型 ID。
- system prompt、任务规划 prompt 和逐步决策 prompt。
- 输入消息结构和图片编码方式。
- 图像输入尺寸。
- 最大序列长度。
- JSON Schema、允许动作目录和参数格式。
- timeout、retry、并发和速率限制策略。
- fallback 策略及其显式状态标记。
- 当前正式配置中的阈值和 memory 检索参数。

不得从论文或其他项目复制训练超参，也不得为了展示效果临时修改正式推理阈值。

#### 真实模型验收

1. 使用 `apikey.txt` 对应的模型接口执行受控多模态请求，但绝不记录或提交密钥。
2. 用相同语言任务分别测试正确图片、遮挡图片和无关图片。
3. 验证模型输出会随视觉输入发生合理变化。
4. 验证总体计划、当前子目标、动作和完成提议均通过 Schema 校验。
5. 验证 API 超时、限流、空输出和格式错误均显式终止或进入标记清楚的 fallback。
6. 验证 `planner_source`、模型 ID、请求 ID 和 `vision_input_used` 被写入审计记录。
7. 使用阶段 10 的固定评估任务集执行回归，不把评估样本用于任何训练。

#### 测试

- 多模态请求序列化测试。
- 图片实际进入请求的审计测试。
- 结构化输出解析和拒绝非法动作测试。
- 视觉因果测试。
- timeout、retry 和限流测试。
- fallback 不冒充真实模型的测试。
- 固定配置重复运行测试。

#### 退出条件

- inference-only 配置完整、可追溯并已冻结。
- 真实模型确实接收图像和语言。
- 模型输出可以驱动总体计划和逐步决策。
- 不存在训练代码、checkpoint 或未经批准的训练参数进入正式 pipeline。

### 阶段 12：标准评估与消融

#### 必须报告的任务指标

- Task Success。
- Exact Success。
- Approximate Success。
- False Success Count。
- Success weighted by Path Length。
- Navigation Error。
- Episode Steps。
- Collision Count/Rate。
- Illegal Action Count。
- Action Execution Success。
- Postcondition Pass Rate。
- Subgoal Completion Rate。
- Replan Count。
- API Failure/Fallback Rate。

#### STOP 与完成判定指标

- completion precision。
- completion recall。
- false-stop rate。
- missed-stop rate。
- success evidence completeness。

完成判定 precision 必须作为高优先级门禁；任何错误成功都需要单独分析，不能被平均指标掩盖。

#### 感知指标

- target detection precision/recall。
- IoU。
- track consistency。
- last-seen accuracy。
- oracle 与非 oracle 差距。

已有配置中的 `min_success_iou=0.3` 保持不变，除非正式实验批准修改。

#### 地图指标

- pose/grid 回投误差。
- free/occupied cell accuracy。
- reachable frontier ratio。
- repeated frontier rate。
- hidden-target leakage test。

#### Memory 指标

- Recall@K。
- nDCG@K。
- MRR。
- context precision。
- irrelevant-memory injection rate。
- stale-memory-induced action count。
- repeated failure rate。
- recovery success rate。
- skill reuse success rate。
- negative transfer rate。
- memory query latency 和 token 开销。

#### 必须做的消融

- 无总体计划 vs 有总体计划。
- 无 frontier vs VLFM-lite frontier。
- 无 memory vs 分层 memory。
- 无 failure memory vs 有 failure memory。
- oracle 感知 vs 非 oracle 感知。
- VLM completion proposal vs 仅 verifier。
- 语言输入 vs 点选多模态输入。

#### 评估规则

- 使用固定数据版本、配置、模型和 seed。
- 失败 episode 不得删除。
- API 错误不得自动重新标成规则成功。
- exact 与 approximate 分开报告。
- replay 和真实 Unity 分开报告。
- 不在 test split 上调整阈值。

#### 退出条件

- 所有核心功能均有量化指标和失败样本分析。
- 论文逻辑带来的收益通过消融证明，而不是只靠界面效果。

### 阶段 13：远端部署、真实演示与视频检查

#### 同步原则

1. 本地只提交经过审核的文件。
2. GitHub `main` 作为唯一代码来源。
3. 远端 `/home/scale/kangjay/kaohe` 使用 `git pull --ff-only`。
4. 密钥、环境、模型权重、视频和 runtime 保持在 Git 外。
5. 本地、GitHub、远端 Git SHA 必须一致。

#### 部署前门禁

```powershell
git diff --check
python -B -m compileall -q src tests tools
python -B -m unittest discover -s tests -v
git status --short
```

#### 远端门禁

- Python 与 AI2-THOR 版本记录。
- AI2-THOR 5.0.0 动作目录匹配。
- 全量测试通过。
- 真实模型视觉调用通过。
- Unity 启动和销毁无残留。
- 第三方相机地图可用。
- 无 `map_camera_fallback`。

#### 必须录制的 demo

1. 纯语言视觉搜索。
2. 点选目标多模态搜索。
3. “找到沙发并坐下”的近似执行。
4. Open -> Pickup -> Put 交互链。
5. Memory 命中并帮助重复任务。
6. 失败或能力不支持时正确结束。

#### 视频检查

每个视频必须逐项检查：

- 画面来自真实 Unity。
- 动作标签与观察时序一致。
- TURN_LEFT/TURN_RIGHT 方向正确。
- 当前子目标实时更新。
- 没有长时间空白或卡死。
- 最终状态与 verifier 一致。
- 没有把 approximate 写成 exact。
- 视频可被浏览器解码。
- 没有泄露密钥、内部日志或个人路径。

#### 退出条件

- 用户可通过本地转发 URL 运行完整 demo。
- 录制视频、事件流和最终验证 JSON 相互一致。

---

## 8. 每个模块必须通过的测试门禁

任何阶段都不得以“代码已经写完”作为完成标准。每个模块必须依次通过以下门禁，前一层失败时不得进入后一层。

### 8.1 通用测试顺序

1. **静态检查**
   - 检查配置字段、类型、导入、路径和许可证边界。
   - 执行 `git diff --check`，不得存在空白错误或冲突标记。
   - 执行 `python -B -m compileall -q src tests tools`。
2. **模块单元测试**
   - 只验证当前模块的输入、输出、异常、边界和状态转换。
   - 必须包含正常样本、失败样本、非法参数和空结果。
3. **相邻模块集成测试**
   - 验证 schema、planner、executor、simulator、verifier、memory 和 UI 之间的真实接口。
   - 禁止用与生产结构不一致的 mock 掩盖接口错误。
4. **全量本地回归**
   - 执行 `python -B -m unittest discover -s tests -v`。
   - 现有测试不得因新功能失效。
5. **真实模型测试**
   - 使用受控任务真实调用多模态 API。
   - 结果中必须记录模型、provider、请求 ID、视觉输入是否实际发送、结构化输出及解析状态。
   - 禁止在模型失败后静默切换到规则输出并仍标记为模型成功。
6. **真实 Unity 测试**
   - 仅在远端 AI2-THOR 环境执行。
   - 检查动作返回、metadata、对象状态、agent pose、第三方相机和后置条件。
7. **浏览器端到端测试**
   - 从网页提交任务，持续接收事件，完成动作执行并显示最终验证结果。
   - 浏览器显示内容必须与服务端事件日志及 episode JSON 一致。
8. **人工视频检查**
   - 逐帧抽查动作前后画面、方向、目标位置、最终状态和文字标注。

### 8.2 VLM 真实性门禁

真实模型运行必须同时满足：

- `planner_source` 明确标记为真实模型规划器。
- `vision_input_used=true`，并可从请求审计信息证明图片已经进入模型请求。
- 模型输出通过严格 JSON Schema 校验。
- 输出动作来自当前允许动作目录。
- 参数来自场景中可绑定对象或合法坐标。
- 解析失败、超时、限流和拒绝必须显式暴露。
- fallback 结果不得伪装成 VLM 结果。
- 展示给用户的是结构化决策摘要，不记录或伪造隐藏思维链。

### 8.3 Simulator 与 Verifier 门禁

- 所有成功状态必须由环境 metadata 或明确的近似谓词证明。
- API `lastActionSuccess=true` 只表示动作调用成功，不等于总体任务成功。
- `Done` 只表示 VLM 提议结束，最终状态由 verifier 决定。
- 对象交互必须绑定具体 `objectId`，不得只依赖类别字符串。
- 关键动作必须验证前置条件、执行结果和后置条件。
- 任务失败、能力不支持和时间上限终止必须是不同状态。

### 8.4 阶段晋级规则

只有同时具备以下证据，阶段才能标为完成：

- 实现代码。
- 针对性测试。
- 全量回归结果。
- 配置审计结果。
- 失败样例及修复记录。
- 产物位置。
- 对应 ChangeRecord。

---

## 9. 计划修改的文件与最小新增边界

优先在既有模块内修复，不为了“架构完整”创建空目录或无使用方的抽象层。以下是预计改动范围，实际修改前必须再次读取当前文件和其他协作者的未提交变更。

### 9.1 既有文件

- `configs/agent_config.json`
  - 只扩展经过代码和测试支持的动作、模型、记忆和验证字段。
  - 保留已有正式超参，未经实验审查不得修改数值。
- `src/types/schema.py`
  - 增加任务计划、子目标、参数化动作、证据、任务状态和验证结果 schema。
- `src/agent/model_adapter.py`
  - 增加多模态任务规划、逐步决策、重新规划和完成提议接口。
- `src/agent/controller.py`
  - 改造成分层执行循环，不再把单次目标检测直接当成任务完成。
- `src/agent/task_semantics.py`
  - 解析任务类型、目标对象、关系、交互动作和完成谓词。
- `src/memory/session_memory.py`
  - 保留当前会话轨迹职责，避免继续堆叠所有记忆类型。
- `src/memory/episodic_store.py`
  - 增加任务级 episode、失败原因、恢复方式和可复用经验。
- `src/simulation/ai2thor_adapter.py`
  - 支持完整动作参数传递、第三方相机、metadata 和严格错误传播。
- `src/simulation/ai2thor_interactions.py`
  - 建立动作能力目录、前置条件、参数绑定和 AI2-THOR 调用。
- `src/simulation/ai2thor_postconditions.py`
  - 建立动作后置条件检查，不把动作 API 成功等同于任务完成。
- `src/ui/app.py`
  - 增加流式事件接口、任务状态查询、episode 审计和视频产物接口。
- `src/ui/static/index.html`
  - 显示总体计划、当前子目标、进度、动作结果、地图、记忆命中和最终验证。

### 9.2 只有确有职责时才新增

- `src/simulation/task_predicates.py`
  - 放置可复用、无 UI 依赖的任务完成谓词。
- `src/simulation/task_verifier.py`
  - 聚合谓词并输出 exact、approximate、failed 或 terminated。
- `src/memory/spatial_memory.py`
  - 存储 pose、可达网格、frontier、探索状态和对象空间证据。
- `src/memory/object_memory.py`
  - 存储对象实例、objectId、类别、置信度、最后观测位置和状态。

如果当前既有文件已经承担相同职责，则不新增重复模块。

### 9.3 必须新增或扩展的测试

- `tests/test_task_planner.py`
- `tests/test_task_semantics.py`
- `tests/test_task_verifier.py`
- `tests/test_action_catalog.py`
- `tests/test_ai2thor_interactions.py`
- `tests/test_ai2thor_postconditions.py`
- `tests/test_spatial_memory.py`
- `tests/test_object_memory.py`
- `tests/test_streaming_events.py`
- `tests/test_visual_causality.py`
- `tests/test_sofa_approximate_sit.py`
- `tests/test_open_pickup_put.py`

测试文件是否拆分，以单一职责和现有测试布局为准；禁止创建内容重复的测试脚本。

---

## 10. Git、代码库和产物清洁规则

### 10.1 禁止提交

- `apikey.txt`、环境变量文件和任何密钥。
- 原始或生成视频。
- 截帧目录。
- API 完整原始响应中可能包含的敏感信息。
- 模型权重、checkpoint、缓存和下载压缩包。
- AI2-THOR Unity 缓存。
- 临时日志、数据库、浏览器 profile、coverage 临时文件。
- `research/codebases` 中完整第三方仓库。

### 10.2 允许提交

- 自研源代码和测试。
- 去敏后的配置模板。
- 固定版本的研究 manifest。
- 小型、可审计、许可证允许的适配代码。
- 指标摘要、验证 JSON 和必要的文档。

### 10.3 每次提交前

1. `git status --short`
2. 检查所有 untracked 文件的用途。
3. `git diff --check`
4. `python -B -m compileall -q src tests tools`
5. `python -B -m unittest discover -s tests -v`
6. 检查是否包含密钥、绝对个人路径、视频、缓存或生成物。
7. 只暂存本阶段已经审核的明确文件，禁止直接 `git add .`。
8. 提交信息必须说明阶段和验证结果。

### 10.4 协作冲突处理

- 发现其他人的未提交修改时先阅读 diff，不覆盖、不回退。
- 同一文件存在并行修改时，先确认双方职责，再做最小合并。
- 重复报告文件中的独有信息应先归并到正式文档，再决定是否删除。
- 删除文件前必须证明没有 import、文档链接、运行脚本或部署流程依赖。
- 本地、GitHub 和远端运行目录最终必须指向同一 Git SHA。

---

## 11. ChangeRecord 标准

每个通过验收的阶段，在 `ChangeRecord/1-9/` 增加一个按现有编号顺序排列的记录。记录必须基于事实，不得把“计划完成”写成“功能完成”。

每份记录至少包含：

1. 阶段目标。
2. 修改文件及职责。
3. 是否修改配置或超参。
4. 配置和超参的事实来源。
5. 运行的测试命令。
6. 测试结果和失败次数。
7. 发现的问题、根因和修复。
8. 本地与远端验证差异。
9. 产物和证据路径。
10. 已知限制。
11. Git commit 和远端 SHA。
12. 下一阶段的进入条件。

没有真实模型调用、真实 Unity 轨迹或浏览器证据时，不得写成对应能力已完成。

---

## 12. 不可跳步的关键路径

严格执行以下顺序：

1. **冻结基线**
   - 解决工作区来源、密钥、配置和协作变更问题。
2. **统一状态语义**
   - 先区分 success、failed、terminated、unsupported 和 approximate。
3. **实现总体任务规划**
   - 让 VLM 把自然语言任务拆成可验证子目标。
4. **实现动作目录与参数绑定**
   - 只有动作可执行，计划才有意义。
5. **实现独立 verifier**
   - 只有完成谓词可靠，Agent 才能正确决定是否结束。
6. **先打通沙发任务**
   - 验证多步规划、移动、姿态动作和 approximate success。
7. **再打通 Open -> Pickup -> Put**
   - 验证 objectId、前置条件、交互状态和连续动作。
8. **加入 VLFM-lite 几何搜索**
   - 替换 UI 3x3 区域式伪地图和固定转向搜索。
9. **加入开放词汇感知**
   - 降低对 AI2-THOR oracle segmentation 的依赖。
10. **加入分层 Memory**
    - 在执行语义稳定后再存储可复用空间、对象和 episode 经验。
11. **加入流式网页**
    - UI 消费已经稳定的事件协议，而不是反向决定 Agent 结构。
12. **完成标准评估任务集**
    - 构建覆盖多场景、多任务、成功和失败的冻结回归版本，不用于模型训练。
13. **冻结 inference-only 模型配置**
    - 验证真实多模态 API、视觉因果关系、结构化输出和 fallback 审计。
14. **完整评估和消融**
    - 证明规划、地图、感知和 memory 的独立价值。
15. **远端部署与视频验收**
    - 最后统一 Git SHA，录制并逐项检查。

依赖关系是：

```text
状态语义
  -> 总体计划
  -> 可执行动作
  -> 可验证完成
  -> 多步任务闭环
  -> 几何探索与开放感知
  -> 分层记忆
  -> 流式展示
  -> 数据、评估和部署
```

不得先制作漂亮 UI 或视频，再用静态轨迹、oracle 信息或硬编码逻辑补齐底层能力。

---

## 13. 第一批立即实施的工作包

第一批只处理“任务被错误地一步结束”这一根问题，不同时大规模改地图和 UI。

### 13.1 工作内容

1. 审阅当前 `task_semantics.py` 和 `model_adapter.py` 的协作者修改。
2. 在 schema 中区分：
   - `planner_done_proposal`
   - `subgoal_completed`
   - `task_success`
   - `episode_terminated`
   - `termination_reason`
3. 实现 `plan_task()`：
   - 输入语言、当前图像、场景能力和动作目录。
   - 输出任务类型、结构化子目标、完成条件和失败策略。
4. 将总体计划持久化到 session state。
5. 每步只让 VLM 选择当前子目标的下一动作，而不是重新猜整个任务。
6. 将 agent pose、heldObjects、目标 objectId、可见性、距离和对象状态传给完成判断。
7. 禁止“检测到 sofa”直接触发 STOP。
8. 完成 sofa 的 locate -> approach -> align -> crouch -> verify 近似闭环。
9. 增加任务未完成时拒绝 STOP 的回归测试。
10. 本地全量测试后，在远端执行真实 API 与 Unity episode。

### 13.2 第一批不得混入

- 不重写全部 UI。
- 不引入完整 ConceptGraphs。
- 不复制 AriGraph 或 VLFM 的整套训练框架。
- 不引入训练、微调、checkpoint 或来源不明的模型参数。
- 不提前声称支持 AI2-THOR 全部动作。
- 不把近似坐下标成原生坐下。

### 13.3 第一批退出条件

- “找到沙发”可以在检测和验证后成功。
- “走到沙发旁”只有在距离和朝向满足后成功。
- “走到沙发并坐下”不会因看见 sofa 提前停止。
- 若场景中存在 sofa，Agent 能执行多步接近和 `Crouch`。
- 若近似谓词通过，结果标记 `approximate_success`。
- 若谓词不通过，必须继续规划或以明确原因终止。
- 真实模型请求确认使用图像。
- 真实 Unity 轨迹和网页状态一致。

---

## 14. 最终验收清单

### 14.1 PPT 基本要求

- [ ] 支持语言指令。
- [ ] 支持图片或点选目标输入。
- [ ] 使用真实 AI2-THOR 场景。
- [ ] 具有机器人第一视角。
- [ ] 具有真实第三方全局相机或明确标记的回退地图。
- [ ] 展示结构化计划、当前子目标、动作、证据和进度。
- [ ] 支持自动执行和人工动作。
- [ ] 支持轨迹管理和回放。
- [ ] 目标完成由视觉和环境状态共同确认。
- [ ] 形成可运行网页和完整演示视频。

### 14.2 Agent 完整性

- [ ] 多模态模型真实接收语言和图像。
- [ ] VLM 先产生总体任务计划。
- [ ] 每个子目标拥有明确完成谓词。
- [ ] 支持重新规划。
- [ ] 动作与 AI2-THOR 参数正确绑定。
- [ ] STOP 受独立 verifier 约束。
- [ ] 能区分 exact、approximate、failed、unsupported 和 terminated。
- [ ] 不把目标检测等同于任务执行完成。

### 14.3 交互完整性

- [ ] OpenObject -> PickupObject -> PutObject 在真实 Unity 中连续通过。
- [ ] Pickup、Put、Open、Close、Toggle、Slice、Throw、Drop、Crouch、Stand 等纳入能力目录。
- [ ] 每项宣称支持的动作均有前置条件、参数、后置条件和真实测试。
- [ ] 未验证动作不得显示为完整支持。

### 14.4 Search 与 Memory

- [ ] 搜索基于真实 pose、可达区域和 frontier。
- [ ] 不使用隐藏目标真值驱动非 oracle 决策。
- [ ] 开放词汇感知可在非 oracle 模式工作。
- [ ] 对象记忆按实例和空间位置维护。
- [ ] 任务记忆记录子目标、证据和失败。
- [ ] episode 记忆能被检索并影响后续决策。
- [ ] 过期或冲突记忆能够失效。

### 14.5 评估任务集与回归

- [ ] 评估任务集有 schema、版本、split、覆盖报告和泄漏检查。
- [ ] 评估任务集仅用于回归、验证和最终评估，不用于模型训练。
- [ ] inference-only 的模型、prompt、Schema 和推理配置均已冻结并可追溯。
- [ ] 开发集用于修复问题，测试集只用于最终评估。
- [ ] 报告 success、SPL、false-stop、动作成功、完成谓词、感知、地图和 memory 指标。
- [ ] 完成规划、地图、感知、memory 和 verifier 消融。
- [ ] 失败 episode 不被删除或隐藏。

### 14.6 工程与演示

- [ ] 全量测试通过，无 `#REF` 类似的运行错误、未捕获异常或静默 fallback。
- [ ] 本地、GitHub、远端代码 SHA 一致。
- [ ] 代码库无密钥、视频、缓存、死代码和无用途报告。
- [ ] 页面流式显示真实进度，不使用预录轨迹冒充在线执行。
- [ ] 视频动作方向、文字时序、地图位置和最终状态逐项核实。
- [ ] 所有演示产物均可追溯到配置、模型、commit 和 episode。

---

## 15. Definition of Done

本计划只有在以下条件全部满足后才算完成：

1. PPT 要求逐项有代码、测试和演示证据。
2. 语言、视觉、总体计划、子目标执行、动作交互、验证、地图和 memory 形成真实闭环。
3. 至少一个视觉搜索任务、一个 sofa 近似坐下任务和一个 Open -> Pickup -> Put 任务在真实 Unity 中通过。
4. 模型失败、能力不支持和任务失败均不会被标记为成功。
5. 所有模型、prompt、结构和推理配置均可追溯；未批准的值没有进入正式 pipeline。
6. 评估任务集准备、回归验证、最终测试和消融符合标准流程，正式 pipeline 不包含训练或微调步骤。
7. 浏览器事件、episode JSON、Unity metadata 和视频内容一致。
8. 全量测试、远端回归、代码清洁和 Git SHA 检查通过。
9. ChangeRecord 完整记录事实、问题、修复和证据。
10. 项目能够由他人在固定环境中复现，而不依赖个人机器上的未提交文件。

该 Definition of Done 高于“网页能打开”“单个 replay 能播放”或“目标物体被检测到”。只有整个任务执行与验证 pipeline 闭环，项目才算真正完成。
