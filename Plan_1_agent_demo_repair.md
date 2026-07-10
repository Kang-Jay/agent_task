# Plan 1: Embodied Visual Search Agent Demo 修复与验收路线

## 0. 目标与边界

本文件是当前项目从“可演示雏形”修复到“可验收完整 demo”的执行指导文件。目标不是快速堆功能，而是在不破坏现有代码和他人改动的前提下，按标准化流程逐步补齐：

- 真实 Agent 主链路：图像观察 + 语言指令输入，模型或明确标注的 fallback planner 输出结构化 action/skill。
- AI2-THOR 可复现演示：真实仿真环境、轨迹、日志、视频、验证报告形成闭环。
- 点选多模态闭环：在网页中点选目标，自动形成 target crop + language instruction，并进入 Agent/demo/video 流程。
- 标准化数据与评估：数据准备、处理、训练或微调、验证、评估、回归测试都有明确入口和指标。
- 代码库清洁：无临时文件、无死代码、无无意义文件夹、无未说明的大文件、无密钥提交风险。

当前已知阻塞项：

- Agent 主链路未调用 `ModelAdapter` 做动作规划。
- 点选坐标只停留在前端 UI，没有进入 demo run 请求。
- 数据集只有极小 happy path，缺少 split、负例、困难样本、多步轨迹、IoU 和 SPL。
- AI2-THOR demo 的 `structured_thought` 与 adapter 覆写后的 action/thought 可能不同步。
- demo/video 生成依赖预生成产物和硬编码路径，复现证据不足。
- 新增测试脚本未纳入常规测试，且在 Windows GBK 控制台下失败。

## 1. 总体原则

1. 每完成一个模块，必须立即运行对应测试；测试不通过不得进入下一模块。
2. 每次改代码后都要检查配置、超参、结构设置是否仍与 `configs/agent_config.json`、schema、pipeline 文档一致。
3. 不允许为了通过 demo 私自降低阈值、减少步数、删掉失败样本、跳过验证或改小训练/评估规模。
4. 不允许把 fallback 伪装成模型决策。所有响应必须标注动作来源：`model_planner`、`rule_fallback`、`simulator_oracle` 或 `human_manual`。
5. 不允许提交 `apikey.txt`、视频大文件、缓存、frame dump、日志 dump、临时图片、`__pycache__`。
6. 不覆盖别人正在修改的文件。每次开始实施前先看 `git status --short` 和相关 diff。
7. 不新增无归属文件夹。所有新增文件必须属于现有目录职责，或者先在本文档中说明新增目录目的。

## 2. 阶段 0: 工作区冻结与基线确认

### 2.1 入口条件

- 本地仓库在 `D:\cache\SummerCap\kaohe\zju`。
- 已确认远程仓库、远程服务器、当前分支和正在被其他人修改的文件。
- 当前阶段只做审查和记录，不修改业务逻辑。

### 2.2 操作步骤

1. 执行：

   ```powershell
   git status --short
   git branch -vv
   git remote -v
   ```

2. 将 dirty files 分类：

   - 本人修改。
   - 他人修改。
   - 生成物。
   - 应纳入版本库的新源码或文档。
   - 应删除或忽略的临时文件。

3. 对当前核心文件保存审查快照，不创建临时副本：

   ```powershell
   git diff -- src/agent src/simulation src/ui src/types src/evaluation src/data configs tests
   ```

4. 明确不得提交内容：

   - `apikey.txt`
   - `*.mp4`
   - frame directories
   - cache directories
   - `__pycache__`
   - 临时测试图片

### 2.3 测试门槛

必须通过：

```powershell
python -m py_compile src/agent/controller.py src/simulation/ai2thor_adapter.py src/simulation/room_simulator.py src/types/schema.py src/ui/app.py
python -B -m unittest discover -s tests -v
git diff --check
```

若失败，停止后续阶段，先定位失败文件和失败原因。

### 2.4 阶段验收

- 清楚知道每个 dirty file 的来源和处理策略。
- 没有删除或覆盖他人改动。
- 基线测试结果已记录到后续 ChangeRecord。

## 3. 阶段 1: 配置与 schema 冻结

### 3.1 目标

先冻结接口和配置，避免后续模块各自定义字段、阈值和结构，导致 pipeline 漂移。

### 3.2 必须确认的文件

- `configs/agent_config.json`
- `src/task/config.py`
- `src/types/schema.py`
- `src/ui/app.py`
- `src/simulation/room_simulator.py`
- `src/simulation/ai2thor_adapter.py`

### 3.3 操作步骤

1. 在 `AgentResponse` 中明确这些字段：

   - `action`
   - `skill_call`
   - `planner_source`
   - `thought`
   - `structured_thought`
   - `observation`
   - `memory_summary`
   - `search_map`
   - `confidence_trace`
   - `target_binding`

2. 如果当前没有 `skill_call` 和 `planner_source`，先补 schema，再补测试。

3. `skill_call` 不应随意发散，先定义最小但完整的 schema：

   ```json
   {
     "name": "TURN_RIGHT",
     "args": {},
     "preconditions": [],
     "expected_observation": "camera heading changes to the right"
   }
   ```

4. `planner_source` 只能来自枚举：

   - `model_planner`
   - `rule_fallback`
   - `simulator_oracle`
   - `human_manual`

5. 所有新增字段必须在 `to_dict()`、demo step、API 返回、UI render 中一致传递。

6. 不改阈值，除非有明确实验依据。当前阈值必须从 `configs/agent_config.json` 读取：

   - `stop_confidence_threshold`
   - `target_visible_threshold`
   - `max_steps`
   - `terminal_actions`
   - `allowed_actions`

### 3.4 测试门槛

新增或更新测试：

- schema 序列化测试。
- `planner_source` 枚举合法性测试。
- `skill_call.name` 必须属于 `allowed_actions`。
- `terminal_actions` 与 `done` 逻辑一致。

运行：

```powershell
python -B -m unittest discover -s tests -v
python -m py_compile src/types/schema.py src/ui/app.py
```

### 3.5 阶段验收

- API 响应结构稳定。
- UI、simulator、evaluation 使用同一 schema。
- 没有自定义未登记字段。

## 4. 阶段 2: 接入真实 Agent 规划主链路

### 4.1 目标

让 `apikey.txt` 或环境变量对应的模型进入 `EmbodiedSearchAgent.step()` 主流程。模型必须接收标准化上下文，输出结构化 action/skill。规则 planner 只作为 fallback，且必须显式标注。

### 4.2 输入结构

模型 planner 输入必须包含：

- `instruction`
- observation image path 或压缩图像摘要
- target crop 信息，如果用户点选了目标
- vision candidates
- current confidence
- session memory
- negative memory
- explored regions
- retrieved hints
- allowed actions
- terminal actions
- current step id
- max steps

### 4.3 输出结构

模型必须返回可解析 JSON：

```json
{
  "thought_summary": "short visible summary, no hidden chain-of-thought",
  "action": {
    "type": "TURN_RIGHT",
    "args": {}
  },
  "skill_call": {
    "name": "TURN_RIGHT",
    "args": {},
    "preconditions": [],
    "expected_observation": "camera heading changes to the right"
  },
  "confidence": 0.42,
  "stop_reason": null
}
```

注意：只展示简短可解释摘要，不输出隐藏思维链。

### 4.4 操作步骤

1. 改造 `src/agent/model_adapter.py`：

   - 支持从环境变量读取 key。
   - 保留 `apikey.txt` 读取，但必须继续被 `.gitignore` 忽略。
   - 增加 `plan_action(payload)` 方法。
   - 明确超时、重试和 JSON 解析失败策略。

2. 改造 `src/agent/controller.py`：

   - 先做视觉分析。
   - 组装 planner payload。
   - 调用 `ModelAdapter.plan_action()`。
   - 校验 action 是否在 `allowed_actions`。
   - 若模型失败或输出非法，调用 `_plan_action()`，并设置 `planner_source=rule_fallback`。
   - 若 AI2-THOR adapter 后续覆盖动作，必须改成 `planner_source=simulator_oracle`。

3. 所有 fallback 都要记录原因：

   - API key 缺失。
   - API 超时。
   - JSON 解析失败。
   - 非法 action。
   - 置信度违反停止规则。

4. 不允许模型直接绕过停止规则。`STOP` 必须同时满足：

   - action 是 `STOP`。
   - confidence 达到配置阈值。
   - target visible 或 segmentation confirmed。

### 4.5 测试门槛

新增测试：

- mock model 返回合法 action，Agent 使用 `model_planner`。
- mock model 返回非法 action，Agent fallback 到 `rule_fallback`。
- mock model 返回 `STOP` 但 confidence 不足，Agent 不允许停止。
- API key 缺失时系统仍可用，但 response 明确 fallback。
- `planner_source` 出现在 API 返回和 demo step 中。

运行：

```powershell
python -B -m unittest discover -s tests -v
python -m py_compile src/agent/controller.py src/agent/model_adapter.py
```

### 4.6 阶段验收

- 可以证明 Agent 主链路调用了模型 planner。
- 可以证明 fallback 没有被伪装成模型输出。
- 可以证明停止规则没有被模型绕过。

## 5. 阶段 3: 点选多模态闭环

### 5.1 目标

实现 PPT 要求的点选目标闭环：用户在网页图像上点选目标，系统自动生成 target crop，并将 crop + language instruction 输入 Agent，最终影响 action/skill 和 demo video。

### 5.2 操作步骤

1. 前端修复：

   - `clickedPoint` 不能只保存到 JS 变量。
   - `runDemo()` 请求必须传递 `clicked_point`。
   - 页面必须显示当前是否为 `language_only` 或 `multimodal`。
   - 点击后可以单步调用 `/api/agent/step` 验证响应。

2. 后端修复：

   - `/api/demo/run` 支持 `clicked_point`。
   - `/api/demo/ai2thor/run` 支持 `clicked_point`。
   - simulator 将第一帧 observation 与 clicked point 送入 Agent。
   - AI2-THOR adapter 在第一帧生成 crop 或绑定 target object。

3. 数据结构修复：

   - `target_binding.clicked_point`
   - `target_binding.target_crop`
   - `target_binding.mode`
   - `target_binding.crop_source`
   - `target_binding.crop_bbox`

4. UI 文案修复：

   - 未接入前不得声称“下次运行将使用多模态搜索”。
   - 接入后必须显示实际传入后端的 `clicked_point` 和 `crop_bbox`。

### 5.3 测试门槛

新增测试：

- 前端请求体包含 `clicked_point`。
- `/api/demo/run` 接收到 clicked point 后返回 `target_binding.mode=multimodal`。
- `/api/agent/step` 使用 clicked point 自动生成 crop。
- 点击错误区域时不应强行高置信 STOP。
- 视频 summary 中记录 target binding。

运行：

```powershell
python -B -m unittest discover -s tests -v
python -m py_compile src/ui/app.py src/simulation/room_simulator.py src/simulation/ai2thor_adapter.py
```

如果使用浏览器测试：

```powershell
python -m src.ui.app
```

然后检查：

- 点击坐标出现在 network payload。
- API response 中 `target_binding.mode` 为 `multimodal`。
- UI 中 mode label 与 response 一致。

### 5.4 阶段验收

- 点选行为真实改变 Agent 输入。
- 点选行为能进入 demo trajectory。
- 点选行为能进入 video 或 summary。

## 6. 阶段 4: AI2-THOR 真实仿真与 demo 证据链

### 6.1 目标

让 AI2-THOR demo 可复现、可审计、可证明，不依赖陈旧产物或硬编码路径。

### 6.2 操作步骤

1. 移除 demo 生成脚本中的硬编码仓库路径。

   - 使用 `Path(__file__).resolve().parents[...]`。
   - 或通过 CLI 参数传入 project root。

2. 将 demo 生成拆成三个命令：

   - run AI2-THOR trajectory。
   - verify trajectory。
   - render video。

3. 每次 AI2-THOR run 必须记录：

   - scene name。
   - AI2-THOR package version。
   - Unity/CloudRendering mode。
   - controller initialization status。
   - start pose。
   - action list。
   - visible objects。
   - segmentation target confirmation。
   - final stop reason。
   - planner source per step。

4. 修复 `structured_thought` 不同步：

   - adapter 覆写 `action/thought/confidence` 时，同步覆写 `structured_thought`。
   - UI 展示“当前观察”“下一步动作”“执行后观察”，避免 TURN_RIGHT 画面方向误解。

5. 严格模式和 fallback 模式分离：

   - strict AI2-THOR 不 fallback。
   - local demo 必须明确标注 local。
   - fallback 只在用户显式选择时发生。

### 6.3 测试门槛

本地无 AI2-THOR 时：

```powershell
python -B -m unittest discover -s tests -v
python -m py_compile src/simulation/ai2thor_adapter.py tools/make_cinematic_demo.py
```

远程 3090GPU2 上：

```bash
python -m src.ui.app
python tools/run_ai2thor_demo.py --scene FloorPlan211 --instruction "Find the television in the room" --strict
python tools/verify_ai2thor_demo.py --summary docs/ai2thor_outputs/ai2thor_demo_summary.json
python tools/make_cinematic_demo.py --summary docs/ai2thor_outputs/ai2thor_demo_summary.json
```

验收必须包含：

- summary JSON。
- verification JSON。
- demo video。
- run log。
- environment/version report。

### 6.4 阶段验收

- 新环境可从命令生成 demo，而不是依赖旧文件。
- 每个 step 的 action、thought、structured thought、planner source 一致。
- strict AI2-THOR 和 local fallback 不混淆。

## 7. 阶段 5: 标准化数据准备与处理

### 7.1 目标

建立可用于训练、验证和评估的数据管线，而不是只靠 3 条 demo annotation。

### 7.2 数据要求

数据至少包含：

- 多场景：厨房、客厅、卧室、浴室、办公室或对应 AI2-THOR FloorPlan。
- 多目标：电视、杯子、书、植物、遥控器、瓶子、椅子等。
- 多语言指令：英文、中文、同义表达、位置描述。
- 多步轨迹：搜索、转向、前进、检查、停止。
- 正例：目标可见、目标部分遮挡、目标远距离。
- 负例：目标不存在、相似干扰物、点击错误物体、低置信度目标。
- 困难样本：反光、遮挡、小目标、多目标同屏。

### 7.3 文件结构建议

使用现有 `datasets/embodied_search_v1`，不要新建无意义目录：

```text
datasets/embodied_search_v1/
  annotations/
    episodes.jsonl
    splits.json
    schema.json
  images/
  crops/
  trajectories/
  metadata/
    dataset_card.md
    generation_config.json
    validation_report.json
```

### 7.4 处理流程

1. 生成或采集 raw episodes。
2. 统一 schema。
3. 自动裁剪目标 crop。
4. 生成 train/val/test split。
5. 检查 split 泄漏：

   - 同一 scene 不应同时出现在 train 和 test，除非文档明确声明。
   - 同一轨迹变体不应跨 split。
   - 同一目标实例不应泄漏到 test。

6. 生成 dataset card。
7. 生成 validation report。

### 7.5 测试门槛

新增测试：

- annotation schema 合法。
- bbox 在图像边界内。
- split 文件存在。
- split 无泄漏。
- 每个 split 包含正例和负例。
- 每个 episode 有终止条件。
- 多步 episode 包含 pose/action/observation。

运行：

```powershell
python -B -m unittest discover -s tests -v
python -m src.data.generate_demo_dataset
python -m src.evaluation.evaluator
```

### 7.6 阶段验收

- 数据不再只是 happy path。
- train/val/test 明确且可复现。
- validation report 能阻止坏 bbox、缺 split、缺负例。

## 8. 阶段 6: 训练或微调 pipeline

### 8.1 目标

如果项目声明训练或微调，就必须提供可复现 pipeline。若不训练，必须明确声明这是 inference-only demo，并用模型 API 做 planning。

### 8.2 两种合法路线

路线 A: inference-only Agent

- 不声称微调。
- 不写虚假训练参数。
- 重点验证 prompt、schema、model planner、fallback、simulation。

路线 B: 训练或微调 Agent

- 提供训练数据。
- 提供训练配置。
- 提供 checkpoint 管理。
- 提供验证和测试指标。

### 8.3 若选择训练路线，必须定义

- model name。
- input format。
- output format。
- train/val/test split。
- batch size。
- learning rate。
- epoch 或 steps。
- optimizer。
- scheduler。
- seed。
- checkpoint interval。
- early stopping。
- validation metric。
- final test metric。

所有超参必须写入配置文件，不能散落在脚本中。

### 8.4 测试门槛

训练前：

```powershell
python -m src.data.validate_dataset
python -m src.training.validate_config
```

训练中：

- 每个 epoch 或固定 step 生成 validation metric。
- 保存 best checkpoint。
- 保存训练日志。

训练后：

```powershell
python -m src.evaluation.evaluator --split test --checkpoint <best_checkpoint>
```

### 8.5 阶段验收

- 有可复现训练命令。
- 有固定 seed。
- 有 best checkpoint 选择依据。
- test set 只在最终评估使用。

## 9. 阶段 7: 标准评估指标

### 9.1 必须实现的指标

- success rate。
- STOP accuracy。
- illegal action rate。
- average confidence。
- bbox IoU。
- center point error。
- candidate recall@k。
- SPL。
- normalized path length。
- timeout rate。
- repeated exploration rate。
- target absent false stop rate。

### 9.2 评估流程

1. 读取 split。
2. 初始化环境或 replay environment。
3. 对每个 episode 逐步执行 Agent。
4. 每步记录：

   - observation id。
   - action。
   - skill call。
   - planner source。
   - confidence。
   - bbox。
   - pose。
   - memory update。

5. episode 结束后计算指标。
6. 输出 JSON 和 Markdown 报告。

### 9.3 测试门槛

新增 failure-first 测试：

- 错 bbox 必须导致 IoU 失败。
- target absent 时 STOP 计为 false stop。
- path 过长时 SPL 降低。
- 非法 action 被计入 illegal action。
- 超过 max steps 计为 timeout。

运行：

```powershell
python -B -m unittest discover -s tests -v
python -m src.evaluation.evaluator --split val
```

### 9.4 阶段验收

- 指标能区分“碰巧 STOP 成功”和“真实搜索成功”。
- 评估报告能暴露误停、绕路、重复探索、定位错误。

## 10. 阶段 8: Web demo 完整交互

### 10.1 目标

网页必须成为最终可演示入口，而不是只播放旧视频。

### 10.2 必备功能

- 任务指令输入。
- 后端选择：strict AI2-THOR、local demo、fallback demo。
- 机器人 POV。
- 顶视图或搜索地图。
- 当前观察。
- 下一步 action。
- skill call。
- planner source。
- confidence。
- memory 面板。
- negative memory 面板。
- trajectory timeline。
- manual action buttons。
- 点选目标 crop 预览。
- demo recording。
- verification report link。

### 10.3 UI 约束

- 不展示隐藏思维链，只展示简短可解释摘要。
- 不把 fallback 写成 real AI2-THOR。
- 不把 simulator oracle 写成模型自主决策。
- 动作时序必须清楚：

  - current observation。
  - next action。
  - observation after action。

### 10.4 测试门槛

浏览器自动测试：

- 页面能加载。
- run demo 后有 steps。
- 每个 step 有 POV、action、confidence、planner source。
- 点击图像后 payload 包含 clicked point。
- memory panel 更新。
- video path 存在。

运行：

```powershell
python -m src.ui.app
python -B -m unittest discover -s tests -v
```

若使用 Playwright 或浏览器工具，必须保存截图检查：

- desktop viewport。
- mobile viewport。
- run 后状态。
- clicked target 后状态。

### 10.5 阶段验收

- 用户可以在网页完成完整 demo。
- 网页信息与后端 summary 一致。
- demo video 可从当前 run 生成。

## 11. 阶段 9: 最终验收与代码库清洁

### 11.1 最终测试清单

本地：

```powershell
git status --short
git diff --check
python -m py_compile src/agent/controller.py src/agent/model_adapter.py src/simulation/ai2thor_adapter.py src/simulation/room_simulator.py src/types/schema.py src/ui/app.py
python -B -m unittest discover -s tests -v
python -m src.data.generate_demo_dataset
python -m src.evaluation.evaluator --split val
```

远程 AI2-THOR：

```bash
python tools/run_ai2thor_demo.py --scene FloorPlan211 --instruction "Find the television in the room" --strict
python tools/verify_ai2thor_demo.py --summary docs/ai2thor_outputs/ai2thor_demo_summary.json
python tools/make_cinematic_demo.py --summary docs/ai2thor_outputs/ai2thor_demo_summary.json
```

### 11.2 最终人工检查

- README 是否与真实行为一致。
- ChangeRecord 是否记录每阶段改动与测试。
- `Plan_1_agent_demo_repair.md` 中的阶段是否全部完成或明确未完成。
- `.gitignore` 是否覆盖密钥、大视频、缓存、frames、日志。
- 工作树是否只包含应提交源码和文档。
- 没有临时图片。
- 没有测试运行残留。
- 没有乱码文档。
- 没有硬编码本地绝对路径。
- 没有死代码和未调用模块。

### 11.3 最终验收标准

只有同时满足以下条件，才能声明完成：

- Agent 主链路真实使用模型 planner，或明确声明 inference-only/fallback 边界。
- 图像 + 语言 + 点选 crop 能共同影响 action/skill。
- AI2-THOR strict demo 可复现。
- demo video 来自当前 run。
- evaluation 能覆盖正例、负例、困难样本、多步路径和定位质量。
- 测试全部通过。
- 代码库干净。

## 12. 推荐实施顺序

必须按以下顺序执行，不跳步：

1. 工作区冻结与基线确认。
2. schema 和配置冻结。
3. 模型 planner 接入主链路。
4. 点选多模态闭环。
5. AI2-THOR thought/action 同步和证据链修复。
6. 数据集与 split 扩展。
7. 训练或 inference-only 声明与 pipeline 固化。
8. 标准评估指标。
9. Web demo 完整交互。
10. 最终清洁与验收。

每一步失败时，只回到当前阶段修复，不跨阶段补丁式绕过。

## 13. 每次提交前检查模板

提交前必须回答：

- 本次改动属于哪个阶段？
- 是否修改了配置、超参、schema？
- 如果修改了，是否同步了测试、文档、UI、evaluation？
- 是否运行了本阶段要求的测试？
- 是否产生了临时文件？
- 是否有密钥、大视频、缓存、frames 被误加入？
- 是否覆盖了他人改动？
- 是否有 fallback 被误标为模型输出？
- 是否有 AI2-THOR strict 与 local fallback 混淆？

推荐命令：

```powershell
git status --short
git diff --check
python -B -m unittest discover -s tests -v
```

## 14. ChangeRecord 要求

每完成一个阶段，写入 `ChangeRecord/1-9/`：

- 文件名按现有顺序递增。
- 记录改动目的。
- 记录涉及文件。
- 记录测试命令和结果。
- 记录未解决风险。
- 记录是否有远程 AI2-THOR 验证。

不得只写“已完成”，必须写清楚证据。

## 15. 当前最小可行修复包

如果要先做一个能明显提升可信度的短周期修复包，顺序如下：

1. 增加 `planner_source` 和 `skill_call` schema。
2. 将 `ModelAdapter.plan_action()` 接入 `EmbodiedSearchAgent.step()`，失败时 fallback。
3. 修复前端 `clickedPoint` 进入 demo 请求。
4. 修复 AI2-THOR adapter 覆写后 `structured_thought` 不同步。
5. 将 `test_improvements.py` 改成 `tests/test_improvements.py`，修复 Windows 编码问题。
6. 增加至少 5 个 failure-first 测试：

   - 非法模型 action fallback。
   - 低置信 STOP 被拒绝。
   - clicked point 进入 target binding。
   - target absent 不 STOP。
   - structured thought 与 final action 一致。

7. 运行全量测试。
8. 写 ChangeRecord。

这个最小修复包只解决主链路可信度和交互闭环，不替代完整数据、训练、评估 pipeline。
