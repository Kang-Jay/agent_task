# 10020 Plan2 动作后视频对齐、中文字体渲染与流式 POV 修复

## 1. 记录目的

本记录对应 `Plan_2_hierarchical_embodied_agent_upgrade.md` 中真实 AI2-THOR 演示的可审计性和展示一致性要求。

本轮只处理以下问题：

1. 模型决策使用动作前 RGB，但视频错误地继续展示动作前 RGB。
2. 视频主画面、全局地图、机器人状态和任务完成状态不属于同一个时间点。
3. 最终 `Crouch` 虽已由 Unity 和后置条件验证，但原视频无法直观看到动作后的相机状态。
4. AI2-THOR 合成画面没有显式加载中文字体，中文指令和 Thought 可能显示为方框。
5. 中文文本按空格换行，不适用于连续中文句子。
6. 动作后的地图仍使用容易误解为下一动作或动作前状态的标签。
7. 流式网页在 `environment_feedback` 到达后没有立即显示动作后 POV。

本轮不修改 Agent 的任务理解、动作规划、任务完成定义、模型调用参数、动作空间、训练流程或仿真物理参数。

本记录不能用于宣称整个 Plan2 已完成。

## 2. 修改前基线

### 2.1 Git 基线

- 本轮修改前本地 `HEAD`：

```text
3656806020782a30af2f3d431d375f0fc9a317e8
```

- 修改前本地 `main`、GitHub `origin/main` 和 3090GPU2 已提交版本一致。
- 3090GPU2 项目目录：

```text
/home/scale/kangjay/kaohe
```

- 本地项目目录：

```text
D:\cache\SummerCap\kaohe\zju
```

### 2.2 配置冻结基线

`configs/agent_config.json` Git blob：

```text
e9311e26ec93dab9b28941b611d1324bd3cabdf5
```

冻结配置包括但不限于：

- `max_steps = 20`
- `history_window = 6`
- `stop_confidence_threshold = 0.78`
- `target_visible_threshold = 0.58`
- `default_turn_angle_degrees = 30`
- `image_size = [448, 448]`
- `memory.long_term_capacity = 200`
- `memory.negative_memory_capacity = 80`
- `memory.retrieval_top_k = 3`
- AI2-THOR `gridSize = 0.25`
- Kimi 模型 `kimi-k2.6`
- thinking 请求 `temperature = 1.0`
- thinking 请求 `max_tokens = 2048`
- 模型超时 `90s`

本轮不得修改上述值。

### 2.3 修改前测试基线

修改前全量测试：

```text
Ran 194 tests
OK (skipped=2)
```

两项跳过测试是由 `RUN_LIVE_MODEL_TESTS` 显式控制的真实付费 API 测试，不是功能失败。

## 3. 根因分析

### 3.1 动作前后画面混用

原始循环在调用 Agent 前保存：

```text
ai2thor_obs_XX.png
```

该图片来自当前 `event.frame`，属于动作前观察，也确实被发送给 VLM。

执行动作后，系统使用 `next_event` 计算：

- `robot_after`
- 后置条件
- 完成状态
- 可见物体
- Unity 3D 全局地图

但是合成视频和 `DemoStep.observation_path` 仍引用动作前图片。

因此同一个步骤中混入了：

- 动作前 POV
- 动作后地图
- 动作后机器人状态
- 动作后完成状态

这会造成以下错误观感：

- 标注 `TURN_RIGHT`，画面却不像右转后的结果。
- 最终标注 `Crouch`，主 POV 仍是蹲下前的画面。
- 地图位置已经更新，但机器人视角未更新。

### 3.2 机器人轨迹缺少动作后终点

原实现只在执行前将 `robot_before` 加入 `agent_path`。

动作后地图虽然使用 `next_event` 的机器人图标，但轨迹折线不包含当前动作后的终点，导致轨迹和机器人图标之间可能相差一个动作。

### 3.3 地图相位标签错误

原地图使用：

```text
next ACTION
before ACTION
```

但地图本身已经使用动作后的 `next_event`。

因此地图标签必须改成：

```text
after ACTION
```

### 3.4 中文字体没有接入 AI2-THOR 合成器

`RoomSimulator` 已存在 CJK 字体候选：

- Windows 微软雅黑
- Windows 黑体
- Linux Noto Sans CJK
- Linux 文泉驿
- DejaVu Sans 回退

但 AI2-THOR 合成器直接调用 `ImageDraw.text()`，没有传递 `font=`。

Pillow 默认字体不保证中文字符，因此真实演示可能显示缺失字形方框。

### 3.5 中文文本换行算法不适用

原 `_wrapped_text()` 使用：

```python
text.split()
```

连续中文句子没有空格，会被当作一个超长 token，无法按照可用像素宽度换行。

### 3.6 流式页面显示延迟

后端的 `environment_feedback` 已能携带动作后信息，但前端只在 `step_completed` 时重新渲染步骤。

在地图渲染、帧合成和视频准备期间，页面仍停留在动作前 POV，降低流式演示的实时性。

## 4. 标准化实施顺序

以下步骤必须按顺序执行。前一阶段未通过时，不允许进入后一阶段。

### 阶段 A：冻结基线

1. 确认本地分支为 `main`。
2. 确认 `main...origin/main` 没有提交差异。
3. 记录修改前 `HEAD`。
4. 记录 `agent_config.json` blob。
5. 运行修改前全量测试。
6. 确认没有暂存内容。

退出条件：

- Git 基线明确。
- 配置哈希明确。
- 194 项基线测试通过。

### 阶段 B：定义时间相位契约

统一定义：

#### 动作前观察

- 来源：当前 `event.frame`
- 用途：VLM 输入和模型审计
- 文件名：

```text
ai2thor_obs_XX.png
```

- 流事件字段：

```json
{
  "observation_phase": "before_action",
  "purpose": "model_input_audit"
}
```

#### 动作后观察

- 来源：执行后的 `next_event.frame`
- 用途：
  - 网页实时 POV
  - `DemoStep.observation_path`
  - 合成帧主 POV
  - H.264 视频
  - 最终证据
- 文件名：

```text
ai2thor_obs_after_XX.png
```

退出条件：

- 模型输入与展示输出不再使用同一语义含混路径。
- 动作前和动作后文件均被保留。

### 阶段 C：动作后帧实现

1. 执行动作后立即从 `next_event.frame` 构造 RGB 图片。
2. 保存 `ai2thor_obs_after_XX.png`。
3. 合成视频主 POV 改用动作后图片。
4. `DemoStep.observation_path` 改用动作后图片。
5. `DemoStep.robot` 改用 `robot_after`。
6. `environment_feedback` 增加动作后图片路径和相位字段。
7. `STOP`、`Done`、`ASK_CLARIFY` 明确标注没有 Unity 状态迁移。
8. 执行失败或后置条件失败时标注为 action attempt failed，不得伪装成成功。

退出条件：

- 单元测试能够证明模型仍接收动作前红色图像。
- 步骤、视频和网页展示动作后蓝色图像。
- `DemoStep.robot` 与动作后坐标一致。

### 阶段 D：地图对齐

1. 在执行前按需添加 `robot_before`，避免重复点。
2. 在执行后按需添加 `robot_after`，避免重复点。
3. 动作后地图使用完整轨迹终点。
4. Unity 3D 地图标签改成 `after ACTION`。
5. 2D fallback 地图标签改成 `after ACTION`。

退出条件：

- 轨迹最后一点等于当前动作后的机器人位置。
- 地图不再使用 `next ACTION` 或 `before ACTION` 描述动作后状态。

### 阶段 E：CJK 字体模块

1. 从 `RoomSimulator._load_font()` 抽取模块级：

```python
load_render_font(size)
```

2. 保留旧静态方法作为兼容委托。
3. 按既有候选顺序查找字体。
4. 单个字体损坏时继续尝试下一候选。
5. 全部候选缺失时使用 Pillow 默认字体回退。
6. 不增加第三方 Python 依赖。
7. 不将字体二进制提交到仓库。

退出条件：

- 字体候选顺序测试通过。
- 损坏字体跳过测试通过。
- 缺失字体回退测试通过。
- 旧接口兼容测试通过。

### 阶段 F：AI2-THOR 字体接线与中文换行

1. AI2-THOR 合成帧导入 `load_render_font()`。
2. 标题、指令、观察相位、动作、置信度、Thought、Visible 均传入显式字体。
3. Unity 3D 地图和 2D fallback 地图文字均传入显式字体。
4. `_wrapped_text()` 改为：
   - 中英文混合 token 化。
   - 使用 `textbbox()` 测量真实像素宽度。
   - 支持连续中文字符换行。
   - 限制最大行数。
   - 超出时增加省略号。

退出条件：

- 中文指令可见。
- 中文 Thought 可见。
- 中文换行不越出右侧决策面板。
- 不显示方框字或问号替代。

### 阶段 G：网页动作后 POV

1. 前端处理 `environment_feedback`。
2. 仅当：

```text
observation_phase == "after_action"
```

且存在 `observation_path` 时更新 POV。
3. UI phase 更新为：

```text
observing after action
```

退出条件：

- 前端契约测试能够找到动作后相位门控。
- 前端不把任意未知路径误当成动作后观察。

### 阶段 H：本地分模块测试

按顺序运行：

```powershell
python -B -m unittest discover -s tests -p test_render_fonts.py -v
python -B -m unittest discover -s tests -p test_ai2thor_post_action_rendering.py -v
python -B -m unittest discover -s tests -p test_ai2thor_sync.py -v
python -B -m unittest discover -s tests -p test_ui_stream_contract.py -v
```

任一模块失败时：

- 停止进入全量回归。
- 修复当前模块。
- 重跑当前模块。
- 不通过修改配置或降低验收阈值规避失败。

### 阶段 I：本地全量回归

运行：

```powershell
python -B -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
```

退出条件：

- 全量测试全部通过。
- 仅允许由真实 API 开关控制的两项测试跳过。
- 编译通过。
- 无空白错误。
- `agent_config.json` blob 未变化。

### 阶段 J：本地视觉检查

1. 使用 Unicode 转义构造中文测试文本，避免 PowerShell 管道编码替换。
2. 生成 1600×900 合成帧。
3. 记录实际字体路径。
4. 检查：
   - 中文指令。
   - 中文 Thought。
   - Thought 换行。
   - 动作后标签。
   - 文字是否越界。
   - 面板是否遮挡。
5. 比较“中/文/沙/发”与 `?`、`□` 的字形像素摘要。

退出条件：

- 代表性中文字形彼此不同。
- 代表性中文字形不等于缺失字符。
- 人工检查无方框字和布局溢出。

### 阶段 K：提交与三端同步

本地必须先完成所有本地门禁，再执行：

```powershell
git add <本轮精确文件>
git diff --cached --check
git diff --cached
git commit -m "fix: align post-action demo rendering"
git push origin main
```

远端仅执行：

```bash
cd /home/scale/kangjay/kaohe
git pull --ff-only origin main
```

禁止：

- 远端 merge。
- 远端强制 reset。
- `stash pop`。
- 删除既有远端验证产物。

退出条件：

- 本地、GitHub、3090GPU2 `HEAD` 完全一致。

### 阶段 L：远端测试与真实 Unity 演示

1. 在 3090GPU2 运行全量测试。
2. 确认远端字体：

```text
/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

3. 使用真实 AI2-THOR FloorPlan211。
4. 使用真实多模态模型 API。
5. 指令：

```text
找到房间里的沙发并坐下
```

6. 禁止使用 `TeleportFull` 代替 Agent 导航。
7. 必须保存：
   - 动作前 RGB。
   - 动作后 RGB。
   - Unity 3D 全局地图。
   - 合成帧。
   - summary JSON。
   - H.264 MP4。
8. 最终必须满足：
   - 真实导航动作成功。
   - 同一 Sofa objectId 持续绑定。
   - approach 已验证。
   - `Crouch` 真实执行成功。
   - `agent.isStanding=False`。
   - `completion_status.outcome=approximate_success`。
   - 最终视频主 POV 来自动作后 Crouch 帧。

### 阶段 M：视频逐帧验收

1. 下载新 MP4 到本机忽略目录。
2. 验证编码：
   - H.264
   - `yuv420p`
   - 1600×900
   - 2 FPS
3. 顺序解码全部视频帧。
4. 按步骤生成 contact sheet。
5. 检查每一步：
   - 动作标签。
   - 动作后 POV。
   - 动作后地图。
   - 轨迹终点。
   - 机器人 heading。
   - 中文指令。
   - 中文 Thought。
6. 重点检查：
   - TURN_LEFT。
   - TURN_RIGHT。
   - 最后一个 MOVE_FORWARD。
   - Crouch。
7. 最后一组视频帧必须映射到最终 `Crouch` 的动作后合成帧。

## 5. 实际修改文件

### 5.1 生产代码

- `src/simulation/ai2thor_adapter.py`
  - 保存动作前/动作后观察。
  - 展示和视频改用动作后观察。
  - 步骤机器人状态改用 `robot_after`。
  - 轨迹加入动作后终点。
  - 地图标签改为动作后相位。
  - 合成器和地图显式使用共享字体。
  - 中文按像素宽度换行。

- `src/simulation/room_simulator.py`
  - 新增共享 `load_render_font(size)`。
  - 保留 `_load_font()` 兼容接口。

- `src/ui/static/index.html`
  - `environment_feedback` 到达后立即显示动作后 POV。

### 5.2 测试

- `tests/test_ai2thor_post_action_rendering.py`
  - 动作前 VLM 输入。
  - 动作后步骤和视频。
  - `robot_after`。
  - 轨迹终点。
  - 终止动作无 transition 标签。
  - 失败动作标签。
  - 地图动作后相位。
  - 共享字体接线。

- `tests/test_render_fonts.py`
  - CJK 候选顺序。
  - 首个可用字体。
  - 损坏字体跳过。
  - 字体缺失回退。
  - 旧接口委托。
  - 中文字形不等于 `?` 或方框。

- `tests/test_ui_stream_contract.py`
  - 动作后相位门控。
  - 动作后 POV 更新。
  - UI 相位可见。

## 6. 失败记录

### 6.1 子任务额度与限流

8 路并行任务均已启动。

其中三路第一次运行受到外部服务额度或限流影响：

- 两路返回 402 每日额度限制。
- 一路达到 429 重试上限。

处理：

1. 关闭失败会话。
2. 使用轻量高推理会话重新启动相同审计任务。
3. 不让失败会话写入代码。
4. 最终获得 8 路有效结果，并由主线程逐文件交叉验证。

### 6.2 两次补丁上下文失败

为地图字体和 UI 同时应用补丁时，因当前文件上下文与预期不一致，`apply_patch` 两次拒绝应用。

处理：

1. 确认失败补丁没有产生部分写入。
2. 重新读取精确行。
3. 将地图和 UI 拆分为两个小补丁。
4. 使用纯 ASCII 锚点修改 UI，避免旧文件乱码文本影响匹配。

### 6.3 第一次本地中文预览显示问号

第一次预览通过 PowerShell here-string 直接传递中文给 Python，中文在管道中被替换为 `?`。

该问题不是字体实现失败。

处理：

1. 改用 Python Unicode 转义构造同一中文文本。
2. 确认实际字体为：

```text
C:\Windows\Fonts\msyh.ttc
```

3. 生成第二张预览图。
4. 中文正常显示。
5. 检查代表性字形摘要：
   - `中`
   - `文`
   - `沙`
   - `发`
   - `?`
   - `□`
6. 中文字形摘要互不相同，也不等于缺失字符。

## 7. 本地测试结果

### 7.1 字体模块

```text
Ran 6 tests
OK
```

### 7.2 动作后帧模块

```text
Ran 5 tests
OK
```

### 7.3 地图方向与同步模块

```text
Ran 10 tests
OK
```

### 7.4 UI 流式契约

```text
Ran 2 tests
OK
```

### 7.5 最终全量测试

```text
Ran 207 tests
OK (skipped=2)
```

同时通过：

- `python -m compileall -q src tests`
- `git diff --check`

仅存在 Windows `core.autocrlf` 引起的 LF/CRLF 提示，没有空白错误。

最终配置 blob 仍为：

```text
e9311e26ec93dab9b28941b611d1324bd3cabdf5
```

## 8. 本地视觉验收

本地生成：

```text
C:\Users\21147\AppData\Local\Temp\ai2thor_cjk_post_action_preview_unicode.png
```

人工检查结果：

- 中文指令正常。
- 中文 Thought 正常。
- 中文 Thought 自动换行。
- 文字未越过决策面板。
- 无方框字。
- 无问号替代。
- `Decision before action` 标签明确。
- `Observation after action: Crouch` 标签明确。
- 动作、置信度和 Thought 没有互相遮挡。

临时预览文件位于系统临时目录，不提交仓库。

## 9. 数据、训练与验证边界

本轮是展示证据和时间相位修复，不涉及训练。

未执行：

- 数据集扩充。
- train/validation/test 重新切分。
- 模型训练。
- 模型微调。
- checkpoint 生成。
- optimizer 或 scheduler 设置。
- epoch、batch size、learning rate 设置。

原因：

- Plan2 当前正式路线是 inference-only。
- 多模态 API 负责规划。
- 确定性 executor 和 verifier 负责执行与完成验收。
- 数据集用于固定回归、评估和消融，不用于本轮渲染模块训练。

这不意味着整个项目可以缺少标准化评估集。

后续仍需冻结：

- scene
- seed
- initial pose
- instruction
- target objectId
- oracle/non-oracle 模式
- success predicate
- SPL
- collision
- false success

## 10. 仓库清洁约束

本轮提交只允许包含：

- 上述生产代码。
- 上述测试。
- 本 ChangeRecord。

不得提交：

- `apikey.txt`
- MP4
- 运行帧目录
- 浏览器缓存
- Python cache
- 日志
- Unity cache
- 临时预览图
- 原始 API 请求或响应
- 远端未跟踪的历史验证产物

远端现有未跟踪 AI2-THOR 证据目录不得删除或覆盖。

## 11. 当前明确未完成项

### 11.1 当前工作包远端门禁

在本记录首次写入时，下列事项仍待执行：

- 本地提交和推送。
- 3090GPU2 `git pull --ff-only`。
- 远端 207 项测试。
- 真实 Unity 新 episode。
- 新视频下载。
- 新视频逐帧审查。
- 网页服务启动。
- 浏览器流式演示检查。

完成后必须在本记录追加真实证据，不能仅依赖本地 mock。

### 11.2 流式协议剩余问题

本轮没有解决：

- 重复运行的 episode/run 隔离。
- 前端旧流污染新任务。
- 真模型请求与一般 planning 事件的语义分离。
- `model_decision` 与最终执行动作的事件顺序。
- 所有异常路径的唯一终止事件。
- 取消模型请求和 Unity 初始化。
- token usage、request ID、延迟和 provider retry 可视化。

这些属于后续独立工作包，不能混入本轮小修复后宣称已完成。

### 11.3 Cinematic 工具剩余问题

`tools/make_cinematic_demo.py` 仍按旧语义把 `DemoStep.observation_path` 当成动作前观察。

当前 `DemoStep.observation_path` 已改为动作后观察，因此在修复该工具前：

- 禁止使用旧脚本重新生成正式 cinematic 视频。
- 禁止将旧 `NEXT ACTION` 标签作为新时间相位证据。
- 本轮远端验收只使用 `AI2ThorVisualSearchDemo` 直接生成的视频。

后续必须独立修复：

- post-action 语义。
- CLI 参数。
- 硬编码项目路径。
- import 时产生文件副作用。
- overlay 遮挡。
- verification JSON 的 codec、pixel format、commit 和 hash。

### 11.4 Plan2 其余工作

仍未完成：

- 非 oracle RGB-D occupancy map。
- frontier exploration。
- semantic value map。
- oracle 与 non-oracle 分离评估。
- OpenObject → PickupObject → PutObject 真实连续闭环。
- 更完整的交互动作后置条件。
- object/spatial/task/failure/skill/episode 分层 memory。
- memory 可视化。
- 固定评估集和多场景统计。
- API deadline、retry、routing 和认证加固。
- 依赖锁定和研究代码 manifest。

## 12. 当前阶段判定

本地阶段判定：

```text
动作后帧契约：通过
地图相位与轨迹终点：通过
CJK 字体加载：通过
CJK 字形验证：通过
中文像素换行：通过
网页动作后 POV 契约：通过
全量回归：通过
配置一致性：通过
远端真实 Unity：待执行
真实视频逐帧验收：待执行
```

只有在远端真实 Unity、新视频和浏览器检查全部通过后，才能将本工作包标记为完整完成。
