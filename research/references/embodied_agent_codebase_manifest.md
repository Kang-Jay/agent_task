# Embodied Agent Codebase Manifest

更新时间：2026-07-10

本清单记录具身视觉搜索 Agent 研究代码的上游来源、固定提交、许可证和复用边界。第三方完整源码位于 `research/codebases/*/source/`，由 `.gitignore` 排除，不进入主项目 Git 历史。主项目只允许提交本清单、原创适配代码、测试和必要的许可证说明。

## 新下载代码库

| ID | 上游仓库 | 固定提交 | 许可证 | 本地路径 | 使用边界 |
|---|---|---|---|---|---|
| `vlfm` | `https://github.com/rai-opensource/vlfm.git` | `584ed56008754fde7997d904983607def8328322` | MIT | `research/codebases/vlfm/source/` | 优先研究 frontier value map、obstacle map、目标价值更新和探索策略；不可直接引入 Habitat 运行主链路 |
| `vlmaps` | `https://github.com/vlmaps/vlmaps.git` | `58060f97239074338ab419a2090d43fa752d724d` | MIT | `research/codebases/vlmaps/source/` | 研究开放词汇语义地图、空间语言查询和 object-goal navigation；需改写为 AI2-THOR 坐标与深度接口 |
| `concept_graphs` | `https://github.com/concept-graphs/concept-graphs.git` | `93277a02bd89171f8121e84203121cf7af9ebb5d` | MIT | `research/codebases/concept_graphs/source/` | 优先研究对象节点合并、关系图和 AI2-THOR 数据生成接口；不得直接搬入完整 SLAM/检测依赖 |
| `arigraph` | `https://github.com/AIRI-Institute/AriGraph.git` | `e884b76d7fa5185a3a8a55e5a67393b5a43f5ef2` | MIT | `research/codebases/arigraph/source/` | 研究 episodic + semantic knowledge graph、经验检索和图更新；TextWorld 适配器不能直接复用 |
| `progprompt_vh` | `https://github.com/NVlabs/progprompt-vh.git` | `56e65510747dff809c1b0bac9318508da9d9a2d4` | NVIDIA License | `research/codebases/progprompt_vh/source/` | 仅研究程序化任务计划、断言和恢复结构；复制代码前必须单独确认 NVIDIA 许可证兼容性 |
| `l3mvn` | `https://github.com/ybgdgh/L3MVN.git` | `204250c26060f32e3fb4a3dbba196d2e97fcfc82` | 未发现根许可证 | `research/codebases/l3mvn/source/` | 仅研究 LLM object-room prior 和 frontier ranking；禁止复制源码、权重、TensorBoard 日志或训练产物 |

## 已有论文代码

以下代码已经存在于 `research/papers/code/`，不重复下载：

- `llm_planner`：高层任务分解、动态重规划、ALFRED 谓词式完成验证。
- `voyager`：自动课程、critic、失败重试、技能库和向量检索。
- `react`：Observation/Action 交替执行协议，仅作交互结构参考。
- `reflexion`：失败后的语言反思和 episodic memory。
- `saycan`：语言相关性与环境可执行性联合打分。
- `embodied_reasoner`：视觉搜索、空间推理、规划与验证数据流程。

`research/papers/code/` 中多数内容来自分支 ZIP，缺少 `.git` 元数据。使用前必须回溯 manifest 中的官方仓库，并记录实际适配所依据的 commit；不能把浮动分支内容描述成可复现固定版本。

## 第一批推荐复用模块

1. **任务计划与重规划**
   - 借鉴 LLM-Planner 和 ProgPrompt。
   - 在当前项目实现结构化 `TaskPlan -> Subgoal[]`，每个子目标必须包含动作约束、成功谓词、失败谓词和恢复策略。

2. **空间探索与目标搜索**
   - 借鉴 VLFM 的 frontier value map。
   - 使用 AI2-THOR reachable positions、相机位姿、深度和实例分割重建轻量地图，不引入 Habitat 环境封装。

3. **空间与对象记忆**
   - 借鉴 VLMaps 和 ConceptGraphs。
   - 维护对象节点、观测位置、置信度、最后观测时间、关系和负证据；对象 ID 以 AI2-THOR metadata 为事实来源。

4. **经验和技能记忆**
   - 借鉴 AriGraph、Voyager 和 Reflexion。
   - episodic memory 只提供检索上下文和重规划建议，不能覆盖 simulator verifier。

5. **任务完成判定**
   - 借鉴 LLM-Planner 的谓词验证和 SayCan 的 affordance grounding。
   - VLM 可以提出 `Done` 候选，但最终完成必须由 AI2-THOR metadata、动作后置条件、距离、inventory、物体状态和 agent posture 联合确认。

## 当前任务的完成定义

对于“找到房间里的沙发并坐下”：

1. VLM 生成 `locate_sofa -> approach_sofa -> crouch_near_sofa -> verify_posture`。
2. 视觉与 metadata 联合确认沙发对象。
3. reachable map 和对象距离确认 Agent 已接近沙发。
4. AI2-THOR 成功执行 `Crouch`。
5. metadata 确认 `agent.isStanding == false`。
6. 结果必须标记为 `approximate_sit`，因为 AI2-THOR iTHOR 没有原生 `SitOnObject` 或“坐在指定家具上”的精确状态。

仅“看见沙发”、仅输出 `INSPECT`、仅由 VLM 输出 `STOP`，均不构成任务完成。

## 禁止事项

- 不把第三方完整源码、模型权重、日志、数据集或 Unity runtime 加入主仓库。
- 不直接复制无许可证的 L3MVN 代码。
- 不把第三方复现仓库当作 PaLM-E、RT-2、SIMA 的官方实现。
- 不让自然语言反思或隐藏思维链代替环境状态验证。
- 不在没有回归测试和真实 AI2-THOR 验证的情况下更换现有主链路。
