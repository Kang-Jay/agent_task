# 10023 Plan2 非 Oracle VLFM-lite 几何搜索纯模块

## 阶段目标

- 根据 `Plan_2_hierarchical_embodied_agent_upgrade.md` 阶段 6，补齐非 Oracle 几何搜索的基础模块。
- 建立 RGB-D occupancy mapping、frontier extraction/filtering、semantic value evidence、deterministic exploration planner 的可测试接口。
- 本阶段只完成纯模块与专项测试，不接入 UI、stream、memory、evaluation、interaction executor 或现有 AI2-THOR runtime 主循环。

## 修改文件及职责

- `src/mapping/observations.py`
  - 定义 planner-safe `CameraIntrinsics`、`CameraPose`、`RGBDObservation`。
  - 明确 RGB-D 观察包不包含 AI2-THOR object metadata、instance masks、target pose、interactable poses 或 reachable-position oracle 数据。
- `src/mapping/occupancy_grid.py`
  - 定义 `GridSpec`、`GridCellState`、`OccupancyGrid`、`MapUpdateResult`。
  - 支持 unknown/free/occupied、visited、blocked、world/grid 坐标转换和 4 邻接遍历。
- `src/mapping/depth_projector.py`
  - 用 depth、camera intrinsics、camera pose 投影 free ray 和 terminal obstacle。
  - 忽略 0、NaN、无效深度，不使用 simulator object truth。
- `src/mapping/frontier.py`
  - 从已知 free cell 与 unknown cell 的边界提取 frontier。
  - 支持 reachable cell 过滤，并按距离、未知邻居数、free 邻居数和坐标稳定排序。
- `src/exploration/frontier_policy.py`
  - 定义 `SemanticValueMap`、`rank_frontiers`、`reject_oracle_fields`。
  - 对 `objects`、`objectId`、`target_pose`、`matched_pose`、`recommended_action`、`instance_masks` 等 oracle 字段直接拒绝。
  - 最终审查发现 `target_pose`、`matched_pose`、`recommended_action` 会被 key normalization 去掉下划线，已将黑名单改为 normalization 后的 key，并补充回归测试。
  - 语义 evidence 只接受 RGB/文本候选字段，例如 label、confidence、world_x/world_z 或 bearing/range。
- `src/planning/grid_planner.py`
  - 定义 BFS/A* 风格的 deterministic grid planner。
  - 返回 `MOVE_FORWARD`、`TURN_LEFT`、`TURN_RIGHT`、`INSPECT` 等现有抽象动作。
  - 避开 occupied/blocked cell，并保持 AI2-THOR yaw 约定：0 度朝 +Z，90 度朝 +X。
- `src/planning/exploration_planner.py`
  - 串联 RGB-D map update、frontier extraction、semantic ranking、path planning、next action selection。
  - 输出 `ExplorationDecision`，包含 action、map_update、path、frontiers、ranked_frontiers、selected_frontier 和 summary。
- `src/mapping/__init__.py`、`src/exploration/__init__.py`、`src/planning/__init__.py`
  - 暴露本阶段主接口。
- `tests/test_occupancy_grid.py`
- `tests/test_depth_mapping.py`
- `tests/test_frontier.py`
- `tests/test_semantic_value_map.py`
- `tests/test_grid_planner.py`
- `tests/test_exploration_planner.py`
  - 覆盖坐标转换、occupancy update、depth projection、frontier、semantic evidence、oracle leakage、path planning 和编排层。

## 是否修改配置或超参

- 未修改 `configs/agent_config.json`。
- 未修改任何已有模型、微调、训练、动作空间、UI 或 runtime 参数。
- 新模块默认参数仅作为纯模块接口默认值：
  - `max_depth_m=5.0`
  - `sample_stride=4`
  - `turn_angle_degrees=30.0`
  - `semantic_map.decay(factor=0.92)`
- 这些值没有写入项目配置，也没有声明为最终 runtime 超参；主链路接入时必须按已有配置或正式实验记录统一收敛。

## 事实依据和边界

- 当前 AI2-THOR adapter 已有 RGB frame、agent pose、action feedback、reachable positions 和 third-party camera visualization。
- 子代理审查确认当前 runtime 尚未启用或暴露 depth，因此本阶段实现的是可接入的 RGB-D 纯模块，不声称真实 Unity 主循环已经使用 depth。
- `event.metadata["objects"]`、`event.instance_masks`、object position/distance、target pose、interactable poses 只能用于 verifier/oracle/debug，不进入本阶段 non-oracle planner。

## 测试命令

```powershell
python -B -m unittest discover -s tests -p "test_occupancy_grid.py" -v
python -B -m unittest discover -s tests -p "test_depth_mapping.py" -v
python -B -m unittest discover -s tests -p "test_frontier.py" -v
python -B -m unittest discover -s tests -p "test_semantic_value_map.py" -v
python -B -m unittest discover -s tests -p "test_grid_planner.py" -v
python -B -m unittest discover -s tests -p "test_exploration_planner.py" -v
python -B -m compileall -q src\mapping src\exploration src\planning tests\test_occupancy_grid.py tests\test_depth_mapping.py tests\test_frontier.py tests\test_semantic_value_map.py tests\test_grid_planner.py tests\test_exploration_planner.py
```

## 测试结果

- `test_occupancy_grid.py`: 3 tests OK。
- `test_depth_mapping.py`: 3 tests OK。
- `test_frontier.py`: 2 tests OK。
- `test_semantic_value_map.py`: 4 tests OK。
- `test_grid_planner.py`: 5 tests OK。
- `test_exploration_planner.py`: 4 tests OK。
- 合计：21 项专项测试通过。
- `compileall` 通过。
- 第一次运行 `python -B -m unittest tests.test_...` 失败，原因是当前 `tests` 目录不是 importable package；改用 `unittest discover` 后通过。
- `test_semantic_candidate_can_prioritize_farther_frontier` 初始失败，原因是测试把语义目标放在不可达孤岛上。已改成远但可达 corridor，保持“可达性优先、语义 evidence 可提升排序”的正确约束。
- 最终只读审查发现 oracle key normalization blocker：`target_pose`、`matched_pose`、`recommended_action` 未被正确拒绝。已修复黑名单并新增 `test_rejects_underscore_oracle_keys_after_normalization` 与 planner context 回归用例。

## 已完成能力

- 可从 planner-safe RGB-D observation 更新 occupancy grid。
- 可提取 free/unknown 边界 frontier。
- 可拒绝明显的 AI2-THOR oracle metadata 泄漏。
- 可拒绝归一化前后均危险的 oracle key，包括 `target_pose`、`matched_pose`、`recommended_action`。
- 可用 RGB/文本 candidate 形成 semantic value evidence。
- 可对 frontier 做 semantic + information gain + travel cost 排序。
- 可规划避障路径并输出下一步抽象导航动作。
- 可通过 `ExplorationPlanner.decide()` 一步串联 map update、frontier、semantic ranking、path 和 action。

## 尚未完成能力

- 尚未把 `ExplorationPlanner` 接入 `EmbodiedSearchAgent.step()` 或 `AI2ThorVisualSearchDemo.run_demo()`。
- 尚未在 `create_controller_safely(...)` 中启用 `renderDepthImage=True`。
- 尚未从真实 AI2-THOR event 读取 `depth_frame` 并转换为 `RGBDObservation`。
- 尚未实现 controller payload 中的 `navigation_state`。
- 尚未替换 `_rule_fallback_planner()` 的固定 turn/forward 策略。
- 尚未用真实 Unity episode 验证非 Oracle frontier 行为。
- 尚未完成 “找到右边的门，然后走出去” 与 “把花瓶放到纸箱里” 两个任务执行；这两个任务需要在本模块接入 runtime、interaction executor、postcondition verifier 后再做真实验证。

## 主线集成点

1. 在 AI2-THOR controller 启动参数中启用 depth 渲染，并以独立、planner-safe reader 暴露 RGB-D、pose、intrinsics。
2. 在 agent/session 层为每个 session 持有 `OccupancyGrid` 和 `ExplorationPlanner`，不要复用当前 3x3 image-region search map 充当 metric map。
3. 在 `EmbodiedSearchAgent.step()` 的 fallback/model payload 中加入 sanitized `navigation_state`，仅包含 grid summary、frontier summary、selected frontier、path summary 和 blocked/attempted frontier 统计。
4. 在视觉目标未达到阈值时，用 `ExplorationPlanner.decide()` 的 action 替换固定 `_rule_fallback_planner()` turn template。
5. 在 runtime 执行失败或碰撞后，把失败 forward cell 写入 blocked，再进行下一轮 frontier selection。
6. 真实 AI2-THOR 验证必须分别标记 `planner_source="non_oracle_frontier"`、`model_planner`、`simulator_oracle`，不能混淆。

## 已知限制

- 本阶段是基础几何模块，不是完整 Plan2 完成态。
- semantic evidence 当前使用候选字典接口，尚未绑定现有 `HeuristicVision.Candidate` 或 VLM detection 输出。
- `world_x/world_z` 在测试中用于非 oracle candidate 位置表达；真实接入时必须由 RGB-D projection 或明确 planner-safe detection 输出提供，不能由 AI2-THOR object metadata 填充。
- 当前 frontier extraction 为 cell-level，不含 connected component medoid、clearance dilation 和 repeated component ID；这些是下一步增强项。

## 代码库清洁

- 已清理 touched area 下的 `__pycache__`。
- 未提交视频、缓存、密钥或临时运行目录。
- 未修改已有配置文件。
- 当前工作区仍存在协作者已有修改和 untracked 文件，本阶段没有回退或覆盖这些内容。

## 下一阶段进入条件

- 本阶段 scoped tests 必须继续通过。
- 先实现 AI2-THOR RGB-D observation reader，并新增 depth snapshot 测试。
- 再将 `ExplorationPlanner` 接入 agent fallback 路径。
- 接入后必须跑真实 AI2-THOR 非 Oracle episode，并确认 planner 没有读取 object metadata、instance masks 或 target pose。
