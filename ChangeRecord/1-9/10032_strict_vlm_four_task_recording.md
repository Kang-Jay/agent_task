# 10032 四任务 Strict VLM 录像

## 目标

对以下任务生成真实 AI2-THOR 视频，并保证整条规划链路不使用规则或 simulator oracle fallback：

1. `television` / `FloorPlan211`
2. `vase_into_box` / `FloorPlan203`
3. `right_door_exit` / `FloorPlan402`
4. `sofa_sit` / `FloorPlan211`

## Strict VLM 约束

- 最终 demo runner 使用 `AI2ThorVisualSearchDemo(strict_vlm=True)`。
- 每一步必须满足 `planner_source == model_planner`。
- 每一步 `fallback_reason` 必须为空。
- 每一步必须记录真实 provider/model、`status == ok` 和 `vision_input_used == true`。
- 模型不可用、API 失败、JSON 解析失败、非法动作、交互绑定失败、verifier 连续拒绝终止动作时，任务明确失败，不执行替代规则动作。
- Strict 模式不使用 `simulator_oracle` approach，不使用实例分割改写 VLM 搜索动作。AI2-THOR metadata 仅用于动作合法性、环境反馈和 verifier 完成证据。
- VLM 图像严格按配置的 `448x448` 输入，不再直接发送 Unity `960x540` 原图。
- Strict thinking 请求使用一次 180 秒 deadline、SDK retry 为 0，避免 90 秒请求被 SDK 三次重试放大到 270 秒。
- 真实 `vase_into_box` 审计显示 2048 completion tokens 中 2047 为 reasoning，`finish_reason=length` 且最终 JSON 为空；因此仅 Strict 模式将 `max_tokens` 提升到 4096，普通模式继续保持 2048，模型与温度不变。

## 验证顺序

1. 本地运行模型规划、最终 runner、post-action rendering 和 runtime 单元测试。
2. 仅同步本轮 Strict VLM 相关代码到 3090GPU2。
3. 远端再次运行相同测试。
4. 四任务顺序执行，避免 Unity/Xvfb/GPU 资源争抢。
5. 每个任务检查 summary、逐步 planner audit、verifier 终态和 MP4 可解码性。
6. 将完整 run 目录复制回 `docs/ai2thor_outputs/final_demo_validation_<RUN_ID>`，不提交视频、帧和 API key。

## 本地测试

- `test_model_planner.py`: 34 passed
- `test_final_agent_demo_runner.py`: 12 passed
- `test_ai2thor_post_action_rendering.py`: 5 passed
- `test_ai2thor_runtime.py`: 11 passed

`test_ai2thor_approach.py` 当前有一个由并行修改引入的独立失败：微小路径拐点期望 `MOVE_FORWARD`，实际返回 `TURN_LEFT`。本轮 Strict VLM 不调用 simulator-oracle approach，因此该失败不作为四任务严格 VLM 录像的放行依据；远端基线测试仍需单独核验。
