# 🚀 快速开始指南

## 当前状态

✅ **Phase 1 & 2 已完成** - Schema 冻结 + 模型规划器集成
⏳ **Phase 3-5 待完成** - 点选交互、AI2-THOR 同步、数据准备

## 立即验证

### 在远程服务器 (3090GPU2) 上验证

```bash
# 1. SSH 登录
ssh 3090GPU2

# 2. 进入项目目录
cd /home/scale/kangjay/kaohe

# 3. 同步最新代码（如果本地已推送到 GitHub）
git pull origin main

# 或者从本地直接同步修改的文件
scp D:/cache/SummerCap/kaohe/zju/src/types/schema.py 3090GPU2:/home/scale/kangjay/kaohe/src/types/schema.py
scp D:/cache/SummerCap/kaohe/zju/src/agent/model_adapter.py 3090GPU2:/home/scale/kangjay/kaohe/src/agent/model_adapter.py
scp D:/cache/SummerCap/kaohe/zju/src/agent/controller.py 3090GPU2:/home/scale/kangjay/kaohe/src/agent/controller.py
scp D:/cache/SummerCap/kaohe/zju/tests/test_schema.py 3090GPU2:/home/scale/kangjay/kaohe/tests/test_schema.py
scp D:/cache/SummerCap/kaohe/zju/tests/test_model_planner.py 3090GPU2:/home/scale/kangjay/kaohe/tests/test_model_planner.py
scp D:/cache/SummerCap/kaohe/zju/validate_phase1_phase2.py 3090GPU2:/home/scale/kangjay/kaohe/validate_phase1_phase2.py

# 4. 运行验证脚本
python validate_phase1_phase2.py

# 5. 运行测试
python -B -m unittest tests.test_schema -v
python -B -m unittest tests.test_model_planner -v
python -B -m unittest tests.test_agent -v
```

### 预期输出

```
============================================================
Phase 1 & 2 Validation Script
============================================================

Test 1: Checking imports...
✅ All imports successful

Test 2: Validating SkillCall structure...
✅ SkillCall structure valid

Test 3: Validating AgentResponse new fields...
✅ AgentResponse new fields valid

...

============================================================
✅ ALL VALIDATION TESTS PASSED
============================================================
```

## 提交代码

### 1. 检查修改

```bash
cd D:\cache\SummerCap\kaohe\zju
git status --short
git diff src/types/schema.py
git diff src/agent/model_adapter.py
git diff src/agent/controller.py
```

### 2. 提交 Phase 1 & 2

```bash
# 添加核心修改
git add src/types/schema.py
git add src/agent/model_adapter.py
git add src/agent/controller.py

# 添加测试
git add tests/test_schema.py
git add tests/test_model_planner.py

# 添加文档
git add ChangeRecord/1-9/10008_phase1_phase2_schema_model_planner.md
git add PROGRESS_REPORT.md
git add COMPLETION_SUMMARY.md
git add validate_phase1_phase2.py

# 提交
git commit -m "feat: Phase 1 & 2 - Schema freeze and model planner integration

According to Plan_1_agent_demo_repair.md:

Phase 1 - Schema Freeze:
- Add SkillCall dataclass to src/types/schema.py
- Add skill_call and planner_source fields to AgentResponse
- All fields properly serialized
- Tests: tests/test_schema.py (8 cases)

Phase 2 - Model Planner Integration:
- Add ModelAdapter.plan_action() with env var support
- Integrate model planner into controller main loop
- Implement fallback with explicit labeling
- Enforce stop confidence threshold
- Tests: tests/test_model_planner.py (7 cases)

Key guarantees:
- No configuration thresholds modified
- Fallback never disguised as model output
- All actions validated against allowed_actions
- Stop rule enforced (confidence >= 0.78)

Validation: validate_phase1_phase2.py (10 checks)
ChangeRecord: ChangeRecord/1-9/10008_phase1_phase2_schema_model_planner.md

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"

# 推送到远程
git push origin main
```

## 使用新功能

### 检查模型规划器状态

```python
from src.agent.controller import EmbodiedSearchAgent

agent = EmbodiedSearchAgent()

# 检查模型适配器是否可用
print(agent.model_adapter.available())  # True 如果有 API key

# 审计配置
print(agent.audit())
```

### 查看 planner_source

```python
from src.types.schema import AgentRequest
from PIL import Image

# 创建测试图像
img = Image.new('RGB', (448, 448), (200, 200, 200))
img.save('test.png')

# 运行 agent
response = agent.step(AgentRequest(
    session_id="test",
    instruction="找到红色杯子",
    observation_image="test.png",
    step_id=0
))

# 查看决策来源
print(f"Planner Source: {response.planner_source}")
# 输出: "model_planner" 或 "rule_fallback"

print(f"Skill Call: {response.skill_call}")
# 输出: SkillCall(name='TURN_RIGHT', args={}, ...)
```

### 设置 API Key (可选)

```bash
# 方法 1: 环境变量（推荐）
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"

# 方法 2: apikey.txt 文件（已在 .gitignore）
echo "sk-..." > apikey.txt

# 方法 3: 自定义 base URL
export MODEL_API_KEY="sk-..."
export MODEL_BASE_URL="https://api.your-provider.com/v1"
export MODEL_NAME="your-model"
```

## 查看文档

### 核心文档
- `COMPLETION_SUMMARY.md` - 完成总结（本次执行）
- `PROGRESS_REPORT.md` - 进度报告
- `Plan_1_agent_demo_repair.md` - 完整修复计划
- `ChangeRecord/1-9/10008_phase1_phase2_schema_model_planner.md` - 详细变更记录

### 测试和验证
- `validate_phase1_phase2.py` - 10个集成验证检查
- `tests/test_schema.py` - Schema 测试
- `tests/test_model_planner.py` - 模型规划器测试

## 下一步 (Phase 3)

### 需要修改的文件

```
src/ui/static/index.html       - 前端点击交互
src/ui/app.py                  - 后端 API 接受 clicked_point
src/simulation/room_simulator.py    - 传递 clicked_point
src/simulation/ai2thor_adapter.py   - 传递 clicked_point
```

### 启动 Phase 3

```bash
# 参考 Plan 第5节
# 或等待下一轮指令
```

## 故障排查

### 问题: 模型始终返回 rule_fallback

**原因:** 没有 API key

**解决:**
```bash
export OPENAI_API_KEY="sk-your-key"
python -m src.agent.model_adapter  # 测试
```

### 问题: 测试失败 "No module named ..."

**原因:** Python 环境或依赖缺失

**解决:**
```bash
pip install -r requirements.txt
```

### 问题: Agent 返回 ASK_CLARIFY

**原因:** 模型返回了非法动作，系统 fallback 并拦截

**说明:** 这是正常行为，检查 `fallback_reason` 字段了解详情

## 关键提醒

1. ⚠️ **不要修改** `configs/agent_config.json` 中的阈值
2. ⚠️ **不要提交** `apikey.txt` 文件
3. ⚠️ **不要删除** 测试文件
4. ✅ **始终检查** `planner_source` 字段了解决策来源
5. ✅ **始终验证** 修改后运行测试

## 联系和反馈

如有问题，参考：
- `Plan_1_agent_demo_repair.md` - 完整规范
- `ChangeRecord/1-9/10008_*.md` - 变更记录
- `COMPLETION_SUMMARY.md` - 执行总结

---

**生成时间:** 2024-07-10
**适用版本:** Phase 1 & 2 完成后
**下一阶段:** Phase 3 点选多模态集成
