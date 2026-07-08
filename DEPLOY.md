# 远程部署和运行指南

## 📋 在远端 3090GPU2 上操作

### 方法 1: 使用部署脚本（推荐）

```bash
# SSH 登录到远端服务器
ssh 3090GPU2

# 进入项目目录
cd /path/to/kaohe/zju

# 拉取最新代码并运行
bash deploy_and_run.sh
```

### 方法 2: 手动步骤

```bash
# 1. SSH 登录
ssh 3090GPU2

# 2. 进入项目目录
cd /path/to/kaohe/zju

# 3. 拉取最新代码
git pull origin main

# 4. 安装依赖（如果有新依赖）
pip install -r requirements.txt

# 5. 运行测试（可选）
python test_improvements.py

# 6. 启动服务
python -m src.ui.app
```

---

## 🌐 从本地访问远程服务

服务在远端运行后，使用 SSH 端口转发从本地访问：

```bash
# 在本地 Windows 打开新的终端
ssh -N -L 18000:127.0.0.1:8000 3090GPU2
```

然后在本地浏览器打开：
```
http://localhost:18000
```

---

## 🔍 验证部署

### 检查服务状态
```bash
# 在远端服务器上
curl http://127.0.0.1:8000/api/agent/audit
```

应该返回类似：
```json
{
  "config_path": "...",
  "pipeline": [...],
  "status": "ok",
  "model_adapter": {
    "available": true,
    "providers": [...]
  }
}
```

### 测试新功能
```bash
# 在远端运行测试
python test_improvements.py
```

预期输出：
```
=== 测试 1: 中文输出 ===
✓ 中文输出测试通过！

=== 测试 2: 结构化思考 ===
✓ 结构化思考测试通过！

=== 测试 3: 点击交互 ===
✓ 点击交互测试通过！

=== 测试 4: 完整演示流程 ===
✓ 完整演示测试通过！

✅ 所有测试通过！
```

---

## 🐛 常见问题

### 问题 1: Git pull 冲突
```bash
# 如果远端有修改导致冲突
git stash          # 暂存本地修改
git pull           # 拉取最新代码
git stash pop      # 恢复本地修改（可能需要手动解决冲突）
```

### 问题 2: 端口被占用
```bash
# 检查端口 8000 是否被占用
lsof -i :8000

# 杀掉占用进程
kill -9 <PID>

# 或者修改端口
python -m src.ui.app --port 8001
```

### 问题 3: 依赖问题
```bash
# 重新安装所有依赖
pip install -r requirements.txt --force-reinstall

# 或使用虚拟环境
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 问题 4: AI2-THOR 无法运行
这是正常的，系统会自动回退到本地模拟器：
```python
# 在 UI 中选择 "本地 FloorPlan211 兼容演示"
```

---

## 📊 监控服务

### 查看日志
```bash
# 在后台运行并记录日志
nohup python -m src.ui.app > app.log 2>&1 &

# 实时查看日志
tail -f app.log
```

### 停止服务
```bash
# 查找进程
ps aux | grep "src.ui.app"

# 停止进程
pkill -f "src.ui.app"

# 或使用 Ctrl+C（如果是前台运行）
```

---

## 🎯 验证改进功能

访问 http://localhost:18000 后：

1. ✅ **验证中文输出**
   - 点击"▶ 运行演示"
   - 查看"智能体决策"面板是否显示中文

2. ✅ **验证结构化思考**
   - 查看是否有"👁 视觉观察"和"🧠 推理过程"两个独立区域

3. ✅ **验证点击交互**
   - 点击左侧机器人视角图像
   - 查看是否出现圆圈标记
   - 状态栏应显示"已选择目标点：(x, y)"

4. ✅ **验证 UI 美化**
   - 置信度是否有颜色编码（绿/黄/红）
   - 信息指标是否显示 4 个（后端/场景/步骤/模式）

---

## 📝 部署清单

- [x] 本地代码已推送到 GitHub
- [ ] SSH 登录到 3090GPU2
- [ ] 进入项目目录
- [ ] git pull 拉取最新代码
- [ ] 运行测试验证功能
- [ ] 启动服务
- [ ] 本地 SSH 端口转发
- [ ] 浏览器验证所有功能

---

## 🔗 相关链接

- GitHub 仓库: https://github.com/Kang-Jay/agent_task.git
- 本地访问（通过 SSH 隧道）: http://localhost:18000
- 远端直接访问: http://127.0.0.1:8000

---

## 💡 提示

1. 确保远端服务器有 GPU 访问权限（用于 AI2-THOR）
2. 如果 AI2-THOR 无法运行，系统会自动使用本地模拟器
3. API key 已经在 `apikey.txt` 中配置好
4. 视频录制功能需要 opencv-python 和 ffmpeg

---

## 🎉 完成！

如果所有步骤都成功，你应该可以：
- 从本地浏览器访问远端运行的服务
- 看到完整的中文界面
- 测试点击交互功能
- 查看结构化的思考输出
