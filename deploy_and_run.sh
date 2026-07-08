#!/bin/bash
# 在远端 3090GPU2 上拉取并运行的脚本

echo "=========================================="
echo "在远端 3090GPU2 上部署和运行"
echo "=========================================="
echo ""

# 1. 拉取最新代码
echo "📥 正在拉取最新代码..."
git pull origin main

if [ $? -ne 0 ]; then
    echo "❌ Git pull 失败"
    exit 1
fi

echo "✅ 代码拉取成功"
echo ""

# 2. 检查依赖
echo "📦 检查 Python 依赖..."
python -m pip install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "❌ 依赖安装失败"
    exit 1
fi

echo "✅ 依赖检查完成"
echo ""

# 3. 运行测试（可选）
echo "🧪 运行功能测试..."
python test_improvements.py

if [ $? -ne 0 ]; then
    echo "⚠️  测试未完全通过，但继续启动服务"
else
    echo "✅ 所有测试通过"
fi
echo ""

# 4. 启动服务
echo "🚀 启动 Web 服务..."
echo "服务将在 http://127.0.0.1:8000 运行"
echo "使用 SSH 端口转发可以从本地访问："
echo "  ssh -N -L 18000:127.0.0.1:8000 3090GPU2"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

python -m src.ui.app
