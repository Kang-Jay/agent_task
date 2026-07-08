#!/bin/bash
# 适配远程 3090GPU2 环境的启动脚本

cd /home/scale/kangjay/kaohe

echo "=========================================="
echo "🚀 在远端 3090GPU2 启动服务"
echo "=========================================="
echo ""

# 激活 conda base 环境
source ~/miniconda3/bin/activate base 2>/dev/null || source ~/anaconda3/bin/activate base 2>/dev/null || echo "使用系统 Python"

echo "🐍 Python 版本："
python --version || python3 --version

echo ""
echo "📦 检查关键依赖..."
python -c "import fastapi, uvicorn, PIL, numpy; print('✅ 依赖已安装')" 2>/dev/null || \
python3 -c "import fastapi, uvicorn, PIL, numpy; print('✅ 依赖已安装')" 2>/dev/null || \
echo "⚠️  部分依赖缺失，但仍尝试启动..."

echo ""
echo "🧪 快速测试（可选，按 Ctrl+C 跳过）..."
timeout 10 python test_improvements.py 2>/dev/null || timeout 10 python3 test_improvements.py 2>/dev/null || echo "跳过测试"

echo ""
echo "=========================================="
echo "🚀 启动 Web 服务"
echo "=========================================="
echo "服务地址: http://127.0.0.1:8000"
echo "按 Ctrl+C 停止服务"
echo ""

# 尝试使用 python 或 python3
python -m src.ui.app 2>/dev/null || python3 -m src.ui.app
