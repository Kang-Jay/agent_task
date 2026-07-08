#!/bin/bash
# 一键部署到远端 3090GPU2 的快捷脚本

echo "=========================================="
echo "部署到远端 3090GPU2"
echo "=========================================="
echo ""

REMOTE_HOST="3090GPU2"
REMOTE_PATH="/home/scale/kangjay/kaohe/zju"

echo "📡 正在连接到 $REMOTE_HOST..."
echo ""

ssh $REMOTE_HOST << 'ENDSSH'
cd /home/scale/kangjay/kaohe/zju

echo "=========================================="
echo "📥 拉取最新代码..."
echo "=========================================="
git pull origin main

echo ""
echo "=========================================="
echo "📦 检查依赖..."
echo "=========================================="
pip install -r requirements.txt -q

echo ""
echo "=========================================="
echo "🧪 运行功能测试..."
echo "=========================================="
python test_improvements.py

echo ""
echo "=========================================="
echo "🚀 启动 Web 服务..."
echo "=========================================="
echo "服务将在后台运行..."
echo "访问地址: http://127.0.0.1:8000"
echo ""

# 停止旧进程
pkill -f "src.ui.app" 2>/dev/null

# 启动服务
nohup python -m src.ui.app > app.log 2>&1 &
echo "✅ 服务已启动！"
echo ""
echo "查看日志: tail -f /home/scale/kangjay/kaohe/zju/app.log"
ENDSSH

echo ""
echo "=========================================="
echo "🔗 建立 SSH 端口转发..."
echo "=========================================="
echo "本地端口: 18000 -> 远端端口: 8000"
echo ""
echo "在浏览器中打开: http://localhost:18000"
echo ""
echo "按 Ctrl+C 停止端口转发"
echo "=========================================="
echo ""

ssh -N -L 18000:127.0.0.1:8000 $REMOTE_HOST
