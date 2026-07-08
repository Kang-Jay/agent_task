#!/bin/bash
# 一键远程部署和启动脚本
# 本脚本在本地执行，会自动 SSH 到远端并启动服务

REMOTE_HOST="3090GPU2"
REMOTE_PATH="/home/scale/kangjay/kaohe/zju"
LOCAL_PORT="18000"
REMOTE_PORT="8000"

echo "=========================================="
echo "远程部署和启动服务到 3090GPU2"
echo "=========================================="
echo ""

# 1. 检查 SSH 连接
echo "🔍 检查 SSH 连接..."
ssh -o ConnectTimeout=5 $REMOTE_HOST "echo '✅ SSH 连接成功'" || {
    echo "❌ 无法连接到 $REMOTE_HOST"
    echo "请检查："
    echo "  1. SSH 配置是否正确"
    echo "  2. 服务器是否在线"
    echo "  3. 网络连接是否正常"
    exit 1
}
echo ""

# 2. 在远端拉取代码并运行
echo "📥 在远端拉取最新代码并启动服务..."
ssh $REMOTE_HOST "cd $REMOTE_PATH && bash deploy_and_run.sh" &
SSH_PID=$!
echo "远端服务进程 PID: $SSH_PID"
echo ""

# 等待服务启动
echo "⏳ 等待服务启动（10秒）..."
sleep 10

# 3. 建立 SSH 端口转发
echo "🔗 建立 SSH 端口转发..."
echo "本地端口: $LOCAL_PORT -> 远端端口: $REMOTE_PORT"
ssh -N -L $LOCAL_PORT:127.0.0.1:$REMOTE_PORT $REMOTE_HOST &
TUNNEL_PID=$!
echo "SSH 隧道进程 PID: $TUNNEL_PID"
echo ""

# 4. 完成
echo "=========================================="
echo "✅ 部署完成！"
echo "=========================================="
echo ""
echo "🌐 在浏览器中打开："
echo "   http://localhost:$LOCAL_PORT"
echo ""
echo "📊 查看远端日志："
echo "   ssh $REMOTE_HOST 'cd $REMOTE_PATH && tail -f app.log'"
echo ""
echo "🛑 停止服务："
echo "   kill $SSH_PID $TUNNEL_PID"
echo "   ssh $REMOTE_HOST 'pkill -f src.ui.app'"
echo ""
echo "按 Ctrl+C 停止本地隧道（远端服务会继续运行）"
echo ""

# 保持隧道打开
wait $TUNNEL_PID
