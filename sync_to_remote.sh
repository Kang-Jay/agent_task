#!/bin/bash
# 直接同步本地修改到远程服务器的脚本

LOCAL_PATH="D:/cache/SummerCap/kaohe/zju"
REMOTE_HOST="3090GPU2"
REMOTE_PATH="/home/scale/kangjay/kaohe"

echo "=========================================="
echo "同步本地修改到远程 3090GPU2"
echo "=========================================="
echo ""
echo "本地路径: $LOCAL_PATH"
echo "远程路径: $REMOTE_HOST:$REMOTE_PATH"
echo ""

# 需要同步的文件列表
FILES=(
    "src/agent/controller.py"
    "src/types/schema.py"
    "src/simulation/room_simulator.py"
    "src/simulation/ai2thor_adapter.py"
    "src/ui/static/index.html"
    "test_improvements.py"
    "deploy_and_run.sh"
    "DEPLOY.md"
    "README_改进说明.md"
    "改进完成清单.md"
)

echo "=========================================="
echo "开始同步文件..."
echo "=========================================="
echo ""

for file in "${FILES[@]}"; do
    echo "📤 同步: $file"
    scp "$LOCAL_PATH/$file" "$REMOTE_HOST:$REMOTE_PATH/$file" 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✅ 成功"
    else
        echo "   ❌ 失败"
    fi
    echo ""
done

echo "=========================================="
echo "✅ 同步完成！"
echo "=========================================="
echo ""
echo "现在可以在远程服务器上运行："
echo "  ssh $REMOTE_HOST"
echo "  cd $REMOTE_PATH"
echo "  python test_improvements.py"
echo "  python -m src.ui.app"
echo ""
