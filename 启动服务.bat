@echo off
chcp 65001 >nul
echo ====================================
echo 启动具身视觉搜索智能体 Web 服务
echo ====================================
echo.
echo 正在启动服务器...
python -m src.ui.app
pause
