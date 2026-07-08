@echo off
chcp 65001 >nul
echo ========================================
echo 远程部署到 3090GPU2 并启动服务
echo ========================================
echo.
echo 第1步: SSH 登录到远程服务器并拉取代码
echo.
echo 请在远程服务器上执行以下命令：
echo.
echo cd /path/to/kaohe/zju
echo bash deploy_and_run.sh
echo.
echo ========================================
echo 第2步: 在本地建立 SSH 隧道
echo ========================================
echo.
echo 在新的终端窗口执行：
echo ssh -N -L 18000:127.0.0.1:8000 3090GPU2
echo.
echo 然后在浏览器打开: http://localhost:18000
echo.
pause
