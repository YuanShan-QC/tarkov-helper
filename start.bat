@echo off
setlocal

rem ============================================================
rem  塔科夫市场查询助手 - 一键启动脚本
rem  用法：双击本文件，自动准备环境并启动后端服务
rem ============================================================

rem 切换到脚本所在目录（项目根目录）
cd /d "%~dp0"

echo ================================================
echo    塔科夫市场查询助手 - 一键启动
echo ================================================
echo.

rem ----- 1. 定位 Python（优先系统 Python 3.11，其次 PATH）-----
set "PYTHON_EXE="
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
) else (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)
if not defined PYTHON_EXE (
    echo [错误] 未检测到 Python，请先安装 Python 3.9 以上版本
    echo.
    pause
    exit /b 1
)
echo [1/3] Python 环境已就绪

rem ----- 2. 检查并准备后端环境 -----
if not exist "backend\.venv\Scripts\python.exe" (
    echo [2/3] 首次运行，正在创建虚拟环境并安装依赖（约需 1 分钟）...
    "%PYTHON_EXE%" -m venv backend\.venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        echo.
        pause
        exit /b 1
    )
    backend\.venv\Scripts\python.exe -m pip install --upgrade pip -q
    backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt -q
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络后重试
        echo.
        pause
        exit /b 1
    )
    echo.
) else (
    echo [2/3] 后端环境已就绪
)

rem ----- 3. 检查配置 -----
if exist "backend\.env" (
    echo [3/3] 配置文件已就绪
) else (
    echo [3/3] [警告] 未找到 backend\.env
)

rem ----- 4. 启动后端 -----
echo.
echo 正在启动后端服务（端口 8100）...
cd /d "%~dp0backend"
start "Tarkov-Helper-Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8100"
cd /d "%~dp0"

echo.
echo ================================================
echo    启动完成！
echo.
echo    聊天页面 : http://localhost:8100/
echo    API 文档 : http://localhost:8100/docs
echo.
echo    浏览器即将自动打开聊天页面
echo    关闭「Tarkov-Helper-Backend」窗口即可停止服务
echo ================================================
echo.

timeout /t 5 /nobreak >nul
start http://localhost:8100/

echo.
echo 按任意键关闭本窗口（不影响后端运行）...
pause

endlocal
