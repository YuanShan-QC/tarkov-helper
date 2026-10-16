# -*- coding: utf-8 -*-
"""
塔科夫市场查询助手 —— 后端启动入口

为什么需要这个文件（而不是直接用 `python -m uvicorn main:app`）：
    Windows 上 asyncio 默认用 ProactorEventLoop。uvicorn 只有在开启了
    use_subprocess（--reload / --workers）时才会切换到 SelectorEventLoop
    （见 uvicorn/loops/asyncio.py: if sys.platform == "win32" and use_subprocess），
    单进程启动时保持 Proactor，accept() 会在第一个连接上抛
        OSError: [WinError 10014] 系统检测到在一个调用中尝试使用指针参数时的无效指针地址。
    监听套接字就此失效，浏览器表现为「无法访问此网站 / 一直转圈」。

    这里在创建事件循环之前先把策略切成 WindowsSelectorEventLoopPolicy，
    从根上避开 Proactor 的 accept 缺陷。

启动：
    cd backend
    .venv\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8100
    （等价于本文件：python run.py）
"""
import sys

# ===== 必须在 import uvicorn / 创建事件循环之前执行 =====
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn  # noqa: E402  （必须在设置事件循环策略之后导入）

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8100,
        loop="asyncio",      # 使用标准 asyncio 循环（策略已在上方设为 Selector）
        log_level="info",
    )
