# -*- coding: utf-8 -*-
"""
FastAPI 入口 —— 塔科夫市场查询助手

接口：
  GET  /           聊天页面（前端界面）
  GET  /api/health 健康检查
  POST /api/chat   对话（输入 message，返回最终回复 + 完整思考步骤）
  POST /api/reset  重置会话记忆
"""
import asyncio
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

# ===== Windows 事件循环兜底（必须在事件循环创建前设置，import 期执行刚好）=====
# Windows 默认的 ProactorEventLoop 在本机 accept() 会抛
#   OSError: [WinError 10014] 系统检测到在一个调用中尝试使用指针参数时的无效指针地址
# 导致监听套接字失效、页面完全打不开。uvicorn 只在 use_subprocess
# （--reload/--workers）时才切 Selector，单进程启动会保留 Proactor，
# 所以这里显式切换。
#
# 但这段只在「入口先执行、循环后创建」的顺序下有效：
#   ✅ python run.py      ✅ python main.py
#   ❌ python -m uvicorn main:app —— uvicorn 先建循环再 import 本模块，
#      此处设置来不及生效，实测仍会 10014。要用 uvicorn 请走 backend/run.py。
# 设 TARKOV_DISABLE_SELECTOR_LOOP=1 可临时关闭该兜底（仅用于诊断对比）。
if sys.platform == "win32" and not os.getenv("TARKOV_DISABLE_SELECTOR_LOOP"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from agent import ReActAgent
from tarkov_client import tarkov, CATEGORY_DISPLAY_ORDER
from price_store import price_store
from knowledge_base import kb
import config  # 模式可运行时切换，必须动态读取 config.GAME_MODE
import tools  # 用 tools.last_trend_chart 读取（模块属性访问，避免值拷贝陷阱）

# 关闭 httpx 的噪音日志
logging.getLogger("httpx").setLevel(logging.WARNING)

# 会话隔离：每个 session 一个 Agent（各自独立的 scratchpad）
# 以前是模块级单例，两个标签页/两个用户同时提问会互相串题（实测 2/6 概率答非所问）。
_sessions: "OrderedDict[str, ReActAgent]" = OrderedDict()
_sessions_lock = threading.Lock()
MAX_SESSIONS = 32  # 最多保留 32 个会话，超出丢弃最久未用的，防止内存无限涨

MAX_STEPS = 8  # 单轮最多执行 8 次工具调用，防止死循环


def _get_agent(session_id: str) -> ReActAgent:
    """按 session_id 取（或新建）Agent 实例，LRU 淘汰。"""
    session_id = (session_id or "default").strip() or "default"
    with _sessions_lock:
        agent = _sessions.get(session_id)
        if agent is None:
            agent = ReActAgent()
            _sessions[session_id] = agent
            while len(_sessions) > MAX_SESSIONS:
                _sessions.popitem(last=False)
        else:
            _sessions.move_to_end(session_id)
        return agent



@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动：本地缓存秒级加载 → 后台刷新数据 → 价格管道 → 后台构建知识库"""
    print(f"[启动] 价格数据模式：{config.GAME_MODE.upper()}（{'PvE' if config.GAME_MODE == 'pve' else 'PvP'}）")
    # 物品数据：优先读本地磁盘缓存（秒级完成），无缓存才联网（首次运行）
    try:
        tarkov.load()
        print(f"[启动] 已加载 {len(tarkov.items)} 个物品数据")
    except Exception as e:
        print(f"[警告] 预加载数据失败（将在后台重试）: {e}")

    # 后台周期性刷新 tarkov.dev 数据：缓存过期才联网拉新数据，不阻塞启动。
    # 每 TARKOV_REFRESH_INTERVAL 秒检查一次（默认 30 分钟），
    # 保证长时间运行的实例数据也不会过期（物品 ≤6h、知识库 ≤24h）。
    refresh_interval = int(os.getenv("TARKOV_REFRESH_INTERVAL", "1800"))

    def _refresh_loop():
        while True:
            try:
                tarkov.maybe_refresh_all()
            except Exception as e:
                print(f"[缓存] 后台刷新异常：{e}")
            time.sleep(refresh_interval)

    threading.Thread(target=_refresh_loop, daemon=True, name="tarkov-refresh").start()

    # 启动价格管道：后台线程每小时抓一次全量价格快照
    try:
        def _fetch_prices():
            return {iid: it.get("avg24hPrice")
                    for iid, it in tarkov.items.items()
                    if it.get("avg24hPrice")}
        price_store.start_scheduler(_fetch_prices)
        print("[启动] 价格管道已启动")
    except Exception as e:
        print(f"[警告] 价格管道启动失败: {e}")

    # 后台构建知识库（不阻塞启动；首次检索时也会自动兜底构建）
    def _build_kb():
        try:
            kb.build()
            print(f"[知识库] 构建完成，共 {len(kb.docs)} 条文档")
        except Exception as e:
            print(f"[知识库] 构建失败: {e}")
    threading.Thread(target=_build_kb, daemon=True).start()

    yield


app = FastAPI(title="塔科夫市场查询助手", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"   # 会话标识：不同标签页/不同用户互不干扰


class ResetRequest(BaseModel):
    session_id: str = "default"


@app.get("/")
def index():
    """返回聊天页面（前端界面）。no-cache：每次打开都拿最新版本，避免改版后看不到更新。"""
    return FileResponse(
        Path(__file__).parent / "static" / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/vendor/{path:path}")
def vendor_files(path: str):
    """前端静态资源（本地化的 echarts 等）。长缓存：文件带版本号，内容不变则直接复用。"""
    base = (Path(__file__).parent / "static" / "vendor").resolve()
    target = (base / path).resolve()
    # 防目录穿越：目标必须在 vendor 目录内
    if target != base and base not in target.parents:
        raise HTTPException(status_code=404, detail="Not Found")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(
        target,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/favicon.svg")
def favicon_svg():
    """站点图标（浏览器标签页显示），消除 /favicon.ico 404 噪音日志"""
    return FileResponse(
        Path(__file__).parent / "static" / "favicon.svg",
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/favicon.ico")
def favicon_ico():
    """兼容仍请求 .ico 的浏览器：重定向到 SVG 图标"""
    return RedirectResponse(url="/favicon.svg", status_code=307)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "塔科夫市场查询助手", "version": "0.1.0", "mode": config.GAME_MODE}


class ModeRequest(BaseModel):
    mode: str


@app.post("/api/mode")
def set_mode(req: ModeRequest):
    """
    运行时切换价格模式（pvp / pve）：
    更新数据源 → 重载对应模式数据（优先本地缓存，无缓存才联网）→ 返回切换结果。
    页面右上角按钮调用；切换后价格、趋势、任务/配方数据均按新模式提供。
    """
    try:
        result = tarkov.set_mode(req.mode)
        return {"status": "ok", **result}
    except ValueError as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": f"切换失败：{e}"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    对话主入口：执行 ReAct 循环直到 Agent 给出最终回复。
    返回 reply（最终回复）和 steps（思考/工具调用过程，便于前端展示 Agent 轨迹）。
    """
    message = req.message.strip()
    if not message:
        return {"reply": "请输入你想问的问题", "steps": []}

    agent = _get_agent(req.session_id)
    # 同一会话串行执行：防止双击发送/两个请求交错把 scratchpad 搅乱
    with agent._turn_lock:
        # 上一轮已结束的话，恢复 thinking 状态（保留历史记忆，支持多轮对话）
        if agent.state == "done":
            agent.state = "thinking"

        steps = []      # 记录每一步（thought + observation），用于展示
        reply = ""
        try:
            # 第一步带用户输入，后续步骤靠 scratchpad 记忆
            result = agent.step(message)
            for _ in range(MAX_STEPS):
                if result["type"] == "error":
                    reply = f"出错了：{result['content']}"
                    break
                if result["type"] == "observation":
                    # 记录思考 + 工具观察
                    steps.append({
                        "thought": result.get("thought", ""),
                        "tool": result.get("tool", ""),
                        "observation": result.get("content", ""),
                    })
                    result = agent.step()  # 继续下一步（不带输入）
                    continue
                if result["type"] == "done":
                    reply = result["reply"]
                    steps.append({"thought": result.get("thought", ""), "tool": "", "observation": ""})
                    break
            else:
                # 步数用尽：让模型用已经查到的结果作答，而不是把整轮丢掉
                reply = agent.force_finalize()
        except Exception as e:
            # 兜底，绝不让用户看到白屏
            reply = f"服务异常：{str(e)[:200]}"

    return {"reply": reply, "steps": steps}


def _run_agent_stream(agent: ReActAgent, message: str):
    """流式执行 ReAct 循环，逐步 yield SSE 数据（供前端实时展示工具调用轨迹）"""
    # 整个会话上锁：生产者生成器在流式返回期间独占该 Agent
    with agent._turn_lock:
        if agent.state == "done":
            agent.state = "thinking"
        try:
            result = agent.step(message)
            for _ in range(MAX_STEPS):
                if result["type"] == "error":
                    yield f'data: {json.dumps({"type": "error", "content": result["content"]}, ensure_ascii=False)}\n\n'
                    return
                if result["type"] == "observation":
                    # 每走一步工具，就把这一步实时推给前端
                    payload = {
                        "type": "step",
                        "thought": result.get("thought", ""),
                        "tool": result.get("tool", ""),
                        "observation": result.get("content", ""),
                    }
                    yield f'data: {json.dumps(payload, ensure_ascii=False)}\n\n'
                    # 趋势查询成功后，附带推送折线图数据给前端渲染
                    if result.get("tool") == "get_price_trend" and tools.last_trend_chart:
                        chart_payload = {"type": "chart", "chart": tools.last_trend_chart}
                        yield f'data: {json.dumps(chart_payload, ensure_ascii=False)}\n\n'
                    result = agent.step()
                    continue
                if result["type"] == "done":
                    yield f'data: {json.dumps({"type": "done", "reply": result["reply"]}, ensure_ascii=False)}\n\n'
                    return
            else:
                # 步数用尽：让模型用已经查到的结果作答
                yield f'data: {json.dumps({"type": "done", "reply": agent.force_finalize()}, ensure_ascii=False)}\n\n'
        except Exception as e:
            yield f'data: {json.dumps({"type": "error", "content": str(e)[:200]}, ensure_ascii=False)}\n\n'


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """流式对话接口：SSE 逐步返回 Agent 每一步（前端实时显示进度）"""
    message = req.message.strip()
    if not message:
        return {"reply": "请输入你想问的问题", "steps": []}
    agent = _get_agent(req.session_id)
    return StreamingResponse(_run_agent_stream(agent, message), media_type="text/event-stream")


@app.post("/api/reset")
def reset(req: "ResetRequest" = None):
    """重置会话，清空多轮记忆"""
    sid = (req.session_id if req else "") or "default"
    _get_agent(sid).reset()
    return {"status": "ok", "message": "会话已重置"}


@app.get("/api/price/trend/{item_id}")
def price_trend(item_id: str, days: int = 7):
    """获取某物品最近 N 天价格趋势（供前端折线图直接调用）"""
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 7
    days = max(1, min(days, 365))  # 限制在 1~365 天，防止异常值

    # 本地没有覆盖到窗口起点的历史时，自动从 tarkov.dev 回补一次（每个物品每天最多一次）
    points = price_store.get_trend_with_history(item_id, days)

    name = tarkov.name_by_id(item_id) or item_id
    return {"item_id": item_id, "name": name, "days": days, "points": points}


# ===== 多页面展示接口（直接查数据，不经过 Agent）=====

@app.get("/api/search")
def search_items(keyword: str = "", limit: int = 20):
    """物品搜索（供价格栏使用）"""
    if not keyword.strip():
        return {"items": []}
    try:
        limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        limit = 20
    try:
        results = tarkov.search(keyword.strip(), limit=limit)
        return {"items": results}
    except Exception as e:
        return {"items": [], "error": f"搜索失败：{e}"}


@app.get("/api/items")
def all_items():
    """全部物品（按分类分组，供价格栏"全部物品"收束展示）"""
    try:
        items = tarkov.list_items()
        groups = {}
        for it in items:
            t = it["type"]
            if t not in groups:
                groups[t] = []
            groups[t].append(it)
        # 固定分类顺序，未知分类排最后
        def sort_key(t):
            return CATEGORY_DISPLAY_ORDER.index(t) if t in CATEGORY_DISPLAY_ORDER else 999
        order = sorted(groups.keys(), key=sort_key)
        return {"groups": [{"type": t, "items": groups[t]} for t in order]}
    except Exception as e:
        return {"groups": [], "error": f"加载物品失败：{e}"}


@app.get("/api/item/{item_id}")
def item_detail(item_id: str):
    """物品详情 + 当前价格（供价格栏使用）"""
    try:
        tarkov.load()
        item = tarkov.items.get(item_id)
        if not item:
            return {"error": "物品不存在"}
        type_map = {"gun": "武器", "ammo": "弹药", "armor": "护甲", "barter": "物资",
                    "wearable": "穿戴", "container": "容器", "key": "钥匙"}
        type_zh = "/".join(type_map.get(t, t) for t in item.get("types", [])) or "未知"
        return {
            "id": item_id,
            "name": tarkov.item_name(item),
            "normalizedName": item.get("normalizedName", ""),
            "type": type_zh,
            "width": item.get("width"),
            "height": item.get("height"),
            "weight": item.get("weight"),
            "basePrice": item.get("basePrice"),
            "minLevelForFlea": item.get("minLevelForFlea"),
            "price": tarkov.get_price(item_id),
        }
    except Exception as e:
        return {"error": f"获取物品失败：{e}"}


@app.get("/api/tasks")
def task_list(keyword: str = "", limit: int = 100):
    """任务列表/搜索（供任务栏使用）；keyword 为空时返回全部任务"""
    try:
        limit = max(1, min(int(limit), 1000))
    except (TypeError, ValueError):
        limit = 100
    try:
        tasks = tarkov.search_tasks(keyword, limit=limit)
        return {"tasks": tasks, "total": len(tasks)}
    except Exception as e:
        return {"tasks": [], "total": 0, "error": f"加载任务失败：{e}"}


@app.get("/api/tasks/{task_id}")
def task_detail(task_id: str):
    """任务详情（供任务栏使用）"""
    try:
        detail = tarkov.task_detail(task_id)
        if not detail:
            return {"error": "任务不存在"}
        return detail
    except Exception as e:
        return {"error": f"获取任务详情失败：{e}"}


@app.get("/api/crafts")
def crafts():
    """藏身处制造配方列表（供藏身处栏使用）"""
    try:
        return {"crafts": tarkov.craft_list()}
    except Exception as e:
        return {"crafts": [], "error": f"加载制造配方失败：{e}"}


@app.get("/api/barters")
def barters():
    """商人以物换物列表（供藏身处栏使用）"""
    try:
        return {"barters": tarkov.barter_list()}
    except Exception as e:
        return {"barters": [], "error": f"加载以物换物失败：{e}"}


@app.get("/api/hideout")
def hideout():
    """藏身处工作站列表（供藏身处栏使用）"""
    try:
        return {"stations": tarkov.hideout_list()}
    except Exception as e:
        return {"stations": [], "error": f"加载藏身处失败：{e}"}
