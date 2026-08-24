# -*- coding: utf-8 -*-
"""
FastAPI 入口 —— 塔科夫市场查询助手

接口：
  GET  /           聊天页面（前端界面）
  GET  /api/health 健康检查
  POST /api/chat   对话（输入 message，返回最终回复 + 完整思考步骤）
  POST /api/reset  重置会话记忆
"""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent import ReActAgent
from tarkov_client import tarkov
from price_store import price_store
from knowledge_base import kb
import tools  # 用 tools.last_trend_chart 读取（模块属性访问，避免值拷贝陷阱）

# 关闭 httpx 的噪音日志
logging.getLogger("httpx").setLevel(logging.WARNING)

# 全局 Agent 实例（单用户场景，多轮对话共享记忆）
_agent = ReActAgent()
MAX_STEPS = 8  # 最大工具调用步数，防止死循环


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时预加载数据 + 启动价格管道 + 后台构建知识库"""
    try:
        tarkov.load()
        print(f"[启动] 已加载 {len(tarkov.items)} 个物品数据")
    except Exception as e:
        print(f"[警告] 预加载数据失败（将在首次查询时重试）: {e}")

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
    import threading
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


@app.get("/")
def index():
    """返回聊天页面（前端界面）"""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "塔科夫市场查询助手", "version": "0.1.0"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    对话主入口：执行 ReAct 循环直到 Agent 给出最终回复。
    返回 reply（最终回复）和 steps（思考/工具调用过程，便于前端展示 Agent 轨迹）。
    """
    message = req.message.strip()
    if not message:
        return {"reply": "请输入你想问的问题", "steps": []}

    # 上一轮已结束的话，恢复 thinking 状态（保留历史记忆，支持多轮对话）
    if _agent.state == "done":
        _agent.state = "thinking"

    steps = []      # 记录每一步（thought + observation），用于展示
    reply = ""
    try:
        # 第一步带用户输入，后续步骤靠 scratchpad 记忆
        result = _agent.step(message)
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
                result = _agent.step()  # 继续下一步（不带输入）
                continue
            if result["type"] == "done":
                reply = result["reply"]
                steps.append({"thought": result.get("thought", ""), "tool": "", "observation": ""})
                break
        else:
            # 超过最大步数仍未结束，强制收尾
            reply = "抱歉，这个问题涉及的信息有点多，我一时没查完，换个更具体的问题试试？"
    except Exception as e:
        # 兜底，绝不让用户看到白屏
        reply = f"服务异常：{str(e)[:200]}"

    return {"reply": reply, "steps": steps}


def _run_agent_stream(message: str):
    """流式执行 ReAct 循环，逐步 yield SSE 数据（供前端实时展示工具调用轨迹）"""
    if _agent.state == "done":
        _agent.state = "thinking"
    try:
        result = _agent.step(message)
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
                result = _agent.step()
                continue
            if result["type"] == "done":
                yield f'data: {json.dumps({"type": "done", "reply": result["reply"]}, ensure_ascii=False)}\n\n'
                return
        else:
            # 超过最大步数仍未结束，强制收尾
            reply = "抱歉，这个问题涉及的信息有点多，我一时没查完，换个更具体的问题试试？"
            yield f'data: {json.dumps({"type": "done", "reply": reply}, ensure_ascii=False)}\n\n'
    except Exception as e:
        yield f'data: {json.dumps({"type": "error", "content": str(e)[:200]}, ensure_ascii=False)}\n\n'


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """流式对话接口：SSE 逐步返回 Agent 每一步（前端实时显示进度）"""
    message = req.message.strip()
    if not message:
        return {"reply": "请输入你想问的问题", "steps": []}
    return StreamingResponse(_run_agent_stream(message), media_type="text/event-stream")


@app.post("/api/reset")
def reset():
    """重置会话，清空多轮记忆"""
    _agent.reset()
    return {"status": "ok", "message": "会话已重置"}


@app.get("/api/price/trend/{item_id}")
def price_trend(item_id: str, days: int = 7):
    """获取某物品最近 N 天价格趋势（供前端折线图直接调用）"""
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 7
    days = max(1, min(days, 365))  # 限制在 1~365 天，防止异常值

    points = price_store.get_trend(item_id, days)
    if len(points) < 2:
        price_store.import_history(item_id)
        points = price_store.get_trend(item_id, days)

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
    """任务列表/搜索（供任务栏使用）"""
    try:
        limit = max(1, min(int(limit), 500))
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
