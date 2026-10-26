# -*- coding: utf-8 -*-
"""
Agent 工具集 —— 塔科夫查询助手可用的工具

工具清单（第一层 MVP）：
  search_item(keyword)        搜索物品，返回候选列表
  get_price(item_name)        查物品当前价格（24h 均价 / 最低价 / 48h 涨跌）
  get_price_trend(item_name, days)  查价格历史趋势
  get_item_info(item_name)    查物品详情（类型 / 格子 / 基础价）
  compare_sell(item_name)     比价：跳蚤市场 vs 卖给商人哪个划算

每个工具接收 args 字典，返回字符串（作为 LLM 的 OBSERVATION）。
"""
import datetime
from tarkov_client import tarkov
from alias import resolve_alias
from price_store import price_store
from knowledge_base import kb

# 模块级变量：记录最近一次趋势查询的图表数据，供 main.py 生成 chart SSE 事件
last_trend_chart = None  # {"item_id", "name", "days", "points": [{ts, price}]}


def _find_item(name: str):
    """
    根据用户给的物品名（可能是黑话）定位物品。
    返回 dict：唯一命中 → {"id", "name"}；多个候选 → {"candidates": [...]}；没找到 → None
    """
    keyword = resolve_alias(name)
    candidates = tarkov.search(keyword, limit=5)
    if not candidates:
        return None
    # 精确命中（完整名/短名与关键词完全一致）直接采用，
    # 避免「实体比特币」被「实体比特币的右半边」等干扰而反复让用户确认
    exact = [c for c in candidates if c.get("name") == keyword or c.get("shortName") == keyword]
    if exact:
        c = exact[0]
        return {"id": c["id"], "name": c["name"]}
    if len(candidates) == 1:
        c = candidates[0]
        return {"id": c["id"], "name": c["name"]}
    return {"candidates": candidates}


def _candidates_text(candidates: list) -> str:
    """把候选列表格式化成文本"""
    lines = [f"{i + 1}. {c['name']}（英文: {c['normalizedName']}）" for i, c in enumerate(candidates)]
    prefix = "找到多个候选物品，需要确认具体是哪个：" if len(candidates) > 1 else "找到该物品："
    return prefix + "\n" + "\n".join(lines)


def tool_search_item(args: dict) -> str:
    """搜索物品"""
    keyword = args.get("keyword", "").strip()
    if not keyword:
        return "请提供要搜索的物品关键词"
    candidates = tarkov.search(keyword, limit=5)
    if not candidates:
        return f"没找到和「{keyword}」相关的物品，试试其他叫法（比如英文名或简称）"
    return _candidates_text(candidates)


def tool_get_price(args: dict) -> str:
    """查物品当前价格"""
    name = args.get("item_name", "").strip()
    found = _find_item(name)
    if not found:
        return f"没找到「{name}」这个物品"
    if "candidates" in found:
        return _candidates_text(found["candidates"])

    p = tarkov.get_price(found["id"])
    if p is None:
        return f"无法获取「{found['name']}」的价格信息"
    if p["noFlea"]:
        return (f"{p['name']} 禁止在跳蚤市场交易，无市场价格。"
                f"基础参考价 {p['basePrice']} 卢布。")
    return (f"{p['name']} 当前价格：\n"
            f"- 24小时均价: {p['avg24hPrice']} 卢布\n"
            f"- 24小时最低/最高: {p['low24hPrice']} / {p['high24hPrice']}\n"
            f"- 最近最低价: {p['lastLowPrice']}\n"
            f"- 48小时涨跌: {p['changeLast48hPercent']}%\n"
            f"- 基础参考价: {p['basePrice']}")


def tool_get_price_trend(args: dict) -> str:
    """查价格历史趋势（最近 N 天，数据来自本地 SQLite 攒的价格管道）"""
    global last_trend_chart
    last_trend_chart = None  # 每次查询前重置

    name = args.get("item_name", "").strip()
    try:
        days = int(args.get("days", 7))
    except (TypeError, ValueError):
        days = 7
    found = _find_item(name)
    if not found:
        return f"没找到「{name}」这个物品"
    if "candidates" in found:
        return _candidates_text(found["candidates"])

    # 从本地 SQLite 读趋势；没覆盖到窗口起点时自动回补该物品完整历史
    points = price_store.get_trend_with_history(found["id"], days)

    if len(points) < 2:
        return f"「{found['name']}」最近 {days} 天价格数据不足，暂无法判断趋势（等价格管道攒一段时间后就有了）"

    prices = [int(p["price"]) for p in points]  # SQLite 读出的 price 是 float，转 int 供 :d 格式化
    first, last = prices[0], prices[-1]
    change = last - first
    pct = (change / first * 100) if first else 0
    trend = "上涨" if change > 0 else ("下跌" if change < 0 else "持平")

    # 暴露图表数据（供前端画折线图）
    last_trend_chart = {
        "item_id": found["id"],
        "name": found["name"],
        "days": days,
        "points": points,
    }

    return (f"{found['name']} 最近 {days} 天价格趋势：\n"
            f"- 数据点数: {len(points)} 个\n"
            f"- 区间起点: {first} 卢布 → 最新: {last} 卢布\n"
            f"- 变化: {change:+d} 卢布（{pct:+.1f}%）\n"
            f"- 走势: {trend}\n"
            f"- 区间最低/最高: {min(prices)} / {max(prices)}")


def tool_get_item_info(args: dict) -> str:
    """查物品详情"""
    name = args.get("item_name", "").strip()
    found = _find_item(name)
    if not found:
        return f"没找到「{name}」这个物品"
    if "candidates" in found:
        return _candidates_text(found["candidates"])

    item = tarkov.items.get(found["id"])
    if not item:
        return f"无法获取「{found['name']}」的详情"
    types = item.get("types", [])
    type_map = {
        "gun": "武器", "ammo": "弹药", "armor": "护甲", "barter": "物资",
        "wearable": "穿戴", "container": "容器", "key": "钥匙",
    }
    type_zh = "/".join(type_map.get(t, t) for t in types) or "未知"
    return (f"{found['name']} 详情：\n"
            f"- 英文名: {item.get('normalizedName', '')}\n"
            f"- 类型: {type_zh}\n"
            f"- 占格: {item.get('width')}x{item.get('height')}\n"
            f"- 重量: {item.get('weight')} 千克\n"
            f"- 基础参考价: {item.get('basePrice')} 卢布\n"
            f"- 跳蚤市场最低等级: {item.get('minLevelForFlea', '未知')}")


def tool_compare_sell(args: dict) -> str:
    """比价：跳蚤市场 vs 卖给商人"""
    name = args.get("item_name", "").strip()
    found = _find_item(name)
    if not found:
        return f"没找到「{name}」这个物品"
    if "candidates" in found:
        return _candidates_text(found["candidates"])

    item = tarkov.items.get(found["id"])
    if not item:
        return f"无法获取「{found['name']}」的数据"

    # 是否禁止跳蚤交易：用类型字段判断，比「价格缺失」更准确
    no_flea = "noFlea" in item.get("types", [])
    flea = item.get("avg24hPrice") or item.get("lastLowPrice")

    # 商人收购价列表，取最高价
    sell_list = item.get("sellToTrader", [])
    if sell_list:
        best = max(sell_list, key=lambda x: x.get("priceRUB", 0))
        best_trader = tarkov.trader_name(best.get("trader", ""))
        trader_price = best.get("priceRUB", 0)
    else:
        best_trader, trader_price = None, 0

    if no_flea:
        return (f"{found['name']} 禁止跳蚤交易"
                + (f"，卖给商人最高 {trader_price} 卢布（{best_trader}）" if best_trader else ""))

    if flea is None:
        return f"{found['name']} 暂无跳蚤市场价格数据"

    if best_trader:
        diff = flea - trader_price
        advice = (f"跳蚤市场更高，建议挂跳蚤（多赚 {diff} 卢布）" if diff > 0
                  else f"卖给商人更划算（多 {abs(diff)} 卢布）")
        return (f"{found['name']} 出手比价：\n"
                f"- 跳蚤市场均价: {flea} 卢布\n"
                f"- 卖给商人最高: {trader_price} 卢布（{best_trader}）\n"
                f"- 结论: {advice}")
    return f"{found['name']} 跳蚤市场均价 {flea} 卢布（无商人收购信息）"


def tool_rag_search(args: dict) -> str:
    """语义检索知识库（物品百科、任务需求、藏身处配方）"""
    query = args.get("query", "").strip()
    if not query:
        return "请提供要检索的问题"
    top_k = args.get("top_k", 5)
    if isinstance(top_k, str):
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5

    results = kb.search(query, top_k=top_k)
    if not results:
        return "知识库暂不可用，或未检索到相关内容"

    type_label = {"item": "物品", "task": "任务", "craft": "制造配方",
                  "barter": "以物换物", "hideout": "藏身处"}
    lines = [f"知识库检索到 {len(results)} 条相关内容："]
    for i, r in enumerate(results, 1):
        label = type_label.get(r["meta"].get("type", ""), "其他")
        lines.append(f"{i}. [{label}] {r['text']}")
    return "\n".join(lines)


# 工具注册表：工具名 → 处理函数（供 Agent 调度）
TOOLS = {
    "search_item": tool_search_item,
    "get_price": tool_get_price,
    "get_price_trend": tool_get_price_trend,
    "get_item_info": tool_get_item_info,
    "compare_sell": tool_compare_sell,
    "rag_search": tool_rag_search,
}
