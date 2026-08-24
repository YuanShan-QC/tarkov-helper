# -*- coding: utf-8 -*-
"""
tarkov.dev 数据层 —— 拉取物品、价格、中文名

数据源（REST API，免费、无需 key、无速率限制）：
  GET /regular/items           → 全部物品数据（含价格字段，name 是翻译 key）
  GET /regular/items_zh        → 中文翻译映射（翻译 key → 中文）
  GET /regular/prices/{itemId} → 某物品的价格历史时间序列

关键点：
  物品的 name / shortName 字段存的是翻译 key（如 "57347ca924597744596b4e71 Name"），
  真实中文名需要到 items_zh 映射里查。本模块封装好这个转换，对外只暴露中文名。
"""
import httpx
from config import TARKOV_API_BASE


class TarkovClient:
    """塔科夫数据客户端 —— 负责加载数据、搜索物品、查询价格"""

    def __init__(self):
        self.items = {}      # {物品id: 完整物品数据}
        self.zh = {}         # {翻译key: 中文}
        self.traders = {}    # {商人id: 商人数据}
        self.traders_zh = {}  # {翻译key: 商人中文名}
        self._loaded = False  # 是否已加载（懒加载 + 内存缓存）

        # ===== RAG 知识库数据（任务 / 配方 / 藏身处）=====
        self.tasks = {}       # {任务id: 任务数据}
        self.tasks_zh = {}    # {翻译key: 中文}
        self.quest_items = {}  # {任务物品id: 任务物品数据}（如文件、包裹等任务专属品）
        self.crafts = []      # 制造配方列表
        self.barters = []     # 以物换物列表
        self.hideout = {}     # {工作站id: 工作站数据}
        self.hideout_zh = {}  # {翻译key: 中文}
        self._kb_loaded = False  # 知识库数据是否已加载

    def load(self):
        """拉取物品数据 + 中文名 + 商人，建立内存索引（只加载一次）"""
        if self._loaded:
            return
        try:
            # 拉取物品列表、中文翻译、商人数据
            items_resp = httpx.get(f"{TARKOV_API_BASE}/regular/items", timeout=60)
            zh_resp = httpx.get(f"{TARKOV_API_BASE}/regular/items_zh", timeout=60)
            traders_resp = httpx.get(f"{TARKOV_API_BASE}/regular/traders", timeout=30)
            traders_zh_resp = httpx.get(f"{TARKOV_API_BASE}/regular/traders_zh", timeout=30)
            items_resp.raise_for_status()
            zh_resp.raise_for_status()
            traders_resp.raise_for_status()
            traders_zh_resp.raise_for_status()

            self.items = items_resp.json()["data"]["items"]
            self.zh = zh_resp.json()["data"]
            self.traders = traders_resp.json()["data"]
            self.traders_zh = traders_zh_resp.json()["data"]
            self._loaded = True
        except Exception as e:
            # 加载失败时抛出明确错误，避免上层拿到空数据
            raise RuntimeError(f"加载 tarkov.dev 数据失败：{e}")

    def trader_name(self, trader_id: str) -> str:
        """根据商人 id 获取中文名（兜底英文名/原 id）"""
        trader = self.traders.get(trader_id, {})
        name_key = trader.get("name", "")
        return self.traders_zh.get(name_key, name_key or trader_id)

    def load_knowledge(self):
        """加载任务/配方/藏身处数据（RAG 知识库数据源，懒加载一次）"""
        if self._kb_loaded:
            return
        try:
            # 一次性拉取任务、配方、以物换物、藏身处 + 各自中文翻译
            tasks_resp = httpx.get(f"{TARKOV_API_BASE}/regular/tasks", timeout=60)
            tasks_zh_resp = httpx.get(f"{TARKOV_API_BASE}/regular/tasks_zh", timeout=60)
            crafts_resp = httpx.get(f"{TARKOV_API_BASE}/regular/crafts", timeout=60)
            barters_resp = httpx.get(f"{TARKOV_API_BASE}/regular/barters", timeout=60)
            hideout_resp = httpx.get(f"{TARKOV_API_BASE}/regular/hideout", timeout=60)
            hideout_zh_resp = httpx.get(f"{TARKOV_API_BASE}/regular/hideout_zh", timeout=60)
            for r in (tasks_resp, tasks_zh_resp, crafts_resp,
                      barters_resp, hideout_resp, hideout_zh_resp):
                r.raise_for_status()

            tasks_data = tasks_resp.json()["data"]
            self.tasks = tasks_data["tasks"]
            self.quest_items = tasks_data.get("questItems", {})
            self.tasks_zh = tasks_zh_resp.json()["data"]
            self.crafts = crafts_resp.json()["data"]
            self.barters = barters_resp.json()["data"]
            self.hideout = hideout_resp.json()["data"]
            self.hideout_zh = hideout_zh_resp.json()["data"]
            self._kb_loaded = True
        except Exception as e:
            raise RuntimeError(f"加载知识库数据失败：{e}")

    def name_by_id(self, item_id: str):
        """物品 id → 中文名（RAG 构建时高频调用，找不到返回 None）"""
        self.load()
        item = self.items.get(item_id)
        if not item:
            return None
        return self.item_name(item)

    def task_name(self, task: dict) -> str:
        """任务中文名"""
        return self.tasks_zh.get(task.get("name", ""), task.get("normalizedName", "未知任务"))

    def quest_item_name(self, qid: str) -> str:
        """任务专属物品 id → 中文名"""
        qi = self.quest_items.get(qid, {})
        return self.tasks_zh.get(qi.get("name", ""), qid)

    def hideout_name(self, station_id: str) -> str:
        """藏身处工作站中文名"""
        station = self.hideout.get(station_id, {})
        key = station.get("name", "")
        return self.hideout_zh.get(key, station.get("normalizedName", station_id))

    def item_name(self, item: dict) -> str:
        """获取物品中文名（优先完整名，其次短名，兜底英文规范名）"""
        full = self.zh.get(item.get("name", ""), "")
        short = self.zh.get(item.get("shortName", ""), "")
        return full or short or item.get("normalizedName", "未知物品")

    def search(self, keyword: str, limit: int = 5) -> list:
        """按关键词搜索物品，返回候选列表（中文名/短名/英文规范名模糊匹配）"""
        self.load()
        keyword = keyword.strip().lower()
        if not keyword:
            return []

        results = []
        for iid, item in self.items.items():
            zh_full = self.zh.get(item.get("name", ""), "")
            zh_short = self.zh.get(item.get("shortName", ""), "")
            norm = item.get("normalizedName", "").lower()
            # 三个维度模糊匹配：中文完整名、中文短名、英文规范名
            if keyword in zh_full or keyword in zh_short or keyword in norm:
                results.append({
                    "id": iid,
                    "name": zh_full or zh_short or item.get("normalizedName", ""),
                    "shortName": zh_short,
                    "normalizedName": item.get("normalizedName", ""),
                })
                if len(results) >= limit:
                    break
        return results

    def get_price(self, item_id: str) -> dict:
        """获取某物品的当前价格信息"""
        self.load()
        item = self.items.get(item_id)
        if not item:
            return None
        types = item.get("types", [])
        return {
            "name": self.item_name(item),
            "avg24hPrice": item.get("avg24hPrice"),        # 24小时均价
            "low24hPrice": item.get("low24hPrice"),        # 24小时最低
            "high24hPrice": item.get("high24hPrice"),      # 24小时最高
            "lastLowPrice": item.get("lastLowPrice"),      # 最近最低价
            "changeLast48h": item.get("changeLast48h"),    # 48小时价格变化绝对值
            "changeLast48hPercent": item.get("changeLast48hPercent"),  # 48小时涨跌幅%
            "basePrice": item.get("basePrice"),            # 基础价（商人参考）
            "lastScan": item.get("lastScan"),              # 最后扫描时间
            "noFlea": "noFlea" in types,                   # 是否禁止跳蚤市场
        }

    def get_price_history(self, item_id: str) -> list:
        """获取某物品的价格历史（时间序列）"""
        try:
            resp = httpx.get(f"{TARKOV_API_BASE}/regular/prices/{item_id}", timeout=30)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            # 价格历史拉取失败不致命，返回空列表并提示
            return []

    # ===== 以下为多页面展示用的解析方法（直接返回中文，供 REST 接口调用）=====

    def search_tasks(self, keyword: str = "", limit: int = 100) -> list:
        """按关键词搜索任务（匹配中文名/英文规范名），返回 [{id, name, normalizedName}]"""
        self.load_knowledge()
        keyword = keyword.strip().lower()
        results = []
        for tid, task in self.tasks.items():
            name = self.task_name(task)
            norm = task.get("normalizedName", "").lower()
            if not keyword or keyword in name.lower() or keyword in norm:
                results.append({
                    "id": tid,
                    "name": name,
                    "normalizedName": task.get("normalizedName", ""),
                })
                if len(results) >= limit:
                    break
        return results

    def task_detail(self, task_id: str):
        """任务详情：目标（含所需物品中文名）、完成奖励、商人、前置等级"""
        self.load_knowledge()
        task = self.tasks.get(task_id)
        if not task:
            return None
        objectives = []
        for obj in task.get("objectives", []):
            desc = self.tasks_zh.get(obj.get("description", ""), "")
            otype = obj.get("type", "")
            count = obj.get("count", 1)
            item_names = []
            for iid in obj.get("items", []):
                nm = self.name_by_id(iid) or self.quest_item_name(iid)
                if nm:
                    item_names.append(nm)
            objectives.append({
                "type": otype, "desc": desc, "count": count, "items": item_names,
            })
        rewards = task.get("finishRewards") or {}
        reward_items = []
        for it in rewards.get("items", []):
            nm = self.name_by_id(it.get("item", ""))
            if nm:
                reward_items.append({"name": nm, "count": it.get("count", 1)})
        return {
            "id": task_id,
            "name": self.task_name(task),
            "normalizedName": task.get("normalizedName", ""),
            "trader": self.trader_name(task.get("trader", "")),
            "minPlayerLevel": task.get("minPlayerLevel"),
            "objectives": objectives,
            "reward_items": reward_items,
            "neededKeys": task.get("neededKeys", []),
            "wikiLink": task.get("wikiLink", ""),
        }

    def craft_list(self) -> list:
        """藏身处制造配方列表（中文解析）"""
        self.load_knowledge()
        result = []
        for c in self.crafts:
            prod = c.get("productItem", {})
            reqs = []
            for ri in c.get("requiredItems", []):
                nm = self.name_by_id(ri.get("item", "")) or ri.get("item", "")
                reqs.append({"name": nm, "count": ri.get("count", 1)})
            result.append({
                "station": self.hideout_name(c.get("station", "")),
                "level": c.get("level", 1),
                "product": self.name_by_id(prod.get("item", "")) or prod.get("item", ""),
                "product_count": prod.get("count", 1),
                "requirements": reqs,
                "duration": c.get("duration", 0),
            })
        return result

    def barter_list(self) -> list:
        """商人以物换物列表（中文解析）"""
        self.load_knowledge()
        result = []
        for b in self.barters:
            offered = b.get("offeredItem", {})
            reqs = []
            for ri in b.get("requiredItems", []):
                nm = self.name_by_id(ri.get("item", "")) or ri.get("item", "")
                reqs.append({"name": nm, "count": ri.get("count", 1)})
            result.append({
                "trader": self.trader_name(b.get("trader", "")),
                "offered": self.name_by_id(offered.get("item", "")) or offered.get("item", ""),
                "offered_count": offered.get("count", 1),
                "requirements": reqs,
                "minTraderLevel": b.get("minTraderLevel"),
            })
        return result

    def hideout_list(self) -> list:
        """藏身处工作站列表（含各级升级材料）"""
        self.load_knowledge()
        result = []
        for sid, st in self.hideout.items():
            levels = []
            for lv in st.get("levels", []):
                reqs = []
                for ri in lv.get("itemRequirements", []):
                    nm = self.name_by_id(ri.get("item", "")) or ri.get("item", "")
                    reqs.append({"name": nm, "count": ri.get("count", 1)})
                levels.append({
                    "level": lv.get("level"),
                    "requirements": reqs,
                    "constructionTime": lv.get("constructionTime"),
                })
            result.append({"id": sid, "name": self.hideout_name(sid), "levels": levels})
        return result


# 全局单例，所有模块共用同一份缓存数据
tarkov = TarkovClient()
