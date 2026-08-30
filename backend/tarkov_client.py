# -*- coding: utf-8 -*-
"""
tarkov.dev 数据层 —— 拉取物品、价格、中文名

数据源（REST API，免费、无需 key、无速率限制）：
  GET /regular/items           → 全部物品数据（含价格字段，name 是翻译 key）
  GET /regular/items_zh        → 中文翻译映射（翻译 key → 中文）
  GET /regular/prices/{itemId} → 某物品的价格历史时间序列

游戏模式：
  tarkov.dev 提供两套独立经济数据：/regular/*（PvP 经典模式）与 /pve/*（PvE 模式）。
  通过 .env 的 TARKOV_MODE=pvp|pve 切换（默认 pvp），缓存按模式分开存放。

关键点：
  物品的 name / shortName 字段存的是翻译 key（如 "57347ca924597744596b4e71 Name"），
  真实中文名需要到 items_zh 映射里查。本模块封装好这个转换，对外只暴露中文名。

启动加速：
  网络数据会落盘缓存（backend/data/*.json），重启时秒级加载本地缓存，
  再在后台线程联网刷新，避免每次开机都干等网络导致页面打不开。
"""
import json
import os
import threading
import time
from pathlib import Path

import httpx
import config  # 模式可运行时切换，必须动态读取 config.GAME_MODE / config.TARKOV_MODE_PREFIX
from config import TARKOV_API_BASE

# ===== 本地磁盘缓存（启动加速：秒读缓存，后台联网刷新）=====
CACHE_DIR = Path(__file__).resolve().parent / "data"


def _items_cache_file() -> Path:
    """当前模式的物品缓存文件（PvP / PvE 分开存放，避免互相覆盖）"""
    return CACHE_DIR / f"tarkov_items_cache_{config.GAME_MODE}.json"


def _knowledge_cache_file() -> Path:
    """当前模式的知识库缓存文件（任务/配方/以物换物/藏身处）"""
    return CACHE_DIR / f"tarkov_knowledge_cache_{config.GAME_MODE}.json"


# 缓存有效期（秒）：物品/价格数据 6 小时刷新一次；任务配方等知识数据 24 小时刷新一次
ITEMS_CACHE_TTL = int(os.getenv("TARKOV_ITEMS_TTL", "21600"))
KNOWLEDGE_CACHE_TTL = int(os.getenv("TARKOV_KNOWLEDGE_TTL", "86400"))

# ===== 物品分类（收束分组用）=====
# tarkov.dev 的 types 是细分类（mods/pistolGrip/noFlea…），noFlea/markedOnly 只是标记不是分类。
# 这里按优先级把细分类归并为中文大类：命中第一个即为主分类。
CATEGORY_ORDER = [
    ("gun", "武器"), ("preset", "武器"),
    ("ammo", "弹药"), ("ammoBox", "弹药"),
    ("armor", "护甲"), ("armorPlate", "护甲"),
    ("rig", "穿戴"), ("backpack", "穿戴"), ("helmet", "穿戴"),
    ("glasses", "穿戴"), ("headphones", "穿戴"), ("wearable", "穿戴"),
    ("meds", "医疗"), ("injectors", "医疗"),
    ("keys", "钥匙"),
    ("grenade", "投掷物"),
    ("container", "容器"),
    ("provisions", "补给"),
    ("barter", "物资"),
    ("mods", "改装件"), ("pistolGrip", "改装件"), ("suppressor", "改装件"),
    ("specialSlot", "杂物"), ("poster", "杂物"),
]
# 分类展示顺序（去重后）
CATEGORY_DISPLAY_ORDER = []
for _t, zh_cat in CATEGORY_ORDER:
    if zh_cat not in CATEGORY_DISPLAY_ORDER:
        CATEGORY_DISPLAY_ORDER.append(zh_cat)


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
        self._load_lock = threading.RLock()   # 物品数据加载/刷新/切换互斥（可重入）
        self._kb_lock = threading.RLock()     # 知识库数据加载/刷新互斥（可重入）

    # ---------- 本地磁盘缓存 ----------
    @staticmethod
    def _cache_age(file: Path):
        """缓存文件距今多少秒；不存在返回极大值（视为过期）"""
        try:
            return time.time() - file.stat().st_mtime
        except OSError:
            return float("inf")

    @staticmethod
    def _save_json_atomic(file: Path, payload: dict):
        """原子写 JSON（先写临时文件再替换，避免写一半崩溃损坏缓存）"""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, file)
        except Exception as e:
            print(f"[缓存] 写入失败 {file.name}：{e}")

    @staticmethod
    def _load_json_cache(file: Path):
        """读缓存 JSON；不存在或损坏返回 None"""
        try:
            if not file.exists():
                return None
            with open(file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def load(self, refresh: bool = False) -> bool:
        """
        加载物品数据 + 中文名 + 商人（只加载一次）。
        refresh=False：优先读本地缓存（秒级），无缓存才联网；
        refresh=True ：强制联网拉取并覆盖缓存（后台线程调用）。
        联网失败时回退本地缓存，保证服务始终可用。
        返回 True=使用本地缓存，False=联网拉取。
        """
        if self._loaded and not refresh:
            return True
        with self._load_lock:
            if self._loaded and not refresh:
                return True
            # 非刷新场景：先尝试本地缓存
            if not refresh:
                cached = self._load_json_cache(_items_cache_file())
                if cached and cached.get("items"):
                    self.items = cached["items"]
                    self.zh = cached.get("zh", {})
                    self.traders = cached.get("traders", {})
                    self.traders_zh = cached.get("traders_zh", {})
                    self._loaded = True
                    print(f"[缓存] 物品数据从本地加载（{len(self.items)} 个物品，模式 {config.GAME_MODE.upper()}）")
                    return True
            try:
                # 拉取物品列表、中文翻译、商人数据（按 TARKOV_MODE 选 PvP/PvE 端点）
                items_resp = httpx.get(f"{TARKOV_API_BASE}{config.TARKOV_MODE_PREFIX}/items", timeout=60)
                zh_resp = httpx.get(f"{TARKOV_API_BASE}{config.TARKOV_MODE_PREFIX}/items_zh", timeout=60)
                traders_resp = httpx.get(f"{TARKOV_API_BASE}{config.TARKOV_MODE_PREFIX}/traders", timeout=30)
                traders_zh_resp = httpx.get(f"{TARKOV_API_BASE}{config.TARKOV_MODE_PREFIX}/traders_zh", timeout=30)
                items_resp.raise_for_status()
                zh_resp.raise_for_status()
                traders_resp.raise_for_status()
                traders_zh_resp.raise_for_status()

                items = items_resp.json()["data"]["items"]
                zh = zh_resp.json()["data"]
                traders = traders_resp.json()["data"]
                traders_zh = traders_zh_resp.json()["data"]
                # 整包替换（原子性），避免读写线程拿到半新半旧的数据
                self.items, self.zh = items, zh
                self.traders, self.traders_zh = traders, traders_zh
                self._loaded = True
                self._save_json_atomic(_items_cache_file(), {
                    "items": items, "zh": zh,
                    "traders": traders, "traders_zh": traders_zh,
                })
                return False
            except Exception as e:
                # 联网失败时回退本地缓存（如果还没加载过），保证服务可用
                if not self._loaded:
                    cached = self._load_json_cache(_items_cache_file())
                    if cached and cached.get("items"):
                        self.items = cached["items"]
                        self.zh = cached.get("zh", {})
                        self.traders = cached.get("traders", {})
                        self.traders_zh = cached.get("traders_zh", {})
                        self._loaded = True
                        print(f"[缓存] 联网刷新失败，已回退本地缓存：{e}")
                        return True
                raise RuntimeError(f"加载 tarkov.dev 数据失败：{e}")

    def trader_name(self, trader_id: str) -> str:
        """根据商人 id 获取中文名（兜底英文名/原 id）"""
        trader = self.traders.get(trader_id, {})
        name_key = trader.get("name", "")
        return self.traders_zh.get(name_key, name_key or trader_id)

    def load_knowledge(self, refresh: bool = False):
        """加载任务/配方/藏身处数据（RAG 知识库数据源）。优先本地缓存，可强制联网刷新。"""
        if self._kb_loaded and not refresh:
            return
        with self._kb_lock:
            if self._kb_loaded and not refresh:
                return
            if not refresh:
                cached = self._load_json_cache(_knowledge_cache_file())
                if cached and cached.get("tasks") is not None:
                    self.tasks = cached["tasks"]
                    self.quest_items = cached.get("quest_items", {})
                    self.tasks_zh = cached.get("tasks_zh", {})
                    self.crafts = cached.get("crafts", [])
                    self.barters = cached.get("barters", [])
                    self.hideout = cached.get("hideout", {})
                    self.hideout_zh = cached.get("hideout_zh", {})
                    self._kb_loaded = True
                    print(f"[缓存] 知识库数据从本地加载（{len(self.tasks)} 个任务，模式 {config.GAME_MODE.upper()}）")
                    return
            try:
                # 一次性拉取任务、配方、以物换物、藏身处 + 各自中文翻译（按模式选端点）
                tasks_resp = httpx.get(f"{TARKOV_API_BASE}{config.TARKOV_MODE_PREFIX}/tasks", timeout=60)
                tasks_zh_resp = httpx.get(f"{TARKOV_API_BASE}{config.TARKOV_MODE_PREFIX}/tasks_zh", timeout=60)
                crafts_resp = httpx.get(f"{TARKOV_API_BASE}{config.TARKOV_MODE_PREFIX}/crafts", timeout=60)
                barters_resp = httpx.get(f"{TARKOV_API_BASE}{config.TARKOV_MODE_PREFIX}/barters", timeout=60)
                hideout_resp = httpx.get(f"{TARKOV_API_BASE}{config.TARKOV_MODE_PREFIX}/hideout", timeout=60)
                hideout_zh_resp = httpx.get(f"{TARKOV_API_BASE}{config.TARKOV_MODE_PREFIX}/hideout_zh", timeout=60)
                for r in (tasks_resp, tasks_zh_resp, crafts_resp,
                          barters_resp, hideout_resp, hideout_zh_resp):
                    r.raise_for_status()

                tasks_data = tasks_resp.json()["data"]
                tasks = tasks_data["tasks"]
                quest_items = tasks_data.get("questItems", {})
                tasks_zh = tasks_zh_resp.json()["data"]
                crafts = crafts_resp.json()["data"]
                barters = barters_resp.json()["data"]
                hideout = hideout_resp.json()["data"]
                hideout_zh = hideout_zh_resp.json()["data"]
                self.tasks, self.quest_items = tasks, quest_items
                self.tasks_zh, self.crafts = tasks_zh, crafts
                self.barters, self.hideout = barters, hideout
                self.hideout_zh = hideout_zh
                self._kb_loaded = True
                self._save_json_atomic(_knowledge_cache_file(), {
                    "tasks": tasks, "quest_items": quest_items,
                    "tasks_zh": tasks_zh, "crafts": crafts,
                    "barters": barters, "hideout": hideout,
                    "hideout_zh": hideout_zh,
                })
            except Exception as e:
                if not self._kb_loaded:
                    cached = self._load_json_cache(_knowledge_cache_file())
                    if cached and cached.get("tasks") is not None:
                        self.tasks = cached["tasks"]
                        self.quest_items = cached.get("quest_items", {})
                        self.tasks_zh = cached.get("tasks_zh", {})
                        self.crafts = cached.get("crafts", [])
                        self.barters = cached.get("barters", [])
                        self.hideout = cached.get("hideout", {})
                        self.hideout_zh = cached.get("hideout_zh", {})
                        self._kb_loaded = True
                        print(f"[缓存] 知识库联网刷新失败，已回退本地缓存：{e}")
                        return
                raise RuntimeError(f"加载知识库数据失败：{e}")

    def set_mode(self, mode: str) -> dict:
        """
        运行时切换价格模式（pvp / pve）：更新配置 + 重载对应模式数据。
        优先读新模式本地缓存；无缓存则联网拉取（首次切换该模式会稍慢）。
        """
        mode = mode.strip().lower()
        if mode not in ("pvp", "pve"):
            raise ValueError(f"未知模式：{mode}（仅支持 pvp / pve）")
        with self._load_lock:
            if mode == config.GAME_MODE:
                return {"mode": mode, "items": len(self.items), "from_cache": True,
                        "message": "已处于该模式"}
            config.set_game_mode(mode)
            # 清空旧模式内存数据，重新加载新模式数据（锁内完成，防止读到半新半旧状态）
            self.items, self.zh = {}, {}
            self.traders, self.traders_zh = {}, {}
            self._loaded = False
            from_cache = self.load()
            with self._kb_lock:
                # 知识库数据重置为懒加载（下次访问时读新模式缓存/联网）
                self.tasks, self.tasks_zh = {}, {}
                self.quest_items, self.crafts = {}, []
                self.barters, self.hideout = [], {}
                self.hideout_zh = {}
                self._kb_loaded = False
            return {"mode": mode, "items": len(self.items), "from_cache": from_cache,
                    "message": f"已切换到 {mode.upper()} 模式"}

    def maybe_refresh_all(self):
        """后台刷新：缓存过期才联网拉新数据（启动时由后台线程调用，不阻塞）"""
        try:
            if self._cache_age(_items_cache_file()) > ITEMS_CACHE_TTL:
                print("[缓存] 物品数据缓存已过期，后台联网刷新中…")
                self.load(refresh=True)
                print(f"[缓存] 物品数据刷新完成（{len(self.items)} 个物品，模式 {config.GAME_MODE.upper()}）")
        except Exception as e:
            print(f"[缓存] 物品数据后台刷新失败：{e}")
        try:
            if self._cache_age(_knowledge_cache_file()) > KNOWLEDGE_CACHE_TTL:
                print("[缓存] 知识库数据缓存已过期，后台联网刷新中…")
                self.load_knowledge(refresh=True)
                print(f"[缓存] 知识库数据刷新完成（{len(self.tasks)} 个任务，模式 {config.GAME_MODE.upper()}）")
        except Exception as e:
            print(f"[缓存] 知识库数据后台刷新失败：{e}")

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
            resp = httpx.get(f"{TARKOV_API_BASE}{TARKOV_MODE_PREFIX}/prices/{item_id}", timeout=30)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            # 价格历史拉取失败不致命，返回空列表并提示
            return []

    # ===== 以下为多页面展示用的解析方法（直接返回中文，供 REST 接口调用）=====

    def list_items(self) -> list:
        """全部物品列表（含中文分类），供价格栏"全部物品"分组展示"""
        self.load()
        results = []
        for iid, item in self.items.items():
            types = item.get("types", [])
            cat = "其他"
            for t, zh_cat in CATEGORY_ORDER:
                if t in types:
                    cat = zh_cat
                    break
            results.append({
                "id": iid,
                "name": self.item_name(item),
                "normalizedName": item.get("normalizedName", ""),
                "type": cat,
            })
        return results

    def search_tasks(self, keyword: str = "", limit: int = 100) -> list:
        """按关键词搜索任务（匹配中文名/英文规范名），返回 [{id, name, normalizedName, trader}]"""
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
                    "trader": self.trader_name(task.get("trader", "")) or "其他",
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
