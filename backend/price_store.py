# -*- coding: utf-8 -*-
"""
价格存储管道 —— 定时抓取 tarkov.dev 价格，存入本地 SQLite，攒出 7/30 天趋势

设计思路：
  1. 查询某物品趋势时，若本地数据在窗口内不够密（大半天数没有点），就自动从
     {前缀}/prices/{itemId} 导入完整历史（立刻有图，不用等慢慢攒；
     同一物品同一模式每天最多试一次）
  2. 后台线程每隔 FETCH_INTERVAL 秒抓一次全量价格，把有价格的物品写入快照
  3. 提供趋势查询：按物品 id + 天数，返回 [{ts(毫秒), price}] 供前端画折线图

为什么用「窗口内覆盖的天数」而不是「点数够不够」判断要不要回补：
  后台快照一次就会给全部 ~3500 个物品各写 1 个点，所以「本地点数 >= 2」这个条件
  几乎对所有物品立刻成立，历史导入永远轮不到 —— 表现就是趋势图只有启动那几天的
  数据。改成检查「窗口里到底有多少天有数据」，才能保证每个物品都真的回补到历史。

数据表：
  price_history(item_id, ts, price, mode)  —— 价格历史（含导入的历史 + 定时快照）
  注意：前缀随模式变化（/regular=PvP，/pve=PvE），趋势数据按 mode 列隔离，
  切换模式后同一物品的 PvP/PvE 历史不会混在一起。
"""
import os
import sqlite3
import threading
import time
from pathlib import Path

import httpx
import config  # 模式可运行时切换，必须动态读取
from config import TARKOV_API_BASE

# 数据库文件：backend/data/prices.db
# 可用 PRICE_DB_PATH 指到别的文件（批量回补脚本 / 测试用独立库时很方便）
DB_PATH = Path(os.getenv("PRICE_DB_PATH")
               or (Path(__file__).resolve().parent / "data" / "prices.db"))

# 抓取间隔（秒），默认 1 小时；可通过环境变量覆盖
FETCH_INTERVAL = int(os.getenv("PRICE_FETCH_INTERVAL", "3600"))


class PriceStore:
    """本地价格存储：SQLite 持久化 + 定时快照 + 趋势查询"""

    def __init__(self):
        self._conn = None
        self._conn_lock = threading.Lock()  # 懒加载连接的建立保护
        self._lock = threading.Lock()  # 多线程读写保护
        self._scheduler_thread = None
        # 历史回补去重：{(item_id, mode): 'YYYY-MM-DD'} 记录当天已尝试过的物品，
        # 避免用户反复点同一个物品时每次都去拉 1800 多个点的历史
        self._backfill_lock = threading.Lock()
        self._backfill_attempts = {}
        self._init_db()

    def _get_conn(self):
        """获取 SQLite 连接（懒加载，允许跨线程访问；加锁避免并发建多个连接）"""
        if self._conn is None:
            with self._conn_lock:
                if self._conn is None:  # 双重检查：锁内可能已被别的线程创建
                    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                    # timeout=30：导入历史与后台快照可能同时写，等锁而不是立刻报错
                    self._conn = sqlite3.connect(
                        str(DB_PATH), check_same_thread=False, timeout=30)
                    self._conn.execute("PRAGMA journal_mode=WAL")  # WAL 提升并发读写
        return self._conn

    def _init_db(self):
        """建表（幂等）；老库自动补 mode 列（存量数据都是 PvP 快照）"""
        conn = self._get_conn()
        with self._lock:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    item_id TEXT NOT NULL,
                    ts       INTEGER NOT NULL,   -- 时间戳（毫秒）
                    price    REAL NOT NULL,
                    mode     TEXT NOT NULL DEFAULT 'pvp',  -- pvp / pve
                    PRIMARY KEY (item_id, ts)
                )
            """)
            # 老版本库升级：补 mode 列
            cols = [r[1] for r in conn.execute("PRAGMA table_info(price_history)")]
            if "mode" not in cols:
                conn.execute(
                    "ALTER TABLE price_history ADD COLUMN mode TEXT NOT NULL DEFAULT 'pvp'")
            # 记录「某物品的历史已经完整导入过」：冷门物品的交易稀疏，导入后窗口内
            # 天数依然不够密，没有这张表就会天天重复下载同一份历史
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_import (
                    item_id TEXT NOT NULL,
                    mode    TEXT NOT NULL,
                    at      INTEGER NOT NULL,  -- 导入完成时间（毫秒）
                    rows    INTEGER NOT NULL,  -- 实际写入条数（0 = 源头也没有历史）
                    PRIMARY KEY (item_id, mode)
                )
            """)
            conn.commit()

    def _count(self, item_id: str) -> int:
        """某物品本地已存的价格条数（当前模式）"""
        conn = self._get_conn()
        with self._lock:
            cur = conn.execute(
                "SELECT COUNT(*) FROM price_history WHERE item_id=? AND mode=?",
                (item_id, config.GAME_MODE))
            return cur.fetchone()[0]

    def import_history(self, item_id: str) -> int:
        """
        从 tarkov.dev 导入某物品的完整价格历史（首次查询时调用）。

        返回：>0 写入条数；0 = 临时失败（网络/服务异常，可重试）；
        -1 = 上游明确没有该物品的价格历史（404 或 data 为空，重试也没用）。
        """
        try:
            resp = httpx.get(
                f"{TARKOV_API_BASE}{config.TARKOV_MODE_PREFIX}/prices/{item_id}", timeout=60)
            if resp.status_code == 404:
                return -1  # 比如卢布这种不参与市场交易的物品，上游根本没有历史
            resp.raise_for_status()
            history = resp.json().get("data", [])
        except Exception:
            return 0

        if not history:
            return -1

        mode = config.GAME_MODE
        rows = []
        for h in history:
            ts = h.get("timestamp")
            price = h.get("price")
            if ts and price:
                rows.append((item_id, int(ts), float(price), mode))

        if not rows:
            return -1

        conn = self._get_conn()
        with self._lock:
            conn.executemany(
                "INSERT OR IGNORE INTO price_history(item_id, ts, price, mode) VALUES(?,?,?,?)",
                rows)
            conn.commit()
        return len(rows)

    def record_snapshot(self, prices: dict) -> int:
        """
        记录一次全量价格快照。prices: {item_id: price}
        同一秒内同一物品去重，返回实际写入条数。
        """
        if not prices:
            return 0
        ts = int(time.time() * 1000)
        mode = config.GAME_MODE
        rows = [(iid, ts, float(p), mode) for iid, p in prices.items() if p]
        if not rows:
            return 0
        conn = self._get_conn()
        with self._lock:
            conn.executemany(
                "INSERT OR IGNORE INTO price_history(item_id, ts, price, mode) VALUES(?,?,?,?)",
                rows)
            conn.commit()
        return len(rows)

    def get_trend(self, item_id: str, days: int = 7) -> list:
        """
        查询某物品最近 days 天的价格序列（仅当前模式），返回 [{ts, price}]（按时间升序）。
        数据不足时返回空列表。
        """
        cutoff = int((time.time() - days * 86400) * 1000)
        conn = self._get_conn()
        with self._lock:
            cur = conn.execute(
                "SELECT ts, price FROM price_history "
                "WHERE item_id=? AND ts>=? AND mode=? ORDER BY ts ASC",
                (item_id, cutoff, config.GAME_MODE))
            rows = cur.fetchall()
        return [{"ts": ts, "price": price} for ts, price in rows]

    def needs_backfill(self, item_id: str, days: int = 7) -> bool:
        """
        本地数据在 days 天窗口里是否「够密」，不够就认为缺历史、需要回补。

        判据是「窗口内有多少天有数据」而不是「点数够不够」：
        后台快照一次就会给每个物品写 1 个点，用点数判断会让历史导入永远触发不了
        （趋势图只剩启动那几天数据的根因）。而只看「有没有更早的点」也不够 ——
        快照可能集中在很久以前的几天，窗口里其实只有 1~2 个点，图还是秃的。

        阈值取窗口的 60%：tarkov.dev 的历史基本是每天 1 个点，正常回补后能覆盖
        接近整段窗口；低价冷门物品本身交易稀疏、历史点少，会保持「缺」的状态，
        由 ensure_history 的「每天最多试一次」兜底，不会反复打接口。
        """
        cutoff = int((time.time() - days * 86400) * 1000)
        need = max(1, int(days * 0.6))
        conn = self._get_conn()
        with self._lock:
            row = conn.execute(
                "SELECT COUNT(DISTINCT ts / 86400000) FROM price_history "
                "WHERE item_id=? AND mode=? AND ts>=?",
                (item_id, config.GAME_MODE, cutoff)).fetchone()
        return row[0] < need

    def _has_imported(self, item_id: str) -> bool:
        """该物品当前模式的历史是否已经完整导入过（导入过就不再重下）"""
        conn = self._get_conn()
        with self._lock:
            row = conn.execute(
                "SELECT 1 FROM price_import WHERE item_id=? AND mode=?",
                (item_id, config.GAME_MODE)).fetchone()
        return row is not None

    def _mark_imported(self, item_id: str, rows: int):
        conn = self._get_conn()
        with self._lock:
            conn.execute(
                "INSERT OR REPLACE INTO price_import(item_id, mode, at, rows) VALUES(?,?,?,?)",
                (item_id, config.GAME_MODE, int(time.time() * 1000), int(rows)))
            conn.commit()

    def ensure_history(self, item_id: str, days: int = 7) -> int:
        """
        需要时从 tarkov.dev 导入一次完整历史。

        两道去重：
          1. 内存：同一物品 + 同一模式每天最多尝试一次（防止反复点图表时重复下载）
          2. 落库：price_import 表记下已导入过的物品，冷门物品即使导入后
             窗口仍然「不够密」，也不会天天重下
        返回实际导入条数；不需要导入 / 已经导入过 / 今天已试过 / 导入失败都返回 0。
        """
        if not self.needs_backfill(item_id, days):
            return 0
        if self._has_imported(item_id):
            return 0
        key = (item_id, config.GAME_MODE)
        today = time.strftime("%Y-%m-%d")
        with self._backfill_lock:
            if self._backfill_attempts.get(key) == today:
                return 0
            self._backfill_attempts[key] = today
        rows = self.import_history(item_id)
        if rows > 0:
            self._mark_imported(item_id, rows)
            print(f"[回补] {item_id} 导入历史 {rows} 条")
        elif rows < 0:
            # 上游明确没有这个物品的历史（如卢布），标记掉，别天天来问
            self._mark_imported(item_id, 0)
        return max(rows, 0)

    def get_trend_with_history(self, item_id: str, days: int = 7) -> list:
        """
        查趋势；若窗口内数据不够密（缺历史），先自动回补一次再查。

        前端折线图和 Agent 的 price_history 工具都走这里，保证「点开就有完整
        历史曲线」，而不是只有程序运行那几天的碎点。
        """
        self.ensure_history(item_id, days)
        return self.get_trend(item_id, days)

    def start_scheduler(self, fetch_fn):
        """
        启动后台定时抓取线程。fetch_fn 是无参函数，返回 {item_id: price} 快照。
        幂等：已启动则忽略。
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return

        def _loop():
            while True:
                try:
                    prices = fetch_fn()
                    if prices:
                        n = self.record_snapshot(prices)
                        print(f"[价格管道] 已快照 {n} 个物品价格")
                except Exception as e:
                    # 单次抓取失败不影响后续循环
                    print(f"[价格管道] 快照失败：{e}")
                time.sleep(FETCH_INTERVAL)

        self._scheduler_thread = threading.Thread(
            target=_loop, daemon=True, name="price-fetcher")
        self._scheduler_thread.start()


# 全局单例，所有模块共用同一份价格存储
price_store = PriceStore()
