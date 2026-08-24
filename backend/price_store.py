# -*- coding: utf-8 -*-
"""
价格存储管道 —— 定时抓取 tarkov.dev 价格，存入本地 SQLite，攒出 7/30 天趋势

设计思路：
  1. 首次查询某物品趋势时，从 /regular/prices/{itemId} 导入完整历史（立刻有图，不用等慢慢攒）
  2. 后台线程每隔 FETCH_INTERVAL 秒抓一次 /regular/items 全量价格，把有价格的物品写入快照
  3. 提供趋势查询：按物品 id + 天数，返回 [{ts(毫秒), price}] 供前端画折线图

数据表：
  price_history(item_id, ts, price)  —— 价格历史（含导入的历史 + 定时快照）
"""
import os
import sqlite3
import threading
import time
from pathlib import Path

import httpx

from config import TARKOV_API_BASE

# 数据库文件：backend/data/prices.db
DB_PATH = Path(__file__).resolve().parent / "data" / "prices.db"

# 抓取间隔（秒），默认 1 小时；可通过环境变量覆盖
FETCH_INTERVAL = int(os.getenv("PRICE_FETCH_INTERVAL", "3600"))


class PriceStore:
    """本地价格存储：SQLite 持久化 + 定时快照 + 趋势查询"""

    def __init__(self):
        self._conn = None
        self._lock = threading.Lock()  # 多线程读写保护
        self._scheduler_thread = None
        self._init_db()

    def _get_conn(self):
        """获取 SQLite 连接（懒加载，允许跨线程访问）"""
        if self._conn is None:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")  # WAL 提升并发读写
        return self._conn

    def _init_db(self):
        """建表（幂等）"""
        conn = self._get_conn()
        with self._lock:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    item_id TEXT NOT NULL,
                    ts       INTEGER NOT NULL,   -- 时间戳（毫秒）
                    price    REAL NOT NULL,
                    PRIMARY KEY (item_id, ts)
                )
            """)
            conn.commit()

    def _count(self, item_id: str) -> int:
        """某物品本地已存的价格条数"""
        conn = self._get_conn()
        with self._lock:
            cur = conn.execute(
                "SELECT COUNT(*) FROM price_history WHERE item_id=?", (item_id,))
            return cur.fetchone()[0]

    def import_history(self, item_id: str) -> int:
        """
        从 tarkov.dev 导入某物品的完整价格历史（首次查询时调用）。
        返回导入条数；失败返回 0（不抛异常，上层兜底）。
        """
        try:
            resp = httpx.get(
                f"{TARKOV_API_BASE}/regular/prices/{item_id}", timeout=60)
            resp.raise_for_status()
            history = resp.json().get("data", [])
        except Exception:
            return 0

        if not history:
            return 0

        rows = []
        for h in history:
            ts = h.get("timestamp")
            price = h.get("price")
            if ts and price:
                rows.append((item_id, int(ts), float(price)))

        if not rows:
            return 0

        conn = self._get_conn()
        with self._lock:
            conn.executemany(
                "INSERT OR IGNORE INTO price_history(item_id, ts, price) VALUES(?,?,?)",
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
        rows = [(iid, ts, float(p)) for iid, p in prices.items() if p]
        if not rows:
            return 0
        conn = self._get_conn()
        with self._lock:
            conn.executemany(
                "INSERT OR IGNORE INTO price_history(item_id, ts, price) VALUES(?,?,?)",
                rows)
            conn.commit()
        return len(rows)

    def get_trend(self, item_id: str, days: int = 7) -> list:
        """
        查询某物品最近 days 天的价格序列，返回 [{ts, price}]（按时间升序）。
        数据不足时返回空列表。
        """
        cutoff = int((time.time() - days * 86400) * 1000)
        conn = self._get_conn()
        with self._lock:
            cur = conn.execute(
                "SELECT ts, price FROM price_history "
                "WHERE item_id=? AND ts>=? ORDER BY ts ASC",
                (item_id, cutoff))
            rows = cur.fetchall()
        return [{"ts": ts, "price": price} for ts, price in rows]

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
