"""价格存储的历史回补逻辑测试 —— 不联网、不碰真实数据库。

重点验证：
  · 为什么不能用「本地点数 < 2」判断要不要导历史（快照一次就给所有物品写点，
    条件立刻成立，历史永远导不进来 —— 趋势图只剩启动那几天数据的根因）
  · 回补的触发/去重/失败兜底是否都符合预期

运行：
    cd C:\\Users\\Lenovo\\DSH
    pytest tarkov-helper -v
"""
import os
import sys
import tempfile
import time
from pathlib import Path

# 必须在 import price_store 之前指定独立数据库，避免动到真实数据
_TMP_DIR = tempfile.mkdtemp(prefix="tarkov_price_test_")
os.environ["PRICE_DB_PATH"] = str(Path(_TMP_DIR) / "prices.db")

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import price_store as ps  # noqa: E402

DAY_MS = 86400 * 1000


def _store():
    """每个测试一个干净的 store（同一临时库文件，表结构幂等）"""
    return ps.PriceStore()


def _insert(store, item_id, ts, price=100.0, mode=None):
    conn = store._get_conn()
    with store._lock:
        conn.execute(
            "INSERT OR IGNORE INTO price_history(item_id,ts,price,mode) VALUES(?,?,?,?)",
            (item_id, int(ts), price, mode or config.GAME_MODE))
        conn.commit()


def _clear(store, item_id, mode=None):
    conn = store._get_conn()
    with store._lock:
        conn.execute("DELETE FROM price_history WHERE item_id=? AND mode=?",
                     (item_id, mode or config.GAME_MODE))
        conn.execute("DELETE FROM price_import WHERE item_id=? AND mode=?",
                     (item_id, mode or config.GAME_MODE))
        conn.commit()


class TestNeedsBackfill:
    """判定「缺不缺历史」：看窗口内有几天数据，而不是点数多少/有没有更早的点"""

    def test_只有今天一个快照点也要回补(self):
        store = _store()
        config.set_game_mode("pvp")
        iid = "test_only_today"
        _clear(store, iid)
        _insert(store, iid, int(time.time() * 1000))
        # 只有今天 1 个点：旧逻辑 len(points) < 2 不成立 → 永远不回补（bug）
        assert store.needs_backfill(iid, days=7) is True

    def test_窗口内每天都有点就不回补(self):
        store = _store()
        config.set_game_mode("pvp")
        iid = "test_has_history"
        _clear(store, iid)
        now = int(time.time() * 1000)
        for d in range(8):  # 最近 8 天每天都一个点
            _insert(store, iid, now - d * DAY_MS)
        assert store.needs_backfill(iid, days=7) is False
        assert store.needs_backfill(iid, days=30) is True  # 30 天窗口只有 8 天，不够

    def test_很久以前的点不能算窗口内的密度(self):
        """真实 bug：物品有 65 天跨度的数据，但 30 天窗口里只有 1 个点"""
        store = _store()
        config.set_game_mode("pvp")
        iid = "test_old_point_only"
        _clear(store, iid)
        now = int(time.time() * 1000)
        _insert(store, iid, now)                 # 今天启动时的快照
        _insert(store, iid, now - 65 * DAY_MS)   # 65 天前还有过一次快照
        assert store.needs_backfill(iid, days=30) is True, \
            "窗口内只有 1 天数据就该判定为缺历史，否则图还是只有启动那天"

    def test_模式隔离_pvp历史不算pve的(self):
        store = _store()
        iid = "test_mode_iso"
        _clear(store, iid, "pvp")
        _clear(store, iid, "pve")
        now = int(time.time() * 1000)
        for d in range(8):
            _insert(store, iid, now - d * DAY_MS, mode="pvp")
        _insert(store, iid, now, mode="pve")
        config.set_game_mode("pve")
        assert store.needs_backfill(iid, days=7) is True
        config.set_game_mode("pvp")
        assert store.needs_backfill(iid, days=7) is False


class TestEnsureHistory:
    """回补触发 + 当天去重 + 失败兜底"""

    def test_回补后就不再重复导入(self):
        store = _store()
        config.set_game_mode("pvp")
        iid = "test_ensure_ok"
        _clear(store, iid)
        now = int(time.time() * 1000)
        _insert(store, iid, now)

        calls = []

        def fake_import(item_id):
            calls.append(item_id)
            for d in range(1, 8):  # 导入后窗口内每天都有点
                _insert(store, item_id, now - d * DAY_MS, 50.0)
            return 7

        store.import_history = fake_import
        assert store.ensure_history(iid, days=7) == 7
        assert calls == [iid]
        # 回补后窗口已够密，第二次直接返回 0，不再联网
        assert store.ensure_history(iid, days=7) == 0
        assert calls == [iid]

    def test_同一天内失败只尝试一次(self):
        store = _store()
        config.set_game_mode("pvp")
        iid = "test_ensure_fail"
        _clear(store, iid)
        _insert(store, iid, int(time.time() * 1000))

        calls = []
        store.import_history = lambda item_id: (calls.append(item_id), 0)[1]
        assert store.ensure_history(iid, days=7) == 0
        assert store.ensure_history(iid, days=7) == 0
        assert len(calls) == 1, "同一天内不应反复重试，防止刷爆接口"

    def test_已覆盖窗口时不联网(self):
        store = _store()
        config.set_game_mode("pvp")
        iid = "test_ensure_skip"
        _clear(store, iid)
        now = int(time.time() * 1000)
        for d in range(8):
            _insert(store, iid, now - d * DAY_MS)

        store.import_history = lambda item_id: (_ for _ in ()).throw(AssertionError("不该联网"))
        assert store.ensure_history(iid, days=7) == 0

    def test_导入过就不再重复下载_即使仍然稀疏(self):
        """冷门物品历史本身稀疏：导入一次后不能天天重下"""
        store = _store()
        config.set_game_mode("pvp")
        iid = "test_imported_sparse"
        _clear(store, iid)
        _insert(store, iid, int(time.time() * 1000))

        calls = []

        def fake_import(item_id):
            calls.append(item_id)
            return 142  # 有历史，但窗口内依然只有 1~2 天（模拟冷门物品）

        store.import_history = fake_import
        assert store.ensure_history(iid, days=30) == 142
        assert calls == [iid]

        # 模拟「第二天」：清掉内存去重，落库标记仍应拦住
        store._backfill_attempts.clear()
        assert store.ensure_history(iid, days=30) == 0
        assert calls == [iid], "已导入过的物品不应再次联网"

    def test_导入失败不落库标记_第二天可重试(self):
        store = _store()
        config.set_game_mode("pvp")
        iid = "test_import_fail_retry"
        _clear(store, iid)
        _insert(store, iid, int(time.time() * 1000))

        calls = []
        store.import_history = lambda item_id: (calls.append(item_id), 0)[1]
        assert store.ensure_history(iid, days=7) == 0
        assert store._has_imported(iid) is False

        store._backfill_attempts.clear()  # 第二天
        assert store.ensure_history(iid, days=7) == 0
        assert len(calls) == 2, "上次没导入成功，第二天应允许重试"

    def test_上游没有历史就标记_不天天重试(self):
        """卢布这类上游 404 的物品，标记掉即可，不用天天去问"""
        store = _store()
        config.set_game_mode("pvp")
        iid = "test_no_upstream"
        _clear(store, iid)
        _insert(store, iid, int(time.time() * 1000))

        calls = []
        store.import_history = lambda item_id: (calls.append(item_id), -1)[1]
        assert store.ensure_history(iid, days=7) == 0
        assert store._has_imported(iid) is True

        store._backfill_attempts.clear()  # 第二天
        assert store.ensure_history(iid, days=7) == 0
        assert len(calls) == 1

    def test_导入标记按模式隔离(self):
        store = _store()
        iid = "test_import_mode"
        _clear(store, iid, "pvp")
        _clear(store, iid, "pve")
        _insert(store, iid, int(time.time() * 1000), mode="pvp")
        _insert(store, iid, int(time.time() * 1000), mode="pve")

        config.set_game_mode("pvp")
        store.import_history = lambda item_id: 5
        assert store.ensure_history(iid, days=7) == 5
        assert store._has_imported(iid) is True

        config.set_game_mode("pve")
        assert store._has_imported(iid) is False, "PvP 的导入记录不能当成 PvE 的"


class TestGetTrendWithHistory:
    """对外入口：查趋势 + 缺历史时自动回补"""

    def test_自动回补后能画出完整曲线(self):
        store = _store()
        config.set_game_mode("pvp")
        iid = "test_trend_backfill"
        _clear(store, iid)
        now = int(time.time() * 1000)
        _insert(store, iid, now, 100.0)

        def fake_import(item_id):
            for d in range(1, 8):
                _insert(store, item_id, now - d * DAY_MS, 100.0 - d)
            return 7

        store.import_history = fake_import
        points = store.get_trend_with_history(iid, days=7)
        assert len(points) >= 7
        # 升序返回，前端按顺序画点
        assert points == sorted(points, key=lambda p: p["ts"])

    def test_完全没有数据的物品也能拿到历史(self):
        store = _store()
        config.set_game_mode("pvp")
        iid = "test_trend_empty"
        _clear(store, iid)

        def fake_import(item_id):
            _insert(store, item_id, int(time.time() * 1000) - 3 * DAY_MS, 42.0)
            return 1

        store.import_history = fake_import
        points = store.get_trend_with_history(iid, days=7)
        assert len(points) == 1
