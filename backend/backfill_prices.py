# -*- coding: utf-8 -*-
"""
一次性批量回补价格历史

背景：定时快照只有程序跑着的时候才写数据，所以趋势图里会出现一段段空白
     （只有「启动那几天」有点）。本脚本把 tarkov.dev 的完整历史（每个物品约
     1800 个点）一次性灌进本地库，把快照开始之前的空白补齐。

    注意：网页上点开某个物品时也会自动回补（每天每物品一次），本脚本只是
    不想等用户一个个点、想一次性把整个市场补全时才用。

用法（在 backend 目录下执行）：
    .venv\\Scripts\\python.exe backfill_prices.py                # 当前模式，所有缺历史的物品
    .venv\\Scripts\\python.exe backfill_prices.py --mode pve     # 指定模式
    .venv\\Scripts\\python.exe backfill_prices.py --limit 50     # 只补前 50 个（先试水）
    .venv\\Scripts\\python.exe backfill_prices.py --days 30      # 判定窗口，默认 30 天
    .venv\\Scripts\\python.exe backfill_prices.py --dry-run      # 只统计要补多少，不联网
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import config
from price_store import price_store
from tarkov_client import tarkov

# 每个物品之间的礼貌间隔（秒）—— tarkov.dev 是免费公共服务，别打太狠
DEFAULT_DELAY = 0.2


def parse_args():
    p = argparse.ArgumentParser(description="批量回补 tarkov.dev 价格历史到本地 SQLite")
    p.add_argument("--mode", choices=["pvp", "pve"], help="游戏模式（默认当前 config.GAME_MODE）")
    p.add_argument("--days", type=int, default=30, help="判定「缺历史」的窗口天数，默认 30")
    p.add_argument("--limit", type=int, default=0, help="最多处理多少个物品（0=不限制）")
    p.add_argument("--workers", type=int, default=4, help="并发数，默认 4")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="相邻请求间隔秒数")
    p.add_argument("--dry-run", action="store_true", help="只统计待回补数量，不实际导入")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode:
        config.set_game_mode(args.mode)

    print(f"[回补] 模式={config.GAME_MODE}  判定窗口={args.days} 天  并发={args.workers}")

    # 物品清单：优先用本地缓存，缓存不存在时联网下载一次
    if not tarkov.items:
        print("[回补] 正在加载物品清单…")
        tarkov.load()
    item_ids = list(tarkov.items)
    print(f"[回补] 物品总数 {len(item_ids)}")

    pending = [iid for iid in item_ids if price_store.needs_backfill(iid, args.days)]
    print(f"[回补] 其中缺历史的 {len(pending)} 个")
    if args.limit:
        pending = pending[:args.limit]
        print(f"[回补] 按 --limit 截取前 {len(pending)} 个")
    if args.dry_run or not pending:
        return 0

    done = imported = failed = 0
    started = time.time()

    def _one(iid: str) -> int:
        time.sleep(args.delay)  # 简单限速，不排队也没关系
        # 走 ensure_history：除了导入，还会落 price_import 标记，
        # 这样网页上再点这个物品时不会重复下载
        return price_store.ensure_history(iid, args.days)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for n in pool.map(_one, pending, chunksize=1):
                done += 1
                if n:
                    imported += n
                else:
                    failed += 1
                if done % 25 == 0 or done == len(pending):
                    elapsed = time.time() - started
                    per = elapsed / done
                    eta = per * (len(pending) - done)
                    print(f"[回补] {done}/{len(pending)}  新写入 {imported} 点  "
                          f"失败 {failed}  已用 {elapsed:.0f}s  预计剩余 {eta:.0f}s")
    except KeyboardInterrupt:
        print("\n[回补] 已中断，已处理的部分已写入本地库")

    print(f"\n[回补] 完成：处理 {done} 个物品，写入 {imported} 个历史点，失败 {failed} 个，"
          f"耗时 {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
