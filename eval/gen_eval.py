#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动扩充评测集：从物品库抽样，按 4 种问法生成查询，跑 before/after 对比。

目的是给人工评测集（20 条）做量级校验——人工集更接近真实用户说法但样本小，
自动集样本大但更"规整"。两个数一起看才可信。

用法（tarkov-helper/backend 的 venv）：
  python gen_eval.py --items 60 --seeds 20261027
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from tarkov_client import tarkov          # noqa: E402
from alias import resolve_alias           # noqa: E402

# (用户问句模板, 检索关键词取自哪个字段, 类型标签)
FORMS = [
    ("{name}多少钱", "full", "exact"),
    ("{short}价格", "short", "slang"),
    ("{norm} 什么价", "norm", "abbr"),
    ("{name}值不值", "full", "exact"),
]


def search_after(keyword, k=5):
    return [c["id"] for c in tarkov.search(resolve_alias(keyword.strip()), limit=k)]


def search_before(keyword, k=5):
    kw = keyword.strip().lower()
    out = []
    for iid, item in tarkov.items.items():
        zh_full = tarkov.zh.get(item.get("name", ""), "")
        zh_short = tarkov.zh.get(item.get("shortName", ""), "")
        norm = item.get("normalizedName", "").lower()
        if kw in zh_full or kw in zh_short or kw in norm:
            out.append(iid)
            if len(out) >= k:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=60, help="抽样多少个物品")
    ap.add_argument("--seeds", type=int, default=20261027)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    tarkov.load()
    random.seed(args.seeds)

    pool = []
    for iid, item in tarkov.items.items():
        full = tarkov.zh.get(item.get("name", ""), "")
        short = tarkov.zh.get(item.get("shortName", ""), "")
        norm = item.get("normalizedName", "")
        if full and short and norm and len(full) <= 30:
            pool.append({"id": iid, "full": full, "short": short, "norm": norm})
    sample = random.sample(pool, min(args.items, len(pool)))

    rows = []
    for it in sample:
        for tpl, kw_field, qtype in FORMS:
            kw = it[kw_field]
            rows.append({"query": tpl.format(name=it["full"], short=it["short"], norm=it["norm"]),
                         "kw": kw, "type": qtype, "gold": [it["id"]]})

    stats = {}
    for label, fn in (("before", search_before), ("after", search_after)):
        hit = 0
        miss = []
        for r in rows:
            top = fn(r["kw"], args.k)
            if r["gold"][0] in top:
                hit += 1
            else:
                miss.append(r)
        stats[label] = (hit, len(rows), miss)

    print(f"自动评测集：{len(sample)} 个物品 × {len(FORMS)} 种问法 = {len(rows)} 条\n")
    for label in ("before", "after"):
        hit, n, _ = stats[label]
        print(f"  {label}: Hit@{args.k} = {hit}/{n} = {hit / n:.1%}")

    miss_after = stats["after"][2]
    print(f"\nafter 的 badcase（{len(miss_after)} 条，前 15）：")
    for r in miss_after[:15]:
        print(f"   {r['kw']}")

    with open("queries.auto240.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"query": r["query"], "kw": r["kw"],
                                "type": r["type"], "gold": r["gold"]}, ensure_ascii=False) + "\n")
    print("\n已写入 queries.auto240.jsonl")


if __name__ == "__main__":
    main()
