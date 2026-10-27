#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑真实检索，导出 preds.jsonl（喂给 run_eval.py）。

对比两组：
  before —— 修复前的检索逻辑（无别名库、中文名不做小写归一）
  after  —— 当前生产的检索逻辑（别名库 + 中文名小写归一）

用法（在 tarkov-helper/backend 目录下，用项目 venv）：
  ..\..\eval_kit> .venv\Scripts\python.exe export_preds.py \
        --queries queries.review20.gold.jsonl --k 5 --out-dir .
"""
import argparse
import json
import os
import sys

# 允许从 backend 目录导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from tarkov_client import tarkov          # noqa: E402
from alias import resolve_alias           # noqa: E402


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def search_after(keyword, k):
    """当前线上逻辑：先过别名库，再模糊匹配（中文名已小写归一）"""
    kw = resolve_alias(keyword.strip())
    return [c["id"] for c in tarkov.search(kw, limit=k)]


def search_before(keyword, k):
    """修复前的逻辑：没有别名库，中文名直接做大小写敏感的子串匹配"""
    kw = keyword.strip().lower()
    if not kw:
        return []
    out = []
    for iid, item in tarkov.items.items():
        zh_full = tarkov.zh.get(item.get("name", ""), "")
        zh_short = tarkov.zh.get(item.get("shortName", ""), "")
        norm = item.get("normalizedName", "").lower()
        if kw in zh_full or kw in zh_short or kw in norm:   # 旧代码：中文名没 lower()
            out.append(iid)
            if len(out) >= k:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    tarkov.load()
    rows = load_jsonl(args.queries)

    after, before, detail = [], [], []
    for q in rows:
        kw = q.get("kw") or (q.get("gold_name") or "")
        if not kw:
            continue
        a = search_after(kw, args.k)
        b = search_before(kw, args.k)
        after.append({"id": q["id"], "top5": a})
        before.append({"id": q["id"], "top5": b})
        gold = set(q.get("gold") or [])
        detail.append({
            "id": q["id"], "query": q["query"], "kw": kw, "type": q.get("type"),
            "gold": sorted(gold),
            "after_hit_rank": next((i for i, x in enumerate(a, 1) if x in gold), None),
            "before_hit_rank": next((i for i, x in enumerate(b, 1) if x in gold), None),
            "after_top3": a[:3], "before_top3": b[:3],
        })

    for name, data in (("preds_after.jsonl", after), ("preds_before.jsonl", before)):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"写入 {path}（{len(data)} 条）")

    with open(os.path.join(args.out_dir, "detail.jsonl"), "w", encoding="utf-8") as f:
        for r in detail:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("写入 detail.jsonl")

    nm = [d for d in detail if d["gold"]]
    for label, key in (("before", "before_hit_rank"), ("after", "after_hit_rank")):
        hit = sum(1 for d in nm if d[key])
        print(f"{label}: {hit}/{len(nm)} = {hit / len(nm):.1%}")


if __name__ == "__main__":
    main()
