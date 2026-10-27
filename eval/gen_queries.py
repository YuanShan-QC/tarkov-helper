#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从物品表批量生成模板化查询（正确答案自动已知，不用人工标注）。

输入 items.csv，首行表头，至少包含 item_id,name；可选 abbr,alias,category
  item_id,name,abbr,alias,category
  item_0481,显卡,GPU,显示卡,硬件

用法：
  python gen_queries.py --items items.csv --out queries.gen.jsonl

生成的每行：
  {"id":1,"query":"显卡多少钱","type":"exact","gold":["item_0481"],"source":"template"}
"""
import argparse
import csv
import json


def row_queries(row, idx):
    """一个物品最多产出 4 条客观题；缺字段就跳过对应类型。"""
    iid = row["item_id"].strip()
    name = (row.get("name") or "").strip()
    abbr = (row.get("abbr") or "").strip()
    alias = (row.get("alias") or "").strip()
    out = []
    if name:
        out.append(("exact", f"{name}多少钱"))
        out.append(("typo", f"{name}那里买"))      # 那里/哪里 是最常见的别字
    if abbr:
        out.append(("abbr", f"{abbr} 价格"))
    if alias:
        out.append(("slang", f"{alias}值不值"))
    res = []
    for t, q in out:
        idx[0] += 1
        res.append({"id": idx[0], "query": q, "type": t,
                    "gold": [iid], "source": "template"})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=10, help="最多取多少个物品")
    args = ap.parse_args()

    with open(args.items, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    idx = [0]
    allq = []
    for row in rows[: args.limit]:
        allq.extend(row_queries(row, idx))

    with open(args.out, "w", encoding="utf-8") as f:
        for q in allq:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"生成 {len(allq)} 条模板查询 -> {args.out}")
    from collections import Counter
    print("类型分布：", dict(Counter(q["type"] for q in allq)))
    print("\n注意：这些是『干净』查询，命中率会虚高。"
          "务必再手工补 20 条真实玩家问法（贴吧/B站评论/Wiki FAQ）混进去。")


if __name__ == "__main__":
    main()
