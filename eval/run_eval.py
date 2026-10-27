#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG 检索评测：Hit@K / MRR / 分类型表现 / badcase 导出。

输入（都不依赖你的项目代码，纯文件交换）：
  --queries  eval/queries.jsonl   人工标注的评测集
             {"id":1,"query":"显卡多少钱","type":"exact","gold":["item_0481"]}
  --pred     eval/preds.jsonl     你的系统实际返回的 top-k
             {"id":1,"top5":["item_0481","item_0102","item_0077"]}

用法：
  python run_eval.py --queries queries.example.jsonl --pred mock        # 看输出长什么样
  python run_eval.py --queries eval/queries.jsonl --pred eval/preds.jsonl \
                     --out eval/report.md --badcase eval/badcase.csv
"""
import argparse
import json
import random
import sys
from collections import defaultdict


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{ln} JSON 解析失败：{e}")
    return rows


def synth_preds(queries, k):
    """故意造一份有缺陷的预测，用来看报表长什么样。别用于真实结论。"""
    random.seed(42)
    by_type = {"exact": 0.95, "abbr": 0.70, "slang": 0.45, "typo": 0.60}
    preds = []
    for q in queries:
        gold = q.get("gold") or []
        r = by_type.get(q.get("type", "exact"), 0.6)
        top = []
        if gold and random.random() < r:
            rank = random.randint(0, k - 1)
            top = [f"item_fake_{i}" for i in range(k)]
            top[rank] = gold[0]
        else:
            top = [f"item_fake_{i}" for i in range(k)]
        preds.append({"id": q["id"], "top5": top[:k]})
    return preds


def first_rank(top, gold):
    gs = set(gold)
    for i, item in enumerate(top, 1):
        if item in gs:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--pred", required=True, help="预测文件，或写 mock")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--skip-types", default="vague",
                    help="这些类型不计命中率（无唯一正确答案），逗号分隔")
    ap.add_argument("--out", default=None, help="markdown 报表输出路径")
    ap.add_argument("--badcase", default=None, help="badcase csv 输出路径")
    ap.add_argument("--label", default="run", help="报表标题")
    args = ap.parse_args()

    queries = load_jsonl(args.queries)
    if args.pred.strip().lower() == "mock":
        preds = synth_preds(queries, args.k)
        print("[提示] 当前是 mock 数据，只用于检查报表格式。\n")
    else:
        preds = load_jsonl(args.pred)
    pred_map = {p["id"]: p.get("top5") or [] for p in preds}

    skip = {s.strip() for s in args.skip_types.split(",") if s.strip()}
    stats = defaultdict(lambda: {"n": 0, "hit": 0, "rr": 0.0})
    bad = []
    missing = []
    excluded = defaultdict(int)

    for q in queries:
        qtype = q.get("type", "exact")
        gold = q.get("gold") or []
        if q["id"] not in pred_map:
            missing.append(q["id"])
            continue
        if qtype in skip or not gold:
            excluded[qtype] += 1
            continue
        top = pred_map[q["id"]][: args.k]
        rank = first_rank(top, gold)
        s = stats[qtype]
        s["n"] += 1
        if rank:
            s["hit"] += 1
            s["rr"] += 1.0 / rank
        else:
            bad.append((q, top))

    if missing:
        print(f"[警告] 有 {len(missing)} 条 query 没有预测结果：{missing[:10]}")

    total_n = sum(s["n"] for s in stats.values())
    if total_n == 0:
        sys.exit("没有可评测的样本（检查 --skip-types 与 gold 标注）")
    total_hit = sum(s["hit"] for s in stats.values())
    total_rr = sum(s["rr"] for s in stats.values())

    L = []
    L.append(f"# 检索评测报告（{args.label}）\n")
    L.append(f"- 评测集：`{args.queries}`，有效样本 {total_n} 条"
             f"（另有 {sum(excluded.values())} 条无唯一答案的样本单独统计，不计入命中率）")
    L.append(f"- 指标口径：正确物品出现在 top-{args.k} 记为命中；MRR 取首个正确物品名次的倒数\n")
    L.append(f"**整体 Hit@{args.k} = {total_hit/total_n:.1%}"
             f"（{total_hit}/{total_n}），MRR@{args.k} = {total_rr/total_n:.3f}**\n")
    L.append("| 类型 | 样本数 | 命中 | Hit@%d | MRR@%d |" % (args.k, args.k))
    L.append("|---|---|---|---|---|")
    order = ["exact", "abbr", "slang", "typo", "vague"]
    keys = [t for t in order if t in stats] + [t for t in stats if t not in order]
    for t in keys:
        s = stats[t]
        L.append(f"| {t} | {s['n']} | {s['hit']} | {s['hit']/s['n']:.0%} | {s['rr']/s['n']:.3f} |")
    L.append("")
    if excluded:
        L.append("无唯一答案的样本（考察是否反问/是否给候选，不按命中率算）："
                 + "、".join(f"{t} {n} 条" for t, n in excluded.items()) + "\n")
    L.append(f"## Badcase（{len(bad)} 条）\n")
    if bad:
        L.append("| id | 查询 | 类型 | 正确答案 | 实际返回 |")
        L.append("|---|---|---|---|---|")
        for q, top in bad[:30]:
            L.append(f"| {q['id']} | {q['query']} | {q.get('type','')} | "
                     f"{','.join(q['gold'])} | {','.join(top[:3])} |")
        if len(bad) > 30:
            L.append(f"\n（仅显示前 30 条，完整列表见 badcase csv）")
    L.append("")
    report = "\n".join(L)
    print(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报表已写入 {args.out}")
    if args.badcase and bad:
        with open(args.badcase, "w", encoding="utf-8-sig") as f:
            f.write("id,query,type,gold,predicted\n")
            for q, top in bad:
                f.write(f"{q['id']},\"{q['query']}\",{q.get('type','')},"
                        f"\"{'|'.join(q['gold'])}\",\"{'|'.join(top)}\"\n")
        print(f"badcase 已写入 {args.badcase}")


if __name__ == "__main__":
    main()
