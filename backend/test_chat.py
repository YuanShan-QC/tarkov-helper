# -*- coding: utf-8 -*-
"""Agent 对话测试 —— 验证多工具编排、别名解析、多轮记忆"""
import json
import httpx

BASE = "http://127.0.0.1:8100"


def chat(message: str):
    """发送一条消息，打印最终回复 + 工具轨迹"""
    r = httpx.post(f"{BASE}/api/chat", json={"message": message}, timeout=120)
    data = r.json()
    print("=" * 60)
    print(f"用户：{message}")
    # 打印 Agent 思考/工具轨迹
    for i, s in enumerate(data.get("steps", []), 1):
        if s.get("tool"):
            print(f"  [步骤{i}] 思考: {s['thought'][:40]}...")
            print(f"          工具: {s['tool']}")
            print(f"          观察: {s['observation'][:80]}...")
    print(f"助手：{data.get('reply', '')}")
    print()
    return data


if __name__ == "__main__":
    # 测试1：多工具编排（查价 + 比价，一句话触发多个工具）
    chat("捡了个显卡，现在多少钱，卖给商人还是挂跳蚤划算")
    # 测试2：别名 + 单工具
    chat("gpu 多少钱")
