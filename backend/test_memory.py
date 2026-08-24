# -*- coding: utf-8 -*-
"""多轮对话记忆测试 —— 省略主语时 Agent 能否记住上下文"""
import json
import httpx

BASE = "http://127.0.0.1:8100"


def chat(message: str):
    r = httpx.post(f"{BASE}/api/chat", json={"message": message}, timeout=120)
    data = r.json()
    print("=" * 60)
    print(f"用户：{message}")
    print(f"助手：{data.get('reply', '')}")
    print()
    return data


if __name__ == "__main__":
    # 第一轮：问 LEDX 价格
    chat("LEDX 现在多少钱")
    # 第二轮：省略主语，说「那它涨了没」—— 测试 Agent 是否记得「它」= LEDX
    chat("那它48小时涨了没")
    # 第三轮：继续省略，「比价一下呢」
    chat("那卖给商人还是挂跳蚤")
