# -*- coding: utf-8 -*-
"""
物品别名库 —— 塔科夫玩家黑话 → 搜索关键词

tarkov.dev 的中文短名已覆盖大部分常见叫法（显卡、LEDX、金链本身就是短名），
这里主要补充英文缩写、大小写变体等短名没覆盖的黑话。

解析流程：
  用户输入 → 别名库映射 → 得到搜索关键词 → search 工具模糊匹配 → 定位物品
"""
# 玩家黑话（统一转小写）→ tarkov.dev 搜索关键词
ALIASES = {
    # 英文缩写 / 大小写变体
    "gpu": "显卡",
    "显卡": "显卡",
    "ledx": "LEDX",
    "btc": "实体比特币",
    "比特币": "实体比特币",
    "金链子": "金链",
    "金项链": "金链",
}


def resolve_alias(keyword: str) -> str:
    """
    把用户输入转成搜索关键词。
    命中别名库则返回映射后的关键词，否则原样返回（交给 search 做模糊匹配兜底）。
    """
    if not keyword:
        return keyword
    return ALIASES.get(keyword.strip().lower(), keyword)
