# -*- coding: utf-8 -*-
"""工具层单元测试 —— 不依赖 LLM，直接验证数据层和工具函数"""
from tarkov_client import tarkov
from tools import tool_search_item, tool_get_price, tool_get_price_trend, tool_get_item_info, tool_compare_sell

print("正在加载数据...")
tarkov.load()
print(f"已加载 {len(tarkov.items)} 个物品\n")

print("=" * 60)
print("【测试1】搜索物品 search_item('显卡')")
print(tool_search_item({"keyword": "显卡"}))
print()

print("=" * 60)
print("【测试2】查价格 get_price('LEDX')")
print(tool_get_price({"item_name": "LEDX"}))
print()

print("=" * 60)
print("【测试3】查趋势 get_price_trend('显卡', days=7)")
print(tool_get_price_trend({"item_name": "显卡", "days": 7}))
print()

print("=" * 60)
print("【测试4】查详情 get_item_info('金链')")
print(tool_get_item_info({"item_name": "金链"}))
print()

print("=" * 60)
print("【测试5】比价 compare_sell('显卡')")
print(tool_compare_sell({"item_name": "显卡"}))
print()

print("=" * 60)
print("【测试6】别名解析 resolve_alias('gpu')")
from alias import resolve_alias
print("gpu →", resolve_alias("gpu"))
print("LEDX →", resolve_alias("ledx"))
