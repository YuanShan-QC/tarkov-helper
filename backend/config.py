# -*- coding: utf-8 -*-
"""
配置模块 —— 加载 .env，初始化 DeepSeek 客户端

所有模块从这里拿 LLM 客户端和 API 地址，避免重复初始化。
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 加载 backend/.env 里的环境变量
load_dotenv(Path(__file__).resolve().parent / ".env")

# DeepSeek 客户端（复用学生成绩系统同一个 key）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
)

# 模型名
MODEL = "deepseek-v4-pro"

# tarkov.dev 数据 API 基地址（免费，无需 key）
TARKOV_API_BASE = os.getenv("TARKOV_API_BASE", "https://json.tarkov.dev")
