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

# ===== 游戏模式（价格数据源）=====
# tarkov.dev 提供两套独立经济：regular（PvP 经典模式，默认）与 pve（PvE 模式）。
# 可通过 .env 设置 TARKOV_MODE=pvp 或 pve，也可在页面右上角一键切换（POST /api/mode）。
GAME_MODE = os.getenv("TARKOV_MODE", "pvp").strip().lower()
if GAME_MODE not in ("pvp", "pve"):
    print(f"[配置] 未知 TARKOV_MODE={GAME_MODE!r}，回退为 pvp")
    GAME_MODE = "pvp"
# 端点前缀：/regular/* = PvP，/pve/* = PvE
TARKOV_MODE_PREFIX = "/pve" if GAME_MODE == "pve" else "/regular"


def set_game_mode(mode: str):
    """运行时切换游戏模式（更新模块级变量；其他模块需通过 config.XXX 动态读取）"""
    global GAME_MODE, TARKOV_MODE_PREFIX
    GAME_MODE = mode
    TARKOV_MODE_PREFIX = "/pve" if mode == "pve" else "/regular"
