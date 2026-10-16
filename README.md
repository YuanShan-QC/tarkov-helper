# 塔科夫市场查询助手（Tarkov Helper）

基于 **ReAct Agent** 的《逃离塔科夫》市场智能查询助手。用自然语言问物品价格、涨跌趋势、出手比价，Agent 自主编排多个工具完成查询。

## 核心亮点（技术视角）

| 能力 | 实现 |
|------|------|
| 多工具编排 | ReAct 循环，Agent 自主决定调用哪些工具、调用几步（搜索 → 查价 → 比价） |
| 别名解析 | 玩家黑话「显卡 / GPU / LEDX」→ 自动映射到游戏物品「显示卡 / LEDX皮肤透照仪」 |
| 多轮记忆 | scratchpad 记录对话状态，省略主语也能接上（「那它涨了没」→ 知道「它」指 LEDX） |
| 实时数据 | 接入 tarkov.dev 免费 REST API（5312 个物品 + 实时价格 + 价格历史） |
| RAG 知识库 | 物品百科 / 任务需求 / 藏身处配方 6858 条文档向量化，语义检索「哪些任务要 LEDX」 |
| 本地价格管道 | 定时抓取价格存 SQLite，自攒 7/30 天趋势，前端 ECharts 折线图 |

## 技术栈

- **后端**：FastAPI + ReAct Agent（自主规划 + 工具调用）
- **LLM**：DeepSeek（`deepseek-v4-pro`）
- **向量检索**：fastembed（bge-small-zh-v1.5，ONNX 本地推理，无需 GPU/联网）
- **存储**：SQLite（价格历史快照）+ numpy 向量缓存
- **数据源**：tarkov.dev（免费公开 API，含实时价格、历史趋势、任务、藏身处配方）

## 可用工具（Agent 可调度的 6 个工具）

1. `search_item` —— 按关键词搜索物品
2. `get_price` —— 查当前价格（24h 均价、最低价、48h 涨跌）
3. `get_price_trend` —— 查最近 N 天价格趋势（数据来自本地 SQLite 价格管道）
4. `get_item_info` —— 查物品详情（类型、占格、重量）
5. `compare_sell` —— 比价：跳蚤市场 vs 卖给商人
6. `rag_search` —— 语义检索知识库（任务需求 / 藏身处配方 / 物品百科）

## 项目结构

```
backend/
├── run.py            # 启动入口（Windows 上先切 SelectorEventLoop，再拉起 uvicorn）
├── main.py           # FastAPI 入口（/api/chat 对话、/api/chat/stream 流式、/api/price/trend 趋势）
├── config.py         # 配置（DeepSeek 客户端 + API 地址）
├── tarkov_client.py  # tarkov.dev 数据层（物品/任务/配方/藏身处 + 中文名解析）
├── alias.py          # 物品别名库（玩家黑话 → 搜索关键词）
├── tools.py          # Agent 工具集（6 个查询工具）
├── agent.py          # ReAct Agent 核心（多工具编排 + 多轮记忆）
├── knowledge_base.py # RAG 向量知识库（fastembed 向量化 + 余弦检索 + 缓存）
├── price_store.py    # SQLite 价格管道（定时快照 + 历史导入 + 趋势查询）
├── models/           # embedding 模型缓存（bge-small-zh，本地化免下载）
├── data/             # prices.db（价格） + kb_cache（向量缓存）
└── static/index.html # 前端聊天页（SSE 流式 + ECharts 折线图）
```

## 快速开始

```bash
cd backend
pip install -r requirements.txt
# 配置 backend/.env（DeepSeek API key）
python run.py                      # 推荐：等价于 uvicorn main:app --host 127.0.0.1 --port 8100
# 首次启动会自动：下载 embedding 模型（或复用 models/ 本地缓存）+ 后台构建向量库
```

> **Windows 必读：请用 `python run.py` 启动，不要直接用 `python -m uvicorn main:app`。**
> Windows 上 asyncio 默认的 **ProactorEventLoop** 会让 uvicorn 的 `accept()` 抛
> `OSError: [WinError 10014] 系统检测到在一个调用中尝试使用指针参数时的无效指针地址`，
> 监听套接字随即失效，浏览器表现为「无法访问此网站 / 一直转圈」。
> uvicorn 只在 `--reload` / `--workers`（`use_subprocess`）时才切到 SelectorEventLoop，
> 单进程启动会保留 Proactor。`run.py` 在创建事件循环前显式设置
> `WindowsSelectorEventLoopPolicy` 从根上规避该问题；
> `main.py` 顶部也加了同样的兜底，直接跑 uvicorn 同样安全。
> （桌面一键启动 `start.bat` 已同步改为调用 `run.py`。）

可选配置（backend/.env）：

```bash
TARKOV_MODE=pvp            # 价格数据模式：pvp（regular 经典模式，默认）或 pve
TARKOV_ITEMS_TTL=21600     # 物品/价格缓存有效期（秒），默认 6 小时
TARKOV_KNOWLEDGE_TTL=86400 # 任务/配方等知识数据缓存有效期（秒），默认 24 小时
TARKOV_REFRESH_INTERVAL=1800  # 后台联网刷新检查间隔（秒），默认 30 分钟
```

> 说明：tarkov.dev 提供 PvP（/regular/*）与 PvE（/pve/*）两套独立价格经济，
> 价格差异很大（如显卡 PvP 约 32 万 vs PvE 约 78 万卢布）。
> 两种模式的本地缓存分开存放，切换模式互不影响；切换后首次启动会联网拉取对应模式的数据。
> 页面右上角也有 **PvP/PvE 一键切换按钮**（POST /api/mode），无需改配置、无需重启。

接口：

```bash
POST /api/chat           # {"message": "捡了个显卡，卖给商人还是挂跳蚤划算"}
POST /api/chat/stream    # 流式对话（SSE，含工具调用轨迹 + chart 图表数据）
GET  /api/price/trend/{item_id}?days=30   # 本地价格趋势（折线图数据源）
GET  /api/items          # 全部物品（按分类分组，价格栏"全部"视图）
POST /api/reset          # 重置会话
POST /api/mode           # 运行时切换 PvP/PvE 价格模式 {"mode":"pve"}
```

## 对话示例

```
用户：捡了个显卡，现在多少钱，卖给商人还是挂跳蚤划算
助手：显卡（显示卡）24 小时均价约 30 万卢布，48 小时基本持平。
      建议挂跳蚤市场，能卖约 30 万，比卖给商人多赚近 20 万。

用户：那 LEDX 呢
助手：LEDX 现在约 58.3 万，但 48 小时跌了 11.7%，在掉价，建议观望。
```
