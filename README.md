# 塔科夫市场 AI 查询助手

用自然语言问《逃离塔科夫》的物品价格、涨跌趋势、卖给商人还是挂跳蚤——**RAG 检索 + 手写 ReAct 编排（不依赖 Agent 框架）**，把「翻 Wiki → 查行情 → 比价」三步压缩成一次提问。

| 关键数字 | |
|---|---|
| 检索命中率（自建 260 条评测集，Hit@5） | 50.4% → **95.4%** |
| 单次问答 | 平均 **2.33 步 / 8.9 秒**（12 条真实问句回归，12/12 有效） |
| 知识库 | 7075 条文档向量化，本地 ONNX 推理，冷启动免重建 |
| 压测 | 定位 8 个问题，修 6 处 + 补 2 处设计，`pytest` 37 passed |

> 个人项目，代码按「能跑通、能验证」的标准写，工程规范上还有不少欠缺，欢迎指正。

## 核心亮点（技术视角）

| 能力 | 实现 |
|------|------|
| 多工具编排 | ReAct 循环，Agent 自主决定调用哪些工具、调用几步（搜索 → 查价 → 比价） |
| 别名解析 | 玩家黑话「显卡 / GPU / LEDX」→ 自动映射到游戏物品「显示卡 / LEDX皮肤透照仪」 |
| 多轮记忆 | scratchpad 记录对话状态，省略主语也能接上（「那它涨了没」→ 知道「它」指 LEDX） |
| 实时数据 | 接入 tarkov.dev 免费 REST API（5480 个物品 + 实时价格 + 价格历史） |
| RAG 知识库 | 物品百科 / 任务需求 / 藏身处配方 7075 条文档向量化，语义检索「哪些任务要 LEDX」 |
| 本地价格管道 | 定时抓取价格存 SQLite，自攒 7/30 天趋势，前端 ECharts 折线图 |

## 评测：把「好像还挺准」变成数字

检索层做得对不对，凭感觉判断不了，所以自建了 260 条评测集，用 **Hit@5 / MRR** 度量：

- **自动集 240 条**：从 5480 个物品里随机抽 60 个，每个生成四种问法（全名+多少钱 / 短名+价格 / 英文名+什么价 / 全名+值不值）。
- **人工集 20 条**：专门设计难例——英文缩写（`GPU`、`LEDX`）、玩家黑话（`显卡`、`油桶`）、模糊描述（`现在什么最值钱`）、比价（`比特币怎么卖最值钱`）、缺信息（该反问而不是猜）。

| 集合 | 修复前 | 修复后 |
|---|---|---|
| 自动集 240 条 | 50.4% | **95.4%** |
| 人工集 14 条（有唯一答案的） | 78.6% | 92.9% |
| 自动集剔除重名物品 | 53.5% | **100%** |

**人工集反而「显得更好」，这正是小样本会骗人的地方**：早期人工集里 `LEDX` 是靠英文名 `ledx-skin-transilluminator` 兜住才命中的，掩盖了中文名检索的窟窿；换成 240 条随机样本，问题立刻暴露成 50.4%。

定位并修掉的两个真问题：

1. **大小写不一致**：5480 个物品里 **4605 个（84%）** 的中文译名含大写拉丁字母（`LEDX皮肤透照仪`、`M4A1`、`GPU`），而检索侧把关键词转小写后没有同步转译名。结果「LEDX 现在什么价」要绕 6 步 80 秒才命中，修完同样的问句是 **2 步 11.5 秒**。
2. **短名被长名挤掉**：候选没有排序，搜「显卡」时含该子串的长名排在真身前。改为按匹配度分档排序，命中率 94.2% → 95.4%。

剩下的 11 条 miss 全部是**同名物品**——5480 个物品里有 2105 个（38%）的中文全名或短名被两个以上物品共用（32 个完全同名，32 个都叫「海报」，8 个都叫「EXFIL」）。这类在名字层面无法区分，正确处理是**反问用户**而不是猜。

> 评测脚本与结果都在 [`eval/`](eval/)，可复现：`python eval/export_preds.py` 生成预测，`python eval/run_eval.py` 出指标。

## 稳定性：黑盒压测与护栏

Agent 这类"看起来能跑"的程序，问题大多藏在并发、边界和统计口径里。本项目做过一轮黑盒压测
（HTTP 探测脚本 + 进程内 trace 抓模型原文与工具参数 + 直接查 SQLite 核对数字），定位 8 个问题，下面 6 处已经修掉：

| 问题 | 现象 | 处理 |
|------|------|------|
| 并发串答案 | Agent 是模块级单例，共享一份 scratchpad，3 条并发实测 **2/6 答错人** | 按 `session_id` 的 LRU 会话池（上限 32）+ 每会话一把锁，前端用 `sessionStorage` 生成 tab 级 id |
| 步数用尽丢结果 | 8 步上限一到直接回"抱歉，信息有点多"，已查到的价格全扔 | `force_finalize()`：基于已有观察收尾，查不到的明说 |
| 工具空参数空转 | 模型偶发吐 `search_item()`，连烧 3 步换回"请提供关键词" | 必填参数校验 + 回"正确写法示例"让模型一步自纠；`temperature` 1.0 → 0.3 |
| 多轮"原话"失真 | 追问"你刚才怎么说的"，丢掉具体阈值却仍称"原话" | `DONE:` 结果写回历史，模型能看到自己上一轮说过什么 |
| 稀疏数据照样给"7 天趋势" | 某物品 7 天窗口只有 4 个采样点，也输出"基本持平" | 覆盖度闸门：点数 < 10 **或** 实际覆盖 < 窗口 60% → 降级为"样本不足，只给当前水平" |
| 极端挂单污染行情 | 中位数 59.7 万，却因一条 344 万的孤立点说"最高冲到 344 万" | 趋势改中位数 + P10~P90 + 首尾各 3 点中位数算涨跌，异常点统计后声明已排除 |

另外加了领域围栏：与市场行情无关的请求（写诗、写代码、闲聊）一句话婉拒并给出可做的示例。

回归结果：12 条真实问句 **12/12 有效**，平均 **2.33 步 / 8.9 秒**，其中 10 条 2 步内完成；`pytest` 37 passed。

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
POST /api/chat           # {"message": "捡了个显卡，卖给商人还是挂跳蚤划算", "session_id": "tab-xxx"}（session_id 可省，默认 default）
POST /api/chat/stream    # 流式对话（SSE，含工具调用轨迹 + chart 图表数据）
GET  /api/price/trend/{item_id}?days=30   # 本地价格趋势（折线图数据源）
GET  /api/items          # 全部物品（按分类分组，价格栏"全部"视图）
POST /api/reset          # 重置会话
POST /api/mode           # 运行时切换 PvP/PvE 价格模式 {"mode":"pve"}
```

### 价格历史是自动补全的

定时快照只会在程序运行期间写点，所以光靠快照攒，趋势图永远只有「启动那几天」。

现在改为**按需回补**：第一次点开某物品的趋势图（或 Agent 调用 `get_price_trend`）时，
如果本地数据在窗口内不够密，会立刻从 tarkov.dev 拉该物品的完整历史（约 1800 个点）写入本地库，
点开就有完整曲线，不用等。

- 去重：同一物品同一天最多联网一次，导入成功后落 `price_import` 表标记，之后不再重复下载；
  上游明确没有历史的物品（如卢布，接口 404）也会被标记，不会天天重试。
- 想一次性把整个市场补全、而不是等用户逐个点开：

```bash
cd backend
.venv\Scripts\python.exe backfill_prices.py --dry-run    # 先看有多少物品缺历史
.venv\Scripts\python.exe backfill_prices.py --limit 200  # 先补 200 个试试
.venv\Scripts\python.exe backfill_prices.py              # 全部补齐（~3500 个物品，约 10 分钟）
```

## 对话示例

```
用户：捡了个显卡，现在多少钱，卖给商人还是挂跳蚤划算
助手：显卡（显示卡）24 小时均价约 30 万卢布，48 小时基本持平。
      建议挂跳蚤市场，能卖约 30 万，比卖给商人多赚近 20 万。

用户：那 LEDX 呢
助手：LEDX 现在约 58.3 万，但 48 小时跌了 11.7%，在掉价，建议观望。
```
