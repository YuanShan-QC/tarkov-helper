# -*- coding: utf-8 -*-
"""
ReAct 自主 Agent —— 塔科夫市场查询助手

核心循环：用户输入 → THOUGHT 思考 → ACT 调用工具 → OBSERVATION 观察 → 继续或 DONE
支持多轮对话记忆（scratchpad 记录全部思考/动作/观察，供后续轮次参考）。

对比学生成绩系统的 Agent，这里去掉了写操作/确认机制（查询类工具只读），
保留：多工具编排、多轮记忆、防死循环。
"""
import json, re, threading
from config import client, MODEL
from tools import TOOLS

# ===== 观察结果（OBSERVATION）长度预算 =====
# 历史预算按「字符」粗略折算（中文约 1 token/字），给模型留足上下文空间。
# 注意：这里是「每步工具结果」的预算，不是上下文窗口硬限制——
# DeepSeek 上下文远大于此，所以不需要抠得太紧。
# 这些值偏大是有意的：RAG 检索一次可能返回好几条任务/配方，
# 之前 300 字硬切会把「收藏家」这类长清单切成 7 项就断，
# 模型因此误判「知识库到这儿就没了」。
MAX_OBS_CHARS = 8000     # 单条观察上限：够放下一次完整 RAG 检索结果
MAX_HISTORY_CHARS = 16000  # 整个任务记录上限（约 2~4 步的完整观察）
LLM_MAX_TOKENS = 1500   # 单轮输出上限：要能列完「收藏家」这类 40+ 项的长清单

# 模型没吐出 ACT/DONE 时的重试提示（实测模型偶尔只输出一段 THOUGHT 就停了）
RETRY_INSTRUCTION = ("你刚才的输出里没有可执行的 ACT，也没有 DONE。"
                     "请严格按格式只输出下一步："
                     "要么 `ACT: 工具名({\"参数\": \"值\"})`，要么 `DONE: 给用户的最终回复`。")

# 步数用尽时的收尾提示：宁可让模型用已有观察作答，也不要把整轮结果丢掉
FINALIZE_INSTRUCTION = ("工具调用步数已用尽，不要再调用任何工具。"
                        "请立刻基于上面已有的 OBSERVATION 输出 `DONE: <给玩家的最终回答>`。"
                        "已经查到的写清楚；没查到的直接说明哪几项没查到，不要编造价格。")

# 每个工具的必填参数：模型偶尔会吐 `search_item()` 这种空参数，
# 直接执行只会得到「请提供关键词」这类无效观察，白白烧掉一步。
# 这里在调用前拦下来，并把「正确格式」回给它，让模型下一步就能自纠。
TOOL_REQUIRED = {
    "search_item": "keyword",
    "get_price": "item_name",
    "get_price_trend": "item_name",
    "get_item_info": "item_name",
    "compare_sell": "item_name",
    "rag_search": "query",
}


def _clip_observation(text: str) -> str:
    """裁剪过长的观察结果。

    尽量在分隔符（换行 / ；）处断开，避免把一个条目从中间切成两半，
    让模型误以为「数据本身到这儿就没了」。被裁剪时补明确标记，
    模型和思考过程都能看出这是展示层裁剪、而非数据缺失。
    """
    if len(text) <= MAX_OBS_CHARS:
        return text
    head = text[:MAX_OBS_CHARS]
    # 优先在换行处断开，其次在「；」处断开，都没有才硬切
    cut = max(head.rfind("\n"), head.rfind("；"))
    if cut > MAX_OBS_CHARS * 0.6:
        head = head[:cut]
    return (head
            + f"…（本条共 {len(text)} 字，此处仅保留前 {len(head)} 字。"
              "上述内容并未到结尾，请基于已给数据回答，"
              "需要剩余部分时用更精确的查询再检索一次）")


# ===== Agent 系统提示词 =====
SYSTEM_PROMPT = """你是《逃离塔科夫》（Escape from Tarkov）的市场查询助手，帮玩家查物品价格、判断涨跌、比价出手。

## 可用工具
1. search_item(keyword) — 按关键词搜索物品，返回候选列表。不确定物品确切名称时先用它
2. get_price(item_name) — 查物品当前价格（24小时均价、最低价、48小时涨跌）
3. get_price_trend(item_name, days) — 查物品最近 N 天价格趋势（默认 7 天，数据来自本地价格管道）
4. get_item_info(item_name) — 查物品详情（类型、占格、重量、基础价）
5. compare_sell(item_name) — 比价：跳蚤市场 vs 卖给商人哪个划算
6. rag_search(query) — 语义检索知识库（物品百科、任务需求、藏身处配方/制造/以物换物）。用户问「哪些任务需要X」「怎么制造Y」「Z物品有什么用/在哪换」「藏身处升级要什么材料」这类百科/任务/配方问题时，用它检索回答，不要自己瞎编

## 响应格式
每轮只能输出一个 ACT（执行一个工具）或一个 DONE（结束任务）。

调用工具：
THOUGHT: <分析当前状态，决定下一步做什么、为什么>
ACT: tool_name({"arg": "value"})

结束任务：
THOUGHT: <总结任务已完成>
DONE: <给用户的最终回复，自然语言>

## 重要规则
- 每轮只有一个 ACT 或一个 DONE
- 玩家常用黑话/简称，例如「显卡」=显示卡、「LEDX」=LEDX皮肤透照仪、「金链」=金项链。不确定物品名时先用 search_item 搜索确认，不要乱猜
- 多轮对话要记住上下文：用户先说「查 LEDX 价格」，再说「那它涨了没」，你要知道「它」指的是 LEDX
- 价格单位是卢布，回答用简洁自然的中文，直接给结论
- 一个简单问题尽量一步查完，不要反复调用工具浪费轮次
- 涉及判断（涨还是跌、卖哪边划算）时，基于工具返回的真实数据下结论，不要凭空猜测
- 任务需求、藏身处配方、制造/以物换物、物品用途这类百科类问题，必须用 rag_search 检索知识库回答，不要凭空编造"""



class ReActAgent:
    """塔科夫查询 Agent —— 维护对话状态，执行工具调用循环"""

    def __init__(self):
        self.scratchpad = []   # [{"role": "thought/act/obs/done", "content": "..."}]
        self.state = "thinking"  # thinking | done
        self._last_user_input = None  # 本轮用户输入，供解析失败重试时复用
        self._last_act = None   # (工具名, 参数) 上一次动作，用于识别原地打转
        # 一个会话同一时刻只处理一轮对话：否则双击发送 / 两个标签页会把
        # 彼此的 scratchpad 搅在一起，答案张冠李戴。
        self._turn_lock = threading.RLock()

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """执行工具并返回观察结果（先做参数校验与重复调用提醒）"""
        if tool_name not in TOOLS:
            return f"错误: 未知工具 '{tool_name}'，可用: {list(TOOLS.keys())}"

        required = TOOL_REQUIRED.get(tool_name)
        if required and not str(args.get(required, "")).strip():
            return (f"参数错误：{tool_name} 必须传入 {required}，你传的是 "
                    f"{json.dumps(args, ensure_ascii=False)}。"
                    f"正确写法示例：ACT: {tool_name}({{\"{required}\": \"物品名\"}})")

        signature = (tool_name, json.dumps(args, ensure_ascii=False, sort_keys=True))
        repeat_note = ""
        if signature == self._last_act:
            repeat_note = ("⚠️ 这一步和上一步调用的工具与参数完全相同，结果不会变化。"
                           "请换一个参数，或直接用已有信息 DONE 回答。\n")
        self._last_act = signature

        try:
            return repeat_note + TOOLS[tool_name](args)
        except Exception as e:
            return repeat_note + f"工具 {tool_name} 执行出错: {str(e)[:200]}"

    def _build_history(self) -> str:
        """构建对话历史文本，限制长度避免超出上下文窗口"""
        if not self.scratchpad:
            return ""
        lines = []
        total = 0
        # 从最新往前取，总字符数不超过 MAX_HISTORY_CHARS
        for entry in reversed(self.scratchpad):
            role = entry["role"]
            content = entry["content"]
            if role == "thought":
                line = f"THOUGHT: {content}"
            elif role == "act":
                line = f"ACT: {content}"
            elif role == "obs":
                line = f"OBSERVATION: {_clip_observation(content)}"
            elif role == "done":
                # 自己上一轮说过的话必须进历史：否则用户追问「你刚才怎么说的」，
                # 模型只能靠 OBSERVATION 反推，还会把反推结果当成「原话」，属于幻觉。
                line = f"DONE: {content[:400]}"
            else:
                continue
            if total + len(line) > MAX_HISTORY_CHARS:
                break
            lines.insert(0, line)
            total += len(line)
        if not lines:
            return ""
        # 清理 scratchpad，只保留最近使用的条目
        keep = min(len(lines) * 2, len(self.scratchpad))
        self.scratchpad = self.scratchpad[-keep:]
        return "## 任务记录\n" + "\n".join(lines)

    def step(self, user_input: str = None) -> dict:
        """
        执行一步 Agent 循环。返回:
          {"type": "thinking", "content": "..."}
          {"type": "acting", "tool": "...", "args": {...}}
          {"type": "observation", "content": "..."}
          {"type": "done", "reply": "..."}
          {"type": "error", "content": "..."}
        """
        # 调用 LLM（消息拼装统一放在 _ask_llm，重试路径也复用它）
        self._last_user_input = user_input
        try:
            content = self._ask_llm(user_input=user_input)
        except Exception as e:
            return {"type": "error", "content": f"Agent 调用失败: {str(e)[:150]}"}

        # 解析 LLM 输出
        return self._parse_response(content)

    def _ask_llm(self, user_input: str = None, extra_instruction: str = None) -> str:
        """按当前历史向 LLM 要下一步输出（THOUGHT + ACT，或 DONE）。

        user_input：本轮用户输入（仅第一步传，后续步骤靠 scratchpad 记忆）。
        extra_instruction：附加在末尾的追问指令（例如解析失败后的重试提示）。
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 添加历史记录（多轮记忆）
        history_text = self._build_history()
        if history_text:
            messages.append({"role": "user", "content": history_text})

        if user_input:
            messages.append({"role": "user", "content": f"用户最新输入: {user_input}"})

        directive = extra_instruction or "请输出你的下一步 THOUGHT 和 ACT（或 DONE）："
        messages.append({"role": "user", "content": directive})

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.3,   # 工具调用要稳：默认 1.0 时同一句话时而漏参数、时而绕远路
            extra_body={"thinking": {"type": "disabled"}},
        )
        return resp.choices[0].message.content.strip()

    def _parse_response(self, content: str, allow_retry: bool = True) -> dict:
        """解析 LLM 响应，提取 THOUGHT/ACT 或 DONE，并执行工具

        allow_retry：既没解析出 ACT 也没解析出 DONE 时，允许再问模型一次。
        模型偶尔只吐一段 THOUGHT 就停了（没给 ACT 也没给 DONE），
        这时既不该把内部推理当成回复丢给用户，也不该就此结束整轮对话。
        """
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=\nACT:|\nDONE:|$)', content, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        # 提取 ACT（括号深度匹配，避免 JSON 内部括号干扰）
        act_match = re.search(r'ACT:\s*(\w+)\(', content)
        tool_name = None
        args = {}
        if act_match:
            tool_name = act_match.group(1).strip()
            start = act_match.end()
            depth = 1
            end = start
            while end < len(content) and depth > 0:
                ch = content[end]
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                end += 1
            args_str = content[start:end - 1].strip()
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = self._parse_simple_args(args_str)

        done_match = re.search(r'DONE:\s*(.+)', content, re.DOTALL)

        if tool_name:
            # 记录思考与动作
            if thought:
                self.scratchpad.append({"role": "thought", "content": thought})
            act_desc = f"{tool_name}({json.dumps(args, ensure_ascii=False)})"
            self.scratchpad.append({"role": "act", "content": act_desc})

            # 执行工具，得到观察结果
            observation = self._execute_tool(tool_name, args)
            self.scratchpad.append({"role": "obs", "content": observation})

            return {"type": "observation", "thought": thought,
                    "tool": tool_name, "args": args, "content": observation}

        if done_match:
            if thought:
                self.scratchpad.append({"role": "thought", "content": thought})
            reply = done_match.group(1).strip()
            self.scratchpad.append({"role": "done", "content": reply})
            self.state = "done"
            return {"type": "done", "thought": thought, "reply": reply}

        # 既没有 ACT 也没有 DONE：模型可能只吐了半截（常见于只输出一段 THOUGHT）
        if allow_retry:
            retry = self._ask_llm(user_input=self._last_user_input,
                                  extra_instruction=RETRY_INSTRUCTION)
            if retry:
                # 再解析一次；这次不允许再重试，避免无限循环
                return self._parse_response(retry, allow_retry=False)

        # 重试后仍解析不出来，降级：把原文当回复，但必须剥掉内部 THOUGHT 标记，
        # 否则模型的内心独白会原样出现在用户的聊天框里。
        if thought:
            self.scratchpad.append({"role": "thought", "content": thought})
        self.state = "done"
        reply = re.sub(r'^\s*THOUGHT:\s*', '', content).strip()
        if not reply:
            reply = "抱歉，我没能给出有效回答，请重试或换个说法问一次。"
        return {"type": "done", "thought": thought, "reply": reply[:200]}

    def _parse_simple_args(self, args_str: str) -> dict:
        """简易参数解析（JSON 解析失败时的后备）"""
        result = {}
        for key in ["keyword", "item_name", "days", "name"]:
            m = re.search(r'["\']' + key + r'["\']\s*:\s*["\'](.+?)["\']', args_str, re.DOTALL)
            if m and key not in result:
                result[key] = m.group(1)
            m = re.search(r'["\']' + key + r'["\']\s*:\s*(\d+)', args_str)
            if m and key not in result:
                result[key] = int(m.group(1))
        if not result and args_str.strip():
            return {"keyword": args_str.strip()}
        return result

    def reset(self):
        """重置 Agent 状态（开启新会话）"""
        self.scratchpad = []
        self.state = "thinking"
        self._last_user_input = None
        self._last_act = None

    def force_finalize(self) -> str:
        """步数用尽时，用已有观察逼模型给一个最终回答。

        以前的做法是直接返回「换个更具体的问题试试」，等于把已经跑完的
        7~8 次工具调用的结果全扔掉；用户问 8 个物品，最后拿到一句道歉。
        这里改成再问一次模型，让它把已经查到的写清楚、没查到的说明白。
        """
        try:
            content = self._ask_llm(user_input=self._last_user_input,
                                    extra_instruction=FINALIZE_INSTRUCTION)
        except Exception as e:
            return f"服务异常：{str(e)[:150]}"

        done_match = re.search(r'DONE:\s*(.+)', content, re.DOTALL)
        reply = done_match.group(1).strip() if done_match else content.strip()
        reply = re.sub(r'^\s*THOUGHT:.*?(?=\n|$)', '', reply, flags=re.DOTALL).strip()
        if not reply:
            return "抱歉，这个问题涉及的信息有点多，我一时没查完，换个更具体的问题试试？"

        self.scratchpad.append({"role": "done", "content": reply})
        self.state = "done"
        return reply
