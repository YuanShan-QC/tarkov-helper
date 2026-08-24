# -*- coding: utf-8 -*-
"""
ReAct 自主 Agent —— 塔科夫市场查询助手

核心循环：用户输入 → THOUGHT 思考 → ACT 调用工具 → OBSERVATION 观察 → 继续或 DONE
支持多轮对话记忆（scratchpad 记录全部思考/动作/观察，供后续轮次参考）。

对比学生成绩系统的 Agent，这里去掉了写操作/确认机制（查询类工具只读），
保留：多工具编排、多轮记忆、防死循环。
"""
import json, re
from config import client, MODEL
from tools import TOOLS

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

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """执行工具并返回观察结果"""
        if tool_name in TOOLS:
            try:
                return TOOLS[tool_name](args)
            except Exception as e:
                return f"工具 {tool_name} 执行出错: {str(e)[:200]}"
        return f"错误: 未知工具 '{tool_name}'，可用: {list(TOOLS.keys())}"

    def _build_history(self) -> str:
        """构建对话历史文本，限制长度避免超出上下文窗口"""
        if not self.scratchpad:
            return ""
        lines = []
        total = 0
        # 从最新往前取，总字符数不超过 2500
        for entry in reversed(self.scratchpad):
            role = entry["role"]
            content = entry["content"]
            if role == "thought":
                line = f"THOUGHT: {content}"
            elif role == "act":
                line = f"ACT: {content}"
            elif role == "obs":
                line = f"OBSERVATION: {content[:300]}"  # 观察结果截断
            else:
                continue
            if total + len(line) > 2500:
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
        # 构建对话
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 添加历史记录（多轮记忆）
        history_text = self._build_history()
        if history_text:
            messages.append({"role": "user", "content": history_text})

        if user_input:
            messages.append({"role": "user", "content": f"用户最新输入: {user_input}"})

        messages.append({"role": "user", "content": "请输出你的下一步 THOUGHT 和 ACT（或 DONE）："})

        # 调用 LLM
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=800,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = resp.choices[0].message.content.strip()
        except Exception as e:
            return {"type": "error", "content": f"Agent 调用失败: {str(e)[:150]}"}

        # 解析 LLM 输出
        return self._parse_response(content)

    def _parse_response(self, content: str) -> dict:
        """解析 LLM 响应，提取 THOUGHT/ACT 或 DONE，并执行工具"""
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

        # 解析失败，把原始输出当最终回复
        if thought:
            self.scratchpad.append({"role": "thought", "content": thought})
        self.state = "done"
        return {"type": "done", "thought": thought, "reply": content[:200]}

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
