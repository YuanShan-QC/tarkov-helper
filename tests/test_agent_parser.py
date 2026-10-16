"""
tarkov-helper 的 ReAct Agent 解析器与工具层测试 —— 不需要 API Key、不联网。

测试重点（也是面试会被追问的点）：
  · 为什么不能用 `json.loads` 直接解模型输出 → 括号深度匹配的必要性
  · 解析失败的兜底路径有几种，各自退化成什么行为
  · 工具执行抛异常时，Agent 是崩掉还是把错误当成 OBSERVATION 继续（答案：继续）

运行：
    cd C:\\Users\\Lenovo\\DSH
    pytest tarkov-helper -v
"""

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


@pytest.fixture
def agent_cls():
    """
    加载 tarkov-helper 的 ReActAgent 类。

    为什么用 importlib 而不是 `from agent import ReActAgent`：
      两个项目各有一个叫 agent 的模块，谁先进 sys.modules 谁就赢，
      直接 import 会串味。按文件路径加载最稳。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tarkov_agent_module", BACKEND / "agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReActAgent


@pytest.fixture
def agent(agent_cls):
    return agent_cls()


# ============================================================================
# 一、解析器：ACT 提取
# ============================================================================

class TestParseAct:
    def test_标准单行_ACT(self, agent):
        result = agent._parse_response(
            'THOUGHT: 用户要查显卡价格，先搜一下\nACT: search_item({"keyword": "显卡"})'
        )
        assert result["type"] == "observation"
        assert result["tool"] == "search_item"
        assert result["args"] == {"keyword": "显卡"}
        assert result["thought"] == "用户要查显卡价格，先搜一下"

    def test_参数里含右括号时括号深度匹配仍然正确(self, agent):
        """这是整个解析器存在的理由：JSON 字符串里的 ) 会骗过正则的贪婪匹配"""
        result = agent._parse_response(
            'THOUGHT: 查询带括号的名字\n'
            'ACT: search_item({"keyword": "AK-74 (5.45mm)"})'
        )
        assert result["tool"] == "search_item"
        assert result["args"] == {"keyword": "AK-74 (5.45mm)"}

    def test_数字参数会被解析成_int_而不是字符串(self, agent):
        result = agent._parse_response(
            'THOUGHT: 看趋势\nACT: get_price_trend({"item_name": "LEDX", "days": 7})'
        )
        assert result["args"] == {"item_name": "LEDX", "days": 7}
        assert isinstance(result["args"]["days"], int)

    def test_JSON_破损时降级到正则兜底解析(self, agent):
        """模型偶尔会漏引号：{"keyword": 显卡} —— 这时不能整轮失败"""
        result = agent._parse_response(
            'THOUGHT: 兜底\nACT: search_item({"keyword": 显卡})'
        )
        assert result["tool"] == "search_item"

    def test_JSON_完全解析不了时把整段当成_keyword(self, agent):
        result = agent._parse_response('THOUGHT: 兜底\nACT: search_item(显卡)')
        assert result["tool"] == "search_item"
        assert result["args"] == {"keyword": "显卡"}

    def test_调用不存在的工具会返回可读错误而不是抛异常(self, agent):
        result = agent._parse_response('THOUGHT: 乱调\nACT: no_such_tool({"a": 1})')
        assert result["type"] == "observation"
        assert "未知工具" in result["content"]


# ============================================================================
# 二、解析器：DONE 与降级
# ============================================================================

class TestParseDone:
    def test_正常_DONE(self, agent):
        result = agent._parse_response("THOUGHT: 查完了\nDONE: 显卡现在 52 万卢布，比昨天涨了 3%")
        assert result["type"] == "done"
        assert "52 万卢布" in result["reply"]
        assert agent.state == "done"

    def test_既没有_ACT_也没有_DONE_时整段原文当回复(self, agent):
        """降级策略：宁可回一段不完美的原文，也不让用户看到报错

        allow_retry=False —— 这里只测「重试后仍解析不出」的最终降级形态，
        免得测试真的去调一次 LLM。重试行为另有专门测试。
        """
        result = agent._parse_response("我直接回答你吧：显卡在涨价", allow_retry=False)
        assert result["type"] == "done"
        assert result["reply"].startswith("我直接回答你吧")

    def test_回复过长时截断到_200_字(self, agent):
        result = agent._parse_response("DONE: " + "很长的回答" * 100)
        assert len(result["reply"]) <= 200 + len("很长的回答") * 100  # DONE 分支不截断
        result2 = agent._parse_response("没有标记的超长输出" * 100, allow_retry=False)
        assert len(result2["reply"]) == 200

    def test_只输出_THOUGHT_时不能把内部推理泄漏给用户(self, agent):
        """回归：模型偶尔只吐一段 THOUGHT 就停了。

        修之前会把 "THOUGHT: ..." 原文当成回复显示在聊天框里（内心独白泄漏），
        而且整轮对话就此结束。现在先重试；重试仍失败才降级，且必须剥掉标记。
        """
        thinking_only = "THOUGHT: 用户问的是「收藏家任务」，我需要先检索知识库确认需求物品。"
        result = agent._parse_response(thinking_only, allow_retry=False)
        assert result["type"] == "done"
        assert not result["reply"].startswith("THOUGHT:")  # 标记必须剥掉
        assert "收藏家任务" in result["reply"]              # 内容保留，不是空回复

    def test_没有_ACT_DONE_时会重试一次(self, agent):
        """重试时用的是本轮用户输入，且重试得到的输出正常参与解析。"""
        calls = []

        def fake_ask(user_input=None, extra_instruction=None):
            calls.append((user_input, extra_instruction))
            return 'THOUGHT: 查一下\nACT: search_item({"keyword": "显卡"})'

        agent._ask_llm = fake_ask
        agent._last_user_input = "显卡多少钱"
        result = agent._parse_response("THOUGHT: 我先想想。")

        assert len(calls) == 1, "只重试一次，这一次就拿到了可用输出"
        assert calls[0][0] == "显卡多少钱", "重试必须带上本轮用户输入"
        assert calls[0][1], "重试必须带额外格式指令"
        assert result["type"] == "observation"
        assert result["tool"] == "search_item"

    def test_重试仍失败时不会无限重试(self, agent):
        """第二次还是只有 THOUGHT —— 必须收尾，不能一直问下去。"""
        calls = []

        def fake_ask(user_input=None, extra_instruction=None):
            calls.append(extra_instruction)
            return "THOUGHT: 我还是只想了个开头。"

        agent._ask_llm = fake_ask
        result = agent._parse_response("THOUGHT: 我还是只想了个开头。")

        assert len(calls) == 1, "重试最多一次"
        assert result["type"] == "done"
        assert not result["reply"].startswith("THOUGHT:")


# ============================================================================
# 三、scratchpad 记忆与裁剪
# ============================================================================

class TestScratchpad:
    def test_每轮把_thought_act_obs_都记进任务记录(self, agent):
        agent._parse_response('THOUGHT: 第一步\nACT: search_item({"keyword": "显卡"})')
        entries = [e["role"] for e in agent.scratchpad]
        assert entries == ["thought", "act", "obs"]

    def test_历史文本里带_THOUGHT_和_OBSERVATION_标记(self, agent):
        agent._parse_response('THOUGHT: 第一步\nACT: search_item({"keyword": "显卡"})')
        history = agent._build_history()
        assert "THOUGHT: 第一步" in history
        assert "OBSERVATION:" in history

    def test_历史文本有长度上限不会无限增长(self, agent):
        from agent import MAX_HISTORY_CHARS

        for i in range(60):
            agent.scratchpad.append({"role": "thought", "content": f"思考{i}" * 100})
        history = agent._build_history()
        # 上限 MAX_HISTORY_CHARS 字符（加上 "## 任务记录\n" 前缀和换行的余量）
        assert len(history) < MAX_HISTORY_CHARS + 100
        # 关键：60 条不可能全塞进去，必须真的发生了裁剪
        assert len(history) < sum(len(e["content"]) for e in agent.scratchpad)

    def test_长观察结果不会被切成半截_且带截断标记(self, agent):
        """回归：RAG 长结果曾被 content[:300] 硬切，模型误以为数据到结尾就没了。

        「收藏家」任务有 44 项要上交的物品，300 字只能放下 7 项，
        导致回答里出现「以及还有其他物品（知识库该条信息末尾被截断）」。
        """
        from agent import MAX_OBS_CHARS, _clip_observation

        # 常规长度（一次完整 RAG 检索结果）必须原样保留
        normal = "任务「收藏家」需求：" + "、".join(f"物品{i}" for i in range(44))
        assert len(normal) < MAX_OBS_CHARS
        assert _clip_observation(normal) == normal

        # 超长时必须带明确标记，别让模型以为是数据本身到结尾了
        clipped = _clip_observation("X" * (MAX_OBS_CHARS * 2))
        assert "仅保留前" in clipped
        assert len(clipped) < MAX_OBS_CHARS * 2

    def test_reset_清空状态(self, agent):
        agent._parse_response('THOUGHT: 干活\nACT: search_item({"keyword": "显卡"})')
        agent.reset()
        assert agent.scratchpad == []
        assert agent.state == "thinking"


# ============================================================================
# 四、工具层
# ============================================================================

class TestTools:
    def test_六个工具都注册在_TOOLS_表里(self):
        from tools import TOOLS

        assert set(TOOLS.keys()) == {
            "search_item", "get_price", "get_price_trend",
            "get_item_info", "compare_sell", "rag_search",
        }

    def test_工具抛异常时被捕获成观察结果_Agent_不会崩(self, agent):
        """工具报错 = 一条 OBSERVATION，模型可以据此换策略，而不是整个请求 500"""
        import tools as tools_module

        original = tools_module.TOOLS["search_item"]
        try:
            def boom(args):
                raise RuntimeError("上游 API 超时")

            tools_module.TOOLS["search_item"] = boom
            result = agent._parse_response('THOUGHT: 试一下\nACT: search_item({"keyword": "显卡"})')
            assert result["type"] == "observation"
            assert "执行出错" in result["content"]
        finally:
            tools_module.TOOLS["search_item"] = original

    def test_异常信息会被截断_不会把堆栈灌进上下文(self, agent):
        import tools as tools_module

        original = tools_module.TOOLS["search_item"]
        try:
            def boom(args):
                raise RuntimeError("X" * 5000)

            tools_module.TOOLS["search_item"] = boom
            result = agent._parse_response('THOUGHT: 试\nACT: search_item({"keyword": "a"})')
            # str(e)[:200] —— 200 字符上限，防止一次异常吃掉整个上下文窗口
            assert len(result["content"]) < 300
        finally:
            tools_module.TOOLS["search_item"] = original


# ============================================================================
# 五、知识库检索的词法加权（不联网、不加载向量，只测打分逻辑）
# ============================================================================

class TestLexicalBoost:
    """回归：短查询/英文名查询曾让目标文档掉出 top-k。

    用户反馈「搜索收藏家显示知识库没有这一项」——因为「收藏家」三个字
    的语义排名在 1400 名开外，纯向量排序根本捞不回来。
    纯函数测试，不需要向量库。
    """

    @pytest.fixture
    def kbmod(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "tarkov_kb_module", BACKEND / "knowledge_base.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_标题被查询包含时给满分(self, kbmod):
        meta = {"type": "task", "name": "收藏家", "normalizedName": "collector"}
        for q in ["收藏家", "收藏家任务", "那收藏家任务呢", "Collector 任务", "collector"]:
            assert kbmod._title_lexical_score(q, meta) >= 0.9, q

    def test_泛化疑问词不该给目标文档加权(self, kbmod):
        """「显卡有什么用」曾因为和「U盘里有什么？」共享「有什么」而误加权。"""
        meta = {"type": "task", "name": "U盘里有什么？"}
        assert kbmod._title_lexical_score("显卡有什么用", meta) < 0.2

    def test_无关查询不给加权(self, kbmod):
        meta = {"type": "task", "name": "收藏家"}
        assert kbmod._title_lexical_score("显卡现在多少钱", meta) == 0.0
        assert kbmod._title_lexical_score("", meta) == 0.0

    def test_短标题不作为强制命中信号(self, kbmod):
        """标题太短（如 2 字）不能当「被点名」处理，否则会误伤一大片。"""
        assert kbmod._substring_eligible("显卡") is False     # 中文 2 字，不够
        assert kbmod._substring_eligible("收藏家") is True    # 中文 3 字，够
        assert kbmod._substring_eligible("ledx") is False     # 英文 4 字符，不够
        assert kbmod._substring_eligible("collector") is True  # 英文 9 字符，够
