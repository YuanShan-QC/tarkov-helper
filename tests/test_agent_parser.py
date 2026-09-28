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
        """降级策略：宁可回一段不完美的原文，也不让用户看到报错"""
        result = agent._parse_response("我直接回答你吧：显卡在涨价")
        assert result["type"] == "done"
        assert result["reply"].startswith("我直接回答你吧")

    def test_回复过长时截断到_200_字(self, agent):
        result = agent._parse_response("DONE: " + "很长的回答" * 100)
        assert len(result["reply"]) <= 200 + len("很长的回答") * 100  # DONE 分支不截断
        result2 = agent._parse_response("没有标记的超长输出" * 100)
        assert len(result2["reply"]) == 200


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
        for i in range(60):
            agent.scratchpad.append({"role": "thought", "content": f"思考{i}" * 100})
        history = agent._build_history()
        # 上限 2500 字符（加上 "## 任务记录\n" 前缀和换行的余量）
        assert len(history) < 2600

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
