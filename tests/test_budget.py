"""AC6, AC7 — budget monitoring and threshold offload."""

from config import TOOL_RESULT_EXCERPT_LIMIT as _unused  # noqa
from budget import BudgetMonitor
from llm.scripted import ScriptedLLMClient


def test_ac6_budget_statuses():
    """AC6: with budget 1000, 40%/65%/85% -> ok/warning/critical; estimate == len(chars)//4."""
    mon = BudgetMonitor(budget=1000)
    for fraction, expected in ((0.40, "ok"), (0.65, "warning"), (0.85, "critical")):
        tokens = int(1000 * fraction)
        s = "x" * (tokens * 4)  # len(chars)//4 == tokens
        assert mon.estimate(s) == len(s) // 4
        assert mon.status_for(mon.estimate(s)) == expected


def test_ac7_threshold_offload(make_manager, embedder):
    """AC7 (scripted): when assembled context is critical, the conversation section is
    replaced by an offload stub, a [Summary ID] reference appears under Summary Memory, and
    the # Question text is byte-identical."""
    from agent import Agent
    from config import INABILITY_MESSAGE

    manager = make_manager("ac7.sqlite")
    llm = ScriptedLLMClient()
    # ensure there is unconsolidated conversation to consolidate
    for i in range(6):
        manager.conversational.add("t7", "user", f"prior question {i}", f"2026-01-0{i+1}T00:00:00Z")
        manager.conversational.add("t7", "assistant", f"prior answer {i}", f"2026-01-0{i+1}T00:00:01Z")

    agent = Agent(manager, llm, embedder, budget=20)  # tiny budget -> any context is critical
    query = "What method did Kestrel use?"
    result = agent.call_agent("t7", query)

    assert result["offloaded"] is True
    assert result["summary_id"]
    assert result["question_block"] == f"# Question\n{query}"
    ctx = result["turn_context"]
    # conversation replaced by offload stub
    assert "consolidated into summary memory" in ctx
    assert "no unconsolidated messages" not in ctx
    # summary reference present
    assert "[Summary ID:" in ctx
    # final answer still produced
    assert result["final_answer"] not in ("", INABILITY_MESSAGE)
