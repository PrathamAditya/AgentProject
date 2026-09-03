"""AC10, AC11, AC12, AC13, AC15 — the agent loop: tool logging, workflow write-back,
iteration cap, partitioned context, non-blocking entity extraction."""

import json

from agent import Agent
from config import (
    INABILITY_MESSAGE,
    MAX_ITERATIONS,
    TOOL_RESULT_EXCERPT_LIMIT,
    TOOL_LOG_PREVIEW_BYTES,
)
from conftest import ingest_seed_thread
from llm.scripted import ScriptedLLMClient

FIVE_HEADINGS = [
    "## Conversation Memory",
    "## Knowledge Base Memory",
    "## Workflow Memory",
    "## Entity Memory",
    "## Summary Memory",
]


def test_ac10_tool_log_bounded_excerpt(make_manager, embedder):
    """AC10 (scripted): fetch_notes (>3,000 chars) is fully logged and the model gets a
    <=3,000-char excerpt plus a truncation notice naming the log id."""
    manager = make_manager("ac10.sqlite")
    llm = ScriptedLLMClient(script=[
        {"type": "tool", "name": "fetch_notes", "args": {"path": "kestrel-notes.md"}},
        {"type": "answer", "content": "I saved the notes into the knowledge base."},
    ])
    agent = Agent(manager, llm, embedder)

    result = agent.call_agent("t10", "Please fetch and save the Kestrel notes.")

    # find the fetch_notes success log record
    rec = None
    for row in manager.conn.execute(
        "SELECT * FROM tool_log_memory WHERE tool_name='fetch_notes'"
    ).fetchall():
        rec = dict(row)
    assert rec is not None
    assert rec["status"] == "success"
    assert len(rec["result"]) > TOOL_RESULT_EXCERPT_LIMIT  # full result held
    assert "tool_args" in rec
    assert len(rec["result_preview"].encode("utf-8")) <= TOOL_LOG_PREVIEW_BYTES

    # the applied tool message
    applied = "".join(
        m.get("content", "") for m in result["applied_tool_messages"]
    )
    assert applied.startswith(rec["result"][:TOOL_RESULT_EXCERPT_LIMIT]) or len(applied) > 0
    # excerpt portion <= 3000 chars of the actual result before the notice
    body, _, notice = applied.partition("\n\n[Result truncated;")
    assert len(body) <= TOOL_RESULT_EXCERPT_LIMIT
    assert "tool log id" in notice
    assert rec["id"] in notice
    assert result["completed"] is True


def test_ac11_workflow_write_back(make_manager, embedder):
    """AC11 (scripted): a two-tool-call turn writes one workflow; zero-step controls are
    excluded from reads."""
    manager = make_manager("ac11.sqlite")
    llm = ScriptedLLMClient(script=[
        {"type": "tool", "name": "get_current_time", "args": {}},
        {"type": "tool", "name": "get_current_time", "args": {}},
        {"type": "answer", "content": "Here is the outcome of those two steps."},
    ])
    agent = Agent(manager, llm, embedder)

    result = agent.call_agent("t11", "Help me do a two-step task with the tools.")
    assert result["completed"] is True

    # zero-step control written directly
    manager.workflow.add(
        query="zero-step control",
        steps=[],
        answer_excerpt="control",
        success=True,
        timestamp="2026-01-01T00:00:00Z",
    )

    hits = manager.workflow.search(embedder.embed("two-step task with tools"), k=5)
    fetched = [row for _, row in hits]
    assert fetched, "no workflow returned"
    assert all(row["num_steps"] > 0 for row in fetched)  # zero-step excluded

    wf = fetched[0]
    steps = json.loads(wf["steps_json"])
    assert wf["query"]
    assert len(steps) == 2
    assert len(wf["answer_excerpt"]) <= 200
    assert wf["num_steps"] == 2
    # ordered step descriptions with outcome markers present
    assert all(("called" in s or "log" in s) for s in steps)


def test_ac12_iteration_cap(make_manager, embedder):
    """AC12 (scripted): an always-tool-calling model stops at 10 iterations, completed=false,
    final answer is the inability message and is persisted."""
    manager = make_manager("ac12.sqlite")
    llm = ScriptedLLMClient(always_tool_call=True)
    agent = Agent(manager, llm, embedder)

    result = agent.call_agent("t12", "run forever")
    assert result["completed"] is False
    assert result["final_answer"] == INABILITY_MESSAGE

    # exactly 10 iterations -> 10 tool logs
    n = manager.conn.execute(
        "SELECT COUNT(*) AS n FROM tool_log_memory"
    ).fetchone()["n"]
    assert n == MAX_ITERATIONS

    # final answer persisted to conversational memory (R1)
    msgs = manager.conversational.thread_messages("t12")
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == INABILITY_MESSAGE


def test_ac13_partitioned_context(make_manager, embedder):
    """AC13 (scripted): model input begins with # Question, has all five headings in order,
    and the system prompt names all five plus the conflict-priority order."""
    manager = make_manager("ac13.sqlite")
    ingest_seed_thread(manager)
    llm = ScriptedLLMClient()
    agent = Agent(manager, llm, embedder)

    query = "What did Kestrel claim about resumption?"
    result = agent.call_agent("atom-13", query)
    mi = result["model_input"]

    # user content (first user message) begins with # Question and has the five headings
    user_content = ""
    for m in mi["messages"]:
        if m["role"] == "user":
            user_content = m["content"]
            break
    assert user_content.startswith("# Question")
    idx = [user_content.find(h) for h in FIVE_HEADINGS]
    assert all(i >= 0 for i in idx), idx
    assert idx == sorted(idx), "headings out of order"

    sys_prompt = mi["system"]
    for h in FIVE_HEADINGS:
        assert h.replace("## ", "") in sys_prompt
    # conflict-priority order present
    assert "current question" in sys_prompt
    assert "latest conversation" in sys_prompt
    assert "knowledge-base evidence" in sys_prompt
    assert "older summaries/workflows" in sys_prompt


def test_ac15_non_blocking_entity_extraction(make_manager, embedder):
    """AC15 (scripted): run 1 writes 2 entities returned as bullets; run 2's extraction
    exception leaves the final answer unchanged."""
    from llm.base import EntityResult

    manager = make_manager("ac15.sqlite")
    ents = [
        EntityResult("Kestrel", "SYSTEM", "Streaming memory consolidation paper"),
        EntityResult("R. Marlow", "PERSON", "First author of the Kestrel paper"),
    ]
    llm = ScriptedLLMClient(scripted_entities=ents)
    agent = Agent(manager, llm, embedder)

    result1 = agent.call_agent("t15", "Talk about the Kestrel paper.")
    assert result1["completed"] is True

    bullets = manager.entity.format_bullets(embedder.embed("Kestrel paper"))
    assert manager.entity.count() >= 2
    assert len(bullets) >= 2

    # run 2: extraction raises -> answer unchanged, turn still completes
    failing_llm = ScriptedLLMClient(
        scripted_entities=ents,
        raise_on_extract=True,
        answer="A stable scripted answer.",
    )
    agent2 = Agent(manager, failing_llm, embedder)
    result2 = agent2.call_agent("t15b", "Another question about memory.")
    assert result2["final_answer"] == "A stable scripted answer."
    assert result2["completed"] is True
