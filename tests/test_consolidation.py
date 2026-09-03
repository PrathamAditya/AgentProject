"""AC8, AC9, AC14 — recoverable consolidation and structured summarization."""

import re

from conftest import ingest_seed_thread
from consolidation import consolidate_thread, expand_summary
from config import SUMMARY_ID_LEN
from llm.scripted import ScriptedLLMClient

FIRST_USER_MSG = "Find the Kestrel paper about streaming memory consolidation."

HEADINGS = [
    "Technical Information",
    "Emotional Context",
    "Entities & References",
    "Action Items & Decisions",
]


def _summary_id_re():
    return re.compile(r"^[0-9a-f]{" + str(SUMMARY_ID_LEN) + r"}$")


def test_ac8_consolidate_marks_exactly_once(make_manager, embedder):
    """AC8 (scripted): run 1 consolidates all seed rows with a shared valid id; run 2 finds
    nothing to summarize."""
    manager = make_manager("ac8.sqlite")
    ingest_seed_thread(manager)
    llm = ScriptedLLMClient()

    result1 = consolidate_thread(manager, llm, embedder, "seed-01")
    assert result1 is not None
    assert result1["consumed"] == 12
    sid = result1["summary_id"]
    assert _summary_id_re().match(sid)

    # every seed row carries the same summary_id
    rows = manager.conversational.thread_messages("seed-01")
    assert all(r["summary_id"] == sid for r in rows)
    # unconsolidated count is 0
    assert manager.conversational.unconsolidated_count("seed-01") == 0
    # conversation read reports no unconsolidated messages
    assert manager.conversational.read("seed-01") == []

    # run 2 -> nothing to summarize
    assert consolidate_thread(manager, llm, embedder, "seed-01") is None


def test_ac9_expand_recovers_originals(make_manager, embedder):
    """AC9 (scripted): expand returns the stored summary AND all 12 originals
    chronologically, including the first user message verbatim."""
    manager = make_manager("ac9.sqlite")
    ingest_seed_thread(manager)
    llm = ScriptedLLMClient()

    res = consolidate_thread(manager, llm, embedder, "seed-01")
    sid = res["summary_id"]
    expanded = expand_summary(manager, sid)
    assert expanded["found"] is True

    # stored summary text present
    assert res["summary_id"] == sid
    assert "Technical Information" in expanded["summary"]

    msgs = expanded["messages"]
    assert len(msgs) == 12
    # chronological
    ts = [m["timestamp"] for m in msgs]
    assert ts == sorted(ts)
    # first user message verbatim
    assert any(m["role"] == "user" and m["content"] == FIRST_USER_MSG for m in msgs)


def test_ac14_scripted_summary_headings(make_manager, embedder):
    """AC14 (scripted): output has the four exact headings, valid 8-char id, 8-12 word label
    that is not generic."""
    manager = make_manager("ac14.sqlite")
    ingest_seed_thread(manager)
    llm = ScriptedLLMClient()

    res = consolidate_thread(manager, llm, embedder, "seed-01")
    sid = res["summary_id"]
    stored = manager.summary.get_by_id(sid)
    assert stored is not None
    summary = stored["summary"]

    for h in HEADINGS:
        assert f"## {h}" in summary

    assert _summary_id_re().match(sid)

    label = stored["description"]
    words = label.split()
    assert 8 <= len(words) <= 12
    assert label.strip().lower().rstrip(".") not in {
        "conversation summary", "conversation", "summary", "thread summary", "chat summary",
    }
