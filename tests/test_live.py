"""AC19 (live-keyless), AC20 (live-keyed). Excluded from the default offline run via
the 'live' marker (see pyproject.toml)."""

import json
import os

import pytest

from tools import paper_search
from config import ABSTRACT_CAP

pytestmark = pytest.mark.live


@pytest.mark.live_keyless
def test_ac19_arxiv_candidate_search():
    """AC19 (live-keyless): the arXiv candidate-search tool returns a JSON array whose
    elements carry arxiv_id, entry_id, title, authors, published, abstract (<=2,500)."""
    out = paper_search("agent memory")
    data = json.loads(out)
    assert isinstance(data, list) and len(data) > 0
    for item in data:
        for key in ("arxiv_id", "entry_id", "title", "authors", "published", "abstract"):
            assert key in item, f"missing {key}"
        assert len(item["abstract"]) <= ABSTRACT_CAP


@pytest.mark.live_keyed
def test_ac20_end_to_end_memory_recovery():
    """AC20 (live-keyed): the course end-to-end sequence. Turn 2 resolves 'the paper' from
    conversation memory; turn 4's answer names the first question after consolidating and
    expanding."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")

    from agent import Agent
    from memory.manager import MemoryManager
    from llm.openai_client import OpenAILLMClient
    from embeddings import get_embedder
    import tempfile

    embedder = get_embedder()
    tmp = tempfile.mkdtemp()
    manager = MemoryManager(path=os.path.join(tmp, "ac20.sqlite"))
    llm = OpenAILLMClient()
    agent = Agent(manager, llm, embedder, augment_tools=True)
    thread = "ac20"
    try:
        r1 = agent.call_agent(thread, "Find the MemGPT paper")
        assert r1["completed"]

        r2 = agent.call_agent(thread, "Save the content of the paper")
        assert r2["completed"]

        r3 = agent.call_agent(thread, "Summarize the conversation so far using your tool")
        assert r3["completed"]

        r4 = agent.call_agent(thread, "What was my first question?")
        assert r4["completed"]
        ans = r4["final_answer"].lower()
        # the first question should be named/recoverable (MemGPT / the paper)
        assert "memgpt" in ans or "paper" in ans or "find" in ans
    except Exception as e:  # noqa: BLE001
        # A live AC that cannot run is SKIPPED with a reason, never reported as passed.
        msg = str(e)
        if "insufficient_quota" in msg or "credit_balance_exhausted" in msg or "no credits" in msg:
            pytest.skip(f"OpenAI account has no credits (billing/quota): {msg[:120]}")
        raise
    finally:
        manager.close()
