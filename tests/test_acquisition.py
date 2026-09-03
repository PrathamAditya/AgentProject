"""AC16, AC17, AC18 — acquisition search-and-store, configuration coherence, tool
augmentation."""

from config import CHUNK_SIZE, CHUNK_OVERLAP, TOOLBOX_K, DISTANCE_STRATEGY
from conftest import register_all_tools, kestrel_notes_text
from tools import fetch_notes, build_tools, register_tools
from llm.scripted import ScriptedLLMClient


def test_ac16_fetch_notes_search_and_store(make_manager, embedder):
    """AC16 (offline): executing fetch_notes on kestrel-notes stores chunk rows with source
    metadata within the tool call itself, respecting the 1,500/200 chunking default."""
    manager = make_manager("ac16.sqlite")
    before = manager.knowledge_base.count()

    out = fetch_notes("kestrel-notes.md", {
        "manager": manager,
        "embedder": embedder,
        "thread_id": "t16",
    })

    after = manager.knowledge_base.count()
    assert after > before  # KB gained chunk rows within the tool call

    rows = [(dict(r)) for r in manager.knowledge_base.all()]
    kestrel_rows = [r for r in rows if r["source"] == "kestrel-notes.md"]
    assert kestrel_rows, "no kestrel-notes chunks stored"
    for r in kestrel_rows:
        assert "source" in r and r["source"] == "kestrel-notes.md"
        assert r["chunk_id"] is not None
        assert r["num_chunks"] == len(kestrel_rows)
        assert len(r["text"]) <= CHUNK_SIZE + CHUNK_OVERLAP
    # oversized return guarantees the >3,000-char case (AC10)
    assert len(out) > 3000


def test_ac17_configuration_coherence(make_manager, embedder):
    """AC17 (offline): every vector store and the toolbox retriever share one distance
    strategy, and retrieval k resolves from a single config point."""
    manager = make_manager("ac17.sqlite")
    for store in manager.vector_stores():
        assert store.distance_strategy == DISTANCE_STRATEGY
    assert manager.knowledge_base.distance_strategy == "cosine"
    assert manager.workflow.distance_strategy == "cosine"
    assert manager.entity.distance_strategy == "cosine"
    assert manager.summary.distance_strategy == "cosine"
    assert manager.toolbox.distance_strategy == "cosine"

    # toolbox k is a single config value
    assert manager.toolbox.k_default == TOOLBOX_K
    assert TOOLBOX_K == 5


def test_ac18_llm_augmented_registration(make_manager, embedder):
    """AC18 (scripted): with augmentation on, the stored description is the enriched text
    (not the raw docstring) and the embedded text contains the 5 synthetic queries."""
    manager = make_manager("ac18.sqlite")
    enriched = "Richly rewritten description of the paper search tool."
    queries = ["find papers", "search agent memory", "lookup tool", "use tool", "help tool"]
    llm = ScriptedLLMClient(scripted_augment=(enriched, queries))

    # register a single tool (paper_search) with augmentation on
    tools = [t for t in build_tools() if t.name == "paper_search"]
    register_tools(manager, llm, embedder, tools, augment=True)

    row = manager.conn.execute(
        "SELECT * FROM toolbox_memory WHERE tool_name='paper_search'"
    ).fetchone()
    assert row is not None
    assert row["description"] == enriched
    assert row["description"] != ""  # not the raw one-line docstring
    stored_text = row["text"]
    for q in queries:
        assert q in stored_text
