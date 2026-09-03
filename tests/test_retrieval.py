"""AC3, AC4, AC5 — semantic KB retrieval, toolbox retrieval schema, idempotent registration."""

from conftest import ingest_kb, register_all_tools

REQUIRED_SCHEMA_KEYS = {"type", "function"}


def test_ac3_kb_true_nearest_neighbor(make_manager, embedder):
    """AC3 (offline): heron-notes is the top hit for 'entity graphs for routing tools'."""
    manager = make_manager("ac3.sqlite")
    ingest_kb(manager, embedder)
    hits = manager.knowledge_base.search(
        embedder.embed("entity graphs for routing tools"), k=3
    )
    top = hits[0][1]
    assert "heron" in top["source"].lower(), top["source"]
    assert top["source"]  # carries source-file metadata
    assert top["chunk_id"] is not None
    assert top["num_chunks"] is not None


def test_ac4_toolbox_retrieval_schemas(make_manager, embedder):
    """AC4 (offline): at most 5 valid RetrievedToolSchema, unique names, paper_search present."""
    manager = make_manager("ac4.sqlite")
    register_all_tools(manager, embedder)
    schemas = manager.toolbox.retrieve_schemas(embedder.embed("find research papers on agent memory"), k=5)
    assert len(schemas) <= 5
    assert len(schemas) > 0
    names = [s["function"]["name"] for s in schemas]
    assert len(names) == len(set(names))  # unique
    assert "paper_search" in names
    for s in schemas:
        assert set(s.keys()) == {"type", "function"}
        fn = s["function"]
        assert set(fn.keys()) == {"name", "description", "parameters"}
        assert fn["name"]
        assert fn["description"]
        p = fn["parameters"]
        assert p["type"] == "object"
        assert "properties" in p
        assert "required" in p


def test_ac5_idempotent_registration(make_manager, embedder):
    """AC5 (offline): re-registering paper_search yields exactly one row."""
    manager = make_manager("ac5.sqlite")
    results1 = register_all_tools(manager, embedder)
    assert results1["paper_search"] is True
    results2 = register_all_tools(manager, embedder)
    assert results2["paper_search"] is False  # already present, no duplicate

    conn = manager.conn
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM toolbox_memory WHERE tool_name = 'paper_search'"
    ).fetchone()["n"]
    assert n == 1
