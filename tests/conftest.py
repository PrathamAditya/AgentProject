"""Shared fixtures for the acceptance-test suite.

The embedder is loaded once per session (one-time model download). Each test gets an
isolated SQLite DB under tmp_path so no test leaks session state.
"""

import json

import pytest

from memory.manager import MemoryManager
from chunker import chunk_text
from setup_data import FIXTURES_DIR


@pytest.fixture(scope="session")
def embedder():
    from embeddings import get_embedder

    return get_embedder()


@pytest.fixture
def make_manager(tmp_path):
    created = []

    def _make(name: str = "db.sqlite") -> MemoryManager:
        path = tmp_path / name
        m = MemoryManager(path=str(path))
        created.append(m)
        return m

    yield _make

    for m in created:
        try:
            m.close()
        except Exception:
            pass


def ingest_kb(manager: MemoryManager, embedder) -> None:
    for md in sorted((FIXTURES_DIR / "kb").glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            manager.knowledge_base.add_chunk(
                chunk,
                embedder.embed(chunk),
                source=md.name,
                chunk_id=i,
                num_chunks=len(chunk_text(text)),
                title=md.stem,
            )


def ingest_seed_thread(manager: MemoryManager) -> None:
    path = FIXTURES_DIR / "conversation" / "seed-thread.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for msg in data:
        manager.conversational.add(
            thread_id=msg["thread_id"],
            role=msg["role"],
            content=msg["content"],
            timestamp=msg["timestamp"],
            row_id=msg.get("id"),
        )


def register_all_tools(manager: MemoryManager, embedder, scripted_client=None):
    if scripted_client is None:
        from llm.scripted import ScriptedLLMClient

        scripted_client = ScriptedLLMClient()
    from tools import register_tools, build_tools

    return register_tools(manager, scripted_client, embedder, build_tools(), augment=True)


def kestrel_notes_text() -> str:
    return (FIXTURES_DIR / "kb" / "kestrel-notes.md").read_text(encoding="utf-8")
