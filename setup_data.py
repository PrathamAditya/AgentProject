"""Fixture ingestion (process 1 for AC1): write the seed thread and ingest KB notes.
Used at setup; AC1 re-opens the same DB from a new OS process to assert persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

from db import connect, init_schema
from memory.manager import MemoryManager
from chunker import chunk_text

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def ingest_kb(manager: MemoryManager, embedder):
    for md in sorted((FIXTURES_DIR / "kb").glob("*.md")):
        text = md.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            emb = embedder.embed(chunk)
            manager.knowledge_base.add_chunk(
                chunk,
                emb,
                source=md.name,
                chunk_id=i,
                num_chunks=len(chunks),
                title=md.stem,
            )


def ingest_seed_thread(manager: MemoryManager):
    path = FIXTURES_DIR / "conversation" / "seed-thread.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for msg in data:
        manager.conversational.add(
            thread_id=msg["thread_id"],
            role=msg["role"],
            content=msg["content"],
            timestamp=msg["timestamp"],
            row_id=msg.get("id"),
            metadata=msg.get("metadata"),
        )


def setup_all(
    manager: MemoryManager = None, embedder=None, *, ingest_seed: bool = True
) -> MemoryManager:
    import os

    os.makedirs(FIXTURES_DIR / "kb", exist_ok=True)
    if manager is None:
        manager = MemoryManager()
    if embedder is None:
        from embeddings import get_embedder

        embedder = get_embedder()
    ingest_kb(manager, embedder)
    if ingest_seed:
        ingest_seed_thread(manager)
    return manager
