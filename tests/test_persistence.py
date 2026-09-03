"""AC1, AC2 — persistence across process restarts and thread-scoped reads."""

import json
import os
import subprocess
import sys

from conftest import ingest_kb, ingest_seed_thread

DB_ENV = "AGENT_MEMORY_DB"


def test_ac1_memory_survives_process_restart(make_manager, embedder, tmp_path):
    """AC1 (offline, cross-process): seed + KB written by process 1 are readable by a new
    OS process opening the same DB file."""
    db_file = tmp_path / "ac1.sqlite"
    manager = make_manager("ac1.sqlite")
    ingest_kb(manager, embedder)
    ingest_seed_thread(manager)
    assert manager.conversational.thread_messages("seed-01")  # sanity: 12 written
    manager.close()

    # new OS process reads the same file
    db_str = str(db_file)
    probe = f"""
import json, os
from db import connect
from memory.manager import MemoryManager
from embeddings import get_embedder
from config import KB_K
m = MemoryManager(path={db_str!r})
conv = m.conversational.thread_messages('seed-01')
assert len(conv) == 12, f'expected 12 got {{len(conv)}}'
roles = [c['role'] for c in conv]
assert roles == ['user','assistant']*6, roles
# chronological ascending
ts = [c['timestamp'] for c in conv]
assert ts == sorted(ts), ts
e = get_embedder()
hits = m.knowledge_base.search(e.embed('streaming memory consolidation'), k=3)
top = hits[0][1]
assert 'kestrel' in top['source'].lower(), top['source']
print('AC1_OK', len(conv), top['source'])
"""
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={**os.environ, DB_ENV: str(db_file)},
    )
    assert proc.returncode == 0, proc.stderr
    assert "AC1_OK 12" in proc.stdout
    assert "kestrel-notes" in proc.stdout


def test_ac2_thread_scoped_reads(make_manager, embedder):
    """AC2 (offline): reading seed-01 returns only its unconsolidated messages ascending,
    never other-01 content."""
    manager = make_manager("ac2.sqlite")
    ingest_seed_thread(manager)
    # two messages in another thread
    manager.conversational.add("other-01", "user", "someone else's question", "2026-02-01T00:00:00Z")
    manager.conversational.add("other-01", "assistant", "someone else's answer", "2026-02-01T00:00:01Z")

    msgs = manager.conversational.read("seed-01", limit=100)
    assert len(msgs) == 12
    assert all(m["thread_id"] == "seed-01" for m in msgs)
    assert all(m["role"] in ("user", "assistant") for m in msgs)
    assert all("timestamp" in m for m in msgs)
    ts = [m["timestamp"] for m in msgs]
    assert ts == sorted(ts)
    contents = json.dumps(msgs)
    assert "someone else's" not in contents
