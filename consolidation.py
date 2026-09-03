"""Consolidation pipeline (R7/R8/R9): threshold/tool-triggered summarization with
write-back links and just-in-time expansion. Summary of the thread's unconsolidated
conversation is stored; the exact source rows are marked with the summary_id; the
conversation segment is replaced by a stub pointing at summary references.
"""

from __future__ import annotations

import datetime

from llm.base import LLMClient

# four headings (R8)
HEADINGS = [
    "Technical Information",
    "Emotional Context",
    "Entities & References",
    "Action Items & Decisions",
]


def build_summary_prompt(transcript: str) -> str:
    return (
        "Summarize the following conversation using EXACTLY these four headings:\n\n"
        + "\n".join(f"## {h}" for h in HEADINGS)
        + "\n\nCONVERSATION:\n"
        + transcript
    )


def consolidate_thread(manager, llm: LLMClient, embedder, thread_id: str) -> dict | None:
    """Summarize a thread's unconsolidated messages, store the summary, and mark the
    consumed rows (R9). Returns a dict about what happened, or None if nothing to do."""
    # R9: consume ALL unconsolidated rows (not the preload limit of 10)
    rows = manager.conversational.read_unconsolidated_all(thread_id)
    if not rows:
        return None

    # R8: summarize from at most the first 6,000 chars
    transcript = "\n".join(f"{r['role']}: {r['content']}" for r in rows)
    transcript = transcript[:6000]

    result = llm.summarize(transcript)

    manager.summary.add(
        summary_id=result.summary_id,
        summary=result.summary,
        description=result.description,
        full_content="\n".join(
            f"{r['timestamp']} {r['role']}: {r['content']}" for r in rows
        ),
        thread_id=thread_id,
        embedding=embedder.embed(result.summary + " " + result.description),
    )

    # mark exactly the consumed rows (R9)
    for r in rows:
        manager.conn.execute(
            "UPDATE conversational_memory SET summary_id = ? WHERE id = ?",
            (result.summary_id, r["id"]),
        )
    manager.conn.commit()

    return {
        "summary_id": result.summary_id,
        "description": result.description,
        "consumed": len(rows),
    }


def expand_summary(manager, summary_id: str) -> dict:
    """R9 JIT expansion: return stored summary text plus all originals,
    chronological, with timestamps."""
    row = manager.summary.get_by_id(summary_id)
    if row is None:
        return {"summary_id": summary_id, "summary": "", "messages": [], "found": False}
    full = row["full_content"]
    messages = []
    for line in full.splitlines():
        parts = line.split(" ", 2)
        if len(parts) >= 3:
            ts = parts[0]
            role = parts[1].rstrip(":")
            content = parts[2]
            messages.append({"timestamp": ts, "role": role, "content": content})
    return {
        "summary_id": summary_id,
        "summary": row["summary"],
        "messages": messages,
        "found": True,
    }


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
