"""Agent-triggered memory-operation tools (D8): summarize-and-store, expand-summary
(JIT), and toolbox self-lookup. Registered into the toolbox alongside the fixture tools
so the model can invoke consolidation and recovery on demand (AC20).
"""

from __future__ import annotations

import datetime
import inspect

from config import TOOLBOX_K


def summarize_and_store(ctx=None):
    """Summarize the current thread's unconsumed conversation and store the summary in
    summary memory, marking the consumed source rows (R8/R9).

    Returns the new summary id and a short description. Use this when the conversation
    has grown long and you want a durable, recoverable summary.
    """
    from consolidation import consolidate_thread

    manager = ctx["manager"]
    thread_id = ctx["thread_id"]
    llm = ctx["llm"]
    embedder = ctx["embedder"]
    result = consolidate_thread(manager, llm, embedder, thread_id)
    if result is None:
        return "Nothing to summarize; no unconsolidated conversation remains."
    return f"Stored summary {result['summary_id']}: {result['description']}"


def expand_summary(summary_id: str, ctx=None):
    """Expand a previously stored summary, returning the stored summary text and the full
    original messages, chronological, with timestamps (R9). Call before relying on detail
    that exists only inside a summary.
    """
    from consolidation import expand_summary as _expand

    manager = ctx["manager"]
    out = _expand(manager, summary_id)
    if not out["found"]:
        return f"No summary found with id {summary_id}."
    lines = [f"[Summary {summary_id}] {out['summary']}", ""]
    lines.append("Original messages:")
    for m in out["messages"]:
        lines.append(f"{m['timestamp']} [{m['role']}] {m['content']}")
    return "\n".join(lines)


def read_toolbox(query: str = "", ctx=None):
    """Look up the currently registered tools that best match a query (self-lookup).
    Returns the top tool names and descriptions. Use mid-execution if your initial toolset
    proves insufficient.
    """
    manager = ctx["manager"]
    embedder = ctx["embedder"]
    emd = embedder.embed(query or "all tools")
    schemas = manager.toolbox.retrieve_schemas(emd, k=TOOLBOX_K)
    return "\n".join(f"- {s['function']['name']}: {s['function']['description']}" for s in schemas)


def _make_parameters(func) -> dict:
    sig = inspect.signature(func)
    props = {}
    required = []
    for name, param in sig.parameters.items():
        if name == "ctx":
            continue
        ann = param.annotation
        ptype = "string"
        if ann in (int, float):
            ptype = "number"
        props[name] = {"type": ptype}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


def _src(func) -> str:
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        return func.__doc__ or ""


from tools import Tool  # noqa: E402


def build_memory_tools() -> list[Tool]:
    import json as _json

    tools = []
    for f in (summarize_and_store, expand_summary, read_toolbox):
        tools.append(
            Tool(
                name=f.__name__,
                func=f,
                docstring=inspect.getdoc(f) or "",
                parameters=_make_parameters(f),
                needs_context=True,
            )
        )
    return tools
