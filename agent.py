"""Memory-aware agent harness (R1-R16, CTX-A). Deterministic preload + partitioned
context + budget guard + focused tool retrieval + logged tool loop + write-back.
"""

from __future__ import annotations

import datetime

from budget import BudgetMonitor
from config import (
    CONVERSATION_LIMIT,
    KB_K,
    WORKFLOW_K,
    ENTITY_K,
    SUMMARY_K,
    TOOLBOX_K,
    MAX_ITERATIONS,
    INABILITY_MESSAGE,
    TOOL_RESULT_EXCERPT_LIMIT,
    TOOL_LOG_PREVIEW_BYTES,
)
from consolidation import consolidate_thread, now_iso
from llm.base import LLMClient
from memory.manager import MemoryManager

SYSTEM_PROMPT = """You are a memory-aware research assistant. You have access to several
memory segments that are loaded before this turn. Each segment is labeled and
self-describing:

- ## Conversation Memory: prior messages in this thread. Consult it before asking
  the user to repeat anything already discussed.
- ## Knowledge Base Memory: facts the assistant has previously stored. Treat it as
  your factual reference for claims.
- ## Workflow Memory: past multi-step procedures and their outcomes. Reuse a
  matching workflow rather than reinventing it.
- ## Entity Memory: known people, places, and systems. Use it to anchor references.
- ## Summary Memory: consolidated summaries of older parts of this thread, each with
  a summary id. If a detail is mentioned as existing only inside a summary, expand
  that summary with the expand tool before relying on it.

Conflict priority, highest to lowest: current question > latest conversation >
knowledge-base evidence > older summaries/workflows.

Use the provided tools with judgment; call them only when needed. If you do not have
evidence for a factual claim available in the Knowledge Base Memory, state your
uncertainty instead of asserting it."""


def read_context_segments(manager: MemoryManager, embedder, thread_id: str, query: str) -> dict:
    qemb = embedder.embed(query)

    conv = manager.conversational.read(thread_id, limit=CONVERSATION_LIMIT)

    kb = manager.knowledge_base.search(qemb, k=KB_K)

    wf = manager.workflow.search(qemb, k=WORKFLOW_K)

    ents = manager.entity.format_bullets(qemb, k=ENTITY_K)

    summ = manager.summary.search(qemb, k=SUMMARY_K, thread_id=thread_id)

    return {"conv": conv, "kb": kb, "workflow": wf, "entities": ents, "summary": summ}


def render_segments(segments: dict, offloaded: bool = False) -> str:
    lines = []

    conv = segments["conv"]
    if offloaded:
        lines.append("## Conversation Memory")
        lines.append("(consolidated — prior thread context moved to summary memory)")
        lines.append("> Conversation has been consolidated into summary memory. See the Summary Memory segment.")
    elif conv:
        lines.append("## Conversation Memory")
        lines.append("(prior thread messages)")
        for m in conv:
            lines.append(f"- [{m['role']}] {m['content']}")
    else:
        lines.append("## Conversation Memory")
        lines.append("(no unconsolidated messages)")

    lines.append("")
    lines.append("## Knowledge Base Memory")
    if segments["kb"]:
        for sim, row in segments["kb"]:
            lines.append(f"- [{row['source']}] {row['text']}")
    else:
        lines.append("(no knowledge-base matches)")

    lines.append("")
    lines.append("## Workflow Memory")
    if segments["workflow"]:
        for sim, row in segments["workflow"]:
            lines.append(f"- {row['query']}: {row['answer_excerpt']}")
    else:
        lines.append("(no workflow matches)")

    lines.append("")
    lines.append("## Entity Memory")
    if segments["entities"]:
        lines.extend(segments["entities"])
    else:
        lines.append("(no entities)")

    lines.append("")
    lines.append("## Summary Memory")
    if segments["summary"]:
        for sim, row in segments["summary"]:
            lines.append(f"- [Summary ID: {row['summary_id']}] {row['description']}")
    else:
        lines.append("(no summaries)")

    return "\n".join(lines)


def assemble_turn_context(
    manager, embedder, thread_id, query, budget_monitor: BudgetMonitor
) -> dict:
    """Assemble partitioned context and report the pre-model budget status.

    Offload (R7) is handled by call_agent, which has the llm client needed to run
    consolidation. This helper exists so the preload assembly is testable in isolation.
    """
    segments = read_context_segments(manager, embedder, thread_id, query)
    context = render_segments(segments, offloaded=False)
    question_block = f"# Question\n{query}"
    full = question_block + "\n\n" + context
    tokens = budget_monitor.estimate(full)
    status = budget_monitor.status_for(tokens)

    return {
        "context": context,
        "segments": segments,
        "offloaded": False,
        "budget_status": status,
        "summary_ref": None,
        "tokens": tokens,
    }


def build_system_message() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def _preview_bytes(result: str) -> str:
    return result.encode("utf-8")[:TOOL_LOG_PREVIEW_BYTES].decode("utf-8", errors="ignore")


class Agent:
    def __init__(
        self,
        manager: MemoryManager,
        llm: LLMClient,
        embedder,
        *,
        budget: int | None = None,
        toolbox_k: int = TOOLBOX_K,
        augment_tools: bool = True,
    ):
        self.manager = manager
        self.llm = llm
        self.embedder = embedder
        self.budget_monitor = BudgetMonitor(budget=budget, model=getattr(llm, "agent_model", None))
        self.toolbox_k = toolbox_k
        self.augment_tools = augment_tools

        from tools import register_tools, build_tools
        from memory_tools import build_memory_tools

        register_tools(manager, llm, embedder, build_tools(), augment=augment_tools)
        register_tools(manager, llm, embedder, build_memory_tools(), augment=False)
        self.tools_by_name = {t.name: t for t in build_tools() + build_memory_tools()}

    # -- per-turn deterministic write-backs (R1, R12) --
    def _write_workflow(self, thread_id, query, steps, final_answer, timestamp):
        if not steps:
            return
        excerpt = final_answer[:200]
        self.manager.workflow.add(
            query=query,
            steps=steps,
            answer_excerpt=excerpt,
            success=True,
            timestamp=timestamp,
        )

    def _extract_and_write_entities(self, text: str):
        try:
            entities = self.llm.extract_entities(text)
        except Exception:
            return
        for e in entities:
            if not e.name:
                continue
            emd = self.embedder.embed(e.name + " " + e.type_ + " " + e.description)
            self.manager.entity.add(
                name=e.name, type_=e.type_, description=e.description, embedding=emd
            )

    def call_agent(self, thread_id: str, query: str) -> dict:
        """Run one full turn. Returns AgentTurnResult-shaped dict."""
        manager = self.manager
        ts = now_iso()

        # R1: persist the user query
        manager.conversational.add(thread_id, "user", query, ts)

        steps: list[str] = []
        completed = False
        final_answer = INABILITY_MESSAGE
        self.last_applied_tool_messages = []

        # deterministic preload (R2/R3) + budget check (R6/R7)
        segments = read_context_segments(manager, self.embedder, thread_id, query)
        context = render_segments(segments, offloaded=False)
        question_block = f"# Question\n{query}"
        full = question_block + "\n\n" + context
        budget_status = self.budget_monitor.status_for(self.budget_monitor.estimate(full))

        offloaded = False
        summary_id = None
        if self.budget_monitor.is_critical(self.budget_monitor.estimate(full)):
            # R7 threshold offload: consolidate the thread's unconsolidated conversation
            cons = consolidate_thread(manager, self.llm, self.embedder, thread_id)
            if cons:
                summary_id = cons["summary_id"]
                segments = read_context_segments(manager, self.embedder, thread_id, query)
                context = render_segments(segments, offloaded=True)
                offloaded = True

        # focused tool retrieval (R4)
        qemb = self.embedder.embed(query)
        tool_schemas = manager.toolbox.retrieve_schemas(qemb, k=self.toolbox_k)

        # build user content (D9 partition)
        user_content = question_block + "\n\n" + context

        messages = [{"role": "user", "content": user_content}]

        executed_tools: list[str] = []
        for iteration in range(MAX_ITERATIONS):
            result = self.llm.chat(
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tool_schemas or None,
            )
            if result.has_tool_calls:
                tc = result.tool_calls[0]
                tool = self.tools_by_name.get(tc.name)
                executed_tools.append(tc.name)
                messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": _dumps(tc.arguments)}}
                ]})
                if tool is None:
                    log_id = manager.tool_log.log(
                        thread_id=thread_id,
                        tool_name=tc.name,
                        tool_args=tc.arguments,
                        result="unknown tool",
                        result_preview="unknown tool",
                        status="failed",
                        error_message=f"tool {tc.name} not registered",
                        timestamp=now_iso(),
                    )
                    tool_msg = [{"role": "tool", "tool_call_id": tc.id,
                                 "content": f"[tool error: unknown tool; log {log_id}]"}]
                    messages.extend(tool_msg)
                    self.last_applied_tool_messages.extend(tool_msg)
                    steps.append(f"tool {tc.name} (unknown) failed")
                else:
                    out = tool.execute(tc.arguments, {
                        "manager": manager, "embedder": self.embedder, "thread_id": thread_id,
                        "llm": self.llm,
                    })
                    # R10: full result to tool log; bounded excerpt to model
                    preview = _preview_bytes(out)
                    log_id = manager.tool_log.log(
                        thread_id=thread_id,
                        tool_name=tc.name,
                        tool_args=tc.arguments,
                        result=out,
                        result_preview=preview,
                        status="success",
                        timestamp=now_iso(),
                        tool_call_id=tc.id,
                    )
                    truncated = len(out) > TOOL_RESULT_EXCERPT_LIMIT
                    excerpt = out[:TOOL_RESULT_EXCERPT_LIMIT]
                    tool_msgs = self.llm.tool_result_messages(tc, excerpt, truncated, log_id)
                    messages.extend(tool_msgs)
                    self.last_applied_tool_messages.extend(tool_msgs)
                    steps.append(f"called {tc.name} [log {log_id}]")
                continue

            # no tool calls -> this is the final answer
            final_answer = (result.content or "").strip() or INABILITY_MESSAGE
            completed = True
            break

        # R1: persist the final answer
        manager.conversational.add(thread_id, "assistant", final_answer, now_iso())

        # R11: workflow write-back when >=1 tool call
        self._write_workflow(thread_id, query, steps, final_answer, ts)

        # R12: non-blocking entity extraction (after query and after answer)
        self._extract_and_write_entities(query)
        self._extract_and_write_entities(final_answer)

        return {
            "thread_id": thread_id,
            "final_answer": final_answer,
            "steps": steps,
            "completed": completed,
            "budget_status": budget_status,
            "offloaded": offloaded,
            "summary_id": summary_id,
            "turn_context": context,
            "question_block": question_block,
            "model_input": {
                "system": SYSTEM_PROMPT,
                "messages": messages,
                "tools": tool_schemas,
            },
            "applied_tool_messages": self.last_applied_tool_messages,
        }


def _dumps(d: dict) -> str:
    import json

    return json.dumps(d)
