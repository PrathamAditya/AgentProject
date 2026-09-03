"""Scripted LLM client [H] — a deterministic stand-in honoring the D4 tool-calling
interface. Lets the offline/scripted oracle run without a key. Chat behavior is driven
by a `script` list (FIFO) so tests can choreograph tool-call-then-answer sequences.
Never modify fixtures to make a test pass.
"""

from __future__ import annotations

import secrets

from config import AGENT_MODEL
from .base import LLMClient, ChatResult, ToolCall, SummaryResult, EntityResult

_CANNED_SUMMARY = """## Technical Information

Discussed Kestrel (KX-2025-011), a tiered summary-ledger approach to streaming memory
consolidation that reduces resumption errors by 41% on long-horizon tasks.

## Emotional Context

The user expressed clarity and satisfaction with the explanation and asked for a note
about which paper is the primary reference.

## Entities & References

- Kestrel (KX-2025-011, R. Marlow, T. Iversen, 2025)
- Heron (HX-2024-007, F. Adeyemi, 2024)

## Action Items & Decisions

Save Kestrel as the primary reference for consolidation design work and Heron as the
secondary reference for retrieval and tool routing."""

_CANNED_LABEL = "Choosing Kestrel primary reference for consolidation design work"

_CANNED_ENTITIES = [
    EntityResult("Kestrel", "SYSTEM", "Streaming memory consolidation paper"),
    EntityResult("R. Marlow", "PERSON", "First author of the Kestrel paper"),
]


def _gen_summary_id() -> str:
    return secrets.token_hex(4)


class ScriptedLLMClient(LLMClient):
    def __init__(
        self,
        *,
        script=None,
        always_tool_call: bool = False,
        scripted_summary: str = _CANNED_SUMMARY,
        scripted_label: str = _CANNED_LABEL,
        scripted_entities=None,
        raise_on_extract: bool = False,
        scripted_augment=None,
        answer: str = "This is the scripted final answer.",
        tool_name: str = "get_current_time",
        agent_model: str = AGENT_MODEL,
    ):
        # script: list of behaviors consumed per chat call. Each element is either
        # {"type": "answer", "content": str} or {"type": "tool", "name": str, "args": dict}.
        self.script = list(script or [])
        self.always_tool_call = always_tool_call
        self.tool_name = tool_name
        self.scripted_summary = scripted_summary
        self.scripted_label = scripted_label
        self.scripted_entities = scripted_entities if scripted_entities is not None else _CANNED_ENTITIES
        self.raise_on_extract = raise_on_extract
        self.scripted_augment = scripted_augment or (
            "Augmented description of this tool for semantic retrieval.",
            ["find papers", "search agent memory", "lookup tool", "use tool", "help tool"],
        )
        self.answer = answer
        self.agent_model = agent_model

        self.captured_inputs: list[dict] = []
        self.tool_call_count = 0

    def chat(self, *, system, messages, tools=None, model=None):
        self.captured_inputs.append({"system": system, "messages": messages, "tools": tools})

        if self.script:
            step = self.script.pop(0)
            if step.get("type") == "tool":
                self.tool_call_count += 1
                return ChatResult(
                    None,
                    [ToolCall(step.get("id") or f"call_{self.tool_call_count}",
                              step.get("name", self.tool_name), step.get("args", {}))],
                )
            return ChatResult(step.get("content", self.answer), None)

        if self.always_tool_call:
            self.tool_call_count += 1
            return ChatResult(None, [ToolCall(f"call_{self.tool_call_count}", self.tool_name, {})])
        return ChatResult(self.answer, None)

    def tool_result_messages(self, tool_call, result_text, truncated, log_id):
        text = result_text
        notice = ""
        if truncated:
            notice = f"\n\n[Result truncated; full output in tool log id {log_id}]"
        return [
            {"role": "tool", "tool_call_id": tool_call.id, "content": text + notice}
        ]

    def summarize(self, transcript: str) -> SummaryResult:
        return SummaryResult(self.scripted_summary, self.scripted_label, _gen_summary_id())

    def extract_entities(self, text: str) -> list[EntityResult]:
        if self.raise_on_extract:
            raise RuntimeError("scripted extraction failure")
        return list(self.scripted_entities)

    def augment_tool(self, docstring: str, source: str) -> tuple[str, list[str]]:
        return self.scripted_augment
