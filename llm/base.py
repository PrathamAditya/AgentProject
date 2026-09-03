"""LLM client interface. The agent loop, summarizer, entity extractor, and tool
augmenter all depend on this abstraction so a scripted stand-in can drive the
offline/scripted acceptance tests without a key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ToolCall:
    def __init__(self, id: str, name: str, arguments: dict):
        self.id = id
        self.name = name
        self.arguments = arguments

    def __repr__(self):
        return f"ToolCall(id={self.id!r}, name={self.name!r}, args={self.arguments!r})"


class ChatResult:
    def __init__(self, content: str | None, tool_calls: list[ToolCall] | None):
        self.content = content
        self.tool_calls = tool_calls or []

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class SummaryResult:
    def __init__(self, summary: str, description: str, summary_id: str):
        self.summary = summary
        self.description = description
        self.summary_id = summary_id


class EntityResult:
    def __init__(self, name: str, type_: str, description: str):
        self.name = name
        self.type_ = type_
        self.description = description


class LLMClient(ABC):
    """Native (OpenAI-style) tool calling + system role + the three auxiliary ops."""

    agent_model: str

    @abstractmethod
    def chat(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
    ) -> ChatResult:
        """One reasoning turn. Returns either content or tool calls."""

    @abstractmethod
    def tool_result_messages(
        self,
        tool_call: ToolCall,
        result_text: str,
        truncated: bool,
        log_id: str | None,
    ) -> list[dict]:
        """Messages to append after a tool executes. R10: bounded excerpt + notice."""

    @abstractmethod
    def summarize(self, transcript: str) -> SummaryResult:
        """R8 structured summary: four headings, 8-char id, 8-12-word label."""

    @abstractmethod
    def extract_entities(self, text: str) -> list[EntityResult]:
        """R12 entities (PERSON/PLACE/SYSTEM) from at most 500 chars."""

    @abstractmethod
    def augment_tool(self, docstring: str, source: str) -> tuple[str, list[str]]:
        """D11: enrich a docstring into a description + ~5 synthetic queries."""
