"""Real OpenAI client implementing LLMClient (D4: gpt-5-mini loop, gpt-5 aux)."""

from __future__ import annotations

import os
import re

from config import (
    AGENT_MODEL,
    AUGMENTATION_MODEL,
    EXTRACTION_MODEL,
    ENTITY_COMPLETION_TOKENS,
    LABEL_COMPLETION_TOKENS,
    SUMMARY_COMPLETION_TOKENS,
    GENERIC_LABELS,
)
from .base import LLMClient, ChatResult, ToolCall, SummaryResult, EntityResult


def _gen_summary_id() -> str:
    import secrets

    return secrets.token_hex(4)


_FALLBACK_SUMMARY = """Technical Information

No technical content could be extracted from the provided transcript.

Emotional Context

No emotional context could be determined.

Entities & References

No entities or references could be identified.

Action Items & Decisions

No action items or decisions could be determined."""

_FALLBACK_LABEL = "Discussion about conversation research notes memory topics"


def _label_word_count(label: str) -> int:
    return len(label.split())


def _is_generic(label: str) -> bool:
    return label.strip().lower().rstrip(".") in GENERIC_LABELS


class OpenAILLMClient(LLMClient):
    def __init__(
        self,
        *,
        agent_model: str = AGENT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
        self.agent_model = agent_model

    # -- primary loop turn --
    def chat(self, *, system, messages, tools=None, model=None):
        kwargs = {}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self._client.chat.completions.create(
            model=model or self.agent_model,
            messages=[{"role": "system", "content": system}] + messages,
            **kwargs,
        )
        msg = resp.choices[0].message
        content = msg.content
        tool_calls = []
        for tc in msg.tool_calls or []:
            try:
                args = eval(tc.function.arguments or "{}")
            except Exception:
                args = {}
            tool_calls.append(ToolCall(tc.id, tc.function.name, args))
        return ChatResult(content, tool_calls or None)

    def tool_result_messages(self, tool_call, result_text, truncated, log_id):
        text = result_text
        notice = ""
        if truncated:
            notice = f"\n\n[Result truncated; full output in tool log id {log_id}]"
        return [
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": text + notice,
            }
        ]

    # -- summarization (R8) --
    def summarize(self, transcript: str) -> SummaryResult:
        summary = self._summarize_text(transcript)
        label = self._generate_label(summary)
        return SummaryResult(summary, label, _gen_summary_id())

    def _summarize_text(self, transcript: str) -> str:
        prompt = (
            "Summarize the following conversation using EXACTLY these four markdown "
            "headings, in this order:\n\n"
            "## Technical Information\n## Emotional Context\n## Entities & References\n"
            "## Action Items & Decisions\n\n"
            "Keep the content concise and factual under each heading. Do not add any "
            "headings beyond these four.\n\nCONVERSATION:\n" + transcript
        )
        try:
            resp = self._client.chat.completions.create(
                model=AUGMENTATION_MODEL,
                max_tokens=SUMMARY_COMPLETION_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text
        except Exception:
            pass
        # one simpler-prompt retry
        try:
            resp = self._client.chat.completions.create(
                model=AUGMENTATION_MODEL,
                max_tokens=SUMMARY_COMPLETION_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "In four labeled sections (Technical Information, Emotional "
                            "Context, Entities & References, Action Items & Decisions) "
                            "summarize:\n" + transcript
                        ),
                    }
                ],
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text
        except Exception:
            pass
        return _FALLBACK_SUMMARY

    def _generate_label(self, summary: str) -> str:
        prompt = (
            "Write a specific 8-12 word label for the following conversation summary. "
            "It must be specific enough to distinguish this conversation from any other; "
            'never use a generic label like "Conversation summary". Reply with only the '
            "label, no quotes.\n\nSUMMARY:\n" + summary
        )
        try:
            resp = self._client.chat.completions.create(
                model=AUGMENTATION_MODEL,
                max_tokens=LABEL_COMPLETION_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            label = (resp.choices[0].message.content or "").strip().strip('"')
            if label and not _is_generic(label) and 8 <= _label_word_count(label) <= 12:
                return label
        except Exception:
            pass
        return _FALLBACK_LABEL

    # -- entity extraction (R12) --
    def extract_entities(self, text: str) -> list[EntityResult]:
        cap = text[:500]
        prompt = (
            "Extract named entities from the text. Classify each as PERSON, PLACE, SYSTEM, "
            "or UNKNOWN. Return a JSON list with objects `{\"name\", \"type\", "
            "\"description\"}`. If none, return [].\nTEXT:\n" + cap
        )
        try:
            resp = self._client.chat.completions.create(
                model=EXTRACTION_MODEL,
                max_tokens=ENTITY_COMPLETION_TOKENS,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.choices[0].message.content or "{}"
            m = re.search(r"\[.*\]", content, re.DOTALL)
            items = []
            if m:
                items = self._safe_loads(m.group(0))
            return [
                EntityResult(
                    it.get("name", ""),
                    it.get("type", "UNKNOWN"),
                    it.get("description", ""),
                )
                for it in items
                if isinstance(it, dict) and it.get("name")
            ]
        except Exception:
            return []

    @staticmethod
    def _safe_loads(text: str):
        try:
            import json

            return json.loads(text)
        except Exception:
            return []

    # -- tool augmentation (D11) --
    def augment_tool(self, docstring: str, source: str) -> tuple[str, list[str]]:
        prompt = (
            "Rewrite the following tool docstring into a richer description (summarize what "
            "it does, its steps, when to call it, and caveats). Then generate 5 short example "
            "user queries that would naturally invoke this tool. Return JSON with keys "
            '`description` and `queries` (a list of 5 strings).\n\n'
            "DOCSTRING:\n"
            + docstring
            + "\n\nSOURCE:\n"
            + (source[:2000])
        )
        try:
            resp = self._client.chat.completions.create(
                model=AUGMENTATION_MODEL,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            import json

            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            description = data.get("description") or docstring
            queries = data.get("queries") or []
            return description, queries
        except Exception:
            return docstring, []
