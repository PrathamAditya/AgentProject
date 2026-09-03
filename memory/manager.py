"""Memory manager — the one abstraction over all seven stores (D7). The agent code
never touches storage details directly.
"""

import sqlite3

from config import DISTANCE_STRATEGY
from db import connect

from .conversational import ConversationalStore
from .tool_log import ToolLogStore
from .vector_store import VectorStore
from .knowledge_base import KnowledgeBaseStore
from .workflow import WorkflowStore
from .toolbox import ToolboxStore
from .entity import EntityStore
from .summary import SummaryStore


class MemoryManager:
    def __init__(self, conn: sqlite3.Connection | None = None, path: str | None = None):
        self.conn = conn or connect(path or None)
        from db import init_schema

        init_schema(self.conn)

        self.conversational = ConversationalStore(self.conn)
        self.tool_log = ToolLogStore(self.conn)
        self.knowledge_base = KnowledgeBaseStore(VectorStore(self.conn, "knowledge_base", 3))
        self.workflow = WorkflowStore(VectorStore(self.conn, "workflow_memory", 3))
        self.toolbox = ToolboxStore(VectorStore(self.conn, "toolbox_memory", 5))
        self.entity = EntityStore(VectorStore(self.conn, "entity_memory", 5))
        self.summary = SummaryStore(VectorStore(self.conn, "summary_memory", 10))

    def close(self):
        self.conn.close()

    @property
    def distance_strategy(self) -> str:
        return DISTANCE_STRATEGY

    def vector_stores(self):
        return [
            self.knowledge_base,
            self.workflow,
            self.toolbox,
            self.entity,
            self.summary,
        ]
