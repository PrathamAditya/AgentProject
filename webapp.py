"""Web UI for the memory-aware research assistant.

A FastAPI server that wraps the agent loop with a browser chat interface: a threads
sidebar, a memory panel, and per-thread conversation. Run with: python -m webapp
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

_turn_lock = threading.Lock()


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    final_answer: str
    steps: list[str]
    completed: bool
    budget_status: str | None = None
    offloaded: bool = False
    summary_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    from embeddings import get_embedder
    from memory.manager import MemoryManager
    from llm.openai_client import OpenAILLMClient
    from agent import Agent

    embedder = get_embedder()
    manager = MemoryManager()
    llm = OpenAILLMClient()
    # Augment tool descriptions only when a key is available, so the UI starts instantly
    # offline and still enriches descriptions live.
    agent = Agent(manager, llm, embedder, augment_tools=bool(os.environ.get("OPENAI_API_KEY")))
    app.state.embedder = embedder
    app.state.manager = manager
    app.state.agent = agent
    yield
    manager.close()


app = FastAPI(title="Memory-Aware Research Assistant", lifespan=lifespan)


@app.get("/api/threads")
def list_threads():
    return [dict(t) for t in app.state.manager.conversational.list_threads()]


@app.get("/api/threads/{thread_id}/messages")
def thread_messages(thread_id: str):
    return app.state.manager.conversational.thread_messages(thread_id)


@app.post("/api/chat")
def chat(req: ChatRequest):
    thread_id = (req.thread_id or "main-01").strip()
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is empty")
    agent = app.state.agent
    try:
        with _turn_lock:  # sqlite turns are serialized
            result = agent.call_agent(thread_id, message)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {e.__class__.__name__}: {e}",
        ) from e
    return ChatResponse(
        thread_id=result["thread_id"],
        final_answer=result["final_answer"],
        steps=result["steps"],
        completed=result["completed"],
        budget_status=result["budget_status"],
        offloaded=result["offloaded"],
        summary_id=result["summary_id"],
    )


@app.get("/api/memory")
def memory_overview():
    m = app.state.manager
    return {
        "counts": {
            "conversational": m.conversational.count(),
            "knowledge_base": m.knowledge_base.count(),
            "workflow": m.workflow.count(),
            "toolbox": m.toolbox.count(),
            "entity": m.entity.count(),
            "summary": m.summary.count(),
        },
        "kb_latest": [
            {"source": r["source"], "preview": r["text"][:120]}
            for r in m.knowledge_base.all()[-8:]
        ],
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp:app", host="127.0.0.1", port=8000, reload=False)
