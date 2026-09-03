"""REPL chat loop for the memory-aware research assistant (D1)."""

from __future__ import annotations

import sys

from dotenv import load_dotenv


def main():
    load_dotenv()
    from embeddings import get_embedder
    from memory.manager import MemoryManager
    from llm.openai_client import OpenAILLMClient
    from agent import Agent

    embedder = get_embedder()
    manager = MemoryManager()
    llm = OpenAILLMClient()
    agent = Agent(manager, llm, embedder)

    thread_id = input("Thread id [default: main-01]: ").strip() or "main-01"
    print("Memory-aware research assistant. Type 'exit' or Ctrl-C to quit.")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break
        result = agent.call_agent(thread_id, query)
        print("\n" + result["final_answer"] + "\n")


if __name__ == "__main__":
    main()
