# Kestrel: Streaming Memory Consolidation for Long-Horizon Agents

**Authors:** R. Marlow, T. Iversen (2025)
**Id:** KX-2025-011
**Claim:** Tiered summary ledgers cut resumption errors by 41% on synthetic long-horizon tasks.

## Abstract Notes

Kestrel addresses the perennial problem of long-horizon agents drifting out of
coherence as their working context grows beyond what a single session can hold.
The authors observe that naive approaches to context management tend to fall into
one of two failure modes: either the agent eagerly evicts information and later
regrets the loss, or it hoards everything and silently degrades latency and cost.

Their solution is a tiered summary-ledger architecture. Rather than maintaining a
single continuously-growing transcript, the system maintains several summary
levels, each capturing the conversation at a different resolution. A lower tier
holds fine-grained detail about the most recent turns, while higher tiers hold
increasingly compressed distillations of older activity. When a user returns to a
session after a long pause, the agent reconstructs the relevant thread by walking
down the ledger from the coarsest to the finest level, pulling in only the detail
the current task actually needs.

## Evaluation Highlights

The paper reports experiments on a suite of synthetic long-horizon task
benchmarks designed to stress resumption behavior. Agents built on the tiered
ledger cut resumption errors by 41% compared with a baseline that used a single
flat rolling window. The authors also measure token spend, showing that the
ledger approach keeps cumulative context costs roughly flat even as task length
grows, because stale detail is never reloaded into the window unless it is needed.

## Key Takeaways

This work is a useful reference for anyone building memory-aware agents that must
survive interrupted or long-running sessions. The central lesson — that
consolidation must remain recoverable, with the original content reachable by id —
has direct design implications for the storage layer behind the agent loop.
