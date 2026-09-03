"""Single home for every config value the spec pins (R15).

Every tunable is defined in exactly one place here. Distance strategy,
toolbox k, and token budget are each single-sourced (R15, AC17).
"""

import os

# --- distance strategy (single-sourced, R15/K2, AC17) ---
DISTANCE_STRATEGY = "cosine"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    n = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return n / (na * nb)


# --- model / provider (D4) ---
AGENT_MODEL = "gpt-5-mini"
AUGMENTATION_MODEL = "gpt-5"
EXTRACTION_MODEL = "gpt-5"

# token budget: 256000 for the default model, 128000 fallback (R6)
DEFAULT_TOKEN_BUDGET = 256000
FALLBACK_TOKEN_BUDGET = 128000


def resolve_token_budget(model: str) -> int:
    if model == AGENT_MODEL:
        return DEFAULT_TOKEN_BUDGET
    return FALLBACK_TOKEN_BUDGET


# budget thresholds (R6)
BUDGET_OK = 0.50
BUDGET_WARNING = 0.79
BUDGET_CRITICAL = 0.80

# per-segment retrieval defaults (R2)
CONVERSATION_LIMIT = 10
KB_K = 3
WORKFLOW_K = 3
ENTITY_K = 5
SUMMARY_K = 10

# toolbox retrieval (R4) -- single value
TOOLBOX_K = 5

# tool logging (R10)
TOOL_RESULT_EXCERPT_LIMIT = 3000
TOOL_LOG_PREVIEW_BYTES = 2000

# summarization (R8)
SUMMARY_INPUT_CAP = 6000
SUMMARY_COMPLETION_TOKENS = 4000
LABEL_COMPLETION_TOKENS = 2000
SUMMARY_ID_LEN = 8
GENERIC_LABELS = {
    "conversation summary",
    "conversation",
    "summary",
    "thread summary",
    "chat summary",
}
_SUMMARY_ID_HEX = "0123456789abcdef"

# entity extraction (R12)
ENTITY_INPUT_CAP = 500
ENTITY_COMPLETION_TOKENS = 2000

# workflow (R11)
ANSWER_EXCERPT_LIMIT = 200

# agent loop (R13)
MAX_ITERATIONS = 10
INABILITY_MESSAGE = (
    "I was unable to complete this request within the allowed number of steps. "
    "Please try rephrasing or breaking the request into smaller parts."
)

# chunking for deep ingest (R14 / CTX-C7)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# abstract cap for arxiv candidates (R14 / CTX-C7)
ABSTRACT_CAP = 2500

# store names (single-valued course constants)
CONVERSATIONAL_MEMORY = "CONVERSATIONAL_MEMORY"
TOOL_LOG_MEMORY = "TOOL_LOG_MEMORY"
SEMANTIC_MEMORY = "SEMANTIC_MEMORY"  # knowledge base
WORKFLOW_MEMORY = "WORKFLOW_MEMORY"
TOOLBOX_MEMORY = "TOOLBOX_MEMORY"
ENTITY_MEMORY = "ENTITY_MEMORY"
SUMMARY_MEMORY = "SUMMARY_MEMORY"

# db file
DB_PATH = os.environ.get("AGENT_MEMORY_DB", "agent_memory.db")

# embedding model (section 2)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
