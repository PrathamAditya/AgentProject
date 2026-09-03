"""Context budget monitor (R6). Token estimate = len(chars)//4 against the model's
budget; status ok (<50%), warning (50-79%), critical (>=80%). Budget injectable so
tests can shrink it (R6 [H], AC6).
"""

from config import (
    BUDGET_OK,
    BUDGET_WARNING,
    BUDGET_CRITICAL,
    resolve_token_budget,
)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


class BudgetMonitor:
    def __init__(self, budget: int | None = None, model: str | None = None):
        self.budget = budget if budget is not None else resolve_token_budget(model)
        self.model = model

    def estimate(self, text: str) -> int:
        return estimate_tokens(text)

    def status_for(self, tokens: int) -> str:
        fraction = tokens / self.budget
        if fraction >= BUDGET_CRITICAL:
            return "critical"
        if fraction >= BUDGET_OK:
            return "warning"
        return "ok"

    def is_critical(self, tokens: int) -> bool:
        return self.status_for(tokens) == "critical"
