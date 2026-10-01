# Wall-clock budgets for latency tests. Shared CI runners are noisy, so CI scales every budget
# with STELLIUM_SLA_SCALE instead of letting a 0.7 ms p99 blip fail an unrelated change.
import os


def budget_ms(local_ms: float) -> float:
    return local_ms * float(os.environ.get("STELLIUM_SLA_SCALE", "1"))
