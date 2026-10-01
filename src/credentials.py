# Cohere credential pool shared by chat and embeddings.
# Keys come only from the environment and are never logged; callers see aliases. A key whose
# monthly allowance is spent is parked for the rest of the process, so requests move straight
# to the next key instead of retrying a call that cannot succeed until the month rolls over.
from __future__ import annotations

import os
import re
import threading

_NAMED_ALIASES = ("COHERE_CHAT_API_KEY", "COHERE", "COHERE_BACKUP", "COHERE_KEY")
_NUMBERED_ALIAS = re.compile(r"COHERE_KEY_[A-Z0-9_]+")
_MONTHLY_CAP = re.compile(r"calls\s*/\s*month|per month|monthly", re.IGNORECASE)

_exhausted: set[str] = set()
_lock = threading.Lock()


def cohere_keys() -> list[tuple[str, str]]:
    # Named aliases first, then COHERE_KEY_* variables, then a comma-separated COHERE_KEYS list.
    candidates = [(name, os.environ.get(name, "")) for name in _NAMED_ALIASES]
    candidates += sorted(
        (name, value) for name, value in os.environ.items() if _NUMBERED_ALIAS.fullmatch(name)
    )
    candidates += [
        (f"COHERE_KEYS[{index}]", value)
        for index, value in enumerate(os.environ.get("COHERE_KEYS", "").split(","))
    ]
    configured: list[tuple[str, str]] = []
    seen: set[str] = set()
    for alias, raw in candidates:
        key = raw.strip()
        if key and key not in seen:
            seen.add(key)
            configured.append((alias, key))
    return configured


def available_cohere_keys() -> list[tuple[str, str]]:
    with _lock:
        return [(alias, key) for alias, key in cohere_keys() if key not in _exhausted]


def is_monthly_cap(status_code: int, body: str) -> bool:
    # Trial keys answer 429 with a message naming the monthly call limit; per-minute throttles
    # also use 429 but recover after a short wait, so only the monthly message parks a key.
    return status_code == 429 and bool(_MONTHLY_CAP.search(str(body)))


def park_exhausted_key(key: str) -> None:
    with _lock:
        _exhausted.add(key)


def parked_keys(keys: list[str]) -> set[str]:
    with _lock:
        return {key for key in keys if key in _exhausted}


def reset_exhausted_keys() -> None:
    # Tests and long-lived servers can clear parked keys after the monthly window resets.
    with _lock:
        _exhausted.clear()
