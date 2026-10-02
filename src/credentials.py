# Cohere credential pool shared by chat and embeddings.
# Keys come only from the environment and are never logged; callers see aliases. A key whose
# monthly allowance is spent (or that the provider rejects) is parked for the rest of the
# process, so requests move straight to the next key instead of retrying a call that cannot
# succeed. A key that is only throttled rests briefly and comes back.
# COHERE_KEY_TIERS orders the pool, e.g. "A,B|C,D": calls spread across the first tier that
# still has a live key, and fall through to the next tier only when it is spent.
from __future__ import annotations

import os
import re
import threading
import time

_NAMED_ALIASES = ("COHERE_CHAT_API_KEY", "COHERE", "COHERE_BACKUP", "COHERE_KEY")
_NUMBERED_ALIAS = re.compile(r"COHERE_KEY_[A-Z0-9_]+")
_MONTHLY_CAP = re.compile(r"calls\s*/\s*month|per month|monthly", re.IGNORECASE)

_exhausted: set[str] = set()
_resting: dict[str, float] = {}
_lock = threading.Lock()


def _tiered_aliases() -> list[list[str]]:
    raw = os.environ.get("COHERE_KEY_TIERS", "")
    tiers = [[name.strip() for name in tier.split(",") if name.strip()] for tier in raw.split("|")]
    return [tier for tier in tiers if tier]


def cohere_key_tiers() -> list[list[tuple[str, str]]]:
    # Tiers named in COHERE_KEY_TIERS come first; every other discovered key forms a last tier.
    tiers: list[list[tuple[str, str]]] = []
    seen: set[str] = set()
    for names in _tiered_aliases():
        tier = []
        for name in names:
            key = os.environ.get(name, "").strip()
            if key and key not in seen:
                seen.add(key)
                tier.append((name, key))
        if tier:
            tiers.append(tier)
    rest = [(alias, key) for alias, key in _discovered_keys() if key not in seen]
    if rest:
        tiers.append(rest)
    return tiers


def cohere_keys() -> list[tuple[str, str]]:
    return [pair for tier in cohere_key_tiers() for pair in tier]


def _discovered_keys() -> list[tuple[str, str]]:
    # Named aliases first, then COHERE_KEY_* variables, then a comma-separated COHERE_KEYS list.
    candidates = [(name, os.environ.get(name, "")) for name in _NAMED_ALIASES]
    candidates += sorted(
        (name, value)
        for name, value in os.environ.items()
        if _NUMBERED_ALIAS.fullmatch(name) and name != "COHERE_KEY_TIERS"
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
    # The first tier with an awake key; a resting key yields to any awake one, even a lower tier's.
    # When every live key is resting, the first live tier is returned so callers wait, not fail.
    now = time.monotonic()
    with _lock:
        tiers = [
            [pair for pair in tier if pair[1] not in _exhausted] for tier in cohere_key_tiers()
        ]
        for tier in tiers:
            awake = [(alias, key) for alias, key in tier if _resting.get(key, 0.0) <= now]
            if awake:
                return awake
        return next((tier for tier in tiers if tier), [])


def live_key_count() -> int:
    with _lock:
        return sum(key not in _exhausted for _, key in cohere_keys())


def rest_key(key: str, seconds: float) -> None:
    # A per-minute throttle or a provider hiccup: try the other keys first for a little while.
    with _lock:
        _resting[key] = time.monotonic() + seconds


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
        _resting.clear()
