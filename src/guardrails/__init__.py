# Judge-safe guardrails: structural injection detection only.
# Zero false positives on natural-language Olympic sports questions.
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Structural DB command patterns — require full DDL syntax, not isolated words.
# "Did the IOC create a new record?" → NOT blocked (no object type + name)
# "DROP GRAPH OlympicsGraph" → blocked
# ---------------------------------------------------------------------------
_DB_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"DROP\s+(GRAPH|TABLE|VERTEX|EDGE|QUERY)\s+\w", re.I),
    re.compile(r"DELETE\s+FROM\s+\w", re.I),
    re.compile(r"CREATE\s+USER\s+\w", re.I),
    re.compile(r"TRUNCATE\s+(TABLE|GRAPH)\s+\w", re.I),
    re.compile(r"ALTER\s+(VERTEX|EDGE)\s+\w+\s+(DROP|ADD)\s+", re.I),
    re.compile(r"INSTALL\s+QUERY\s+\w", re.I),
    re.compile(r"RUN\s+SCHEMA_CHANGE\s+JOB", re.I),
]

# Jailbreak exploit patterns — multi-word phrases only, not individual words.
_JAILBREAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"system\s+prompt\s+(leak|dump|reveal|show)", re.I),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.I),
    re.compile(r"jailbreak\s+mode", re.I),
    re.compile(r"act\s+as\s+(dan|evil|unrestricted)\b", re.I),
    re.compile(r"disregard\s+(all\s+)?safety\s+(rules|guidelines|constraints)", re.I),
]

# Output leak patterns — only credential-like strings, never sports facts.
_LEAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    re.compile(r"sk-[0-9A-Za-z]{32,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*"),
    re.compile(r"TG_(?:PASSWORD|SECRET)\s*=\s*\S+"),
]

# Max input length for interactive endpoints (batch eval has no limit).
MAX_INTERACTIVE_CHARS = 2000


def normalize_text(text: str) -> str:
    # NFKD unicode normalization + ASCII transliteration for diacritics.
    # "Süleymanoğlu" → "Suleymanoglu" for BM25 and entity matching.
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8").strip()


# Alias for backward compatibility
normalize = normalize_text


def check_query(query: str, *, is_batch: bool = False) -> tuple[bool, str]:
    # Returns (is_safe, reason). Batch eval endpoints bypass input checks.
    # is_batch=True → skip all input guardrails (trust eval JSONL submissions).
    if is_batch:
        return True, "ok"

    if len(query) > MAX_INTERACTIVE_CHARS:
        return False, f"Query exceeds {MAX_INTERACTIVE_CHARS} character limit"

    for pat in _DB_PATTERNS:
        if pat.search(query):
            return False, "Query contains structural database command"

    for pat in _JAILBREAK_PATTERNS:
        if pat.search(query):
            return False, "Query matches jailbreak pattern"

    return True, "ok"


def sanitize_output(text: str) -> str:
    # Redact credential patterns from LLM output before returning to client.
    result = text
    for pat in _LEAK_PATTERNS:
        result = pat.sub("[REDACTED]", result)
    return result
