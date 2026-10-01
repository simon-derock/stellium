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
    re.compile(r"jina_[0-9a-zA-Z]{20,}"),
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


_GSQL_FORBIDDEN_TOKENS = {
    "alter",
    "call",
    "create",
    "delete",
    "drop",
    "exec",
    "export",
    "file",
    "grant",
    "import",
    "install",
    "insert",
    "load",
    "revoke",
    "run",
    "schema",
    "system",
    "truncate",
    "update",
    "upsert",
    "while",
}
_GSQL_ALLOWED_GRAPH_OBJECTS = {
    "chunk",
    "document",
    "event",
    "venue",
    "has_chunk",
    "documented_in",
    "held_at",
    "precedes",
    "succeeds",
    "olympicsgraph",
}
_GSQL_ALLOWED_ATTRIBUTES = {
    "chunk_id",
    "doc_id",
    "chunk_index",
    "section_title",
    "text",
    "raw_text",
    "prev_chunk_id",
    "next_chunk_id",
    "filter_mask",
    "title",
    "url",
    "wikidata_qid",
    "wikipedia_pageid",
    "approx_tokens",
    "event_id",
    "name",
    "year",
    "season",
    "sport",
    "gender",
    "venue",
    "competitor_count",
    "nation_count",
    "gold_athlete",
    "silver_athlete",
    "bronze_athlete",
    "gold_noc",
    "silver_noc",
    "bronze_noc",
    "prev_event_id",
    "next_event_id",
    "valid_from",
    "valid_to",
    "superseded_by",
    "source_authority",
    "venue_id",
    "start_date",
    "end_date",
    "time_diff",
}
_GSQL_ALLOWED_WORDS = {
    "accum",
    "and",
    "asc",
    "as",
    "avgaccum",
    "by",
    "count",
    "countaccum",
    "desc",
    "else",
    "end",
    "false",
    "for",
    "from",
    "graph",
    "group",
    "if",
    "in",
    "interpret",
    "int",
    "listaccum",
    "like",
    "limit",
    "lower",
    "maxaccum",
    "minaccum",
    "not",
    "or",
    "order",
    "post_accum",
    "print",
    "query",
    "select",
    "setaccum",
    "sumaccum",
    "true",
    "where",
}
_GSQL_SAFE_PREFIX = re.compile(
    r"^\s*INTERPRET\s+QUERY\s*\(\s*\)\s+FOR\s+GRAPH\s+OlympicsGraph\s*\{",
    re.I,
)
_GSQL_STRING = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'")
_GSQL_FIELD_ACCESS = re.compile(r"\.\s*([A-Za-z_]\w*)")


def normalize_gsql_string_quotes(query: str) -> tuple[str, bool]:
    # TigerGraph's interpreted-query endpoint requires double-quoted string literals.
    # Convert only complete, unescaped single-quoted literals; leave ambiguous input untouched.
    parts: list[str] = []
    index = 0
    changed = False
    while index < len(query):
        char = query[index]
        if char == '"':
            end = index + 1
            while end < len(query):
                if query[end] == "\\":
                    end += 2
                    continue
                if query[end] == '"':
                    end += 1
                    break
                end += 1
            parts.append(query[index:end])
            index = end
            continue
        if char != "'":
            parts.append(char)
            index += 1
            continue

        end = index + 1
        while end < len(query) and query[end] not in "'\\\n\r":
            end += 1
        if end >= len(query) or query[end] != "'":
            return query, False
        literal = query[index + 1 : end]
        if '"' in literal:
            return query, False
        parts.extend(('"', literal, '"'))
        index = end + 1
        changed = True

    return ("".join(parts), True) if changed else (query, False)


def validate_generated_gsql(query: str) -> tuple[bool, str]:
    # Restrict generated ad hoc GSQL to bounded, read-only queries over public corpus types.
    if not query or len(query) > 20_000:
        return False, "Generated GSQL is empty or exceeds the query size limit"
    if not _GSQL_SAFE_PREFIX.match(query):
        return False, "Generated GSQL must target OlympicsGraph with INTERPRET QUERY"
    if not re.search(r"\}\s*$", query):
        return False, "Generated GSQL must end at the query body"
    if "//" in query or "/*" in query or "*/" in query or "\\" in query:
        return False, "Comments and escape sequences are not permitted in generated GSQL"

    string_values = _GSQL_STRING.findall(query)
    remainder = _GSQL_STRING.sub('""', query)
    if any(any(char in value for char in ";{}\n\r") for value in string_values):
        return False, "String literals contain prohibited query delimiters"

    words = [word.casefold() for word in re.findall(r"[A-Za-z_]\w*", remainder)]
    if any(word in _GSQL_FORBIDDEN_TOKENS for word in words):
        return False, "Generated GSQL contains a prohibited command"
    declared_identifiers = {
        name.casefold() for name in re.findall(r"(?:^|[;{}])\s*(\w+)\s*=", remainder)
    }
    declared_identifiers.update(name.casefold() for name in re.findall(r":\s*(\w+)", remainder))
    declared_identifiers.update(
        name.casefold() for name in re.findall(r"\bAS\s+(\w+)", remainder, re.I)
    )
    declared_identifiers.update(name.casefold() for name in re.findall(r"@@(\w+)", remainder))
    allowed_identifiers = (
        _GSQL_ALLOWED_WORDS
        | _GSQL_ALLOWED_GRAPH_OBJECTS
        | _GSQL_ALLOWED_ATTRIBUTES
        | declared_identifiers
    )
    if any(word not in allowed_identifiers for word in words):
        return False, "Generated GSQL contains an unsupported identifier or keyword"
    graph_objects = re.findall(r"\{\s*(\w+)\s*\.\s*\*\s*\}", remainder)
    graph_objects.extend(re.findall(r"-\s*\(\s*(\w+)", remainder))
    graph_objects.extend(re.findall(r"->\s*(\w+)\s*:", remainder))
    declared_sets = {
        name.casefold() for name in re.findall(r"(\w+)\s*=\s*\{\s*\w+\s*\.\s*\*\s*\}", remainder)
    }
    from_objects = re.findall(r"\bFROM\s+(\w+)\s*:\s*\w+", remainder, re.I)
    if any(
        item.casefold() not in _GSQL_ALLOWED_GRAPH_OBJECTS and item.casefold() not in declared_sets
        for item in graph_objects + from_objects
    ):
        return False, "Generated GSQL references an unsupported vertex or edge type"
    if any(
        field.casefold() not in _GSQL_ALLOWED_ATTRIBUTES
        for field in _GSQL_FIELD_ACCESS.findall(remainder)
    ):
        return False, "Generated GSQL accesses an unsupported vertex or edge attribute"
    # Parenthesized Boolean groups such as `AND (...)` are not function calls.
    functions = [
        function.casefold()
        for function in re.findall(r"\b(\w+)\s*\(", remainder)
        if function.casefold() not in {"and", "or", "not"}
    ]
    if any(function not in {"interpret", "query", "lower", "count"} for function in functions):
        return False, "Generated GSQL calls an unsupported function"

    if remainder.count("{") != remainder.count("}") or remainder.count("{") < 1:
        return False, "Generated GSQL has unbalanced query delimiters"
    if len(re.findall(r"\bSELECT\b", remainder, re.I)) != 1:
        return False, "Generated GSQL must contain exactly one bounded SELECT"
    if not re.search(r"\bPRINT\b", remainder, re.I):
        return False, "Generated GSQL must return its result with PRINT"
    limits = re.findall(r"\bLIMIT\s+(\d+)\b", remainder, re.I)
    if not limits or any(int(limit) > 2500 for limit in limits):
        return False, "Generated GSQL must cap results at 2,500 rows"
    if (
        re.search(r"-\s*\(\s*HELD_AT\s*:", remainder, re.I)
        and not re.search(r"\bORDER\s+BY\b", remainder, re.I)
        and max(map(int, limits)) < 20
    ):
        return (
            False,
            "Venue/date queries must return up to 20 candidates so ambiguous matches can be resolved",
        )
    if remainder.count(";") > 4:
        return False, "Generated GSQL contains too many statements"
    return True, "ok"


def sanitize_output(text: str) -> str:
    # Redact credential patterns from LLM output before returning to client.
    result = text
    for pat in _LEAK_PATTERNS:
        result = pat.sub("[REDACTED]", result)
    return result
