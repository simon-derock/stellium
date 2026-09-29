from __future__ import annotations


def normalize_gender_filter(value: str) -> str:
    # Convert a possessive question form into the category stored on Event vertices.
    normalized = value.strip().lower().replace("’", "'")
    if normalized.endswith("'s"):
        normalized = normalized[:-2].strip()
    return normalized
