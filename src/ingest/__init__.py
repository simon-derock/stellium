# Table-aware corpus parser and chunker.
# Parses Wikipedia Olympic event articles from corpus.jsonl.
# Extracts infobox fields, creates doubly-linked chunks with metadata headers.
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path

from src.models import Chunk, CorpusDoc, ParsedInbox

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Optimal chunk size for Olympic articles (they are 1000-3000 chars each).
# Most articles fit in 1-2 chunks. Avoid splitting mid-infobox.
CHUNK_SIZE_TOKENS = 400  # approximate tokens per chunk
CHUNK_OVERLAP = 40  # overlap tokens between chunks

# Average chars per token for Wikipedia English text
_CHARS_PER_TOKEN = 4


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Unicode / Text Normalization
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    # NFKD normalization for diacritics: "Süleymanoğlu" → "Suleymanoglu"
    # Applied consistently to ALL text before BM25 indexing and entity matching.
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8").strip()


# ---------------------------------------------------------------------------
# Infobox Parser
# ---------------------------------------------------------------------------

# Maps common infobox field name variants to canonical field names.
_FIELD_ALIASES: dict[str, str] = {
    "games": "games",
    "year": "year",
    "sport": "sport",
    "event": "event",
    "venue": "venue",
    "dates": "dates",
    "date": "dates",
    "competitors": "competitors",
    "competitor": "competitors",
    "nations": "nations",
    "nation": "nations",
    "gold": "gold",
    "silver": "silver",
    "bronze": "bronze",
    "goldnoc": "gold_noc",
    "silvernoc": "silver_noc",
    "bronzenoc": "bronze_noc",
    "prev": "prev",
    "previous": "prev",
    "next": "next",
}

_INFOBOX_START = re.compile(r"\[Infobox Olympic event\]", re.I)
_FIELD_LINE = re.compile(r"^\s*([^:=|]+?)\s*[:=|]\s*(.+)$")
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
_NOC_RE = re.compile(r"\(([A-Z]{2,3})\)\s*$")

# Splits concatenated athlete names at title-case boundaries.
# e.g. "Rudolf DombiRoland Kökény" → ["Rudolf Dombi", "Roland Kökény"]
_NAME_SPLIT_RE = re.compile(r"(?<=[a-züöäåæøéèà])(?=[A-ZÜÖÄÅÆØÉÈÀ])")


def _split_athlete_names(raw: str) -> list[str]:
    # Handle both space-separated and concatenated name formats.
    parts = _NAME_SPLIT_RE.split(raw.strip())
    return [p.strip() for p in parts if p.strip()]


def parse_infobox(text: str) -> ParsedInbox:
    # Extract structured fields from the [Infobox Olympic event] header block.
    # Returns ParsedInbox with None values for missing fields.
    fields: dict[str, str] = {}
    in_infobox = False

    for line in text.splitlines():
        if _INFOBOX_START.search(line):
            in_infobox = True
            continue
        if in_infobox:
            # Infobox ends at first blank line after start.
            if not line.strip():
                break
            m = _FIELD_LINE.match(line)
            if m:
                raw_key = m.group(1).lower().strip()
                val = m.group(2).strip()
                canonical = _FIELD_ALIASES.get(raw_key)
                if canonical:
                    fields[canonical] = val

    # Extract year from "games" field or any 4-digit year found.
    year: int | None = None
    games_str = fields.get("games", "")
    year_match = _YEAR_RE.search(games_str)
    if year_match:
        year = int(year_match.group(1))

    # Extract season
    season: str | None = None
    if "summer" in games_str.lower():
        season = "Summer"
    elif "winter" in games_str.lower():
        season = "Winter"

    # Extract competitor and nation counts
    def _parse_int(raw: str | None) -> int | None:
        if not raw:
            return None
        m = re.search(r"\d+", raw)
        return int(m.group()) if m else None

    # Parse prev/next years
    prev_year: int | None = None
    next_year: int | None = None
    prev_str = fields.get("prev", "")
    next_str = fields.get("next", "")
    prev_m = _YEAR_RE.search(prev_str)
    next_m = _YEAR_RE.search(next_str)
    if prev_m:
        prev_year = int(prev_m.group(1))
    if next_m:
        next_year = int(next_m.group(1))

    # Extract NOC codes from gold/silver/bronze fields
    def _noc(raw: str | None) -> str | None:
        if not raw:
            return None
        m = _NOC_RE.search(raw)
        return m.group(1) if m else None

    def _athlete(raw: str | None) -> str | None:
        if not raw:
            return None
        # Strip NOC code at end
        cleaned = _NOC_RE.sub("", raw).strip()
        athletes = _split_athlete_names(cleaned)
        return ", ".join(athletes) if athletes else cleaned

    return ParsedInbox(
        year=year,
        season=season,
        sport=fields.get("sport"),
        event_name=fields.get("event"),
        venue=fields.get("venue"),
        start_date=fields.get("dates"),
        competitor_count=_parse_int(fields.get("competitors")),
        nation_count=_parse_int(fields.get("nations")),
        gold_athlete=_athlete(fields.get("gold")),
        silver_athlete=_athlete(fields.get("silver")),
        bronze_athlete=_athlete(fields.get("bronze")),
        gold_noc=fields.get("gold_noc") or _noc(fields.get("gold")),
        silver_noc=fields.get("silver_noc") or _noc(fields.get("silver")),
        bronze_noc=fields.get("bronze_noc") or _noc(fields.get("bronze")),
        prev_year=prev_year,
        next_year=next_year,
    )


# ---------------------------------------------------------------------------
# Metadata Header Injection
# ---------------------------------------------------------------------------


def _build_metadata_header(doc: CorpusDoc, infobox: ParsedInbox) -> str:
    # Deterministic metadata header prepended to first chunk.
    # Provides structured signals to both the embedding model and BM25 index.
    # Zero LLM tokens — pure field extraction + formatting.
    parts = [f"Title: {doc.title}"]
    if infobox.year:
        parts.append(f"Year: {infobox.year}")
    if infobox.season:
        parts.append(f"Season: {infobox.season}")
    if infobox.sport:
        parts.append(f"Sport: {infobox.sport}")
    if infobox.event_name:
        parts.append(f"Event: {infobox.event_name}")
    if infobox.venue:
        parts.append(f"Venue: {infobox.venue}")
    if infobox.competitor_count is not None:
        parts.append(f"Competitors: {infobox.competitor_count}")
    if infobox.nation_count is not None:
        parts.append(f"Nations: {infobox.nation_count}")
    if infobox.gold_athlete:
        parts.append(f"Gold: {infobox.gold_athlete}")
    if infobox.silver_athlete:
        parts.append(f"Silver: {infobox.silver_athlete}")
    if infobox.bronze_athlete:
        parts.append(f"Bronze: {infobox.bronze_athlete}")
    if infobox.prev_year:
        parts.append(f"Prev Edition: {infobox.prev_year}")
    if infobox.next_year:
        parts.append(f"Next Edition: {infobox.next_year}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def chunk_document(doc: CorpusDoc) -> list[Chunk]:
    # Splits a document into chunks.
    # Chunk 0 always includes the full infobox text + injected metadata header.
    # Subsequent chunks are sliding window splits of the body text.
    infobox = parse_infobox(doc.text)
    header = _build_metadata_header(doc, infobox)

    # Split text into sections at double-newlines (Wikipedia paragraph structure).
    # This prevents splitting mid-sentence across paragraphs.
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", doc.text) if p.strip()]

    # Group paragraphs into chunks of ~CHUNK_SIZE_TOKENS each.
    chunks: list[Chunk] = []
    current_paras: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_tokens(para)
        if current_tokens + para_tokens > CHUNK_SIZE_TOKENS and current_paras:
            chunks.append(
                _make_chunk(
                    doc,
                    len(chunks),
                    current_paras,
                    header if not chunks else None,
                    infobox if not chunks else None,
                )
            )
            # Overlap: keep last paragraph for context continuity
            current_paras = current_paras[-1:] + [para]
            current_tokens = _approx_tokens(current_paras[0]) + para_tokens
        else:
            current_paras.append(para)
            current_tokens += para_tokens

    # Final chunk with remaining paragraphs
    if current_paras:
        chunks.append(
            _make_chunk(
                doc,
                len(chunks),
                current_paras,
                header if not chunks else None,
                infobox if not chunks else None,
            )
        )

    # If document is tiny (fits in zero paragraphs somehow), create one chunk.
    if not chunks:
        chunks.append(_make_chunk(doc, 0, [doc.text], header, infobox))

    # Set doubly-linked chunk pointers
    n = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.prev_chunk_id = f"{doc.doc_id}#{i - 1}" if i > 0 else None
        chunk.next_chunk_id = f"{doc.doc_id}#{i + 1}" if i < n - 1 else None

    return chunks


def _make_chunk(
    doc: CorpusDoc,
    index: int,
    paras: list[str],
    header: str | None,
    infobox: ParsedInbox | None,
) -> Chunk:
    raw_text = "\n\n".join(paras)
    # Prepend metadata header to first chunk only for richer embedding signal.
    text = f"{header}\n\n{raw_text}" if header else raw_text

    return Chunk(
        chunk_id=f"{doc.doc_id}#{index}",
        doc_id=doc.doc_id,
        chunk_index=index,
        section_title=doc.title,
        text=text,
        raw_text=raw_text,
        infobox=infobox if index == 0 else None,
        filter_mask=0,  # Filled in by build_filter_mask() after infobox parse
    )


# ---------------------------------------------------------------------------
# Corpus Reader
# ---------------------------------------------------------------------------


def iter_corpus(corpus_path: str | Path) -> Iterator[CorpusDoc]:
    # Stream 2,951 documents from corpus.jsonl without loading all into memory.
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield CorpusDoc.model_validate_json(line)


def load_all_chunks(corpus_path: str | Path) -> list[Chunk]:
    # Parse all 2,951 documents and return flat list of all chunks.
    all_chunks: list[Chunk] = []
    for doc in iter_corpus(corpus_path):
        all_chunks.extend(chunk_document(doc))
    return all_chunks
