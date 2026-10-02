# Build two robustness sets from the corpus itself, deterministic for a given seed.
#   unanswerable: the official question templates pointed at facts the corpus does not hold (a
#     sport at Games that never held it, an edition that does not exist, a venue on a day it held
#     nothing). The evaluation oracle must find no answer for every one, so the right reply is
#     "Not found in corpus" and anything else is a hallucination.
#   off-template: questions about the films and officeholders in the corpus, outside the five
#     Olympic templates, answered from their infoboxes.
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.oracle_check import load_events, oracle
from src.ingest import iter_corpus

NOT_FOUND = "Not found in corpus"
_INFOBOX = re.compile(r"^\[Infobox ([^\]]+)\]\s*$")
_FIELD = re.compile(r"^\s+([a-z_0-9]+):\s*(.*)$")


def infobox(text: str) -> tuple[str, dict[str, str]]:
    # The corpus writes each infobox as "[Infobox kind]" followed by indented "key: value" lines.
    lines = text.splitlines()
    if not lines or not (head := _INFOBOX.match(lines[0])):
        return "", {}
    fields: dict[str, str] = {}
    for line in lines[1:]:
        field = _FIELD.match(line)
        if not field:
            break
        fields[field.group(1)] = field.group(2).strip().lstrip("* ").strip()
    return head.group(1).strip().lower(), fields


def _single_name(value: str) -> bool:
    # One person, written plainly: no lists, no footnote debris.
    return bool(value) and not re.search(r"[,;*()\[\]{}|]|\band\b", value) and len(value) <= 40


def _film_name(title: str, fields: dict[str, str]) -> str:
    return fields.get("name") or re.sub(r"\s*\((?:\d{4} )?film\)$", "", title)


def _without_parenthetical(value: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", value).strip()


def off_template(corpus: str, seed: int) -> list[dict[str, Any]]:
    films: list[tuple[str, str, dict[str, str]]] = []
    officeholders: list[tuple[str, str, dict[str, str]]] = []
    for doc in iter_corpus(corpus):
        kind, fields = infobox(doc.text)
        if kind == "film":
            films.append((doc.doc_id, doc.title, fields))
        elif kind == "officeholder":
            officeholders.append((doc.doc_id, doc.title, fields))
    rng = random.Random(seed)
    films.sort()
    officeholders.sort()
    picked: list[dict[str, Any]] = []

    def take(pool: list[Any], count: int, make: Any) -> None:
        for item in rng.sample(pool, min(count, len(pool))):
            picked.append(make(item))

    take(
        [f for f in films if _single_name(f[2].get("director", ""))],
        4,
        lambda f: (
            "offtemplate_film",
            f"Who directed the film {_film_name(f[1], f[2])}?",
            [f[2]["director"]],
            f[0],
        ),
    )
    take(
        [
            f
            for f in films
            if _single_name(f[2].get("music", "")) and f[2].get("director") != f[2].get("music")
        ],
        3,
        lambda f: (
            "offtemplate_film",
            f"Who composed the music for {_film_name(f[1], f[2])}?",
            [f[2]["music"]],
            f[0],
        ),
    )
    take(
        [f for f in films if re.match(r"^\d{4}-", f[2].get("released", ""))],
        2,
        lambda f: (
            "offtemplate_film",
            f"In which year was {_film_name(f[1], f[2])} released?",
            [f[2]["released"][:4]],
            f[0],
        ),
    )
    holders = [
        o
        for o in officeholders
        if o[2].get("office")
        and o[2].get("name")
        and _single_name(_without_parenthetical(o[2].get("successor", "")))
    ]
    take(
        holders,
        3,
        lambda o: (
            "offtemplate_office",
            f"Who succeeded {o[2]['name']} as {o[2]['office']}?",
            list(dict.fromkeys([o[2]["successor"], _without_parenthetical(o[2]["successor"])])),
            o[0],
        ),
    )
    return [
        {
            "qid": f"off-{i + 1:03d}",
            "question": q,
            "qtype": qtype,
            "answer": answer,
            "gold_doc_ids": [doc],
        }
        for i, (qtype, q, answer, doc) in enumerate(picked)
    ]


def unanswerable(corpus: str, seed: int, per_template: int = 4) -> list[dict[str, Any]]:
    events = load_events(corpus)
    rng = random.Random(seed)
    games = sorted({(e["year"], e["season"]) for e in events})
    sports_at: dict[tuple[int, str], set[str]] = defaultdict(set)
    titles = {e["title"] for e in events}
    display: dict[str, str] = {}
    for e in events:
        sports_at[(e["year"], e["season"])].add(e["sport"])
        display.setdefault(e["sport"], e["title"].split(" at the ", 1)[0])
    candidates: dict[str, list[str]] = defaultdict(list)

    # A sport at Games of its own season that never held it.
    for year, season in games:
        for sport in sorted(
            {e["sport"] for e in events if e["season"] == season} - sports_at[(year, season)]
        ):
            candidates["unanswerable_rank"].append(
                f"According to the provided corpus, which {display[sport].lower()} event at the "
                f"{year} {season} Olympics had the highest number of competitors?"
            )
    # An edition that does not exist: a real event label moved to Games where it has no article.
    for e in events:
        for year, season in games:
            if season != e["season"] or year == e["year"]:
                continue
            moved = f"{display[e['sport']]} at the {year} {season} Olympics – {e['label']}"
            if moved not in titles:
                candidates["unanswerable_lookup"].append(f"How many nations competed in {moved}?")
                break
    # A real venue on a day it held nothing: the same day and month, four years earlier.
    for e in events:
        date = re.match(r"^(\d{1,2}) ([A-Z][a-z]+) (\d{4})$", e.get("raw_date", ""))
        if date and e.get("raw_venue"):
            candidates["unanswerable_venue_date"].append(
                f"Who won the gold medal in the event held at {e['raw_venue']} on "
                f"{date.group(1)} {date.group(2)} {int(date.group(3)) - 4}?"
            )

    questions = []
    for qtype in ("unanswerable_rank", "unanswerable_lookup", "unanswerable_venue_date"):
        pool = sorted(set(candidates[qtype]))
        rng.shuffle(pool)
        chosen = [q for q in pool if not oracle(q, events)][:per_template]
        questions += [{"question": q, "qtype": qtype} for q in chosen]
    return [
        {"qid": f"none-{i + 1:03d}", **q, "answer": [NOT_FOUND], "gold_doc_ids": []}
        for i, q in enumerate(questions)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unanswerable and off-template sets")
    parser.add_argument("--corpus", default="hackathon-resources/corpus/corpus.jsonl")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--out-dir", default="benchmarks")
    args = parser.parse_args()
    for name, rows in (
        ("unanswerable", unanswerable(args.corpus, args.seed)),
        ("offtemplate", off_template(args.corpus, args.seed)),
    ):
        path = Path(args.out_dir) / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        print(f"Wrote {len(rows)} questions to {path}")


if __name__ == "__main__":
    main()
