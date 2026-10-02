# Build a small compositional benchmark from parsed corpus facts: each question needs two
# chained graph steps (rank the events of one Games, then read an attribute of the winner),
# which a single fixed lookup cannot express. Gold answers come from the same infobox data the
# graph holds; groups with tied maxima or a missing attribute are skipped so every question has
# exactly one answer. Deterministic for a given seed.
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.ingest import iter_corpus, parse_infobox

_TITLE = re.compile(r"^(.+?) at the (\d{4}) (Summer|Winter) Olympics\s*[–—-]\s*(.+)$")

# (qtype, attribute, question template); {sport}, {year}, {season} are filled per Games.
_TEMPLATES = (
    (
        "compositional_gold",
        "gold_athlete",
        "Who won the gold medal in the {sport} event with the most competitors at the "
        "{year} {season} Olympics?",
    ),
    (
        "compositional_venue",
        "venue",
        "At which venue was the {sport} event with the most competitors at the "
        "{year} {season} Olympics held?",
    ),
    (
        "compositional_nations",
        "nation_count",
        "How many nations competed in the {sport} event with the most competitors at the "
        "{year} {season} Olympics?",
    ),
)


def _games_groups(corpus: str) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for doc in iter_corpus(corpus):
        title = _TITLE.match(doc.title)
        if not title:
            continue
        infobox = parse_infobox(doc.text, doc.title)
        if not infobox.competitor_count:
            continue
        groups[(title.group(1), int(title.group(2)), title.group(3))].append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "competitors": infobox.competitor_count,
                "gold_athlete": infobox.gold_athlete,
                "venue": infobox.venue,
                "nation_count": infobox.nation_count,
            }
        )
    return groups


def build(corpus: str, per_template: int, seed: int) -> list[dict[str, Any]]:
    groups = _games_groups(corpus)
    eligible = []
    for (sport, year, season), events in sorted(groups.items()):
        if len(events) < 4:
            continue
        ranked = sorted(events, key=lambda event: -event["competitors"])
        if ranked[0]["competitors"] == ranked[1]["competitors"]:
            continue  # a tie has no single answer
        eligible.append((sport, year, season, ranked[0]))
    rng = random.Random(seed)
    questions = []
    for qtype, attribute, template in _TEMPLATES:
        candidates = [item for item in eligible if item[3].get(attribute)]
        for sport, year, season, top in rng.sample(candidates, min(per_template, len(candidates))):
            questions.append(
                {
                    "qid": f"comp-{len(questions) + 1:03d}",
                    "question": template.format(sport=sport.lower(), year=year, season=season),
                    "qtype": qtype,
                    "answer": [str(top[attribute])],
                    "gold_doc_ids": [top["doc_id"]],
                }
            )
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compositional benchmark set")
    parser.add_argument("--corpus", default="hackathon-resources/corpus/corpus.jsonl")
    parser.add_argument("--per-template", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--out", default="benchmarks/compositional.jsonl")
    args = parser.parse_args()
    questions = build(args.corpus, args.per_template, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions))
    print(f"Wrote {len(questions)} questions to {out}")


if __name__ == "__main__":
    main()
