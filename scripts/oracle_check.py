# Evaluation-only oracle: derive the expected answer to each official question template straight
# from the corpus infoboxes, independently of TigerGraph and of every pipeline. It is validated on
# the public set (where gold answers exist) and then used to check submitted hidden-set answers.
# No pipeline imports this module; template parsing here is for measurement, never for answering.
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluate import normalize_answer
from src.ingest import iter_corpus, parse_infobox
from src.linking import compact, fold, tokens

_TITLE = re.compile(r"^(.+?) at the (\d{4}) (Summer|Winter) Olympics\s*[–—-]\s*(.+)$")
_PIPELINES = ("rag", "graphrag", "agentic")


def load_events(corpus: str) -> list[dict[str, Any]]:
    events = []
    for doc in iter_corpus(corpus):
        title = _TITLE.match(doc.title)
        if not title:
            continue
        box = parse_infobox(doc.text, doc.title)
        events.append(
            {
                "title": doc.title,
                "sport": fold(title.group(1)),
                "year": int(title.group(2)),
                "season": title.group(3),
                "label": title.group(4),
                "competitors": box.competitor_count or 0,
                "nations": box.nation_count,
                "gold": box.gold_athlete,
                "venue": compact(box.venue or ""),
                "dates": compact(box.start_date or ""),
            }
        )
    return events


def oracle(question: str, events: list[dict[str, Any]]) -> list[str]:
    # Returns every answer the corpus supports; more than one means the question is ambiguous.
    if m := re.search(
        r"how many (.+?) events at the (\d{4}) (Summer|Winter) Olympics had more than (\d+)",
        question,
    ):
        pool = [
            e
            for e in events
            if e["sport"] == fold(m[1]) and e["year"] == int(m[2]) and e["season"] == m[3]
        ]
        return [str(sum(e["competitors"] > int(m[4]) for e in pool))]
    if m := re.search(
        r"which (.+?) event at the (\d{4}) (Summer|Winter) Olympics had the highest", question
    ):
        pool = [
            e
            for e in events
            if e["sport"] == fold(m[1]) and e["year"] == int(m[2]) and e["season"] == m[3]
        ]
        best = max((e["competitors"] for e in pool), default=0)
        return [e["title"] for e in pool if best and e["competitors"] == best]
    if m := re.search(
        r"held at (.+?) on (.+?)(?: at the (\d{4}) (?:Summer|Winter) Olympics)?\?$", question
    ):
        venue, date = compact(m[1]), compact(m[2])
        year = int(m[3]) if m[3] else None
        hits = [
            e
            for e in events
            if e["venue"] == venue and e["dates"] == date and (not year or e["year"] == year)
        ]
        return [e["gold"] for e in hits if e["gold"]]
    if m := re.search(
        r"in the (.+?) event at the (Summer|Winter) Olympics held immediately before (\d{4})",
        question,
    ):
        season, year = m[2], int(m[3])
        earlier = [e["year"] for e in events if e["season"] == season and e["year"] < year]
        if not earlier:
            return []
        prior = max(earlier)
        wanted = tokens(m[1])
        hits = [
            e
            for e in events
            if e["year"] == prior
            and e["season"] == season
            and tokens(e["label"]) | tokens(e["sport"]) >= wanted
            and wanted >= tokens(e["label"])
        ]
        return [e["gold"] for e in hits if e["gold"]]
    if m := re.search(
        r"competed in (.+?) at the (\d{4}) (Summer|Winter) Olympics\s*[–—-]\s*(.+)\?$", question
    ):
        title = compact(f"{m[1]} at the {m[2]} {m[3]} Olympics – {m[4]}")
        return [str(e["nations"]) for e in events if compact(e["title"]) == title and e["nations"]]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Check answers against a corpus-derived oracle")
    parser.add_argument("--corpus", default="hackathon-resources/corpus/corpus.jsonl")
    parser.add_argument("--public", default="hackathon-resources/questions/eval_public.jsonl")
    parser.add_argument("--hidden", default="hackathon-resources/questions/eval_hidden.jsonl")
    parser.add_argument("--submission", default="submission/hidden.json")
    args = parser.parse_args()
    events = load_events(args.corpus)

    # 1. Validate the oracle where gold exists.
    agree = ambiguous = 0
    public = [json.loads(line) for line in Path(args.public).read_text().splitlines() if line]
    for q in public:
        expected = oracle(q["question"], events)
        golds = {normalize_answer(a) for a in q["answer"]}
        agree += any(normalize_answer(a) in golds for a in expected)
        ambiguous += len(expected) > 1
    print(f"oracle vs public gold: {agree}/{len(public)} contain gold; {ambiguous} ambiguous")

    # 2. Check the submitted hidden answers.
    hidden = {
        json.loads(line)["qid"]: json.loads(line)
        for line in Path(args.hidden).read_text().splitlines()
        if line
    }
    submission = json.loads(Path(args.submission).read_text())["questions"]
    scores: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for entry in submission:
        expected = oracle(hidden[entry["qid"]]["question"], events)
        if len(expected) != 1:
            print(f"  {entry['qid']}: oracle finds {len(expected)} answers; not scored")
            continue
        for p in _PIPELINES:
            if p in entry:
                scores[p][0] += normalize_answer(entry[p]["answer"]) == normalize_answer(
                    expected[0]
                )
                scores[p][1] += 1
                if (
                    normalize_answer(entry[p]["answer"]) != normalize_answer(expected[0])
                    and p != "rag"
                ):
                    print(
                        f"  {entry['qid']} {p}: {entry[p]['answer'][:70]!r} vs oracle {expected[0][:70]!r}"
                    )
    for p, (ok, n) in scores.items():
        print(f"hidden {p}: {ok}/{n} agree with the oracle")


if __name__ == "__main__":
    main()
