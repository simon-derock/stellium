# Compositional benchmark generation on an invented corpus: one answer per question, ties skipped.
from __future__ import annotations

import json
from pathlib import Path

from scripts.build_compositional_set import build


def _article(
    doc_id: str, title: str, competitors: int, gold: str, venue: str = "Hall A"
) -> dict[str, object]:
    text = (
        "[Infobox Olympic event]\n"
        f"  competitors: {competitors}\n  nations: {competitors // 2}\n"
        f"  gold: {gold}\n  venue: {venue}\n\nBody text."
    )
    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "url": "",
        "wikidata_qid": doc_id,
        "wikipedia_pageid": 1,
        "approx_tokens": 40,
    }


def _corpus(tmp_path: Path) -> Path:
    articles = [
        # Rowing 2004: a clear winner by competitor count.
        _article("R1", "Rowing at the 2004 Summer Olympics – Men's eight", 40, "Ada Stone"),
        _article("R2", "Rowing at the 2004 Summer Olympics – Men's pair", 30, "Bo Reed"),
        _article("R3", "Rowing at the 2004 Summer Olympics – Women's pair", 20, "Cy Moss"),
        _article("R4", "Rowing at the 2004 Summer Olympics – Women's eight", 10, "Di Lake"),
        # Judo 2008: a tie at the top, so no question may be generated for it.
        _article("J1", "Judo at the 2008 Summer Olympics – Men's 60 kg", 35, "Ed Fox"),
        _article("J2", "Judo at the 2008 Summer Olympics – Men's 66 kg", 35, "Fay Oak"),
        _article("J3", "Judo at the 2008 Summer Olympics – Men's 73 kg", 20, "Gil Ash"),
        _article("J4", "Judo at the 2008 Summer Olympics – Men's 81 kg", 15, "Hal Elm"),
    ]
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(a) + "\n" for a in articles))
    return path


def test_build_asks_two_step_questions_with_one_answer_each(tmp_path: Path) -> None:
    questions = build(str(_corpus(tmp_path)), per_template=5, seed=7)

    # Only the untied Rowing 2004 group qualifies, once per template.
    assert [q["qtype"] for q in questions] == [
        "compositional_gold",
        "compositional_venue",
        "compositional_nations",
    ]
    assert all(
        "rowing event with the most competitors at the 2004 Summer" in q["question"]
        for q in questions
    )
    assert [q["answer"] for q in questions] == [["Ada Stone"], ["Hall A"], ["20"]]
    assert all(q["gold_doc_ids"] == ["R1"] for q in questions)


def test_build_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    corpus = str(_corpus(tmp_path))
    assert build(corpus, per_template=5, seed=7) == build(corpus, per_template=5, seed=7)
