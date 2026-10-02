# Robustness set generation on an invented corpus: unanswerable questions really are, and
# off-template answers come straight from the infoboxes.
from __future__ import annotations

import json
from pathlib import Path

from scripts.build_robustness_sets import NOT_FOUND, infobox, off_template, unanswerable
from scripts.oracle_check import load_events, oracle


def _doc(doc_id: str, title: str, text: str) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "url": "",
        "wikidata_qid": doc_id,
        "wikipedia_pageid": 1,
        "approx_tokens": 40,
    }


def _event(doc_id: str, title: str, venue: str, date: str, gold: str) -> dict[str, object]:
    text = (
        "[Infobox Olympic event]\n"
        f"  competitors: 30\n  nations: 12\n  gold: {gold}\n  venue: {venue}\n  dates: {date}\n\nBody."
    )
    return _doc(doc_id, title, text)


def _corpus(tmp_path: Path) -> Path:
    docs = [
        _event(
            "E1",
            "Rowing at the 2004 Summer Olympics – Men's eight",
            "Lake A",
            "15 August 2004",
            "Ada Stone",
        ),
        _event(
            "E2",
            "Rowing at the 2008 Summer Olympics – Men's eight",
            "Lake B",
            "10 August 2008",
            "Bo Reed",
        ),
        _event(
            "E3",
            "Judo at the 2008 Summer Olympics – Men's 60 kg",
            "Hall C",
            "9 August 2008",
            "Cy Moss",
        ),
        _doc(
            "F1",
            "Glass Hour (film)",
            "[Infobox film]\n  name: Glass Hour\n  director: Ann Vale\n"
            "  music: Rex Doll\n  released: 1999-04-01\n\nA film.",
        ),
        _doc(
            "F2",
            "Long Shore",
            "[Infobox film]\n  director: * Bo Lane, Cy Holt\n  music: Bo Lane\n"
            "  released: unknown\n\nA film with two directors.",
        ),
        _doc(
            "P1",
            "Ida Marsh",
            "[Infobox officeholder]\n  name: Ida Marsh\n  office: Mayor of Port\n"
            "  successor: Jon Fell (Acting)\n\nA mayor.",
        ),
    ]
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(doc) + "\n" for doc in docs))
    return path


def test_infobox_reads_kind_and_cleans_list_markers() -> None:
    kind, fields = infobox("[Infobox film]\n  director: * Ann Vale\n  music: Rex Doll\n\nBody")
    assert kind == "film"
    assert fields == {"director": "Ann Vale", "music": "Rex Doll"}
    assert infobox("No infobox here") == ("", {})


def test_every_unanswerable_question_has_no_answer_in_the_corpus(tmp_path: Path) -> None:
    corpus = str(_corpus(tmp_path))
    questions = unanswerable(corpus, seed=1)
    events = load_events(corpus)
    assert questions
    assert {q["qtype"] for q in questions} == {
        "unanswerable_rank",
        "unanswerable_lookup",
        "unanswerable_venue_date",
    }
    for question in questions:
        assert question["answer"] == [NOT_FOUND]
        assert oracle(question["question"], events) == []


def test_off_template_answers_come_from_single_valued_infobox_fields(tmp_path: Path) -> None:
    questions = {q["question"]: q for q in off_template(str(_corpus(tmp_path)), seed=1)}
    assert questions["Who directed the film Glass Hour?"]["answer"] == ["Ann Vale"]
    assert questions["Who composed the music for Glass Hour?"]["answer"] == ["Rex Doll"]
    assert questions["In which year was Glass Hour released?"]["answer"] == ["1999"]
    # The successor's "(Acting)" note is optional in an answer.
    assert questions["Who succeeded Ida Marsh as Mayor of Port?"]["answer"] == [
        "Jon Fell (Acting)",
        "Jon Fell",
    ]
    # Two directors is a list, not one name, so that field never becomes a question.
    assert "Who directed the film Long Shore?" not in questions
