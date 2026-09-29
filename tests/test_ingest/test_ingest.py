# Unit tests for infobox parsing, name splitting, and chunking with prev/next pointers.
from src.ingest import (
    _split_athlete_names,
    chunk_document,
    parse_infobox,
)
from src.models import CorpusDoc

SAMPLE_TEXT = """[Infobox Olympic event]
  event: Men's canoe sprint K-2 1,000 metres
  games: 2012 Summer
  venue: Eton Dorney
  date: 6 to 8 August
  competitors: 24
  nations: 12
  gold: Rudolf DombiRoland Kökény
  goldNOC: HUN
  silver: Fernando PimentaEmanuel Silva
  silverNOC: POR
  bronze: Andreas IhleMartin Hollstein
  bronzeNOC: GER
  prev: 2008
  next: 2016

The men's canoe sprint K-2 1,000 metres competition at the 2012 Olympic Games in London
took place between 6 and 8 August at Eton Dorney.

== Competition format ==
The competition comprised heats, semifinals, and a final."""


def test_infobox_extraction() -> None:
    infobox = parse_infobox(SAMPLE_TEXT)
    assert infobox.year == 2012
    assert infobox.season == "Summer"
    assert infobox.gender == "Men"
    assert infobox.venue == "Eton Dorney"
    assert infobox.competitor_count == 24
    assert infobox.nation_count == 12
    assert infobox.prev_year == 2008
    assert infobox.next_year == 2016
    assert infobox.gold_noc == "HUN"
    assert infobox.gold_athlete == "Rudolf Dombi, Roland Kökény"


def test_infobox_preserves_internal_capital_when_article_confirms_full_name() -> None:
    text = """[Infobox Olympic event]
  event: Women's trampoline
  games: 2012 Summer
  gold: Rosannagh MacLennan
  goldNOC: CAN

Results
1 | He Wenna | CHN
4 | Rosannagh MacLennan | CAN
"""

    parsed = parse_infobox(text)

    assert parsed.gold_athlete == "Rosannagh MacLennan"


def test_infobox_gender_uses_explicit_event_title_qualifier() -> None:
    parsed = parse_infobox(
        "[Infobox Olympic event]\n  event: 100 metres",
        title="Athletics at the 2012 Summer Olympics – Women's 100 metres",
    )
    assert parsed.gender == "Women"


def test_split_athlete_names() -> None:
    # Concatenated title-case names should be cleanly split
    names = _split_athlete_names("Rudolf DombiRoland Kökény")
    assert "Rudolf Dombi" in names
    assert "Roland Kökény" in names

    names_silver = _split_athlete_names("Fernando PimentaEmanuel Silva")
    assert "Fernando Pimenta" in names_silver
    assert "Emanuel Silva" in names_silver

    assert _split_athlete_names("Rosannagh MacLennan") == ["Rosannagh MacLennan"]
    assert _split_athlete_names("Rosannagh MacLennanOther Person") == [
        "Rosannagh MacLennan",
        "Other Person",
    ]
    assert _split_athlete_names("Michel AndrieuxJean-Christophe Rolland") == [
        "Michel Andrieux",
        "Jean-Christophe Rolland",
    ]


def test_chunking_doubly_linked_pointers() -> None:
    doc = CorpusDoc(
        doc_id="Q1000",
        wikidata_qid="Q1000",
        wikipedia_pageid=123,
        title="Canoeing Men K-2",
        url="https://en.wikipedia.org/wiki/test",
        text=SAMPLE_TEXT,
        approx_tokens=200,
    )
    chunks = chunk_document(doc)
    assert len(chunks) >= 1

    # Chunk 0 has metadata header and parsed infobox
    assert chunks[0].chunk_id == "Q1000#0"
    assert chunks[0].infobox is not None
    assert "Year: 2012" in chunks[0].text
    assert "Venue: Eton Dorney" in chunks[0].text

    # Verify pointer consistency
    for i, c in enumerate(chunks):
        if i == 0:
            assert c.prev_chunk_id is None
        else:
            assert c.prev_chunk_id == f"Q1000#{i - 1}"

        if i == len(chunks) - 1:
            assert c.next_chunk_id is None
        else:
            assert c.next_chunk_id == f"Q1000#{i + 1}"
