# Shared in-memory graph seeded with invented Olympic events (never benchmark data).
from __future__ import annotations

from typing import Any

from src.coprocessor import Coprocessor
from src.graph import GraphClient
from src.graph.mock import MockTigerGraphConnection, create_mock_graph_client
from src.linking import EventCatalog
from src.models import Chunk
from src.pipelines.toolkit import GraphToolkit

EVENTS: list[dict[str, Any]] = [
    # id, title, competitors, gold, venue, dates, previous edition id
    {
        "id": "R04M",
        "name": "Rowing at the 2004 Summer Olympics – Men's single sculls",
        "competitors": 30,
        "gold": "Ada Stone",
        "venue": "Lake A",
        "dates": "15 to 21 August",
    },
    {
        "id": "R04W",
        "name": "Rowing at the 2004 Summer Olympics – Women's single sculls",
        "competitors": 24,
        "gold": "Bea Moss",
        "venue": "Lake A",
        "dates": "15 to 21 August",
    },
    {
        "id": "R08M",
        "name": "Rowing at the 2008 Summer Olympics – Men's single sculls",
        "competitors": 33,
        "gold": "Cal Reed",
        "venue": "Lake B",
        "dates": "9–16 August",
        "prev": "R04M",
    },
    {
        "id": "R08W",
        "name": "Rowing at the 2008 Summer Olympics – Women's single sculls",
        "competitors": 33,
        "gold": "Dee Lake",
        "venue": "Lake B",
        "dates": "9–17 August",
        "prev": "R04W",
    },
    {
        "id": "R08X",
        "name": "Rowing at the 2008 Summer Olympics – Men's eight",
        "competitors": 0,
        "gold": "Team Nine",
        "venue": "Lake B",
        "dates": "10–17 August",
    },
    {
        "id": "R12W",
        "name": "Rowing at the 2012 Summer Olympics – Women's single sculls",
        "competitors": 28,
        "gold": "Eve Wren",
        "venue": "Lake C",
        "dates": "28 July – 3 August",
    },
]


def seeded_graph() -> tuple[GraphClient, EventCatalog, Coprocessor]:
    conn = MockTigerGraphConnection()
    chunks = []
    for event in EVENTS:
        year = int(event["name"].split(" at the ")[1][:4])
        gender = "Women" if "Women" in event["name"] else "Men"
        conn.upsertVertex(
            "Event",
            event["id"],
            {
                "name": event["name"],
                "year": year,
                "season": "Summer",
                "sport": "Rowing",
                "gender": gender,
                "venue": event["venue"],
                "competitor_count": event["competitors"],
                "nation_count": event["competitors"] // 2,
                "gold_athlete": event["gold"],
            },
        )
        conn.upsertVertex("Document", event["id"], {"wikidata_qid": event["id"]})
        conn.upsertEdge("Event", event["id"], "DOCUMENTED_IN", "Document", event["id"])
        conn.upsertEdge(
            "Event", event["id"], "HELD_AT", "Venue", "v", {"start_date": event["dates"]}
        )
        if "prev" in event:
            conn.upsertEdge("Event", event["id"], "PRECEDES", "Event", event["prev"])
        text = f"Gold: {event['gold']}\nThere were {event['competitors']} competitors."
        chunks.append(
            Chunk(
                chunk_id=f"{event['id']}#0",
                doc_id=event["id"],
                chunk_index=0,
                section_title=event["name"],
                text=f"Title: {event['name']} | Gold: {event['gold']}\n\n{text}",
                raw_text=text,
            )
        )
    coprocessor = Coprocessor()
    coprocessor.build(chunks)
    graph = create_mock_graph_client(conn)
    return graph, EventCatalog.from_graph(conn), coprocessor


def seeded_toolkit() -> GraphToolkit:
    return GraphToolkit(*seeded_graph())
