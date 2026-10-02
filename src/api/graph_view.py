# Projects the graph-loaded event catalog into the node/edge shape the graph canvas draws.
# Every node is an Event, a Venue, or a grouping read off Event attributes (Games, sport), and
# every edge says where it came from, so the picture never shows a relation the graph lacks.
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from src.linking import EventCatalog, EventRecord, compact
from src.models import GraphEdgeDTO, GraphNodeDTO, SnapshotDTO

VIEWS = ("constellation", "venues", "lineage", "investigation")

# A query neighbourhood stays readable: the focused events, their Games, sport and venue,
# their editions either side, and a capped set of rivals from the same pool.
_RIVALS_PER_FOCUS = 12
_LINEAGE_LIMIT = 400


def _weight(count: int) -> float:
    # Node radius grows with sqrt(weight) on the canvas; log keeps the 400-event Games readable.
    return round(1 + 2.2 * math.log1p(count), 2)


def _games_id(record: EventRecord) -> str:
    return f"games:{record.year}-{record.season.lower()}"


def _sport_id(sport: str) -> str:
    return f"sport:{compact(sport)}"


def _venue_id(venue: str) -> str:
    return f"venue:{compact(venue)}"


def _edition_key(record: EventRecord) -> tuple[str, str, str]:
    # The same competition across Games: sport, season and the event label after the dash.
    return compact(record.sport), record.season, compact(record.label)


class _Builder:
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNodeDTO] = {}
        self.edges: dict[str, GraphEdgeDTO] = {}

    def node(
        self,
        node_id: str,
        type_: str,
        layer: str,
        label: str,
        count: int = 1,
        confidence: float = 1.0,
    ) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = GraphNodeDTO(
                id=node_id,
                type=type_,
                layer=layer,
                label=label,
                weight=_weight(count),
                source_count=count,
                confidence=confidence,
            )

    def games(self, record: EventRecord, count: int = 1) -> str:
        node_id = _games_id(record)
        self.node(node_id, "Games", "games", f"{record.year} {record.season}", count)
        return node_id

    def sport(self, sport: str, count: int = 1) -> str:
        node_id = _sport_id(sport)
        self.node(node_id, "Sport", "sport", sport, count)
        return node_id

    def venue(self, venue: str, count: int = 1) -> str:
        node_id = _venue_id(venue)
        self.node(node_id, "Venue", "venue", venue, count)
        return node_id

    def event(self, record: EventRecord, layer: str = "event", label: str | None = None) -> str:
        self.node(record.event_id, "Event", layer, label or record.label)
        return record.event_id

    def edge(self, source: str, target: str, relationship: str, provenance: str) -> None:
        edge_id = f"{source}|{relationship}|{target}"
        if edge_id not in self.edges and source != target:
            self.edges[edge_id] = GraphEdgeDTO(
                id=edge_id,
                source=source,
                target=target,
                relationship_type=relationship,
                provenance_label=provenance,
            )

    def snapshot(self, disclosure: str) -> SnapshotDTO:
        return SnapshotDTO(
            graph_nodes=list(self.nodes.values()),
            graph_edges=list(self.edges.values()),
            disclosure=disclosure,
        )


def _editions(catalog: EventCatalog) -> dict[tuple[str, str, str], list[EventRecord]]:
    chains: dict[tuple[str, str, str], list[EventRecord]] = defaultdict(list)
    for record in catalog.records:
        if record.year and record.label:
            chains[_edition_key(record)].append(record)
    for chain in chains.values():
        chain.sort(key=lambda record: record.year)
    return chains


def constellation(catalog: EventCatalog) -> SnapshotDTO:
    # Every Games and every sport, joined wherever the Games held at least one event in it.
    build = _Builder()
    per_games = Counter(_games_id(record) for record in catalog.records if record.year)
    per_sport = Counter(record.sport for record in catalog.records if record.sport)
    pairs = Counter(
        (_games_id(record), record.sport)
        for record in catalog.records
        if record.year and record.sport
    )
    for record in catalog.records:
        if record.year and record.sport:
            games = build.games(record, per_games[_games_id(record)])
            sport = build.sport(record.sport, per_sport[record.sport])
            build.edge(
                games, sport, "HELD_SPORT", f"{pairs[(games, record.sport)]} events, Event.sport"
            )
    return build.snapshot(
        f"{len(per_games)} Games and {len(per_sport)} sports across {len(catalog.records)} events."
    )


def venues(catalog: EventCatalog) -> SnapshotDTO:
    build = _Builder()
    per_venue = Counter(record.venue for record in catalog.records if record.venue)
    per_games = Counter(_games_id(record) for record in catalog.records if record.year)
    for record in catalog.records:
        if record.venue and record.year:
            games = build.games(record, per_games[_games_id(record)])
            venue = build.venue(record.venue, per_venue[record.venue])
            build.edge(venue, games, "HOSTED", "Event HELD_AT Venue")
    return build.snapshot(f"{len(per_venue)} venues linked to the Games they hosted.")


def lineage(catalog: EventCatalog, sport: str | None = None) -> SnapshotDTO:
    # One sport's events chained edition to edition, the relation PRECEDES stores in the graph.
    resolved = catalog.resolve_sport(sport) if sport else None
    busiest = Counter(record.sport for record in catalog.records if record.sport).most_common(1)
    chosen = resolved or (busiest[0][0] if busiest else "")
    build = _Builder()
    shown = 0
    for chain in _editions(catalog).values():
        if chain[0].sport != chosen or len(chain) < 2:
            continue
        for earlier, later in zip(chain, chain[1:], strict=False):
            if shown >= _LINEAGE_LIMIT:
                break
            for record in (earlier, later):
                build.event(record, label=f"{record.label} {record.year}")
                build.edge(
                    record.event_id, build.games(record), "AT_GAMES", "Event.year, Event.season"
                )
            build.edge(later.event_id, earlier.event_id, "PRECEDES", "previous edition")
            shown += 1
    return build.snapshot(f"{chosen}: each event linked to its previous edition.")


def investigation(catalog: EventCatalog, focus: Sequence[str]) -> SnapshotDTO:
    # The neighbourhood a pipeline actually touched, plus the pool it was ranked against.
    build = _Builder()
    chains = _editions(catalog)
    records = [record for event_id in dict.fromkeys(focus) if (record := catalog.get(event_id))]
    # Focused events claim their layer first, so a rival of one focus still reads as a focus.
    for record in records:
        build.event(record, layer="evidence", label=record.name)
    for record in records:
        event = record.event_id
        build.edge(event, build.games(record), "AT_GAMES", "Event.year, Event.season")
        build.edge(event, build.sport(record.sport), "IN_SPORT", "Event.sport")
        if record.venue:
            build.edge(event, build.venue(record.venue), "HELD_AT", "Event HELD_AT Venue")
        chain = chains.get(_edition_key(record), [])
        at = next((i for i, other in enumerate(chain) if other.event_id == record.event_id), -1)
        for neighbour in _neighbours(chain, at):
            build.event(neighbour, label=f"{neighbour.label} {neighbour.year}")
            older, newer = sorted((record, neighbour), key=lambda r: r.year)
            build.edge(newer.event_id, older.event_id, "PRECEDES", "previous edition")
        rivals = catalog.events(sport=record.sport, year=record.year, season=record.season)
        for rival in _capped(rivals, record.event_id):
            build.event(rival)
            build.edge(rival.event_id, build.games(rival), "AT_GAMES", "Event.year, Event.season")
    return build.snapshot(
        f"{len(records)} focused events with their Games, venue, editions and rivals."
        if records
        else "No focused events: run a question to see the subgraph it used."
    )


def _neighbours(chain: list[EventRecord], at: int) -> Iterable[EventRecord]:
    if at < 0:
        return []
    return [chain[i] for i in (at - 1, at + 1) if 0 <= i < len(chain)]


def _capped(rivals: list[EventRecord], skip: str) -> list[EventRecord]:
    return [rival for rival in rivals if rival.event_id != skip][:_RIVALS_PER_FOCUS]


def project(
    catalog: EventCatalog, view: str, focus: Sequence[str] = (), sport: str | None = None
) -> SnapshotDTO:
    if view == "venues":
        return venues(catalog)
    if view == "lineage":
        return lineage(catalog, sport)
    if view == "investigation":
        return investigation(catalog, focus)
    return constellation(catalog)
