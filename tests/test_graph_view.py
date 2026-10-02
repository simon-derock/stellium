# Graph canvas projections over the seeded catalog: every edge joins drawn nodes and names its source.
from __future__ import annotations

import pytest

from src.api.graph_view import VIEWS, project
from src.linking import EventCatalog
from src.models import SnapshotDTO
from tests.graph_fixtures import seeded_graph


@pytest.fixture(scope="module")
def catalog() -> EventCatalog:
    return seeded_graph()[1]


def _ids(snapshot: SnapshotDTO, type_: str | None = None) -> set[str]:
    return {node.id for node in snapshot.graph_nodes if type_ is None or node.type == type_}


def _edges(snapshot: SnapshotDTO, relationship: str) -> set[tuple[str, str]]:
    return {
        (edge.source, edge.target)
        for edge in snapshot.graph_edges
        if edge.relationship_type == relationship
    }


@pytest.mark.parametrize("view", VIEWS)
def test_every_edge_joins_drawn_nodes_and_says_where_it_came_from(
    catalog: EventCatalog, view: str
) -> None:
    snapshot = project(catalog, view, focus=["R08W"])
    ids = _ids(snapshot)
    assert snapshot.graph_nodes
    for edge in snapshot.graph_edges:
        assert edge.source in ids and edge.target in ids
        assert edge.provenance_label


def test_constellation_joins_each_games_to_its_sports(catalog: EventCatalog) -> None:
    snapshot = project(catalog, "constellation")
    assert _ids(snapshot, "Games") == {
        "games:2004-summer",
        "games:2008-summer",
        "games:2012-summer",
    }
    assert _ids(snapshot, "Sport") == {"sport:rowing"}
    held = {edge.source: edge.provenance_label for edge in snapshot.graph_edges}
    assert held["games:2008-summer"].startswith("3 events")


def test_venues_link_to_the_games_they_hosted(catalog: EventCatalog) -> None:
    snapshot = project(catalog, "venues")
    assert _edges(snapshot, "HOSTED") == {
        ("venue:lakea", "games:2004-summer"),
        ("venue:lakeb", "games:2008-summer"),
        ("venue:lakec", "games:2012-summer"),
    }


def test_lineage_chains_each_event_to_its_previous_edition(catalog: EventCatalog) -> None:
    snapshot = project(catalog, "lineage", sport="rowing")
    assert _edges(snapshot, "PRECEDES") == {
        ("R08M", "R04M"),
        ("R08W", "R04W"),
        ("R12W", "R08W"),
    }
    # An event with no other edition has nothing to chain to.
    assert "R08X" not in _ids(snapshot)


def test_investigation_centres_on_the_focused_event(catalog: EventCatalog) -> None:
    snapshot = project(catalog, "investigation", focus=["R08W", "R08W", "missing"])
    layers = {node.id: node.layer for node in snapshot.graph_nodes}
    assert layers["R08W"] == "evidence"
    assert {"games:2008-summer", "sport:rowing", "venue:lakeb"} <= set(layers)
    # Both editions either side, and the rest of the 2008 rowing pool it was ranked against.
    assert {("R08W", "R04W"), ("R12W", "R08W")} == _edges(snapshot, "PRECEDES")
    assert {"R08M", "R08X"} <= set(layers)
    assert snapshot.disclosure.startswith("1 focused event")


def test_a_rival_that_is_also_focused_keeps_the_focus_layer(catalog: EventCatalog) -> None:
    snapshot = project(catalog, "investigation", focus=["R08W", "R08M"])
    layers = {node.id: node.layer for node in snapshot.graph_nodes}
    assert layers["R08W"] == layers["R08M"] == "evidence"


def test_investigation_without_focus_says_so(catalog: EventCatalog) -> None:
    snapshot = project(catalog, "investigation")
    assert not snapshot.graph_nodes
    assert "run a question" in snapshot.disclosure


def test_lineage_on_an_empty_catalog_is_empty() -> None:
    assert not project(EventCatalog([]), "lineage").graph_nodes
