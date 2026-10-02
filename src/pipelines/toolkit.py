# Typed graph tools shared by GraphRAG (fixed dispatch) and the agent (LLM-chosen calls).
# Names are linked in memory against the graph-loaded catalog; every answer value is read back
# from TigerGraph, so these tools cost zero LLM tokens. Each outcome says whether it is
# conclusive, which specialist handled it, and which source passage supports the value.
from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.coprocessor import Coprocessor
from src.graph import GraphClient
from src.linking import EventCatalog, EventRecord, LinkResult, fold
from src.models import EvidenceItem

ENTITY_LINKER = "EntityLinkingAgent"
AGGREGATOR = "AggregationAgent"
TRAVERSER = "GraphTraversalAgent"
EVALUATOR = "EvidenceEvaluationAgent"

# Event attributes a caller may ask for; "dates" comes from the HELD_AT edge.
EVENT_ATTRIBUTES = (
    "gold_athlete",
    "silver_athlete",
    "bronze_athlete",
    "gold_noc",
    "silver_noc",
    "bronze_noc",
    "competitor_count",
    "nation_count",
    "venue",
    "dates",
)
_MAX_LISTED_EVENTS = 25
# What kind of value each attribute and each event-level tool returns.
_ATTRIBUTE_KINDS = {
    "gold_athlete": "person",
    "silver_athlete": "person",
    "bronze_athlete": "person",
    "gold_noc": "nation",
    "silver_noc": "nation",
    "bronze_noc": "nation",
    "competitor_count": "number",
    "nation_count": "number",
    "venue": "venue",
    "dates": "date",
}
_TOOL_KINDS = {"count_events": "number", "rank_events": "event", "find_events": "event"}
# Head nouns after "which"/"what" that name the kind of answer wanted.
_HEAD_KINDS = {
    **dict.fromkeys(("venue", "venues", "stadium", "arena"), "venue"),
    **dict.fromkeys(("nation", "nations", "country", "countries", "noc"), "nation"),
    **dict.fromkeys(("athlete", "athletes", "person", "competitor", "medallist"), "person"),
    **dict.fromkeys(("date", "dates", "day"), "date"),
    **dict.fromkeys(("number", "count"), "number"),
    **dict.fromkeys(("event", "events", "competition", "discipline"), "event"),
}
# Words that close a wh-phrase: the verb or preposition after its head noun.
_PHRASE_ENDS = frozenset(
    "at in of on for with from to by was were is are had has did does do held won took".split()
)
_WH_KINDS = {"who": "person", "whom": "person", "whose": "person", "where": "venue", "when": "date"}
# Infobox fields read "  competitors: 41"; prose never starts with a short key and a colon.
_INFOBOX_LINE = re.compile(r"^\s*(?:\[infobox|[a-z_ ]{2,30}\s*[:=])", re.IGNORECASE)


_catalogs: dict[int, EventCatalog] = {}

# Typed tools and the arguments each accepts; both pipelines dispatch through this table.
TOOL_PARAMETERS: dict[str, tuple[str, ...]] = {
    "count_events": ("sport", "year", "season", "gender", "comparison", "threshold"),
    "rank_events": ("sport", "year", "season", "gender", "order"),
    "event_attribute": ("event", "attribute", "sport", "year", "season", "gender"),
    "previous_edition": ("event", "year", "attribute", "sport", "season", "gender"),
    "event_at_venue_date": ("venue", "date", "year", "attribute"),
    "find_events": ("event", "sport", "year", "season", "gender"),
}
# Models sometimes reuse argument names from older tool schemas; accept the obvious synonyms.
_ARGUMENT_ALIASES = {
    "target_year": "year",
    "current_year": "year",
    "event_name": "event",
    "event_name_fragment": "event",
    "event_fragment": "event",
    "venue_name_fragment": "venue",
    "venue_fragment": "venue",
    "target_date_fragment": "date",
    "order_by": "order",
}
_INTEGER_ARGUMENTS = {"year", "threshold"}


def catalog_for(graph: GraphClient) -> EventCatalog:
    # Read once per connection from TigerGraph and shared by every later run.
    key = id(graph.conn)
    if key not in _catalogs:
        _catalogs[key] = EventCatalog.from_graph(graph.conn)
    return _catalogs[key]


def unsupported_arguments(tool: str, raw: dict[str, Any]) -> list[str]:
    # Non-empty arguments the tool cannot use usually mean the wrong tool was chosen, e.g. an
    # event name sent to count_events when the question asks for that event's nation count.
    return [
        key
        for key, value in raw.items()
        if value not in (None, "", 0)
        and _ARGUMENT_ALIASES.get(key, key) not in TOOL_PARAMETERS[tool]
    ]


def tool_arguments(tool: str, raw: dict[str, Any]) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for key, value in raw.items():
        name = _ARGUMENT_ALIASES.get(key, key)
        if name not in TOOL_PARAMETERS[tool] or value in (None, ""):
            continue
        if name in _INTEGER_ARGUMENTS:
            try:
                arguments[name] = int(value)
            except (TypeError, ValueError):
                continue
        else:
            arguments[name] = str(value).strip()
    return arguments


@dataclass
class ToolOutcome:
    tool: str
    observation: dict[str, Any]
    answer: str | None = None
    candidates: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def conclusive(self) -> bool:
        return self.answer is not None

    @property
    def ambiguous(self) -> bool:
        return self.answer is None and len(self.candidates) > 1

    @property
    def answer_kind(self) -> str | None:
        if self.tool in _TOOL_KINDS:
            return _TOOL_KINDS[self.tool]
        return _ATTRIBUTE_KINDS.get(str(self.observation.get("attribute", "")))


def asked_kind(question: str) -> str | None:
    # Expected answer type from the wh-phrase: "how many" wants a number, "who" a person. After
    # "which"/"what" the phrase's head noun decides, i.e. its last noun before the verb or
    # preposition, so "which cross-country skiing event" asks for an event, not a country.
    # None when the wording does not say, and then the tool's value is trusted.
    words = fold(question).split()
    for index, word in enumerate(words):
        if word == "how" and words[index + 1 : index + 2] == ["many"] or word == "count":
            return "number"
        if word in _WH_KINDS:
            return _WH_KINDS[word]
        if word in {"which", "what", "name"}:
            heads: list[str | None] = [None]
            for following in words[index + 1 : index + 7]:
                if following in _PHRASE_ENDS:
                    break
                if following in _HEAD_KINDS:
                    heads.append(_HEAD_KINDS[following])
            return heads[-1]
    return None


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bounds(comparison: str, threshold: int) -> tuple[int, int]:
    # Inclusive competitor bounds. Unknown counts are stored as 0, so upper-bounded comparisons
    # start at 1 to keep events with missing data out of "fewer than" answers.
    kind = fold(comparison).replace(" ", "_")
    if kind in {"more_than", "greater_than", "over", "above"}:
        return threshold + 1, 0
    if kind in {"at_least", "minimum", "no_fewer_than"}:
        return threshold, 0
    if kind in {"fewer_than", "less_than", "under", "below"}:
        return 1, threshold - 1
    if kind in {"at_most", "maximum", "no_more_than"}:
        return 1, threshold
    if kind in {"exactly", "equal_to", "equals"}:
        return threshold, threshold
    raise ValueError(f"unsupported comparison {comparison!r}")


class GraphToolkit:
    def __init__(
        self, graph: GraphClient, catalog: EventCatalog, coprocessor: Coprocessor | None = None
    ) -> None:
        self.graph = graph
        self.catalog = catalog
        self.coprocessor = coprocessor

    def invoke(self, tool: str, raw_arguments: dict[str, Any]) -> ToolOutcome:
        # Dispatch by name with model-supplied arguments normalised to the method signature.
        method: Callable[..., ToolOutcome] = getattr(self, tool)
        return method(**tool_arguments(tool, raw_arguments))

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _constraints(
        self, sport: str = "", season: str = "", gender: str = ""
    ) -> tuple[str | None, str | None, str | None, list[str]]:
        # Resolve free-text constraints to catalog values, reporting anything that did not link.
        problems = []
        resolved_sport = self.catalog.resolve_sport(sport) if sport else None
        if sport and resolved_sport is None:
            problems.append(f"sport {sport!r} matches no graph sport")
        resolved_season = self.catalog.resolve_season(season) if season else None
        if season and resolved_season is None:
            problems.append(f"season {season!r} matches no graph season")
        resolved_gender = self.catalog.resolve_gender(gender) if gender else None
        if gender and resolved_gender is None:
            problems.append(f"gender {gender!r} matches no graph gender")
        return resolved_sport, resolved_season, resolved_gender, problems

    def _fetch_events(self, records: list[EventRecord]) -> dict[str, dict[str, Any]]:
        # One REST read for the linked vertices; attribute values always come from TigerGraph.
        if not records:
            return {}
        rows = self.graph.vertices_by_id("Event", [record.event_id for record in records])
        return {str(row["v_id"]): dict(row.get("attributes", {})) for row in rows}

    def _prose_confirming_count(self, doc_id: str, count: int) -> str | None:
        # Return the article sentence that restates a competitor count, ignoring infobox lines.
        if self.coprocessor is None:
            return None
        restated = re.compile(
            rf"\b{count}\s+(?:[a-z-]+\s+){{0,2}}(?:from|representing)\s+\d+\s+(?:nations|countries)"
            rf"|\b{count}\s+(?:competitors|athletes|entrants|participants)\b",
            re.IGNORECASE,
        )
        index = 0
        while (chunk := self.coprocessor.get_chunk(f"{doc_id}#{index}")) is not None:
            for line in chunk.raw_text.splitlines():
                if _INFOBOX_LINE.match(line):
                    continue
                for sentence in re.split(r"(?<=[.!?])\s+", line):
                    if restated.search(sentence):
                        return sentence.strip()
            index += 1
        return None

    def _attribute_value(self, record: EventRecord, attributes: dict[str, Any], name: str) -> str:
        if name == "dates":
            return record.dates
        value = attributes.get(name)
        if isinstance(value, int) and name.endswith("_count") and value <= 0:
            return ""
        return str(value or "")

    def verify(self, value: str, doc_ids: list[str]) -> list[EvidenceItem]:
        # Evidence evaluation: find the source line in the cited article that states the value.
        if self.coprocessor is None or not value:
            return []
        wanted = fold(value)
        evidence = []
        for doc_id in doc_ids[:5]:
            chunk = self.coprocessor.get_chunk(f"{doc_id}#0")
            if chunk is None:
                continue
            line = next(
                (
                    candidate.strip()
                    for candidate in chunk.raw_text.splitlines() + chunk.text.split(" | ")
                    if wanted and wanted in fold(candidate)
                ),
                "",
            )
            if line:
                evidence.append(
                    EvidenceItem(doc_id=doc_id, chunk_id=chunk.chunk_id, text=line, source="gsql")
                )
        return evidence

    def _finish(
        self,
        outcome: ToolOutcome,
        started: float,
        link: LinkResult | None = None,
        verify_answer: bool = True,
    ) -> ToolOutcome:
        if link is not None:
            outcome.observation["linking"] = {"method": link.method, "matches": link.summary()}
            outcome.agents.insert(0, ENTITY_LINKER)
        if outcome.answer is not None and verify_answer:
            outcome.evidence = self.verify(outcome.answer, outcome.citations)
            outcome.observation["source_check"] = [
                {"doc_id": item.doc_id, "line": item.text} for item in outcome.evidence
            ] or "value not stated verbatim in the cited article's opening section"
            outcome.agents.append(EVALUATOR)
        if outcome.citations and not outcome.evidence:
            outcome.evidence = [
                EvidenceItem(doc_id=doc_id, text="", source="gsql") for doc_id in outcome.citations
            ]
        outcome.latency_ms = (time.perf_counter() - started) * 1000
        return outcome

    def _unlinked(self, tool: str, started: float, problems: list[str]) -> ToolOutcome:
        observation: dict[str, Any] = {"error": "; ".join(problems)}
        if any(problem.startswith("sport") for problem in problems):
            observation["graph_sports"] = self.catalog.sports
        return self._finish(
            ToolOutcome(tool=tool, observation=observation, agents=[ENTITY_LINKER]), started
        )

    def _link_ambiguity(self, tool: str, started: float, link: LinkResult) -> ToolOutcome:
        names = [match.record.name for match in link.matches]
        observation: dict[str, Any] = {
            "status": "ambiguous" if names else "no_match",
            "detail": (
                "Several events fit; add a distinguishing qualifier or report the ambiguity."
                if names
                else "No event in the graph fits these constraints."
            ),
        }
        return self._finish(
            ToolOutcome(tool=tool, observation=observation, candidates=names), started, link
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def count_events(
        self,
        sport: str = "",
        year: int = 0,
        season: str = "",
        gender: str = "",
        comparison: str = "more_than",
        threshold: int = 0,
    ) -> ToolOutcome:
        started = time.perf_counter()
        tool = "count_events"
        resolved_sport, resolved_season, resolved_gender, problems = self._constraints(
            sport, season, gender
        )
        if problems:
            return self._unlinked(tool, started, problems)
        try:
            minimum, maximum = _bounds(comparison, threshold)
        except ValueError as exc:
            return self._finish(
                ToolOutcome(tool=tool, observation={"error": str(exc)}, agents=[AGGREGATOR]),
                started,
            )
        if maximum and maximum < minimum:
            # "fewer than 1" and similar bounds admit no event; 0 would mean "unbounded" below.
            result: dict[str, Any] = {"count": 0, "events": [], "gold_doc_ids": []}
        else:
            result = self.graph.run_aggregation(
                sport=resolved_sport or "",
                year=year,
                min_competitors=minimum,
                max_competitors=maximum,
            )
        names = [str(name) for name in result.get("events", [])]
        doc_ids = [str(doc_id) for doc_id in result.get("gold_doc_ids", [])]
        count = _as_int(result.get("count"))
        if resolved_season or resolved_gender:
            # The installed count query filters sport, year, and bounds; season and gender narrow
            # its returned rows here, so the count must be recomputed from those rows.
            wanted = {
                record.name
                for record in self.catalog.events(
                    sport=resolved_sport, year=year, season=resolved_season, gender=resolved_gender
                )
            }
            names = [name for name in names if name in wanted]
            count = len(names)
        observation = {
            "count": count,
            "filters": {
                "sport": resolved_sport,
                "year": year or None,
                "season": resolved_season,
                "gender": resolved_gender,
                "competitors_min": minimum,
                "competitors_max": maximum or None,
            },
            "matching_events": names[:_MAX_LISTED_EVENTS],
        }
        return self._finish(
            ToolOutcome(
                tool=tool,
                observation=observation,
                answer=str(count),
                citations=doc_ids,
                agents=[AGGREGATOR],
            ),
            started,
            verify_answer=False,
        )

    def rank_events(
        self,
        sport: str = "",
        year: int = 0,
        season: str = "",
        gender: str = "",
        order: str = "desc",
        attribute: str = "competitor_count",
    ) -> ToolOutcome:
        # Highest or lowest competitor count; ties are reported, never broken arbitrarily.
        started = time.perf_counter()
        tool = "rank_events"
        resolved_sport, resolved_season, resolved_gender, problems = self._constraints(
            sport, season, gender
        )
        if attribute != "competitor_count":
            problems.append("only competitor_count rankings are supported")
        if problems:
            return self._unlinked(tool, started, problems)
        descending = fold(order) not in {"asc", "ascending", "lowest", "fewest", "min"}
        pool = self.catalog.events(
            sport=resolved_sport, year=year, season=resolved_season, gender=resolved_gender
        )
        # Rank TigerGraph's stored counts for the whole linked pool. The installed superlative
        # query cannot do this: its ACCUM lists ignore ORDER BY order, and LIMIT 1 hides ties.
        # Missing counts are stored as 0 and never win a ranking.
        fetched = self._fetch_events(pool)
        doc_by_name = {record.name: record.event_id for record in pool}
        counted = [
            (record.name, _as_int(fetched.get(record.event_id, {}).get("competitor_count")))
            for record in pool
        ]
        ranked = sorted(
            [(name, count) for name, count in counted if count > 0],
            key=lambda item: (-item[1] if descending else item[1], item[0]),
        )
        direction = "desc" if descending else "asc"
        if not ranked:
            return self._finish(
                ToolOutcome(
                    tool=tool,
                    observation={"status": "no_match", "detail": "No ranked events fit."},
                    agents=[AGGREGATOR],
                ),
                started,
            )
        best = ranked[0][1]
        leaders = [name for name, count in ranked if count == best]
        observation: dict[str, Any] = {
            "order": direction,
            "top": [{"event": name, "competitor_count": count} for name, count in ranked[:5]],
            "events_ranked": len(ranked),
        }
        if len(leaders) > 1:
            observation["tie"] = {"competitor_count": best, "events": leaders}
            # Corroboration breaks a tie only when exactly one article states the count again in
            # its own prose; the infobox figure alone is a single source for every tied event.
            confirmations = {
                name: sentence
                for name in leaders
                if (sentence := self._prose_confirming_count(doc_by_name[name], best))
            }
            if len(confirmations) == 1:
                ((winner, sentence),) = confirmations.items()
                observation["tie"]["resolved_by"] = "count restated in the article's prose"
                observation["tie"]["evidence"] = {"event": winner, "sentence": sentence}
                leaders = [winner]
        outcome = ToolOutcome(
            tool=tool,
            observation=observation,
            answer=leaders[0] if len(leaders) == 1 else None,
            candidates=leaders,
            citations=[doc_by_name[name] for name in leaders],
            agents=[AGGREGATOR],
        )
        return self._finish(outcome, started, verify_answer=False)

    # ------------------------------------------------------------------
    # Traversal and lookup
    # ------------------------------------------------------------------

    def event_attribute(
        self,
        event: str,
        attribute: str = "gold_athlete",
        sport: str = "",
        year: int = 0,
        season: str = "",
        gender: str = "",
    ) -> ToolOutcome:
        started = time.perf_counter()
        tool = "event_attribute"
        if attribute not in EVENT_ATTRIBUTES:
            return self._finish(
                ToolOutcome(
                    tool=tool,
                    observation={
                        "error": f"unknown attribute {attribute!r}",
                        "attributes": list(EVENT_ATTRIBUTES),
                    },
                ),
                started,
            )
        resolved_sport, resolved_season, resolved_gender, problems = self._constraints(
            sport, season, gender
        )
        if problems:
            return self._unlinked(tool, started, problems)
        link = self.catalog.link_event(
            event, sport=resolved_sport, year=year, season=resolved_season, gender=resolved_gender
        )
        record = link.resolved
        if record is None:
            return self._link_ambiguity(tool, started, link)
        return self._read_attribute(tool, started, record, attribute, link)

    def _read_attribute(
        self,
        tool: str,
        started: float,
        record: EventRecord,
        attribute: str,
        link: LinkResult | None,
        extra: dict[str, Any] | None = None,
    ) -> ToolOutcome:
        attributes = self._fetch_events([record]).get(record.event_id, {})
        value = self._attribute_value(record, attributes, attribute)
        observation: dict[str, Any] = {
            "event": record.name,
            "attribute": attribute,
            "value": value or None,
            **(extra or {}),
        }
        return self._finish(
            ToolOutcome(
                tool=tool,
                observation=observation,
                answer=value or None,
                citations=[record.event_id],
                agents=[TRAVERSER],
            ),
            started,
            link,
        )

    def previous_edition(
        self,
        event: str,
        year: int,
        attribute: str = "gold_athlete",
        sport: str = "",
        season: str = "",
        gender: str = "",
    ) -> ToolOutcome:
        # The same event at the Games immediately before `year`: follow the PRECEDES edge from
        # the linked `year` edition, or link the earlier Games directly when `year` has none.
        started = time.perf_counter()
        tool = "previous_edition"
        resolved_sport, resolved_season, resolved_gender, problems = self._constraints(
            sport, season, gender
        )
        if problems:
            return self._unlinked(tool, started, problems)
        current = self.catalog.link_event(
            event, sport=resolved_sport, year=year, season=resolved_season, gender=resolved_gender
        )
        record = current.resolved
        if record is not None:
            result = self.graph.run_temporal(
                sport=record.sport, event_name_fragment=record.name, current_year=record.year
            )
            previous = [self.catalog.get(str(doc_id)) for doc_id in result.get("gold_doc_ids", [])]
            linked_previous = [item for item in previous if item is not None]
            if len(linked_previous) == 1:
                prior = linked_previous[0]
                games_year = self.catalog.previous_games_year(record.year, record.season)
                extra = {
                    "from_event": record.name,
                    "traversal": "PRECEDES",
                    "previous_games_year": games_year,
                }
                if games_year and prior.year != games_year:
                    extra["warning"] = (
                        f"The event was not held at the {games_year} Games; "
                        f"its previous edition is {prior.year}."
                    )
                return self._read_attribute(tool, started, prior, attribute, current, extra)
        season_for_games = resolved_season or (record.season if record else None)
        if season_for_games is None:
            seasons = {item.season for item in self.catalog.events(year=year) if item.season}
            season_for_games = seasons.pop() if len(seasons) == 1 else None
        games_year = (
            self.catalog.previous_games_year(year, season_for_games) if season_for_games else None
        )
        if games_year is None:
            return self._link_ambiguity(tool, started, current)
        prior_link = self.catalog.link_event(
            event,
            sport=resolved_sport,
            year=games_year,
            season=season_for_games,
            gender=resolved_gender,
        )
        prior_record = prior_link.resolved
        if prior_record is None:
            return self._link_ambiguity(tool, started, prior_link)
        return self._read_attribute(
            tool,
            started,
            prior_record,
            attribute,
            prior_link,
            {"traversal": "previous_games", "previous_games_year": games_year},
        )

    def event_at_venue_date(
        self, venue: str, date: str, year: int = 0, attribute: str = "gold_athlete"
    ) -> ToolOutcome:
        # Multi-hop: Venue <-HELD_AT(date)- Event, then read the requested attribute.
        started = time.perf_counter()
        tool = "event_at_venue_date"
        if attribute not in EVENT_ATTRIBUTES:
            attribute = "gold_athlete"
        link = self.catalog.link_venue_date(venue, date, year or None)
        record = link.resolved
        if record is not None:
            return self._read_attribute(tool, started, record, attribute, link)
        if not link.matches:
            return self._link_ambiguity(tool, started, link)
        # Several events share the venue and date: return each one's value so the caller can
        # disambiguate with other evidence or report all of them.
        records = [match.record for match in link.matches]
        fetched = self._fetch_events(records)
        options = [
            {
                "event": item.name,
                attribute: self._attribute_value(item, fetched.get(item.event_id, {}), attribute),
            }
            for item in records
        ]
        observation: dict[str, Any] = {
            "status": "ambiguous",
            "detail": "Several events were held at this venue on this date.",
            "events": options,
        }
        return self._finish(
            ToolOutcome(
                tool=tool,
                observation=observation,
                candidates=[str(option[attribute]) for option in options],
                citations=[item.event_id for item in records],
                agents=[TRAVERSER],
            ),
            started,
            link,
        )

    def find_events(
        self,
        event: str = "",
        sport: str = "",
        year: int = 0,
        season: str = "",
        gender: str = "",
    ) -> ToolOutcome:
        # Exploration for the planner: which canonical events fit these constraints?
        started = time.perf_counter()
        tool = "find_events"
        resolved_sport, resolved_season, resolved_gender, problems = self._constraints(
            sport, season, gender
        )
        if problems:
            return self._unlinked(tool, started, problems)
        if event:
            link = self.catalog.link_event(
                event,
                sport=resolved_sport,
                year=year,
                season=resolved_season,
                gender=resolved_gender,
            )
            names = [match.record.name for match in link.matches]
            outcome = ToolOutcome(tool=tool, observation={"events": names}, candidates=names)
            return self._finish(outcome, started, link)
        records = self.catalog.events(
            sport=resolved_sport, year=year, season=resolved_season, gender=resolved_gender
        )
        names = [record.name for record in records]
        outcome = ToolOutcome(
            tool=tool,
            observation={"total": len(names), "events": names[:_MAX_LISTED_EVENTS]},
            candidates=names,
            agents=[ENTITY_LINKER],
        )
        return self._finish(outcome, started)
