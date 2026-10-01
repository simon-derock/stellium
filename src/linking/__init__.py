# Graph-backed entity linking for Olympic events, venues, and dates.
# Everything the linker can return is loaded from TigerGraph (Event vertices plus HELD_AT edges),
# so no sport, venue, or event list lives in code. Matching is order-insensitive token overlap,
# which is what lets "500 metres speed skating" find "Speed skating – Women's 500 metres".
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "–": "-", "—": "-"})
_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
# Words that frame a question rather than name an event; they never distinguish two events.
_FRAMING_WORDS = frozenset(
    {"the", "at", "in", "of", "event", "olympic", "olympics", "game", "games", "edition"}
)
# Canonical title grammar, e.g. "Judo at the 2016 Summer Olympics – Women's 57 kg".
_TITLE = re.compile(r"^(.+?)\s+at the\s+(\d{4})\s+(\w+)\s+Olympics\s*[–—-]\s*(.+)$", re.I)
_MIN_LINK_SCORE = 0.5
_UNIQUE_MARGIN = 0.15


def fold(text: str) -> str:
    # Lowercase ASCII words: accents, dashes, and possessive apostrophes stop mattering.
    ascii_text = (
        unicodedata.normalize("NFKD", text.translate(_TYPOGRAPHY))
        .encode("ascii", "ignore")
        .decode()
    )
    # "+80 kg" and "80 kg" are different weight classes, so the plus sign survives as a word.
    ascii_text = ascii_text.casefold().replace("'", "").replace("+", " plus ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_text).split())


def compact(text: str) -> str:
    # Spacing-insensitive form, for strings the corpus sometimes glues together
    # ("TechnologyUniversity") or writes with different dash spacing.
    return fold(text).replace(" ", "")


def _stem(token: str) -> str:
    # Plural-insensitive enough for "metres"/"metre" and "mens"/"men" without a stemmer library.
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokens(text: str) -> frozenset[str]:
    return frozenset(_stem(token) for token in fold(text).split())


def _date_key(text: str) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    # Day numbers, month names, and years, so "August 12, 2008" equals "12 August".
    words = fold(text).split()
    days = frozenset(word for word in words if word.isdigit() and len(word) <= 2)
    years = frozenset(word for word in words if word.isdigit() and len(word) == 4)
    months = frozenset(word for word in words if word in _MONTHS)
    return days, months, years


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    name: str
    year: int
    season: str
    sport: str
    gender: str
    venue: str
    dates: str

    @property
    def label(self) -> str:
        # The part of the canonical title after the Games, e.g. "Men's épée".
        return re.split(r"\s[–—-]\s", self.name, maxsplit=1)[-1]


@dataclass(frozen=True)
class EventMatch:
    record: EventRecord
    score: float


@dataclass(frozen=True)
class LinkResult:
    matches: tuple[EventMatch, ...]
    method: str

    @property
    def resolved(self) -> EventRecord | None:
        # One clear winner: a single match, or a top match well ahead of the runner-up.
        if not self.matches:
            return None
        if len(self.matches) == 1:
            return self.matches[0].record
        top, runner_up = self.matches[0].score, self.matches[1].score
        if top == 1.0 > runner_up or top - runner_up >= _UNIQUE_MARGIN:
            return self.matches[0].record
        return None

    @property
    def ambiguous(self) -> bool:
        return bool(self.matches) and self.resolved is None

    def summary(self, limit: int = 5) -> list[dict[str, Any]]:
        return [
            {"event": match.record.name, "event_id": match.record.event_id, "score": match.score}
            for match in self.matches[:limit]
        ]


class EventCatalog:
    def __init__(self, records: Iterable[EventRecord]) -> None:
        self.records = tuple(records)
        self.sports = sorted({record.sport for record in self.records if record.sport})
        self.genders = sorted({record.gender for record in self.records if record.gender})
        self._by_id = {record.event_id: record for record in self.records}
        self._by_title = {compact(record.name): record for record in self.records}
        self._years = {record.year for record in self.records if record.year}

    @classmethod
    def from_graph(cls, conn: Any) -> EventCatalog:
        # Two bulk reads at startup; every later lookup is in memory.
        dates = {
            str(edge["from_id"]): str(edge.get("attributes", {}).get("start_date", "") or "")
            for edge in conn.getEdgesByType("HELD_AT")
        }
        records = []
        for row in conn.getVertices("Event"):
            attributes = row.get("attributes", {})
            event_id = str(row["v_id"])
            records.append(
                EventRecord(
                    event_id=event_id,
                    name=str(attributes.get("name", "")),
                    year=int(attributes.get("year", 0) or 0),
                    season=str(attributes.get("season", "") or ""),
                    sport=str(attributes.get("sport", "") or ""),
                    gender=str(attributes.get("gender", "") or ""),
                    venue=str(attributes.get("venue", "") or ""),
                    dates=dates.get(event_id, ""),
                )
            )
        return cls(records)

    def get(self, event_id: str) -> EventRecord | None:
        return self._by_id.get(event_id)

    def resolve_sport(self, phrase: str) -> str | None:
        # Exact token match first ("alpine skiing" -> "Alpine skiing"), then the single best
        # overlapping sport name if one clearly dominates.
        wanted = tokens(phrase)
        if not wanted:
            return None
        for sport in self.sports:
            if tokens(sport) == wanted:
                return sport
        scored = sorted(
            (
                (len(wanted & tokens(sport)) / len(wanted | tokens(sport)), sport)
                for sport in self.sports
            ),
            reverse=True,
        )
        if scored and scored[0][0] >= _MIN_LINK_SCORE:
            if len(scored) == 1 or scored[0][0] > scored[1][0]:
                return scored[0][1]
        return None

    def _sport_named_in(self, phrase: str) -> str | None:
        # A phrase like "women's 200 metre freestyle swimming" names its sport; prefer the
        # longest sport name whose words all appear, so "speed skating" beats "skating".
        words = tokens(phrase)
        named = [sport for sport in self.sports if tokens(sport) and tokens(sport) <= words]
        return max(named, key=lambda sport: len(tokens(sport))) if named else None

    def resolve_gender(self, phrase: str) -> str | None:
        wanted = fold(phrase).replace(" ", "")
        for gender in self.genders:
            if fold(gender) == wanted or fold(gender) + "s" == wanted:
                return gender
        return None

    def resolve_season(self, phrase: str) -> str | None:
        wanted = fold(phrase)
        seasons = {record.season for record in self.records if record.season}
        return next((season for season in seasons if fold(season) == wanted), None)

    def previous_games_year(self, year: int, season: str) -> int | None:
        # The Games immediately before `year` for the same season, as recorded in the graph.
        earlier = [
            record.year
            for record in self.records
            if record.season == season and 0 < record.year < year
        ]
        return max(earlier) if earlier else None

    def events(
        self,
        sport: str | None = None,
        year: int | None = None,
        season: str | None = None,
        gender: str | None = None,
    ) -> list[EventRecord]:
        return [
            record
            for record in self.records
            if (not sport or record.sport == sport)
            and (not year or record.year == year)
            and (not season or record.season == season)
            and (not gender or record.gender == gender)
        ]

    def link_event(
        self,
        phrase: str,
        sport: str | None = None,
        year: int | None = None,
        season: str | None = None,
        gender: str | None = None,
    ) -> LinkResult:
        # Rank events in the constrained pool by F1 overlap between the phrase and each title's
        # event label. Numbers must agree exactly: "200 metre" never links to "4 × 200 metre".
        exact = self._by_title.get(compact(phrase))
        if exact is not None and (not year or exact.year == year):
            return LinkResult(matches=(EventMatch(record=exact, score=1.0),), method="exact_title")
        titled = _TITLE.match(phrase.strip())
        if titled:
            # A phrase in title form carries its own sport, Games, and season constraints.
            sport = sport or self.resolve_sport(titled.group(1))
            year = year or int(titled.group(2))
            season = season or self.resolve_season(titled.group(3))
            phrase = titled.group(4)
        if not year:
            # A single Games year inside a free phrase constrains the year, not the event label.
            named_years = {int(t) for t in tokens(phrase) if len(t) == 4 and t.isdigit()}
            if len(named_years & self._years) == 1:
                year = (named_years & self._years).pop()
        if not sport:
            sport = self._sport_named_in(phrase)
        pool = self.events(sport=sport, year=year, season=season, gender=gender)
        noise = set(_FRAMING_WORDS) | set(tokens(sport or "")) | set(tokens(season or ""))
        if year:
            noise.add(str(year))
        wanted = tokens(phrase) - noise
        if not wanted:
            return LinkResult(matches=(), method="empty_phrase")
        wanted_numbers = {token for token in wanted if token.isdigit()}
        matches = []
        for record in pool:
            label_tokens = tokens(record.label)
            overlap = len(wanted & label_tokens)
            if not overlap:
                continue
            precision, recall = overlap / len(wanted), overlap / len(label_tokens)
            score = 2 * precision * recall / (precision + recall)
            if {token for token in label_tokens if token.isdigit()} != wanted_numbers:
                score *= 0.5
            if score >= _MIN_LINK_SCORE:
                matches.append(EventMatch(record=record, score=round(score, 3)))
        matches.sort(key=lambda match: (-match.score, match.record.name))
        return LinkResult(matches=tuple(matches[:10]), method="label_token_f1")

    def link_venue_date(self, venue: str, date: str, year: int | None = None) -> LinkResult:
        # Exact (spacing-insensitive) venue and date first; questions usually copy the infobox.
        # Then the same venue with an equivalent date written in another order or format.
        venue_key, date_key = compact(venue), compact(date)
        pool = [record for record in self.records if not year or record.year == year]
        exact = [
            EventMatch(record=record, score=1.0)
            for record in pool
            if venue_key
            and compact(record.venue) == venue_key
            and (not date_key or compact(record.dates) == date_key)
        ]
        if exact:
            return LinkResult(matches=tuple(exact), method="exact_venue_date")
        days, months, years = _date_key(date)
        relaxed = []
        for record in pool:
            record_venue = compact(record.venue)
            if not record_venue or not venue_key:
                continue
            if venue_key not in record_venue and record_venue not in venue_key:
                continue
            record_days, record_months, record_years = _date_key(record.dates)
            if days != record_days or months != record_months:
                continue
            if years and not years <= (record_years | {str(record.year)}):
                continue
            relaxed.append(EventMatch(record=record, score=0.8))
        return LinkResult(matches=tuple(relaxed), method="venue_and_equivalent_date")
