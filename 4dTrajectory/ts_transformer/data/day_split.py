"""The two-tier line's split BY OPERATING DAY (prior design §3.3): which days are test, validation,
internal selection and train.

A flight belongs to the operating day of its LANDING (`landing_day`): the landing time is the one
time both harvest rosters carry (the tracks roster, where the landing statistics come from, has no
entry time), and a flight must fall on the same day in both — its track in the arrivals roster, its
landing in the statistics. Entry and landing fall on different operating days for 20 of the 72,247
eligible arrivals (2026-09-24).

The days are dealt by COUNT, not by a threshold on each day's hash: sorted by sha256(seed:day), the
first round(15 % of N) are test, the next round(15 % of N) validation, then round(1/7 of the rest)
internal selection, the remainder train. With 90 days a per-day threshold misses its share by
±3.4 days (one binomial standard deviation at 15 %) — a quarter of the test set; counting hits it.
The price: a day's split depends on the whole day list — seven more days would move a development day
into test, sixteen a test day out of it. So the deal is made once and committed (`PINNED_DAY_SPLIT`),
and a harvest whose days differ from it is refused: extending the split is a decision (one that keeps
these test days sealed), never a silent re-deal.

Test days are sealed: `DaySplit.development_split` refuses them by name (`SealedDay`), so a reader on
this line cannot open a test day's flight, count its landing or put it in a scene.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

#: An OPERATING day starts at the overnight traffic minimum, not at UTC midnight: at these US
#: airports 22Z–03Z carries 27–33 % of arrivals, so a UTC-midnight cut splits the evening session
#: and puts one session's flights in two day folds (R1 review, 2026-09-13; the quietest hours are
#: 07–11Z — 03:00–07:00 EDT, 00:00–04:00 PDT).
OPERATIONAL_DAY_SHIFT = timedelta(hours=9)

DAY_SPLIT_SCHEMA = "ts-day-split-v1"
#: Every split, in the order the sorted days are dealt to them.
DAY_SPLITS = ("test", "val", "select", "train")
#: The splits the two-tier line may open (everything but test).
DEVELOPMENT_SPLITS = ("train", "select", "val")
#: Prior design §3.3.
DAY_SPLIT_SEED = 1337
TEST_SHARE = 0.15
VAL_SHARE = 0.15
#: The internal selection set's share of the days left after test and validation.
SELECT_SHARE = 1 / 7
DAY_SPLIT_METHOD = ("operating days sorted by sha256(seed:day); the first round(0.15 N) test, the next round(0.15 N) "
                    "val, round(1/7 of the rest) select, the remainder train; rounding half up")
#: The deal the two-tier line uses: the 90 operating days of the harvest of 2026-09-23 (04-30 … 09-21), dealt with
#: `DAY_SPLIT_SEED` and committed.
PINNED_DAY_SPLIT = Path(__file__).with_name("day_split_20260924.json")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def operational_day(time: datetime) -> str:
    """The operating day a UTC time belongs to (see `OPERATIONAL_DAY_SHIFT`), ISO date."""
    return (time - OPERATIONAL_DAY_SHIFT).date().isoformat()


def operating_day_span_s(day: str) -> tuple[float, float]:
    """The UTC epoch seconds an operating day covers, ``[start, end)``."""
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + OPERATIONAL_DAY_SHIFT
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def landing_day(landing_time_utc: str) -> str:
    """A flight's operating day: the one its landing falls on."""
    return operational_day(parse_utc(landing_time_utc))


class SealedDay(ValueError):
    """A reader on the two-tier line reached a test day."""


def _count(share: float, n: int) -> int:
    return math.floor(share * n + 0.5)


@dataclass(frozen=True)
class DaySplit:
    seed: int
    #: split → its days, sorted; every split of `DAY_SPLITS` present, no day in two (read-only once made)
    days: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "days", MappingProxyType({split: tuple(days) for split, days in self.days.items()}))
        if tuple(self.days) != DAY_SPLITS:
            raise ValueError(f"a day split has the splits {DAY_SPLITS}, not {tuple(self.days)}")
        listed = [day for split in DAY_SPLITS for day in self.days[split]]
        if len(set(listed)) != len(listed):
            raise ValueError("a day is listed in two splits")
        object.__setattr__(self, "_split_of", {day: split for split in DAY_SPLITS for day in self.days[split]})

    @property
    def listed(self) -> frozenset[str]:
        """Every day of the split."""
        return frozenset(self._split_of)

    def split_of(self, day: str) -> str:
        """The split of a listed day. A day outside the list belongs to no split: the list was dealt
        as a whole, so a day it never saw is refused, never guessed."""
        if day not in self._split_of:
            raise KeyError(f"operating day {day} is not in this day split ({len(self._split_of)} days)")
        return self._split_of[day]

    def development_split(self, day: str) -> str:
        """The split of a day the two-tier line may open; a test day is refused (`SealedDay`)."""
        split = self.split_of(day)
        if split == "test":
            raise SealedDay(f"operating day {day} is a sealed test day")
        return split

    def to_dict(self) -> dict[str, Any]:
        return {"schema": DAY_SPLIT_SCHEMA, "method": DAY_SPLIT_METHOD, "seed": self.seed,
                "operational_day_shift_h": OPERATIONAL_DAY_SHIFT.total_seconds() / 3600,
                "counts": {split: len(days) for split, days in self.days.items()},
                "days": {split: list(days) for split, days in self.days.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DaySplit":
        if data["schema"] != DAY_SPLIT_SCHEMA:
            raise ValueError(f"not a {DAY_SPLIT_SCHEMA} record: {data['schema']}")
        if data["method"] != DAY_SPLIT_METHOD or data["operational_day_shift_h"] != OPERATIONAL_DAY_SHIFT.total_seconds() / 3600:
            raise ValueError("the recorded day split was dealt by another method than this code's")
        if set(data["days"]) != set(DAY_SPLITS) or data["counts"] != {name: len(data["days"][name]) for name in DAY_SPLITS}:
            raise ValueError(f"a day split record holds the splits {DAY_SPLITS} and their day counts")
        split = cls(int(data["seed"]), {name: tuple(data["days"][name]) for name in DAY_SPLITS})
        if split != split_days([day for days in split.days.values() for day in days], split.seed):
            raise ValueError("the recorded day split is not what its day list and seed deal")
        return split


def split_days(days: Iterable[str], seed: int) -> DaySplit:
    """Deal the operating days (see the module docstring)."""
    listed = sorted(set(days))
    if not listed:
        raise ValueError("no operating days to split")
    ordered = sorted(listed, key=lambda day: hashlib.sha256(f"{seed}:{day}".encode()).digest())
    n_test, n_val = _count(TEST_SHARE, len(ordered)), _count(VAL_SHARE, len(ordered))
    n_select = _count(SELECT_SHARE, len(ordered) - n_test - n_val)
    bounds = {"test": (0, n_test), "val": (n_test, n_test + n_val),
              "select": (n_test + n_val, n_test + n_val + n_select), "train": (n_test + n_val + n_select, len(ordered))}
    return DaySplit(seed, {split: tuple(sorted(ordered[lo:hi])) for split, (lo, hi) in bounds.items()})


def pinned_day_split() -> DaySplit:
    """The committed deal (`PINNED_DAY_SPLIT`)."""
    return DaySplit.from_dict(json.loads(PINNED_DAY_SPLIT.read_text(encoding="utf-8")))


def harvest_operating_days(tracks_manifests: Sequence[Path]) -> list[str]:
    """Every operating day with an assigned landing in these tracks rosters — roster metadata only,
    no track opened. The arrivals are a subset of the assigned landings, so every arrival's day is
    listed, and so is every landing the statistics may count."""
    days: set[str] = set()
    for path in tracks_manifests:
        for row in json.loads(Path(path).read_text(encoding="utf-8"))["records"]:
            if row["outcome"] == "assigned":
                days.add(landing_day(row["landing_time_utc"]))
    return sorted(days)
