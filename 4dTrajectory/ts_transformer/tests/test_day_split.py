"""The split by operating day (`data.day_split`, prior design §3.3)."""

import json
from datetime import date, timedelta

import pytest

from ts_transformer.data.day_split import (
    DAY_SPLIT_SEED,
    DAY_SPLITS,
    DaySplit,
    SealedDay,
    harvest_operating_days,
    landing_day,
    operating_day_span_s,
    operational_day,
    pinned_day_split,
    split_days,
)
from ts_transformer.tests.support import fixture_days


def _days(n: int) -> list[str]:
    return [(date(2026, 5, 1) + timedelta(days=i)).isoformat() for i in range(n)]


def test_ninety_days_are_dealt_by_count():
    split = split_days(_days(90), DAY_SPLIT_SEED)
    # round half up: 13.5 → 14 test, 14 val; of the 62 left, 62/7 = 8.9 → 9 select; 53 train
    assert {name: len(days) for name, days in split.days.items()} == {"test": 14, "val": 14, "select": 9, "train": 53}
    listed = [day for days in split.days.values() for day in days]
    assert sorted(listed) == _days(90) and all(list(days) == sorted(days) for days in split.days.values())


def test_the_deal_depends_on_the_day_list_and_seed_only():
    days = _days(40)
    assert split_days(list(reversed(days)) + days[:5], 7) == split_days(days, 7)
    assert split_days(days, 7) != split_days(days, 8)


def test_an_operating_day_turns_at_nine_utc():
    assert landing_day("2026-06-02T08:59:59Z") == "2026-06-01"
    assert landing_day("2026-06-02T09:00:00Z") == "2026-06-02"
    assert landing_day("2026-06-01T23:30:00.250Z") == "2026-06-01"


def test_an_operating_day_spans_nine_utc_to_nine_utc():
    from datetime import datetime, timezone

    start, end = operating_day_span_s("2026-06-01")
    assert end - start == 86_400.0
    for t, day in ((start, "2026-06-01"), (end - 0.001, "2026-06-01"), (end, "2026-06-02")):
        assert operational_day(datetime.fromtimestamp(t, tz=timezone.utc)) == day


def test_a_test_day_is_sealed_and_an_unlisted_day_belongs_to_no_split():
    split = split_days(_days(20), DAY_SPLIT_SEED)
    test_day, train_day = split.days["test"][0], split.days["train"][0]
    assert split.split_of(test_day) == "test"
    with pytest.raises(SealedDay, match=f"{test_day} is a sealed test day"):
        split.development_split(test_day)
    assert split.development_split(train_day) == "train"
    with pytest.raises(KeyError, match="not in this day split"):
        split.split_of("2027-01-01")


def test_the_record_round_trips_and_a_tampered_one_is_refused():
    split = split_days(_days(20), DAY_SPLIT_SEED)
    record = json.loads(json.dumps(split.to_dict()))
    assert DaySplit.from_dict(record) == split
    moved = json.loads(json.dumps(record))
    moved["days"]["train"].append(moved["days"]["test"].pop())
    moved["counts"] = {name: len(days) for name, days in moved["days"].items()}
    with pytest.raises(ValueError, match="not what its day list and seed deal"):
        DaySplit.from_dict(moved)
    with pytest.raises(ValueError, match="another method"):
        DaySplit.from_dict({**record, "operational_day_shift_h": 0.0})
    with pytest.raises(ValueError, match="holds the splits"):
        DaySplit.from_dict({**record, "days": {**record["days"], "extra": []}})
    with pytest.raises(ValueError, match="holds the splits"):
        DaySplit.from_dict({**record, "counts": {**record["counts"], "test": 0}})
    with pytest.raises(ValueError, match="has the splits"):
        DaySplit(DAY_SPLIT_SEED, {name: split.days[name] for name in reversed(DAY_SPLITS)})
    with pytest.raises(ValueError, match="listed in two splits"):
        DaySplit(DAY_SPLIT_SEED, {**split.days, "val": split.days["val"] + split.days["test"][:1]})


def test_the_harvest_days_are_the_assigned_landings_of_every_roster(tmp_path):
    rosters = []
    for name, rows in {
        "A": [{"outcome": "assigned", "landing_time_utc": "2026-06-02T08:00:00Z"},
              {"outcome": "not_landing", "landing_time_utc": None}],
        "B": [{"outcome": "assigned", "landing_time_utc": "2026-06-03T12:00:00Z"},
              {"outcome": "ambiguous", "landing_time_utc": "2026-06-09T12:00:00Z"}],
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"records": rows}), encoding="utf-8")
        rosters.append(path)
    assert harvest_operating_days(rosters) == ["2026-06-01", "2026-06-03"]


def test_the_pinned_deal_is_ninety_days_dealt_with_the_seed_and_cannot_be_edited():
    pinned = pinned_day_split()          # from_dict re-deals the recorded list and compares
    assert pinned.seed == DAY_SPLIT_SEED and len(pinned.listed) == 90
    assert {name: len(days) for name, days in pinned.days.items()} == {"test": 14, "val": 14, "select": 9, "train": 53}
    with pytest.raises(TypeError):
        pinned.days["test"] = ()


def _arrivals(tmp_path, airport: str, landings: dict[str, str]):
    path = tmp_path / f"{airport}.json"
    path.write_text(json.dumps({"records": [{"flight_key": k, "landing_time_utc": t} for k, t in landings.items()]}),
                    encoding="utf-8")
    return path


def test_the_signals_runner_deals_flights_by_landing_day_and_never_builds_a_test_day(tmp_path):
    from ts_transformer.experiments.instruction_signals import build_jobs, keys_by_day
    from ts_transformer.tests.support import landing_on

    days = fixture_days()
    manifests = {"KAAA": _arrivals(tmp_path, "KAAA", {"a1": landing_on("train"), "a2": landing_on("test"),
                                                      "a3": landing_on("select"), "late": "2026-06-30T12:00:00Z"}),
                 "KBBB": _arrivals(tmp_path, "KBBB", {"b1": landing_on("val"), "b2": landing_on("test")})}
    # the provenance lists airports in its own (sorted) order and only the eligible flights
    provenance = {"manifests": [{"airport": "KBBB", "source_records": [{"flight_key": "b1"}, {"flight_key": "b2"}]},
                                {"airport": "KAAA", "source_records": [{"flight_key": k} for k in ("a1", "a2", "a3")]}]}
    keys = keys_by_day(provenance, manifests, days)
    assert keys == {"test": ["KBBB:b2", "KAAA:a2"], "val": ["KBBB:b1"], "select": ["KAAA:a3"], "train": ["KAAA:a1"]}
    jobs = build_jobs(keys, manifests, limit=0)
    built = [key for _, _, _, chunk in jobs for key in chunk]
    assert sorted(built) == ["KAAA:a1", "KAAA:a3", "KBBB:b1"] and {split for split, *_ in jobs} == {"train", "select", "val"}
    assert all(key.startswith(f"{airport}:") and manifest == str(manifests[airport]) for _, airport, manifest, chunk in jobs
               for key in chunk)
    # an eligible flight landing on a day the deal never saw is refused, never guessed into a split
    provenance["manifests"][1]["source_records"].append({"flight_key": "late"})
    with pytest.raises(KeyError, match="2026-06-30 is not in this day split"):
        keys_by_day(provenance, manifests, days)
