from types import SimpleNamespace

import pytest

import run_ts_overfit_diagnostic as diagnostic


def _series(airport: str, count: int):
    return [
        SimpleNamespace(airport=airport, dataset_id=f"{airport}:flight-{index}")
        for index in range(count)
    ]


def test_balanced_subset_is_deterministic_and_airport_balanced():
    population = _series("KAAA", 10) + _series("KBBB", 10)

    first = diagnostic.select_balanced_subset(
        population, ("KAAA", "KBBB"), samples_per_airport=4, seed=17
    )
    second = diagnostic.select_balanced_subset(
        list(reversed(population)), ("KAAA", "KBBB"), samples_per_airport=4, seed=17
    )

    assert [item.dataset_id for item in first] == [item.dataset_id for item in second]
    assert sum(item.airport == "KAAA" for item in first) == 4
    assert sum(item.airport == "KBBB" for item in first) == 4


def test_balanced_subset_rejects_an_undersized_airport():
    with pytest.raises(ValueError, match="KAAA has only 2"):
        diagnostic.select_balanced_subset(
            _series("KAAA", 2), ("KAAA",), samples_per_airport=3, seed=17
        )
