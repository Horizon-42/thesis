"""Cut verbatim from tests/test_autopilot.py on 2026-09-24 with the runner it tests (archive README). It does not run."""

from __future__ import annotations

import numpy as np


def test_the_heading_comparison_tags_each_row_by_its_dataset_id_not_its_position(monkeypatch):
    """Review of §10.1's runner: `fly_variant` returns the rows grouped by airport, while the sample interleaves the
    airports — the rows must find their flight, stratum and H1 intercept turn by dataset id."""
    from types import SimpleNamespace

    from ts_transformer.experiments.heading_reading_compare import in_group, tag_rows

    def h1(turns):
        return SimpleNamespace(checks={"turns": turns})

    ids = ["KRDU:a", "KSJC:b", "KRDU:c"]                       # the sample: airports interleaved
    turn = {"turn_deg": -150.0, "kind": "intercept"}
    readings = {0: h1([]), 1: h1([turn])}                      # flight 2 refused by H1
    rows = [{"dataset_id": "KRDU:a"}, {"dataset_id": "KRDU:c"}, {"dataset_id": "KSJC:b"}]   # grouped by airport
    import ts_transformer.experiments.heading_reading_compare as compare
    monkeypatch.setattr(compare, "flight_record",
                        lambda reading: {"stratum": "vectored" if reading.checks["turns"] else "straight-in"})
    tag_rows(rows, ids, readings)
    assert [(r["sample_index"], r["stratum"], r["h1_intercept_turn_deg"]) for r in rows] == [
        (0, "straight-in", 0.0), (2, "refused by H1", None), (1, "vectored", 150.0)]
    assert [in_group(r, "onto final") for r in rows] == [False, False, True]
    assert [in_group(r, "not onto final") for r in rows] == [True, True, False]
