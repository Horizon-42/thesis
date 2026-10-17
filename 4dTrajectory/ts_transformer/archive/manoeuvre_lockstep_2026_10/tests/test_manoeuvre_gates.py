"""The gates two-tier v3 still reads (`manoeuvre/gates.py`): the grid gate's verdict reads the
lockstep artefacts, refuses a single seed, and applies the pre-registered rule exactly.

Gates T / X / P / E / S and their tests were ARCHIVED 2026-09-20 with the intent-code layer
(`archive/manoeuvre_codes_2026_09/gates_manoeuvre.py`, `.../tests/test_manoeuvre_gates.py` —
the original of this file, unmodified). The relative gate's own tests live in
`tests/test_executor_relative_gate.py`.
"""

from __future__ import annotations

import pytest

from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_ESTABLISHED, STRATUM_STRAIGHT_IN, STRATUM_VECTORED
from ts_transformer.manoeuvre import gates as g


def _none_payload(*, established_all: float, established_vectored: float, flyable: float = 0.99, vectored_ade: float = 2000.0, n: int = 1400) -> dict:
    """A protocol-none lockstep payload with the cells the grid gate reads."""
    def cell(established, ade):
        return {"n": 10, "ade_mean_m": ade, "ade_p50_m": ade * 0.8, "fde_p50_m": ade, "fully_flyable_share": flyable,
                "established_share": established}
    return {"executor_name": "twin", "protocol": "none", "segment_s": 20.0, "executed_s": 20.0, "flights": n, "strata": {
        STRATUM_ALL: cell(established_all, vectored_ade * 0.6), STRATUM_VECTORED: cell(established_vectored, vectored_ade),
        STRATUM_STRAIGHT_IN: cell(1.0, 300.0), STRATUM_ESTABLISHED: {"n": 0},
    }}


def _other_protocol_payload() -> dict:
    """A payload flown under anything but ``none`` — what the grid gate must refuse."""
    return {**_none_payload(established_all=0.7, established_vectored=0.3), "protocol": "A"}


def test_the_grid_gate_reads_the_seed_line_off_the_grid_and_breaks_ties_shorter():
    """Two-tier v3 A1 (§3.3, D9): established over all flights and over the vectored group,
    both seeds; fully flyable ≥ 0.95 on both seeds; the seed line is the grid's own p75 of the
    seed differences; ties within it go to the shorter segment, then the shorter lookback."""
    def grid(values):
        return {cell: {1337: _none_payload(established_all=a1, established_vectored=v1),
                       2024: _none_payload(established_all=a2, established_vectored=v2)}
                for cell, (a1, a2, v1, v2) in values.items()}
    cells = grid({
        (30.0, 20.0): (0.70, 0.71, 0.30, 0.31),
        (60.0, 20.0): (0.70, 0.72, 0.30, 0.32),      # ties the best within the line: shorter L wins the tie
        (30.0, 60.0): (0.75, 0.77, 0.40, 0.42),      # the best on both seeds, both primaries
        (60.0, 60.0): (0.74, 0.76, 0.39, 0.41),      # within 0.01 of the best: a tie; longer L loses
        (30.0, 90.0): (0.76, 0.75, 0.41, 0.40),      # within the line of the best on both seeds; longer Δ loses
    })
    verdict = g.gate_grid(cells)
    assert verdict["seeds"] == [1337, 2024] and verdict["seed_line"]["established_all"]["p75"] == pytest.approx(0.02)
    assert verdict["eligible"] == ["L30_D20", "L60_D20", "L30_D60", "L60_D60", "L30_D90"] and verdict["ineligible"] == []
    assert verdict["leaders"]["established_all"] == ["L30_D60", "L60_D60", "L30_D90"]
    assert verdict["winners"] == ["L30_D60", "L60_D60", "L30_D90"]
    assert verdict["selected"] == {"cell": "L30_D60", "lookback_s": 30.0, "segment_s": 60.0} and verdict["decisive"]
    assert verdict["best_on_both_seeds"]["established_all"] is None      # seed 1337's best is L30_D90, seed 2024's L30_D60
    # the flyable constraint removes a cell on ONE seed's failure
    cells[(30.0, 60.0)][2024] = _none_payload(established_all=0.77, established_vectored=0.42, flyable=0.90)
    verdict = g.gate_grid(cells)
    assert verdict["ineligible"] == ["L30_D60"] and verdict["selected"]["cell"] == "L60_D60"
    # the primaries disagree: all-flights established decides, and the note says so
    cells = grid({(30.0, 20.0): (0.80, 0.80, 0.20, 0.20), (30.0, 60.0): (0.70, 0.70, 0.40, 0.40)})
    verdict = g.gate_grid(cells)
    assert verdict["selected"]["cell"] == "L30_D20" and "decides" in verdict["note"]
    # every cell within the seed line: not decisive, the shortest cell is taken
    cells = grid({(30.0, 20.0): (0.70, 0.72, 0.30, 0.32), (60.0, 60.0): (0.71, 0.71, 0.31, 0.31), (120.0, 120.0): (0.71, 0.73, 0.30, 0.32)})
    verdict = g.gate_grid(cells)
    assert not verdict["decisive"] and verdict["selected"]["cell"] == "L30_D20" and "not the deciding" in verdict["note"]
    with pytest.raises(ValueError, match="two seeds"):
        g.gate_grid({(30.0, 20.0): {1337: _none_payload(established_all=0.7, established_vectored=0.3)}})
    with pytest.raises(ValueError, match="protocol"):
        g.gate_grid({(30.0, 20.0): {1337: _other_protocol_payload(), 2024: _other_protocol_payload()}})


def test_the_cell_reading_carries_the_executed_step_beside_the_segment():
    """A receding reading (A3-a) flies only the first `executed_s` of each `segment_s` forecast:
    both are carried, never silent."""
    reading = g.cell_reading({**_none_payload(established_all=0.7, established_vectored=0.3), "executed_s": 10.0})
    assert reading["segment_s"] == 20.0 and reading["executed_s"] == 10.0 and reading["n"] == 1400
    assert g.cell_name(60.0, 20.0) == "L60_D20"
