"""Test vectors for the tutorial's JavaScript ports: the Python code runs random cases, the JS must give the same answers.

    conda run -n aeroviz python scripts/two_tier_tutorial/tests/make_vectors.py
    node scripts/two_tier_tutorial/tests/test_ports.js
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "4dTrajectory"), str(ROOT / "geokit" / "src")]

from ts_transformer.autopilot.lateral import stopping_rate_deg_s, word_rate  # noqa: E402
from ts_transformer.autopilot.params import ExecutorParams  # noqa: E402
from ts_transformer.instructions.artefact import load_spec  # noqa: E402
from ts_transformer.instructions.grammar import InForce, _ONE, _rules, column_mask, column_words  # noqa: E402
from ts_transformer.instructions.labeller.lateral import per_step_words  # noqa: E402
from ts_transformer.instructions.words import Words  # noqa: E402

ARTEFACT = Path("/home/supercomputing/studys/thesis/4dTrajectory/outputs/POOLED/instruction_language/v12_20261005")


def main() -> None:
    spec = load_spec(ARTEFACT)
    words = Words(spec)
    rng = np.random.default_rng(7)
    out: dict = {"levels": words.altitude_levels.tolist(), "tol": words.altitude_tolerances.tolist(),
                 "counts": words.class_counts(), "n_descent": words.n_descent}
    # 1. the rules, on random rows
    cases = []
    for _ in range(4000):
        n = int(rng.integers(1, 9))
        first = bool(rng.random() < 0.15)
        in_runway = int(rng.integers(0, n))
        in_go = bool(rng.random() < 0.25)
        in_alt = int(rng.integers(0, words.n_altitude_levels + 1))
        in_angle = int(rng.integers(0, words.n_descent + 2))
        step = [int(rng.choice([-1, -2] + list(range(0, n + 1)))), int(rng.choice([-1, 0, 3, 71])),
                int(rng.choice([-1, words.altitude_no_level_off] + list(rng.integers(0, words.n_altitude_levels, 4)))),
                int(rng.choice([-1] + list(range(0, words.n_descent + 2)))), int(rng.choice([-1, 5, 47]))]
        height = float(rng.uniform(0, 3000))
        code, runway, go, alt, angle = _rules(_ONE, words, first, in_runway, in_go, in_alt, in_angle, *step, height, n)
        cases.append({"first": first, "in": [in_runway, in_go, in_alt, in_angle], "step": step, "height": height, "n": n,
                      "code": int(code), "after": [int(runway), bool(go), int(alt), int(angle)]})
    out["rules"] = cases
    # 2. the mask of each column (D62)
    masks = []
    for _ in range(120):
        n = int(rng.integers(2, 6))
        first = bool(rng.random() < 0.2)
        in_force = None if first else InForce(int(rng.integers(0, n)), bool(rng.random() < 0.3), int(rng.integers(0, 72)),
                                              int(rng.integers(0, words.n_altitude_levels + 1)), int(rng.integers(0, 6)),
                                              int(rng.integers(0, 48)))
        column = int(rng.integers(0, 5))
        height = float(rng.uniform(0, 2500))
        said = [int(rng.choice(column_words(c, words, n))) for c in range(column)]
        permitted = {}
        for c in range(column + 1, 5):
            if rng.random() < 0.5:
                allow = rng.random(len(column_words(c, words, n))) < 0.6
                if not allow.any():
                    allow[-1] = True
                permitted[c] = allow
        got = column_mask([in_force], np.array([said], dtype=np.int64).reshape(1, column), column, np.array([height]), [n],
                          words, {c: a[None, :] for c, a in permitted.items()})[0]
        masks.append({"in": None if in_force is None else [in_force.runway, in_force.go_around, in_force.heading,
                                                           in_force.altitude, in_force.angle, in_force.speed],
                      "said": said, "column": column, "height": height, "n": n,
                      "permitted": {str(c): [int(w) for w, ok in zip(column_words(c, words, n), a) if ok] for c, a in permitted.items()},
                      "mask": [bool(v) for v in got]})
    out["masks"] = masks
    # 3. the heading reading
    reads = []
    for _ in range(60):
        rows = int(rng.integers(20, 120))
        track = np.cumsum(rng.normal(0, 1.5, rows)) + rng.uniform(0, 360)
        course = np.full(rows, float(rng.uniform(0, 360)))
        lead = int(rng.integers(0, 4))
        reads.append({"track": track.tolist(), "course": course.tolist(), "lead": lead,
                      "words": [[int(r), float(t)] for r, t in per_step_words(track, course, 5.0, lead)]})
    out["reads"] = reads
    # 4. the lateral law
    params = ExecutorParams(cycle_s=1.0, bank_rate_deg_s=5.0, path_time_constant_s=2.0, path_rate_factor=2.0,
                            timeout_factor=1.5, start_rule="displacement-2s")
    rates = []
    for _ in range(300):
        e, to_go, v = float(rng.uniform(-170, 170)), float(rng.uniform(-2, 40)), float(rng.uniform(40, 250))
        rate = word_rate(torch.tensor([e]), torch.tensor([to_go]), torch.tensor([v]), params, spec)
        stop = stopping_rate_deg_s(torch.tensor([e]), torch.tensor([v]), params)
        rates.append({"e": e, "to_go": to_go, "v": v, "rate": float(rate[0]), "stop": float(stop[0])})
    out["rates"] = rates
    out["rmax"] = spec.turn_rate_max_deg_s
    (HERE / "vectors.json").write_text(json.dumps(out), encoding="utf-8")
    print("wrote", len(cases), "rule cases,", len(masks), "mask cases,", len(reads), "readings,", len(rates), "rates")


if __name__ == "__main__":
    main()
