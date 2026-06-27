import json
import shutil
import tempfile
import unittest
from pathlib import Path

from aeroviz_backend import dynamics_comparison_history as history
from aeroviz_backend.dynamics_comparison_backend import DynamicsComparisonBackend


def _payload(**overrides):
    payload = {
        "initialState": {
            "lat": 35.4092,
            "lon": -78.7346,
            "altM": 2300.0,
            "speedMps": 130.0,
            "headingDeg": 45.0,
            "flightPathDeg": -2.0,
            "aircraftType": "A320",
        },
        "control": {"thrustN": 70000.0, "bankDeg": 0.0, "loadFactor": 1.0},
        "durationS": 120.0,
        "dtS": 0.2,
    }
    payload.update(overrides)
    return payload


class _TempHistory(unittest.TestCase):
    """Point the run-history dir at a temp dir so tests never touch the repo."""

    def setUp(self):
        self._orig_history_dir = history.HISTORY_DIR
        self._tmp = Path(tempfile.mkdtemp())
        history.HISTORY_DIR = self._tmp

    def tearDown(self):
        history.HISTORY_DIR = self._orig_history_dir
        shutil.rmtree(self._tmp, ignore_errors=True)


class TestDynamicsComparisonBackend(_TempHistory):
    def setUp(self):
        super().setUp()
        self.result = DynamicsComparisonBackend().run(_payload())

    def test_run_reports_six_systems_with_b_as_reference(self):
        systems = self.result["systems"]
        # A/B/C/D plus F (geodetic RHS, FULL/exact transport) and N (geodetic RHS
        # in the optimizer's normalized coords).
        self.assertEqual([s["key"] for s in systems], ["A", "B", "C", "D", "F", "N"])
        reference = [s["key"] for s in systems if s["isReference"]]
        self.assertEqual(reference, ["B"])
        # every system carries a distinct rgba colour for the legend / path
        colours = {tuple(s["colorRgba"]) for s in systems}
        self.assertEqual(len(colours), 6)

    def test_czml_has_one_hideable_entity_per_system(self):
        czml = self.result["playback"]["czml"]
        self.assertEqual(czml[0]["id"], "document")
        ids = [packet["id"] for packet in czml[1:]]
        self.assertEqual(ids, ["dyncmp-A", "dyncmp-B", "dyncmp-C", "dyncmp-D", "dyncmp-F", "dyncmp-N"])
        # each entity has a time-sampled position, a colored path, and a colored
        # aircraft model (its colour matches the path)
        for packet in czml[1:]:
            position = packet["position"]["cartographicDegrees"]
            self.assertGreater(len(position), 4)
            self.assertEqual(len(position) % 4, 0)
            self.assertIn("path", packet)
            path_color = packet["path"]["material"]["solidColor"]["color"]["rgba"]
            self.assertIn("model", packet)
            self.assertTrue(packet["model"]["gltf"].endswith(".glb"))
            self.assertEqual(packet["model"]["color"]["rgba"], path_color)

    def test_clock_multiplier_keeps_playback_short(self):
        # 120 s at the doc multiplier should land near the ~40 s wall-time target.
        multiplier = self.result["playback"]["multiplier"]
        self.assertGreaterEqual(multiplier, 1)
        wall_time = self.result["durationS"] / multiplier
        self.assertLessEqual(wall_time, 80.0)

    def test_chart_series_excludes_reference_and_aligns_with_distance(self):
        chart = self.result["chart"]
        self.assertEqual(set(chart["series"]), {"A", "C", "D", "F", "N"})
        n = len(chart["distanceKm"])
        self.assertEqual(len(chart["timeS"]), n)
        for key in ("A", "C", "D", "F", "N"):
            for field in ("horiz", "alt", "head", "speed", "fpa"):
                self.assertEqual(len(chart["series"][key][field]), n)
        # distance grows monotonically along the reference
        self.assertTrue(
            all(chart["distanceKm"][i] <= chart["distanceKm"][i + 1] for i in range(n - 1))
        )

    def test_normalized_system_overlays_geodetic(self):
        # System N (geodetic RHS in the optimizer's NORMALIZED metric coords)
        # is a pure change of variables, so its error vs B matches system C's to
        # machine precision -- the proof that normalization changes nothing.
        chart = self.result["chart"]
        for field in ("horiz", "alt", "head", "speed", "fpa"):
            for n_val, c_val in zip(chart["series"]["N"][field], chart["series"]["C"][field]):
                self.assertAlmostEqual(n_val, c_val, places=4)

    def test_geodetic_rhs_matches_reanchored_reference(self):
        # System C (geodetic RHS + transport) should track B to sub-metre, while
        # A (fixed tangent) drifts far more — this is the whole point of the study.
        final = self.result["chart"]["final"]
        self.assertLess(abs(final["C"]["horiz"]), 1.0)
        self.assertGreater(abs(final["A"]["horiz"]), abs(final["C"]["horiz"]) + 10.0)

    def test_full_transport_tracks_reference_like_approx(self):
        # System F (geodetic RHS, FULL/exact transport) also tracks B to
        # sub-metre, and differs from C (approx transport) only by the tiny psi
        # cross term — so its final error sits right next to C's.  (The cross
        # term's effect is sub-metre, below RK4 truncation at the run dt, so we
        # assert closeness, not a strict F<C ordering, which would be noisy.)
        final = self.result["chart"]["final"]
        self.assertLess(abs(final["F"]["horiz"]), 1.0)
        self.assertLess(abs(final["F"]["horiz"] - final["C"]["horiz"]), 0.5)

    def test_run_output_is_json_serialisable(self):
        # No numpy scalars leak through (would raise on json.dumps).
        json.dumps(self.result)

    def test_control_and_horizon_are_clamped(self):
        result = DynamicsComparisonBackend().run(
            _payload(
                control={"thrustN": 1e9, "bankDeg": 999.0, "loadFactor": 50.0},
                durationS=1.0,  # below MIN_DURATION_S
                dtS=0.0001,     # below MIN_DT_S
            )
        )
        self.assertTrue(result["ok"])
        # clamped horizon still yields a usable playback
        self.assertGreaterEqual(result["durationS"], 5.0)
        self.assertGreater(len(result["chart"]["distanceKm"]), 1)

    def test_run_persists_a_record(self):
        # setUp already ran one comparison.
        self.assertEqual(self.result["historyCount"], 1)


class TestDynamicsComparisonHistory(_TempHistory):
    def test_count_increments_and_average_spans_shortest_run(self):
        backend = DynamicsComparisonBackend()
        self.assertEqual(backend.history_count()["historyCount"], 0)

        r1 = backend.run(_payload(durationS=120.0, dtS=0.2))
        r2 = backend.run(_payload(durationS=200.0, dtS=0.2))
        self.assertEqual(r2["historyCount"], 2)

        averaged = backend.average()
        self.assertTrue(averaged["ok"])
        self.assertEqual(averaged["runCount"], 2)
        chart = averaged["chart"]
        # averaged onto one common grid; series excludes the reference B
        self.assertEqual(set(chart["series"]), {"A", "C", "D", "F", "N"})
        n = len(chart["distanceKm"])
        self.assertGreater(n, 1)
        for key in ("A", "C", "D"):
            for field in ("horiz", "alt", "head", "speed", "fpa"):
                self.assertEqual(len(chart["series"][key][field]), n)
        # the common grid spans only as far as the shorter-range run (every run
        # covers it), and starts at 0
        self.assertEqual(chart["distanceKm"][0], 0.0)
        shortest = min(r1["chart"]["distanceKm"][-1], r2["chart"]["distanceKm"][-1])
        self.assertAlmostEqual(chart["distanceKm"][-1], shortest, places=3)
        json.dumps(averaged)  # serialisable (no numpy scalars)

    def test_average_without_history_reports_not_ok(self):
        result = DynamicsComparisonBackend().average()
        self.assertFalse(result["ok"])

    def test_clear_empties_history(self):
        backend = DynamicsComparisonBackend()
        backend.run(_payload())
        self.assertEqual(backend.clear()["historyCount"], 0)
        self.assertEqual(backend.history_count()["historyCount"], 0)


if __name__ == "__main__":
    unittest.main()
