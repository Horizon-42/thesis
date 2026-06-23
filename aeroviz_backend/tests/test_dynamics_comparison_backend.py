import json
import unittest

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


class TestDynamicsComparisonBackend(unittest.TestCase):
    def setUp(self):
        self.result = DynamicsComparisonBackend().run(_payload())

    def test_run_reports_four_systems_with_b_as_reference(self):
        systems = self.result["systems"]
        self.assertEqual([s["key"] for s in systems], ["A", "B", "C", "D"])
        reference = [s["key"] for s in systems if s["isReference"]]
        self.assertEqual(reference, ["B"])
        # every system carries a distinct rgba colour for the legend / path
        colours = {tuple(s["colorRgba"]) for s in systems}
        self.assertEqual(len(colours), 4)

    def test_czml_has_one_hideable_entity_per_system(self):
        czml = self.result["playback"]["czml"]
        self.assertEqual(czml[0]["id"], "document")
        ids = [packet["id"] for packet in czml[1:]]
        self.assertEqual(ids, ["dyncmp-A", "dyncmp-B", "dyncmp-C", "dyncmp-D"])
        # each entity has a time-sampled position and a colored path
        for packet in czml[1:]:
            position = packet["position"]["cartographicDegrees"]
            self.assertGreater(len(position), 4)
            self.assertEqual(len(position) % 4, 0)
            self.assertIn("path", packet)
            self.assertIn("rgba", packet["path"]["material"]["solidColor"]["color"])

    def test_clock_multiplier_keeps_playback_short(self):
        # 120 s at the doc multiplier should land near the ~40 s wall-time target.
        multiplier = self.result["playback"]["multiplier"]
        self.assertGreaterEqual(multiplier, 1)
        wall_time = self.result["durationS"] / multiplier
        self.assertLessEqual(wall_time, 80.0)

    def test_chart_series_excludes_reference_and_aligns_with_distance(self):
        chart = self.result["chart"]
        self.assertEqual(set(chart["series"]), {"A", "C", "D"})
        n = len(chart["distanceKm"])
        self.assertEqual(len(chart["timeS"]), n)
        for key in ("A", "C", "D"):
            for field in ("horiz", "alt", "head", "speed"):
                self.assertEqual(len(chart["series"][key][field]), n)
        # distance grows monotonically along the reference
        self.assertTrue(
            all(chart["distanceKm"][i] <= chart["distanceKm"][i + 1] for i in range(n - 1))
        )

    def test_geodetic_rhs_matches_reanchored_reference(self):
        # System C (geodetic RHS + transport) should track B to sub-metre, while
        # A (fixed tangent) drifts far more — this is the whole point of the study.
        final = self.result["chart"]["final"]
        self.assertLess(abs(final["C"]["horiz"]), 1.0)
        self.assertGreater(abs(final["A"]["horiz"]), abs(final["C"]["horiz"]) + 10.0)

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


if __name__ == "__main__":
    unittest.main()
