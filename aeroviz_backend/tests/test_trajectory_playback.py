import math
import unittest

import numpy as np

from aeroviz_backend import trajectory_playback
from aircraft.aircraft_sets import A320
from common import GeodeticState


def _trail_packets(czml: list) -> list:
    return [
        packet for packet in czml
        if str(packet["id"]).startswith("optimized-trajectory-trail-")
    ]


def _initial_state() -> GeodeticState:
    return GeodeticState(
        latitude=35.95,
        longitude=-78.90,
        altitude=1500.0,
        V=95.0,
        psi=math.radians(220.0),
        gamma=math.radians(-2.0),
        m=A320.mass.max_takeoff_kg,
    )


class TestTrajectoryPlayback(unittest.TestCase):
    def test_build_playback_emits_document_aircraft_and_segment_polylines(self):
        node_control = np.array([
            [40000.0, math.radians(5.0), 1.0],
            [40000.0, math.radians(25.0), 1.1],
            [40000.0, math.radians(40.0), 1.2],
            [40000.0, math.radians(0.0), 1.0],
        ])

        playback = trajectory_playback.build_optimized_trajectory_playback(
            "casadiDirectCollocation",
            _initial_state(),
            node_control,
            40.0,
            A320,
        )

        self.assertIsNotNone(playback)
        self.assertEqual(playback["multiplier"], 1)
        czml = playback["czml"]

        # document packet first, with a clock spanning the horizon. The interval
        # end keeps millisecond precision — a second-truncated end parks the
        # frontend clock up to 1 s (≈75 m of flight) before the true final
        # sample (the phantom "25 m final horiz err").
        self.assertEqual(czml[0]["id"], "document")
        self.assertEqual(
            czml[0]["clock"]["interval"],
            "2026-01-01T00:00:00Z/2026-01-01T00:00:40.000Z",
        )

        # one aircraft entity with time-sampled position; orientation is set on
        # the frontend (so the model banks like the live Pilot aircraft).
        aircraft = czml[1]
        self.assertEqual(aircraft["id"], "optimized-trajectory-aircraft")
        position = aircraft["position"]["cartographicDegrees"]
        self.assertEqual(len(position) % 4, 0)
        self.assertNotIn("orientation", aircraft)

        # the trail is one short polyline per sample interval
        sample_count = len(playback["samples"])
        trail = _trail_packets(czml)
        self.assertEqual(len(trail), sample_count - 1)
        # each trail step is a 2-point polyline
        self.assertEqual(
            len(trail[0]["polyline"]["positions"]["cartographicDegrees"]), 6
        )

    def test_colour_tracks_segment_order(self):
        # Colour is by control-segment order: first segment cool, last warm.
        node_control = np.array([
            [40000.0, math.radians(0.0), 1.0],
            [40000.0, math.radians(10.0), 1.0],
            [40000.0, math.radians(20.0), 1.0],
            [40000.0, math.radians(30.0), 1.0],
        ])

        playback = trajectory_playback.build_optimized_trajectory_playback(
            "casadiDirectCollocation",
            _initial_state(),
            node_control,
            40.0,
            A320,
        )
        trail = _trail_packets(playback["czml"])
        first_rgb = trail[0]["polyline"]["material"]["solidColor"]["color"]["rgba"]
        last_rgb = trail[-1]["polyline"]["material"]["solidColor"]["color"]["rgba"]

        # first segment is blue-dominant, last segment is red-dominant
        self.assertGreater(first_rgb[2], first_rgb[0])
        self.assertGreater(last_rgb[0], last_rgb[2])

    def test_trail_grows_via_availability(self):
        node_control = np.array([
            [40000.0, math.radians(5.0), 1.0],
            [40000.0, math.radians(5.0), 1.0],
        ])
        playback = trajectory_playback.build_optimized_trajectory_playback(
            "casadiDirectCollocation",
            _initial_state(),
            node_control,
            20.0,
            A320,
        )
        trail = _trail_packets(playback["czml"])
        starts = [packet["availability"].split("/")[0] for packet in trail]
        # availability windows open progressively (the trail grows over time)
        self.assertEqual(starts, sorted(starts))
        self.assertLess(starts[0], starts[-1])
        # the first step does not appear at t=0 — it trails the aircraft by one step
        self.assertNotEqual(starts[0], "2026-01-01T00:00:00.000Z")

    def test_samples_are_monotonic_and_carry_control(self):
        node_control = np.array([
            [40000.0, math.radians(10.0), 1.05],
            [40000.0, math.radians(20.0), 1.10],
        ])
        playback = trajectory_playback.build_optimized_trajectory_playback(
            "casadiDirectCollocation",
            _initial_state(),
            node_control,
            20.0,
            A320,
        )
        samples = playback["samples"]
        self.assertGreater(len(samples), 2)
        times = [sample["t"] for sample in samples]
        self.assertTrue(all(times[i] < times[i + 1] for i in range(len(times) - 1)))
        self.assertEqual(samples[0]["t"], 0.0)
        self.assertIn("loadFactor", samples[0])
        self.assertIn("bankDeg", samples[0])

    def test_alpha_optimizer_uses_attack_control(self):
        node_control = np.array([[40000.0, math.radians(5.0), math.radians(3.0)]])
        playback = trajectory_playback.build_optimized_trajectory_playback(
            "transcription",
            _initial_state(),
            node_control,
            10.0,
            A320,
        )
        self.assertIsNotNone(playback)
        self.assertIn("attackDeg", playback["samples"][0])
        self.assertNotIn("loadFactor", playback["samples"][0])

    def test_returns_none_without_controls(self):
        self.assertIsNone(
            trajectory_playback.build_optimized_trajectory_playback(
                "casadiDirectCollocation", _initial_state(), [], 40.0, A320,
            )
        )
        self.assertIsNone(
            trajectory_playback.build_optimized_trajectory_playback(
                "casadiDirectCollocation",
                _initial_state(),
                np.array([[40000.0, 0.0, 1.0]]),
                0.0,
                A320,
            )
        )


if __name__ == "__main__":
    unittest.main()


class TestPlaybackTerminalDrift(unittest.TestCase):
    """The drift guard's measurement: last rollout sample vs the target, no re-simulation."""

    def test_zero_drift_when_the_rollout_ends_on_the_target(self):
        playback = {"samples": [
            {"lat": 35.9, "lon": -78.9}, {"lat": 35.87445, "lon": -78.80196},
        ]}
        drift = trajectory_playback.playback_terminal_drift_m(playback, 35.87445, -78.80196)
        self.assertAlmostEqual(drift, 0.0, places=6)

    def test_drift_measures_the_final_sample_only(self):
        # ~0.01 deg of latitude ≈ 1113 m; the earlier (far) sample must not matter.
        playback = {"samples": [
            {"lat": 40.0, "lon": -70.0}, {"lat": 35.88445, "lon": -78.80196},
        ]}
        drift = trajectory_playback.playback_terminal_drift_m(playback, 35.87445, -78.80196)
        self.assertAlmostEqual(drift, 1113.0, delta=5.0)

    def test_none_without_samples(self):
        self.assertIsNone(trajectory_playback.playback_terminal_drift_m({"samples": []}, 0.0, 0.0))
        self.assertIsNone(trajectory_playback.playback_terminal_drift_m({}, 0.0, 0.0))


def test_document_packet_keeps_fractional_end_seconds():
    # Regression: iso() second-truncation of the clock interval end parked the
    # frontend clock 0-1 s before the trajectory's final sample (~25 m at 75 m/s).
    from datetime import timedelta

    from aeroviz_backend.czml_common import EPOCH, document_packet

    packet = document_packet(EPOCH + timedelta(seconds=324.3384), 1, "x")
    assert packet["clock"]["interval"] == "2026-01-01T00:00:00Z/2026-01-01T00:05:24.338Z"
