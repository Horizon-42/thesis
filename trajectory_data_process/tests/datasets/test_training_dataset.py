from __future__ import annotations

import unittest

from trajectory_data_process.datasets.training_dataset import build_training_event, make_raw_track_record
from trajectory_data_process.trajectory import Trajectory, TrajectoryPoint


def _trajectory(geo_values: list[float | None]) -> Trajectory:
    points = [
        TrajectoryPoint(i * 10, 0.0, 0.0, geo, None if geo is None else geo + 20, None, False)
        for i, geo in enumerate(geo_values)
    ]
    return Trajectory(icao24="abc123", callsign="TST", dep_airport=None, arr_airport="KRDU", points=points)


def _event(start: int, end: int) -> dict:
    return {
        "event_id": "evt-1",
        "airport": "KRDU",
        "label": "landing",
        "event_time": {"entry_time": 0},
        "point_range": {"start_index": start, "end_index": end},
        "quality": {"num_points": end - start + 1},
    }


class TrainingDatasetTests(unittest.TestCase):
    def test_make_raw_track_record_embeds_trajectory(self) -> None:
        record = make_raw_track_record(
            airport="KRDU", fetch_profile="airport_ops", source_response_ids=["x"], traj=_trajectory([900.0, 950.0])
        )
        self.assertTrue(record["raw_track_id"].startswith("raw_track:sha256:"))
        self.assertEqual(record["source"]["endpoint"], "traffic.data.opensky.history")
        self.assertEqual(len(record["trajectory"]["points"]), 2)

    def test_build_training_event_attaches_points(self) -> None:
        traj = _trajectory([900.0, 950.0, 1000.0])
        ready, quarantine = build_training_event(_event(0, 2), raw_track_id="rt", traj=traj)

        self.assertIsNone(quarantine)
        assert ready is not None
        self.assertEqual(len(ready["training_points"]), 3)
        self.assertEqual(ready["quality"]["points_with_geo_altitude"], 3)
        self.assertEqual(ready["source"]["raw_track_id"], "rt")

    def test_missing_geo_altitude_is_quarantined(self) -> None:
        traj = _trajectory([900.0, None, 1000.0])
        ready, quarantine = build_training_event(_event(0, 2), raw_track_id="rt", traj=traj, require_geo_altitude=True)

        self.assertIsNone(ready)
        assert quarantine is not None
        self.assertEqual(quarantine["reason"], "missing_geo_altitude")


if __name__ == "__main__":
    unittest.main()
