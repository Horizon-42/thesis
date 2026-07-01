from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trajectory_data_process.summarize_heading_rejected import (
    find_rejected_files,
    format_report,
    histogram,
    load_rejected,
    metric_error,
    summarize,
)


def _rec(runway: str, course: float | None, track: float | None, tol: float = 20.0) -> dict:
    return {
        "icao24": "abc", "runway": runway, "heading_ok": False,
        "course_error_deg": course, "track_error_deg": track, "heading_tolerance_deg": tol,
    }


class MetricAndHistogramTests(unittest.TestCase):
    def test_metric_error_selects_and_handles_missing(self) -> None:
        rec = _rec("23R", course=40.0, track=3.0)
        self.assertEqual(metric_error(rec, "course"), 40.0)
        self.assertEqual(metric_error(rec, "track"), 3.0)
        self.assertEqual(metric_error(rec, "max"), 40.0)
        # max falls back to whichever value is present
        self.assertEqual(metric_error(_rec("23R", course=None, track=12.0), "max"), 12.0)
        self.assertIsNone(metric_error(_rec("23R", course=None, track=None), "max"))

    def test_histogram_bins_values(self) -> None:
        bins = histogram([5.0, 22.0, 40.0, 178.0], bin_width=10.0)
        self.assertEqual(len(bins), 18)
        counts = {int(lo): c for lo, _hi, c in bins}
        self.assertEqual(counts[0], 1)     # 5°
        self.assertEqual(counts[20], 1)    # 22°
        self.assertEqual(counts[40], 1)    # 40°
        self.assertEqual(counts[170], 1)   # 178° clamps into the last bin
        self.assertEqual(sum(c for _lo, _hi, c in bins), 4)


class SummaryTests(unittest.TestCase):
    RECORDS = [
        _rec("23R", course=5.0, track=45.0),    # geometry_ok_track_bad
        _rec("23R", course=22.0, track=24.0),   # both_bad + borderline
        _rec("05L", course=178.0, track=178.0),  # both_bad + opposite_end
        _rec("30L", course=40.0, track=3.0),    # track_ok_geometry_bad
    ]

    def test_signals_and_stats(self) -> None:
        summary = summarize(self.RECORDS, metric="course", bin_width=10.0)

        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["with_error"], 4)
        self.assertEqual(summary["borderline"], 1)          # only the 22° one
        self.assertEqual(summary["stats"]["max"], 178.0)

        sig = summary["signals"]
        self.assertEqual(sig["geometry_ok_track_bad"], 1)
        self.assertEqual(sig["track_ok_geometry_bad"], 1)
        self.assertEqual(sig["both_bad"], 2)
        self.assertEqual(sig["opposite_end"], 1)

    def test_format_report_is_renderable(self) -> None:
        text = format_report(summarize(self.RECORDS, metric="course", bin_width=10.0))
        self.assertIn("Heading-rejected landings: 4", text)
        self.assertIn("geometry_ok_track_bad", text)

    def test_empty_report(self) -> None:
        text = format_report(summarize([], metric="course", bin_width=10.0))
        self.assertIn("nothing to summarize", text)


class FileScanTests(unittest.TestCase):
    def test_find_and_load_across_airports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "KRDU").mkdir()
            (root / "KSJC").mkdir()
            (root / "KRDU" / "KRDU_23R_heading_rejected.json").write_text(
                json.dumps([_rec("23R", 40.0, 3.0), _rec("23R", 22.0, 24.0)]), encoding="utf-8"
            )
            (root / "KSJC" / "KSJC_30L_heading_rejected.json").write_text(
                json.dumps([_rec("30L", 178.0, 178.0)]), encoding="utf-8"
            )
            # a normal landings file must NOT be picked up
            (root / "KRDU" / "KRDU_23R_landings.json").write_text(json.dumps([{"icao24": "x"}]), encoding="utf-8")

            self.assertEqual(len(find_rejected_files(root)), 2)
            self.assertEqual(len(find_rejected_files(root, airports={"KRDU"})), 1)

            records = load_rejected(find_rejected_files(root))
            self.assertEqual(len(records), 3)
            self.assertEqual({r["_airport"] for r in records}, {"KRDU", "KSJC"})
            self.assertEqual(summarize(records, "course", 10.0)["per_runway"]["KRDU 23R"], 2)


if __name__ == "__main__":
    unittest.main()
