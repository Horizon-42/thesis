import csv
import math
import unittest
from argparse import Namespace
from pathlib import Path

from tnm_elevation_downloader.download_tnm_elevation import (
    DATASET_SPECS,
    DEFAULT_RADIUS_KM,
    TnmProduct,
    bbox_from_point,
    build_query_contexts,
    dem_candidate_keys,
    download_one,
    format_bytes,
    latest_per_footprint,
    load_airports,
    normalize_codes,
    progress_bar,
    target_path,
    validate_args,
)


def write_airport_csv(path: Path) -> None:
    fieldnames = [
        "ident",
        "type",
        "name",
        "latitude_deg",
        "longitude_deg",
        "icao_code",
        "iata_code",
        "gps_code",
        "local_code",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "ident": "KRDU",
                "type": "large_airport",
                "name": "Raleigh Durham International Airport",
                "latitude_deg": "35.87764",
                "longitude_deg": "-78.78747",
                "icao_code": "KRDU",
                "iata_code": "RDU",
                "gps_code": "KRDU",
                "local_code": "RDU",
            }
        )


class TnmElevationDownloaderTests(unittest.TestCase):
    def test_bbox_from_point_uses_kilometres(self) -> None:
        bbox = bbox_from_point(35.87764, -78.78747, DEFAULT_RADIUS_KM)
        lat_delta = bbox[3] - 35.87764
        expected_lat_delta = DEFAULT_RADIUS_KM / 111.32
        self.assertTrue(math.isclose(lat_delta, expected_lat_delta, rel_tol=1e-6))
        self.assertLess(bbox[0], -78.78747)
        self.assertGreater(bbox[2], -78.78747)
        self.assertLess(bbox[1], 35.87764)
        self.assertGreater(bbox[3], 35.87764)

    def test_bbox_rejects_less_than_five_km(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 5"):
            bbox_from_point(35.0, -78.0, 4.9)

    def test_load_airports_matches_aliases(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "airports.csv"
            write_airport_csv(csv_path)
            airports = load_airports(csv_path, ["RDU", "KRDU"])
        self.assertEqual(len(airports), 1)
        self.assertEqual(airports[0].code, "KRDU")
        self.assertEqual(airports[0].latitude, 35.87764)

    def test_normalize_codes_accepts_commas_and_dedupes(self) -> None:
        self.assertEqual(
            normalize_codes(["krdu,kden", "KRDU", " ksfo "]),
            ["KRDU", "KDEN", "KSFO"],
        )

    def test_build_query_contexts_for_lat_lon(self) -> None:
        args = Namespace(
            airports=[],
            airport_csv=Path("unused.csv"),
            lat=35.87764,
            lon=-78.78747,
            label="KRDU_POINT",
            bbox=None,
            radius_km=5.0,
        )
        contexts = build_query_contexts(args)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].group, "KRDU_POINT")
        self.assertLess(contexts[0].bbox[0], -78.78747)
        self.assertGreater(contexts[0].bbox[2], -78.78747)

    def test_dem_fallback_order_starts_at_selected_dataset(self) -> None:
        self.assertEqual(
            dem_candidate_keys("dem_s1m", True),
            ["dem_s1m", "dem_opr", "dem_13", "dem_1"],
        )
        self.assertEqual(dem_candidate_keys("dem_1m", False), ["dem_1m"])

    def test_format_bytes_uses_binary_units(self) -> None:
        self.assertEqual(format_bytes(0), "0 B")
        self.assertEqual(format_bytes(1536), "1.5 KiB")
        self.assertEqual(format_bytes(1024 * 1024), "1.0 MiB")

    def test_progress_bar_bounds_fraction(self) -> None:
        self.assertEqual(progress_bar(-1, width=4), "[----]")
        self.assertEqual(progress_bar(0.5, width=4), "[##--]")
        self.assertEqual(progress_bar(2, width=4), "[####]")

    def test_target_path_uses_dataset_subdir(self) -> None:
        context = Namespace(group="KRDU", label="KRDU", bbox=[], latitude=0.0, longitude=0.0)
        product = TnmProduct(
            context=context,
            spec=DATASET_SPECS["dsm_lpc"],
            item={
                "title": "USGS Lidar Point Cloud sample",
                "urls": {"LAZ": "https://example.test/path/sample.laz"},
            },
        )
        self.assertEqual(
            target_path(Path("out"), product),
            Path("out/KRDU/dsm/source_laz/sample.laz"),
        )

    def test_latest_per_footprint_keeps_newest_product(self) -> None:
        context = Namespace(group="KRDU", label="KRDU", bbox=[], latitude=0.0, longitude=0.0)
        old = TnmProduct(
            context=context,
            spec=DATASET_SPECS["dem_13"],
            item={
                "title": "old",
                "publicationDate": "2020-01-01",
                "boundingBox": {"minX": -79, "minY": 35, "maxX": -78, "maxY": 36},
                "downloadURL": "https://example.test/old.tif",
            },
        )
        new = TnmProduct(
            context=context,
            spec=DATASET_SPECS["dem_13"],
            item={
                "title": "new",
                "publicationDate": "2025-01-01",
                "boundingBox": {"minX": -79, "minY": 35, "maxX": -78, "maxY": 36},
                "downloadURL": "https://example.test/new.tif",
            },
        )
        self.assertEqual(latest_per_footprint([old, new]), [new])

    def test_validate_args_requires_input(self) -> None:
        args = Namespace(
            airports=[],
            lat=None,
            lon=None,
            bbox=None,
            radius_km=5.0,
            limit=None,
            page_size=100,
            workers=4,
        )
        with self.assertRaisesRegex(ValueError, "provide at least one"):
            validate_args(args)

    def test_validate_args_rejects_nonpositive_workers(self) -> None:
        args = Namespace(
            airports=["KRDU"],
            lat=None,
            lon=None,
            bbox=None,
            radius_km=5.0,
            limit=None,
            page_size=100,
            workers=0,
        )
        with self.assertRaisesRegex(ValueError, "workers"):
            validate_args(args)

    def test_download_one_reports_progress(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.tif"
            source.write_bytes(b"x" * 4096)
            context = Namespace(group="KRDU", label="KRDU", bbox=[], latitude=0.0, longitude=0.0)
            product = TnmProduct(
                context=context,
                spec=DATASET_SPECS["dem_13"],
                item={
                    "title": "local",
                    "downloadURL": source.as_uri(),
                    "sizeInBytes": 4096,
                },
            )
            updates: list[int] = []
            result = download_one(
                product,
                out_dir=tmp_path / "out",
                overwrite=False,
                timeout=10,
                retries=1,
                progress_callback=lambda _target, byte_count: updates.append(byte_count),
            )

        self.assertEqual(result.status, "downloaded")
        self.assertEqual(result.size_in_bytes, 4096)
        self.assertEqual(updates[-1], 4096)
