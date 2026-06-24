import json
import unittest

from aeroviz_backend.procedure_constraint import (
    AltitudeWindow,
    ProcedureConstraint,
    altitude_window_from_cifp,
)


def _leg(seq, terminator, end_fix, role, altitude, geometry_ft):
    return {
        "legId": f"leg:R:{seq:03d}",
        "sequence": seq,
        "segmentType": "final",
        "path": {
            "pathTerminator": terminator,
            "constructionMethod": "track_to_fix",
            "startFixRef": None,
            "endFixRef": end_fix,
        },
        "termination": {"kind": "fix", "fixRef": end_fix},
        "constraints": {"altitude": altitude, "speedKt": None, "geometryAltitudeFt": geometry_ft},
        "roleAtEnd": role,
        "sourceRefs": ["src:cifp-detail"],
        "quality": {"status": "exact", "sourceLine": seq, "renderedInPlanView": True},
    }


# A minimal KRDU R05LY-shaped detail document (final branch only), using the
# authoritative published chart values (RNAV (GPS) Y RWY 5L). Self-contained so
# the backend suite needs no generated-data bundle; the real parser-vs-chart
# guarantee lives in the aeroviz-4d python suite, which builds from CIFP.
R05LY_DOCUMENT = {
    "procedureUid": "KRDU-R05LY-RW05L",
    "airport": {"icao": "KRDU", "faa": "RDU", "name": "Raleigh-Durham International Airport"},
    "runway": {"ident": "RW05L", "threshold": {"lon": -78.80196389, "lat": 35.87445, "elevationFt": 367}},
    "procedure": {"runwayIdent": "RW05L", "baseBranchIdent": "R", "procedureFamily": "RNAV_GPS"},
    "fixes": [
        {"fixId": "fix:SCHOO", "ident": "SCHOO", "position": {"lon": -78.92647222, "lat": 35.77341389}},
        {"fixId": "fix:WEPAS", "ident": "WEPAS", "position": {"lon": -78.88295556, "lat": 35.80876667}},
        {"fixId": "fix:RW05L", "ident": "RW05L", "position": {"lon": -78.80196389, "lat": 35.87445}},
    ],
    "branches": [
        {
            "branchId": "branch:R",
            "branchIdent": "R",
            "branchRole": "final",
            "legs": [
                _leg(10, "IF", "fix:SCHOO", "IF", {"qualifier": "atOrAbove", "valueFt": 3000, "rawText": "3000 ft"}, 3000),
                _leg(20, "TF", "fix:WEPAS", "FAF", {"qualifier": "atOrAbove", "valueFt": 2200, "rawText": "2200 ft"}, 2200),
                _leg(30, "TF", "fix:RW05L", "MAPt", {"qualifier": "at", "valueFt": 424, "rawText": "424 ft"}, 367),
            ],
        }
    ],
    "verticalProfiles": [{"glidepathAngleDeg": 3.0, "thresholdCrossingHeightFt": 57.4}],
    "displayHints": {"nominalSpeedKt": 140.0},
}


class TestAltitudeWindowFromCifp(unittest.TestCase):
    def test_none_for_uncoded_altitude(self):
        self.assertIsNone(altitude_window_from_cifp(None))

    def test_at_or_above(self):
        window = altitude_window_from_cifp(
            {"qualifier": "atOrAbove", "valueFt": 3000, "rawText": "3000 ft"}
        )
        self.assertEqual(window, AltitudeWindow("AT_OR_ABOVE", min_ft_msl=3000.0, source_text="3000 ft"))
        self.assertEqual(window.reference_ft, 3000.0)

    def test_at_or_below(self):
        window = altitude_window_from_cifp(
            {"qualifier": "atOrBelow", "valueFt": 5000, "rawText": "5000 ft"}
        )
        self.assertEqual(window.kind, "AT_OR_BELOW")
        self.assertEqual(window.reference_ft, 5000.0)

    def test_plain_crossing_is_at_with_equal_bounds(self):
        window = altitude_window_from_cifp(
            {"qualifier": "at", "valueFt": 2200, "rawText": "2200 ft"}
        )
        self.assertEqual(window, AltitudeWindow("AT", 2200.0, 2200.0, "2200 ft"))

    def test_block_keeps_both_bounds(self):
        # Matches the frontend consolidation AND the parser fix: a block
        # altitude is a WINDOW with both bounds, not a single AT value.
        window = altitude_window_from_cifp(
            {"qualifier": "block", "valueFt": 7000, "rawText": "6000-7000 ft"}
        )
        self.assertEqual(window, AltitudeWindow("WINDOW", 6000.0, 7000.0, "6000-7000 ft"))


class TestProcedureConstraintFromDetailDocument(unittest.TestCase):
    """Golden read of a R05LY-shaped document against its published chart."""

    @classmethod
    def setUpClass(cls):
        cls.constraint = ProcedureConstraint.from_detail_document(R05LY_DOCUMENT)

    def test_final_branch_waypoint_sequence(self):
        idents = [wp.ident for wp in self.constraint.waypoints]
        roles = [wp.role for wp in self.constraint.waypoints]
        self.assertEqual(idents, ["SCHOO", "WEPAS", "RW05L"])
        self.assertEqual(roles, ["IF", "FAF", "MAPt"])

    def test_published_crossing_altitudes_match_chart(self):
        by_ident = {wp.ident: wp for wp in self.constraint.waypoints}
        # Chart: SCHOO (IF) 3000, WEPAS (FAF) 2200, RW05L threshold-crossing
        # 424 = threshold 367 + TCH 57.
        self.assertEqual(by_ident["SCHOO"].altitude_ref_ft, 3000.0)
        self.assertEqual(by_ident["WEPAS"].altitude_ref_ft, 2200.0)
        self.assertEqual(by_ident["RW05L"].altitude_ref_ft, 424.0)
        self.assertEqual(by_ident["RW05L"].geometry_alt_ft, 367.0)

    def test_coded_glidepath_matches_chart(self):
        self.assertIsNotNone(self.constraint.glidepath)
        self.assertAlmostEqual(self.constraint.glidepath.angle_deg, 3.0)  # chart: GP 3.00
        self.assertAlmostEqual(self.constraint.glidepath.tch_ft, 57.4, places=1)  # chart: TCH 57

    def test_final_course_is_runway_aligned(self):
        # WEPAS -> RW05L: ~054 deg magnetic, ~9 deg W variation -> ~045 deg true.
        self.assertGreater(self.constraint.approach_course_deg, 40.0)
        self.assertLess(self.constraint.approach_course_deg, 50.0)

    def test_descent_is_monotonic(self):
        self.assertTrue(self.constraint.is_monotonic_descent())

    def test_summary_shape(self):
        self.assertEqual(
            self.constraint.summary(),
            {
                "waypointCount": 3,
                "monotonicDescent": True,
                "firstFixIdent": "SCHOO",
                "lastFixIdent": "RW05L",
            },
        )


class TestProcedureConstraintRoundTrip(unittest.TestCase):
    """The frontend builds the constraint and ships JSON; the backend parses the
    same shape. Round-tripping it through the wire payload must be lossless on
    the fields the optimizer reads."""

    def test_from_payload_round_trips_waypoints(self):
        built = ProcedureConstraint.from_detail_document(R05LY_DOCUMENT)

        payload = {
            "procedureUid": built.procedure_uid,
            "airportIcao": built.airport_icao,
            "runwayIdent": built.runway_ident,
            "branchId": built.branch_id,
            "approachCourseDeg": built.approach_course_deg,
            "glidepath": {"angleDeg": built.glidepath.angle_deg, "tchFt": built.glidepath.tch_ft},
            "nominalSpeedKt": built.nominal_speed_kt,
            "waypoints": [
                {
                    "fixId": wp.fix_id,
                    "ident": wp.ident,
                    "role": wp.role,
                    "legType": wp.leg_type,
                    "lonDeg": wp.lon_deg,
                    "latDeg": wp.lat_deg,
                    "altitude": wp.altitude.to_payload() if wp.altitude else None,
                    "altitudeRefFt": wp.altitude_ref_ft,
                    "geometryAltFt": wp.geometry_alt_ft,
                    "speedMaxKt": wp.speed_max_kt,
                    "distanceFromStartM": wp.distance_from_start_m,
                }
                for wp in built.waypoints
            ],
        }
        # JSON serialisability is part of the contract.
        parsed = ProcedureConstraint.from_payload(json.loads(json.dumps(payload)))

        self.assertEqual(
            [wp.ident for wp in parsed.waypoints],
            [wp.ident for wp in built.waypoints],
        )
        self.assertEqual(
            [wp.altitude_ref_ft for wp in parsed.waypoints],
            [wp.altitude_ref_ft for wp in built.waypoints],
        )
        self.assertEqual(parsed.summary(), built.summary())

    def test_from_payload_rejects_empty(self):
        self.assertIsNone(ProcedureConstraint.from_payload(None))
        self.assertIsNone(ProcedureConstraint.from_payload({"waypoints": []}))


if __name__ == "__main__":
    unittest.main()
