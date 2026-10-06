"""IAF→runway path assembly and the shortest-path ranking proxy (synthetic procedures)."""

from procedure.constraint import Glidepath, ProcedureConstraint, ProcedureConstraintWaypoint
from procedure.iaf import concat_to_runway, path_length_m


def _pc(waypoints, *, branch_id="branch:X", nominal_kt=140.0):
    return ProcedureConstraint(
        procedure_uid="UID", airport_icao="KRDU", runway_ident="RW05L",
        branch_id=branch_id, approach_course_deg=45.0,
        glidepath=Glidepath(3.0, 50.0), nominal_speed_kt=nominal_kt,
        waypoints=tuple(waypoints),
    )


def _wp(ident, lat, lon, *, alt_ft=None, fix_id=None):
    return ProcedureConstraintWaypoint(
        fix_id=ident if fix_id is None else fix_id, ident=ident, role="IF", leg_type="TF",
        lon_deg=lon, lat_deg=lat, altitude=None, altitude_ref_ft=alt_ft,
        geometry_alt_ft=None, speed_max_kt=None, distance_from_start_m=0.0,
    )


def test_concat_to_runway_joins_transition_and_final():
    # transition CHWDR -> SCHOO  +  final SCHOO -> RW05L  =>  CHWDR -> SCHOO -> RW05L
    trans = _pc([_wp("CHWDR", 36.10, -78.70), _wp("SCHOO", 36.00, -78.60)], branch_id="branch:T")
    final = _pc([_wp("SCHOO", 36.00, -78.60), _wp("RW05L", 35.88, -78.78)], branch_id="branch:R")
    merged = concat_to_runway(trans, final)
    assert [w.ident for w in merged.waypoints] == ["CHWDR", "SCHOO", "RW05L"]
    assert merged.branch_id == "branch:T"                 # labelled by the transition (the IAF)
    dists = [w.distance_from_start_m for w in merged.waypoints]
    assert dists[0] == 0.0 and dists[1] < dists[2]        # cumulative, recomputed

    # a transition that does not end at the final's first fix does not feed it
    stray = _pc([_wp("CHWDR", 36.10, -78.70), _wp("ELSEW", 36.20, -78.90)], branch_id="branch:T")
    assert concat_to_runway(stray, final) is None

    # Two identifier-LESS endpoints ("" is the constraint layer's missing-fixId default)
    # must not read as the same fix — "" == "" used to satisfy the fix_id half of the
    # check and concatenate a transition onto a final it does not feed.
    blank_end = _pc([_wp("CHWDR", 36.10, -78.70),
                     _wp("SCHOO", 36.00, -78.60, fix_id="")], branch_id="branch:T")
    blank_start = _pc([_wp("ELSEW", 36.05, -78.65, fix_id=""),
                       _wp("RW05L", 35.88, -78.78)], branch_id="branch:R")
    assert concat_to_runway(blank_end, blank_start) is None


def test_path_length_ranks_shorter_path_lower():
    # The naive selector ranks IAFs by this polyline length (no solve; the old
    # Lagrange-curve proxy inflated cornered routes and could invert the ranking).
    # A near, direct path must score lower than a longer one that enters farther out.
    short = _pc([_wp("SCHOO", 36.00, -78.60, alt_ft=3000.0), _wp("RW05L", 35.88, -78.78, alt_ft=400.0)])
    long = _pc([
        _wp("OTTOS", 36.30, -78.40, alt_ft=6000.0), _wp("CHWDR", 36.15, -78.55, alt_ft=5000.0),
        _wp("SCHOO", 36.00, -78.60, alt_ft=3000.0), _wp("RW05L", 35.88, -78.78, alt_ft=400.0),
    ])
    assert path_length_m(short) < path_length_m(long)


def test_path_length_is_horizontal():
    # An IAF often has no coded altitude: the ranking must not read it as 0 ft (the 3D form did).
    coded = _pc([_wp("SCHOO", 36.00, -78.60, alt_ft=3000.0), _wp("RW05L", 35.88, -78.78, alt_ft=400.0)])
    uncoded = _pc([_wp("SCHOO", 36.00, -78.60), _wp("RW05L", 35.88, -78.78, alt_ft=400.0)])
    assert path_length_m(coded) == path_length_m(uncoded)
