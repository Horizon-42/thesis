"""End-to-end build wiring: manifest arrival + aircraft -> FlightScenario."""

import json
from pathlib import Path

import pytest

from aircraft.aircraft_sets import A320
from aircraft.performance_index import PERFORMANCE_INDEX_SCHEMA, performance_index_identity
from flight_scenarios.__main__ import (
    airport_for_manifest,
    discover_arrival_manifests,
    scenario_output_name,
)
from flight_scenarios.build import NoAircraftDynamics, build_scenario, build_scenarios_from_arrivals

# A minimal model-ready arrival record.
# icao24 3949ea is a real Air France transponder address -> resolves to B772 via OpenAP.
FLIGHT = {
    "id": "AFR074",
    "callsign": "AFR074",
    "type": "UNK",
    "icao24": "3949ea",
    "runway": "05L",
    "entry_time_utc": "2026-06-18T10:03:07Z",
    # Real harvest data always declares its datum; the modeling loader converts on it.
    "altitude_source": "opensky_history_geoaltitude_m",
    "runway_target": {
        "lat": 35.74, "lon": -78.47, "elevation_hae_m": 100.0,
        "elevation_msl_m": 133.5, "hae_minus_msl_m": -33.5,
        "course_deg": 50.0, "threshold_crossing_height_m": 15.0,
        "published_glidepath_deg": 3.0,
        "position_source": "faa_cifp_path_point",
        "vertical_source": "faa_cifp_path_point",
    },
    "waypoints": [
        [0.0, -78.45, 35.74, 2500.0],
        [5.0, -78.46, 35.74, 2450.0],
        [10.0, -78.47, 35.74, 2400.0],
    ],
}


def test_build_scenario_resolves_aircraft_from_icao24():
    # No fallback supplied: the aircraft is resolved from the flight's own icao24.
    scen = build_scenario(FLIGHT)
    assert scen.aircraft.code == "B772"
    assert scen.aero.S == scen.aircraft.geometry.wing_area_m2
    assert scen.initial.m == scen.aircraft.landing_mass  # landing weight, not MTOW
    assert scen.source["id"] == "AFR074"
    assert scen.source["runway"] == "05L"
    assert scen.source["entry_time_utc"] == "2026-06-18T10:03:07Z"  # ring-entry carried through
    assert scen.source["n_samples"] == 3
    # both boundary states are populated (initial = track start, target = track end)
    assert scen.initial.longitude == FLIGHT["waypoints"][0][1]
    assert scen.target is not None
    assert scen.target.longitude == FLIGHT["waypoints"][-1][1]


def test_build_scenario_falls_back_to_aircraft_type_when_icao24_unresolvable():
    flight = {**FLIGHT, "icao24": None}  # no transponder address -> use the explicit fallback
    scen = build_scenario(flight, "A320")
    assert scen.aircraft is A320


def test_build_scenario_records_identity_and_dynamics_provenance():
    # Present in the current FAA registry but absent from the stale OpenSky address map.
    # FAA model 737-9GPER standardizes to ICAO B739 and is supported by OpenAP.
    flight = {**FLIGHT, "id": "DAL1450", "callsign": "DAL1450", "icao24": "ad63f7"}

    scen = build_scenario(flight)

    assert scen.aircraft.code == "B739"
    assert scen.source["resolved_typecode"] == "B739"
    assert scen.source["identity_source"] == "faa_registry"
    assert scen.source["typecode_source"] == "faa_registry+opensky_evidence"
    assert scen.source["typecode_standard"] == "ICAO Doc 8643"
    assert scen.source["dynamics_typecode"] == "B739"
    assert scen.source["dynamics_source"].startswith("openap")
    assert scen.source["aircraft_fallback_used"] is False
    assert scen.source["aircraft_fallback_reason"] is None


def test_build_scenario_flies_an_index_type_with_its_own_parameters():
    # BCS3 used to fly as an A320; the performance index gives it its own documents' facts.
    scen = build_scenario({**FLIGHT, "type": "BCS3", "icao24": None})

    assert scen.aircraft.code == "BCS3"
    assert scen.source["resolved_typecode"] == "BCS3"
    assert scen.source["identity_source"] == "declared_type"
    assert scen.source["dynamics_typecode"] == "BCS3"
    assert scen.source["dynamics_source"] == PERFORMANCE_INDEX_SCHEMA
    assert scen.source["dynamics_surrogate_typecode"] is None
    assert scen.source["performance_index_decision"] == "own"
    assert scen.source["aircraft_fallback_used"] is False


def test_build_scenario_flies_a_substitute_under_its_own_code_keeping_the_identity():
    # A306 is an OpenAP synonym of A332; the index flies it AS the A332, so its mass, approach
    # speed and speed gate all belong to one airframe, while the identity stays A306.
    scen = build_scenario({**FLIGHT, "type": "A306", "icao24": None})

    assert scen.aircraft.code == "A332"
    assert scen.source["resolved_typecode"] == "A306"
    assert scen.source["dynamics_typecode"] == "A332"
    assert scen.source["dynamics_source"].startswith("openap-")      # the A332's own OpenAP model
    assert scen.source["dynamics_surrogate_typecode"] == "A332"
    assert scen.source["performance_index_decision"] == "substitute"
    assert scen.source["performance_index_sha256"] == performance_index_identity()["sha256"]
    assert scen.source["aircraft_fallback_used"] is False


def test_an_excluded_type_raises_by_name_unless_a_fallback_is_given():
    flight = {**FLIGHT, "type": "PC12", "icao24": None}
    with pytest.raises(NoAircraftDynamics) as caught:
        build_scenario(flight)
    assert caught.value.typecode == "PC12" and "propeller" in caught.value.reason

    scen = build_scenario(flight, "A320")          # the explicit fallback (ts `all` filter)
    assert scen.aircraft is A320
    assert scen.source["performance_index_decision"] == "exclude"
    assert scen.source["aircraft_fallback_used"] is True
    assert "propeller" in scen.source["aircraft_fallback_reason"]


def test_build_scenario_raises_when_unresolvable_and_no_fallback():
    flight = {**FLIGHT, "icao24": None}
    with pytest.raises(NoAircraftDynamics):
        build_scenario(flight)


def test_the_batch_layer_drops_and_names_a_flight_without_dynamics(monkeypatch):
    import flight_scenarios.dataset as dataset

    flights = [FLIGHT, {**FLIGHT, "id": "PC1", "type": "PC12", "icao24": None}]
    report = dataset.SelectionReport(
        airport="KRDU", target="track-end", max_per_runway=None, available=2, selected=2,
        per_runway={"05L": dataset.RunwaySelection(available=2, selected=2)},
    )
    monkeypatch.setattr(dataset, "select_flight_keys", lambda *a, **k: (["x", "y"], report, "KRDU"))
    monkeypatch.setattr(dataset, "load_model_arrivals_subset", lambda *a, **k: flights)

    scenarios, selection = dataset.build_scenario_dataset("unused", target="track-end")

    assert [s.aircraft.code for s in scenarios] == ["B772"]
    assert selection.selected == 1 and selection.per_runway["05L"].selected == 1
    (dropped,) = selection.excluded_no_dynamics
    assert dropped["typecode"] == "PC12" and "propeller" in dropped["reason"]
    payload = selection.to_dict()
    assert payload["schema_version"] == "flight-scenarios-selection-v2"
    assert payload["performance_index"] == performance_index_identity()
    assert payload["per_runway"]["05L"]["excluded_no_dynamics"] == 1
    assert "1 dropped: no aircraft dynamics" in selection.summary_line()


def test_build_scenario_rejects_mutually_exclusive_targets_before_processing():
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_scenario(
            {"waypoints": []},
            target_from_threshold=True,
            target_from_fitted_adsb=True,
        )


def test_build_scenarios_from_list_resolves_each_flight():
    scens = build_scenarios_from_arrivals([FLIGHT, FLIGHT])
    assert len(scens) == 2
    assert all(s.aircraft.code == "B772" for s in scens)


def test_build_scenario_target_from_threshold():
    flight = {**FLIGHT, "arr_airport": "KRDU", "runway": "05L"}
    scen = build_scenario(flight, target_from_threshold=True)
    assert scen.source["target_source"] == "runway_threshold"
    assert (scen.target.latitude, scen.target.longitude) == (
        FLIGHT["runway_target"]["lat"], FLIGHT["runway_target"]["lon"]
    )
    assert scen.target.V == scen.aircraft.approach.reference_speed_ms(scen.target.m)


def test_build_scenario_target_from_threshold_falls_back_when_unknown():
    flight = {**FLIGHT, "arr_airport": "KRDU", "runway": "99X"}  # no such threshold
    scen = build_scenario(flight, "A320", target_from_threshold=True)
    assert scen.source["target_source"] == "runway_threshold"


# ── CLI discovery (one manifest per airport) ─────────────────────────────────

def _make_manifest_tree(root: Path) -> None:
    for code in ("KRDU", "KMSY"):
        directory = root / code / "arrivals"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps({"schema_version": "harvest-arrivals-v1", "airport": code}),
            encoding="utf-8",
        )


def test_discover_arrival_manifests_single_input(tmp_path):
    _make_manifest_tree(tmp_path)
    manifest = tmp_path / "KRDU" / "arrivals" / "manifest.json"
    assert discover_arrival_manifests(input_path=manifest) == [manifest]
    assert airport_for_manifest(manifest) == "KRDU"


def test_discover_arrival_manifest_for_one_airport(tmp_path):
    _make_manifest_tree(tmp_path)
    found = discover_arrival_manifests(airport="krdu", harvest_root=tmp_path)
    assert found == [tmp_path / "KRDU" / "arrivals" / "manifest.json"]


def test_discover_arrival_manifests_for_all_airports(tmp_path):
    _make_manifest_tree(tmp_path)
    found = discover_arrival_manifests(harvest_root=tmp_path)
    assert found == [
        tmp_path / "KMSY" / "arrivals" / "manifest.json",
        tmp_path / "KRDU" / "arrivals" / "manifest.json",
    ]


def test_scenario_output_name_distinguishes_target_mode():
    assert scenario_output_name("KRDU", threshold=False) == "KRDU_arrivals_scenarios.json"
    assert scenario_output_name(
        "KRDU", threshold=False, fitted_adsb=True
    ) == "KRDU_arrivals_fitted_adsb_scenarios.json"
    assert scenario_output_name("KRDU", threshold=True) == "KRDU_arrivals_threshold_scenarios.json"


def test_resolve_airframe_names_the_type_even_when_openap_has_no_dynamics(monkeypatch):
    """The harvest's observed records need the TYPE (the published speed window's key)
    far more than a modelled mass: an identity OpenAP cannot fly still resolves, with
    no mass; an unresolvable identity returns None."""
    from types import SimpleNamespace

    import flight_scenarios.build as build

    class _Resolver:
        def __init__(self, typecode):
            self.typecode = typecode

        def resolve(self, *, declared_type, icao24):
            return SimpleNamespace(typecode=self.typecode)

    # A type the index flies with its own parameters carries its own landing mass ...
    monkeypatch.setattr(build, "get_default_identity_resolver", lambda: _Resolver("E55P"))
    assert build.resolve_airframe("abc123") == (7568.0, "E55P")
    # ... a substituted or excluded type keeps its type but no mass (a substitute's mass
    # is not the observed type's).
    for typecode in ("GLF4", "PC12"):
        monkeypatch.setattr(build, "get_default_identity_resolver", lambda t=typecode: _Resolver(t))
        assert build.resolve_airframe("abc123") == (None, typecode)

    monkeypatch.setattr(build, "get_default_identity_resolver", lambda: _Resolver("A320"))
    mass, typecode = build.resolve_airframe("abc123")
    assert typecode == "A320" and mass == A320.landing_mass

    monkeypatch.setattr(build, "get_default_identity_resolver", lambda: _Resolver(None))
    assert build.resolve_airframe("abc123") is None
