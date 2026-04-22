from preprocess_obstacles import (
    resolve_airport_config_path,
    resolve_obstacle_center,
    resolve_obstacle_output_path,
)
from preprocess_waypoints import resolve_waypoint_center, resolve_waypoint_output_path


def test_resolve_waypoint_center_defaults_to_airport_record(tmp_path):
    csv_path = tmp_path / "airports.csv"
    csv_path.write_text(
        "\n".join([
            '"id","ident","name","latitude_deg","longitude_deg","elevation_ft","gps_code","icao_code"',
            '1,"CYLW","Kelowna International Airport",49.9561,-119.377998,1421,"CYLW","CYLW"',
        ]),
        encoding="utf-8",
    )

    center_lon, center_lat = resolve_waypoint_center("CYLW", csv_path, None, None)

    assert center_lon == -119.377998
    assert center_lat == 49.9561


def test_resolve_waypoint_output_path_defaults_to_airport_folder():
    output_path = resolve_waypoint_output_path("cylw", None)
    assert output_path.as_posix().endswith("/public/data/airports/CYLW/waypoints.geojson")


def test_resolve_airport_config_path_defaults_to_airport_folder():
    airport_path = resolve_airport_config_path("krdu")
    assert airport_path.as_posix().endswith("/public/data/airports/KRDU/airport.json")


def test_resolve_obstacle_center_reads_airport_json(tmp_path):
    airport_json = tmp_path / "airport.json"
    airport_json.write_text(
        '{"code":"KRDU","lon":-78.7873,"lat":35.878659,"height":15000}',
        encoding="utf-8",
    )

    center_lon, center_lat = resolve_obstacle_center(None, None, airport_json)

    assert center_lon == -78.7873
    assert center_lat == 35.878659


def test_resolve_obstacle_output_path_defaults_to_airport_folder():
    output_path = resolve_obstacle_output_path("krdu", None)
    assert output_path.as_posix().endswith("/public/data/airports/KRDU/obstacles.geojson")


def test_resolve_obstacle_center_requires_lon_and_lat_together(tmp_path):
    airport_json = tmp_path / "airport.json"
    airport_json.write_text(
        '{"code":"KRDU","lon":-78.7873,"lat":35.878659,"height":15000}',
        encoding="utf-8",
    )

    try:
        resolve_obstacle_center(-78.7, None, airport_json)
    except ValueError as error:
        assert "Provide both --center-lon and --center-lat" in str(error)
    else:
        raise AssertionError("Expected resolve_obstacle_center to reject partial explicit center")
