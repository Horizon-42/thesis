import json

from preprocess_usgs_tnm_terrain import (
    DEFAULT_LAZ_SOURCE_SRS_BY_AIRPORT,
    GeoBBox,
    StagedTerrainSource,
    build_laz_to_dsm_pipeline,
    default_target_srs_for_bbox,
    ensure_readable_source_files,
    pdal_bounds_string,
    read_manifest_bboxes,
    read_manifest_download_urls,
    resolve_processing_bbox,
    select_highest_precision_source,
    source_kinds_to_stage,
    staging_dir_for_source,
    transform_bbox_with_gdaltransform,
    transformation_matrix_for_z_scale,
    union_bboxes,
)


def test_read_manifest_bboxes_uses_airport_group_and_product_bbox(tmp_path):
    manifest_path = tmp_path / "download_manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "status,message,group,label,product,product_bbox,query_bbox",
                'downloaded,,KRDU,RDU,dsm,"-78.8,35.8,-78.7,35.9","0,0,1,1"',
                'downloaded,,KSJC,SJC,dsm,"-122.0,37.0,-121.9,37.1","0,0,1,1"',
                'downloaded,,krdu,RDU,dsm,"-78.9,35.7,-78.75,35.95","0,0,1,1"',
            ]
        ),
        encoding="utf-8",
    )

    bbox = union_bboxes(read_manifest_bboxes(manifest_path, "KRDU"))

    assert bbox == GeoBBox(west=-78.9, south=35.7, east=-78.7, north=35.95)


def test_read_manifest_download_urls_uses_airport_product_and_target(tmp_path):
    target = tmp_path / "KSJC" / "dem" / "tile.tif"
    other_target = tmp_path / "KRDU" / "dem" / "tile.tif"
    manifest_path = tmp_path / "download_manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "status,message,group,label,product,url,target",
                f"downloaded,,KSJC,SJC,dem,https://example.test/ksjc.tif,{target}",
                f"downloaded,,KRDU,RDU,dem,https://example.test/krdu.tif,{other_target}",
                f"downloaded,,KSJC,SJC,dsm,https://example.test/ksjc.laz,{tmp_path / 'surface.laz'}",
            ]
        ),
        encoding="utf-8",
    )

    urls = read_manifest_download_urls(manifest_path, "ksjc", product="dem")

    assert urls == {target.resolve(): "https://example.test/ksjc.tif"}


def test_ensure_readable_source_files_redownloads_manifest_repair(monkeypatch, tmp_path):
    source = tmp_path / "KSJC" / "dem" / "tile.tif"
    source.parent.mkdir(parents=True)
    source.write_text("broken", encoding="utf-8")
    manifest_path = tmp_path / "download_manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "status,message,group,label,product,url,target",
                f"downloaded,,KSJC,SJC,dem,https://example.test/tile.tif,{source}",
            ]
        ),
        encoding="utf-8",
    )
    states = iter(["bad tiff", None])
    downloads = []

    monkeypatch.setattr("preprocess_usgs_tnm_terrain.geotiff_read_error", lambda path: next(states))

    def fake_download(url, target):
        downloads.append((url, target))
        target.write_text("repaired", encoding="utf-8")

    monkeypatch.setattr("preprocess_usgs_tnm_terrain.download_file", fake_download)

    assert ensure_readable_source_files(
        airport_code="KSJC",
        usgs_root=tmp_path,
        product="dem",
        source_paths=[source],
        dry_run=False,
    ) == [source]
    assert downloads == [("https://example.test/tile.tif", source)]
    assert source.read_text(encoding="utf-8") == "repaired"


def test_resolve_processing_bbox_prefers_explicit_bbox(tmp_path):
    explicit_bbox = GeoBBox(west=-2, south=3, east=-1, north=4)

    bbox = resolve_processing_bbox(
        airport_code="KRDU",
        explicit_bbox=explicit_bbox,
        usgs_root=tmp_path,
        fallback_radius_km=20,
    )

    assert bbox == explicit_bbox


def test_resolve_processing_bbox_uses_manifest_union(tmp_path):
    manifest_path = tmp_path / "download_manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "status,message,group,label,product,product_bbox,query_bbox",
                'downloaded,,KRDU,RDU,dsm,"-78.8,35.8,-78.7,35.9","0,0,1,1"',
                'downloaded,,KRDU,RDU,dsm,"-78.9,35.75,-78.72,35.95","0,0,1,1"',
            ]
        ),
        encoding="utf-8",
    )

    bbox = resolve_processing_bbox(
        airport_code="KRDU",
        explicit_bbox=None,
        usgs_root=tmp_path,
        fallback_radius_km=20,
    )

    assert bbox == GeoBBox(west=-78.9, south=35.75, east=-78.7, north=35.95)


def test_default_target_srs_uses_nad83_utm_zone_for_krdu_bbox():
    bbox = GeoBBox(west=-78.875, south=35.812, east=-78.687, north=35.938)

    assert default_target_srs_for_bbox(bbox) == "EPSG:26917"


def test_transform_bbox_with_gdaltransform_parses_transformed_corner_bounds(monkeypatch):
    calls = []

    class Result:
        stdout = "10 40 0\n10 50 0\n20 30 0\n20 60 0\n"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Result()

    monkeypatch.setattr("preprocess_usgs_tnm_terrain.shutil.which", lambda name: "/usr/bin/gdaltransform")
    monkeypatch.setattr("preprocess_usgs_tnm_terrain.subprocess.run", fake_run)

    bounds = transform_bbox_with_gdaltransform(
        GeoBBox(west=-78.875, south=35.812, east=-78.687, north=35.938),
        "EPSG:26917",
    )

    assert bounds == (10.0, 30.0, 20.0, 60.0)
    assert calls[0][0] == ["gdaltransform", "-s_srs", "EPSG:4326", "-t_srs", "EPSG:26917"]
    assert "-78.8750000000 35.8120000000" in calls[0][1]["input"]


def test_krdu_laz_default_source_srs_matches_inspected_stateplane_data():
    assert DEFAULT_LAZ_SOURCE_SRS_BY_AIRPORT["KRDU"] == "EPSG:2264"


def test_build_laz_to_dsm_pipeline_merges_reprojects_scales_and_writes_max(tmp_path):
    output_tif = tmp_path / "dsm.tif"

    pipeline = build_laz_to_dsm_pipeline(
        laz_paths=[tmp_path / "a.laz", tmp_path / "b.laz"],
        output_tif=output_tif,
        source_srs="EPSG:2264",
        target_srs="EPSG:26917",
        target_bounds=(690000.0, 3965000.0, 707000.0, 3980000.0),
        resolution_m=2.0,
        vertical_scale=0.3048,
    )

    assert [stage["type"] for stage in pipeline] == [
        "readers.las",
        "readers.las",
        "filters.merge",
        "filters.reprojection",
        "filters.transformation",
        "writers.gdal",
    ]
    assert pipeline[0]["override_srs"] == "EPSG:2264"
    assert pipeline[3]["out_srs"] == "EPSG:26917"
    assert pipeline[4]["matrix"] == transformation_matrix_for_z_scale(0.3048)
    assert pipeline[5] == {
        "type": "writers.gdal",
        "filename": str(output_tif),
        "resolution": 2.0,
        "output_type": "max",
        "data_type": "float32",
        "nodata": -999999.0,
        "bounds": pdal_bounds_string((690000.0, 3965000.0, 707000.0, 3980000.0)),
        "gdalopts": "COMPRESS=LZW,TILED=YES,BIGTIFF=IF_SAFER",
    }


def test_pipeline_json_is_serializable(tmp_path):
    pipeline = build_laz_to_dsm_pipeline(
        laz_paths=[tmp_path / "a.laz"],
        output_tif=tmp_path / "dsm.tif",
        source_srs="EPSG:2264",
        target_srs="EPSG:26917",
        target_bounds=(1.0, 2.0, 3.0, 4.0),
        resolution_m=2.0,
        vertical_scale=0.3048,
    )

    encoded = json.dumps({"pipeline": pipeline})

    assert "writers.gdal" in encoded


def test_source_kinds_to_stage_supports_both():
    assert source_kinds_to_stage("dem") == ["dem"]
    assert source_kinds_to_stage("dsm") == ["dsm"]
    assert source_kinds_to_stage("both") == ["dem", "dsm"]


def test_source_kinds_to_stage_auto_uses_available_source_shapes(tmp_path):
    airport_root = tmp_path / "KRDU"
    (airport_root / "dem").mkdir(parents=True)
    (airport_root / "dem" / "dem.tif").write_text("placeholder", encoding="utf-8")
    (airport_root / "dsm" / "source_laz").mkdir(parents=True)
    (airport_root / "dsm" / "source_laz" / "surface.laz").write_text(
        "placeholder",
        encoding="utf-8",
    )

    assert source_kinds_to_stage("auto", airport_code="KRDU", usgs_root=tmp_path) == [
        "dem",
        "dsm",
    ]


def test_source_kinds_to_stage_auto_accepts_tnm_dem_img(tmp_path):
    airport_root = tmp_path / "KRDU"
    (airport_root / "dem").mkdir(parents=True)
    (airport_root / "dem" / "dem.img").write_text("placeholder", encoding="utf-8")

    assert source_kinds_to_stage("auto", airport_code="KRDU", usgs_root=tmp_path) == [
        "dem",
    ]


def test_select_highest_precision_source_uses_resolution_not_source_kind(tmp_path):
    selected = select_highest_precision_source(
        {
            "dem": StagedTerrainSource(
                source_kind="dem",
                source_dir=tmp_path / "dem",
                output_tif=tmp_path / "dem.tif",
                horizontal_resolution_m=10.0,
            ),
            "dsm": StagedTerrainSource(
                source_kind="dsm",
                source_dir=tmp_path / "dsm",
                output_tif=tmp_path / "dsm.tif",
                horizontal_resolution_m=2.0,
            ),
        }
    )

    assert selected == "dsm"


def test_staging_dir_defaults_under_airport_local_terrain_sources():
    path = staging_dir_for_source("krdu", "dem")

    assert path.as_posix().endswith(
        "/public/data/airports/KRDU/local-terrain/sources/usgs-tnm-dem"
    )
