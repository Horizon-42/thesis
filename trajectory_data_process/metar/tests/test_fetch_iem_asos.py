"""The IEM ASOS fetcher: station mapping, the landing span, the request, the files."""

from __future__ import annotations

import io
import json
from datetime import date

import pytest

from trajectory_data_process.metar import fetch_iem_asos as fetcher


def test_the_iem_station_is_the_faa_identifier():
    assert fetcher.station_for("KRDU") == "RDU"
    assert fetcher.station_for("ksjc") == "SJC"
    with pytest.raises(ValueError, match="K-prefixed"):
        fetcher.station_for("CYYC")


def test_the_landing_span_covers_only_assigned_landings(tmp_path):
    manifest = tmp_path / "KAAA" / "tracks" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"records": [
        {"outcome": "assigned", "landing_time_utc": "2026-05-03T10:00:00Z"},
        {"outcome": "not_landing", "landing_time_utc": "2026-04-01T00:00:00Z"},
        {"outcome": "assigned", "landing_time_utc": "2026-06-30T23:59:00+00:00"},
    ]}), encoding="utf-8")
    assert fetcher.landing_span("KAAA", tmp_path) == (date(2026, 5, 3), date(2026, 6, 30))
    manifest.write_text(json.dumps({"records": [{"outcome": "not_landing"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="no assigned landings"):
        fetcher.landing_span("KAAA", tmp_path)


def test_the_request_names_the_wind_columns_utc_and_an_exclusive_end():
    url = fetcher.request_url("RDU", date(2026, 7, 15), date(2026, 7, 16))
    assert url.startswith(fetcher.IEM_ASOS_URL + "?")
    for fragment in (
        "station=RDU", "data=drct", "data=sknt", "data=gust", "tz=Etc%2FUTC",
        "format=onlycomma", "year1=2026", "month1=7", "day1=15", "year2=2026",
        "month2=7", "day2=16", "report_type=3", "report_type=4",
    ):
        assert fragment in url


def test_fetch_writes_the_archive_csv_and_its_provenance(tmp_path, monkeypatch):
    body = "station,valid,drct,sknt,gust\nRDU,2026-07-15 00:51,100.00,3.00,M\n"
    seen: list[str] = []

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(url, timeout):
        seen.append(url)
        return _Response(body.encode("utf-8"))

    monkeypatch.setattr(fetcher.urllib.request, "urlopen", fake_urlopen)
    path = fetcher.fetch("KRDU", date(2026, 7, 15), date(2026, 7, 15), out_root=tmp_path)
    assert path == tmp_path / "KRDU" / "asos_2026-07-15_2026-07-15.csv"
    assert path.read_text(encoding="utf-8") == body
    # The archive's end day is exclusive: one inclusive day asks for day1=15, day2=16.
    assert "day1=15" in seen[0] and "day2=16" in seen[0]
    provenance = json.loads(path.with_suffix("").with_suffix(".provenance.json").read_text())
    assert provenance["rows"] == 1 and provenance["station"] == "RDU"
    assert provenance["url"] == seen[0] and provenance["units"]["drct"] == "deg true"


def test_fetch_refuses_a_response_that_is_not_the_archive(tmp_path, monkeypatch):
    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        fetcher.urllib.request, "urlopen",
        lambda url, timeout: _Response(b"<html>maintenance</html>\n"),
    )
    with pytest.raises(ValueError, match="unexpected response header"):
        fetcher.fetch("KRDU", date(2026, 7, 15), date(2026, 7, 15), out_root=tmp_path)
    assert not (tmp_path / "KRDU").exists()


def test_start_and_end_go_together(capsys):
    with pytest.raises(SystemExit):
        fetcher.main(["--airport", "KRDU", "--start", "2026-07-15"])
    assert "--start and --end go together" in capsys.readouterr().err
