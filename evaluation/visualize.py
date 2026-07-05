"""Render one batch's evaluation into an HTML report (tables + Plotly charts).

    python -m evaluation.visualize --input <batch dir> --output evaluation_report.html

Recomputes the evaluation from the record files (same gates / flags as
``python -m evaluation``, so the HTML can never go stale against a previously
written report JSON), embeds everything as one ``const DATA`` blob, and renders:

  * summary cards + aggregate tables (solve/success rates, deviation spreads,
    flight times, observed-track comparison)
  * charts — per-flight final lateral deviation vs the gate (log scale),
    final vertical deviation vs the WCH window, optimized-vs-observed flight
    times + Δtime distribution, per-flight path-shape deviation
  * per-flight track overlay (plan view + altitude profile, optimized vs
    observed, arc-length resampled) behind a flight selector
  * the full per-trajectory verdict table

Plotly comes from its CDN (the same convention as the project's other
interactive docs, e.g. ``4dTrajectory/docs/dynamics_comparison_30km.zh.html``);
everything else is embedded. Track overlays are capped at ``--max-tracks``
(evenly sampled) to keep the file small — the cap is stated in the page, never
silent.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence

from evaluation.metrics import evaluate_batch
from evaluation.records import TrajectoryRecord, load_records
from evaluation.reference import (
    horizontal_arc_length_m,
    load_reference,
    resample_by_arc_length,
)
from evaluation.thresholds import DeviationThresholds

DEFAULT_MAX_TRACKS = 30
RESAMPLE_N = 101


def build_payload(
    records: Sequence[TrajectoryRecord],
    thresholds: DeviationThresholds = DeviationThresholds(),
    *,
    max_tracks: int = DEFAULT_MAX_TRACKS,
) -> dict[str, Any]:
    """The HTML page's data blob (embedded as ``const DATA``), JSON-ready:

        {"report": the metrics.evaluate_batch report dict (schema documented there),
         "tracks": [_track_entry() result per selected solved record],
         "tracksShown": int,    # len(tracks)
         "tracksTotal": int}    # all solved records; > tracksShown ⇒ capped/sampled
    """
    report = evaluate_batch(records, thresholds)
    # Only arc-length-drawable paths: a zero-horizontal-extent record (e.g. a rollout
    # truncated at its first step) has no path to overlay and would crash the resampler.
    drawable = [r for r in records if r.solved and horizontal_arc_length_m(r.states) > 0.0]
    selected = _sample_evenly(drawable, max_tracks)
    tracks = [_track_entry(record) for record in selected]
    return {
        "report": report,
        "tracks": tracks,
        "tracksShown": len(selected),
        "tracksTotal": len(drawable),
    }


def render_html(payload: dict[str, Any], *, title: str, source_label: str) -> str:
    """Bake a :func:`build_payload` dict into the page template.

    The JSON blob is embedded inside a ``<script>`` element, where a literal
    ``</`` in any string (a flight id, a solver failure message) would terminate
    the script early — escape it as ``<\\/`` (identical after JS string parsing).
    """
    return _TEMPLATE.replace("__TITLE__", html.escape(title)) \
        .replace("__SOURCE__", html.escape(source_label)) \
        .replace("__DATA__", json.dumps(payload).replace("</", "<\\/"))


def main(argv: list[str] | None = None) -> None:
    defaults = DeviationThresholds()
    parser = argparse.ArgumentParser(
        description="Render an evaluation batch as an HTML report (tables + charts)."
    )
    parser.add_argument("--input", required=True,
                        help="one record JSON, or a directory of them")
    parser.add_argument("--pattern", default="*_eval.json",
                        help="filename glob when --input is a directory")
    parser.add_argument("--output", default="evaluation_report.html",
                        help="where to write the HTML report")
    parser.add_argument("--title", default="Trajectory Evaluation Report")
    parser.add_argument("--max-tracks", type=int, default=DEFAULT_MAX_TRACKS,
                        help="track overlays to embed (evenly sampled over the solved records)")
    parser.add_argument("--lateral-max-m", type=float, default=defaults.lateral_max_m)
    parser.add_argument("--vertical-below-max-m", type=float,
                        default=defaults.vertical_below_max_m)
    parser.add_argument("--vertical-above-max-m", type=float,
                        default=defaults.vertical_above_max_m)
    args = parser.parse_args(argv)

    thresholds = DeviationThresholds(
        lateral_max_m=args.lateral_max_m,
        vertical_below_max_m=args.vertical_below_max_m,
        vertical_above_max_m=args.vertical_above_max_m,
    )
    records = load_records(args.input, pattern=args.pattern)
    payload = build_payload(records, thresholds, max_tracks=args.max_tracks)
    out = Path(args.output)
    out.write_text(
        render_html(payload, title=args.title, source_label=str(args.input)),
        encoding="utf-8",
    )
    print(f"✓ evaluation HTML ({payload['report']['total']} trajectories, "
          f"{payload['tracksShown']}/{payload['tracksTotal']} track overlays) -> {out}")


def _sample_evenly(items: Sequence[TrajectoryRecord], count: int) -> list[TrajectoryRecord]:
    if count <= 0 or len(items) <= count:
        return list(items)
    step = (len(items) - 1) / (count - 1) if count > 1 else 0.0
    indices = sorted({round(i * step) for i in range(count)})
    return [items[i] for i in indices]


def _track_entry(record: TrajectoryRecord) -> dict[str, Any]:
    """One flight's overlay: resampled optimized path (+ reference when pointed at).

        {"id": str,
         "optimized": _polyline() result,
         "target": {"lat", "lon", "alt"},   # degrees / degrees / metres
         "reference": _polyline() result}   # only when the record has a reference_file
    """
    entry: dict[str, Any] = {
        "id": str(record.source.get("id") or (record.path.stem if record.path else "trajectory")),
        "optimized": _polyline(resample_by_arc_length(record.states, RESAMPLE_N)),
        "target": {
            "lat": record.target_state["lat"],
            "lon": record.target_state["lon"],
            "alt": record.target_state["alt"],
        },
    }
    if record.reference_file is not None:
        reference = load_reference(record)
        # A degenerate observed track (zero horizontal extent) cannot be resampled —
        # draw the optimized path alone rather than crash the page build.
        if horizontal_arc_length_m(reference.states) > 0.0:
            entry["reference"] = _polyline(resample_by_arc_length(reference.states, RESAMPLE_N))
    return entry


def _polyline(points: list[tuple[float, float, float]]) -> dict[str, list[float]]:
    """``(lat, lon, alt)`` tuples (resample_by_arc_length's output) → parallel
    plot arrays ``{"lat": […], "lon": […], "alt": […]}`` of equal length."""
    return {
        "lat": [p[0] for p in points],
        "lon": [p[1] for p in points],
        "alt": [p[2] for p in points],
    }


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js" charset="utf-8"></script>
<style>
  body { font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
         margin: 0; background: #f5f6f8; color: #1c2733; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 24px 20px 60px; }
  h1 { font-size: 22px; margin: 8px 0 2px; }
  h2 { font-size: 16px; margin: 34px 0 10px; border-left: 4px solid #2f6fed; padding-left: 8px; }
  .meta { color: #667; font-size: 12px; margin-bottom: 18px; }
  .cards { display: flex; flex-wrap: wrap; gap: 12px; }
  .card { background: #fff; border-radius: 10px; padding: 12px 18px; min-width: 130px;
          box-shadow: 0 1px 3px rgba(20,40,80,.08); }
  .card .num { font-size: 22px; font-weight: 600; }
  .card .lbl { font-size: 12px; color: #667; margin-top: 2px; }
  .ok   .num { color: #178a4c; }
  .bad  .num { color: #c43d3d; }
  table { border-collapse: collapse; width: 100%; background: #fff; font-size: 12.5px;
          box-shadow: 0 1px 3px rgba(20,40,80,.08); border-radius: 8px; overflow: hidden; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #eef0f3; text-align: right;
           white-space: nowrap; }
  th { background: #eef2f9; color: #345; position: sticky; top: 0; }
  td:first-child, th:first-child { text-align: left; }
  tr.fail td { background: #fdf3f3; }
  tr.unsolved td { background: #f3f4f6; color: #889; }
  .scroll { max-height: 480px; overflow: auto; border-radius: 8px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 900px) { .grid2 { grid-template-columns: 1fr; } }
  .chart { background: #fff; border-radius: 10px; box-shadow: 0 1px 3px rgba(20,40,80,.08);
           padding: 6px; }
  select { font-size: 13px; padding: 4px 8px; margin: 0 0 10px; }
  .note { font-size: 12px; color: #778; margin: 6px 0; }
  .viol { text-align: left; white-space: normal; max-width: 340px; color: #a33; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="meta">input: <code>__SOURCE__</code></div>

  <div class="cards" id="cards"></div>

  <h2>Aggregates</h2>
  <div id="aggTable"></div>
  <div class="note">
    Gate sources (FAA Order 8260.58D): lateral — the LPV (Localizer Performance with
    Vertical guidance) lateral-guidance semiwidth floor at the runway threshold
    (Formula 3-1-1, 350 ft); vertical — the WCH (Wheel Crossing Height) window around
    the published TCH (Threshold Crossing Height) (§1-3-2, 20–50 ft about the 30 ft design value).
  </div>

  <h2>Final-state deviations vs regulation gates</h2>
  <div class="grid2">
    <div class="chart" id="chartLateral"></div>
    <div class="chart" id="chartVertical"></div>
  </div>

  <div id="refSection" style="display:none">
    <h2>Comparison vs observed ADS-B (Automatic Dependent Surveillance–Broadcast) tracks</h2>
    <div class="grid2">
      <div class="chart" id="chartTime"></div>
      <div class="chart" id="chartDtime"></div>
    </div>
    <div class="grid2" style="margin-top:16px">
      <div class="chart" id="chartPathDev"></div>
      <div class="chart" style="visibility:hidden"></div>
    </div>
  </div>

  <h2>Track overlay</h2>
  <div class="note" id="trackNote"></div>
  <select id="flightSelect"></select>
  <div class="grid2">
    <div class="chart" id="chartPlan"></div>
    <div class="chart" id="chartProfile"></div>
  </div>

  <h2>Per-trajectory verdicts</h2>
  <div class="scroll"><table id="rowTable"></table></div>
</div>

<script>
const DATA = __DATA__;
const R = DATA.report, ROWS = R.trajectories, TH = R.thresholds;
const solvedRows = ROWS.filter(r => r.solved);
const refRows = solvedRows.filter(r => r.reference && r.reference.flight_time_delta_s !== undefined);
const GREEN = "#178a4c", RED = "#c43d3d", GRAY = "#98a2ad", BLUE = "#2f6fed", DARK = "#48505c";
const fmt = (v, d=1) => (v === null || v === undefined) ? "—" : Number(v).toFixed(d);
const pct = v => (100 * v).toFixed(1) + "%";
// Escape data-derived strings (flight ids, solver messages) before any HTML sink.
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
// Chart label: source ids can recur (same callsign lands twice) and would merge
// Plotly categories — the record FILE name base is unique per trajectory.
const label = r => (r.file || r.id).replace(/_eval\\.json$/, "");
const LAYOUT = { margin: {l: 60, r: 16, t: 42, b: 46}, height: 330,
                 paper_bgcolor: "#fff", plot_bgcolor: "#fff", font: {size: 11.5} };

// ── summary cards ─────────────────────────────────────────────────────────────
function card(num, lbl, cls) {
  return `<div class="card ${cls || ""}"><div class="num">${num}</div><div class="lbl">${lbl}</div></div>`;
}
document.getElementById("cards").innerHTML =
  card(R.total, "total trajectories") +
  card(`${R.solved}/${R.total}`, `solve rate ${pct(R.solve_rate)}`,
       R.solve_rate >= 0.9 ? "ok" : "bad") +
  card(`${R.successful}/${R.total}`, `success rate ${pct(R.success_rate)}`,
       R.success_rate >= 0.9 ? "ok" : "bad") +
  (R.success_rate_among_solved !== null
     ? card(pct(R.success_rate_among_solved), "success among solved") : "") +
  (R.final_time_s ? card(fmt(R.final_time_s.mean) + " s", "mean flight time") : "") +
  (R.reference ? card(fmt(R.reference.flight_time_delta_s.mean, 1) + " s",
                      "mean Δt vs observed (optimized − flown)") : "");

// ── aggregate table ───────────────────────────────────────────────────────────
function aggTable() {
  const rows = [];
  rows.push(["gates", `lateral ≤ ${TH.lateral_max_m} m · vertical −${TH.vertical_below_max_m}/+${TH.vertical_above_max_m} m`, "", ""]);
  if (R.lateral_m) rows.push(["final lateral deviation (m)", fmt(R.lateral_m.mean), fmt(R.lateral_m.p95), fmt(R.lateral_m.max)]);
  if (R.vertical_m) rows.push(["final vertical |deviation| (m)", fmt(R.vertical_m.mean_abs), fmt(R.vertical_m.p95_abs), fmt(R.vertical_m.max_abs)]);
  if (R.final_time_s) rows.push(["flight time (s)", fmt(R.final_time_s.mean), fmt(R.final_time_s.min) + " (min)", fmt(R.final_time_s.max) + " (max)"]);
  if (R.reference) {
    const d = R.reference.flight_time_delta_s;
    rows.push([`Δt vs observed (s, ${R.reference.compared} flights)`, fmt(d.mean), fmt(d.min) + " (min)", fmt(d.max) + " (max)"]);
    rows.push(["path-shape deviation, lateral (m)", fmt(R.reference.path_lateral_m.mean), "", fmt(R.reference.path_lateral_m.max) + " (max)"]);
    rows.push(["path-shape deviation, |vertical| (m)", fmt(R.reference.path_vertical_m.mean_abs), "", fmt(R.reference.path_vertical_m.max_abs) + " (max)"]);
  }
  document.getElementById("aggTable").innerHTML =
    "<table><tr><th>metric</th><th>mean</th><th>p95 / min</th><th>max</th></tr>" +
    rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join("") +
    "</table>";
}
aggTable();

// ── deviation charts ──────────────────────────────────────────────────────────
function lateralChart() {
  if (!solvedRows.length) return;
  const colors = solvedRows.map(r => r.success ? GREEN : RED);
  Plotly.newPlot("chartLateral", [{
    type: "bar", x: solvedRows.map(label),
    y: solvedRows.map(r => Math.max(r.lateral_m, 0.01)),
    marker: {color: colors},
    hovertext: solvedRows.map(r => `${esc(label(r))}: ${fmt(r.lateral_m, 2)} m`), hoverinfo: "text",
  }], {...LAYOUT, title: "Final lateral deviation (log scale; red dashed = gate)",
       yaxis: {type: "log", title: "m"}, xaxis: {tickangle: -40},
       shapes: [{type: "line", xref: "paper", x0: 0, x1: 1,
                 y0: TH.lateral_max_m, y1: TH.lateral_max_m,
                 line: {color: RED, dash: "dash", width: 1.5}}]},
    {displayModeBar: false});
}
function verticalChart() {
  if (!solvedRows.length) return;
  const ys = solvedRows.map(r => r.vertical_m);
  Plotly.newPlot("chartVertical", [{
    type: "bar", x: solvedRows.map(label), y: ys,
    marker: {color: solvedRows.map(r => r.success ? GREEN : RED)},
    hovertext: solvedRows.map(r => `${esc(label(r))}: ${fmt(r.vertical_m, 2)} m`), hoverinfo: "text",
  }], {...LAYOUT, title: "Final vertical deviation (green band = WCH window)",
       yaxis: {title: "m"}, xaxis: {tickangle: -40},
       shapes: [{type: "rect", xref: "paper", x0: 0, x1: 1,
                 y0: -TH.vertical_below_max_m, y1: TH.vertical_above_max_m,
                 fillcolor: "rgba(23,138,76,.12)", line: {width: 0}}]},
    {displayModeBar: false});
}
lateralChart(); verticalChart();

// ── reference charts ──────────────────────────────────────────────────────────
if (refRows.length) {
  document.getElementById("refSection").style.display = "";
  const obs = refRows.map(r => r.reference.flight_time_s);
  const opt = refRows.map(r => r.final_time_s);
  const lim = [Math.min(...obs, ...opt) * 0.95, Math.max(...obs, ...opt) * 1.05];
  Plotly.newPlot("chartTime", [{
    type: "scatter", mode: "markers", x: obs, y: opt,
    marker: {color: refRows.map(r => r.success ? GREEN : RED), size: 8},
    hovertext: refRows.map(r => `${esc(label(r))}: observed ${fmt(r.reference.flight_time_s)} s → optimized ${fmt(r.final_time_s)} s`),
    hoverinfo: "text",
  }], {...LAYOUT, title: "Flight time: optimized vs observed (diagonal = equal)",
       xaxis: {title: "observed (s)", range: lim},
       yaxis: {title: "optimized (s)", range: lim},
       shapes: [{type: "line", x0: lim[0], y0: lim[0], x1: lim[1], y1: lim[1],
                 line: {color: GRAY, dash: "dot", width: 1}}]},
    {displayModeBar: false});
  Plotly.newPlot("chartDtime", [{
    type: "histogram", x: refRows.map(r => r.reference.flight_time_delta_s),
    marker: {color: BLUE},
  }], {...LAYOUT, title: "Δ flight-time distribution (optimized − observed; negative = faster)",
       xaxis: {title: "s"}, yaxis: {title: "count"}},
    {displayModeBar: false});
  Plotly.newPlot("chartPathDev", [{
    type: "bar", x: refRows.map(label),
    y: refRows.map(r => r.reference.path_lateral_m.mean),
    marker: {color: BLUE},
    hovertext: refRows.map(r => `${esc(label(r))}: mean ${fmt(r.reference.path_lateral_m.mean)} m / max ${fmt(r.reference.path_lateral_m.max)} m`),
    hoverinfo: "text",
  }], {...LAYOUT, title: "Path-shape deviation (mean lateral offset, arc-matched)",
       yaxis: {title: "m"}, xaxis: {tickangle: -40}},
    {displayModeBar: false});
}

// ── track overlay ─────────────────────────────────────────────────────────────
const note = document.getElementById("trackNote");
note.textContent = DATA.tracksTotal > DATA.tracksShown
  ? `showing ${DATA.tracksShown} of ${DATA.tracksTotal} solved flights (evenly sampled; adjust with --max-tracks)`
  : `${DATA.tracksShown} solved flight(s)`;
const select = document.getElementById("flightSelect");
DATA.tracks.forEach((t, i) => {
  const option = document.createElement("option");
  option.value = i; option.textContent = t.id;
  select.appendChild(option);
});
function drawTrack(i) {
  const t = DATA.tracks[i];
  if (!t) return;
  const planTraces = [];
  const profTraces = [];
  const frac = t.optimized.lat.map((_, k) => k / (t.optimized.lat.length - 1));
  if (t.reference) {
    planTraces.push({type: "scatter", mode: "lines", name: "observed (ADS-B)",
                     x: t.reference.lon, y: t.reference.lat, line: {color: DARK, width: 2}});
    profTraces.push({type: "scatter", mode: "lines", name: "observed (ADS-B)",
                     x: frac, y: t.reference.alt, line: {color: DARK, width: 2}});
  }
  planTraces.push({type: "scatter", mode: "lines", name: "optimized",
                   x: t.optimized.lon, y: t.optimized.lat, line: {color: BLUE, width: 2}});
  profTraces.push({type: "scatter", mode: "lines", name: "optimized",
                   x: frac, y: t.optimized.alt, line: {color: BLUE, width: 2}});
  planTraces.push({type: "scatter", mode: "markers", name: "target",
                   x: [t.target.lon], y: [t.target.lat],
                   marker: {symbol: "star", size: 13, color: RED}});
  Plotly.react("chartPlan", planTraces,
    {...LAYOUT, title: `Plan view · ${esc(t.id)}`, xaxis: {title: "lon (°)"},
     yaxis: {title: "lat (°)", scaleanchor: "x"}, showlegend: true,
     legend: {orientation: "h", y: -0.25}},
    {displayModeBar: false});
  Plotly.react("chartProfile", profTraces,
    {...LAYOUT, title: "Altitude profile (normalized by each path's arc length)",
     xaxis: {title: "arc fraction"},
     yaxis: {title: "alt (m)"}, showlegend: true, legend: {orientation: "h", y: -0.25}},
    {displayModeBar: false});
}
select.addEventListener("change", () => drawTrack(Number(select.value)));
if (DATA.tracks.length) drawTrack(0);
else note.textContent = "No solved trajectories to display.";

// ── per-trajectory table ──────────────────────────────────────────────────────
function rowTable() {
  const table = document.getElementById("rowTable");
  const head = ["flight", "solved", "success", "lateral (m)", "vertical (m)",
                "ΔV (m/s)", "T (s)", "Δt vs observed (s)", "path dev mean (m)", "notes"];
  const header = "<tr>" + head.map(h => `<th>${h}</th>`).join("") + "</tr>";
  const body = ROWS.map(r => {
    const cls = !r.solved ? "unsolved" : (r.success ? "" : "fail");
    const ref = r.reference || {};
    const why = r.reason || (r.violations || []).join("; ");
    return `<tr class="${cls}">` +
      `<td>${esc(r.id)}</td><td>${r.solved ? "✓" : "✗"}</td><td>${r.success ? "✓" : "✗"}</td>` +
      `<td>${fmt(r.lateral_m, 2)}</td><td>${fmt(r.vertical_m, 2)}</td>` +
      `<td>${fmt(r.speed_ms, 1)}</td><td>${fmt(r.final_time_s, 1)}</td>` +
      `<td>${fmt(ref.flight_time_delta_s, 1)}</td>` +
      `<td>${ref.path_lateral_m ? fmt(ref.path_lateral_m.mean, 0) : "—"}</td>` +
      `<td class="viol">${esc(why || "")}</td></tr>`;
  }).join("");
  table.innerHTML = header + body;
}
rowTable();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
