"""Render an evaluation batch as a self-contained data report plus Plotly CDN."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Sequence, TypeVar

from evaluation.cli import add_context_args, contexts_for_input, winds_for_input
from evaluation.context import ContextKey
from evaluation.metrics import evaluate_batch
from evaluation.records import TrajectoryRecord, iter_records, load_record
from evaluation.reference import N_RESAMPLE, horizontal_arc_length_m, load_reference, resample_by_arc_length
from evaluation.thresholds import AssessmentContext
from evaluation.wind import WindTable

DEFAULT_MAX_TRACKS = 30
T = TypeVar("T")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_payload(
    input_path: str | Path,
    *,
    contexts: dict[ContextKey, AssessmentContext],
    winds: dict[str, WindTable] | None = None,
    max_tracks: int = DEFAULT_MAX_TRACKS,
) -> dict[str, Any]:
    """The report plus ``max_tracks`` overlays, in two passes and one record at a time.

    The report needs every record; the overlay needs only ``max_tracks`` of them. Holding
    the whole batch to draw 30 tracks costs ~1 MB per flight of resolved state, so pass one
    streams the metrics and remembers only which FILES were drawable, and pass two reloads
    the sampled few. There is no list-based variant: a second path that had to agree with
    this one verdict for verdict was only ever exercised by tests.
    """
    report_records = _CountedStream(iter_records(input_path))
    report = evaluate_batch(report_records, contexts=contexts, winds=winds)
    drawable = report_records.drawable
    selected = _sample_evenly(drawable, max_tracks)
    return {
        "report": report,
        "tracks": [_track_entry(load_record(file)) for file in selected],
        "tracksShown": len(selected),
        "tracksTotal": len(drawable),
    }


def _drawable(record: TrajectoryRecord) -> bool:
    """A solved record whose MEASURED path moved (an appended crossing row is not path)."""
    return record.solved and horizontal_arc_length_m(record.measured_states) > 0.0


class _CountedStream:
    """Pass-through iterator that notes each drawable record's PATH as it goes by."""

    def __init__(self, records: Iterator[TrajectoryRecord]) -> None:
        self._records = records
        self.drawable: list[Path] = []

    def __iter__(self) -> Iterator[TrajectoryRecord]:
        for record in self._records:
            if record.path is not None and _drawable(record):
                self.drawable.append(record.path)
            yield record


def render_html(payload: dict[str, Any], *, title: str, source_label: str) -> str:
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    return (_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__SOURCE__", html.escape(source_label))
            .replace("__DATA__", data))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a terminal-verdict HTML report")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="evaluation_report.html")
    parser.add_argument("--title", default="Terminal Approach Evaluation Report")
    parser.add_argument("--max-tracks", type=_positive_int, default=DEFAULT_MAX_TRACKS)
    add_context_args(parser)
    args = parser.parse_args(argv)
    contexts = contexts_for_input(args.input, args)
    payload = build_payload(
        args.input, contexts=contexts, winds=winds_for_input(contexts, args),
        max_tracks=args.max_tracks,
    )
    out = Path(args.output)
    out.write_text(
        render_html(payload, title=args.title, source_label=str(args.input)),
        encoding="utf-8",
    )
    print(
        f"evaluation HTML ({payload['report']['total']} trajectories, "
        f"{payload['tracksShown']}/{payload['tracksTotal']} track overlays) -> {out}"
    )


def _sample_evenly(items: Sequence[T], count: int) -> list[T]:
    if count <= 0:
        raise ValueError("count must be greater than zero")
    if len(items) <= count:
        return list(items)  # includes the empty batch
    step = (len(items) - 1) / (count - 1) if count > 1 else 0.0
    indices = sorted({round(index * step) for index in range(count)})
    return [items[index] for index in indices]


def _track_entry(record: TrajectoryRecord) -> dict[str, Any]:
    record_id = str(record.source.get("id") or "trajectory")
    flight_key = record.source.get("flight_key")
    file_name = record.path.name if record.path else None
    stable = str(flight_key or file_name or record_id)
    label = record_id if stable == record_id else f"{record_id} — {stable}"
    entry: dict[str, Any] = {
        "id": record_id,
        "flight_key": flight_key,
        "file": file_name,
        "label": label,
        # The MEASURED path: an observed record's appended fitted-crossing row is a
        # modeling boundary, not flown trajectory (records.measured_states).
        "trajectory": _polyline(resample_by_arc_length(record.measured_states, N_RESAMPLE)),
        "target": {
            "lat": record.target_state["lat"],
            "lon": record.target_state["lon"],
            "alt": record.target_state["alt"],
        },
    }
    if record.reference_file:
        reference = load_reference(record)
        if _drawable(reference):
            entry["reference"] = _polyline(
                resample_by_arc_length(reference.measured_states, N_RESAMPLE)
            )
    return entry


def _polyline(points: list[tuple[float, float, float]]) -> dict[str, list[float]]:
    return {
        "lat": [point[0] for point in points],
        "lon": [point[1] for point in points],
        "alt": [point[2] for point in points],
    }


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>__TITLE__</title><script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>body{font:14px system-ui;margin:0;background:#f5f7fa;color:#18212b}.wrap{max-width:1180px;margin:auto;padding:24px}h1{font-size:24px}.meta,.note{color:#667085}.cards{display:flex;gap:12px;flex-wrap:wrap}.card{background:white;padding:12px 18px;border-radius:9px;box-shadow:0 1px 3px #0001}.num{font-size:22px;font-weight:700}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.chart,table{background:white;border-radius:9px}table{width:100%;border-collapse:collapse}th,td{padding:7px;border-bottom:1px solid #eee;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto;max-height:520px}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style></head>
<body><div class="wrap"><h1>__TITLE__</h1><div class="meta">input: __SOURCE__</div>
<p class="note">Runway-threshold geometric verdict. Lateral is half the published runway width (did the crossing lie over the pavement) — not a navigation-containment bound. Vertical uses the published-TCH path and the 22 m RNAV/RNP terminal vertical bound. Speed uses the type's PUBLISHED approach-speed window [V_ref,lo·√n, V_ref,hi + 20 kt] (FAA Aircraft Characteristics Database speed at MALW, scaled by √(m/MALW); docs/reference_speeds): computed subjects at their crossing mass and load factor n are judged on the crossing model airspeed; observed baselines, whose mass is unmeasured, over the type's published mass range on the fitted crossing GROUND speed corrected by the field's METAR headwind into an airspeed estimate (shown as est) when a report is usable, else on the raw ground speed as a stated proxy (shown as GS); their load factor is inverted from the flight's own final 20 s of kinematics. This is not touchdown or landing certification. Observed events reuse the producer-side threshold event, carry explicitly uncalibrated uncertainty, and are never refitted by evaluation.</p>
<div id="cards" class="cards"></div><h2>Terminal deviations</h2><div class="grid"><div id="lat" class="chart"></div><div id="vert" class="chart"></div></div>
<h2>Track overlay</h2><div id="trackNote" class="note"></div><select id="selector"></select><div class="grid"><div id="plan" class="chart"></div><div id="profile" class="chart"></div></div>
<h2>Per-trajectory verdicts</h2><div class="scroll"><table id="rows"></table></div></div>
<script>const DATA=__DATA__,R=DATA.report,ROWS=R.trajectories;const esc=s=>String(s??"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const card=(n,l)=>`<div class="card"><div class="num">${n}</div><div>${l}</div></div>`;document.getElementById("cards").innerHTML=card(R.total,"total")+card(R.verdict_counts.pass,"pass")+card(R.verdict_counts.fail,"fail")+card(R.verdict_counts.indeterminate,"indeterminate");
const M=ROWS.filter(r=>r.deviation);const colors=M.map(r=>r.verdict==="pass"?"#16864b":r.verdict==="fail"?"#c33":"#c68a12");
Plotly.newPlot("lat",[{type:"bar",x:M.map(r=>r.file||r.id),y:M.map(r=>r.cross_track_m),marker:{color:colors},name:"cross-track"},{type:"scatter",mode:"markers",x:M.map(r=>r.file||r.id),y:M.map(r=>r.bounds.lateral_m),marker:{symbol:"line-ew",size:14},name:"+ runway half-width"}],{title:"Signed cross-track at threshold (m)",margin:{b:90}},{displayModeBar:false});
Plotly.newPlot("vert",[{type:"bar",x:M.map(r=>r.file||r.id),y:M.map(r=>r.vertical_m),marker:{color:colors}}],{title:"Signed vertical deviation from published-TCH path (m)",margin:{b:90}},{displayModeBar:false});
const sel=document.getElementById("selector");DATA.tracks.forEach((t,i)=>{const o=document.createElement("option");o.value=i;o.textContent=t.label;sel.appendChild(o)});document.getElementById("trackNote").textContent=`${DATA.tracksShown}/${DATA.tracksTotal} drawable tracks shown`;
function draw(i){const t=DATA.tracks[i];if(!t)return;const plan=[],prof=[],f=t.trajectory.lat.map((_,k)=>k/(t.trajectory.lat.length-1));if(t.reference){plan.push({x:t.reference.lon,y:t.reference.lat,mode:"lines",name:"reference"});prof.push({x:f,y:t.reference.alt,mode:"lines",name:"reference"})}plan.push({x:t.trajectory.lon,y:t.trajectory.lat,mode:"lines",name:"trajectory"});prof.push({x:f,y:t.trajectory.alt,mode:"lines",name:"trajectory"});Plotly.react("plan",plan,{title:`Plan · ${t.label}`,xaxis:{title:"lon"},yaxis:{title:"lat",scaleanchor:"x"}},{displayModeBar:false});Plotly.react("profile",prof,{title:"Altitude by arc fraction",xaxis:{title:"arc fraction"},yaxis:{title:"m MSL"}},{displayModeBar:false})}sel.onchange=()=>draw(Number(sel.value));draw(0);
document.getElementById("rows").innerHTML="<tr><th>flight identity</th><th>subject</th><th>benchmark</th><th>event</th><th>lateral</th><th>vertical</th><th>speed</th><th>overall</th><th>x (m)</th><th>z (m)</th><th>V judged (m/s)</th><th>n</th><th>reason</th></tr>"+ROWS.map(r=>`<tr><td>${esc(r.flight_key||r.file||r.id)}</td><td>${r.subject}</td><td>${r.benchmark}</td><td>${r.event_status}</td><td>${r.lateral_result}</td><td>${r.vertical_result}</td><td>${r.speed_result}</td><td>${r.verdict}</td><td>${r.cross_track_m==null?"—":r.cross_track_m.toFixed(2)}</td><td>${r.vertical_m==null?"—":r.vertical_m.toFixed(2)}</td><td>${r.crossing_airspeed_estimate_ms!=null?r.crossing_airspeed_estimate_ms.toFixed(1)+" est":r.deviation&&r.deviation.crossing_speed_ms!=null?r.deviation.crossing_speed_ms.toFixed(1):r.deviation&&r.deviation.crossing_ground_speed_ms!=null?r.deviation.crossing_ground_speed_ms.toFixed(1)+" GS":"—"}</td><td>${r.deviation?r.deviation.crossing_load_factor.toFixed(2)+(r.deviation.crossing_load_factor_source==="assumed_1g"?"*":""):"—"}</td><td>${esc(r.reason)}</td></tr>`).join("");</script></body></html>"""


if __name__ == "__main__":
    main()
