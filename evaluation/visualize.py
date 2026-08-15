"""Render an evaluation batch as a self-contained data report plus Plotly CDN."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence, TypeVar

from evaluation.cli import add_context_args, contexts_from_args
from evaluation.context import ContextKey
from evaluation.metrics import evaluate_batch
from evaluation.records import TrajectoryRecord, load_records
from evaluation.reference import N_RESAMPLE, horizontal_arc_length_m, load_reference, resample_by_arc_length
from evaluation.thresholds import AssessmentContext

DEFAULT_MAX_TRACKS = 30
T = TypeVar("T")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_payload(
    records: Sequence[TrajectoryRecord],
    *,
    contexts: dict[ContextKey, AssessmentContext],
    max_tracks: int = DEFAULT_MAX_TRACKS,
) -> dict[str, Any]:
    if max_tracks <= 0:
        raise ValueError("max_tracks must be greater than zero")
    report = evaluate_batch(records, contexts=contexts)
    drawable = [
        record for record in records
        if record.solved and horizontal_arc_length_m(record.states) > 0.0
    ]
    selected = _sample_evenly(drawable, max_tracks)
    return {
        "report": report,
        "tracks": [_track_entry(record) for record in selected],
        "tracksShown": len(selected),
        "tracksTotal": len(drawable),
    }


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
    records = load_records(args.input)
    payload = build_payload(
        records, contexts=contexts_from_args(records, args), max_tracks=args.max_tracks
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
        return list(items)
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
        "trajectory": _polyline(resample_by_arc_length(record.states, N_RESAMPLE)),
        "target": {
            "lat": record.target_state["lat"],
            "lon": record.target_state["lon"],
            "alt": record.target_state["alt"],
        },
    }
    if record.reference_file:
        reference = load_reference(record)
        if horizontal_arc_length_m(reference.states) > 0.0:
            entry["reference"] = _polyline(resample_by_arc_length(reference.states, N_RESAMPLE))
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
<p class="note">Runway-threshold geometric verdict. Vertical uses the published-TCH path and the 22 m RNAV/RNP terminal vertical bound. This is not touchdown or landing certification. Observed events reuse the producer-side threshold event, carry explicitly uncalibrated uncertainty, and are never refitted by evaluation.</p>
<div id="cards" class="cards"></div><h2>Terminal deviations</h2><div class="grid"><div id="lat" class="chart"></div><div id="vert" class="chart"></div></div>
<h2>Track overlay</h2><div id="trackNote" class="note"></div><select id="selector"></select><div class="grid"><div id="plan" class="chart"></div><div id="profile" class="chart"></div></div>
<h2>Per-trajectory verdicts</h2><div class="scroll"><table id="rows"></table></div></div>
<script>const DATA=__DATA__,R=DATA.report,ROWS=R.trajectories;const esc=s=>String(s??"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const card=(n,l)=>`<div class="card"><div class="num">${n}</div><div>${l}</div></div>`;document.getElementById("cards").innerHTML=card(R.total,"total")+card(R.verdict_counts.pass,"pass")+card(R.verdict_counts.fail,"fail")+card(R.verdict_counts.indeterminate,"indeterminate");
const M=ROWS.filter(r=>r.deviation);const colors=M.map(r=>r.verdict==="pass"?"#16864b":r.verdict==="fail"?"#c33":"#c68a12");
Plotly.newPlot("lat",[{type:"bar",x:M.map(r=>r.file||r.id),y:M.map(r=>r.cross_track_m),marker:{color:colors},name:"cross-track"},{type:"scatter",mode:"markers",x:M.map(r=>r.file||r.id),y:M.map(r=>r.bounds.effective_lateral_m),marker:{symbol:"line-ew",size:14},name:"+ effective bound"}],{title:"Signed cross-track at threshold (m)",margin:{b:90}},{displayModeBar:false});
Plotly.newPlot("vert",[{type:"bar",x:M.map(r=>r.file||r.id),y:M.map(r=>r.vertical_m),marker:{color:colors}}],{title:"Signed vertical deviation from published-TCH path (m)",margin:{b:90}},{displayModeBar:false});
const sel=document.getElementById("selector");DATA.tracks.forEach((t,i)=>{const o=document.createElement("option");o.value=i;o.textContent=t.label;sel.appendChild(o)});document.getElementById("trackNote").textContent=`${DATA.tracksShown}/${DATA.tracksTotal} drawable tracks shown`;
function draw(i){const t=DATA.tracks[i];if(!t)return;const plan=[],prof=[],f=t.trajectory.lat.map((_,k)=>k/(t.trajectory.lat.length-1));if(t.reference){plan.push({x:t.reference.lon,y:t.reference.lat,mode:"lines",name:"reference"});prof.push({x:f,y:t.reference.alt,mode:"lines",name:"reference"})}plan.push({x:t.trajectory.lon,y:t.trajectory.lat,mode:"lines",name:"trajectory"});prof.push({x:f,y:t.trajectory.alt,mode:"lines",name:"trajectory"});Plotly.react("plan",plan,{title:`Plan · ${t.label}`,xaxis:{title:"lon"},yaxis:{title:"lat",scaleanchor:"x"}},{displayModeBar:false});Plotly.react("profile",prof,{title:"Altitude by arc fraction",xaxis:{title:"arc fraction"},yaxis:{title:"m MSL"}},{displayModeBar:false})}sel.onchange=()=>draw(Number(sel.value));draw(0);
document.getElementById("rows").innerHTML="<tr><th>flight identity</th><th>subject</th><th>benchmark</th><th>event</th><th>lateral</th><th>vertical</th><th>overall</th><th>x (m)</th><th>z (m)</th><th>reason</th></tr>"+ROWS.map(r=>`<tr><td>${esc(r.flight_key||r.file||r.id)}</td><td>${r.subject}</td><td>${r.benchmark}</td><td>${r.event_status}</td><td>${r.lateral_result}</td><td>${r.vertical_result}</td><td>${r.verdict}</td><td>${r.cross_track_m==null?"—":r.cross_track_m.toFixed(2)}</td><td>${r.vertical_m==null?"—":r.vertical_m.toFixed(2)}</td><td>${esc(r.reason)}</td></tr>`).join("");</script></body></html>"""


if __name__ == "__main__":
    main()
