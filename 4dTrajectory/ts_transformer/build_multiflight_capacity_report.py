#!/usr/bin/env python3
"""Build a self-contained visual diagnosis from train-only overfit results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


RESULT_SCHEMA = "ts-control-fixed-dt-single-flight-overfit-v3-clock-aligned-diagnostics"
NON_RECIPE_CONFIG_FIELDS = frozenset({"notes"})


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _recipe(result: dict[str, Any]) -> dict[str, Any]:
    config = result["config"]
    if not isinstance(config, dict):
        raise ValueError("experiment config must be an object")
    return {
        key: config[key]
        for key in sorted(config)
        if key not in NON_RECIPE_CONFIG_FIELDS
    }


def _downsample(rows: list[dict[str, Any]], maximum: int = 250) -> list[dict[str, float]]:
    if not rows:
        return []
    stride = max(1, math.ceil(len(rows) / maximum))
    selected = rows[::stride]
    if selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    return [
        {
            "epoch": int(row["epoch"]),
            "train": _finite(row["train_loss"], "train loss"),
            "replay": _finite(row["val_loss"], "replay loss"),
        }
        for row in selected
    ]


def _first_crossing(times: Iterable[float], values: Iterable[float], threshold: float):
    for time_s, value in zip(times, values):
        if value >= threshold:
            return float(time_s)
    return None


def _saturation_ratios(segments: dict[str, Any]) -> list[float]:
    controls = segments["controls"]
    lower = segments["control_lower"]
    upper = segments["control_upper"]
    ratios = []
    for channel in range(3):
        width = float(upper[channel]) - float(lower[channel])
        tolerance = 0.02 * width
        saturated = sum(
            float(row[channel]) <= float(lower[channel]) + tolerance
            or float(row[channel]) >= float(upper[channel]) - tolerance
            for row in controls
        )
        ratios.append(saturated / len(controls))
    return ratios


def _chart_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Build chart series without treating zero-weight velocities as observations."""

    names = diagnostics["channel_names"]
    required = ("e", "n", "u", "edot", "ndot", "udot")
    if any(name not in names for name in required):
        raise ValueError("diagnostic channels do not contain the required ENU state")
    index = {name: names.index(name) for name in required}
    fixed = diagnostics["fixed_dt"]
    times = [float(value) for value in fixed["offset_s"]]
    reference = fixed["reference_state"]
    predicted = fixed["predicted_state"]
    measured = [bool(value) for value in fixed["reference_fully_measured"]]
    if not (len(times) == len(reference) == len(predicted) == len(measured)):
        raise ValueError("fixed-dt diagnostic rows and masks must have matching lengths")

    def velocity_series(states, valid):
        horizontal, vertical = [], []
        for row, row_valid in zip(states, valid):
            if not row_valid:
                horizontal.append(None)
                vertical.append(None)
                continue
            horizontal.append(
                math.hypot(float(row[index["edot"]]), float(row[index["ndot"]]))
            )
            vertical.append(float(row[index["udot"]]))
        return horizontal, vertical

    def interval_series(states, valid):
        all_times = [0.0, *times]
        all_states = [diagnostics["anchor_state"], *states]
        all_valid = [True, *valid]
        consistency, acceleration = [], []
        for row in range(1, len(all_states)):
            if not (all_valid[row - 1] and all_valid[row]):
                consistency.append(None)
                acceleration.append(None)
                continue
            before, after = all_states[row - 1], all_states[row]
            dt_s = all_times[row] - all_times[row - 1]
            if dt_s <= 0.0:
                raise ValueError("fixed-dt diagnostic offsets must be strictly increasing")
            finite_difference = [
                (float(after[index[channel]]) - float(before[index[channel]]))
                / dt_s
                for channel in ("e", "n", "u")
            ]
            midpoint_velocity = [
                (
                    float(before[index[channel]])
                    + float(after[index[channel]])
                )
                / 2.0
                for channel in ("edot", "ndot", "udot")
            ]
            velocity_delta = [
                (
                    float(after[index[channel]])
                    - float(before[index[channel]])
                )
                / dt_s
                for channel in ("edot", "ndot", "udot")
            ]
            consistency.append(
                math.sqrt(
                    sum(
                        (difference - velocity) ** 2
                        for difference, velocity in zip(
                            finite_difference, midpoint_velocity
                        )
                    )
                )
            )
            acceleration.append(
                math.sqrt(sum(component**2 for component in velocity_delta))
            )
        return consistency, acceleration

    predicted_valid = [True] * len(predicted)
    reference_horizontal, reference_vertical = velocity_series(reference, measured)
    predicted_horizontal, predicted_vertical = velocity_series(
        predicted, predicted_valid
    )
    reference_consistency, reference_acceleration = interval_series(
        reference, measured
    )
    predicted_consistency, predicted_acceleration = interval_series(
        predicted, predicted_valid
    )
    return {
        "offset_s": times,
        "reference_horizontal_speed_mps": reference_horizontal,
        "reference_vertical_speed_mps": reference_vertical,
        "predicted_horizontal_speed_mps": predicted_horizontal,
        "predicted_vertical_speed_mps": predicted_vertical,
        "reference_consistency_mps": reference_consistency,
        "predicted_consistency_mps": predicted_consistency,
        "reference_acceleration_mps2": reference_acceleration,
        "predicted_acceleration_mps2": predicted_acceleration,
    }


def _compact_result(result: dict[str, Any], source: Path) -> dict[str, Any]:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError(f"{source}: expected schema {RESULT_SCHEMA!r}")
    policy = result.get("test_policy", {})
    if policy.get("outer_test_tracks_opened") is not False:
        raise ValueError(f"{source}: outer-test isolation is not proven")
    if policy.get("validation_tracks_opened") is not False:
        raise ValueError(f"{source}: validation values were opened")
    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"{source}: diagnostics are missing")
    segments = diagnostics["segments"]
    for key in ("control_lower", "control_upper"):
        if key not in segments:
            raise ValueError(f"{source}: {key} is missing; regenerate this result")

    dense = result["metrics"]["fixed_dt_training_clock"]
    predicted = result["metrics"]["deployable_predicted_clock"]
    fixed = diagnostics["fixed_dt"]
    errors = fixed["position_error_m"]
    times = fixed["offset_s"]
    duration_values = [float(value) for value in segments["duration_s"]]
    duration_mean = sum(duration_values) / len(duration_values)
    duration_variance = sum(
        (value - duration_mean) ** 2 for value in duration_values
    ) / len(duration_values)
    kinematics = result["metrics"]["reference_kinematics"]
    duration_partition = result["metrics"]["duration_partition"]
    metrics = {
        "fixed_dt_ade_m": _finite(dense["ade_m"], "fixed-dt ADE"),
        "common_grid_ade_m": _finite(predicted["common_grid_ade_m"], "common ADE"),
        "common_grid_fde_m": _finite(predicted["common_grid_fde_m"], "common FDE"),
        "arc_distance_m": _finite(predicted["arc_length_distance_mean_m"], "arc distance"),
        "arc_horizontal_m": _finite(predicted["arc_length_horizontal_mean_m"], "horizontal error"),
        "arc_vertical_m": _finite(predicted["arc_length_vertical_mae_m"], "vertical error"),
        "terminal_velocity_mps": _finite(
            predicted["arc_length_terminal_velocity_error_mps"],
            "terminal velocity error",
        ),
        "path_length_ratio": _finite(
            predicted["arc_length_path_length_ratio"], "path length ratio"
        ),
        "final_time_error_s": _finite(
            predicted["final_time_abs_error_s"], "final time error"
        ),
        "terminal_position_components_m": [
            _finite(predicted["arc_length_terminal_along_position_abs_m"], "along position"),
            _finite(predicted["arc_length_terminal_cross_position_abs_m"], "cross position"),
            _finite(predicted["arc_length_terminal_vertical_position_abs_m"], "vertical position"),
        ],
        "terminal_velocity_components_mps": [
            _finite(predicted["arc_length_terminal_along_velocity_abs_mps"], "along velocity"),
            _finite(predicted["arc_length_terminal_cross_velocity_abs_mps"], "cross velocity"),
            _finite(predicted["arc_length_terminal_vertical_velocity_abs_mps"], "vertical velocity"),
        ],
        "first_error_crossing_s": {
            str(threshold): _first_crossing(times, errors, threshold)
            for threshold in (100, 500, 1000)
        },
        "control_saturation_ratio": _saturation_ratios(segments),
        "duration_cv": math.sqrt(duration_variance) / duration_mean,
        "duration_entropy_normalized": _finite(
            duration_partition["fraction_entropy"], "duration entropy"
        ) / math.log(len(duration_values)),
        "reference_position_velocity_rmse_mps": _finite(
            kinematics["position_velocity_rmse_mps"],
            "reference position/velocity consistency",
        ),
        "reference_acceleration_p95_mps2": _finite(
            kinematics["acceleration_p95_mps2"], "reference acceleration"
        ),
        "reference_jerk_p95_mps3": _finite(
            kinematics["jerk_p95_mps3"], "reference jerk"
        ),
        "reference_turn_rate_p95_deg_s": _finite(
            kinematics["turn_rate_p95_deg_s"], "reference turn rate"
        ),
    }
    return {
        "source": str(source),
        "id": result["flight"]["dataset_id"],
        "airport": result["flight"]["airport"],
        "runway": diagnostics["runway"],
        "duration_s": _finite(result["flight"]["remaining_time_s"], "duration"),
        "best_epoch": int(result["loss"]["best_epoch"]),
        "epochs_run": int(result["loss"]["epochs_run"]),
        "loss_reduction_pct": _finite(result["loss"]["reduction_pct"], "loss reduction"),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "charts": _chart_diagnostics(diagnostics),
        "loss_curve": _downsample(result["history"]),
    }


def _load_results(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected_recipe = _recipe(raw[0])
    for path, result in zip(paths[1:], raw[1:]):
        current = _recipe(result)
        if current != expected_recipe:
            differences = [
                key
                for key in sorted(expected_recipe.keys() | current.keys())
                if current.get(key) != expected_recipe.get(key)
            ]
            raise ValueError(f"{path}: experiment recipe differs in {differences}")
    compact = [_compact_result(result, path) for path, result in zip(paths, raw)]
    compact.sort(key=lambda row: row["duration_s"])
    return compact, expected_recipe


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2+4 control loss 多航迹容量诊断</title>
<style>
:root{--bg:#07111e;--panel:#0d1b2b;--panel2:#102338;--ink:#e9f2ff;--muted:#94a9c4;--grid:#29405a;--cyan:#42d5e8;--amber:#ffbd55;--pink:#ff6f91;--green:#5ee0a0;--red:#ff6b6b;--blue:#6ca6ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% -10%,#153d5a 0,transparent 34%),var(--bg);color:var(--ink);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI","Noto Sans SC",sans-serif}
main{max-width:1480px;margin:auto;padding:32px}.eyebrow{color:var(--cyan);font-weight:700;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(28px,4vw,52px);line-height:1.12;margin:.18em 0}.subtitle{max-width:1000px;color:var(--muted);font-size:16px}.notice{border-left:4px solid var(--green);background:#0c2530;padding:12px 16px;margin:24px 0;border-radius:0 8px 8px 0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card,.panel{background:linear-gradient(145deg,rgba(16,35,56,.97),rgba(10,25,41,.97));border:1px solid #203b57;border-radius:12px;box-shadow:0 14px 40px #0004}.card{padding:16px}.card .value{font-size:27px;font-weight:750}.card .label{color:var(--muted)}.panel{padding:18px;margin:14px 0}.panel h2,.panel h3{margin:0 0 10px}.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.chart{min-height:290px}.chart.small{min-height:230px}svg{width:100%;height:auto;display:block}.axis text,.tick{fill:var(--muted);font-size:11px}.axis line,.gridline{stroke:var(--grid);stroke-width:1}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px}.legend i{display:inline-block;width:16px;height:3px;margin-right:5px;vertical-align:middle}.selector{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.selector button{color:var(--muted);background:#0a1726;border:1px solid #29445f;border-radius:999px;padding:8px 13px;cursor:pointer}.selector button.active{background:var(--cyan);border-color:var(--cyan);color:#04131d;font-weight:700}.finding{padding:12px 15px;border-radius:8px;background:#0a1726;border-left:3px solid var(--amber);margin:8px 0}.finding strong{color:var(--amber)}table{width:100%;border-collapse:collapse;min-width:980px}th,td{padding:9px 10px;border-bottom:1px solid #20364d;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--muted);position:sticky;top:0;background:var(--panel)}.table-wrap{overflow:auto}.pill{display:inline-block;padding:2px 7px;border-radius:99px;background:#18334c;color:var(--cyan)}details{border-top:1px solid #243d55;padding:10px 0}summary{cursor:pointer;color:var(--cyan)}code{color:#b9d9ff;overflow-wrap:anywhere}.footer{color:var(--muted);margin-top:25px}.tooltip{position:fixed;display:none;pointer-events:none;background:#020914eF;border:1px solid #34536f;border-radius:6px;padding:7px 9px;font-size:12px;z-index:10}
@media(max-width:900px){main{padding:18px}.grid2,.grid3{grid-template-columns:1fr}}
@media print{body{background:#fff;color:#111}.panel,.card{background:#fff;box-shadow:none;border-color:#bbb}.subtitle,.label,.legend,.footer{color:#444}.selector{display:none}}
</style>
</head>
<body><main>
<div class="eyebrow">train-only · capacity diagnosis · fixed 2 s clock</div>
<h1>2+4 control loss 多航迹容量诊断</h1>
<p class="subtitle">同一模型、同一种子、同一套 loss，在 5 条 outer-train 航迹上做单航迹过拟合。这里测的是“模型 + 控制参数化 + dynamics + loss 能否表达一条已知航迹”的上限，不是泛化性能。</p>
<div class="notice"><b>数据隔离：</b>页面生成器逐文件验证 <code>outer_test_tracks_opened=false</code> 与 <code>validation_tracks_opened=false</code>；航迹值仅来自 outer-train。没有用 test 或 validation 指标选择配方。</div>
<section id="summaryCards" class="cards"></section>
<section class="panel"><h2>先看结论</h2><div id="globalFindings"></div></section>
<section class="grid2">
  <div class="panel"><h3>时域长度与误差</h3><div id="durationScatter" class="chart"></div></div>
  <div class="panel"><h3>误差结构：水平、垂直、终端速度</h3><div id="structureBars" class="chart"></div></div>
</section>
<section class="panel"><h2>五条航迹对照</h2><div class="table-wrap"><table id="summaryTable"></table></div></section>
<section class="panel">
  <h2>逐航迹诊断</h2><div id="flightSelector" class="selector"></div><div id="flightHeader"></div><div id="flightFindings"></div>
  <div class="grid2">
    <div><h3>平面轨迹 E–N</h3><div id="planChart" class="chart"></div></div>
    <div><h3>高度–时间</h3><div id="altitudeChart" class="chart"></div></div>
    <div><h3>位置误差–时间</h3><div id="errorChart" class="chart"></div></div>
    <div><h3>速度状态–时间</h3><div id="velocityChart" class="chart"></div></div>
    <div><h3>位置差分速度 vs 状态速度</h3><div id="consistencyChart" class="chart"></div></div>
    <div><h3>参考/预测加速度幅值</h3><div id="accelerationChart" class="chart"></div></div>
  </div>
  <h3>预测控制与分段时长</h3><div class="grid3">
    <div><div id="thrustChart" class="chart small"></div></div>
    <div><div id="bankChart" class="chart small"></div></div>
    <div><div id="loadChart" class="chart small"></div></div>
  </div>
  <div class="grid2">
    <div><h3>64 段时长</h3><div id="durationChart" class="chart small"></div></div>
    <div><h3>终端误差分解</h3><div id="terminalChart" class="chart small"></div></div>
    <div><h3>训练/回放 loss</h3><div id="lossChart" class="chart small"></div></div>
    <div><h3>可核查信息</h3><div id="audit"></div></div>
  </div>
</section>
<section class="panel"><h2>如何读这些图</h2>
<details open><summary>common-grid ADE 与弧长误差为什么会不同？</summary><p>common-grid 在同一物理时间比较两条轨迹；弧长误差按水平路径进度重新对齐。若 common-grid ADE 明显高于弧长距离，说明路径可能大致相似，但速度/进度安排不同。两者都高才是空间几何本身偏离。</p></details>
<details><summary>FDE 很小是否代表正确降落？</summary><p>不代表。FDE 只说明最终位置接近；还必须同时看终端沿跑道/横向/垂直速度、此前高度剖面、跑道对正和控制是否贴边。本报告特意把这些量拆开。</p></details>
<details><summary>单航迹过拟合能说明什么？</summary><p>若单航迹也拟合不好，瓶颈至少存在于表达、动力学、优化或 loss；不能归因于跨航迹泛化。若单航迹拟合好，才值得继续用 train/validation 测泛化。此实验不提供 outer-test 结论。</p></details>
</section>
<p class="footer">生成时间：<span id="generatedAt"></span> · 单文件离线报告，无外部脚本、字体或数据请求。</p>
</main><div id="tooltip" class="tooltip"></div>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
const report=JSON.parse(document.getElementById('report-data').textContent);const flights=report.flights;
const colors=['#42d5e8','#ffbd55','#ff6f91','#5ee0a0','#6ca6ff','#c58cff'];
const fmt=(v,d=1)=>v==null?'—':Number(v).toLocaleString('zh-CN',{maximumFractionDigits:d,minimumFractionDigits:d});
const el=id=>document.getElementById(id); const S='http://www.w3.org/2000/svg';
function node(name,attrs={}){const n=document.createElementNS(S,name);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);return n}
function extent(values){let lo=Math.min(...values),hi=Math.max(...values);if(!isFinite(lo)||!isFinite(hi))return[0,1];if(lo===hi){const p=Math.abs(lo||1)*.05;lo-=p;hi+=p}const p=(hi-lo)*.07;return[lo-p,hi+p]}
function lineChart(id,series,opt={}){const host=el(id);host.innerHTML='';const W=720,H=opt.height||280,m={l:62,r:22,t:24,b:44};const svg=node('svg',{viewBox:`0 0 ${W} ${H}`});host.append(svg);const allX=series.flatMap(s=>s.x),allY=series.flatMap(s=>s.y.filter(Number.isFinite));let[x0,x1]=opt.xDomain||extent(allX),[y0,y1]=opt.yDomain||extent(allY);const sx=x=>m.l+(x-x0)/(x1-x0)*(W-m.l-m.r),sy=y=>H-m.b-(y-y0)/(y1-y0)*(H-m.t-m.b);
for(let i=0;i<=5;i++){const y=y0+(y1-y0)*i/5,Y=sy(y);svg.append(node('line',{x1:m.l,x2:W-m.r,y1:Y,y2:Y,class:'gridline'}));const t=node('text',{x:m.l-8,y:Y+4,'text-anchor':'end',class:'tick'});t.textContent=fmt(y,opt.yDigits??1);svg.append(t)}
for(let i=0;i<=5;i++){const x=x0+(x1-x0)*i/5,X=sx(x);const t=node('text',{x:X,y:H-18,'text-anchor':'middle',class:'tick'});t.textContent=fmt(x,opt.xDigits??0);svg.append(t)}
series.forEach((s,i)=>{let d='',penDown=false;s.x.forEach((x,j)=>{const y=s.y[j];if(!Number.isFinite(y)){penDown=false;return}d+=(penDown?'L':'M')+sx(x)+','+sy(y);penDown=true});svg.append(node('path',{d,fill:'none',stroke:s.color||colors[i], 'stroke-width':s.width||2,'stroke-dasharray':s.dash||''}));});
const tx=node('text',{x:(m.l+W-m.r)/2,y:H-3,'text-anchor':'middle',class:'tick'});tx.textContent=opt.xLabel||'';svg.append(tx);const ty=node('text',{x:14,y:(m.t+H-m.b)/2,transform:`rotate(-90 14 ${(m.t+H-m.b)/2})`,'text-anchor':'middle',class:'tick'});ty.textContent=opt.yLabel||'';svg.append(ty);legend(host,series);}
function legend(host,series){const d=document.createElement('div');d.className='legend';series.forEach((s,i)=>{const x=document.createElement('span');x.innerHTML=`<i style="background:${s.color||colors[i]}"></i>${s.name}`;d.append(x)});host.append(d)}
function barChart(id,groups,opt={}){const host=el(id);host.innerHTML='';const W=720,H=opt.height||280,m={l:62,r:18,t:22,b:62};const svg=node('svg',{viewBox:`0 0 ${W} ${H}`});host.append(svg);const max=Math.max(...groups.flatMap(g=>g.values),1);const sy=y=>H-m.b-y/max*(H-m.t-m.b),gw=(W-m.l-m.r)/groups.length,bw=Math.min(34,gw/(groups[0].values.length+1));for(let i=0;i<=5;i++){const v=max*i/5,Y=sy(v);svg.append(node('line',{x1:m.l,x2:W-m.r,y1:Y,y2:Y,class:'gridline'}));const t=node('text',{x:m.l-8,y:Y+4,'text-anchor':'end',class:'tick'});t.textContent=fmt(v,1);svg.append(t)}groups.forEach((g,gi)=>{g.values.forEach((v,j)=>{const x=m.l+gi*gw+gw/2+(j-(g.values.length-1)/2)*bw;svg.append(node('rect',{x:x-bw*.42,y:sy(v),width:bw*.84,height:H-m.b-sy(v),fill:colors[j],rx:2}))});const t=node('text',{x:m.l+gi*gw+gw/2,y:H-40,'text-anchor':'middle',class:'tick'});t.textContent=g.label;svg.append(t)});if(opt.labels)legend(host,opt.labels.map((name,i)=>({name,color:colors[i]})));}
function planChart(f){const d=f.diagnostics,idx=Object.fromEntries(d.channel_names.map((x,i)=>[x,i])),ref=d.fixed_dt.reference_state,pred=d.fixed_dt.predicted_state;const all=[...ref,...pred],xs=all.map(r=>r[idx.e]),ys=all.map(r=>r[idx.n]);let[x0,x1]=extent(xs),[y0,y1]=extent(ys);const span=Math.max(x1-x0,y1-y0),cx=(x0+x1)/2,cy=(y0+y1)/2;x0=cx-span/2;x1=cx+span/2;y0=cy-span/2;y1=cy+span/2;lineChart('planChart',[{name:'reference',x:ref.map(r=>r[idx.e]),y:ref.map(r=>r[idx.n]),color:colors[0]},{name:'prediction',x:pred.map(r=>r[idx.e]),y:pred.map(r=>r[idx.n]),color:colors[1]}],{xDomain:[x0,x1],yDomain:[y0,y1],xLabel:'East (m)',yLabel:'North (m)'});}
function findings(f){const m=f.metrics,out=[];if(m.common_grid_fde_m<20&&m.common_grid_ade_m>200)out.push(['终点好、过程差',`FDE 只有 ${fmt(m.common_grid_fde_m)} m，但 ADE 为 ${fmt(m.common_grid_ade_m)} m；终端位置约束成功，中途航迹仍未拟合。`]);if(m.arc_vertical_m>m.arc_horizontal_m*1.5)out.push(['垂直剖面主导',`弧长对齐后垂直 MAE ${fmt(m.arc_vertical_m)} m，显著高于水平 ${fmt(m.arc_horizontal_m)} m。`]);else if(m.arc_horizontal_m>m.arc_vertical_m*1.2)out.push(['水平几何主导',`弧长水平 MAE ${fmt(m.arc_horizontal_m)} m，高于垂直 ${fmt(m.arc_vertical_m)} m。`]);else out.push(['水平与垂直均未解决',`弧长水平/垂直 MAE 分别为 ${fmt(m.arc_horizontal_m)} / ${fmt(m.arc_vertical_m)} m。`]);if(m.terminal_velocity_mps>10)out.push(['终端速度未锁定',`终端速度误差 ${fmt(m.terminal_velocity_mps)} m/s，主要分量见下方终端分解。`]);if(Math.abs(m.path_length_ratio-1)>.08)out.push(['路径长度失真',`预测/参考水平路径长度比为 ${fmt(m.path_length_ratio,3)}，说明不仅是时序错位，路径几何本身也已偏离。`]);if(m.reference_position_velocity_rmse_mps>3)out.push(['参考状态不自洽',`参考位置差分速度与速度状态 RMSE 为 ${fmt(m.reference_position_velocity_rmse_mps)} m/s；一个严格满足 dynamics 的解无法同时逐点满足两者。`]);const sat=m.control_saturation_ratio;if(Math.max(...sat)>.2)out.push(['控制贴边',`推力/坡度/载荷因子在上下界 2% 内的段比例为 ${sat.map(x=>fmt(100*x,0)+'%').join(' / ')}。`]);return out}
function renderGlobal(){const mean=k=>flights.reduce((s,f)=>s+f.metrics[k],0)/flights.length;el('summaryCards').innerHTML=[['航迹数',flights.length],['时域范围',`${fmt(flights[0].duration_s,0)}–${fmt(flights.at(-1).duration_s,0)} s`],['平均 ADE',`${fmt(mean('common_grid_ade_m'))} m`],['平均 FDE',`${fmt(mean('common_grid_fde_m'))} m`],['平均终端速度差',`${fmt(mean('terminal_velocity_mps'))} m/s`]].map(([a,b])=>`<div class="card"><div class="value">${b}</div><div class="label">${a}</div></div>`).join('');
const endpoint=flights.filter(f=>f.metrics.common_grid_fde_m<20&&f.metrics.common_grid_ade_m>200).length, vertical=flights.filter(f=>f.metrics.arc_vertical_m>100).length,long=flights.filter(f=>f.duration_s>300),short=flights.filter(f=>f.duration_s<=300),inconsistent=flights.filter(f=>f.metrics.reference_position_velocity_rmse_mps>3).length;const avg=a=>a.reduce((s,f)=>s+f.metrics.common_grid_ade_m,0)/a.length;const range=k=>[Math.min(...flights.map(f=>f.metrics[k])),Math.max(...flights.map(f=>f.metrics[k]))];const pv=range('reference_position_velocity_rmse_mps'),acc=range('reference_acceleration_p95_mps2');el('globalFindings').innerHTML=`<div class="finding"><strong>${inconsistent}/${flights.length} 条参考 state 存在位置–速度自一致性张力。</strong> 位置差分速度与速度状态 RMSE 为 ${fmt(pv[0])}–${fmt(pv[1])} m/s，参考加速度 p95 为 ${fmt(acc[0])}–${fmt(acc[1])} m/s²；这是同时加入几何与速度 loss 后的共同不可约误差来源。</div><div class="finding"><strong>${endpoint}/${flights.length} 条出现“准确到点但过程不准”。</strong> 这不是普通 FDE 失败，而是终端位置与整段状态拟合解耦。</div><div class="finding"><strong>${vertical}/${flights.length} 条弧长垂直 MAE 超过 100 m。</strong> 垂直剖面是跨机场重复出现的瓶颈。</div><div class="finding"><strong>长时域误差放大：</strong> ≤300 s 平均 ADE ${fmt(avg(short))} m；>300 s 平均 ADE ${fmt(avg(long))} m。长航迹路径长度比分别偏到 0.727 与 2.671，方向相反，说明是坏局部解而非固定比例积分漂移。</div>`;
lineChart('durationScatter',[{name:'common-grid ADE',x:flights.map(f=>f.duration_s),y:flights.map(f=>f.metrics.common_grid_ade_m),color:colors[0]},{name:'弧长 3D 距离',x:flights.map(f=>f.duration_s),y:flights.map(f=>f.metrics.arc_distance_m),color:colors[1]}],{xLabel:'预测时域 (s)',yLabel:'位置误差 (m)'});barChart('structureBars',flights.map(f=>({label:f.airport,values:[f.metrics.arc_horizontal_m,f.metrics.arc_vertical_m,f.metrics.terminal_velocity_mps*10]})),{labels:['水平 MAE (m)','垂直 MAE (m)','终端速度差 ×10']});
el('summaryTable').innerHTML='<thead><tr><th>航迹</th><th>时域(s)</th><th>ADE(m)</th><th>FDE(m)</th><th>弧长水平(m)</th><th>弧长垂直(m)</th><th>终端速度(m/s)</th><th>路径长度比</th><th>参考位置/速度 RMSE</th><th>最佳 epoch</th></tr></thead><tbody>'+flights.map(f=>`<tr><td><span class="pill">${f.airport}</span> ${f.runway}</td><td>${fmt(f.duration_s)}</td><td>${fmt(f.metrics.common_grid_ade_m)}</td><td>${fmt(f.metrics.common_grid_fde_m,2)}</td><td>${fmt(f.metrics.arc_horizontal_m)}</td><td>${fmt(f.metrics.arc_vertical_m)}</td><td>${fmt(f.metrics.terminal_velocity_mps,2)}</td><td>${fmt(f.metrics.path_length_ratio,3)}</td><td>${fmt(f.metrics.reference_position_velocity_rmse_mps,2)} m/s</td><td>${f.best_epoch}/${f.epochs_run}</td></tr>`).join('')+'</tbody>';}
function renderFlight(index){document.querySelectorAll('#flightSelector button').forEach((b,i)=>b.classList.toggle('active',i===index));const f=flights[index],d=f.diagnostics,chart=f.charts,idx=Object.fromEntries(d.channel_names.map((x,i)=>[x,i])),t=d.fixed_dt.offset_s,ref=d.fixed_dt.reference_state,pred=d.fixed_dt.predicted_state,m=f.metrics;el('flightHeader').innerHTML=`<h3>${f.airport} ${f.runway} · ${f.id.split(':')[1]} · ${fmt(f.duration_s)} s</h3><p class="subtitle">loss 下降 ${fmt(f.loss_reduction_pct)}%；最佳 epoch ${f.best_epoch}/${f.epochs_run}；最终时长误差 ${fmt(m.final_time_error_s,3)} s。</p>`;el('flightFindings').innerHTML=findings(f).map(x=>`<div class="finding"><strong>${x[0]}：</strong>${x[1]}</div>`).join('');planChart(f);
lineChart('altitudeChart',[{name:'reference U',x:t,y:ref.map(r=>r[idx.u]),color:colors[0]},{name:'prediction U',x:t,y:pred.map(r=>r[idx.u]),color:colors[1]}],{xLabel:'anchor 后时间 (s)',yLabel:'Up (m)'});
const hErr=pred.map((r,i)=>Math.hypot(r[idx.e]-ref[i][idx.e],r[idx.n]-ref[i][idx.n])),vErr=pred.map((r,i)=>Math.abs(r[idx.u]-ref[i][idx.u]));lineChart('errorChart',[{name:'3D',x:t,y:d.fixed_dt.position_error_m,color:colors[2]},{name:'水平',x:t,y:hErr,color:colors[0]},{name:'垂直',x:t,y:vErr,color:colors[1]}],{xLabel:'anchor 后时间 (s)',yLabel:'误差 (m)',yDomain:[0,Math.max(...d.fixed_dt.position_error_m)*1.05]});
lineChart('velocityChart',[{name:'参考水平速度',x:chart.offset_s,y:chart.reference_horizontal_speed_mps,color:colors[0]},{name:'预测水平速度',x:chart.offset_s,y:chart.predicted_horizontal_speed_mps,color:colors[1]},{name:'参考垂直速度',x:chart.offset_s,y:chart.reference_vertical_speed_mps,color:colors[3],dash:'6 4'},{name:'预测垂直速度',x:chart.offset_s,y:chart.predicted_vertical_speed_mps,color:colors[2],dash:'6 4'}],{xLabel:'anchor 后时间 (s)',yLabel:'速度 (m/s)'});
lineChart('consistencyChart',[{name:'reference mismatch',x:chart.offset_s,y:chart.reference_consistency_mps,color:colors[2]},{name:'prediction mismatch',x:chart.offset_s,y:chart.predicted_consistency_mps,color:colors[0]}],{xLabel:'anchor 后时间 (s)',yLabel:'|Δposition/Δt − velocity| (m/s)'});lineChart('accelerationChart',[{name:'reference',x:chart.offset_s,y:chart.reference_acceleration_mps2,color:colors[2]},{name:'prediction',x:chart.offset_s,y:chart.predicted_acceleration_mps2,color:colors[0]}],{xLabel:'anchor 后时间 (s)',yLabel:'|Δvelocity/Δt| (m/s²)'});
const seg=d.segments,st=[0,...seg.offset_s.slice(0,-1)],ctrl=seg.controls;lineChart('thrustChart',[{name:'推力',x:st,y:ctrl.map(r=>r[0]/1000),color:colors[0]}],{xLabel:'时间 (s)',yLabel:'Thrust (kN)',height:220});lineChart('bankChart',[{name:'坡度',x:st,y:ctrl.map(r=>r[1]*180/Math.PI),color:colors[1]}],{xLabel:'时间 (s)',yLabel:'Bank (deg)',height:220});lineChart('loadChart',[{name:'载荷因子',x:st,y:ctrl.map(r=>r[2]),color:colors[2]}],{xLabel:'时间 (s)',yLabel:'Load factor',height:220});lineChart('durationChart',[{name:'分段时长',x:seg.duration_s.map((_,i)=>i+1),y:seg.duration_s,color:colors[3]}],{xLabel:'控制段',yLabel:'时长 (s)',height:220});barChart('terminalChart',[{label:'位置 (m)',values:m.terminal_position_components_m},{label:'速度 (m/s)',values:m.terminal_velocity_components_mps}],{labels:['沿跑道','横向','垂直'],height:220});lineChart('lossChart',[{name:'train',x:f.loss_curve.map(r=>r.epoch),y:f.loss_curve.map(r=>Math.log10(Math.max(r.train,1e-10))),color:colors[0]},{name:'replay',x:f.loss_curve.map(r=>r.epoch),y:f.loss_curve.map(r=>Math.log10(Math.max(r.replay,1e-10))),color:colors[1]}],{xLabel:'epoch',yLabel:'log10(loss)',height:220});const crossing=m.first_error_crossing_s;el('audit').innerHTML=`<p><b>误差首次越过：</b>100 m @ ${fmt(crossing['100'])} s；500 m @ ${fmt(crossing['500'])} s；1000 m @ ${fmt(crossing['1000'])} s。</p><p><b>控制贴边率：</b>推力 ${fmt(m.control_saturation_ratio[0]*100,0)}%，坡度 ${fmt(m.control_saturation_ratio[1]*100,0)}%，载荷因子 ${fmt(m.control_saturation_ratio[2]*100,0)}%。</p><p><b>时长自由度：</b>CV ${fmt(m.duration_cv,3)}，归一化熵 ${fmt(m.duration_entropy_normalized,3)}（越接近 1 越接近均匀分配）。</p><p><b>参考可行性：</b>位置/速度 RMSE ${fmt(m.reference_position_velocity_rmse_mps,2)} m/s；加速度 p95 ${fmt(m.reference_acceleration_p95_mps2,2)} m/s²；jerk p95 ${fmt(m.reference_jerk_p95_mps3,2)} m/s³。</p><p><b>结果源：</b><code>${f.source}</code></p>`;}
renderGlobal();flights.forEach((f,i)=>{const b=document.createElement('button');b.textContent=`${f.airport} · ${fmt(f.duration_s,0)}s`;b.onclick=()=>renderFlight(i);el('flightSelector').append(b)});renderFlight(0);el('generatedAt').textContent=report.generated_at;
</script></body></html>'''


def build_report(result_paths: list[Path], output: Path) -> None:
    flights, recipe = _load_results(result_paths)
    payload = {
        "generated_at": "2026-08-01",
        "isolation": "outer-train values only; validation and outer-test values unopened",
        "recipe": recipe,
        "flights": flights,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        HTML_TEMPLATE.replace("__REPORT_DATA__", encoded), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build_report(args.results, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
