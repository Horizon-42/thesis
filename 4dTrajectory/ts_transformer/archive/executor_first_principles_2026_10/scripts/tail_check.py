import sys, math
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import lead_eval_1000 as run, new_design_eval as nde, lead_eval_tail as lt
import numpy as np
from collections import Counter
from ts_transformer.autopilot import replay
from ts_transformer.autopilot import flights as fm
from ts_transformer.repo_layout import arrival_manifest_path
from ts_transformer.instructions.labeller import read as rm, vertical as vm
from ts_transformer.instructions.words import Words
fm.arrival_manifest_path = lambda a: arrival_manifest_path(a, nde.HARVEST)
with np.load(nde.INSTRUCTIONS / "sentences_train.npz") as d: lab=[int(i) for i in d["signal_index"]]
drawn = replay.draw_flights(nde.INSTRUCTIONS, "train", lab, per_airport=200, seed=1337, groups=(replay.OWN, replay.STAND_IN))
spec = nde.spec_of(4,4,4); words=Words(spec)
rowsA=[]; rowsB=[]
for f in drawn.signals[:1000]:
    try: fl = rm.admit(f, drawn.geometries[f.airport], spec)
    except Exception: continue
    h = fl.smoothed.altitude_m - drawn.geometries[f.airport].elevation_m
    tg = fl.relative.before_threshold_m
    pieces = vm.vertical_pieces(fl.smoothed.distance_m, h, spec, words)
    for p in pieces:
        if p.kind==vm.MOVE and tg[p.start] < 300: rowsA.append((round(float(tg[p.start])), p.rows, round(float(tg[-1])), round(p.angle_deg,2)))
    if pieces and pieces[-1].kind==vm.MOVE and pieces[-1].descending(spec):
        first=len(pieces)-1
        while first>0 and pieces[first-1].kind==vm.MOVE and pieces[first-1].descending(spec): first-=1
        r=np.arange(pieces[first].start, pieces[-1].stop); ins=r[(tg[r]>=300)&(tg[r]<=5000)]
        if len(ins)>=8:
            fit=math.degrees(math.atan(np.polyfit(tg[ins],h[ins],1)[0]))
            rowsB.append((fit, pieces[-1].angle_deg, float(tg[pieces[-1].start]), float(tg[pieces[first].start]), len(ins), float(tg[ins].min()), float(tg[ins].max()), f.dataset_id))
print("A pieces", len(rowsA)); print("A to_go at start pct", np.percentile([r[0] for r in rowsA],[0,10,50,90,100]))
print("A rows", Counter(r[1] for r in rowsA).most_common(6)); print("A end to_go pct", np.percentile([r[2] for r in rowsA],[0,10,50,90,100]))
print("A examples", rowsA[:10])
fits=np.array([r[0] for r in rowsB]); print("B fits pct", np.percentile(fits,[1,5,10,50,90,95,99]))
sh=[r for r in rowsB if r[0]<2.0]; print("B fit<2:", len(sh))
for r in sh[:12]: print("  fit %.2f last %.2f last_start_togo %.0f group_start %.0f n %d window %.0f-%.0f %s"%r)
