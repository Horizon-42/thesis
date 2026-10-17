import json,sys
keys=set(json.load(open(sys.argv[1])))
r=json.load(sys.stdin)
out={t["flight_key"]:{k:t.get(k) for k in ("verdict","lateral_result","vertical_result","speed_result","vertical_m")} for t in r["trajectories"] if t.get("flight_key") in keys}
json.dump(out,sys.stdout)
