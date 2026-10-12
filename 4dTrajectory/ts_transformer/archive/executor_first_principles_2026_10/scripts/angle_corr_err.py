"""e_h at an observed steepening row, and at the rows after, in the official closed loop."""
import numpy as np, sys
d = np.load(sys.argv[1])
W, C, off, vm = d["words"], d["correction"], d["offsets"], d["vertical_m"]
ROW = float(d["row_interval_s"]); ANG = 3
from_level, to_steeper = [], []
for s in range(len(off) - 1):
    a, b = off[s], off[s + 1]
    obs = None
    for r in range(a, b):
        w = W[r, ANG]
        if w >= 0 and not C[r, ANG]:
            if obs is not None and w != obs and 1 <= w <= 4:
                if r + 8 < b:
                    trace = vm[r - 1:r + 8] if r > a else np.r_[np.nan, vm[r:r + 8]]
                    if obs == 0: from_level.append(trace)
                    elif 1 <= obs <= 4 and w > obs: to_steeper.append(trace)
            obs = w
for name, t in (("level -> descent", from_level), ("descent -> steeper descent", to_steeper)):
    t = np.array(t)
    print(f"{name}: {len(t)} events; e_h (m) by row offset from the word row (-1 .. +7), {ROW:g} s rows")
    for q in (10, 50, 90):
        print(f"  p{q}: " + " ".join(f"{v:6.1f}" for v in np.nanpercentile(t, q, axis=0)))
    print("  share |e_h|>15 m: " + " ".join(f"{v:6.2f}" for v in (np.abs(t) > 15).mean(axis=0)))
