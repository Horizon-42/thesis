"""Where do the official closed loop's angle corrections fall, relative to the observed angle word before them?"""
import numpy as np, sys
d = np.load(sys.argv[1])
W, C, off, vm = d["words"], d["correction"], d["offsets"], d["vertical_m"]
ROW = float(d["row_interval_s"])
ANG, ALT = 3, 2
NLO = 40
since_steepen = []      # seconds from an observed steepening (incl. level->descent) to the first steeper correction after it
first_after = {"steeper": 0, "shallower": 0}
corr_total = {"steeper": 0, "shallower": 0}
corr_within = {10: 0, 20: 0, 30: 0}
n_steepen = 0; n_steepen_corrected = 0
final_entry = []        # first angle correction after the observed 'no level-off'
for s in range(len(off) - 1):
    a, b = off[s], off[s + 1]
    obs_ang = None; last_obs_change_row = None; last_change_steepen = False; answered = True
    for r in range(a, b):
        w = W[r, ANG]
        if w >= 0 and not C[r, ANG]:
            if obs_ang is not None and w != obs_ang:
                steep = (1 <= w <= 4) and (obs_ang == 0 or (1 <= obs_ang <= 4 and w > obs_ang))
                last_obs_change_row, last_change_steepen, answered = r, steep, False
                if steep: n_steepen += 1
            obs_ang = w
        if w >= 0 and C[r, ANG] and obs_ang is not None:
            kind = "steeper" if (1 <= w <= 4 and 1 <= obs_ang <= 4 and w > obs_ang) else "shallower"
            corr_total[kind] += 1
            if last_obs_change_row is not None and not answered:
                answered = True
                dt = (r - last_obs_change_row) * ROW
                if last_change_steepen and kind == "steeper":
                    since_steepen.append(dt); n_steepen_corrected += 1
since = np.array(since_steepen)
print(f"row interval {ROW:g} s, sentences {len(off)-1}")
print("angle corrections:", corr_total)
print(f"observed steepenings (level->descent or to a steeper class): {n_steepen}")
print(f"  followed (before any other observed angle change) by a STEEPER correction: {n_steepen_corrected} ({n_steepen_corrected/n_steepen:.1%})")
for q in (25, 50, 75, 90):
    print(f"  delay to that correction p{q}: {np.percentile(since, q):.0f} s")
for t in (10, 20, 30):
    print(f"  within {t} s: {(since <= t).mean():.1%} of them, {(since <= t).sum()/n_steepen:.1%} of all steepenings")
