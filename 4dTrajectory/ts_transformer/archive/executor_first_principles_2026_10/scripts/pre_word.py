"""Before an observed level->descent angle word: how far had the OBSERVED aircraft already sunk? (official closed loop, 2 s)
e_h = flown - observed. Where the flown aircraft is level (|vertical rate| < 0.3 m/s on the rows before), the growth of e_h
before the word row is the observed aircraft's own sink before the word was said."""
import numpy as np, sys
d = np.load(sys.argv[1]); assert float(d["row_interval_s"]) == 2.0
W, C, off, soff, vm, st = d["words"], d["correction"], d["offsets"], d["state_offsets"], d["vertical_m"], d["states"]
start = int(d["start_row"]); ANG = 3; VR = 5
words_cnt = {"level said at row 0": 0, "level said later (observed)": 0, "level said later (correction)": 0}
pre = []
for s in range(len(off) - 1):
    a, b = off[s], off[s + 1]; sa = soff[s]
    obs = None
    for r in range(a, b):
        w = W[r, ANG]
        if w == 0:
            words_cnt["level said at row 0" if r == a else ("level said later (correction)" if C[r, ANG] else "level said later (observed)")] += 1
        if w >= 0 and not C[r, ANG]:
            if obs == 0 and 1 <= w <= 4 and r - a >= 5:
                srow = sa + start + (r - a)
                vr = st[srow - 5:srow, VR]
                if np.all(np.abs(vr) < 0.3):
                    pre.append(vm[r - 5:r + 1] - vm[r - 5])
            obs = w
pre = np.array(pre)
print(words_cnt)
print(f"level->descent words with the flown aircraft level for 10 s before: {len(pre)}")
print("observed sink relative to row -5 (m), rows -5..0 (2 s rows):")
for q in (25, 50, 75, 90):
    print(f"  p{q}: " + " ".join(f"{v:6.1f}" for v in np.percentile(pre, q, axis=0)))
