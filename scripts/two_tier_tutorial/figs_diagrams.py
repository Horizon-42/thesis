"""Paper figures that are diagrams (no data): the architecture, the prior, the decoding, the cross-validation, the
post-training loop, branch training, the traffic features and the judge. English text only, no title, no caption."""
from __future__ import annotations

import math
import sys

import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from figs_common import C, FILL, DOUBLE, SINGLE, anchor, arrow, box, canvas, panel, plt, save, tag


# ------------------------------------------------------------------ Fig 1: the two tiers
def fig01_architecture() -> None:
    fig, ax, H = canvas(DOUBLE, 3.9)
    ax.add_patch(FancyBboxPatch((49.5, 4.2), 50.3, 33.5, boxstyle="round,pad=0,rounding_size=1.2", fc="none", ec=C["word"], lw=0.9, ls=(0, (4, 3)), zorder=1))
    ax.text(51, 35.9, "closed loop", fontsize=7, color=C["word"], fontweight="bold", va="center")
    b = {}
    b["obs"] = box(ax, 1, 41, 17, 9.5, "Observed tracks", FILL["n"], sub="ADS-B, 2 s rows")
    b["lab"] = box(ax, 26, 41, 18, 9.5, "Labeller", FILL["a"], sub="open and closed loop")
    b["sen"] = box(ax, 52, 41, 22, 9.5, "Sentences", FILL["a"], sub="words and flown states")
    b["msk"] = box(ax, 26, 23.5, 18, 10, "Masks", FILL["b"], sub="grammar, procedure, caller")
    b["pri"] = box(ax, 52, 22.5, 22, 11.5, "Prior (speaker)", FILL["b"], bold=True, sub="says one row of words")
    b["exe"] = box(ax, 80, 22.5, 19, 11.5, "Executor", FILL["a"], bold=True, sub="flies only the words")
    b["jdg"] = box(ax, 80, 6.5, 19, 9.5, "Judge", FILL["a"], sub="outcome of the flight")
    b["pst"] = box(ax, 52, 6.5, 22, 9.5, "Post-training", FILL["c"], sub="reward, branch training")
    b["trf"] = box(ax, 26, 6.5, 18, 9.5, "Recorded traffic", FILL["c"], sub="other aircraft")
    for k, (l, x, y) in {"lab": ("A", 26, 50.5), "sen": ("A", 52, 50.5), "msk": ("B", 26, 33.5), "pri": ("B", 52, 34), "exe": ("A", 80, 34), "jdg": ("A", 80, 16), "pst": ("C", 52, 16), "trf": ("C", 26, 16)}.items():
        tag(ax, x + 0.6, y - 0.2, l)
    arrow(ax, anchor(b["obs"], "r"), anchor(b["lab"], "l"))
    arrow(ax, anchor(b["lab"], "r"), anchor(b["sen"], "l"), "sentences")
    arrow(ax, anchor(b["sen"], "b", 0.35), anchor(b["pri"], "t", 0.35), "teacher forcing", lxy=(60.5, 37.5), ha="left")
    arrow(ax, anchor(b["msk"], "r"), anchor(b["pri"], "l"), "blocks words")
    arrow(ax, anchor(b["pri"], "r", 0.7), anchor(b["exe"], "l", 0.7), "rows of words", lxy=(76.5, 31.0))
    arrow(ax, anchor(b["exe"], "b", 0.18), anchor(b["pri"], "b", 0.85), "flown states", rad=-0.33, lxy=(76.5, 18.5))
    arrow(ax, anchor(b["exe"], "b", 0.8), anchor(b["jdg"], "t", 0.8), "flown track", lxy=(94.5, 18.3), ha="left")
    arrow(ax, anchor(b["jdg"], "l"), anchor(b["pst"], "r"), "outcome")
    arrow(ax, anchor(b["pst"], "t", 0.2), anchor(b["pri"], "b", 0.2), "update", lxy=(53.5, 18.3), ha="left")
    arrow(ax, anchor(b["trf"], "r"), anchor(b["pst"], "l"), "traffic", lxy=(48, 11.7))
    for x, l, t in [(3, "A", "stage A: vocabulary, labeller, executor, judge"), (41, "B", "stage B: prior"), (59, "C", "stage C: post-training")]:
        tag(ax, x, 1.7, l); ax.text(x + 1.8, 1.7, t, fontsize=6.5, va="center", color=C["ink2"])
    save(fig, "fig01_architecture")


# ------------------------------------------------------------------ Fig 8: the network of the prior
def fig08_prior_architecture() -> None:
    fig, ax, H = canvas(DOUBLE, 5.5)
    # inputs
    own = box(ax, 1, 46, 25, 8.5, "Own state", FILL["n"], sub="height above E, speed, vertical rate")
    tim = box(ax, 1, 35, 25, 8.5, "Time since each word", FILL["n"], sub="in seconds, for each column")
    wif = box(ax, 1, 24, 25, 8.5, "Words in force", FILL["n"], sub="runway token, G, heading, embeddings")
    pool = box(ax, 1, 13, 25, 10.5, "Candidate pool", FILL["a"], sub="K vectors → one shared network\n→ K tokens → attention (sum 1)")
    ax.add_patch(plt.Circle((33, 33), 1.9, fc="white", ec=C["ink2"], lw=0.9, zorder=4)); ax.text(33, 32.9, "+", ha="center", va="center", fontsize=10, zorder=5)
    for bb in (own, tim, wif, pool): arrow(ax, anchor(bb, "r"), (31.2, 33 + (anchor(bb, "r")[1] - 33) * 0.08), lw=0.8)
    # the layers
    ax.add_patch(FancyBboxPatch((40, 20), 30, 36, boxstyle="round,pad=0,rounding_size=1", fc="white", ec=C["ink2"], lw=0.9, zorder=2))
    ax.text(41.5, 53.8, "× 4 layers (pre-norm, residual)", fontsize=6.5, color=C["ink2"], va="center")
    l1 = box(ax, 42.5, 40, 25, 9, "Causal time attention", FILL["b"], sub="RoPE on the time in seconds")
    l2 = box(ax, 42.5, 29.5, 25, 8.5, "Traffic attention", FILL["c"], sub="reads the other aircraft")
    l3 = box(ax, 42.5, 22, 25, 6, "Feed-forward", FILL["b"])
    arrow(ax, (35, 33), (40, 33)); arrow(ax, anchor(l1, "b"), anchor(l2, "t"), lw=0.7, ms=5); arrow(ax, anchor(l2, "b"), anchor(l3, "t"), lw=0.7, ms=5)
    hb = box(ax, 78, 29, 20, 9, "Hidden state h", FILL["n"], bold=True)
    arrow(ax, (70, 33.5), (78, 33.5))
    # the heads, with a bus
    names = [("Runway", "pointer over the tokens"), ("Heading", "73 classes"), ("Altitude", "42 classes"), ("Angle", "7 classes"), ("Speed", "49 classes")]
    heads = [box(ax, 1 + 19.6 * i, 0.8, 17.2, 7.0, n_, FILL["b"], bold=True, sub=s_, fs=6.8) for i, (n_, s_) in enumerate(names)]
    ax.plot([88, 88, 9.6], [29, 10.8, 10.8], color=C["ink2"], lw=0.9, zorder=2)
    for h in heads: arrow(ax, (h[0] + h[2] / 2, 10.8), (h[0] + h[2] / 2, 7.9), lw=0.9, ms=6)
    for i in range(4): arrow(ax, anchor(heads[i], "r", 0.3), anchor(heads[i + 1], "l", 0.3), lw=1.8, ms=8, color=C["word"])
    ax.text(55, 15.5, "autoregressive over the columns: each head also reads the words that the\nearlier columns chose in this row (purple arrows, in the order of the columns)", ha="center", fontsize=6.4, color=C["word"], va="center")
    ax.text(86.5, 20.5, "masks and a random\nnumber u act on the\nlogits of each head", fontsize=6.4, ha="right", va="center", color=C["word"])
    # autoregressive over the rows: the words said in this row are inputs of the next row
    rail = -3.0
    for h in heads:
        ax.plot([h[0] + h[2] / 2, h[0] + h[2] / 2], [h[1], rail], color=C["word"], lw=0.9, zorder=2)
    ax.plot([heads[0][0] + heads[0][2] / 2, -3.5, -3.5], [rail, rail, 39.2], color=C["word"], lw=0.9, zorder=2)
    ax.plot([heads[-1][0] + heads[-1][2] / 2, heads[0][0] + heads[0][2] / 2], [rail, rail], color=C["word"], lw=0.9, zorder=2)
    for y in (39.2, 28.3):
        arrow(ax, (-3.5, y), (1, y), None, color=C["word"], lw=0.9, ms=6)
    ax.text(50, rail - 1.2, "words said in this row  →  inputs of the next row", fontsize=6.6, color=C["word"], ha="center", va="top", fontweight="bold")
    # the rows in time that the causal attention reads
    ax.text(87, 55.6, "rows in time", fontsize=6.4, color=C["ink2"], ha="center", va="bottom")
    for k, (lab, fc) in enumerate([("t−2", "white"), ("t−1", "white"), ("t", FILL["b"])]):
        box(ax, 75.5 + 8 * k, 48.5, 7, 5, lab, fc, fs=6.0)
        if k < 2: arrow(ax, (82.5 + 8 * k, 51), (83.5 + 8 * k, 51), None, lw=0.8, ms=5)
    ax.text(87, 46.6, "each row reads the rows\nbefore it and itself", fontsize=5.8, color=C["ink2"], ha="center", va="top")
    # the traffic attention in detail: the other aircraft become tokens, and the row attends to them
    ax.add_patch(FancyBboxPatch((1, 58.5), 98, 19, boxstyle="round,pad=0,rounding_size=1", fc="none", ec=C["ink3"], lw=0.8, ls=(0, (4, 3)), zorder=1))
    ax.text(3, 75.2, "Traffic attention, in each layer", fontsize=7, fontweight="bold", va="center", color=C["ink"])
    t1 = box(ax, 5, 62.5, 24, 9.5, "Other aircraft", FILL["n"], sub="edge features and own motion")
    t2 = box(ax, 35, 62.5, 24, 9.5, "Token network", FILL["c"], sub="one token per aircraft")
    t3 = box(ax, 65, 62.5, 24, 9.5, "Attention", FILL["c"], sub="query: the commanded aircraft")
    arrow(ax, anchor(t1, "r"), anchor(t2, "l"), lw=0.9, ms=6); arrow(ax, anchor(t2, "r"), anchor(t3, "l"), "tokens", lxy=(62, 67.6), fs=6.0, lw=0.9, ms=6)
    ax.text(5, 60.4, "The output is added to the row. It is zero at the start, and zero when there is no other aircraft.", fontsize=6.2, color=C["ink2"], va="center")
    ax.plot([72.5, 72.5, 67.6], [62.5, 36.0, 36.0], color=C["ink2"], lw=0.9, ls=(0, (3, 2)), zorder=2)
    arrow(ax, (68.6, 36.0), (67.6, 36.0), lw=0.9, ms=5)
    ax.set_xlim(-7, 100); ax.set_ylim(-8.5, H)
    save(fig, "fig08_prior_architecture")


# ------------------------------------------------------------------ Fig 18: inside the prior: the input, one layer, the heads
def fig18_prior_layers() -> None:
    fig, ax, H = canvas(DOUBLE, 4.7)
    plus = lambda x, y, c=None: (ax.add_patch(plt.Circle((x, y), 1.5, fc="white", ec=c or C["ink2"], lw=0.9, zorder=4)), ax.text(x, y - 0.1, "+", ha="center", va="center", fontsize=8, zorder=5, color=c or C["ink"]))
    line = lambda xs, ys, c=None, lw=0.9, ls="-": ax.plot(xs, ys, color=c or C["ink2"], lw=lw, ls=ls, zorder=2)
    A, B = 43.0, 21.5                        # the lower edges of panels (a) and (b); panel (c) starts at 0
    for y, t in ((A + 22.4, "(a) The input of a row"), (B + 19.4, "(b) One layer (× 4, stacked)"), (18.0, "(c) The five heads, in the order of the columns")):
        ax.text(1, y, t, fontsize=7.2, fontweight="bold", va="center")
    # ---- (a) the input of a row: three inputs are added, then what the row reads of the candidates
    ins = [box(ax, 1 + 22 * i, A + 13.4, 19.5, 7.2, t, FILL["n"], sub=sub, fs=6.6) for i, (t, sub) in enumerate(
        [("Linear", "own state, time since, G, heading"), ("3 embeddings", "altitude, angle, speed words"), ("Linear", "token of the runway in force")])]
    bus = A + 10.8
    for bb in ins:
        cx = bb[0] + bb[2] / 2
        line([cx, cx], [A + 13.4, bus])
    line([ins[0][0] + ins[0][2] / 2, 68.4], [bus, bus])
    plus(70, bus); plus(90, bus)
    arrow(ax, (71.6, bus), (88.4, bus), "x", lxy=(80, bus + 0.8), fs=6.4, lw=0.9, ms=6)
    arrow(ax, (91.6, bus), (99.4, bus), None, lw=0.9, ms=6)
    ax.text(95.5, bus - 1.6, "input of\nlayer 1", ha="center", va="top", fontsize=5.8, color=C["ink2"])
    tok = box(ax, 23, A + 1.0, 29, 6.8, "Token network", FILL["a"], sub="Linear, GELU, Linear; the same for every candidate", fs=6.3)
    pool = box(ax, 58, A + 1.0, 28, 6.8, "Candidate pool", FILL["a"], sub="attention: query = x, keys and values = tokens", fs=6.3)
    ax.text(1, A + 4.4, "candidate vectors", fontsize=6.0, color=C["ink2"], va="center"); arrow(ax, (15.5, A + 4.4), (22.8, A + 4.4), lw=0.8, ms=5)
    arrow(ax, anchor(tok, "r"), anchor(pool, "l"), lw=0.8, ms=5)
    arrow(ax, (80, bus), (80, A + 7.9), None, lw=0.8, ms=5, ls=(0, (2, 2)))
    line([86, 90, 90], [A + 4.4, A + 4.4, bus - 1.7]); arrow(ax, (90, bus - 3.6), (90, bus - 1.7), lw=0.9, ms=5)
    # ---- (b) one layer: the rail is the vector of the row; each group reads a copy and adds to the rail
    rail = B + 1.5
    line([4, 99.4], [rail, rail], lw=1.4)
    ax.text(1, rail, "x", fontsize=6.8, color=C["ink2"], va="center")
    def group(x0, x1, title, boxes, fc, widths=None):
        ax.add_patch(FancyBboxPatch((x0, B + 4.4), x1 - x0, 12.6, boxstyle="round,pad=0,rounding_size=1", fc="none", ec=C["ink3"], lw=0.8, ls=(0, (4, 3)), zorder=1))
        ax.text(x0 + 1.2, B + 15.2, title, fontsize=6.2, color=C["ink2"], va="center", fontweight="bold")
        line([x0 + 1.0, x0 + 1.0], [rail, B + 9.4]); arrow(ax, (x0 + 1.0, B + 9.4), (x0 + 2.2, B + 9.4), lw=0.9, ms=5)
        placed = []
        widths = widths or [(x1 - x0 - 3.4) / len(boxes) - 1.6] * len(boxes)
        bx = x0 + 2.2
        for (t, sub), w in zip(boxes, widths):
            placed.append(box(ax, bx, B + 6.0, w, 6.8, t, fc, sub=sub, fs=6.0)); bx += w + 1.6
        for lo, hi in zip(placed[:-1], placed[1:]): arrow(ax, anchor(lo, "r"), anchor(hi, "l"), lw=0.8, ms=4)
        last = placed[-1]; cx = last[0] + last[2] / 2
        arrow(ax, (cx, B + 6.0), (cx, rail + 1.7), lw=0.9, ms=5); plus(cx, rail)
    group(4, 56, "causal time attention", [("LayerNorm", None), ("Q, K, V", "one Linear"), ("RoPE", "by time in s"), ("Attention", "rows ≤ this row"), ("Linear", None)], FILL["b"])
    group(58, 74, "added module", [("Traffic attention", "no norm")], FILL["c"])
    group(76, 99.4, "feed-forward", [("LayerNorm", None), ("Feed-forward", "Linear, GELU, Linear")], FILL["b"], widths=[7.5, 11.0])
    # ---- (c) the five heads: h feeds all of them; each head passes its word on to the next
    names = ["Runway", "Heading", "Altitude", "Angle", "Speed"]
    xs = [1 + 21.7 * i for i in range(5)]
    box(ax, 1, 10.4, 98.8, 4.6, "h: output of (b), after a final LayerNorm", FILL["n"], fs=6.8)
    for i, n_ in enumerate(names):
        box(ax, xs[i], 0, 11, 7.0, n_, FILL["b"], bold=True, fs=7.2)
        arrow(ax, (xs[i] + 5.5, 10.3), (xs[i] + 5.5, 7.1), lw=0.9, ms=5)
        if i < 4: arrow(ax, (xs[i] + 11, 3.5), (xs[i + 1], 3.5), lw=1.9, ms=7, color=C["word"])
    ax.set_ylim(-0.3, H)
    save(fig, "fig18_prior_layers")




# ------------------------------------------------------------------ Fig 19: the traffic attention
def fig19_traffic_attention() -> None:
    fig, ax, H = canvas(DOUBLE, 2.4)
    plus = lambda x, y: (ax.add_patch(plt.Circle((x, y), 1.5, fc="white", ec=C["ink2"], lw=0.9, zorder=4)), ax.text(x, y - 0.1, "+", ha="center", va="center", fontsize=8, zorder=5))
    ax.text(1, 32.4, "Once for each step: the tokens of the other aircraft", fontsize=7.0, fontweight="bold", va="center")
    ax.text(1, 15.6, "In each layer: one attention from the row to the tokens", fontsize=7.0, fontweight="bold", va="center")
    oth = box(ax, 1, 20.5, 22, 8.0, "Other aircraft", FILL["n"], sub="20 features each", fs=6.6)
    net = box(ax, 28, 20.5, 30, 8.0, "Traffic token network", FILL["c"], sub="Linear, GELU, Linear; run once per step", fs=6.6)
    tok = box(ax, 63, 20.5, 20, 8.0, "Tokens", FILL["c"], sub="N tokens", fs=6.6)
    arrow(ax, anchor(oth, "r"), anchor(net, "l"), lw=0.9, ms=5); arrow(ax, anchor(net, "r"), anchor(tok, "l"), lw=0.9, ms=5)
    ax.text(91.5, 24.5, "N: the other\naircraft of the\nstep (any number)", fontsize=5.8, color=C["ink2"], ha="center", va="center")
    xb = box(ax, 1, 2.0, 11, 7.0, "x", FILL["n"], bold=True, sub="the row", fs=7.0)
    ln = box(ax, 17, 2.0, 17, 7.0, "LayerNorm", FILL["b"], fs=6.6)
    att = box(ax, 40, 2.0, 36, 7.0, "Attention, 4 heads", FILL["c"], sub="query: the row; keys, values: the tokens", fs=6.6)
    lin = box(ax, 81, 2.0, 12, 7.0, "Linear", FILL["c"], sub="starts at 0", fs=6.6)
    arrow(ax, anchor(xb, "r"), anchor(ln, "l"), lw=0.9, ms=5); arrow(ax, anchor(ln, "r"), anchor(att, "l"), lw=0.9, ms=5); arrow(ax, anchor(att, "r"), anchor(lin, "l"), lw=0.9, ms=5)
    arrow(ax, (73, 20.4), (73, 9.1), "keys, values", lxy=(74.2, 13.2), ha="left", fs=6.0, lw=0.9, ms=5)
    plus(97.5, 5.5)
    arrow(ax, anchor(lin, "r"), (95.9, 5.5), lw=0.9, ms=5)
    ax.text(97.5, 1.4, "added\nto x", fontsize=5.6, color=C["ink2"], ha="center", va="top")
    ax.set_ylim(-1.8, H)
    save(fig, "fig19_traffic_attention")


# ------------------------------------------------------------------ Fig 20: one round of stage D, and the loop inside it
def fig20_multi_round() -> None:
    fig, ax, H = canvas(DOUBLE, 4.0)
    kinds = {"data": FILL["n"], "speak": FILL["a"], "learn": FILL["c"]}
    ax.text(1, 54.6, "(a) One round", fontsize=7.2, fontweight="bold", va="center")
    steps = [("1  Windows", "an anchor and the\narrivals after it", "data"),
             ("2  First pass", "run the loop,\njudge each aircraft", "speak"),
             ("3  Reward", "W = sum of the\naircraft's rewards", "learn"),
             ("4  Second pass", "vary one aircraft v,\n8 continuations", "speak"),
             ("5  Advantage", "W less the group mean,\non v's later rows", "learn"),
             ("6  Update", "surrogate, pull,\ndata term", "learn")]
    w, gap, y0, h = 13.6, 3.0, 36.0, 12.0
    boxes = [box(ax, 0.5 + i * (w + gap), y0, w, h, t, kinds[k], sub=sub, fs=6.2) for i, (t, sub, k) in enumerate(steps)]
    for lo, hi in zip(boxes[:-1], boxes[1:]): arrow(ax, anchor(lo, "r"), anchor(hi, "l"), lw=1.0, ms=6)
    cx = boxes[2][0] + w / 2
    arrow(ax, (cx, y0), (cx, 31.0), None, lw=0.9, ms=5, ls=(0, (3, 2)))
    ax.text(cx, 29.6, "W = number of aircraft: no sample", ha="center", va="top", fontsize=5.8, color=C["ink2"])
    gx = (boxes[2][0] + w + boxes[3][0]) / 2
    ax.text(gx, y0 + h + 1.4, "W < number of aircraft", ha="center", fontsize=5.9, color=C["ink2"])
    cx6, cx1 = boxes[5][0] + w / 2, boxes[0][0] + w / 2
    ax.plot([cx6, cx6, cx1], [y0, 33.6, 33.6], color=C["ink2"], lw=0.9, zorder=2)
    arrow(ax, (cx1, 33.6), (cx1, y0), None, lw=0.9, ms=6)
    ax.text(cx6 - 1.0, 34.2, "new weights, next round", ha="right", va="bottom", fontsize=5.9, color=C["ink2"])
    for x, (name, key) in zip((60.0, 69.0, 86.0), (("data", "data"), ("speaking", "speak"), ("learning", "learn"))):
        ax.add_patch(Rectangle((x, 55.3), 2.0, 1.6, fc=kinds[key], ec=C["ink2"], lw=0.7))
        ax.text(x + 2.8, 56.1, name, fontsize=5.9, color=C["ink2"], va="center")
    # (b) the loop that steps 2 and 4 run: one row at a time, all aircraft together
    ax.text(1, 21.5, "(b) Inside steps 2 and 4: one row of the loop", fontsize=7.2, fontweight="bold", va="center")
    loop = [("Join", "each aircraft at its\nown first row", "data"),
            ("Tokens", "one for each\nother aircraft", "data"),
            ("Prior", "speaks for every commanded\naircraft at once", "speak"),
            ("Executor", "flies each aircraft\none row", "speak"),
            ("Separation judge", "a responsible aircraft\nbecomes silent", "speak")]
    w2, gap2, y1, h2 = 17.0, 3.5, 3.5, 12.0
    lb = [box(ax, 0.5 + i * (w2 + gap2), y1, w2, h2, t, kinds[k], sub=sub, fs=6.2) for i, (t, sub, k) in enumerate(loop)]
    for lo, hi in zip(lb[:-1], lb[1:]): arrow(ax, anchor(lo, "r"), anchor(hi, "l"), lw=1.0, ms=6)
    ex = lb[4][0] + w2 / 2
    ax.plot([ex, ex, lb[0][0] + w2 / 2], [y1, 1.0, 1.0], color=C["ink2"], lw=0.9, zorder=2)
    arrow(ax, (lb[0][0] + w2 / 2, 1.0), (lb[0][0] + w2 / 2, y1), None, lw=0.9, ms=6)
    ax.text((ex + lb[0][0] + w2 / 2) / 2, 1.5, "next row, until every commanded aircraft is done or silent", ha="center", va="bottom", fontsize=5.9, color=C["ink2"])
    ax.set_ylim(-0.4, H)
    save(fig, "fig20_multi_round")


# ------------------------------------------------------------------ Fig 21: a random number picks a word
def fig21_random_number() -> None:
    fig, ax, H = canvas(DOUBLE, 2.5)
    words = ["unchanged", "70", "65", "60"]
    sets = [("the situation is the same", [0.55, 0.25, 0.15, 0.05]),
            ("v's words changed it", [0.40, 0.20, 0.30, 0.10])]
    u = 0.62
    x0, x1 = 24.0, 98.0
    ax.text(1, 36.0, "The words of the speed column (in m/s) are laid on a line from 0 to 1. Each word gets a piece as long as its probability.", fontsize=6.2, color=C["ink2"], va="center")
    for i, (title, p) in enumerate(sets):
        y = 25.0 - i * 13.0
        ax.text(x0 - 1.5, y, title, fontsize=6.4, ha="right", va="center", fontweight="bold")
        edge = 0.0
        for word, q in zip(words, p):
            lo, hi = x0 + (x1 - x0) * edge, x0 + (x1 - x0) * (edge + q)
            chosen = edge <= u < edge + q
            ax.add_patch(Rectangle((lo, y - 2.6), hi - lo, 5.2, fc=FILL["b"] if chosen else "white", ec=C["ink2"], lw=0.9, zorder=2))
            ax.text((lo + hi) / 2, y, word, fontsize=5.8, ha="center", va="center", fontweight="bold" if chosen else "normal", zorder=3)
            edge += q
    ux = x0 + (x1 - x0) * u
    ax.plot([ux, ux], [8.4, 30.6], color=C["word"], lw=1.5, zorder=5)
    ax.text(ux + 1.0, 31.8, "the random number u = 0.62", fontsize=6.4, color=C["word"], fontweight="bold", va="center")
    for v in (0.0, 1.0):
        ax.text(x0 + (x1 - x0) * v, 7.2, f"{v:g}", fontsize=5.8, color=C["ink2"], ha="center", va="center")
    ax.text(1, 2.6, "The word whose piece holds u is said. The same u gives the same word while the pieces are the same; when the pieces change, the same u can fall in another word.", fontsize=5.8, color=C["ink2"], va="center")
    ax.set_xlim(0, 100); ax.set_ylim(0.8, 38.0)
    save(fig, "fig21_random_number")


# ------------------------------------------------------------------ Fig 22: how the sentences of one window branch and affect each other
def fig22_window_branches() -> None:
    fig, ax, H = canvas(DOUBLE, 4.6)
    T, D = 40.0, 30.0                              # a panel: time 0..T, distance to the threshold D..0
    pw, ph, y_base = 28.5, 28.0, 19.0
    x0s = [5.0, 37.5, 70.0]
    def P(k, t, d): return x0s[k] + pw * t / T, y_base + ph * d / D
    def line(k, pts, **kw): xy = [P(k, t, d) for t, d in pts]; ax.plot([q[0] for q in xy], [q[1] for q in xy], **kw)
    tb = 10.0
    pre = {"A2": [(0, 28), (10, 17)], "v": [(3, 30), (10, 20)], "A3": [(6, 30), (10, 24)]}
    post1 = {"A2": [(10, 17), (22, 0)], "v": [(10, 20), (24, 0)], "A3": [(10, 24), (28, 0)]}
    post2 = {"A2": [(10, 17), (22, 0)], "v": [(10, 20), (30, 0)], "A3": [(10, 24), (38, 0)]}
    post3 = {"A2": [(10, 17), (22, 0)], "v": [(10, 20), (20, 0)], "A3": [(10, 24), (22, 0)]}
    panels = [("first sentence", "all numbers drawn first", post1, None, [("v", 16.0, 11.4)],
               "loss: v too close behind aircraft 2"),
              ("continuation 1: v slows down", "v: new numbers   2, 3: the same numbers", post2, post1, [],
               "no loss: all three land"),
              ("continuation 2: v speeds up", "v: new numbers   2, 3: the same numbers", post3, post1, [("v", 15.2, 9.6), ("A3", 20.0, 4.0)],
               "losses: v overtakes aircraft 2;\naircraft 3 too close behind it")]
    colours = {"A2": C["flown"], "A3": C["flown"], "v": C["corr"]}
    for k, (title, numbers, post, ghost, losses, events) in enumerate(panels):
        ax.text(x0s[k] + pw / 2, 54.6, title, fontsize=6.8, fontweight="bold", ha="center", va="center")
        ax.text(x0s[k] + pw / 2, 52.0, numbers, fontsize=5.7, color=C["ink2"], ha="center", va="center")
        line(k, [(0, 0), (T, 0)], color=C["ink"], lw=1.0, zorder=1)
        line(k, [(0, 0), (0, D)], color=C["ink3"], lw=0.7, zorder=1)
        line(k, [(tb, 0), (tb, D)], color=C["ink2"], lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(*P(k, T, 0.0), "", fontsize=5)
        ax.text(x0s[k] + pw, y_base - 1.0, "time \u2192", fontsize=5.6, color=C["ink2"], ha="right", va="top")
        if k == 0:
            ax.text(*P(k, tb + 0.6, D - 0.2), "branch point", fontsize=5.6, color=C["ink2"], va="top")
            ax.text(x0s[k], y_base - 1.0, "threshold", fontsize=5.6, color=C["ink2"], ha="left", va="top")
            ax.text(x0s[k] - 1.2, y_base + ph / 2, "distance to the threshold", fontsize=5.6, color=C["ink2"], rotation=90, ha="right", va="center")
        if ghost:
            for name, pts in ghost.items(): line(k, pts, color=C["faint"], lw=1.0, zorder=2)
        for name in ("A2", "v", "A3"):
            line(k, pre[name], color=C["ink3"], lw=1.1, zorder=3)
            pts = post[name]
            loss = [l for l in losses if l[0] == name]
            colour = colours[name]; lw = 2.4 if name == "v" else 1.3
            if loss:
                t_l, d_l = loss[0][1], loss[0][2]
                line(k, [pts[0], (t_l, d_l)], color=colour, lw=lw, zorder=4, solid_capstyle="butt")
                line(k, [(t_l, d_l), pts[-1]], color=colour, lw=1.1, ls=(0, (1, 2)), zorder=4)
                ax.plot(*P(k, t_l, d_l), "x", color=C["block"], ms=5.5, mew=1.9, zorder=6)
            else:
                line(k, pts, color=colour, lw=lw, zorder=4, solid_capstyle="butt")
                ax.plot(*P(k, *pts[-1]), "o", color=C["ok"], ms=3.8, zorder=6)
        ax.text(x0s[k] + pw / 2, 14.4, events, fontsize=5.7, color=C["block"] if "loss" in events and "no loss" not in events else C["ok"], ha="center", va="center", linespacing=1.15)
    # the gap between v and the aircraft ahead, at one time: too small in the first sentence, enough in continuation 1
    for k, (d_front, d_back, colour) in ((0, (8.5, 11.4, C["block"])), (1, (8.5, 14.0, C["ok"]))):
        a_, b_ = P(k, 16.0, d_front), P(k, 16.0, d_back)
        ax.annotate("", xy=b_, xytext=a_, arrowprops=dict(arrowstyle="<->", color=colour, lw=1.0, shrinkA=0, shrinkB=0))
    # who is who
    ax.text(*P(0, 5.0, 30.6), "v", fontsize=6.8, color=C["corr"], fontweight="bold", ha="center", va="bottom")
    ax.text(*P(0, 8.0, 30.4), "aircraft 3", fontsize=5.8, color=C["flown"], ha="left", va="bottom")
    ax.text(*P(0, 0.4, 28.6), "aircraft 2", fontsize=5.8, color=C["flown"], ha="left", va="bottom")
    # aircraft 3 answers v
    for k, (t, d_v, d_3) in ((1, (13.0, 17.4, 20.4)), (2, (13.0, 14.0, 20.6))):
        arrow(ax, P(k, t, d_v), P(k, t + 0.3, d_3 - 1.4), None, lw=0.8, ms=5, color=C["ink2"], ls=(0, (2, 2)))
    ax.text(*P(1, 20.5, 25.0), "aircraft 3 slows down:\nit answers v", fontsize=5.7, color=C["flown"], ha="left", va="center")
    ax.text(*P(2, 24.0, 22.0), "aircraft 3 speeds up:\nit answers v", fontsize=5.7, color=C["flown"], ha="left", va="center")
    ax.text(*P(1, 26.5, 17.0), "grey: the\nfirst sentence", fontsize=5.6, color=C["ink3"], ha="left", va="center")
    for k, (W, adv) in enumerate(((2, 0), (3, 1), (1, -1))):
        x = x0s[k] + pw / 2
        ax.text(x, 10.6, f"W = {W}", fontsize=7.0, fontweight="bold", ha="center", va="center")
        ax.text(x, 7.4, "advantage " + (f"{adv:+d}".replace("-", "\u2212") if adv else "0"), fontsize=6.6, fontweight="bold", color=C["corr"], ha="center", va="center")
    ky = 2.6
    ax.plot([1, 5], [ky, ky], color=C["corr"], lw=2.4); ax.text(6.0, ky, "v, after the branch point", fontsize=5.6, color=C["ink2"], va="center")
    ax.plot([26, 30], [ky, ky], color=C["flown"], lw=1.3); ax.text(31.0, ky, "the other commanded aircraft", fontsize=5.6, color=C["ink2"], va="center")
    ax.plot([55, 59], [ky, ky], color=C["corr"], lw=1.1, ls=(0, (1, 2))); ax.text(60.0, ky, "silent after a loss", fontsize=5.6, color=C["ink2"], va="center")
    ax.plot([76], [ky], "x", color=C["block"], ms=5, mew=1.8); ax.text(77.5, ky, "loss", fontsize=5.6, color=C["ink2"], va="center")
    ax.plot([85], [ky], "o", color=C["ok"], ms=3.8); ax.text(86.5, ky, "landed", fontsize=5.6, color=C["ink2"], va="center")
    ky2 = -0.4
    ax.annotate("", xy=(5.0, ky2 + 1.0), xytext=(5.0, ky2 - 1.0), arrowprops=dict(arrowstyle="<->", color=C["block"], lw=1.0)); ax.text(6.5, ky2, "gap to the aircraft ahead: too small", fontsize=5.6, color=C["ink2"], va="center")
    ax.annotate("", xy=(37.0, ky2 + 1.0), xytext=(37.0, ky2 - 1.0), arrowprops=dict(arrowstyle="<->", color=C["ok"], lw=1.0)); ax.text(38.5, ky2, "gap: enough", fontsize=5.6, color=C["ink2"], va="center")
    ax.plot([55, 59], [ky2, ky2], color=C["ink2"], lw=0.8, ls=(0, (2, 2))); ax.text(60.0, ky2, "aircraft 3 answers v", fontsize=5.6, color=C["ink2"], va="center")
    ax.set_xlim(0, 100); ax.set_ylim(-2.4, 57.0)
    save(fig, "fig22_window_branches")


# ------------------------------------------------------------------ Fig 23: the tree of one window's sentences
def fig23_window_tree() -> None:
    fig, ax, H = canvas(DOUBLE, 5.3)
    def elbow(x1, y1, x2, y2, **kw):
        xm = (x1 + x2) / 2
        ax.plot([x1, xm, xm, x2], [y1, y1, y2, y2], color=kw.get("color", C["ink2"]), lw=kw.get("lw", 1.0), ls=kw.get("ls", "-"), zorder=1)
    def node(x, y, w, h, title, sub=None, fc="white", dashed=False, fs=6.4):
        return box(ax, x, y - h / 2, w, h, title, fc, sub=sub, fs=fs, dashed=dashed, ec=C["ink3"] if dashed else None)
    def dot(x, y, kind, ring=False):
        if ring: ax.add_patch(plt.Circle((x, y), 1.55, fc="none", ec=C["flown"], lw=1.4, zorder=5))
        if kind == "ok": ax.plot([x], [y], "o", color=C["ok"], ms=3.8, zorder=6)
        else: ax.plot([x], [y], "x", color=C["block"], ms=4.6, mew=1.7, zorder=6)
    # column heads
    for x, t in ((1.0, "1  a window"), (20.0, "2  who is varied"), (43.0, "3  branch point"), (64.0, "4  the sentences of the group")):
        ax.text(x, 74.0, t, fontsize=6.8, fontweight="bold", va="center")
    # column 1: the window and its first sentence
    ax.add_patch(FancyBboxPatch((1, 29), 15.5, 17.5, boxstyle="round,pad=0,rounding_size=0.8", fc=FILL["n"], ec=C["ink2"], lw=0.9, zorder=3))
    ax.text(8.75, 43.6, "first sentence", fontsize=6.4, ha="center", va="center", fontweight="bold", zorder=4)
    for i, (name, kind) in enumerate((("v1", "x"), ("v2", "x"), ("v3", "ok"))):
        ax.text(4.2 + i * 4.6, 39.6, name, fontsize=5.8, ha="center", va="center", color=C["ink2"], zorder=4)
        dot(4.2 + i * 4.6, 36.6, kind)
    ax.text(8.75, 32.4, "W = 1", fontsize=6.6, ha="center", va="center", fontweight="bold", zorder=4)
    ax.text(1, 26.0, "three commanded\naircraft; v3 landed\n(r = 1): not varied", fontsize=5.6, color=C["ink2"], va="top")
    # column 2: one varied aircraft at a time
    n1 = node(20, 46, 17.5, 7.0, "vary v1", "r = 0: it lost")
    n2 = node(20, 10, 17.5, 7.0, "vary v2", "r = 0: it failed")
    root_r = (16.5, 37.75)
    elbow(root_r[0], root_r[1], 20, 46); elbow(root_r[0], root_r[1], 20, 10)
    # column 3: the branch points of v1
    b = [node(43, y, 15.5, 7.0, t, s) for y, (t, s) in zip((68, 46, 26), (("b1", "first predicted step"), ("b2", "120 s later"), ("b3", "240 s later")))]
    for y in (68, 46, 26): elbow(37.5, 46, 43, y)
    stub2 = node(43, 10, 15.5, 7.0, "3 branch points", "of v2, the same way", dashed=True)
    elbow(37.5, 10, 43, 10, ls=(0, (3, 2)), color=C["ink3"])
    # column 4: b2 expanded: the group = the first sentence + the continuations
    ax.text(79.0, 62.4, "v1", fontsize=5.8, color=C["corr"], fontweight="bold", ha="center", va="center")
    ax.text(83.3, 62.4, "v2", fontsize=5.8, color=C["ink2"], ha="center", va="center")
    ax.text(87.6, 62.4, "v3", fontsize=5.8, color=C["ink2"], ha="center", va="center")
    ax.text(92.0, 62.4, "W", fontsize=5.8, color=C["ink2"], ha="center", va="center")
    ax.text(96.6, 62.4, "advantage", fontsize=5.4, color=C["corr"], ha="center", va="center")
    leaves = [("first sentence", ("x", "x", "ok"), (False, False, False), 1, "−0.25"),
              ("continuation 1", ("ok", "x", "ok"), (False, False, False), 2, "+0.75"),
              ("continuation 2", ("x", "ok", "ok"), (False, True, False), 2, "+0.75"),
              ("continuation 3", ("x", "x", "x"), (False, False, True), 0, "−1.25")]
    for i, (label, res, rings, W, adv) in enumerate(leaves):
        y = 57.0 - i * 8.0
        node(64, y, 33.5, 6.6, "", fc=FILL["n"] if i == 0 else "white")
        ax.text(65.4, y, label, fontsize=5.9, va="center", color=C["ink"], zorder=4)
        for j, kind in enumerate(res): dot(79.0 + j * 4.3, y, kind, ring=rings[j])
        ax.text(92.0, y, str(W), fontsize=6.4, ha="center", va="center", fontweight="bold", zorder=4)
        ax.text(96.6, y, adv, fontsize=6.0, ha="center", va="center", color=C["corr"], fontweight="bold", zorder=4)
        elbow(58.5, 46, 64, y)
    ax.text(64.4, 62.4, "mean of W: 1.25", fontsize=5.8, color=C["ink2"], va="center")
    # the other branch points of v1 are groups of their own
    for y, t in ((68, "8 continuations"), (26, "8 continuations")):
        node(64, y, 33.5, 6.6, t, dashed=True, fs=5.9)
        elbow(58.5, y, 64, y, ls=(0, (3, 2)), color=C["ink3"])
    # notes under the tree
    ax.text(1, 3.0, "Down a branch: the state of the whole window is copied at the branch point, and only the varied aircraft draws new random numbers.\nThe other aircraft keep theirs, but their situation changed, so they can answer (ring): v2 in continuation 2 lands, v3 in continuation 3 fails.", fontsize=5.8, color=C["ink2"], va="center", linespacing=1.4)
    ax.add_patch(plt.Circle((52.5, 17.4), 0.0, fc="none"))
    ax.set_xlim(0, 100); ax.set_ylim(0.5, 75.7)
    save(fig, "fig23_window_tree")


# ------------------------------------------------------------------ Fig 10: one column of the speaker
def fig10_decoding() -> None:
    fig = plt.figure(figsize=(DOUBLE, 2.7))
    words = ["unchanged", "600 m", "540 m", "480 m", "420 m", "no level-off", "900 m", "300 m"]
    p = np.array([0.80, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.02]); blocked = {4, 7}
    ok = np.array([i not in blocked for i in range(len(p))]); q = np.where(ok, p, 0.0); q = q / q.sum(); u = 0.86
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.38)
    ax = fig.add_subplot(gs[0]); x = np.arange(len(p))
    ax.bar(x, p, color=[C["block"] if i in blocked else C["faint"] for i in x], width=0.7)
    for i in blocked: ax.text(i, p[i] + 0.02, "×", ha="center", color=C["block"], fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(words, rotation=60, ha="right"); ax.set_ylabel("probability"); ax.set_ylim(0, 0.9); ax.grid(axis="x", visible=False)
    ax.text(0.97, 0.93, "model", transform=ax.transAxes, ha="right", fontsize=7, color=C["ink2"]); panel(ax, "a", dx=-0.2)
    ax = fig.add_subplot(gs[1])
    ax.bar(x, q, color=C["flown"], width=0.7); ax.set_xticks(x); ax.set_xticklabels(words, rotation=60, ha="right"); ax.set_ylim(0, 0.9); ax.grid(axis="x", visible=False)
    ax.text(0.97, 0.93, "after the masks", transform=ax.transAxes, ha="right", fontsize=7, color=C["ink2"]); panel(ax, "b", dx=-0.2)
    ax = fig.add_subplot(gs[2]); cum = np.concatenate(([0], np.cumsum(q))); pick = int(np.searchsorted(cum, u, side="right") - 1)
    cols = [C["flown"], "#6da7ec"]
    for i in range(len(q)):
        if q[i] > 0:
            ax.barh(0, q[i], left=cum[i], height=0.5, color=C["word"] if i == pick else cols[i % 2], ec="white", lw=0.8)
            if i == 0: ax.text(cum[i] + q[i] / 2, 0, "unchanged", ha="center", va="center", fontsize=6.8, color="white")
    ax.annotate("", xy=(u, -0.3), xytext=(u, -0.78), arrowprops=dict(arrowstyle="-|>", color=C["ink"], lw=1.0))
    ax.text(u, -0.9, f"u = {u}", ha="center", va="top", fontsize=7)
    ax.text(cum[pick] + 0.5 * q[pick], 0.45, f"says “{words[pick]}”", ha="right", fontsize=7, color=C["word"], fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-1.3, 1.0); ax.set_yticks([]); ax.set_xlabel("cumulative probability"); ax.grid(False); ax.spines["left"].set_visible(False)
    panel(ax, "c", dx=-0.08)
    save(fig, "fig10_decoding")


# ------------------------------------------------------------------ Fig 11: cross-validation by airport
def fig11_cv_design() -> None:
    fig = plt.figure(figsize=(DOUBLE, 2.9)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.15)
    ax = fig.add_subplot(gs[0]); air = ["KMSY", "KRDU", "KSJC", "KSMF", "KSTL"]
    for i in range(5):
        for j in range(5):
            held = i == j
            ax.add_patch(Rectangle((j, 4 - i), 0.92, 0.92, fc=C["corr"] if held else C["flown"], ec="none", alpha=1.0 if held else 0.35))
            ax.text(j + 0.46, 4 - i + 0.46, "read" if held else "train", ha="center", va="center", fontsize=6.3, color="white" if held else C["ink"])
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_xticks(np.arange(5) + 0.46); ax.set_xticklabels(air); ax.set_yticks(np.arange(5) + 0.46); ax.set_yticklabels([f"fold {i + 1}" for i in range(5)][::-1])
    ax.xaxis.tick_top(); ax.grid(False); ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xlabel("airports", labelpad=4); ax.xaxis.set_label_position("top")
    panel(ax, "a", dx=-0.2, dy=1.12)
    ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 100); ax.set_ylim(-3, 60); ax.axis("off")
    steps = [("1", "4 configurations × 5 folds", "20 runs", "variant “full”: choose the configuration with the fewest parameters within 2 seed scales of the best"),
             ("2", "configuration A, second seed", "5 runs", "gives the seed scale"),
             ("3", "variant “constants”, chosen configuration", "5 runs", "chosen only if better than “full” by more than 2 seed scales"),
             ("4", "the base on all five airports", "1 run", "stopped on the select days; the val days are read one time")]
    for k, (n, t1, t2, t3) in enumerate(steps):
        y = 47 - 12.5 * k
        ax.add_patch(plt.Circle((4, y + 4.5), 2.8, fc=C["word"], ec="none")); ax.text(4, y + 4.4, n, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        ax.text(10, y + 7, t1, fontsize=7.4, fontweight="bold", va="center"); ax.text(99, y + 7, t2, fontsize=7.4, va="center", ha="right", color=C["ink2"])
        ax.text(10, y + 2, t3, fontsize=6.4, va="center", color=C["ink2"])
    ax.text(10, -1.5, "31 training runs in one campaign; the score of a run is the mean per-step loss on the held-out airport", fontsize=6.4, color=C["ink2"])
    panel(ax, "b", dx=-0.02, dy=1.0)
    save(fig, "fig11_cv_design")


# ------------------------------------------------------------------ Fig 12: the post-training loop
def _post_training_loop(name, method, msub, usub) -> None:
    """The loop of one post-training method: the top row is the same for both methods; the bottom row is the method."""
    fig, ax, H = canvas(DOUBLE, 2.9)
    oy = 3.0
    win = box(ax, 1, oy + 26, 15, 10, "Window", FILL["n"], sub="recorded traffic", fs=6.8)
    pri = box(ax, 20, oy + 26, 22, 10, "Prior + traffic attention", FILL["b"], bold=True, sub="speaks under its masks", fs=6.6)
    exe = box(ax, 46, oy + 26, 15, 10, "Executor", FILL["a"], sub="flies the words", fs=6.8)
    jdg = box(ax, 65, oy + 26, 18, 10, "Judges", FILL["a"], sub="outcome, separation", fs=6.8)
    rew = box(ax, 87, oy + 26, 12, 10, "Reward", FILL["c"], sub="1, 0.9ⁿ, 0", fs=6.8)
    arrow(ax, anchor(win, "r"), anchor(pri, "l"), None)
    arrow(ax, anchor(pri, "r"), anchor(exe, "l"), None)
    arrow(ax, anchor(exe, "r"), anchor(jdg, "l"), None)
    arrow(ax, anchor(jdg, "r"), anchor(rew, "l"), None)
    met = box(ax, 40, oy + 5, 26, 12, method, FILL["c"], sub=msub, fs=6.8)
    upd = box(ax, 72, oy + 5, 17, 12, "Update of the prior", FILL["c"], bold=True, sub=usub, fs=6.4)
    bas = box(ax, 92, oy + 5, 7.5, 12, "Base", FILL["b"], sub="frozen", fs=6.4)
    ax.plot([93, 93, 53], [oy + 26, oy + 21.5, oy + 21.5], color=C["ink2"], lw=0.9, zorder=2)
    arrow(ax, (53, oy + 21.5), anchor(met, "t"), None)
    ax.text(73, oy + 22.3, "reward", fontsize=5.6, color=C["ink2"], ha="center", va="bottom")
    arrow(ax, anchor(met, "r"), anchor(upd, "l"), "samples", lxy=(69, oy + 11.3), fs=5.6)
    arrow(ax, anchor(bas, "l"), anchor(upd, "r"), None)
    ax.plot([80.5, 80.5, 31, 31], [oy + 5, oy + 1.8, oy + 1.8, oy + 26], color=C["ink2"], lw=0.9, zorder=2)
    arrow(ax, (31, oy + 24.5), (31, oy + 26), None)
    ax.text(55, oy + 2.4, "new weights", fontsize=5.8, color=C["ink2"], ha="center", va="bottom")
    ax.text(50, 1.0, "Data term: teacher forcing on the landed sentences of the train days. The other aircraft of a window fly their records.",
            ha="center", fontsize=5.9, color=C["ink2"], va="center")
    save(fig, name)


def fig12_post_training_loop() -> None:
    _post_training_loop("fig12_post_training_loop", "Branch training", "8 copies at each branch point;\nadvantage inside each group",
                        "clipped ratio\n+ 0.04 × pull + data term")


def fig17_landed_training() -> None:
    _post_training_loop("fig17_landed_training", "Keep the landed sentence", "each window spoken several times;\nthe landing of highest reward kept",
                        "log-likelihood\n+ 0.04 × pull + data term")


# ------------------------------------------------------------------ Fig 13: branch training
def fig13_branch_training() -> None:
    rng = np.random.default_rng(5)
    t_event, every, K, q0 = 380.0, 120.0, 8, 0.16
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.9))
    branch = [0.0, 120.0, 240.0]
    y = 0.0; yticks, ylabs = [], []
    for tb in branch:
        q = q0 * min(1.0, (t_event - tb) / 250.0)
        conts = [(rng.random() < q) for _ in range(K)]
        conts[3 if tb == 0 else 5] = True if tb < 240 else conts[3]   # keep one landing continuation in the first groups, as an example
        rewards = [0.0] + [1.0 if c else 0.0 for c in conts]
        mean = float(np.mean(rewards))
        ax.hlines(y, tb, t_event, color=C["word"], lw=3.2, zorder=3)
        for i, c in enumerate(conts):
            ax.hlines(y - 0.55 - 0.5 * i, tb, t_event - (25 if c else 0), color=C["ok"] if c else C["block"], lw=2.0)
        ax.vlines(tb, y + 0.35, y - 0.55 - 0.5 * (K - 1) - 0.2, color=C["ink"], lw=0.8)
        ax.text(tb + 3, y + 0.55, f"branch point {tb:.0f} s", fontsize=6.6, color=C["ink"], va="bottom")
        ax.text(t_event + 8, y, f"first sentence: A = {0 - mean:+.2f}", fontsize=6.4, va="center", color=C["word"])
        ax.text(t_event + 8, y - 1.55, f"landing continuation: A = {1 - mean:+.2f}", fontsize=6.4, va="center", color=C["ok"])
        ax.text(t_event + 8, y - 2.55, f"failing continuation: A = {0 - mean:+.2f}", fontsize=6.4, va="center", color=C["block"])
        y -= 6.6
    ax.axvline(t_event, color=C["block"], lw=0.9, ls=(0, (4, 3))); ax.text(t_event, 1.5, "event that ended the first sentence", ha="right", fontsize=6.6, color=C["block"], va="bottom", rotation=0)
    ax.set_xlim(-10, 560); ax.set_ylim(-19.2, 2.4); ax.set_yticks([]); ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
    ax.set_xlabel("time since the first predicted step (s)")
    ax.plot([], [], color=C["word"], lw=3.0, label="first sentence (counted after the branch point)"); ax.plot([], [], color=C["ok"], lw=2.0, label="continuation that lands"); ax.plot([], [], color=C["block"], lw=2.0, label="continuation that fails")
    ax.legend(loc="upper center", fontsize=6.4, ncol=3, bbox_to_anchor=(0.5, -0.17))
    ax.set_xticks([0, 120, 240, 360])
    save(fig, "fig13_branch_training")


# ------------------------------------------------------------------ Fig 14: the traffic features
def fig14_traffic_features() -> None:
    fig, ax, H = canvas(DOUBLE, 3.4)
    ax.text(1, H - 2, "(a)", fontweight="bold", fontsize=8, va="top"); ax.text(52, H - 2, "(b)", fontweight="bold", fontsize=8, va="top")
    # (a) the approach clock and the speed-word mask: a final from left (far) to right (threshold)
    y0 = 20.0
    ax.plot([2, 47], [y0, y0], color=C["ink3"], lw=1.0, ls=(0, (4, 3)))
    ax.add_patch(Rectangle((45.5, y0 - 1.0), 3.0, 2.0, fc=C["ink"], ec="none")); ax.text(47, y0 - 4.3, "threshold", ha="center", fontsize=6.4, color=C["ink2"])
    def plane(x, col, lab, up=True):
        ax.add_patch(Polygon([[x + 2, y0], [x - 1.4, y0 + 1.6], [x - 1.4, y0 - 1.6]], fc=col, ec="none", zorder=4)); ax.text(x, y0 + 4.5, lab, ha="center", fontsize=6.6, color=col, fontweight="bold")
    plane(34, C["obs"], "leader"); plane(10, C["flown"], "commanded")
    ax.annotate("", xy=(34, y0 + 9.5), xytext=(10, y0 + 9.5), arrowprops=dict(arrowstyle="<->", color=C["ink2"], lw=0.8))
    ax.text(22, y0 + 10.4, "gap on the approach clock", ha="center", fontsize=6.6, color=C["ink2"])
    ax.annotate("", xy=(10 + 14, y0 - 8.5), xytext=(10, y0 - 8.5), arrowprops=dict(arrowstyle="<->", color=C["corr"], lw=0.8))
    ax.text(17, y0 - 11.2, "distance that the rules require", ha="center", fontsize=6.4, color=C["corr"])
    ax.annotate("", xy=(34 + 8, y0 - 1.5), xytext=(34, y0 - 1.5), arrowprops=dict(arrowstyle="-|>", color=C["obs"], lw=1.0)); ax.text(38, y0 - 4.3, "own speed", fontsize=6.2, color=C["obs"], ha="center")
    ax.annotate("", xy=(10 + 9, y0 - 1.5), xytext=(10, y0 - 1.5), arrowprops=dict(arrowstyle="-|>", color=C["word"], lw=1.0, ls=(0, (3, 2)))); ax.text(15, y0 - 4.3, "each speed word", fontsize=6.2, color=C["word"], ha="center")
    ax.text(25, 3.2, "The speed-word mask predicts both aircraft to the time when the leader crosses its\nthreshold, and blocks a speed word whose predicted gap is less than the required distance.", ha="center", fontsize=6.2, color=C["ink2"], va="center")
    # (b) the relative geometry of another aircraft
    cx, cy = 60.0, 18.0
    ax.add_patch(Polygon([[cx + 2.4, cy], [cx - 1.6, cy + 1.8], [cx - 1.6, cy - 1.8]], fc=C["flown"], ec="none", zorder=4)); ax.text(cx, cy - 4.5, "commanded", fontsize=6.6, color=C["flown"], fontweight="bold", ha="center")
    arrow(ax, (cx, cy), (cx + 20, cy), "front", color=C["flown"], lxy=(cx + 18, cy + 0.8), lc=C["flown"], lw=1.1)
    arrow(ax, (cx, cy), (cx, cy + 22), "left", color=C["flown"], lxy=(cx + 1.2, cy + 20.5), ha="left", lc=C["flown"], lw=1.1)
    ox, oy = cx + 32, cy + 20
    ax.add_patch(Polygon([[ox - 1.8, oy + 0.3], [ox + 1.4, oy + 1.9], [ox + 1.4, oy - 1.5]], fc=C["corr"], ec="none", zorder=4)); ax.text(ox, oy + 4.2, "other aircraft", fontsize=6.6, color=C["corr"], fontweight="bold", ha="center")
    arrow(ax, (ox, oy), (ox - 10, oy - 9), "own motion", color=C["corr"], lxy=(ox - 5, oy - 14.5), lc=C["corr"], lw=1.0)
    arrow(ax, (cx + 2, cy + 1), (ox - 2.2, oy - 0.8), None, color=C["ink3"], ls=(0, (2, 2)), lw=0.7)
    ax.text(cx + 14, cy + 15, "closing rate", fontsize=6.2, color=C["ink2"], rotation=32, ha="center")
    ax.text(97, 5.5, "closest point of approach when both fly straight on:\ntime, horizontal distance, height difference", fontsize=6.0, color=C["ink"], va="center", ha="right")
    save(fig, "fig14_traffic_features")


# ------------------------------------------------------------------ Fig 16: the four kinds of window
def fig16_window_kinds() -> None:
    fig, ax, H = canvas(DOUBLE, 3.2)
    y0 = 29.0

    def plane(x, y, col, label=None, hollow=False, size=1.0):
        ax.add_patch(Polygon([[x + 2 * size, y], [x - 1.4 * size, y + 1.6 * size], [x - 1.4 * size, y - 1.6 * size]],
                             fc="white" if hollow else col, ec=col, lw=1.0 if hollow else 0, ls=(0, (2, 1.5)) if hollow else "-", zorder=4))
        if label:
            ax.text(x, y + 4.2 * size, label, ha="center", fontsize=6.0, color=col, fontweight="bold")

    panels = [("(a) Real", ["the recorded traffic", "of the commanded", "flight"]),
              ("(b) A: inserted", ["one more flight, shifted", "to land within 180 s of", "the commanded one"]),
              ("(c) D: leader moved", ["the aircraft ahead is", "shifted in time by", "−120 s to +120 s"]),
              ("(d) B: start moved", ["the commanded aircraft's", "turn ±15°, height ±300 m,", "speed × (1 ± 0.1)"])]
    for i, (title, lines) in enumerate(panels):
        x = 1 + i * 25
        ax.text(x + 1, 44, title, fontsize=7, fontweight="bold", va="center")
        ax.plot([x + 1, x + 21], [y0, y0], color=C["ink3"], lw=1.0, ls=(0, (4, 3)))
        ax.add_patch(Rectangle((x + 20.2, y0 - 1.0), 1.6, 2.0, fc=C["ink"], ec="none"))
        ax.text(x + 21, y0 - 4.0, "threshold", ha="center", fontsize=5.4, color=C["ink2"])
        plane(x + 4, y0, C["flown"], "commanded" if i != 3 else None)
        plane(x + 13, y0, C["obs"], "leader" if i != 2 else None)
        for k, t in enumerate(lines):
            ax.text(x + 1, 19.5 - k * 3.2, t, fontsize=5.9, color=C["ink2"], va="center")
    # (b) the inserted aircraft
    plane(1 + 25 + 9, y0 + 8.5, C["corr"], "inserted"); ax.plot([26 + 1, 26 + 21], [y0 + 8.5, y0 + 8.5], color=C["ink3"], lw=0.7, ls=(0, (2, 3)))
    # (c) the leader, before and after the shift
    plane(1 + 50 + 13, y0, C["obs"], None, hollow=True)
    plane(1 + 50 + 17.5, y0, C["corr"], "leader moved")
    ax.annotate("", xy=(1 + 50 + 16.2, y0 - 2.6), xytext=(1 + 50 + 13, y0 - 2.6), arrowprops=dict(arrowstyle="-|>", color=C["corr"], lw=0.8))
    # (d) the commanded aircraft, before and after the move
    plane(1 + 75 + 4, y0, C["flown"], None, hollow=True)
    plane(1 + 75 + 7, y0 + 7.5, C["corr"], "commanded, moved")
    ax.annotate("", xy=(1 + 75 + 6.2, y0 + 5.6), xytext=(1 + 75 + 4, y0 + 1.6), arrowprops=dict(arrowstyle="-|>", color=C["corr"], lw=0.8))
    ax.text(50, 6.0, "The commanded aircraft is always the one that speaks; the others fly their records. A round draws the four kinds from the same flights.",
            ha="center", fontsize=6.0, color=C["ink2"], va="center")
    save(fig, "fig16_window_kinds")


# ------------------------------------------------------------------ Fig 15: the outcomes of the judge
def fig15_judge_outcomes() -> None:
    fig, ax, H = canvas(DOUBLE, 4.6)
    ax.set_ylim(0, 65.5)
    Q = [(2, 57, "A state is not finite, or the stall cut-off binds?", "dynamics_failure"),
         (2, 48, "Below the threshold elevation, before the threshold?", "ground_contact"),
         (2, 38, "Approach crossing of the plane of R?\n(G false, track within 30°, inside the landing screen)", None),
         (2, 24, "G false and a lined-up crossing of another candidate\nthat is inside that runway’s limit?", "crossed_other_runway")]
    qb = []
    for (x, y, t, o) in Q:
        h = 7.2 if "\n" in t else 5.8
        qb.append(box(ax, x, y, 38, h, t, FILL["n"], fs=6.3))
        if o:
            ob = box(ax, 44, y + (h - 5) / 2, 18, 5, o, FILL["block"], fs=6.0, ec=C["block"])
            arrow(ax, anchor(qb[-1], "r"), anchor(ob, "l"), "yes", lxy=(41.0, y + h / 2 + 0.6), fs=6.0)
    for i in range(3): arrow(ax, anchor(qb[i], "b"), anchor(qb[i + 1], "t"), "no", lxy=(21.5, qb[i][1] - 2.7), ha="left", fs=6.0)
    to = box(ax, 2, 8, 38, 6, "timeout: none of these within the time limit", FILL["n"], fs=6.3, ec=C["ink3"])
    arrow(ax, anchor(qb[3], "b"), anchor(to, "t"), "no", lxy=(21.5, 18.0), ha="left", fs=6.0)
    # the approach crossing: three more questions
    s1 = box(ax, 66, 56, 20, 6, "Higher than 100 m\nabove the threshold?", FILL["n"], fs=5.9)
    o1 = box(ax, 88, 56.5, 11.5, 5, "crossed_\ntoo_high", FILL["block"], fs=5.8, ec=C["block"])
    s2 = box(ax, 66, 44, 20, 6, "Offset from the\ncentreline > 106.7 m?", FILL["n"], fs=5.9)
    o2 = box(ax, 88, 44.5, 11.5, 5, "crossed_off_\nrunway", FILL["block"], fs=5.8, ec=C["block"])
    s3 = box(ax, 66, 31, 20, 7, "DA check passed?\n(±22 m, inside the FAS cone)", FILL["n"], fs=5.7)
    o3 = box(ax, 66, 19, 9.5, 5, "landed", FILL["ok"], fs=6.0, ec=C["ok"], bold=True)
    o4 = box(ax, 77, 19, 12, 5, "unstable_at\n_minimums", FILL["block"], fs=5.2, ec=C["block"])
    ax.plot([40, 63.5, 63.5], [41.6, 41.6, 59], color=C["ink2"], lw=0.9, zorder=2); arrow(ax, (63.5, 59), (66, 59), None)
    ax.text(52, 42.2, "yes", fontsize=6.0, color=C["ink2"], ha="center")
    arrow(ax, anchor(s1, "r"), anchor(o1, "l"), "yes", lxy=(87, 59.6), fs=5.8)
    arrow(ax, anchor(s1, "b"), anchor(s2, "t"), "no", lxy=(77.2, 52.2), ha="left", fs=5.8)
    arrow(ax, anchor(s2, "r"), anchor(o2, "l"), "yes", lxy=(87, 47.6), fs=5.8)
    arrow(ax, anchor(s2, "b"), anchor(s3, "t"), "no", lxy=(77.2, 40.2), ha="left", fs=5.8)
    arrow(ax, anchor(s3, "b", 0.2), anchor(o3, "t", 0.5), "yes", lxy=(68.5, 27.3), ha="right", fs=5.8)
    arrow(ax, anchor(s3, "b", 0.8), anchor(o4, "t", 0.5), "no", lxy=(83.5, 27.3), ha="left", fs=5.8)
    save(fig, "fig15_judge_outcomes")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["fig01_architecture", "fig08_prior_architecture", "fig10_decoding", "fig11_cv_design", "fig12_post_training_loop", "fig13_branch_training", "fig14_traffic_features", "fig15_judge_outcomes", "fig16_window_kinds", "fig17_landed_training", "fig18_prior_layers", "fig19_traffic_attention", "fig20_multi_round", "fig21_random_number", "fig22_window_branches", "fig23_window_tree"]:
        globals()[name]()
