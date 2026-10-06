"""Paper figures made from data: the real flights of the artefact (train split) and the simulation sketches."""
from __future__ import annotations

import math
import sys

import numpy as np
from matplotlib.gridspec import GridSpec

from figs_common import C, DATA, FILL, SIM, DOUBLE, SINGLE, panel, plt, rel_deg, save

LEVELS, TOL = np.array(SIM["levels"]), np.array(SIM["tol"])


# ------------------------------------------------------------------ Fig 2: the grids of the vocabulary
def fig02_vocabulary_grids() -> None:
    fig = plt.figure(figsize=(DOUBLE, 4.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    # (a) the heading grid, relative to the course of the runway
    ax = fig.add_subplot(gs[0, 0], projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_ylim(0, 1.45); ax.grid(False)
    ax.set_yticks([]); ax.set_xticks([]); ax.spines["polar"].set_visible(False)
    for k in range(72):
        th = math.radians(5 * k)
        big = k % 18 == 0
        ax.plot([th, th], [0.86 if big else 0.92, 1.0], color=C["ink2"] if big else C["faint"], lw=1.0 if big else 0.6)
    ax.plot(np.linspace(0, 2 * np.pi, 200), np.ones(200), color=C["faint"], lw=0.7)
    for k, lab, col in [(0, "class 0\nthe final", C["flown"]), (18, "+90°\nbase leg", C["ink2"]), (36, "class 36\ndownwind", C["ink2"]), (54, "−90°\nbase leg", C["ink2"])]:
        th = math.radians(5 * k)
        ax.annotate("", xy=(th, 0.82), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2 if k == 0 else 0.8, mutation_scale=7))
        ax.text(th, 1.34, lab, ha="center", va="center", fontsize=6.5, color=col, fontweight="bold" if k == 0 else "normal")
    ax.text(math.radians(14), 0.55, "5° steps", fontsize=6.5, color=C["ink3"], ha="left")
    panel(ax, "a", dx=-0.06, dy=1.0)
    # (b) the altitude levels and the band of each
    ax = fig.add_subplot(gs[0, 1])
    for lo, hi, col, lab in [(0, 1260, FILL["a"], "60 m steps"), (1260, 2700, FILL["b"], "120 m steps"), (2700, 5400, FILL["c"], "450 m steps")]:
        ax.axvspan(lo, hi, color=col, lw=0); ax.text((lo + hi) / 2, 262, lab, ha="center", fontsize=6.5, color=C["ink2"])
    ax.step(LEVELS, TOL, where="mid", color=C["word"], lw=1.3)
    ax.plot(LEVELS, np.full_like(LEVELS, -12), "|", color=C["ink2"], ms=5, mew=0.6)
    ax.set_xlim(0, 5400); ax.set_ylim(-25, 285); ax.set_xlabel("height above the airport elevation (m)"); ax.set_ylabel("tolerance ε of the level (m)")
    ax.text(5350, 20, "ticks: the 40 levels", ha="right", va="bottom", fontsize=6.5, color=C["ink2"])
    panel(ax, "b")
    # (c) the angle classes
    ax = fig.add_subplot(gs[1, 0])
    sp = DATA["spec"]; x = np.array([0, 10.0])
    for k, e in enumerate(sp["descent_edges_deg"][1:-1]):
        ax.plot(x, 1000 * x * math.tan(math.radians(e)), color=C["faint"], lw=0.8, ls=(0, (4, 3)))
    cols = [C["flown"], C["ok"], C["word"], C["corr"]]
    for k, d in enumerate(sp["descent_centres_deg"]):
        y = 1000 * 10 * math.tan(math.radians(d)); ax.plot(x, 1000 * x * math.tan(math.radians(d)), color=cols[k], lw=1.3)
        ax.text(10.1, y, f"descent {k + 1}  {d}°", va="center", fontsize=6.5, color=cols[k])
    ax.set_xlim(0, 10); ax.set_ylim(0, 820); ax.set_xlabel("distance flown (km)"); ax.set_ylabel("height lost (m)")
    ax.text(0.3, 750, "dashed: class edges 2.0°, 2.75°, 3.75°", fontsize=6.5, color=C["ink2"])
    ax.set_xlim(0, 12.6); ax.set_xticks([0, 2, 4, 6, 8, 10])
    panel(ax, "c")
    # (d) speed words as steps of 5 m/s
    ax = fig.add_subplot(gs[1, 1])
    s = SIM["speed_steps"]; t = np.array(s["ts"]); obs = np.array(s["obs"]); ev = np.array(s["ev"]); fl = np.array(s["flown"])
    ax.plot(t, obs, color=C["obs"], lw=1.2, label="observed speed")
    ax.step(np.append(ev[:, 0], 180), np.append(ev[:, 1], ev[-1, 1]), where="post", color=C["word"], lw=1.2, ls=(0, (4, 2)), label="speed word in force")
    ax.plot(ev[:, 0], ev[:, 1], "o", color=C["word"], ms=2.8)
    ax.plot(fl[:, 0], fl[:, 1], color=C["flown"], lw=1.3, label="flown speed (a$_\\mathrm{max}$ = 1.4 m/s$^2$)")
    ax.set_xlim(0, 150); ax.set_ylim(70, 160); ax.set_xlabel("time (s)"); ax.set_ylabel("ground speed (m/s)")
    ax.legend(loc="upper right")
    panel(ax, "d")
    save(fig, "fig02_vocabulary_grids")


# ------------------------------------------------------------------ Fig 3: a real sentence
def _states(fl):
    st = np.array(fl["states"]["rows"]); ob = fl["observed"]
    return st, ob


def fig03_sentence_example() -> None:
    fl = DATA["flights"]["vectored"]; st, ob = _states(fl)
    start, ev, dt = fl["start_row"], fl["every"], fl["delta_s"]
    c = fl["candidates"][fl["runway_index"]]; E0 = fl["elevation_m"]
    t_state = np.arange(len(st)) * 2.0; i0 = start * ev
    texts = fl["words"]["text"]; corr = np.array(fl["words"]["correction"]); M = len(texts)
    tw = (start + np.arange(M)) * dt
    fig = plt.figure(figsize=(DOUBLE, 5.6))
    gs = GridSpec(4, 2, figure=fig, width_ratios=[1.0, 1.25], height_ratios=[1.0, 1.0, 1.3, 0.1], hspace=0.5, wspace=0.28)
    # (a) plan view
    ax = fig.add_subplot(gs[0:3, 0])
    cr = math.radians(c["course"])
    ax.plot([(c["e"] - 30000 * math.sin(cr)) / 1000, c["e"] / 1000], [(c["n"] - 30000 * math.cos(cr)) / 1000, c["n"] / 1000], color=C["ink3"], lw=0.8, ls=(0, (4, 3)))
    ax.plot(np.array(ob["e"]) / 1000, np.array(ob["n"]) / 1000, color=C["obs"], lw=1.2, label="observed track")
    ax.plot(st[i0:, 0] / 1000, st[i0:, 1] / 1000, color=C["flown"], lw=1.4, label="flown by the executor")
    ax.plot(c["e"] / 1000, c["n"] / 1000, "s", color=C["ink"], ms=4, label=f"threshold of runway {c['ident']}")
    ax.plot(st[i0, 0] / 1000, st[i0, 1] / 1000, "o", color=C["ink"], mfc="white", ms=4.5, label="first predicted step")
    heads = [(k, t[1]) for k, t in enumerate(texts) if t[1]]
    last = None
    for q, (k, w) in enumerate(heads):
        i = (start + k) * ev; p = (st[i, 0] / 1000, st[i, 1] / 1000)
        if last is None or math.hypot(p[0] - last[0], p[1] - last[1]) > 1.6:
            ax.plot(*p, "o", color=C["word"], ms=3)
            ax.annotate(w + "°", p, textcoords="offset points", xytext=(5, 5), fontsize=6.3, color=C["word"]); last = p
    ax.set_aspect("equal"); ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)"); ax.legend(loc="upper left", fontsize=6.3)
    panel(ax, "a", dx=-0.16)
    # (b) height
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.array(ob["t"]), np.array(ob["alt"]) - E0, color=C["obs"], lw=1.1)
    ax.plot(t_state[i0:], st[i0:, 2] - E0, color=C["flown"], lw=1.3)
    level, t_from, spans = None, None, []
    for k, t in enumerate(texts):
        if t[2]:
            if level is not None: spans.append((t_from, tw[k], level))
            level, t_from = t[2], tw[k]
    spans.append((t_from, t_state[-1], level))
    for a, b, lv in spans:
        if lv == "no level-off": ax.axvspan(a, b, color=FILL["band"], lw=0); ax.text((a + b) / 2, 80, "no level-off", ha="center", fontsize=6.3, color=C["ink2"])
        else: ax.hlines(float(lv), a, b, color=C["word"], lw=1.1, ls=(0, (4, 2)))
    ax.set_ylabel("height above E (m)"); ax.set_xlim(0, t_state[-1]); ax.tick_params(labelbottom=False)
    panel(ax, "b")
    # (c) speed
    ax = fig.add_subplot(gs[1, 1], sharex=ax)
    ax.plot(np.array(ob["t"]), ob["gs"], color=C["obs"], lw=1.1, label="observed")
    ax.plot(t_state[i0:], st[i0:, 4], color=C["flown"], lw=1.3, label="flown")
    sp_t, sp_v = [], []
    for k, t in enumerate(texts):
        if t[4]: sp_t.append(tw[k]); sp_v.append(None if t[4] == "unspecified" else float(t[4]))
    xs, ys = [], []
    for j, (a, v) in enumerate(zip(sp_t, sp_v)):
        b = sp_t[j + 1] if j + 1 < len(sp_t) else t_state[-1]
        if v is None: ax.axvspan(a, b, color=FILL["n"], lw=0); ax.text((a + b) / 2, 138, "unspecified", ha="center", fontsize=6.3, color=C["ink2"])
        else: ax.hlines(v, a, b, color=C["word"], lw=1.1, ls=(0, (4, 2)))
    ax.set_ylabel("ground speed (m/s)"); ax.tick_params(labelbottom=False); ax.legend(loc="lower left", ncol=2, fontsize=6.3)
    panel(ax, "c")
    # (d) the raster of the words
    ax = fig.add_subplot(gs[2, 1], sharex=ax)
    lanes = ["runway", "heading", "altitude", "angle", "speed"]
    ax.set_ylim(-0.7, 4.7); ax.invert_yaxis(); ax.set_yticks(range(5)); ax.set_yticklabels(lanes)
    last_angle = -1e9
    for k, t in enumerate(texts):
        for q in range(5):
            if t[q]:
                col = C["corr"] if corr[k][q] else C["word"]
                ax.plot(tw[k], q, "s", color=col, ms=3.2, mew=0)
                if (q in (0, 2) or (q == 3 and not corr[k][q] and tw[k] - last_angle > 40) or (q == 4 and not t[q].isdigit()) or (q == 1 and k == 0)):
                    if q == 3: last_angle = tw[k]
                    ax.annotate(t[q].replace("descent ", "descent ").replace("no level-off", "no level-off"), (tw[k], q), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=5.6, color=col)
    ax.set_xlabel("time since the first row (s)"); ax.grid(axis="y", visible=False)
    ax.plot([], [], "s", color=C["word"], ms=3.2, label="word of the observed track"); ax.plot([], [], "s", color=C["corr"], ms=3.2, label="correction word")
    ax.legend(loc="upper right", fontsize=6.3, ncol=2, bbox_to_anchor=(1.0, -0.3))
    panel(ax, "d")
    save(fig, "fig03_sentence_example")


# ------------------------------------------------------------------ Fig 4: the three laws of the executor
def fig04_executor_laws() -> None:
    fig = plt.figure(figsize=(DOUBLE, 5.4))
    gs = GridSpec(3, 2, figure=fig, hspace=0.62, wspace=0.28)
    L = SIM["lateral_one"]; t = np.array([s["t"] for s in L])
    ax = fig.add_subplot(gs[0, 0])
    ax.step([0, 5, 60], [0, 90, 90], where="post", color=C["word"], ls=(0, (4, 2)), lw=1.2, label="heading word (track change)")
    ax.plot(t, [s["track"] for s in L], color=C["flown"], lw=1.4, label="flown track")
    ax.axvline(5 + SIM["lead_s"], color=C["faint"], lw=0.8); ax.text(5 + SIM["lead_s"] + 1.0, 52, "lead L = 4 s", fontsize=6.3, color=C["ink2"], rotation=90, va="center")
    ax.set_xlim(0, 60); ax.set_ylim(-5, 108); ax.set_ylabel("track change (deg)"); ax.legend(loc="lower right", fontsize=6.3); ax.tick_params(labelbottom=False)
    panel(ax, "a")
    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    ax.plot(t, [min(s["timeLeft"], 8) if s["t"] >= 5 else np.nan for s in L], color=C["corr"], lw=1.0, label="error ÷ time left")
    ax.plot(t, [min(s["stopping"], 8) if s["t"] >= 5 else np.nan for s in L], color=C["ok"], lw=1.0, label="stopping rate")
    ax.axhline(SIM["rmax"], color=C["obs"], lw=0.9, ls=(0, (4, 2))); ax.text(1, SIM["rmax"] + 0.2, "turn-rate limit 4.7°/s", ha="left", fontsize=6.3, color=C["obs"])
    ax.plot(t, [s["rate"] for s in L], color=C["flown"], lw=1.6, label="rate that flies (smallest)")
    ax.set_ylim(0, 8.4); ax.set_ylabel("turn rate (deg/s)"); ax.legend(loc="upper right", fontsize=6.0, bbox_to_anchor=(1.0, 1.02)); ax.tick_params(labelbottom=False)
    panel(ax, "b")
    ax = fig.add_subplot(gs[2, 0], sharex=ax)
    ax.axhline(32, color=C["faint"], lw=0.8, ls=(0, (4, 2))); ax.axhline(0, color=C["faint"], lw=0.6)
    ax.plot(t, [abs(s["bank"]) for s in L], color=C["flown"], lw=1.4); ax.text(59, 33.5, "bank limit 32°", ha="right", fontsize=6.3, color=C["ink2"])
    ax.set_ylim(-2, 40); ax.set_xlabel("time (s)"); ax.set_ylabel("bank (deg)")
    panel(ax, "c")
    # vertical law
    V = SIM["vertical"]; km = np.array([s["km"] for s in V]); h = np.array([s["h"] for s in V])
    ax = fig.add_subplot(gs[0, 1])
    ax.hlines(600, 0, 12, color=C["word"], ls=(0, (4, 2)), lw=1.1); ax.text(11.8, 612, "level word 600 m", ha="right", fontsize=6.3, color=C["word"])
    ax.plot(km, h, color=C["flown"], lw=1.4)
    cap = next(i for i, s in enumerate(V) if s["captured"] and s["t"] > 2)
    ax.plot(km[cap], h[cap], "o", color=C["corr"], ms=3.8); ax.annotate(f"level-off starts\n{V[cap]['levelOff']:.0f} m above the level", (km[cap], h[cap]), textcoords="offset points", xytext=(10, 14), fontsize=6.3, color=C["corr"], arrowprops=dict(arrowstyle="-", color=C["corr"], lw=0.6))
    ax.set_xlim(0, 12); ax.set_ylim(540, 940); ax.set_ylabel("height above E (m)"); ax.tick_params(labelbottom=False)
    panel(ax, "d")
    ax = fig.add_subplot(gs[1, 1], sharex=ax)
    ax.axhline(-2.5, color=C["word"], lw=0.9, ls=(0, (4, 2))); ax.text(11.8, -2.35, "descent 2: 2.5°", ha="right", fontsize=6.3, color=C["word"], va="bottom")
    ax.plot(km, [-s["gamma"] for s in V], color=C["flown"], lw=1.4)
    ax.axhline(0, color=C["faint"], lw=0.6); ax.set_ylabel("path angle (deg)"); ax.set_ylim(-3.4, 1.0); ax.tick_params(labelbottom=False)
    panel(ax, "e")
    # speed law
    ax = fig.add_subplot(gs[2, 1])
    s = SIM["speed_steps"]; t2 = np.array(s["ts"]); fl = np.array(s["flown"]); ev = np.array(s["ev"])
    ax.step(np.append(ev[:, 0], 180), np.append(ev[:, 1], ev[-1, 1]), where="post", color=C["word"], lw=1.1, ls=(0, (4, 2)), label="speed words (5 m/s steps)")
    ax.plot(fl[:, 0], fl[:, 1], color=C["flown"], lw=1.4, label="flown speed")
    ax.set_xlim(0, 150); ax.set_ylim(70, 160); ax.set_xlabel("time (s)"); ax.set_ylabel("ground speed (m/s)"); ax.legend(loc="upper right", fontsize=6.3)
    panel(ax, "f")
    ax_d = fig.axes[3]; ax_d.set_xlabel("")
    fig.axes[4].set_xlabel("distance flown (km)")
    fig.axes[4].tick_params(labelbottom=True)
    save(fig, "fig04_executor_laws")


# ------------------------------------------------------------------ Fig 5: the closed-loop reading on a real flight
def fig05_closed_loop_real() -> None:
    fl = DATA["flights"]["vectored"]; start, dt = fl["start_row"], fl["delta_s"]
    M = len(fl["words"]["grid"]); tw = (start + np.arange(M)) * dt
    ey = np.array([np.nan if v is None else v for v in fl["lateral_m"]]); eh = np.array([np.nan if v is None else v for v in fl["vertical_m"]])
    corr = np.array(fl["words"]["correction"]); texts = fl["words"]["text"]
    fig = plt.figure(figsize=(DOUBLE, 4.6))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.0, 0.7], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax.axhspan(-30, 30, color=FILL["band"], lw=0); ax.axhline(0, color=C["faint"], lw=0.6)
    ax.plot(tw, ey, color=C["flown"], lw=1.4)
    ch = corr[:, 1].astype(bool); ax.plot(tw[ch], ey[ch], "o", color=C["corr"], ms=3.6, label="a heading correction word is said")
    ax.set_ylabel("lateral error e$_y$ (m)"); ax.legend(loc="lower right"); ax.text(tw[-1] + 4, 27, "tolerance Y = 30 m", fontsize=6.3, color=C["flown"], va="top", ha="right")
    ax.tick_params(labelbottom=False); panel(ax, "a", dx=-0.07)
    ax = fig.add_subplot(gs[1], sharex=ax)
    nlo = False; tol = []
    for t in texts:
        if t[2] == "no level-off": nlo = True
        elif t[2]: nlo = False
        tol.append(10 if nlo else 15)
    tol = np.array(tol, float)
    ax.fill_between(tw, -tol, tol, step="post", color=FILL["band"], lw=0); ax.axhline(0, color=C["faint"], lw=0.6)
    ax.plot(tw, eh, color=C["flown"], lw=1.4)
    ca = corr[:, 3].astype(bool); ax.plot(tw[ca], eh[ca], "o", color=C["corr"], ms=3.6, label="an angle correction word is said")
    ax.set_ylabel("vertical error e$_h$ (m)"); ax.legend(loc="upper right"); ax.text(tw[0] + 2, 46, "tolerance H = 15 m, H$_\\mathrm{final}$ = 10 m", fontsize=6.3, color=C["flown"], va="top", ha="left")
    ax.tick_params(labelbottom=False); panel(ax, "b", dx=-0.07)
    ax = fig.add_subplot(gs[2], sharex=ax)
    for q, name, row in [(1, "heading", 0), (3, "angle", 1)]:
        for k in range(M):
            if texts[k][q]:
                ax.plot(tw[k], row, "s", color=C["corr"] if corr[k][q] else C["word"], ms=3.4, mew=0)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["heading", "angle"]); ax.set_ylim(1.6, -0.6); ax.grid(axis="y", visible=False)
    ax.plot([], [], "s", color=C["word"], ms=3.4, label="observed word"); ax.plot([], [], "s", color=C["corr"], ms=3.4, label="correction word")
    ax.legend(loc="center right", ncol=2, fontsize=6.3); ax.set_xlabel("time since the first row (s)"); panel(ax, "c", dx=-0.07)
    ax.set_xlim(tw[0], tw[-1] + 6)
    save(fig, "fig05_closed_loop_real")


# ------------------------------------------------------------------ Fig 6: the effect of the corrections (a sketch)
def fig06_corrections_effect() -> None:
    on, off = SIM["closed_on"], SIM["closed_off"]; Y = SIM["closed_params"]["Y"]
    fig = plt.figure(figsize=(DOUBLE, 3.1)); gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.15], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    rows = np.array(on["rows"]); ax.plot(rows[:, 0], rows[:, 1] / 1000, color=C["obs"], lw=1.3, label="observed path")
    for d, ls, lab in [(off, (0, (4, 2)), "flown, corrections off"), (on, "-", "flown, corrections on")]:
        f = np.array(d["flown"]); ax.plot(f[:, 0], f[:, 1] / 1000, color=C["flown"], lw=1.4, ls=ls, label=lab)
    ax.axvline(0, color=C["ink3"], lw=0.8, ls=(0, (4, 3))); ax.plot(0, 0, "s", color=C["ink"], ms=4, label="threshold")
    ax.set_xlim(-250, 250); ax.set_ylim(-0.3, 7.0); ax.set_xlabel("offset east of the centreline (m)"); ax.set_ylabel("North (km)")
    h_, l_ = ax.get_legend_handles_labels()
    panel(ax, "a", dx=-0.14)
    ax = fig.add_subplot(gs[1])
    ax.axhspan(-Y, Y, color=FILL["band"], lw=0); ax.axhline(0, color=C["faint"], lw=0.6)
    for d, ls in [(off, (0, (4, 2))), (on, "-")]:
        e = np.array(d["errs"]); ax.plot(e[:, 0], e[:, 1], color=C["flown"], lw=1.4, ls=ls)
    c = np.array(on["corrs"]); ax.plot(c[:, 0], c[:, 1], "o", color=C["corr"], ms=3.8, label="correction word said")
    ax.set_xlabel("time since the first predicted step (s)"); ax.set_ylabel("lateral error e$_y$ (m)")
    h2, l2 = ax.get_legend_handles_labels()
    fig.legend(h_ + h2, l_ + l2, loc="lower center", ncol=5, fontsize=6.3, bbox_to_anchor=(0.5, -0.09))
    panel(ax, "b")
    save(fig, "fig06_corrections_effect")


# ------------------------------------------------------------------ Fig 7: the heading words on the 2, 4 and 8 s grids (real labeller output)
def fig07_delta_grid() -> None:
    from pathlib import Path
    from figs_real import heading_words_2s
    t, rel = heading_words_2s()                    # the real words at 2 s of the flight (labeller output)
    fig = plt.figure(figsize=(DOUBLE, 4.2)); gs = GridSpec(3, 1, figure=fig, hspace=0.35)
    for q, delta in enumerate((2, 4, 8)):
        ax = fig.add_subplot(gs[q], sharex=fig.axes[0] if q else None)
        ax.step(np.append(t, t[-1] + 8), np.append(rel, rel[-1]), where="post", color=C["obs"], lw=0.9)
        ax.plot(t, rel, "o", color=C["obs"], ms=2.4, label="word of the 2 s reading" if q == 0 else None)
        rows = {}
        for ti, r in zip(t, rel):
            rows[int(math.floor(ti / delta + 0.5))] = r         # the nearest row; a tie goes to the later row; the last word of a row wins
        gt = np.array(sorted(rows)) * delta; gv = np.array([rows[k] for k in sorted(rows)])
        for ti, r in zip(t, rel):
            j = math.floor(ti / delta + 0.5) * delta
            if abs(j - ti) > 1e-9: ax.plot([ti, j], [r, r], color=C["corr"], lw=0.8, label="move of a word to its row" if (q == 1 and not hasattr(ax, "_m")) else None); ax._m = True
        ax.step(np.append(gt, gt[-1] + 8), np.append(gv, gv[-1]), where="post", color=C["word"], lw=1.3)
        ax.plot(gt, gv, "s", color=C["word"], ms=3.2, label="word on the Δ grid" if q == 0 else None)
        jump = np.max(np.abs(np.diff(gv))) / 5 if len(gv) > 1 else 0
        ax.text(0.01, 0.06, f"Δ = {delta} s: {len(gt)} words, largest jump {jump:.0f} classes", transform=ax.transAxes, fontsize=6.6, color=C["ink2"])
        ax.set_ylabel("heading word (deg)")
        if q < 2: ax.tick_params(labelbottom=False)
        panel(ax, "abc"[q], dx=-0.065)
    ax.set_xlabel("time since the first row (s)")
    h1, l1 = fig.axes[0].get_legend_handles_labels(); h2, l2 = fig.axes[1].get_legend_handles_labels()
    fig.axes[0].legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.3, ncol=3)
    save(fig, "fig07_delta_grid")


# ------------------------------------------------------------------ Fig 9: the procedure masks on a real final (KRDU 23R)
RE = 6.371e6


def proc_mask(f, E, d, h, off=0.0, joined_before=False, dipped=False, G=False):
    thr = f["elev"] - E
    gp = lambda x: thr + f["tch"] + x * math.tan(math.radians(f["gp"])) + x * x / (2 * RE)
    dfpap = max(f["length"], 9023 * 0.3048); dgarp = dfpap + 1000 * 0.3048; cw = max(350 * 0.3048, math.tan(math.radians(1.5)) * dgarp)
    inside = 0 <= d <= f["faf_m"] and abs(off) <= cw * (d + dgarp) / dgarp
    joined = joined_before or inside; barred = dipped and not joined and not G
    edge = gp(d) - 60 if inside else math.nan; decision = thr + f["da"]
    ok = np.ones(len(LEVELS), bool)
    for i, (L, t) in enumerate(zip(LEVELS, TOL)):
        ok[i] = (L >= edge - t) if inside else (L >= decision - t)
        if barred and L > h + t: ok[i] = False
    return dict(ok=ok, inside=inside, edge=edge, decision=decision, entry=gp(f["faf_m"]), barred=barred, gp=gp, thr=thr)


def fig09_procedure_masks() -> None:
    A = DATA["airports"]["KRDU"]; f = next(c for c in A["candidates"] if c["ident"] == "23R"); E = A["elevation_m"]
    fig = plt.figure(figsize=(DOUBLE, 3.5)); gs = GridSpec(1, 4, figure=fig, width_ratios=[5, 0.9, 5, 0.9], wspace=0.12)
    cases = [("a", dict(d=7000.0, h=330.0), "inside the region: the lower edge acts"),
             ("b", dict(d=13000.0, h=470.0, dipped=True), "before the join, after a dip below the entry height")]
    for q, (letter, st, note) in enumerate(cases):
        m = proc_mask(f, E, **st); ax = fig.add_subplot(gs[0, 2 * q]); rl = fig.add_subplot(gs[0, 2 * q + 1], sharey=ax)
        dd = np.linspace(0, 16000, 160)
        ax.axvspan(0, f["faf_m"] / 1000, color=FILL["a"], lw=0, alpha=0.55)
        ax.plot(dd / 1000, [m["gp"](x) for x in dd], color=C["ink"], lw=1.2, label="published glidepath")
        dr = dd[dd <= f["faf_m"]]; ax.plot(dr / 1000, [m["gp"](x) - 60 for x in dr], color=C["block"], lw=1.2, ls=(0, (4, 2)), label="lower edge (glidepath − 60 m)")
        ax.axhline(m["decision"], color=C["ok"], lw=1.0, ls=(0, (4, 2)), label="decision altitude (DA)")
        ax.axhline(m["entry"], color=C["ink3"], lw=0.8, ls=(1, (1, 2)), label="entry height")
        ax.axvline(f["faf_m"] / 1000, color=C["faint"], lw=0.8); ax.text(f["faf_m"] / 1000 - 0.15, 1330, "FAF", ha="right", fontsize=6.5, color=C["ink2"]); ax.text(f["faf_m"] / 1000 - 0.15, 1230, "region", ha="right", fontsize=6.5, color=C["flown"])
        ax.plot(st["d"] / 1000, st["h"], "o", color=C["block"] if m["barred"] else C["flown"], ms=5.5, mec="white", mew=1.0, zorder=5)
        ax.set_xlim(16, 0); ax.set_ylim(-60, 1400); ax.set_xlabel("distance before the threshold (km)")
        if q == 0: ax.set_ylabel("height above E (m)"); ax.legend(loc="lower left", fontsize=6.0, bbox_to_anchor=(0.0, 0.0))
        else: ax.tick_params(labelleft=False)
        panel(ax, letter, dx=-0.12)
        for i, L in enumerate(LEVELS): rl.plot([0, 1], [L, L], color=C["ok"] if m["ok"][i] else C["block"], lw=1.1)
        rl.set_xlim(0, 1); rl.set_xticks([]); rl.grid(False); rl.spines["bottom"].set_visible(False); rl.tick_params(labelleft=False, left=False)
        rl.set_xlabel("level words", fontsize=6.5)
        if q == 1:
            ax.plot([], [], color=C["ok"], lw=1.2, label="level word permitted"); ax.plot([], [], color=C["block"], lw=1.2, label="level word blocked"); ax.legend(loc="upper right", fontsize=6.0)
    save(fig, "fig09_procedure_masks")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["fig02_vocabulary_grids", "fig03_sentence_example", "fig04_executor_laws", "fig05_closed_loop_real", "fig06_corrections_effect", "fig07_delta_grid", "fig09_procedure_masks"]:
        globals()[name]()
