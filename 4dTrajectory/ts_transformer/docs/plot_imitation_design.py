#!/usr/bin/env python
"""Regenerate the figures for section 12 of the bank-wiggle diagnosis document.

Reads the published KRDU imitation campaign directly, so the figures cannot drift from
the artifacts they describe. Light-surface PNGs, same palette slots as
plot_control_wiggle_diagnosis.py.

    conda run -n aeroviz python 4dTrajectory/ts_transformer/docs/plot_imitation_design.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "4dTrajectory" / "ts_transformer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_control_arms import score  # noqa: E402

CAMPAIGN = REPO / "4dTrajectory" / "outputs" / "KRDU" / "experiments" / "imitation_design"
FIGURES = Path(__file__).resolve().parent / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8f8e88"
GRID = "#e4e3de"
BLUE = "#2a78d6"      # the model
ORANGE = "#eb6834"    # what the observed track asks for
AQUA = "#1baf7a"      # the reachable ceiling

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Ordered by dose, never alphabetically: a glob once put 16x between baseline and 1x.
ORDER = ("baseline", "imitation_1x", "bracket_1p5x", "imitation_4x", "imitation_16x",
         "seed2024_16x", "extend_47x")
LABEL = {
    "baseline": "simple-v2 baseline\n(no imitation term)",
    "imitation_1x": "imitation 0.74x position",
    "bracket_1p5x": "imitation 1.47x position",
    "imitation_4x": "imitation 2.94x position",
    "imitation_16x": "imitation 11.8x position",
    "seed2024_16x": "imitation 11.8x, seed 2024",
    "extend_47x": "imitation 47x position",
}


def _finish(ax, title: str, subtitle: str | None = None) -> None:
    lines = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(title, loc="left", pad=8 + 11 * lines)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5,
                color=INK_2, va="bottom", linespacing=1.35)
    ax.set_axisbelow(True)


def load_arms() -> dict[str, dict]:
    arms = {}
    for key in ORDER:
        pred = CAMPAIGN / f"{key}_pred_val"
        if (pred / "summary.json").is_file():
            result = score(pred)
            if result:
                arms[key] = result
    return arms


def figure_skill_band(arms: dict[str, dict]) -> None:
    """Bank skill is only readable between the floor and the twin ceiling."""
    keys = [k for k in ORDER if k in arms]
    y = np.arange(len(keys))[::-1]
    floor = arms[keys[0]]["skill_floor"]
    twin = arms[keys[0]]["skill_twin"]

    fig, ax = plt.subplots(figsize=(8.4, 0.56 * len(keys) + 2.3))
    ax.axvspan(floor, twin, color=AQUA, alpha=0.10, lw=0)
    ax.axvline(floor, color=ORANGE, lw=2.0, zorder=3)
    ax.axvline(twin, color=AQUA, lw=2.0, zorder=3)
    for row, key in zip(y, keys):
        value = arms[key]["bank_skill"]
        ax.plot([min(value, floor), max(value, floor)], [row, row],
                color=GRID, lw=2.0, zorder=2, solid_capstyle="round")
        ax.plot([value], [row], "o", ms=9, color=BLUE, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{value:+.3f}", (value, row), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [LABEL[k] for k in keys], fontsize=8.5)
    for row, key in zip(y, keys):
        if key == "seed2024_16x":
            ax.get_yticklabels()[list(y).index(row)].set_color(MUTED)
    ax.set_xlabel("per-flight bank skill  (correlation with the flown track's bank)")
    # Headroom above the top row so the reference captions never sit on a marker.
    ax.set_ylim(-0.7, len(keys) + 0.45)
    top = len(keys) - 0.30
    ax.annotate(f"random other flight  {floor:+.3f}", (floor, top), ha="left",
                va="bottom", fontsize=8, color=ORANGE, fontweight="bold",
                xytext=(4, 0), textcoords="offset points")
    ax.annotate(f"same-runway twin  {twin:+.3f}", (twin, top), ha="right",
                va="bottom", fontsize=8, color=AQUA, fontweight="bold",
                xytext=(-4, 0), textcoords="offset points")
    ax.text(0.995, 0.03, "the twin is a yardstick, not a bound", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=MUTED, style="italic")
    _finish(ax, "Bank skill against the references this metric actually has",
            "KRDU validation, 1404 flights. Left of the orange line the model says\n"
            "less about the flown bank than a random real flight does.")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "imitation_skill_band.png", dpi=170)
    plt.close(fig)


def figure_mode_alignment(arms: dict[str, dict]) -> None:
    """The model concentrates its bank into a single mode -- the wrong one."""
    pred = CAMPAIGN / "baseline_pred_val"
    model, observed = _bank_matrices(pred)
    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    x = np.linspace(0, 100, model.shape[1])
    modes = {}
    for matrix, colour, name in ((observed, ORANGE, "flown tracks"),
                                 (model, BLUE, "simple-v2 baseline")):
        residual = matrix - matrix.mean(axis=0)
        _u, s, vt = np.linalg.svd(residual, full_matrices=False)
        mode = vt[0] * np.sign(vt[0][np.argmax(np.abs(vt[0]))])
        modes[name] = mode
        share = 100 * s[0] ** 2 / (s ** 2).sum()
        ax.plot(x, mode, color=colour, lw=2.0,
                label=f"{name} — mode 1 ({share:.0f}% of residual energy)")
    alignment = float(np.corrcoef(modes["flown tracks"],
                                  modes["simple-v2 baseline"])[0, 1])
    ax.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
    ax.axvspan(55, 72, color=BLUE, alpha=0.07, lw=0, zorder=0)
    ax.annotate("a counter-turn the\nflown tracks never make", (63.5, -0.075),
                ha="center", va="top", fontsize=8, color=BLUE)
    ax.set_xlabel("progress through the forecast (%)")
    ax.set_ylabel("mode-1 shape (arbitrary units)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _finish(ax, "The average turn is learned; the counter-turn after it is invented",
            f"KRDU validation. Dominant bank mode of each side, own mean removed; "
            f"shape correlation {alignment:+.2f}.")
    fig.tight_layout()
    fig.savefig(FIGURES / "imitation_mode_alignment.png", dpi=170)
    plt.close(fig)


def _bank_matrices(pred_dir: Path):
    import json
    from control_inverse_dynamics import actual_controls
    aero = np.array([122.6, 2.7, 0.023, 0.0334, 0.8, 0.2])
    grid = 64
    model, observed = [], []
    for path in sorted(pred_dir.glob("*_states.json")):
        payload = json.loads(path.read_text())
        segments = payload["control_segments"]
        future = [r for r in payload["observed_states"] if r["t"] >= 0.0]
        if len(segments) < 2 or len(future) < 8:
            continue
        times = np.array([r["t"] for r in future], float)
        keep = np.concatenate(([True], np.diff(times) > 1e-6))
        times = times[keep]
        states = np.array([[r["lat"], r["lon"], r["alt"], r["V"], r["psi"], r["gamma"],
                            r["m"]] for r, k in zip(future, keep) if k], float)
        try:
            bank = actual_controls(states, times, aero_params=aero,
                                   max_thrust_n=240_000.0)[:, 1]
        except ValueError:
            continue
        progress = (np.arange(grid) + 0.5) / grid
        observed.append(np.degrees(np.interp(progress * (times[-1] - times[0]) + times[0],
                                             times, bank)))
        segment_bank = np.degrees([s["bank_rad"] for s in segments])
        model.append(segment_bank[np.minimum((progress * len(segments)).astype(int),
                                             len(segments) - 1)])
    return np.array(model), np.array(observed)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    arms = load_arms()
    if not arms:
        print(f"no scored arms under {CAMPAIGN}")
        return 1
    print(f"scored arms: {', '.join(arms)}")
    figure_mode_alignment(arms)
    if len(arms) > 1:
        figure_skill_band(arms)
    else:
        print("  skill band needs more than the baseline arm; skipped")
    for path in sorted(FIGURES.glob("imitation_*.png")):
        print(f"  wrote {path.relative_to(REPO)}  ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
