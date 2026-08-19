#!/usr/bin/env python
"""Regenerate the figures for 2026-08-19_control_bank_wiggle_diagnosis.zh.md.

Reads the published KSJC validation batches directly, so the figures cannot drift from
the artifacts they describe. Light-surface PNGs, as the doc is read on a repo page.

    conda run -n aeroviz python 4dTrajectory/ts_transformer/docs/plot_control_wiggle_diagnosis.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "4dTrajectory" / "ts_transformer"))
from control_inverse_dynamics import actual_controls  # noqa: E402

EXPERIMENTS = REPO / "4dTrajectory" / "outputs" / "KSJC" / "experiments"
BEFORE = "flight_model_paired"                 # zeroed control-head weights
AFTER = "flight_model_paired_seeded_head"      # seeded control-head weights
ARM = "first_order_lag"
FIGURES = Path(__file__).resolve().parent / "figures"

# Validated categorical slots (dataviz reference palette, light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8f8e88"
GRID = "#e4e3de"
BLUE = "#2a78d6"      # the model
ORANGE = "#eb6834"    # what the observed track asks for
AQUA = "#1baf7a"      # the attempted fix
AERO = np.array([122.6, 2.7, 0.023, 0.0334, 0.8, 0.2])
MAX_THRUST_N = 240_000.0
N_SEGMENTS = 64

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _tortuosity(rows) -> tuple[float, np.ndarray]:
    metres = np.array([[r["lon"], r["lat"]] for r in rows]) * np.array(
        [111_320.0 * math.cos(math.radians(rows[0]["lat"])), 111_320.0]
    )
    length = float(np.sum(np.linalg.norm(np.diff(metres, axis=0), axis=1)))
    straight = float(np.linalg.norm(metres[-1] - metres[0]))
    return (length / straight if straight > 1.0 else np.inf), metres


def load(campaign: str, arm: str = ARM) -> dict:
    """Model bank, observed-inverted bank, tortuosity and paths, per flight."""
    predictions = EXPERIMENTS / campaign / f"{arm}_pred_val"
    model, observed, tortuosity, records = [], [], [], []
    for path in sorted(predictions.glob("*_states.json")):
        payload = json.loads(path.read_text())
        segments = payload["control_segments"]
        future = [r for r in payload["observed_states"] if r["t"] >= 0.0]
        if len(segments) != N_SEGMENTS or len(future) < 8:
            continue
        states = np.array(
            [[r["lat"], r["lon"], r["alt"], r["V"], r["psi"], r["gamma"], r["m"]]
             for r in future], dtype=float)
        times = np.array([r["t"] for r in future], dtype=float)
        keep = np.concatenate(([True], np.diff(times) > 1e-6))
        states, times = states[keep], times[keep]
        try:
            bank = actual_controls(
                states, times, aero_params=AERO, max_thrust_n=MAX_THRUST_N
            )[:, 1]
        except ValueError:
            continue
        grid = (np.arange(N_SEGMENTS) + 0.5) / N_SEGMENTS * (times[-1] - times[0]) + times[0]
        observed.append(np.degrees(np.interp(grid, times, bank)))
        model.append(np.degrees([s["bank_rad"] for s in segments]))
        tort, _ = _tortuosity(future)
        tortuosity.append(tort)
        records.append({"name": path.name, "payload": payload})
    return {
        "model": np.array(model), "observed": np.array(observed),
        "tortuosity": np.array(tortuosity), "records": records,
    }


def _finish(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", pad=14 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5,
                color=INK_2, va="bottom")
    ax.set_axisbelow(True)


def figure_profiles(data: dict) -> None:
    """The core evidence: one fixed S-curve vs a population mean that is flat."""
    model, observed = data["model"], data["observed"]
    x = np.arange(N_SEGMENTS)
    fig, ax = plt.subplots(figsize=(8.4, 4.0))

    for values, colour, label in ((observed, ORANGE, "what the flown tracks ask for"),
                                  (model, BLUE, "what the model predicts")):
        mean = values.mean(axis=0)
        ax.fill_between(x, np.percentile(values, 25, axis=0),
                        np.percentile(values, 75, axis=0),
                        color=colour, alpha=0.13, linewidth=0)
        ax.plot(x, mean, color=colour, linewidth=2.0, label=label, zorder=3)

    ax.axhline(0.0, color=MUTED, linewidth=1.0, zorder=2)
    ax.set_ylim(-9.5, 15.5)
    ax.annotate("the flown tracks average ≈ 0°",
                xy=(57, observed.mean(axis=0)[57]), xytext=(52, 5.4),
                color=INK_2, fontsize=8.5, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))
    peak = int(np.argmax(model.mean(axis=0)))
    ax.annotate(f"a {model.mean(axis=0)[peak]:.1f}° hump\nthe data never asks for",
                xy=(peak, model.mean(axis=0)[peak]), xytext=(peak - 17, 13.6),
                color=INK_2, fontsize=8.5, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))

    ax.set_xlabel("control segment (0 = anchor, 63 = threshold)")
    ax.set_ylabel("bank angle (deg)")
    ax.set_xlim(0, N_SEGMENTS - 1)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK_2)
    _finish(ax, "The model flies one fixed S-curve on every approach",
            f"mean profile, band = interquartile range across {len(model)} KSJC "
            "validation flights")
    fig.tight_layout()
    fig.savefig(FIGURES / "bank_profile_comparison.png", dpi=170)
    plt.close(fig)


def figure_energy(data: dict) -> None:
    """Where the bank variance lives: shared across flights, or specific to one."""
    def split(values: np.ndarray) -> tuple[float, float]:
        common = values.mean(axis=0)
        shared = float(np.sum(common ** 2) * values.shape[0])
        return 100 * shared / float(np.sum(values ** 2)), float(np.median(values.std(axis=0)))

    after = load(AFTER)
    rows = [
        ("flown tracks (truth)", *split(data["observed"])),
        ("model, as shipped", *split(data["model"])),
        ("model, after the attempted fix", *split(after["model"])),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    y = np.arange(len(rows))[::-1]
    for index, (label, shared, _sd) in zip(y, rows):
        ax.barh(index, shared, height=0.52, color=BLUE if "model" in label else ORANGE,
                zorder=3)
        ax.barh(index, 100 - shared, left=shared + 1.0, height=0.52,
                color=GRID, zorder=3)
        inside = shared > 12.0
        ax.text(shared - 1.5 if inside else shared + 2.5, index, f"{shared:.0f}%",
                ha="right" if inside else "left", va="center",
                color=SURFACE if inside else INK, fontsize=9, fontweight="bold", zorder=4)
        ax.text(101.5, index, f"{100-shared:.0f}% flight-specific", va="center",
                color=INK_2, fontsize=8.5)
    ax.set_yticks(y, [r[0] for r in rows], color=INK)
    ax.set_xlim(0, 148)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.grid(axis="y", visible=False)
    _finish(ax, "Almost all of the model's bank is the same on every flight",
            "share of total bank energy carried by the profile common to all flights")
    fig.tight_layout()
    fig.savefig(FIGURES / "bank_energy_share.png", dpi=170)
    plt.close(fig)


def figure_example(data: dict) -> None:
    """One straight-in approach, drawn: the reference, the prediction, the bank."""
    # Straight reference AND a roughly correct predicted duration, so the figure shows the
    # everyday wiggle rather than one of the four final-time blow-ups (which are a separate
    # defect and would make this look worse than the typical case).
    def usable(i: int) -> bool:
        payload = data["records"][i]["payload"]
        future = [r for r in payload["observed_states"] if r["t"] >= 0.0]
        true_span = float(future[-1]["t"])
        return (data["tortuosity"][i] < 1.005 and true_span > 120.0
                and abs(payload["final_time_s"] - true_span) < 0.2 * true_span)

    straight = [i for i in range(len(data["model"])) if usable(i)]
    worst = max(straight, key=lambda i: float(np.abs(data["model"][i]).max()))
    payload = data["records"][worst]["payload"]
    future = [r for r in payload["observed_states"] if r["t"] >= 0.0]
    predicted = payload["predicted_states"]
    lat0, lon0 = future[0]["lat"], future[0]["lon"]
    mlat, mlon = 111_320.0, 111_320.0 * math.cos(math.radians(lat0))

    def xy(rows):
        return (np.array([(r["lon"] - lon0) * mlon for r in rows]) / 1000.0,
                np.array([(r["lat"] - lat0) * mlat for r in rows]) / 1000.0)

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    ox, oy = xy(future)
    px, py = xy(predicted)
    ax.plot(ox, oy, color=ORANGE, linewidth=2.0, label="flown track (reference)", zorder=3)
    ax.plot(px, py, color=BLUE, linewidth=2.0, label="prediction", zorder=3)
    ax.scatter([ox[0]], [oy[0]], s=42, color=INK, zorder=4)
    ax.annotate("anchor", (ox[0], oy[0]), textcoords="offset points", xytext=(8, -10),
                color=INK_2, fontsize=8.5)
    ax.set_xlabel("east of anchor (km)")
    ax.set_ylabel("north of anchor (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, loc="best", labelcolor=INK_2)
    _finish(ax, "A straight-in approach, drawn",
            f"{data['records'][worst]['name'].split('_')[0]} · reference is straight to "
            "within 0.5 %")

    bank_model = data["model"][worst]
    bank_obs = data["observed"][worst]
    seg = np.arange(N_SEGMENTS)
    bx.plot(seg, bank_obs, color=ORANGE, linewidth=2.0,
            label="inverted from the flown track")
    bx.plot(seg, bank_model, color=BLUE, linewidth=2.0, label="predicted")
    bx.axhline(0.0, color=MUTED, linewidth=1.0)
    bx.set_xlabel("control segment")
    bx.set_ylabel("bank angle (deg)")
    bx.set_xlim(0, N_SEGMENTS - 1)
    bx.legend(frameon=False, loc="best", labelcolor=INK_2)
    _finish(bx, "…and the bank each one requires",
            "the flown track needs almost none")
    fig.tight_layout()
    fig.savefig(FIGURES / "straight_in_example.png", dpi=170)
    plt.close(fig)


def figure_attempted_fix() -> None:
    """The paired ADE change from seeding the head — a regression, not a fix."""
    def rows(campaign):
        summary = json.loads(
            (EXPERIMENTS / campaign / f"{ARM}_pred_val" / "summary.json").read_text())
        return {(r["id"], r["icao24"], r["landing_time_utc"]): r["ade_m"]
                for r in summary["results"]}

    before, after = rows(BEFORE), rows(AFTER)
    shared = sorted(set(before) & set(after))
    delta = np.array([after[k] - before[k] for k in shared])
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    ax.hist(delta, bins=np.linspace(-400, 700, 70), color=AQUA, zorder=3)
    ax.axvline(0.0, color=MUTED, linewidth=1.2, zorder=4)
    ax.axvline(float(np.median(delta)), color=INK, linewidth=1.6, linestyle="--", zorder=5)
    ax.annotate(f"median {np.median(delta):+.0f} m", xy=(np.median(delta), ax.get_ylim()[1]),
                xytext=(6, -14), textcoords="offset points", color=INK, fontsize=9,
                fontweight="bold")
    better = 100.0 * float((delta < 0).mean())
    ax.text(0.98, 0.92, f"the fix helped {better:.0f} % of flights\nand hurt {100-better:.0f} %",
            transform=ax.transAxes, ha="right", va="top", color=INK_2, fontsize=9)
    ax.set_xlabel("change in per-flight ADE after seeding the head (m) — negative is better")
    ax.set_ylabel("flights")
    _finish(ax, "The attempted fix made accuracy worse",
            f"paired over the same {len(delta)} validation flights")
    fig.tight_layout()
    fig.savefig(FIGURES / "attempted_fix_regression.png", dpi=170)
    plt.close(fig)


def figure_final_time_floor() -> None:
    """The second, separate defect: the duration head cannot reach short remaining times."""
    summary = json.loads(
        (EXPERIMENTS / BEFORE / f"{ARM}_pred_val" / "summary.json").read_text())
    true = np.array([r["true_final_time_s"] for r in summary["results"]], dtype=float)
    predicted = np.array([r["predicted_final_time_s"] for r in summary["results"]], dtype=float)
    ok = np.isfinite(true) & np.isfinite(predicted)
    true, predicted = true[ok], predicted[ok]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    limit = max(true.max(), predicted.max()) * 1.03
    ax.plot([0, limit], [0, limit], color=MUTED, linewidth=1.2, zorder=2)
    ax.scatter(true, predicted, s=11, color=BLUE, alpha=0.35, linewidth=0, zorder=3)
    floor = float(predicted.min())
    ax.axhline(floor, color=ORANGE, linewidth=1.8, zorder=4)
    ax.text(limit * 0.98, floor + 14, f"the head never predicts below {floor:.0f} s",
            ha="right", color=ORANGE, fontsize=9, fontweight="bold")
    short = true < 90
    ax.scatter(true[short], predicted[short], s=42, facecolor="none",
               edgecolor=ORANGE, linewidth=1.6, zorder=5)
    ax.annotate("these three loop:\nthe rollout flies for 5 minutes\nwhen the runway is 30 s away",
                xy=(float(true[short].max()), float(predicted[short].min())),
                xytext=(limit * 0.16, limit * 0.62), color=INK_2, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))
    ax.set_xlabel("true remaining time from the anchor (s)")
    ax.set_ylabel("predicted remaining time (s)")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    _finish(ax, "A separate defect: the duration head has a floor",
            f"the diagonal is a perfect prediction · {len(true)} validation flights")
    fig.tight_layout()
    fig.savefig(FIGURES / "final_time_floor.png", dpi=170)
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load(BEFORE)
    print(f"loaded {len(data['model'])} flights from {BEFORE}")
    figure_profiles(data)
    figure_energy(data)
    figure_example(data)
    figure_attempted_fix()
    figure_final_time_floor()
    for path in sorted(FIGURES.glob("*.png")):
        print(f"  wrote {path.relative_to(REPO)}  ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
