"""Shared style and helpers of the paper figures.

Rules (the user's): English text only; the legend, the axis labels and the panel letters are inside the figure, but
**no title and no caption** (the caption is on the web page); a PNG at 300 dpi and a vector PDF for each figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "4dTrajectory" / "ts_transformer" / "docs" / "two_tier" / "tutorial" / "figures"
DATA = json.loads((HERE / "data" / "tutorial_data.json").read_text(encoding="utf-8"))
SIM = json.loads((HERE / "data" / "figure_sim.json").read_text(encoding="utf-8"))

# the colour roles of the page (dataviz palette, light mode); one role, one colour in every figure
C = dict(obs="#6f6e68", flown="#2a78d6", corr="#eb6834", word="#4a3aa7", ok="#1baf7a", block="#e34948", warn="#eda100",
         ink="#161615", ink2="#4b4a46", ink3="#8a8982", grid="#dcdad3", faint="#c4c2b9", bg="#ffffff")
FILL = dict(a="#dbe8f9", b="#e6e2f6", c="#fbe3d9", n="#f4f3f0", ok="#d7f1e6", block="#fadcdc", band="#d6e4f7")

SINGLE, DOUBLE = 3.4, 7.0           # inches: one column and two columns


def style() -> None:
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 7.5, "axes.labelsize": 7.5, "axes.linewidth": 0.6,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.fontsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "text.color": C["ink"], "lines.linewidth": 1.3,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5, "grid.linestyle": (0, (2, 3)),
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 110,
    })


def save(fig, name: str) -> None:
    (OUT / "png").mkdir(parents=True, exist_ok=True)
    (OUT / "pdf").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "png" / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "pdf" / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter: str, dx: float = -0.1, dy: float = 1.04) -> None:
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom", ha="left")


def canvas(width: float, height: float, w: float = 100.0):
    """A blank drawing area for a diagram: the data units are ``w`` across; ``height / width * w`` up."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    h = height / width * w
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.grid(False)
    return fig, ax, h


def box(ax, x, y, w, h, text, fc="white", ec=None, lw=0.9, fs=7, bold=False, dashed=False, sub=None, tc=None, r=0.8,
        align="center", zorder=3):
    """A rounded box with its lower-left corner at (x, y). ``sub`` is a second, lighter line. Returns the box."""
    ec = ec or C["ink2"]
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=zorder,
                       ls=(0, (3, 2)) if dashed else "-")
    ax.add_patch(p)
    tx = x + w / 2 if align == "center" else x + 1.2
    if sub:
        ax.text(tx, y + h * 0.63, text, ha=align, va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc or C["ink"], zorder=zorder + 1)
        ax.text(tx, y + h * 0.28, sub, ha=align, va="center", fontsize=fs - 1.3, color=C["ink2"], zorder=zorder + 1)
    else:
        ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc or C["ink"], zorder=zorder + 1)
    return (x, y, w, h)


def anchor(b, side, f=0.5):
    x, y, w, h = b
    return {"l": (x, y + h * f), "r": (x + w, y + h * f), "t": (x + w * f, y + h), "b": (x + w * f, y)}[side]


def arrow(ax, p0, p1, label=None, color=None, rad=0.0, ls="-", lw=0.9, fs=6.3, lxy=None, ha="center", style="-|>", ms=7,
          zorder=2, lc=None):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color or C["ink2"], ls=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=zorder, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if label:
        x, y = lxy if lxy else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.9)
        ax.text(x, y, label, fontsize=fs, ha=ha, va="bottom", color=lc or C["ink2"], zorder=zorder + 1,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))
    return a


def tag(ax, x, y, letter, fs=6.5):
    """A small stage tag (A, B or C)."""
    col = {"A": C["flown"], "B": C["word"], "C": C["corr"]}[letter]
    ax.add_patch(plt.Circle((x, y), 1.25, fc=col, ec="none", zorder=6))
    ax.text(x, y - 0.05, letter, ha="center", va="center", fontsize=fs, color="white", fontweight="bold", zorder=7)


def rel_deg(track, course):
    return (np.asarray(track) - course + 180.0) % 360.0 - 180.0


style()
