"""Draw the multi-aircraft design's figure of how a scene's segment is cut into teacher-forcing samples (§2.3).

A made-up 49-minute segment — twelve arrivals, one of them a background aircraft — cut by the design's rule, drawn as an
SVG: each flight's time in the scene as a bar, coloured by the sample whose loss its steps are in; the part of a flight
the cut splits that the next sample reads only as its preceding context, hatched; the number of aircraft in the scene
over time, with the windows the cuts were looked for in. The cuts are COMPUTED by the rule (`cut_times`), not placed by
hand, so the figure shows what the rule does. Not real data. When the rule is implemented for training (the design's
M2), this runner draws with that implementation instead of its own.

    python run_ts.py scene_sample_figure            # (re)writes docs/two_tier/figures/scene_samples.svg
    python run_ts.py scene_sample_figure --check    # exit 1 if the file on disk is not what this draws

The SVG is plain text (its Chinese labels are drawn by the viewer's fonts), written deterministically, and
`tests/test_scene_sample_figure.py` holds the committed file to what this runner draws.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from ts_transformer.instructions.measure import SUGGESTED
from ts_transformer.repo_layout import TS_DIR

FIGURE = TS_DIR / "docs" / "two_tier" / "figures" / "scene_samples.svg"
#: The design's rule (§2.3): the steps a sample's loss is on span at most this long (its input is longer by the
#: preceding context of the flights the cut before it splits) ...
SAMPLE_MAX_S = 20 * 60.0
#: ... and a longer segment is cut at the step, from this far after the sample's loss starts (the segment's start or
#: the cut before) to `SAMPLE_MAX_S`, with the fewest aircraft in the scene, the earliest of equals. The step at a cut
#: is the later sample's.
SEARCH_FROM_S = 15 * 60.0
#: The scene's step is the rows' step (prior design §3.2).
STEP_S = SUGGESTED["step_s"]


@dataclass(frozen=True)
class Flight:
    name: str
    start_s: float            # its first row, seconds from the segment's start
    end_s: float              # its last row in the scene: the last before the threshold (a background aircraft: its last)
    speaking: bool = True     # False: a background aircraft (no sentence), in the scene as input only


def minutes(start: float, end: float, name: str, speaking: bool = True) -> Flight:
    return Flight(name, start * 60.0, end * 60.0, speaking)


#: The made-up segment.
EXAMPLE = (
    minutes(0.0, 9.0, "A"), minutes(3.0, 12.5, "B"), minutes(7.0, 15.7, "C"), minutes(12.0, 21.0, "D"),
    minutes(18.0, 27.0, "E"), minutes(19.5, 28.5, "F"), minutes(21.5, 26.0, "X", speaking=False),
    minutes(23.0, 31.8, "G"), minutes(28.0, 37.5, "H"), minutes(33.5, 42.0, "I"), minutes(37.0, 46.0, "J"),
    minutes(40.0, 49.0, "K"),
)


def in_scene(flights: tuple[Flight, ...], t_s: float) -> int:
    """How many aircraft are in the scene at ``t_s`` (a flight from its first row to its last, both included)."""
    return sum(flight.start_s <= t_s <= flight.end_s for flight in flights)


def cut_times(flights: tuple[Flight, ...]) -> list[float]:
    """The design's rule (§2.3), seconds from the segment's start: while what is left runs past `SAMPLE_MAX_S`, cut at
    the step from `SEARCH_FROM_S` to `SAMPLE_MAX_S` after the sample's loss starts with the fewest aircraft in the
    scene, the earliest of equals; what follows the cut is cut the same way."""
    start, end = min(f.start_s for f in flights), max(f.end_s for f in flights)
    cuts = []
    while end - start > SAMPLE_MAX_S:
        steps = [start + k * STEP_S for k in range(math.ceil(SEARCH_FROM_S / STEP_S), int(SAMPLE_MAX_S / STEP_S) + 1)]
        counts = [in_scene(flights, t) for t in steps]
        start = steps[counts.index(min(counts))]
        cuts.append(start)
    return cuts


def carried(flights: tuple[Flight, ...], cut_s: float) -> list[Flight]:
    """The flights the cut splits: in the scene at the cut's step (the later sample's), having entered before it."""
    return [f for f in flights if f.start_s < cut_s <= f.end_s]


# ---- drawing (of `EXAMPLE`, which starts at 0)
END_S = max(f.end_s for f in EXAMPLE)
AXIS_END_S = 300.0 * math.ceil(END_S / 300.0)   # the axis runs to the next five minutes
WIDTH, LEFT, RIGHT = 960, 150, 40
SCALE = (WIDTH - LEFT - RIGHT) / AXIS_END_S     # pixels a second
ROW, BAR = 22, 14
COLOURS = ("#2563eb", "#ea580c", "#16a34a")     # one a sample: the example is cut into three
GREY = "#9ca3af"
FONT = "Noto Sans CJK SC, Noto Sans SC, PingFang SC, Microsoft YaHei, sans-serif"
WINDOW = f"{SEARCH_FROM_S / 60:g}–{SAMPLE_MAX_S / 60:g} 分钟"


def x(t_s: float) -> float:
    return LEFT + t_s * SCALE


def mmss(t_s: float) -> str:
    return f"{int(t_s // 60)}:{int(t_s % 60):02d}"


def text(x_px: float, y_px: float, body: str, *, size: int = 12, anchor: str = "start", colour: str = "#111827",
         weight: str = "normal", halo: bool = False) -> str:
    """`halo`: a white outline under the letters, for a label a line runs through."""
    outline = ' stroke="white" stroke-width="4" stroke-linejoin="round" paint-order="stroke"' if halo else ""
    return (f'<text x="{x_px:.1f}" y="{y_px:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{colour}" '
            f'font-weight="{weight}"{outline}>{body}</text>')


def rect(x0: float, y0: float, width: float, height: float, fill: str, *, stroke: str = "none", extra: str = "") -> str:
    return (f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{max(width, 0.0):.1f}" height="{height:.1f}" fill="{fill}" '
            f'stroke="{stroke}"{extra}/>')


def render() -> str:
    flights = EXAMPLE
    cuts = cut_times(flights)
    loss_starts = [0.0, *cuts]
    loss_ends = [*cuts, END_S]
    context_starts = [0.0, *(min(f.start_s for f in carried(flights, cut)) for cut in cuts)]
    samples = len(loss_starts)

    top = 26
    brackets_y = top + 44                       # the first sample's label; its line 8 px below
    rows_y = brackets_y + samples * 34 + 14
    count_y = rows_y + len(flights) * ROW + 28
    count_h = 90
    axis_y = count_y + count_h
    legend_y = axis_y + 50
    height = legend_y + 92
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" '
             f'font-family="{FONT}">',
             # the hatch of the samples after the first (the flights their cut splits, read as context)
             '<defs>' + "".join(
                 f'<pattern id="context{i}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                 f'<rect width="6" height="6" fill="white"/><line x1="0" y1="0" x2="0" y2="6" stroke="{COLOURS[i]}" '
                 f'stroke-width="3"/></pattern>' for i in range(1, samples)) + '</defs>',
             rect(0, 0, WIDTH, height, "white"),
             text(LEFT, top - 8, f"一段 {END_S / 60:g} 分钟的场景怎样切成 teacher forcing 的样本（示意，不是真实数据）",
                  size=15, weight="bold")]

    # each sample: its preceding context (dashed) and the steps its loss is on (solid); the labels are drawn last, over
    # the cuts' lines
    labels = []
    for i in range(samples):
        y = brackets_y + i * 34 + 8
        colour = COLOURS[i]
        if context_starts[i] < loss_starts[i]:
            parts.append(f'<line x1="{x(context_starts[i]):.1f}" y1="{y}" x2="{x(loss_starts[i]):.1f}" y2="{y}" '
                         f'stroke="{colour}" stroke-width="3" stroke-dasharray="5,4"/>')
        parts.append(f'<line x1="{x(loss_starts[i]):.1f}" y1="{y}" x2="{x(loss_ends[i]):.1f}" y2="{y}" stroke="{colour}" '
                     f'stroke-width="6"/>')
        label = (f"样本 {i + 1}：损失 {mmss(loss_starts[i])}–{mmss(loss_ends[i])}" if context_starts[i] == loss_starts[i]
                 else f"样本 {i + 1}：前文 {mmss(context_starts[i])} 起，损失 {mmss(loss_starts[i])}–{mmss(loss_ends[i])}")
        labels.append(text(x(context_starts[i]), y - 7, label, size=12, colour=colour, weight="bold", halo=True))

    # each flight's time in the scene, split by the sample its loss is in; the part before a cut that splits it hatched
    # in the next sample's colour (that sample reads it as its preceding context)
    for row, flight in enumerate(flights):
        y = rows_y + row * ROW
        name = flight.name if flight.speaking else f"{flight.name}（背景飞机）"
        parts.append(text(LEFT - 8, y + BAR - 2, name, size=12, anchor="end"))
        if not flight.speaking:
            parts.append(rect(x(flight.start_s), y, x(flight.end_s) - x(flight.start_s), BAR, "#f3f4f6", stroke=GREY,
                              extra=' stroke-dasharray="4,3"'))
            continue
        for i in range(samples):
            lo, hi = max(flight.start_s, loss_starts[i]), min(flight.end_s, loss_ends[i])
            if lo >= hi:
                continue
            split_by_next = i + 1 < samples and flight in carried(flights, cuts[i])
            parts.append(rect(x(lo), y, x(hi) - x(lo), BAR, COLOURS[i]))
            if split_by_next:
                # the same steps, read again by the next sample as its context: a hatched band over the lower half
                parts.append(rect(x(lo), y + BAR / 2, x(hi) - x(lo), BAR / 2, f"url(#context{i + 1})"))

    # the aircraft in the scene over time, the windows each cut was looked for in, and the cuts
    steps = [k * STEP_S for k in range(int(END_S / STEP_S) + 2)]
    counts = [in_scene(flights, t) for t in steps]
    unit = count_h / (max(counts) + 1)
    for i, cut in enumerate(cuts):
        lo = loss_starts[i] + SEARCH_FROM_S
        parts.append(rect(x(lo), count_y, x(loss_starts[i] + SAMPLE_MAX_S) - x(lo), count_h, "#fef3c7"))
        # beside the window's top, not over it: the cut's line runs through the window
        parts.append(text(x(loss_starts[i] + SAMPLE_MAX_S) + 4, count_y - 4, f"← 损失起点后第 {WINDOW}", size=10,
                          colour="#92400e"))
    # a step line: a vertex only where the number changes
    points, last = [], None
    for t, count in zip(steps, counts):
        if count != last:
            if last is not None:
                points.append(f"{x(t):.1f},{axis_y - last * unit:.1f}")
            points.append(f"{x(t):.1f},{axis_y - count * unit:.1f}")
            last = count
    parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#111827" stroke-width="1.5"/>')
    for n in range(max(counts) + 1):
        parts.append(text(LEFT - 8, axis_y - n * unit + 4, str(n), size=10, anchor="end", colour="#4b5563"))
    parts.append(text(LEFT - 30, count_y + count_h / 2, "在场架数", size=11, anchor="end", colour="#4b5563"))
    for i, cut in enumerate(cuts):
        parts.append(f'<line x1="{x(cut):.1f}" y1="{top + 20}" x2="{x(cut):.1f}" y2="{axis_y}" stroke="#dc2626" '
                     f'stroke-width="1.5" stroke-dasharray="6,4"/>')
        parts.append(text(x(cut) + 4, top + 28, f"切口 {i + 1}（{mmss(cut)}，在场 {in_scene(flights, cut)} 架）",
                          size=11, colour="#dc2626"))
    parts.extend(labels)

    # the time axis
    parts.append(f'<line x1="{LEFT}" y1="{axis_y}" x2="{x(AXIS_END_S):.1f}" y2="{axis_y}" stroke="#111827"/>')
    for minute in range(0, int(AXIS_END_S / 60) + 1, 5):
        parts.append(f'<line x1="{x(minute * 60.0):.1f}" y1="{axis_y}" x2="{x(minute * 60.0):.1f}" y2="{axis_y + 5}" '
                     f'stroke="#111827"/>')
        parts.append(text(x(minute * 60.0), axis_y + 18, str(minute), size=11, anchor="middle"))
    parts.append(text(x(AXIS_END_S / 2), axis_y + 36,
                      "分钟（从这一段的第一架飞机进入 25 km 范围算起；飞机在场 = 从它的第 0 行到越过入口前的最后一行）",
                      size=11, anchor="middle", colour="#4b5563"))

    # the legend
    legend = [(rect(LEFT, legend_y, 26, BAR, COLOURS[0]), "实心：这一步算在该样本的损失里（颜色 = 哪个样本）"),
              (rect(LEFT, legend_y + 22, 26, BAR, "url(#context1)"), "斜线：切口前的这些步，在下一个样本里只作前文（输入），不算损失"),
              (rect(LEFT, legend_y + 44, 26, BAR, "#f3f4f6", stroke=GREY, extra=' stroke-dasharray="4,3"'),
               "虚框：背景飞机（没有句子），在任何样本里都只作输入"),
              (rect(LEFT, legend_y + 66, 26, BAR, "#fef3c7"),
               f"黄色：每一刀的查找范围——这个样本的损失起点（段的开头或上一刀）之后第 {WINDOW}，在这里找场上架数最少的一步")]
    for i, (swatch, label) in enumerate(legend):
        parts.append(swatch)
        parts.append(text(LEFT + 34, legend_y + 11 + i * 22, label, size=12))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    parser.add_argument("--check", action="store_true", help="only compare the file on disk with what this draws")
    args = parser.parse_args(argv)
    drawn = render()
    if args.check:
        same = FIGURE.is_file() and FIGURE.read_text(encoding="utf-8") == drawn
        print(f"{FIGURE}: {'as drawn' if same else 'differs from what this runner draws'}")
        return 0 if same else 1
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    FIGURE.write_text(drawn, encoding="utf-8")
    cuts = cut_times(EXAMPLE)
    print(f"wrote {FIGURE}: cuts at {[mmss(cut) for cut in cuts]}, "
          f"split flights {[[f.name for f in carried(EXAMPLE, cut)] for cut in cuts]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
