"""Draw the multi-aircraft design's figure of how each aircraft's rows hang on the scene's steps (§2.1, decision 13).

Two aircraft entering the 25 km range at times that are not on the scene's grid, each with a row every `STEP_S` from its
own entry; every row hangs on the nearest even UTC second (`prior.scene.hang`) — at most 1 s off, the same offset for
every row of one aircraft. Inside one step, a quantity between the two is taken at each one's own row time: the other
is moved straight on from its last row at or before that time (`inference.scene_edges`: the row hung on the same step
when it is not later, else the one before it; a first row later than that time is carried back). The rows hung and the
moves are COMPUTED from the example's entry times by those rules, and `tests/test_scene_step_figure.py` checks them
against `prior.scene.hang` and `inference.scene_edges` themselves. Not real data.

    python run_ts.py scene_step_figure            # (re)writes docs/two_tier/figures/scene_steps.svg
    python run_ts.py scene_step_figure --check    # exit 1 if the file on disk is not what this draws

The SVG is plain text (its Chinese labels are drawn by the viewer's fonts), written deterministically, and the test
holds the committed file to what this runner draws.
"""

from __future__ import annotations

import argparse

from ts_transformer.instructions.measure import SUGGESTED
from ts_transformer.prior.scene import hang, scene_steps
from ts_transformer.repo_layout import TS_DIR

FIGURE = TS_DIR / "docs" / "two_tier" / "figures" / "scene_steps.svg"

#: The scene's step is the rows' step (prior design §3.2): every row 2 s after the one before.
STEP_S = SUGGESTED["step_s"]
#: The example: (name, entry time into the 25 km range in seconds of the hour, rows drawn).
AIRCRAFT = (("A", 100.37, 4), ("B", 101.20, 4))
#: The step drawn in detail: where A's row 2 and B's row 1 hang together.
FOCUS_STEP_S = 104.0


def row_times(name: str) -> list[float]:
    """An aircraft's row times: its entry time plus `STEP_S` per row."""
    entry, rows = next((entry, rows) for n, entry, rows in AIRCRAFT if n == name)
    return [round(entry + k * STEP_S, 6) for k in range(rows)]


def hung(name: str) -> list[float]:
    """The step each of an aircraft's rows hangs on (`prior.scene.hang`)."""
    return [float(t) for t in hang(row_times(name), STEP_S)]


def row_on(name: str, step_s: float) -> float:
    """The time of an aircraft's row hung on ``step_s``."""
    return next(t for t, s in zip(row_times(name), hung(name)) if s == step_s)


def seen_from(seer: str, seen: str, step_s: float) -> tuple[float, float]:
    """At ``step_s``, the row of ``seen`` that ``seer``'s row reads (`inference.scene_edges`: its row on the same step
    when that is not later than the seer's or is its first, else the one before it) and how far it is moved on to the
    seer's time (negative: a first row carried back)."""
    at = row_on(seer, step_s)
    there = row_on(seen, step_s)
    source = row_on(seen, step_s - STEP_S) if there > at and step_s - STEP_S in hung(seen) else there
    return source, round(at - source, 6)


# ---- drawing
WIDTH, LEFT, RIGHT = 960, 170, 30
T_MIN, T_MAX = 99.5, 108.5
SCALE = (WIDTH - LEFT - RIGHT) / (T_MAX - T_MIN)       # pixels a second
LANE_Y = {"A": 120, "B": 220}
COLOURS = {"A": "#2563eb", "B": "#ea580c"}
INK, GREY, PALE, BAND = "#111827", "#6b7280", "#9ca3af", "#fef3c7"
FONT = "Noto Sans CJK SC, Noto Sans SC, PingFang SC, Microsoft YaHei, sans-serif"


def x(t_s: float) -> float:
    return LEFT + (t_s - T_MIN) * SCALE


def seconds(t_s: float) -> str:
    return f"{t_s:.2f}"


def text(x_px: float, y_px: float, body: str, *, size: int = 12, anchor: str = "start", colour: str = INK,
         weight: str = "normal", halo: bool = False) -> str:
    """`halo`: a white outline under the letters, for a label a line runs through."""
    outline = ' stroke="white" stroke-width="4" stroke-linejoin="round" paint-order="stroke"' if halo else ""
    return (f'<text x="{x_px:.1f}" y="{y_px:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{colour}" '
            f'font-weight="{weight}"{outline}>{body}</text>')


def line(x1: float, y1: float, x2: float, y2: float, colour: str, *, width: float = 1.0, dash: str = "") -> str:
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}" '
            f'stroke-width="{width}"{dashes}/>')


def arrow(x1: float, x2: float, y: float, colour: str, *, dash: str = "") -> list[str]:
    """A horizontal arrow from ``x1`` to ``x2``, its head at ``x2``."""
    head = 7.0 if x2 > x1 else -7.0
    return [line(x1, y, x2 - head, y, colour, width=2, dash=dash),
            f'<path d="M {x2:.1f} {y} L {x2 - head:.1f} {y - 4.5} L {x2 - head:.1f} {y + 4.5} Z" fill="{colour}"/>']


def render() -> str:
    height = 430
    top, bottom = 62, 268
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" '
             f'font-family="{FONT}">',
             f'<rect x="0" y="0" width="{WIDTH}" height="{height}" fill="white" stroke="none"/>',
             text(20, 26, "每一行挂到最近的场景时间步（示意的数，不是真实数据）", size=15, weight="bold")]

    # the step drawn in detail — every time that hangs on it, a step wide — shaded first so everything else is on top
    parts.append(f'<rect x="{x(FOCUS_STEP_S - STEP_S / 2):.1f}" y="{top}" width="{STEP_S * SCALE:.1f}" '
                 f'height="{bottom - top}" fill="{BAND}" stroke="none"/>')

    # the scene's steps: even UTC seconds
    parts.append(text(20, top - 8, "场景时间步", size=12, colour=GREY))
    parts.append(text(20, top + 8, "（UTC 偶数秒）", size=11, colour=GREY))
    for step in scene_steps(T_MIN, T_MAX, STEP_S):
        parts.append(line(x(step), top, x(step), bottom, PALE, dash="3,3"))
        parts.append(text(x(step), top - 8, f"{step:.0f} s", size=12, anchor="middle", colour=INK, weight="bold"))

    # inside the focused step: each one's row reads the other moved on to its own time (drawn under the rows, so
    # the dotted drop from the row read passes behind its dot and time label)
    for seer, seen, lane_offset in (("A", "B", 34), ("B", "A", 34)):
        source, moved = seen_from(seer, seen, FOCUS_STEP_S)
        at = row_on(seer, FOCUS_STEP_S)
        y = LANE_Y[seen] + lane_offset
        colour = COLOURS[seen]
        # from the row read (a dotted drop from its dot) to the reader's time, ending just before the hollow circle
        parts.append(line(x(source), LANE_Y[seen], x(source), y, colour, dash="1,3"))
        parts.extend(arrow(x(source), x(at) - 7, y, colour, dash="4,3"))
        parts.append(f'<circle cx="{x(at):.1f}" cy="{y}" r="5" fill="white" stroke="{colour}" stroke-width="2"/>')
        parts.append(text(x(at) + 10, y + 4, f"{seer} 在 {seconds(at)} s 读 {seen}：从 {seconds(source)} s 那一行推 "
                                             f"{moved:.2f} s", size=11, colour=colour, halo=True))

    # each aircraft's rows, at their own times, and the step each hangs on
    for name, entry, _ in AIRCRAFT:
        y = LANE_Y[name]
        colour = COLOURS[name]
        parts.append(text(20, y - 4, f"飞机 {name}", size=13, colour=colour, weight="bold"))
        parts.append(text(20, y + 14, f"{seconds(entry)} s 进入 25 km", size=11, colour=GREY))
        for k, (t, s) in enumerate(zip(row_times(name), hung(name))):
            if abs(t - s) > 1e-9:
                parts.extend(arrow(x(t), x(s), y, colour))
            parts.append(f'<circle cx="{x(t):.1f}" cy="{y}" r="6" fill="{colour}"/>')
            parts.append(text(x(t), y - 14, f"第 {k} 行", size=11, anchor="middle", colour=colour))
            parts.append(text(x(t), y + 22, seconds(t), size=11, anchor="middle", colour=colour, halo=True))
        first, step_of = row_times(name)[0], hung(name)[0]
        shift = step_of - first
        parts.append(text((x(first) + x(step_of)) / 2, y - 30,
                          f"挂到 {step_of:.0f} s（{'+' if shift > 0 else '−'}{abs(shift):.2f} s，每行都一样）", size=11,
                          anchor="middle", colour=colour, halo=True))

    a_row, b_row = row_on("A", FOCUS_STEP_S), row_on("B", FOCUS_STEP_S)
    parts.append(text(x(FOCUS_STEP_S), bottom + 18,
                      f"时间步 {FOCUS_STEP_S:.0f} s 里：A 第 {row_times('A').index(a_row)} 行（{seconds(a_row)} s）"
                      f"和 B 第 {row_times('B').index(b_row)} 行（{seconds(b_row)} s）",
                      size=12, anchor="middle", weight="bold"))

    legend = (
        "实心点：这架飞机的行。时刻 = 它进入 25 km 的时刻 + 2 s × 行号，不改；它的输入和说的词按这一行。",
        "实线箭头：这一行挂到最近的时间步（最多差 1 s，正好在中间时挂到后一步）。只用来把各架排进同一个时间步。",
        "空心点、虚线箭头：两架之间的量按本机这一行的真实时刻算。邻机取它在这一时刻之前（含）的最后一行，按那一行的",
        "速度直线推到这一时刻（不超过 2 s；邻机第一行晚于这一时刻时，用第一行往回推；inference/scene_edges.py）。",
        "闭环里由执行器飞的飞机从它挂到的时间步起飞，此后的状态正好在时间步上：两架都由执行器飞时推的时间为 0。",
    )
    for n, body in enumerate(legend):
        parts.append(text(20, bottom + 52 + 22 * n, body, size=12))
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
    print(f"wrote {FIGURE}: A hangs {hung('A')}, B hangs {hung('B')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
