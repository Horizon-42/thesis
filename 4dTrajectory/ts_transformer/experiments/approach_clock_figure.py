"""Draw the multi-aircraft design's figure of the edge feature "ahead or behind on the approach clock" (§2.5).

Two parallel runways whose thresholds are not level (23L's is `STAGGER_M` further along the landing direction than
23R's, as at KRDU), an aircraft on each extended centreline, and the one axis both are projected on: each aircraft's
position along the landing direction is its threshold's position (`Separation.along_nm`, in metres) less how far it
still is before that threshold along the course (`relative_to_runway`'s ``before_threshold_m``); ahead or behind is
the difference of the two positions. The positions are COMPUTED from the example's distances by that rule, and
`tests/test_approach_clock_figure.py` checks the rule's two conventions against the code it cites. Not real data.

    python run_ts.py approach_clock_figure            # (re)writes docs/two_tier/figures/approach_clock.svg
    python run_ts.py approach_clock_figure --check    # exit 1 if the file on disk is not what this draws

The SVG is plain text (its Chinese labels are drawn by the viewer's fonts), written deterministically, and the test
holds the committed file to what this runner draws.
"""

from __future__ import annotations

import argparse

from ts_transformer.repo_layout import TS_DIR

FIGURE = TS_DIR / "docs" / "two_tier" / "figures" / "approach_clock.svg"

#: The example: 23L's threshold is this much further along the landing direction than 23R's (KRDU's: 0.67 NM,
#: `inference/runway_schedule.py`'s module text) ...
STAGGER_M = 1_240.0
#: ... so on the approach clock the thresholds sit here, metres along the landing direction. The origin is the first
#: runway by name, as `runway_schedule.parallel_relations` takes it — of these two; at KRDU itself it is 05L's.
ALONG_M = {"23L": 0.0, "23R": -STAGGER_M}
#: The two aircraft: (name, runway in force, how far before its threshold along the course).
AIRCRAFT = (("i", "23R", 8_000.0), ("j", "23L", 6_000.0))


def position_m(runway: str, before_threshold_m: float) -> float:
    """An aircraft's position along the landing direction: its threshold's, less how far it still is before it."""
    return ALONG_M[runway] - before_threshold_m


def ahead_m() -> float:
    """How far j is ahead of i on the approach clock (the edge feature i → j, before scaling)."""
    (_, runway_i, before_i), (_, runway_j, before_j) = AIRCRAFT
    return position_m(runway_j, before_j) - position_m(runway_i, before_i)


# ---- drawing
WIDTH, LEFT, RIGHT = 960, 60, 40
X_MIN_M, X_MAX_M = -10_000.0, 3_500.0
SCALE = (WIDTH - LEFT - RIGHT) / (X_MAX_M - X_MIN_M)     # pixels a metre
RUNWAY_DRAWN_M = 3_000.0                                  # a runway's drawn length: nominal, drawn at the axis scale
ROW_Y = {"23L": 100, "23R": 170}                          # 23L is the left one looking along the landing direction
COLOURS = {"i": "#2563eb", "j": "#ea580c"}
INK, GREY, PALE = "#111827", "#6b7280", "#9ca3af"
FONT = "Noto Sans CJK SC, Noto Sans SC, PingFang SC, Microsoft YaHei, sans-serif"


def x(along_m: float) -> float:
    return LEFT + (along_m - X_MIN_M) * SCALE


def metres(value: float) -> str:
    return f"{value:,.0f}".replace("-", "−")


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


def bracket(x1: float, x2: float, y: float, colour: str) -> list[str]:
    """A horizontal span with an end tick at each end."""
    return [line(x1, y, x2, y, colour, width=1.5), line(x1, y - 5, x1, y + 5, colour, width=1.5),
            line(x2, y - 5, x2, y + 5, colour, width=1.5)]


def render() -> str:
    axis_y = 262
    height = 432
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" '
             f'font-family="{FONT}">',
             f'<rect x="0" y="0" width="{WIDTH}" height="{height}" fill="white" stroke="none"/>',
             text(LEFT, 24, "进近时钟上的前后：把两架飞机都投到落地方向上比（示意；两条跑道的横向间距不按比例）", size=15,
                  weight="bold"),
             text(830, 56, "落地方向", size=12, anchor="end", colour=GREY),
             line(838, 52, 900, 52, GREY, width=2),
             f'<path d="M 908 52 L 898 47 L 898 57 Z" fill="{GREY}"/>']

    # each aircraft and threshold projected down onto the axis, drawn first so the runways cover the lines
    aircraft = [(name, position_m(runway, before), ROW_Y[runway]) for name, runway, before in AIRCRAFT]
    thresholds = [(f"{runway} 入口", along, ROW_Y[runway]) for runway, along in ALONG_M.items()]
    for _, along, row_y in aircraft + thresholds:
        parts.append(line(x(along), row_y + 12, x(along), axis_y, PALE, dash="2,3"))

    # the runways, their thresholds, and the extended centrelines the aircraft are on
    for runway, y in ROW_Y.items():
        threshold = x(ALONG_M[runway])
        parts.append(f'<rect x="{threshold:.1f}" y="{y - 8}" width="{RUNWAY_DRAWN_M * SCALE:.1f}" height="16" '
                     f'fill="#e5e7eb" stroke="{GREY}"/>')
        parts.append(line(threshold, y - 11, threshold, y + 11, INK, width=3))
        parts.append(text(threshold + RUNWAY_DRAWN_M * SCALE / 2, y + 4, f"跑道 {runway}", size=11, anchor="middle",
                          colour=GREY))
    for name, runway, before in AIRCRAFT:
        y = ROW_Y[runway]
        at = x(position_m(runway, before))
        threshold = x(ALONG_M[runway])
        parts.append(line(at, y, threshold, y, PALE, dash="4,4"))
        parts.append(f'<path d="M {at + 9:.1f} {y} L {at - 7:.1f} {y - 7} L {at - 7:.1f} {y + 7} Z" '
                     f'fill="{COLOURS[name]}"/>')
        parts.append(text(at - 12, y + 5, name, size=14, anchor="end", colour=COLOURS[name], weight="bold"))
        label_y = y - 14 if y < ROW_Y["23R"] else y + 24
        parts.append(text((at + threshold) / 2, label_y, f"{name} 沿航道离 {runway} 入口 {metres(before)} m", size=12,
                          anchor="middle", colour=COLOURS[name], halo=True))

    # the thresholds' stagger
    mid_y = (ROW_Y["23L"] + ROW_Y["23R"]) / 2
    parts.extend(bracket(x(ALONG_M["23R"]), x(ALONG_M["23L"]), mid_y, INK))
    # to the bracket's left, where the row is empty: to its right it would run off the canvas
    parts.append(text(x(ALONG_M["23R"]) - 8, mid_y + 4,
                      f"23L 的入口沿落地方向比 23R 的靠前 {metres(STAGGER_M)} m", size=12, anchor="end"))

    # the one axis: everything projected on the landing direction
    parts.append(line(x(X_MIN_M), axis_y, x(X_MAX_M), axis_y, INK))
    for km in range(int(X_MIN_M / 1000), int(X_MAX_M / 1000) + 1, 2):
        parts.append(line(x(km * 1000.0), axis_y, x(km * 1000.0), axis_y + 4, INK))
        parts.append(text(x(km * 1000.0), axis_y + 16, f"{km} km".replace("-", "−"), size=10, anchor="middle",
                          colour=GREY))
    label_y = axis_y + 36
    for name, along, _ in aircraft:
        parts.append(f'<circle cx="{x(along):.1f}" cy="{axis_y}" r="4" fill="{COLOURS[name]}"/>')
        parts.append(text(x(along), label_y, f"{name} {metres(along)} m", size=12, anchor="middle",
                          colour=COLOURS[name], weight="bold"))
    for name, along, _ in thresholds:
        parts.append(f'<circle cx="{x(along):.1f}" cy="{axis_y}" r="4" fill="{INK}"/>')
        # the origin's label to its right, the other threshold's to its left: they are close
        if along == 0.0:
            parts.append(text(x(along) + 6, label_y, f"{name} 0（本例的原点）", size=12))
        else:
            parts.append(text(x(along) - 6, label_y, f"{name} {metres(along)} m", size=12, anchor="end"))

    # the answer, and why the plain difference of the two distances is wrong
    (_, _, before_i), (_, _, before_j) = AIRCRAFT
    (_, position_i, _), (_, position_j, _) = aircraft
    parts.extend(bracket(x(position_i), x(position_j), axis_y + 64, INK))
    parts.append(text((x(position_i) + x(position_j)) / 2, axis_y + 56, f"j 在 i 前面 {metres(ahead_m())} m", size=13,
                      anchor="middle", weight="bold"))
    parts.append(text(LEFT, axis_y + 100,
                      f"只拿两架各自离入口的距离相减：{metres(before_i)} − {metres(before_j)} = {metres(before_i - before_j)} m，"
                      f"少算了两个入口之间的错位 {metres(STAGGER_M)} m。"))
    parts.append(text(LEFT, axis_y + 124,
                      "每架的位置 = 它生效跑道的入口沿落地方向的位置（Separation.along_nm，换成米）− 它沿航道离这个入口还有多远"
                      "（before_threshold_m）；"))
    parts.append(text(LEFT, axis_y + 144, "前后 = j 的位置 − i 的位置。原点取在哪里都不影响这个差。"))
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
    print(f"wrote {FIGURE}: j ahead of i by {metres(ahead_m())} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
