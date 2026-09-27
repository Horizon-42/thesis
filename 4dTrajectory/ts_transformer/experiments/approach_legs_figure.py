"""Draw the prior design's figure of the legs of an arrival — what 五边 (the final) is (§1.1).

Panel A: the five legs of a traffic pattern around a runway, by their Chinese names (not to scale). Panel B: an arrival as
the radar-vectored ones in our data fly it, to scale — downwind, base, an intercept heading 30° to the final approach
course (meeting it at least 2 miles outside the approach gate, as 7110.65BB 5-9-1 a / 5-9-2 require for 30°), where
the project's labeller puts the approach clearance (the start of the capture turn), the capture into the corridor (the
vocabulary's definition of "established on the final"), the final to the threshold; and a straight-in arrival beside it.
The path is COMPUTED from the turn radius and the legs' placement (`vectored_path`) and every turn is drawn as points of
its computed circle; `tests/test_approach_legs_figure.py` checks the geometry and the intercept rule. Not real data.

    python run_ts.py approach_legs_figure            # (re)writes docs/two_tier/figures/approach_legs.svg
    python run_ts.py approach_legs_figure --check    # exit 1 if the file on disk is not what this draws

The SVG is plain text (its Chinese labels are drawn by the viewer's fonts), written deterministically, and the test
holds the committed file to what this runner draws.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from aerodynamic_model.common import GRAVITY_MPS2
from geokit import NM_M

from ts_transformer.instructions.envelope import corridor_half_width_m
from ts_transformer.instructions.spec import ATC_MAX_INTERCEPT_DEG
from ts_transformer.repo_layout import TS_DIR

FIGURE = TS_DIR / "docs" / "two_tier" / "figures" / "approach_legs.svg"

# ---- panel B's geometry: metres in the runway frame, x along the landing direction from the threshold, y to its left;
# headings in degrees, 0 = the landing direction, counter-clockwise (a left turn raises the heading)
#: MIRROR of the sentence artefact's capture corridor (spec `v5_20260926`: `corridor_half_width_m`,
#: `corridor_widening_deg`, `corridor_course_tolerance_deg`; vocabulary design §2.2, §10) — drawn only.
CORRIDOR_HALF_WIDTH_M = 20.0
CORRIDOR_WIDENING_DEG = 0.45
CORRIDOR_COURSE_TOLERANCE_DEG = 2.0
#: The approach gate (P/CG APPROACH GATE): on the final approach course 1 mile outside the FAF and never closer than
#: 5 miles to the threshold; a 30° intercept must meet the course at least 2 miles outside it (7110.65BB 5-9-1 a,
#: 5-9-2 TBL 5-9-1). The miles are nautical.
GATE_OUTSIDE_FAF_NM, GATE_MIN_NM, INTERCEPT_OUTSIDE_GATE_NM = 1.0, 5.0, 2.0
#: The example: every turn flown at 80 m/s and a 25° bank ...
TURN_SPEED_MPS, TURN_BANK_DEG = 80.0, 25.0
TURN_RADIUS_M = TURN_SPEED_MPS ** 2 / (GRAVITY_MPS2 * math.tan(math.radians(TURN_BANK_DEG)))
#: ... the downwind 5 km to the left of the final approach course, from 4.5 km past the threshold; the base leg ending
#: 2.4 km from the course; the capture 18 km before the threshold; the FAF 10 km before it; the runway 3 km long.
DOWNWIND_Y_M = 5_000.0
DOWNWIND_FROM_X_M = 4_500.0
BASE_END_Y_M = 2_400.0
CAPTURE_X_M = -18_000.0
FAF_X_M = -10_000.0
RUNWAY_LENGTH_M = 3_000.0
#: The straight-in arrival: where it is first drawn, and where it joins the course (a curve tangent to the course there).
STRAIGHT_IN_FROM = (-26_000.0, -1_000.0)
STRAIGHT_IN_JOIN_X_M = -22_000.0


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def on_left_turn(centre: Point, heading_deg: float) -> Point:
    """The point of a left turn about ``centre`` where the heading is ``heading_deg`` (the centre lies 90° to the left
    of the direction of flight)."""
    angle = math.radians(heading_deg - 90.0)
    return Point(centre.x + TURN_RADIUS_M * math.cos(angle), centre.y + TURN_RADIUS_M * math.sin(angle))


def left_turn(centre: Point, from_deg: float, to_deg: float) -> list[Point]:
    """The turn from one heading to another as points of its circle, at most 5° apart, both ends included."""
    steps = max(1, math.ceil((to_deg - from_deg) / 5.0))
    return [on_left_turn(centre, from_deg + (to_deg - from_deg) * k / steps) for k in range(steps + 1)]


@dataclass(frozen=True)
class VectoredPath:
    """The vectored arrival's legs and left turns: downwind (heading 180°) → base (270°) → intercept (360° −
    `ATC_MAX_INTERCEPT_DEG`) → the final approach course (360°)."""

    downwind_start: Point
    downwind_end: Point
    base_turn_centre: Point
    base_start: Point
    base_end: Point
    intercept_turn_centre: Point
    intercept_start: Point
    clearance: Point               # the capture turn's start: where the labeller puts the approach clearance
    capture_turn_centre: Point
    capture: Point                 # on the course: established on the final


def vectored_path() -> VectoredPath:
    """Built backwards from the capture: the capture turn ends on the course at `CAPTURE_X_M`; the intercept leg runs at
    the intercept angle to the course back up to the turn off the base leg, which ends at `BASE_END_Y_M`; the base leg
    runs up to the turn off the downwind at `DOWNWIND_Y_M`."""
    r, intercept = TURN_RADIUS_M, math.radians(ATC_MAX_INTERCEPT_DEG)
    capture = Point(CAPTURE_X_M, 0.0)
    capture_turn_centre = Point(capture.x, r)
    clearance = on_left_turn(capture_turn_centre, 360.0 - ATC_MAX_INTERCEPT_DEG)
    intercept_length = (BASE_END_Y_M - r) / math.sin(intercept)
    base_x = capture.x - r - intercept_length * math.cos(intercept)
    intercept_turn_centre = Point(base_x + r, BASE_END_Y_M)
    base_turn_centre = Point(base_x + r, DOWNWIND_Y_M - r)
    return VectoredPath(
        downwind_start=Point(DOWNWIND_FROM_X_M, DOWNWIND_Y_M), downwind_end=on_left_turn(base_turn_centre, 180.0),
        base_turn_centre=base_turn_centre, base_start=on_left_turn(base_turn_centre, 270.0),
        base_end=on_left_turn(intercept_turn_centre, 270.0), intercept_turn_centre=intercept_turn_centre,
        intercept_start=on_left_turn(intercept_turn_centre, 360.0 - ATC_MAX_INTERCEPT_DEG), clearance=clearance,
        capture_turn_centre=capture_turn_centre, capture=capture)


def interception_point(path: VectoredPath) -> Point:
    """Where the intercept heading, flown on, would meet the final approach course (the point 5-9-2 measures)."""
    return Point(path.clearance.x + path.clearance.y / math.tan(math.radians(ATC_MAX_INTERCEPT_DEG)), 0.0)


def approach_gate_x_m() -> float:
    return -max(-FAF_X_M + GATE_OUTSIDE_FAF_NM * NM_M, GATE_MIN_NM * NM_M)


def straight_in() -> tuple[Point, Point, Point]:
    """``(start, control, join)`` of the straight-in's quadratic curve: the control point on the course, so the curve
    reaches the course along it."""
    start, join = Point(*STRAIGHT_IN_FROM), Point(STRAIGHT_IN_JOIN_X_M, 0.0)
    return start, Point((start.x + join.x) / 2, 0.0), join


def corridor_m(before_threshold_m: float) -> float:
    return float(corridor_half_width_m(before_threshold_m, CORRIDOR_HALF_WIDTH_M, CORRIDOR_WIDENING_DEG))


# ---- drawing
WIDTH = 960
INK, GREY, PALE = "#111827", "#6b7280", "#9ca3af"
BLUE, ORANGE, GREEN = "#2563eb", "#ea580c", "#16a34a"
FONT = "Noto Sans CJK SC, Noto Sans SC, PingFang SC, Microsoft YaHei, sans-serif"

# panel A: schematic units, the threshold at the origin, x along the landing direction, y to its left
A_UNIT_PX, A_ORIGIN_PX = 50.0, (330.0, 215.0)
A_RUNWAY_END, A_PATTERN_FAR, A_PATTERN_NEAR, A_PATTERN_SIDE = 4.0, 7.0, -3.0, 2.6

# panel B: to scale
B_X_MIN_M, B_X_MAX_M, B_Y_MAX_M = -26_000.0, 5_000.0, 5_800.0
B_LEFT_PX, B_TOP_PX = 60.0, 300.0
B_SCALE = (WIDTH - B_LEFT_PX - 40.0) / (B_X_MAX_M - B_X_MIN_M)     # pixels a metre, both axes


def a_px(x: float, y: float) -> tuple[float, float]:
    return A_ORIGIN_PX[0] + x * A_UNIT_PX, A_ORIGIN_PX[1] - y * A_UNIT_PX


def b_px(point: Point) -> tuple[float, float]:
    return B_LEFT_PX + (point.x - B_X_MIN_M) * B_SCALE, B_TOP_PX + (B_Y_MAX_M - point.y) * B_SCALE


def text(x_px: float, y_px: float, body: str, *, size: int = 12, anchor: str = "start", colour: str = INK,
         weight: str = "normal") -> str:
    return (f'<text x="{x_px:.1f}" y="{y_px:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{colour}" '
            f'font-weight="{weight}">{body}</text>')


def line(x1: float, y1: float, x2: float, y2: float, colour: str, *, width: float = 1.0, dash: str = "") -> str:
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}" '
            f'stroke-width="{width}"{dashes}/>')


def arrowhead(x_px: float, y_px: float, dx: float, dy: float, colour: str) -> str:
    """A filled arrowhead at a point, pointing along (dx, dy) in pixels."""
    norm = math.hypot(dx, dy)
    ux, uy = dx / norm, dy / norm
    tip = (x_px + 6 * ux, y_px + 6 * uy)
    left = (x_px - 5 * ux - 4 * uy, y_px - 5 * uy + 4 * ux)
    right = (x_px - 5 * ux + 4 * uy, y_px - 5 * uy - 4 * ux)
    return (f'<path d="M {tip[0]:.1f} {tip[1]:.1f} L {left[0]:.1f} {left[1]:.1f} L {right[0]:.1f} {right[1]:.1f} Z" '
            f'fill="{colour}"/>')


def polyline(points: list[Point], colour: str) -> str:
    coordinates = " ".join(f"{b_px(p)[0]:.1f},{b_px(p)[1]:.1f}" for p in points)
    return f'<polyline points="{coordinates}" fill="none" stroke="{colour}" stroke-width="2"/>'


def dot(point: Point, colour: str) -> str:
    x_px, y_px = b_px(point)
    return f'<circle cx="{x_px:.1f}" cy="{y_px:.1f}" r="4" fill="{colour}"/>'


def panel_a() -> list[str]:
    """The traffic pattern: departure/upwind past the runway, crosswind, downwind, base, final — left-hand."""
    near, far, side = A_PATTERN_NEAR, A_PATTERN_FAR, A_PATTERN_SIDE
    legs = [("五边 final（最后进近）", (near, 0.0), (0.0, 0.0), "below"),
            ("一边 upwind", (A_RUNWAY_END, 0.0), (far, 0.0), "below"),
            ("二边 crosswind", (far, 0.0), (far, side), "right"),
            ("三边 downwind（下风边）：与落地方向相反", (far, side), (near, side), "above"),
            ("四边 base（基线）", (near, side), (near, 0.0), "left")]
    x0, y0 = a_px(0.0, 0.0)
    x1, _ = a_px(A_RUNWAY_END, 0.0)
    parts = [text(40, 52, "A　起落航线的五条边（示意，不按比例；左航线：绕行一侧在落地方向的左边）", size=13, weight="bold"),
             f'<rect x="{x0:.1f}" y="{y0 - 6:.1f}" width="{x1 - x0:.1f}" height="12" fill="#e5e7eb" stroke="{GREY}"/>',
             text((x0 + x1) / 2, y0 + 4, "跑道", size=10, anchor="middle", colour=GREY),
             line(x0, y0 - 9, x0, y0 + 9, INK, width=3),
             text(x0, y0 - 14, "入口", size=11, anchor="middle")]
    for name, start, end, where in legs:
        (xs, ys), (xe, ye) = a_px(*start), a_px(*end)
        parts.append(line(xs, ys, xe, ye, BLUE, width=2))
        parts.append(arrowhead((xs + xe) / 2, (ys + ye) / 2, xe - xs, ye - ys, BLUE))
        mx, my = (xs + xe) / 2, (ys + ye) / 2
        label = {"below": (mx, my + 20, "middle"), "above": (mx, my - 10, "middle"), "right": (mx + 10, my + 4, "start"),
                 "left": (mx - 10, my + 4, "end")}[where]
        parts.append(text(label[0], label[1], name, size=12, anchor=label[2], colour=BLUE))
    fx, fy = a_px(far + 1.2, 0.0)
    parts += [text(fx + 8, fy - 38, "落地方向", size=11, colour=GREY), line(fx + 8, fy - 30, fx + 68, fy - 30, GREY, width=2),
              arrowhead(fx + 70, fy - 30, 1.0, 0.0, GREY),
              text(fx + 8, fy + 20, "起飞后：一边 → 二边", size=11, colour=GREY),
              text(fx + 8, fy + 36, "进场落地：三边 → 四边 → 五边", size=11, colour=GREY)]
    return parts


def panel_b() -> list[str]:
    path = vectored_path()
    threshold = Point(0.0, 0.0)
    intercept = 360.0 - ATC_MAX_INTERCEPT_DEG
    parts = [text(40, B_TOP_PX - 30, "B　雷达引导的进场：本项目数据里的样子（按比例画，只有跑道画宽了）", size=13, weight="bold")]

    # the capture corridor (to scale) and the final approach course: the extended centreline
    far = -B_X_MIN_M
    corridor = [Point(0.0, CORRIDOR_HALF_WIDTH_M), Point(-far, corridor_m(far)), Point(-far, -corridor_m(far)),
                Point(0.0, -CORRIDOR_HALF_WIDTH_M)]
    points = " ".join(f"{b_px(p)[0]:.1f},{b_px(p)[1]:.1f}" for p in corridor)
    parts.append(f'<polygon points="{points}" fill="#bbf7d0" stroke="{GREEN}" stroke-width="0.5"/>')
    (cx0, course_y), (cx1, _) = b_px(Point(B_X_MIN_M, 0.0)), b_px(threshold)
    parts.append(line(cx0, course_y, cx1, course_y, PALE, dash="5,4"))

    # the runway and its threshold; the FAF and the approach gate on the course
    rx1, _ = b_px(Point(RUNWAY_LENGTH_M, 0.0))
    parts += [f'<rect x="{cx1:.1f}" y="{course_y - 4:.1f}" width="{rx1 - cx1:.1f}" height="8" fill="#e5e7eb" '
              f'stroke="{GREY}"/>',
              line(cx1, course_y - 7, cx1, course_y + 7, INK, width=3),
              text(cx1, course_y + 22, "入口", size=11, anchor="middle"),
              text((cx1 + rx1) / 2, course_y + 22, "跑道", size=11, anchor="middle", colour=GREY)]
    faf_x, _ = b_px(Point(FAF_X_M, 0.0))
    gate_x, _ = b_px(Point(approach_gate_x_m(), 0.0))
    parts += [line(faf_x, course_y - 6, faf_x, course_y + 6, INK, width=2),
              text(faf_x + 4, course_y - 10, "FAF（示意）", size=11),
              line(gate_x, course_y - 6, gate_x, course_y + 6, INK, width=2),
              text(gate_x - 4, course_y - 10, "进近门", size=11, anchor="end")]

    # the vectored arrival: legs, and each left turn drawn as points of its computed circle
    route = [path.downwind_start, *left_turn(path.base_turn_centre, 180.0, 270.0),
             *left_turn(path.intercept_turn_centre, 270.0, intercept),
             *left_turn(path.capture_turn_centre, intercept, 360.0), threshold]
    parts.append(polyline(route, BLUE))
    # the straight-in arrival: a quadratic curve reaching the course along it
    start, control, join = straight_in()
    parts.append(f'<path d="M {b_px(start)[0]:.1f} {b_px(start)[1]:.1f} Q {b_px(control)[0]:.1f} {b_px(control)[1]:.1f} '
                 f'{b_px(join)[0]:.1f} {b_px(join)[1]:.1f}" fill="none" stroke="{ORANGE}" stroke-width="2"/>')

    # the direction of flight on each leg, and on the straight-in's curve (the point and its tangent at t = 0.3)
    for a, b in ((path.downwind_start, path.downwind_end), (path.base_start, path.base_end),
                 (path.intercept_start, path.clearance), (Point(-6_000.0, 0.0), Point(-5_000.0, 0.0))):
        (xa, ya), (xb, yb) = b_px(a), b_px(b)
        parts.append(arrowhead((xa + xb) / 2, (ya + yb) / 2, xb - xa, yb - ya, BLUE))
    t = 0.3
    (sx, sy), (kx, ky), (jx, jy) = b_px(start), b_px(control), b_px(join)
    bx = (1 - t) ** 2 * sx + 2 * (1 - t) * t * kx + t ** 2 * jx
    by = (1 - t) ** 2 * sy + 2 * (1 - t) * t * ky + t ** 2 * jy
    parts.append(arrowhead(bx, by, 2 * (1 - t) * (kx - sx) + 2 * t * (jx - kx), 2 * (1 - t) * (ky - sy) + 2 * t * (jy - ky),
                           ORANGE))

    # the two events, and the labels
    parts += [dot(path.clearance, INK), dot(path.capture, GREEN)]
    down_x, down_y = b_px(Point(-6_000.0, DOWNWIND_Y_M))
    base_x, base_y = b_px(Point(path.base_start.x, (path.base_start.y + path.base_end.y) / 2))
    icpt_x, icpt_y = b_px(path.intercept_start)
    clr_x, clr_y = b_px(path.clearance)
    cap_x, cap_y = b_px(path.capture)
    fin_x, _ = b_px(Point(-4_000.0, 0.0))
    parts += [text(down_x, down_y - 10, "三边（下风边）：与落地方向相反，飞过跑道", size=12, anchor="middle", colour=BLUE),
              text(base_x - 10, base_y + 4, "四边（基线）", size=12, anchor="end", colour=BLUE),
              text(icpt_x - 10, icpt_y + 14, f"切入航向：与五边成 {ATC_MAX_INTERCEPT_DEG:g}°", size=12, anchor="end",
                   colour=BLUE),
              text(clr_x + 10, clr_y - 30, "许可加入（本项目标在截获转弯起点）", size=12),
              line(clr_x + 8, clr_y - 26, clr_x + 3, clr_y - 5, INK),
              text(cap_x + 10, cap_y + 28, "截获：进入走廊 = 建立在五边上", size=12, colour=GREEN),
              line(cap_x + 8, cap_y + 16, cap_x + 3, cap_y + 5, GREEN),
              text(fin_x, course_y - 12, "五边：沿中线的延长线下降到入口", size=12, anchor="middle", colour=BLUE),
              text(sx, sy + 22, "直线进近：不飞三边、四边，直接切上五边", size=12, colour=ORANGE)]

    # the landing direction and a scale bar
    lx, ly = b_px(Point(1_500.0, DOWNWIND_Y_M - 1_600.0))
    parts += [text(lx, ly - 8, "落地方向", size=11, colour=GREY), line(lx, ly, lx + 60, ly, GREY, width=2),
              arrowhead(lx + 62, ly, 1.0, 0.0, GREY)]
    (sx0, sb), (sx1, _) = b_px(Point(-5_000.0, -2_600.0)), b_px(Point(0.0, -2_600.0))
    parts += [line(sx0, sb, sx1, sb, INK, width=2), line(sx0, sb - 4, sx0, sb + 4, INK), line(sx1, sb - 4, sx1, sb + 4, INK),
              text((sx0 + sx1) / 2, sb - 6, "5 km", size=11, anchor="middle")]
    return parts


def render() -> str:
    _, bottom = b_px(Point(0.0, -3_000.0))
    notes_y = bottom + 24
    height = int(notes_y + 84)
    far = -B_X_MIN_M
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" '
             f'font-family="{FONT}">',
             f'<rect x="0" y="0" width="{WIDTH}" height="{height}" fill="white" stroke="none"/>',
             text(40, 26, "进场的几段航线：五边是什么（示意，不是真实数据）", size=15, weight="bold"),
             *panel_a(), *panel_b(),
             text(40, notes_y, f"绿色楔形是截获走廊，按比例画：离中线不超过 {CORRIDOR_HALF_WIDTH_M:g} m + d·tan "
                  f"{CORRIDOR_WIDENING_DEG:g}°（d 是离入口的距离，{far / 1000:g} km 处 {corridor_m(far):.0f} m），"
                  f"航迹与跑道航向差不超过 {CORRIDOR_COURSE_TOLERANCE_DEG:g}°。", size=12, colour=GREY),
             text(40, notes_y + 20, f"进近门在 FAF 外 {GATE_OUTSIDE_FAF_NM * NM_M:,.0f} m、离入口不少于 {GATE_MIN_NM * NM_M:,.0f} m；"
                  f"{ATC_MAX_INTERCEPT_DEG:g}° 切入要在进近门外至少 {INTERCEPT_OUTSIDE_GATE_NM * NM_M:,.0f} m 处切上五边"
                  "（7110.65BB 5-9-1、5-9-2）。", size=12, colour=GREY),
             text(40, notes_y + 40, "实际的进近许可常与切入航向一起给（5-9-4 的例子）；录音不在数据里，标注器把「许可加入」放在截获转弯起点"
                  "（词表设计 §2.2）。", size=12, colour=GREY),
             text(40, notes_y + 60, f"转弯半径 {TURN_RADIUS_M:,.0f} m（{TURN_SPEED_MPS:g} m/s、坡度 {TURN_BANK_DEG:g}°）；"
                  f"三边离五边 {DOWNWIND_Y_M / 1000:g} km；每一段转弯都与前后两段相切。", size=12, colour=GREY),
             "</svg>"]
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
    path = vectored_path()
    print(f"wrote {FIGURE}: intercept meets the course at x {interception_point(path).x:,.0f} m, the approach gate at "
          f"{approach_gate_x_m():,.0f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
