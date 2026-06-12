#!/usr/bin/env python3
"""Generate SVG diagrams for the 3-DOF alpha and simplified flight models.

The companion HTML pages intentionally reference generated SVG assets instead
of hand-drawing equations and force decompositions inline. Re-run this script
after changing labels, colors, or force-decomposition geometry.
"""

from __future__ import annotations

from html import escape
from math import cos, pi, sin
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "three_dof_models"


class Svg:
    def __init__(self, width: int, height: int, title: str) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.items: list[str] = []

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        cls: str,
        marker: bool = False,
    ) -> None:
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        self.items.append(
            f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}"{marker_attr}/>'
        )

    def path(self, d: str, cls: str, marker: bool = False) -> None:
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        self.items.append(f'<path class="{cls}" d="{d}"{marker_attr}/>')

    def circle(self, x: float, y: float, r: float, cls: str) -> None:
        self.items.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}"/>')

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, cls: str) -> None:
        self.items.append(
            f'<ellipse class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}"/>'
        )

    def polygon(self, points: list[tuple[float, float]], cls: str) -> None:
        pairs = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.items.append(f'<polygon class="{cls}" points="{pairs}"/>')

    def rect(self, x: float, y: float, w: float, h: float, cls: str) -> None:
        self.items.append(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"/>')

    def text(self, text: str, x: float, y: float, cls: str = "label") -> None:
        self.items.append(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}">{escape(text)}</text>')

    def tspan_text(self, lines: list[str], x: float, y: float, cls: str = "label", dy: int = 22) -> None:
        spans = []
        for i, line in enumerate(lines):
            offset = 0 if i == 0 else dy
            spans.append(f'<tspan x="{x:.1f}" dy="{offset}">{escape(line)}</tspan>')
        self.items.append(f'<text class="{cls}">{"".join(spans)}</text>')

    def render(self) -> str:
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title">
  <title id="title">{escape(self.title)}</title>
  <defs>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L9,3.5 L0,7 Z" fill="#233143"/>
    </marker>
    <linearGradient id="panel" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#eef5f7"/>
    </linearGradient>
  </defs>
  <style>
    svg {{ background: #f6f8fb; }}
    .panel {{ fill: url(#panel); stroke: #c9d5df; stroke-width: 1.2; rx: 14; }}
    .soft {{ fill: #eef6f8; stroke: #bdd6dc; stroke-width: 1.2; rx: 10; }}
    .warnbox {{ fill: #fff4e5; stroke: #e1a154; stroke-width: 1.2; rx: 10; }}
    .aircraft {{ fill: #233143; stroke: #ffffff; stroke-width: 2.2; }}
    .wing {{ fill: #67809a; opacity: .95; }}
    .horizon {{ stroke: #9fb0bf; stroke-width: 2; stroke-dasharray: 7 7; }}
    .path {{ stroke: #007c89; stroke-width: 4; fill: none; }}
    .axis {{ stroke: #233143; stroke-width: 2.4; fill: none; }}
    .force {{ stroke: #b65b13; stroke-width: 4; fill: none; }}
    .lift {{ stroke: #2f62b7; stroke-width: 4; fill: none; }}
    .drag {{ stroke: #8f3f71; stroke-width: 4; fill: none; }}
    .weight {{ stroke: #58687a; stroke-width: 4; fill: none; }}
    .component {{ stroke: #7b8b9b; stroke-width: 2.4; stroke-dasharray: 6 5; fill: none; }}
    .alpha {{ stroke: #bf4b34; stroke-width: 3; fill: none; }}
    .bank {{ stroke: #6f48a8; stroke-width: 3; fill: none; }}
    .flow {{ stroke: #007c89; stroke-width: 2.8; fill: none; }}
    .danger {{ stroke: #b7352d; stroke-width: 3.2; fill: none; }}
    .dot {{ fill: #007c89; stroke: #ffffff; stroke-width: 2; }}
    .title {{ fill: #14202f; font: 700 24px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .subtitle {{ fill: #52677a; font: 500 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .label {{ fill: #182538; font: 650 15px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .small {{ fill: #52677a; font: 500 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .formula {{ fill: #182538; font: 600 17px "Times New Roman", serif; }}
    .mono {{ fill: #233143; font: 600 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .orange {{ fill: #9a4c10; font: 700 15px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .blue {{ fill: #2f62b7; font: 700 15px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .violet {{ fill: #7b3f93; font: 700 15px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .red {{ fill: #b7352d; font: 700 15px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  </style>
  {''.join(self.items)}
</svg>
"""


def arrow_vec(x: float, y: float, length: float, angle_deg: float) -> tuple[float, float]:
    a = angle_deg * pi / 180.0
    return x + length * cos(a), y - length * sin(a)


def draw_aircraft(svg: Svg, cx: float, cy: float, angle_deg: float) -> None:
    a = angle_deg * pi / 180.0
    ca, sa = cos(a), sin(a)

    def rot(px: float, py: float) -> tuple[float, float]:
        return cx + px * ca - py * sa, cy - (px * sa + py * ca)

    fuselage = [rot(48, 0), rot(12, 11), rot(-42, 9), rot(-54, 0), rot(-42, -9), rot(12, -11)]
    wing = [rot(10, 0), rot(-12, 36), rot(-23, 34), rot(-8, 0), rot(-23, -34), rot(-12, -36)]
    tail = [rot(-34, 0), rot(-50, 22), rot(-56, 18), rot(-46, 0), rot(-56, -18), rot(-50, -22)]
    svg.polygon(wing, "wing")
    svg.polygon(tail, "wing")
    svg.polygon(fuselage, "aircraft")


def alpha_model_diagram() -> str:
    svg = Svg(1120, 760, "Alpha-input 3-DOF model force decomposition")
    svg.rect(34, 34, 1052, 692, "panel")
    svg.text("完整模型：以攻角 α 为输入，先计算 L 与 D，再投影到轨迹方向", 64, 78, "title")
    svg.text("风轴：切向 eV 决定 Vdot；法向合力分解为航向转弯方向与飞行路径角方向", 66, 104, "subtitle")

    ox, oy = 365, 402
    gamma = 16
    body_pitch = 27
    vx, vy = arrow_vec(ox, oy, 175, gamma)
    gx, gy = arrow_vec(ox, oy, 118, gamma + 90)
    px, py = arrow_vec(ox, oy, 108, gamma - 90)
    bx, by = arrow_vec(ox, oy, 128, body_pitch)

    svg.line(86, oy + 38, 642, oy + 38, "horizon")
    svg.text("local horizon", 92, oy + 30, "small")
    svg.line(ox - 110, oy + 31, ox + 225, oy - 63, "path")
    svg.line(ox, oy, vx, vy, "axis", True)
    svg.text("V, eV", vx + 12, vy + 8, "label")
    svg.path(f"M{ox + 76},{oy + 39} A82,82 0 0,0 {ox + 76},{oy + 18}", "alpha")
    svg.text("γ", ox + 92, oy + 33, "blue")

    draw_aircraft(svg, ox, oy, body_pitch)
    svg.line(ox, oy, bx, by, "force", True)
    svg.path(f"M{ox + 58},{oy - 18} A72,72 0 0,0 {ox + 52},{oy - 32}", "alpha")
    svg.text("α", ox + 68, oy - 22, "red")

    svg.line(ox, oy, *arrow_vec(ox, oy, 150, body_pitch), "force", True)
    svg.text("T", ox + 130, oy - 68, "orange")
    svg.line(ox, oy, *arrow_vec(ox, oy, 132, gamma), "component", True)
    svg.text("T cos α", ox + 128, oy - 20, "small")
    svg.line(ox, oy, *arrow_vec(ox, oy, 92, gamma + 90), "component", True)
    svg.text("T sin α", ox - 58, oy - 94, "small")

    svg.line(ox, oy, *arrow_vec(ox, oy, 128, gamma + 180), "drag", True)
    svg.text("D", ox - 150, oy + 30, "violet")
    svg.line(ox, oy, gx, gy, "lift", True)
    svg.text("L", gx - 22, gy - 10, "blue")
    svg.line(ox, oy, ox, oy + 150, "weight", True)
    svg.text("mg", ox + 10, oy + 146, "small")
    svg.line(ox, oy, *arrow_vec(ox, oy, 88, gamma + 180), "component", True)
    svg.text("mg sin γ", ox - 115, oy + 62, "small")
    svg.line(ox, oy, px, py, "component", True)
    svg.text("mg cos γ", px - 28, py + 28, "small")

    cx, cy = 760, 340
    svg.ellipse(cx, cy, 145, 72, "soft")
    svg.line(cx, cy, cx, cy - 112, "lift", True)
    svg.text("(L + T sin α) cos φ", cx + 14, cy - 98, "blue")
    svg.line(cx, cy, cx + 130, cy + 52, "lift", True)
    svg.text("(L + T sin α) sin φ", cx + 110, cy + 70, "blue")
    svg.line(cx, cy, cx + 84, cy - 74, "axis", True)
    svg.text("L + T sin α", cx + 72, cy - 84, "label")
    svg.path(f"M{cx + 2},{cy - 50} A58,58 0 0,1 {cx + 46},{cy - 39}", "bank")
    svg.text("φ", cx + 38, cy - 54, "violet")
    svg.text("法向合力的 bank 分解", cx - 112, cy + 108, "label")

    svg.rect(680, 470, 360, 188, "soft")
    svg.tspan_text(
        [
            "L = 1/2 ρ V² S CL(α)",
            "D = 1/2 ρ V² S CD(α)",
            "Vdot = (T cosα - D)/m - g sinγ",
            "ψdot = ((L + T sinα) sinφ)/(mV cosγ)",
            "γdot = ((L + T sinα) cosφ - mg cosγ)/(mV)",
        ],
        704,
        506,
        "formula",
        27,
    )
    return svg.render()


def simplified_model_diagram() -> str:
    svg = Svg(1120, 760, "Simplified load-factor 3-DOF model force decomposition")
    svg.rect(34, 34, 1052, 692, "panel")
    svg.text("简化模型：以载荷因子 n 为输入，直接令 L = nmg", 64, 78, "title")
    svg.text("假设 α≈0、推力沿速度方向；n 是控制命令，但 stall 时只能实现 n_actual", 66, 104, "subtitle")

    ox, oy = 365, 402
    gamma = 12
    vx, vy = arrow_vec(ox, oy, 172, gamma)
    nx, ny = arrow_vec(ox, oy, 130, gamma + 90)
    wx, wy = arrow_vec(ox, oy, 118, gamma - 90)
    svg.line(84, oy + 32, 650, oy + 32, "horizon")
    svg.line(ox - 110, oy + 23, ox + 225, oy - 47, "path")
    svg.line(ox, oy, vx, vy, "axis", True)
    svg.text("V", vx + 12, vy + 4, "label")
    draw_aircraft(svg, ox, oy, gamma)

    svg.line(ox, oy, *arrow_vec(ox, oy, 150, gamma), "force", True)
    svg.text("T", ox + 132, oy - 28, "orange")
    svg.line(ox, oy, *arrow_vec(ox, oy, 125, gamma + 180), "drag", True)
    svg.text("D", ox - 146, oy + 25, "violet")
    svg.line(ox, oy, nx, ny, "lift", True)
    svg.text("L = nmg", nx - 50, ny - 8, "blue")
    svg.line(ox, oy, ox, oy + 148, "weight", True)
    svg.text("mg", ox + 10, oy + 146, "small")
    svg.line(ox, oy, *arrow_vec(ox, oy, 88, gamma + 180), "component", True)
    svg.text("mg sin γ", ox - 115, oy + 54, "small")
    svg.line(ox, oy, wx, wy, "component", True)
    svg.text("mg cos γ", wx - 24, wy + 28, "small")

    cx, cy = 765, 334
    svg.ellipse(cx, cy, 142, 72, "soft")
    svg.line(cx, cy, cx, cy - 112, "lift", True)
    svg.text("nmg cos μ", cx + 14, cy - 94, "blue")
    svg.line(cx, cy, cx + 130, cy + 50, "lift", True)
    svg.text("nmg sin μ", cx + 108, cy + 68, "blue")
    svg.line(cx, cy, cx + 84, cy - 74, "axis", True)
    svg.text("nmg", cx + 74, cy - 86, "label")
    svg.path(f"M{cx + 3},{cy - 49} A57,57 0 0,1 {cx + 46},{cy - 39}", "bank")
    svg.text("μ", cx + 38, cy - 54, "violet")
    svg.text("载荷因子的 bank 分解", cx - 106, cy + 108, "label")

    svg.rect(680, 470, 360, 188, "soft")
    svg.tspan_text(
        [
            "L = nmg",
            "D = 1/2 ρ V² S CD(CL_req)",
            "Vdot = (T - D)/m - g sinγ",
            "ψdot = g n sinμ/(V cosγ)",
            "γdot = g(n cosμ - cosγ)/V",
        ],
        704,
        506,
        "formula",
        27,
    )
    return svg.render()


def bridge_stall_diagram() -> str:
    svg = Svg(1120, 720, "Bridge between alpha and load-factor models with stall conditions")
    svg.rect(34, 34, 1052, 652, "panel")
    svg.text("两种模型的连接：α 决定气动能力，n 是简化模型的升力需求", 64, 78, "title")
    svg.text("同一个升力方程把完整模型和简化模型连接起来；stall 是能力边界，不只是告警。", 66, 104, "subtitle")

    svg.rect(88, 166, 286, 128, "soft")
    svg.text("完整模型输入", 112, 202, "label")
    svg.text("u = [T, φ, α]", 112, 236, "formula")
    svg.text("α → CL(α) → L", 112, 268, "formula")

    svg.rect(418, 166, 286, 128, "soft")
    svg.text("共同升力方程", 442, 202, "label")
    svg.text("L = 1/2 ρ V² S CL", 442, 236, "formula")
    svg.text("n = L/(mg)", 442, 268, "formula")

    svg.rect(748, 166, 286, 128, "soft")
    svg.text("简化模型输入", 772, 202, "label")
    svg.text("u = [T, μ, n_cmd]", 772, 236, "formula")
    svg.text("n_cmd → CL_req → α_req", 772, 268, "formula")

    svg.line(374, 230, 418, 230, "flow", True)
    svg.line(704, 230, 748, 230, "flow", True)
    svg.path("M748,274 C650,362 508,362 374,274", "flow", True)
    svg.text("反推关系", 528, 352, "label")

    svg.rect(116, 404, 412, 166, "warnbox")
    svg.text("完整模型 stall", 142, 438, "red")
    svg.tspan_text(
        [
            "α ≥ αcrit",
            "或 CL(α) ≥ CLmax",
            "动力学中使用 post-stall 的 CL_actual 与 CD_actual",
        ],
        142,
        476,
        "formula",
        28,
    )

    svg.rect(592, 404, 412, 166, "warnbox")
    svg.text("简化模型 stall", 618, 438, "red")
    svg.tspan_text(
        [
            "CL_req = 2mgn_cmd/(ρV²S)",
            "α_req = (CL_req - CL0)/CLα",
            "n_actual = min(n_cmd, nmax) 并增加 stall drag",
        ],
        618,
        476,
        "formula",
        28,
    )

    svg.line(360, 570, 360, 618, "danger", True)
    svg.line(826, 570, 826, 618, "danger", True)
    svg.text("轨迹影响：V 下降、γ 下降、转弯能力低于命令值", 304, 650, "red")
    return svg.render()


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    diagrams = {
        "alpha-model-force-decomposition.svg": alpha_model_diagram(),
        "simplified-model-force-decomposition.svg": simplified_model_diagram(),
        "model-bridge-stall.svg": bridge_stall_diagram(),
    }
    for filename, content in diagrams.items():
        (ASSET_DIR / filename).write_text(content, encoding="utf-8")
    print(f"Wrote {len(diagrams)} diagrams to {ASSET_DIR}")


if __name__ == "__main__":
    main()
