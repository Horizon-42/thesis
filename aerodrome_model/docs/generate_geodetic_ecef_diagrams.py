#!/usr/bin/env python3
"""Generate WGS 84 geodetic/ECEF teaching diagrams as deterministic SVG.

The generated diagrams avoid hand-placed ellipses and arrows. Geometry is
computed from WGS 84 constants, geodetic/ECEF formulas, and an orthographic
camera projection. Some arrows use an explicit visual height scale so the
normal direction can be seen on a page-sized diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import atan, atan2, cos, degrees, pi, sin, sqrt, tan
from pathlib import Path
from typing import Iterable, Sequence


WGS84_A_M = 6_378_137.0
WGS84_INV_F = 298.257223563
WGS84_F = 1.0 / WGS84_INV_F
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_B_OVER_A = 1.0 - WGS84_F

TEACHING_PHI_DEG = 53.809394444
TEACHING_LAMBDA_DEG = 35.0
TEACHING_PHI = TEACHING_PHI_DEG * pi / 180.0
TEACHING_LAMBDA = TEACHING_LAMBDA_DEG * pi / 180.0
VISUAL_H_OVER_A = 0.16

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "geodetic_ecef"


Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(v: Vec3) -> float:
    return sqrt(dot(v, v))


def unit(v: Vec3) -> Vec3:
    n = norm(v)
    return (v[0] / n, v[1] / n, v[2] / n)


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(s: float, v: Vec3) -> Vec3:
    return (s * v[0], s * v[1], s * v[2])


def nu_over_a(phi: float) -> float:
    return 1.0 / sqrt(1.0 - WGS84_E2 * sin(phi) ** 2)


def geodetic_surface_unit(phi: float, lam: float) -> Vec3:
    nu = nu_over_a(phi)
    return (
        nu * cos(phi) * cos(lam),
        nu * cos(phi) * sin(lam),
        (1.0 - WGS84_E2) * nu * sin(phi),
    )


def normal_unit(phi: float, lam: float) -> Vec3:
    return (cos(phi) * cos(lam), cos(phi) * sin(lam), sin(phi))


def east_unit(lam: float) -> Vec3:
    return (-sin(lam), cos(lam), 0.0)


def north_unit(phi: float, lam: float) -> Vec3:
    return (-sin(phi) * cos(lam), -sin(phi) * sin(lam), cos(phi))


def geodetic_to_ecef_m(phi: float, lam: float, h_m: float) -> Vec3:
    nu = WGS84_A_M / sqrt(1.0 - WGS84_E2 * sin(phi) ** 2)
    return (
        (nu + h_m) * cos(phi) * cos(lam),
        (nu + h_m) * cos(phi) * sin(lam),
        ((1.0 - WGS84_E2) * nu + h_m) * sin(phi),
    )


@dataclass
class Camera:
    eye: Vec3 = (1.7, -2.4, 1.25)

    def __post_init__(self) -> None:
        self.view = unit(self.eye)
        self.x_axis = unit(cross((0.0, 0.0, 1.0), self.view))
        self.y_axis = unit(cross(self.view, self.x_axis))

    def raw_project(self, p: Vec3) -> Vec2:
        return (dot(p, self.x_axis), dot(p, self.y_axis))


class Svg:
    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        world_points: Sequence[Vec3],
        camera: Camera | None = None,
        pad: int = 54,
    ) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.camera = camera or Camera()
        raws = [self.camera.raw_project(p) for p in world_points]
        xs = [p[0] for p in raws]
        ys = [p[1] for p in raws]
        x_span = max(xs) - min(xs) or 1.0
        y_span = max(ys) - min(ys) or 1.0
        self.scale = min((width - 2 * pad) / x_span, (height - 2 * pad) / y_span)
        self.x_mid = (max(xs) + min(xs)) / 2.0
        self.y_mid = (max(ys) + min(ys)) / 2.0
        self.items: list[str] = []

    def p(self, world: Vec3) -> Vec2:
        x, y = self.camera.raw_project(world)
        return (
            self.width / 2.0 + (x - self.x_mid) * self.scale,
            self.height / 2.0 - (y - self.y_mid) * self.scale,
        )

    def line(self, a: Vec3, b: Vec3, cls: str, marker: bool = False) -> None:
        ax, ay = self.p(a)
        bx, by = self.p(b)
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        self.items.append(
            f'<line class="{cls}" x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}"{marker_attr}/>'
        )

    def polyline(self, pts: Iterable[Vec3], cls: str) -> None:
        pairs = " ".join(f"{x:.1f},{y:.1f}" for x, y in (self.p(p) for p in pts))
        self.items.append(f'<polyline class="{cls}" points="{pairs}"/>')

    def circle(self, p: Vec3, r: float, cls: str) -> None:
        x, y = self.p(p)
        self.items.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}"/>')

    def text(self, text: str, p: Vec3, dx: float, dy: float, cls: str = "label") -> None:
        x, y = self.p(p)
        self.items.append(
            f'<text class="{cls}" x="{x + dx:.1f}" y="{y + dy:.1f}">{escape(text)}</text>'
        )

    def text_xy(self, text: str, x: float, y: float, cls: str = "label") -> None:
        self.items.append(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}">{escape(text)}</text>')

    def raw(self, item: str) -> None:
        self.items.append(item)

    def render(self) -> str:
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title">
  <title id="title">{escape(self.title)}</title>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#263548"/>
    </marker>
  </defs>
  <style>
    svg {{ background: linear-gradient(180deg, #fbfdff 0%, #f3f7fa 100%); }}
    .title {{ fill: #16212f; font: 700 22px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .subtitle {{ fill: #5f6f7e; font: 500 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .wire {{ fill: none; stroke: #6aa7ad; stroke-width: 1.3; opacity: .58; }}
    .equator {{ fill: none; stroke: #b65b13; stroke-width: 2.2; }}
    .meridian {{ fill: none; stroke: #3d58a8; stroke-width: 2.0; }}
    .axis {{ stroke: #263548; stroke-width: 2.4; }}
    .vector {{ stroke: #3d58a8; stroke-width: 2.4; fill: none; }}
    .normal {{ stroke: #b65b13; stroke-width: 2.7; fill: none; }}
    .helper {{ stroke: #8292a4; stroke-width: 1.5; stroke-dasharray: 6 5; fill: none; }}
.component {{ stroke: #007c89; stroke-width: 2.4; fill: none; }}
.east {{ stroke: #007c89; stroke-width: 2.7; fill: none; }}
.north {{ stroke: #3d58a8; stroke-width: 2.7; fill: none; }}
.curvature {{ fill: none; stroke: #b65b13; stroke-width: 1.8; stroke-dasharray: 7 5; }}
.plane {{ fill: #f7e7d6; opacity: .62; stroke: #b65b13; stroke-width: 1.2; }}
    .point {{ fill: #007c89; stroke: white; stroke-width: 2.0; }}
    .surface {{ fill: #b65b13; stroke: white; stroke-width: 2.0; }}
    .label {{ fill: #16212f; font: 650 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .small {{ fill: #4b5d70; font: 500 12.5px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .teal {{ fill: #006f79; font: 650 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .orange {{ fill: #9a4c10; font: 650 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .indigo {{ fill: #314b99; font: 650 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .formula {{ fill: #263548; font: 600 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  </style>
  {''.join(self.items)}
</svg>
"""


def ellipsoid_lat(phi: float, samples: int = 145) -> list[Vec3]:
    return [geodetic_surface_unit(phi, 2.0 * pi * i / (samples - 1)) for i in range(samples)]


def ellipsoid_lon(lam: float, samples: int = 121) -> list[Vec3]:
    return [geodetic_surface_unit(-pi / 2.0 + pi * i / (samples - 1), lam) for i in range(samples)]


def wire_points() -> list[Vec3]:
    pts: list[Vec3] = []
    for phi_deg in (-60, -30, 0, 30, 60):
        pts.extend(ellipsoid_lat(phi_deg * pi / 180.0))
    for lam_deg in (0, 45, 90, 135, 180, 225, 270, 315, TEACHING_LAMBDA_DEG):
        pts.extend(ellipsoid_lon(lam_deg * pi / 180.0))
    return pts


def draw_wire(svg: Svg) -> None:
    for phi_deg in (-60, -30, 30, 60):
        svg.polyline(ellipsoid_lat(phi_deg * pi / 180.0), "wire")
    for lam_deg in (45, 90, 135, 180, 225, 270, 315):
        svg.polyline(ellipsoid_lon(lam_deg * pi / 180.0), "wire")
    svg.polyline(ellipsoid_lat(0.0), "equator")
    svg.polyline(ellipsoid_lon(0.0), "meridian")
    svg.polyline(ellipsoid_lon(TEACHING_LAMBDA), "meridian")


def tangent_plane_polygon(s: Vec3, phi: float, lam: float, size: float = 0.18) -> list[Vec3]:
    e = east_unit(lam)
    n = north_unit(phi, lam)
    return [
        add(s, add(mul(-size, e), mul(-size * 0.7, n))),
        add(s, add(mul(size, e), mul(-size * 0.7, n))),
        add(s, add(mul(size, e), mul(size * 0.7, n))),
        add(s, add(mul(-size, e), mul(size * 0.7, n))),
    ]


def polygon_points(svg: Svg, pts: Sequence[Vec3], cls: str) -> None:
    pairs = " ".join(f"{x:.1f},{y:.1f}" for x, y in (svg.p(p) for p in pts))
    svg.raw(f'<polygon class="{cls}" points="{pairs}"/>')


def diagram_coordinate_system() -> str:
    s = geodetic_surface_unit(TEACHING_PHI, TEACHING_LAMBDA)
    n = normal_unit(TEACHING_PHI, TEACHING_LAMBDA)
    p = add(s, mul(VISUAL_H_OVER_A, n))
    pts = wire_points() + [(0, 0, 0), (1.45, 0, 0), (0, 1.45, 0), (0, 0, 1.35), s, p]
    svg = Svg(900, 560, "WGS 84 ellipsoid and ECEF axes generated from formulas", pts)
    svg.text_xy("生成式 3D 图：WGS 84 椭球、大地坐标与 ECEF", 30, 36, "title")
    svg.text_xy("教学点 φ=53.809394°，λ=35°；h 箭头沿精确法线方向放大显示", 30, 58, "subtitle")
    draw_wire(svg)
    svg.line((0, 0, 0), (1.45, 0, 0), "axis", True)
    svg.line((0, 0, 0), (0, 1.45, 0), "axis", True)
    svg.line((0, 0, 0), (0, 0, 1.35), "axis", True)
    polygon_points(svg, tangent_plane_polygon(s, TEACHING_PHI, TEACHING_LAMBDA), "plane")
    svg.line((0, 0, 0), s, "vector", True)
    svg.line(s, p, "normal", True)
    svg.circle(s, 5.8, "surface")
    svg.circle(p, 6.6, "point")
    svg.text("X", (1.45, 0, 0), 8, 4)
    svg.text("Y", (0, 1.45, 0), 6, 8)
    svg.text("Z", (0, 0, 1.35), 8, -2)
    svg.text("O 地心", (0, 0, 0), 6, 16, "small")
    svg.text("P(φ, λ, h)", p, 8, -6, "teal")
    svg.text("S 椭球面法线脚点", s, 10, 14, "orange")
    svg.text("h·n̂", add(s, mul(VISUAL_H_OVER_A * 0.55, n)), 10, -8, "orange")
    svg.text("子午线 λ", geodetic_surface_unit(0.45, TEACHING_LAMBDA), 8, -4, "indigo")
    svg.text("赤道", geodetic_surface_unit(0, 2.25), 4, 16, "orange")
    return svg.render()


def svg_2d_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">',
        f'<title id="title">{escape(title)}</title>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#263548"/></marker></defs>',
        """<style>
svg { background: linear-gradient(180deg, #fbfdff 0%, #f3f7fa 100%); }
.title { fill: #16212f; font: 700 22px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.subtitle { fill: #5f6f7e; font: 500 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.axis { stroke: #263548; stroke-width: 2.2; marker-end: url(#arrow); }
.ellipse { fill: #e7f7f8; stroke: #207b84; stroke-width: 2.2; }
.helper { stroke: #8292a4; stroke-width: 1.5; stroke-dasharray: 6 5; fill: none; }
.normal { stroke: #b65b13; stroke-width: 2.7; marker-end: url(#arrow); }
.radius { stroke: #3d58a8; stroke-width: 2.4; marker-end: url(#arrow); }
.tangent { stroke: #3d58a8; stroke-width: 2.0; }
.point { fill: #007c89; stroke: white; stroke-width: 2.0; }
.surface { fill: #b65b13; stroke: white; stroke-width: 2.0; }
.label { fill: #16212f; font: 650 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.small { fill: #4b5d70; font: 500 12.5px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.orange { fill: #9a4c10; font: 650 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.indigo { fill: #314b99; font: 650 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.formula { fill: #263548; font: 600 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
</style>""",
    ]


def diagram_latitudes() -> str:
    width, height = 900, 560
    cx, cy = 410, 305
    scale = 255
    b = WGS84_B_OVER_A
    phi = TEACHING_PHI
    p_s = nu_over_a(phi) * cos(phi)
    z_s = (1.0 - WGS84_E2) * nu_over_a(phi) * sin(phi)
    psi = atan2(z_s, p_s)
    sx, sy = cx + p_s * scale, cy - z_s * scale
    nx, ny = cos(phi), -sin(phi)
    rx, ry = p_s, -z_s
    tangent = (-sin(phi), -cos(phi))
    out = svg_2d_header(width, height, "Geodetic latitude and geocentric latitude")
    out.append('<text class="title" x="30" y="36">精确剖面图：大地纬度 φ 与地心纬度 ψ</text>')
    out.append(f'<text class="subtitle" x="30" y="58">WGS 84 真实扁率；φ={TEACHING_PHI_DEG:.6f}°，ψ={degrees(psi):.6f}°，差值={TEACHING_PHI_DEG - degrees(psi):.6f}°</text>')
    out.append(f'<ellipse class="ellipse" cx="{cx}" cy="{cy}" rx="{scale}" ry="{scale * b:.1f}"/>')
    out.append(f'<line class="axis" x1="{cx - 310}" y1="{cy}" x2="{cx + 335}" y2="{cy}"/>')
    out.append(f'<line class="axis" x1="{cx}" y1="{cy + 285}" x2="{cx}" y2="{cy - 300}"/>')
    out.append(f'<line class="helper" x1="{sx:.1f}" y1="{sy:.1f}" x2="{sx:.1f}" y2="{cy:.1f}"/>')
    out.append(f'<line class="helper" x1="{cx:.1f}" y1="{sy:.1f}" x2="{sx:.1f}" y2="{sy:.1f}"/>')
    out.append(f'<line class="radius" x1="{cx}" y1="{cy}" x2="{sx:.1f}" y2="{sy:.1f}"/>')
    out.append(f'<line class="normal" x1="{sx:.1f}" y1="{sy:.1f}" x2="{sx + nx * 150:.1f}" y2="{sy + ny * 150:.1f}"/>')
    out.append(f'<line class="tangent" x1="{sx - tangent[0] * 120:.1f}" y1="{sy - tangent[1] * 120:.1f}" x2="{sx + tangent[0] * 120:.1f}" y2="{sy + tangent[1] * 120:.1f}"/>')
    out.append(f'<circle class="surface" cx="{sx:.1f}" cy="{sy:.1f}" r="6"/>')
    out.append(f'<text class="label" x="{cx + 342}" y="{cy + 6}">p</text>')
    out.append(f'<text class="label" x="{cx + 10}" y="{cy - 303}">z</text>')
    out.append(f'<text class="small" x="{cx + 8}" y="{cy + 18}">O</text>')
    out.append(f'<text class="orange" x="{sx + nx * 100 + 8:.1f}" y="{sy + ny * 100:.1f}">椭球法线 n̂</text>')
    out.append(f'<text class="indigo" x="{cx + rx * scale * 0.45:.1f}" y="{cy + ry * scale * 0.45 - 8:.1f}">地心方向</text>')
    out.append(f'<text class="label" x="{sx + 10:.1f}" y="{sy - 8:.1f}">S(p_S,z_S)</text>')
    out.append(f'<text class="small" x="{sx + 10:.1f}" y="{cy - 8:.1f}">p_S</text>')
    out.append(f'<text class="small" x="{sx + 8:.1f}" y="{(sy + cy) / 2:.1f}">z_S</text>')
    out.append(f'<text class="indigo" x="{cx + 120}" y="{cy - 28}">ψ = atan2(z_S,p_S)</text>')
    out.append(f'<text class="orange" x="{cx + 300}" y="{cy - 100}">φ = 法线角</text>')
    out.append(f'<text class="formula" x="42" y="512">tan ψ = z_S/p_S；tan φ = z_S / ((1-e²)p_S)。球体 e²=0 时二者才相同。</text>')
    out.append("</svg>")
    return "\n".join(out)


def meridian_radius_over_a(phi: float) -> float:
    return (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * sin(phi) ** 2) ** 1.5


def circle3d(center: Vec3, radius: float, u: Vec3, v: Vec3, samples: int = 145) -> list[Vec3]:
    return [
        add(center, add(mul(radius * cos(2.0 * pi * i / (samples - 1)), u), mul(radius * sin(2.0 * pi * i / (samples - 1)), v)))
        for i in range(samples)
    ]


def diagram_curvature_radii() -> str:
    phi = TEACHING_PHI
    lam = TEACHING_LAMBDA
    s = geodetic_surface_unit(phi, lam)
    n_out = normal_unit(phi, lam)
    n_in = mul(-1.0, n_out)
    east = east_unit(lam)
    north = north_unit(phi, lam)
    nu = nu_over_a(phi)
    meridian = meridian_radius_over_a(phi)
    center_nu = add(s, mul(nu, n_in))
    center_m = add(s, mul(meridian, n_in))
    east_circle = circle3d(center_nu, nu, east, n_out)
    north_circle = circle3d(center_m, meridian, north, n_out)
    pts = (
        wire_points()
        + east_circle
        + north_circle
        + [s, center_nu, center_m, add(s, mul(0.42, east)), add(s, mul(0.42, north)), add(s, mul(0.46, n_out))]
    )
    svg = Svg(900, 600, "Meridian and prime vertical radii of curvature", pts, pad=42)
    svg.text_xy("生成式 3D 图：南北与东西方向曲率半径", 30, 36, "title")
    svg.text_xy(
        f"教学点 φ={TEACHING_PHI_DEG:.6f}°；M={meridian * WGS84_A_M:,.3f} m，ν={nu * WGS84_A_M:,.3f} m",
        30,
        58,
        "subtitle",
    )
    draw_wire(svg)
    svg.polyline(north_circle, "north")
    svg.polyline(east_circle, "east")
    svg.polyline(circle3d(center_m, meridian * 0.08, north, n_out, samples=60), "curvature")
    svg.polyline(circle3d(center_nu, nu * 0.08, east, n_out, samples=60), "curvature")
    svg.line(s, add(s, mul(0.42, north)), "north", True)
    svg.line(s, add(s, mul(0.42, east)), "east", True)
    svg.line(s, add(s, mul(0.46, n_out)), "normal", True)
    svg.line(s, center_m, "helper")
    svg.line(s, center_nu, "helper")
    svg.circle(s, 6.5, "point")
    svg.circle(center_m, 5.0, "surface")
    svg.circle(center_nu, 5.0, "surface")
    svg.text("S", s, 8, -8, "teal")
    svg.text("north tangent：南北方向", add(s, mul(0.42, north)), 10, -4, "indigo")
    svg.text("east tangent：东西方向", add(s, mul(0.42, east)), 10, 12, "teal")
    svg.text("outward normal n̂", add(s, mul(0.46, n_out)), 10, -6, "orange")
    svg.text("M 子午圈曲率半径", add(s, mul(0.48, n_in)), -152, 8, "indigo")
    svg.text("ν 卯酉圈曲率半径", add(s, mul(0.75, n_in)), 10, -8, "orange")
    svg.text_xy("M：meridian radius of curvature，沿南北方向的法截线曲率半径。", 44, 540, "formula")
    svg.text_xy("ν/N：prime vertical radius of curvature，沿东西方向的法截线曲率半径；不是地心半径。", 44, 564, "formula")
    return svg.render()


def diagram_forward() -> str:
    s = geodetic_surface_unit(TEACHING_PHI, TEACHING_LAMBDA)
    n = normal_unit(TEACHING_PHI, TEACHING_LAMBDA)
    p = add(s, mul(VISUAL_H_OVER_A, n))
    q = (p[0], p[1], 0.0)
    x_comp = (p[0], 0.0, 0.0)
    y_comp = (0.0, p[1], 0.0)
    pts = wire_points() + [(0, 0, 0), (1.45, 0, 0), (0, 1.45, 0), (0, 0, 1.35), s, p, q, x_comp, y_comp]
    svg = Svg(900, 580, "Forward conversion geodetic to ECEF", pts)
    svg.text_xy("生成式 3D 坐标变换：大地坐标 → ECEF", 30, 36, "title")
    svg.text_xy("P = S + h·n̂；S 和 n̂ 由 φ、λ、WGS 84 椭球精确计算", 30, 58, "subtitle")
    draw_wire(svg)
    for end, label in [((1.45, 0, 0), "X"), ((0, 1.45, 0), "Y"), ((0, 0, 1.35), "Z")]:
        svg.line((0, 0, 0), end, "axis", True)
        svg.text(label, end, 8, 4)
    svg.line(s, p, "normal", True)
    svg.line((0, 0, 0), q, "component", True)
    svg.line(q, p, "helper")
    svg.line((0, 0, 0), x_comp, "helper")
    svg.line((0, 0, 0), y_comp, "helper")
    svg.line(x_comp, q, "component", True)
    svg.line(y_comp, q, "component", True)
    svg.circle(s, 5.8, "surface")
    svg.circle(q, 5.2, "surface")
    svg.circle(p, 6.6, "point")
    svg.text("S(φ,λ,h=0)", s, 8, 16, "orange")
    svg.text("P", p, 9, -6, "teal")
    svg.text("Q=(X,Y,0)", q, 8, 14, "teal")
    svg.text("h·n̂", add(s, mul(VISUAL_H_OVER_A * 0.55, n)), 10, -8, "orange")
    svg.text("p=sqrt(X²+Y²)", q, 18, -8, "teal")
    svg.text_xy("X=(ν+h)cosφcosλ", 44, 520, "formula")
    svg.text_xy("Y=(ν+h)cosφsinλ", 44, 540, "formula")
    svg.text_xy("Z=((1-e²)ν+h)sinφ", 44, 560, "formula")
    return svg.render()


def diagram_inverse() -> str:
    s = geodetic_surface_unit(TEACHING_PHI, TEACHING_LAMBDA)
    n = normal_unit(TEACHING_PHI, TEACHING_LAMBDA)
    p = add(s, mul(VISUAL_H_OVER_A, n))
    q = (p[0], p[1], 0.0)
    pts = wire_points() + [(0, 0, 0), (1.45, 0, 0), (0, 1.45, 0), (0, 0, 1.35), s, p, q]
    svg = Svg(900, 580, "Inverse conversion ECEF to geodetic", pts)
    svg.text_xy("生成式 3D 坐标变换：ECEF → 大地坐标", 30, 36, "title")
    svg.text_xy("已知 P(X,Y,Z)，先求 Q 和 λ，再寻找使 P-S 平行 n̂ 的椭球法线脚点 S", 30, 58, "subtitle")
    draw_wire(svg)
    for end, label in [((1.45, 0, 0), "X"), ((0, 1.45, 0), "Y"), ((0, 0, 1.35), "Z")]:
        svg.line((0, 0, 0), end, "axis", True)
        svg.text(label, end, 8, 4)
    svg.line((0, 0, 0), p, "vector", True)
    svg.line((0, 0, 0), q, "component", True)
    svg.line(q, p, "helper")
    svg.line(s, p, "normal", True)
    svg.circle(p, 6.6, "point")
    svg.circle(q, 5.2, "surface")
    svg.circle(s, 5.8, "surface")
    svg.text("已知 P(X,Y,Z)", p, 8, -6, "teal")
    svg.text("Q=(X,Y,0)", q, 8, 14, "teal")
    svg.text("S 法线脚点", s, 8, 16, "orange")
    svg.text("h=(P-S)·n̂", add(s, mul(VISUAL_H_OVER_A * 0.55, n)), 10, -8, "orange")
    svg.text("λ=atan2(Y,X)", q, 20, -18, "teal")
    svg.text_xy("p=sqrt(X²+Y²)", 44, 520, "formula")
    svg.text_xy("φ 通过迭代或 Bowring 公式求得；h 是 P-S 在法线方向上的有符号长度", 44, 544, "formula")
    return svg.render()


def diagram_height() -> str:
    width, height = 900, 520
    cx, cy = 380, 290
    scale = 250
    phi = TEACHING_PHI
    b = WGS84_B_OVER_A
    p_s = nu_over_a(phi) * cos(phi)
    z_s = (1.0 - WGS84_E2) * nu_over_a(phi) * sin(phi)
    sx, sy = cx + p_s * scale, cy - z_s * scale
    n = (cos(phi), -sin(phi))
    h_px = 120
    px, py = sx + n[0] * h_px, sy + n[1] * h_px
    out = svg_2d_header(width, height, "Ellipsoidal height along the normal")
    out.append('<text class="title" x="30" y="36">精确高度示意：h 沿法线，不沿地心半径</text>')
    out.append('<text class="subtitle" x="30" y="58">剖面使用 WGS 84 真实扁率；h 箭头沿同一法线方向可视化放大</text>')
    out.append(f'<ellipse class="ellipse" cx="{cx}" cy="{cy}" rx="{scale}" ry="{scale * b:.1f}"/>')
    out.append(f'<line class="axis" x1="{cx - 310}" y1="{cy}" x2="{cx + 335}" y2="{cy}"/>')
    out.append(f'<line class="axis" x1="{cx}" y1="{cy + 245}" x2="{cx}" y2="{cy - 280}"/>')
    out.append(f'<line class="radius" x1="{cx}" y1="{cy}" x2="{px:.1f}" y2="{py:.1f}"/>')
    out.append(f'<line class="normal" x1="{sx:.1f}" y1="{sy:.1f}" x2="{px:.1f}" y2="{py:.1f}"/>')
    out.append(f'<circle class="surface" cx="{sx:.1f}" cy="{sy:.1f}" r="6"/>')
    out.append(f'<circle class="point" cx="{px:.1f}" cy="{py:.1f}" r="7"/>')
    out.append(f'<text class="small" x="{cx + 8}" y="{cy + 18}">O 地心</text>')
    out.append(f'<text class="orange" x="{sx + 10:.1f}" y="{sy + 16:.1f}">S：椭球面法线脚点</text>')
    out.append(f'<text class="label" x="{px + 10:.1f}" y="{py - 6:.1f}">P</text>')
    out.append(f'<text class="orange" x="{(sx + px) / 2 + 8:.1f}" y="{(sy + py) / 2 - 8:.1f}">h = |P-S|，方向为 n̂</text>')
    out.append(f'<text class="indigo" x="{(cx + px) / 2 - 40:.1f}" y="{(cy + py) / 2 - 10:.1f}">r = ||P||，不是 h</text>')
    out.append('<text class="formula" x="42" y="462">P = S + h·n̂；n̂=(cosφcosλ, cosφsinλ, sinφ)。只有球面或赤道/极点等特殊情况，法线才与地心方向重合。</text>')
    out.append('<text class="formula" x="42" y="488">海拔/正高 H 还需要大地水准面起伏 N：常用约定 h = H + N。</text>')
    out.append("</svg>")
    return "\n".join(out)


def diagram_values_readme() -> str:
    phi = TEACHING_PHI
    lam = TEACHING_LAMBDA
    s = geodetic_surface_unit(phi, lam)
    n = normal_unit(phi, lam)
    psi = atan((1.0 - WGS84_E2) * tan(phi))
    epsg_phi = (53 + 48 / 60 + 33.820 / 3600) * pi / 180
    epsg_lam = (2 + 7 / 60 + 46.380 / 3600) * pi / 180
    epsg_p = geodetic_to_ecef_m(epsg_phi, epsg_lam, 73.0)
    epsg_r = norm(epsg_p)
    return f"""# Generated diagram metadata

These SVG files are generated by `../generate_geodetic_ecef_diagrams.py`.

- WGS 84 semi-major axis a: {WGS84_A_M:.1f} m
- WGS 84 inverse flattening: {WGS84_INV_F:.12f}
- WGS 84 first eccentricity squared e^2: {WGS84_E2:.14f}
- Teaching latitude phi: {TEACHING_PHI_DEG:.9f} deg
- Teaching longitude lambda: {TEACHING_LAMBDA_DEG:.9f} deg
- Teaching surface point S/a: ({s[0]:.9f}, {s[1]:.9f}, {s[2]:.9f})
- Teaching normal n-hat: ({n[0]:.9f}, {n[1]:.9f}, {n[2]:.9f})
- Geocentric latitude psi at teaching latitude: {degrees(psi):.9f} deg
- Meridian radius M at teaching latitude: {meridian_radius_over_a(phi) * WGS84_A_M:.3f} m
- Prime vertical radius nu at teaching latitude: {nu_over_a(phi) * WGS84_A_M:.3f} m
- Visual height arrow: {VISUAL_H_OVER_A:.3f} a, used only to make the normal direction visible.
- EPSG example radial distance ||P|| for h=73.0 m: {epsg_r:.3f} m. This is not the ellipsoidal height.
"""


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    write(ASSET_DIR / "01-coordinate-systems.svg", diagram_coordinate_system())
    write(ASSET_DIR / "02-geodetic-vs-geocentric-latitude.svg", diagram_latitudes())
    write(ASSET_DIR / "03-curvature-radii.svg", diagram_curvature_radii())
    write(ASSET_DIR / "04-forward-geodetic-to-ecef.svg", diagram_forward())
    write(ASSET_DIR / "05-inverse-ecef-to-geodetic.svg", diagram_inverse())
    write(ASSET_DIR / "06-height-normal-vs-radial.svg", diagram_height())
    write(ASSET_DIR / "README.md", diagram_values_readme())


if __name__ == "__main__":
    main()
