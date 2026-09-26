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
import re
import unicodedata
from math import atan, atan2, cos, degrees, pi, radians, sin, sqrt, tan
from pathlib import Path
from typing import Iterable, Sequence


WGS84_A_M = 6_378_137.0
WGS84_INV_F = 298.257223563
WGS84_F = 1.0 / WGS84_INV_F
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_B_OVER_A = 1.0 - WGS84_F

REFERENCE_PHI_DEG = 53.809394444
REFERENCE_LAMBDA_DEG = 35.0
REFERENCE_PHI = REFERENCE_PHI_DEG * pi / 180.0
REFERENCE_LAMBDA = REFERENCE_LAMBDA_DEG * pi / 180.0
VISUAL_H_OVER_A = 0.26
# Arrowhead size relative to the original 8x6 marker, in stroke widths.
ARROW_SCALE = 0.62
# Blank border kept around the drawing when the viewBox is fitted to it.
FIT_MARGIN_PX = 14

# The default camera looks almost along the local East axis at the reference point
# (East projects to 0.39 of its length), so the ENU figures turn the eye to azimuth
# -15°, elevation 25°, where E, N and U all project to >= 0.72 and Z stays upright.
ENU_CAMERA_EYE = (cos(radians(25)) * cos(radians(-15)), cos(radians(25)) * sin(radians(-15)), sin(radians(25)))

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
        pad: int = 30,
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

    def raw(self, item: str) -> None:
        self.items.append(item)

    def render(self) -> str:
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title">
  <title id="title">{escape(self.title)}</title>
  <defs>
    <marker id="arrow" viewBox="0 0 8 6" markerWidth="{8 * ARROW_SCALE:g}" markerHeight="{6 * ARROW_SCALE:g}" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 Z" fill="#263548"/>
    </marker>
  </defs>
  <style>
    svg {{ background: linear-gradient(180deg, #fbfdff 0%, #f3f7fa 100%); }}
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
    .hidden {{ fill: none; stroke: #8292a4; stroke-width: 1.2; stroke-dasharray: 4 5; opacity: .45; }}
    .outline {{ fill: none; stroke: #207b84; stroke-width: 1.8; }}
    .arc {{ fill: none; stroke: #3d58a8; stroke-width: 1.8; }}
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
    for lam_deg in (0, 45, 90, 135, 180, 225, 270, 315, REFERENCE_LAMBDA_DEG):
        pts.extend(ellipsoid_lon(lam_deg * pi / 180.0))
    return pts


def draw_wire(svg: Svg) -> None:
    """WGS 84 graticule, near side only; the far halves of the equator and the two
    highlighted meridians are kept faint and dashed, and the silhouette closes the shape."""
    for phi_deg in (-60, -30, 30, 60):
        surface_polyline(svg, ellipsoid_lat(phi_deg * pi / 180.0), "wire")
    for lam_deg in (45, 90, 135, 180, 225, 270, 315):
        surface_polyline(svg, ellipsoid_lon(lam_deg * pi / 180.0), "wire")
    surface_polyline(svg, ellipsoid_lat(0.0), "equator", "hidden")
    surface_polyline(svg, ellipsoid_lon(0.0), "meridian", "hidden")
    surface_polyline(svg, ellipsoid_lon(REFERENCE_LAMBDA), "meridian", "hidden")
    svg.polyline(ellipsoid_outline(svg.camera), "outline")


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
    s = geodetic_surface_unit(REFERENCE_PHI, REFERENCE_LAMBDA)
    n = normal_unit(REFERENCE_PHI, REFERENCE_LAMBDA)
    p = add(s, mul(VISUAL_H_OVER_A, n))
    pts = wire_points() + [(0, 0, 0), (1.45, 0, 0), (0, 1.45, 0), (0, 0, 1.35), s, p]
    svg = Svg(760, 600, "WGS 84 ellipsoid, geodetic coordinates and ECEF axes", pts)
    draw_wire(svg)
    svg.line((0, 0, 0), (1.45, 0, 0), "axis", True)
    svg.line((0, 0, 0), (0, 1.45, 0), "axis", True)
    svg.line((0, 0, 0), (0, 0, 1.35), "axis", True)
    polygon_points(svg, tangent_plane_polygon(s, REFERENCE_PHI, REFERENCE_LAMBDA), "plane")
    svg.line((0, 0, 0), s, "vector", True)
    svg.line(s, p, "normal", True)
    svg.circle(s, 5.0, "surface")
    svg.circle(p, 5.6, "point")
    svg.text("X", (1.45, 0, 0), 6, 16)
    svg.text("Y", (0, 1.45, 0), -2, -12)
    svg.text("Z", (0, 0, 1.35), 10, 6)
    svg.text("O", (0, 0, 0), -20, 6)
    svg.text("P(φ, λ, h)", p, 12, -8, "teal")
    svg.text("S", s, 12, 20, "orange")
    svg.text("h·û", add(s, mul(VISUAL_H_OVER_A * 0.5, n)), -50, -14, "orange")
    svg.text("meridian λ", geodetic_surface_unit(0.45, REFERENCE_LAMBDA), 10, 4, "indigo")
    svg.text("equator", geodetic_surface_unit(0, -0.6), 4, 20, "orange")
    return svg.render()


def diagram_pz_section_geometry() -> str:
    phi = REFERENCE_PHI
    lam = REFERENCE_LAMBDA
    s = geodetic_surface_unit(phi, lam)
    radial = (cos(lam), sin(lam), 0.0)
    p_s = sqrt(s[0] * s[0] + s[1] * s[1])
    z_s = s[2]
    axis_at_z = (0.0, 0.0, z_s)
    z_min = -WGS84_B_OVER_A * 1.03
    z_max = WGS84_B_OVER_A * 1.03

    meridian_half_plane = [
        (0.0, 0.0, z_min),
        add(mul(1.08, radial), (0.0, 0.0, z_min)),
        add(mul(1.08, radial), (0.0, 0.0, z_max)),
        (0.0, 0.0, z_max),
    ]
    meridian_curve = [geodetic_surface_unit(-pi / 2.0 + pi * i / 120.0, lam) for i in range(121)]
    parallel_circle = [
        (p_s * cos(2.0 * pi * i / 144.0), p_s * sin(2.0 * pi * i / 144.0), z_s)
        for i in range(145)
    ]
    cylinder_circles = [
        [
            (p_s * cos(2.0 * pi * i / 96.0), p_s * sin(2.0 * pi * i / 96.0), z)
            for i in range(97)
        ]
        for z in (z_min * 0.82, 0.0, z_s, z_max * 0.82)
    ]
    cylinder_lines = [
        [(p_s * cos(theta), p_s * sin(theta), z_min * 0.82), (p_s * cos(theta), p_s * sin(theta), z_max * 0.82)]
        for theta in (0.0, pi / 3.0, 2.0 * pi / 3.0, pi, 4.0 * pi / 3.0, 5.0 * pi / 3.0)
    ]
    pts = (
        wire_points()
        + meridian_half_plane
        + meridian_curve
        + parallel_circle
        + [p for ring in cylinder_circles for p in ring]
        + [p for line in cylinder_lines for p in line]
        + [(0, 0, z_min), (0, 0, z_max), s, axis_at_z, mul(1.22, radial)]
    )

    svg = Svg(900, 600, "p-z section geometry for WGS 84 ellipsoid", pts)
    polygon_points(svg, meridian_half_plane, "plane")
    draw_wire(svg)
    for ring in cylinder_circles:
        svg.polyline(ring, "wire")
    for line in cylinder_lines:
        svg.polyline(line, "helper")
    surface_polyline(svg, meridian_curve, "meridian", "hidden")
    surface_polyline(svg, parallel_circle, "equator", "hidden")
    svg.line((0, 0, z_min), (0, 0, z_max), "axis", True)
    svg.line(axis_at_z, s, "component", True)
    svg.circle(s, 6.2, "surface")
    svg.circle(axis_at_z, 4.8, "point")
    svg.text("Z axis", (0, 0, z_max), 10, -4)
    svg.text("S", s, 10, -8, "orange")
    svg.text("p_S", add(axis_at_z, mul(0.52, sub(s, axis_at_z))), 8, -8, "teal")
    svg.text("fixed λ: meridian half-plane", geodetic_surface_unit(0.08, lam), 12, -10, "indigo")
    svg.text("fixed z=z_S: parallel / horizontal section", (p_s * cos(lam - 1.2), p_s * sin(lam - 1.2), z_s), -120, 24, "orange")
    svg.text("fixed p=p_S: cylinder", (p_s * cos(lam - 1.45), p_s * sin(lam - 1.45), 0.0), -40, 18, "teal")
    return svg.render()


def svg_2d_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">',
        f'<title id="title">{escape(title)}</title>',
        f'<defs><marker id="arrow" viewBox="0 0 8 6" markerWidth="{8 * ARROW_SCALE:g}" markerHeight="{6 * ARROW_SCALE:g}" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#263548"/></marker></defs>',
        """<style>
svg { background: linear-gradient(180deg, #fbfdff 0%, #f3f7fa 100%); }
.axis { stroke: #263548; stroke-width: 2.2; marker-end: url(#arrow); }
.ellipse { fill: #e7f7f8; stroke: #207b84; stroke-width: 2.2; }
.helper { stroke: #8292a4; stroke-width: 1.5; stroke-dasharray: 6 5; fill: none; }
.normal { stroke: #b65b13; stroke-width: 2.7; marker-end: url(#arrow); }
.radius { stroke: #3d58a8; stroke-width: 2.4; marker-end: url(#arrow); }
.tangent { stroke: #3d58a8; stroke-width: 2.0; }
.arc { fill: none; stroke: #3d58a8; stroke-width: 1.8; }
.point { fill: #007c89; stroke: white; stroke-width: 2.0; }
.surface { fill: #b65b13; stroke: white; stroke-width: 2.0; }
.label { fill: #16212f; font: 650 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.small { fill: #4b5d70; font: 500 12.5px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.orange { fill: #9a4c10; font: 650 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.teal { fill: #006f79; font: 650 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.indigo { fill: #314b99; font: 650 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
</style>""",
    ]


def diagram_latitudes() -> str:
    width, height = 900, 560
    cx, cy = 410, 305
    scale = 255
    b = WGS84_B_OVER_A
    phi = REFERENCE_PHI
    p_s = nu_over_a(phi) * cos(phi)
    z_s = (1.0 - WGS84_E2) * nu_over_a(phi) * sin(phi)
    psi = atan2(z_s, p_s)
    sx, sy = cx + p_s * scale, cy - z_s * scale
    nx, ny = cos(phi), -sin(phi)
    rx, ry = p_s, -z_s
    tangent = (-sin(phi), -cos(phi))
    out = svg_2d_header(width, height, "Geodetic latitude and geocentric latitude")
    out.append(f'<ellipse class="ellipse" cx="{cx}" cy="{cy}" rx="{scale}" ry="{scale * b:.1f}"/>')
    out.append(f'<line class="axis" x1="{cx - 310}" y1="{cy}" x2="{cx + 335}" y2="{cy}"/>')
    out.append(f'<line class="axis" x1="{cx}" y1="{cy + scale * b + 20:.1f}" x2="{cx}" y2="{cy - 300}"/>')
    out.append(f'<line class="helper" x1="{sx:.1f}" y1="{sy:.1f}" x2="{sx:.1f}" y2="{cy:.1f}"/>')
    # The normal through S meets the p axis at F; φ is the angle there, ψ the angle at O.
    fx = sx - (cy - sy) * cos(phi) / sin(phi)
    out.append(f'<line class="helper" x1="{fx:.1f}" y1="{cy:.1f}" x2="{sx:.1f}" y2="{sy:.1f}"/>')
    for vx, radius, angle in ((cx, 62.0, psi), (fx, 46.0, phi)):
        out.append(
            f'<path class="arc" d="M {vx + radius:.1f},{cy:.1f} A {radius:.1f},{radius:.1f} 0 0 0 '
            f'{vx + radius * cos(angle):.1f},{cy - radius * sin(angle):.1f}"/>'
        )
    out.append(f'<line class="helper" x1="{cx:.1f}" y1="{sy:.1f}" x2="{sx:.1f}" y2="{sy:.1f}"/>')
    out.append(f'<line class="radius" x1="{cx}" y1="{cy}" x2="{sx:.1f}" y2="{sy:.1f}"/>')
    out.append(f'<line class="normal" x1="{sx:.1f}" y1="{sy:.1f}" x2="{sx + nx * 150:.1f}" y2="{sy + ny * 150:.1f}"/>')
    out.append(f'<line class="tangent" x1="{sx - tangent[0] * 120:.1f}" y1="{sy - tangent[1] * 120:.1f}" x2="{sx + tangent[0] * 120:.1f}" y2="{sy + tangent[1] * 120:.1f}"/>')
    out.append(f'<circle class="surface" cx="{sx:.1f}" cy="{sy:.1f}" r="6"/>')
    out.append(f'<text class="label" x="{cx + 342}" y="{cy + 6}">p</text>')
    out.append(f'<text class="label" x="{cx + 10}" y="{cy - 303}">z</text>')
    out.append(f'<text class="small" x="{cx + 8}" y="{cy + 18}">O</text>')
    out.append(f'<text class="orange" x="{sx + nx * 100 + 8:.1f}" y="{sy + ny * 100:.1f}">normal û</text>')
    out.append(f'<text class="indigo" x="{cx + rx * scale * 0.5 + 10:.1f}" y="{cy + ry * scale * 0.5 + 20:.1f}">geocentric</text>')
    out.append(f'<text class="label" x="{sx + 10:.1f}" y="{sy - 8:.1f}">S(p_S,z_S)</text>')
    out.append(f'<text class="small" x="{sx + 10:.1f}" y="{cy - 8:.1f}">p_S</text>')
    out.append(f'<text class="small" x="{sx + 8:.1f}" y="{(sy + cy) / 2:.1f}">z_S</text>')
    # With the true flattening F = e²ν cosφ ≈ 25 km from O, so the two arcs are almost concentric.
    out.append(f'<text class="indigo" x="{cx + 74 * cos(psi / 2):.1f}" y="{cy - 74 * sin(psi / 2) + 5:.1f}">ψ</text>')
    out.append(f'<text class="orange" x="{fx + 30 * cos(phi / 2) - 4:.1f}" y="{cy - 30 * sin(phi / 2) + 5:.1f}">φ</text>')
    out.append("</svg>")
    return "\n".join(out)


def diagram_tangent_slope_dp() -> str:
    width, height = 940, 560
    cx, cy = 230, 365
    scale = 255
    b = WGS84_B_OVER_A
    phi = REFERENCE_PHI
    p_s = nu_over_a(phi) * cos(phi)
    z_s = (1.0 - WGS84_E2) * nu_over_a(phi) * sin(phi)
    delta_p = 0.12
    p_2 = p_s + delta_p
    z_2 = b * sqrt(1.0 - p_2 * p_2)
    delta_z = z_2 - z_s
    slope = -(b * b * p_s) / z_s

    def xy(p: float, z: float) -> tuple[float, float]:
        return cx + p * scale, cy - z * scale

    sx, sy = xy(p_s, z_s)
    s2x, s2y = xy(p_2, z_2)
    hx, hy = xy(p_2, z_s)
    tx0 = p_s - 0.20
    tx1 = p_s + 0.13
    ty0 = z_s + slope * (tx0 - p_s)
    ty1 = z_s + slope * (tx1 - p_s)
    tangent_x0, tangent_y0 = xy(tx0, ty0)
    tangent_x1, tangent_y1 = xy(tx1, ty1)
    curve_points = []
    for i in range(130):
        theta = (pi / 2.0) * i / 129.0
        p = cos(theta)
        z = b * sin(theta)
        x, y = xy(p, z)
        curve_points.append(f"{x:.1f},{y:.1f}")

    out = svg_2d_header(width, height, "Meaning of p in dz/dp on the WGS 84 meridian section")
    out.append("""<style>
.secant { stroke: #007c89; stroke-width: 2.0; stroke-dasharray: 8 5; }
.delta { stroke: #007c89; stroke-width: 2.4; marker-end: url(#arrow); }
.delta-z { stroke: #b65b13; stroke-width: 2.4; marker-end: url(#arrow); }
</style>""")
    out.append(f'<polyline class="ellipse" fill="none" points="{" ".join(curve_points)}"/>')
    out.append(f'<line class="axis" x1="{cx}" y1="{cy + 30}" x2="{cx}" y2="{cy - 290}"/>')
    out.append(f'<line class="axis" x1="{cx - 20}" y1="{cy}" x2="{cx + 565}" y2="{cy}"/>')
    out.append(f'<line class="helper" x1="{sx:.1f}" y1="{sy:.1f}" x2="{sx:.1f}" y2="{cy:.1f}"/>')
    out.append(f'<line class="helper" x1="{s2x:.1f}" y1="{s2y:.1f}" x2="{s2x:.1f}" y2="{cy:.1f}"/>')
    out.append(f'<line class="helper" x1="{cx:.1f}" y1="{sy:.1f}" x2="{sx:.1f}" y2="{sy:.1f}"/>')
    out.append(f'<line class="helper" x1="{cx:.1f}" y1="{s2y:.1f}" x2="{s2x:.1f}" y2="{s2y:.1f}"/>')
    out.append(f'<line class="tangent" x1="{tangent_x0:.1f}" y1="{tangent_y0:.1f}" x2="{tangent_x1:.1f}" y2="{tangent_y1:.1f}"/>')
    out.append(f'<line class="secant" x1="{sx:.1f}" y1="{sy:.1f}" x2="{s2x:.1f}" y2="{s2y:.1f}"/>')
    out.append(f'<line class="delta" x1="{sx:.1f}" y1="{sy + 22:.1f}" x2="{hx:.1f}" y2="{hy + 22:.1f}"/>')
    out.append(f'<line class="delta-z" x1="{hx + 14:.1f}" y1="{hy:.1f}" x2="{s2x + 14:.1f}" y2="{s2y:.1f}"/>')
    out.append(f'<circle class="surface" cx="{sx:.1f}" cy="{sy:.1f}" r="6"/>')
    out.append(f'<circle class="point" cx="{s2x:.1f}" cy="{s2y:.1f}" r="5"/>')
    out.append(f'<text class="label" x="{cx + 575}" y="{cy + 5}">p</text>')
    out.append(f'<text class="label" x="{cx + 10}" y="{cy - 292}">z</text>')
    out.append(f'<text class="small" x="{cx + 6}" y="{cy + 18}">O (Z axis)</text>')
    out.append(f'<text class="orange" x="{sx - 100:.1f}" y="{sy + 20:.1f}">S(p_S,z_S)</text>')
    out.append(f'<text class="teal" x="{s2x + 12:.1f}" y="{s2y + 26:.1f}">S′(p_S+Δp, z_S+Δz)</text>')
    out.append(f'<text class="small" x="{sx - 26:.1f}" y="{cy + 22:.1f}">p_S</text>')
    out.append(f'<text class="small" x="{s2x - 4:.1f}" y="{cy + 22:.1f}">p_S+Δp</text>')
    out.append(f'<text class="teal" x="{(sx + hx) / 2 - 6:.1f}" y="{hy + 42:.1f}">Δp</text>')
    out.append(f'<text class="orange" x="{hx + 25:.1f}" y="{(hy + s2y) / 2 + 5:.1f}">Δz</text>')
    out.append(f'<text class="indigo" x="{tangent_x0 - 70:.1f}" y="{tangent_y0 - 12:.1f}">tangent slope = (dz/dp)|S</text>')
    return "\n".join(out + ["</svg>"])


def meridian_radius_over_a(phi: float) -> float:
    return (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * sin(phi) ** 2) ** 1.5


def circle3d(center: Vec3, radius: float, u: Vec3, v: Vec3, samples: int = 145) -> list[Vec3]:
    return [
        add(center, add(mul(radius * cos(2.0 * pi * i / (samples - 1)), u), mul(radius * sin(2.0 * pi * i / (samples - 1)), v)))
        for i in range(samples)
    ]


def diagram_curvature_radii() -> str:
    phi = REFERENCE_PHI
    lam = REFERENCE_LAMBDA
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
    svg = Svg(900, 600, "Meridian and prime vertical radii of curvature", pts)
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
    svg.text("S", s, -22, 20, "teal")
    svg.text("north tangent (north-south)", add(s, mul(0.42, north)), 10, -4, "indigo")
    svg.text("east tangent (east-west)", add(s, mul(0.42, east)), 10, 12, "teal")
    svg.text("normal û", add(s, mul(0.46, n_out)), 10, -6, "orange")
    svg.text("M meridian radius", add(s, mul(0.48, n_in)), -140, 8, "indigo")
    svg.text("ν prime-vertical radius", add(s, mul(0.75, n_in)), 10, -8, "orange")
    return svg.render()


def diagram_prime_vertical_projection() -> str:
    width, height = 940, 620
    phi = REFERENCE_PHI
    lam = REFERENCE_LAMBDA
    cos_phi = cos(phi)
    sin_phi = sin(phi)

    out = svg_2d_header(width, height, "Projection from parallel-circle curvature to prime-vertical normal curvature")
    out.append("""<style>
.panel-title { fill: #16212f; font: 700 16px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.parallel-circle { fill: #eef8f9; stroke: #007c89; stroke-width: 2.2; }
.kcircle { stroke: #007c89; stroke-width: 3.0; marker-end: url(#arrow); fill: none; }
.projection { stroke: #b65b13; stroke-width: 3.0; marker-end: url(#arrow); fill: none; }
.normal-line { stroke: #b65b13; stroke-width: 2.4; marker-end: url(#arrow); fill: none; }
.panel-box { fill: none; stroke: #d8e1e8; stroke-width: 1.2; }
</style>""")
    out.append('<rect class="panel-box" x="30" y="86" width="410" height="410" rx="8"/>')
    out.append('<rect class="panel-box" x="500" y="86" width="410" height="410" rx="8"/>')
    out.append('<text class="panel-title" x="50" y="116">A. Space curvature of the parallel</text>')
    out.append('<text class="panel-title" x="520" y="116">B. Projecting it onto the ellipsoid normal</text>')

    # Left panel: top-down view of the parallel circle at z=z_S.
    cx, cy, radius = 235.0, 300.0, 150.0
    sx = cx + radius * cos(lam)
    sy = cy - radius * sin(lam)
    tangent_len = 96.0
    tx = -sin(lam)
    ty = -cos(lam)
    k_len = 92.0
    kx = -cos(lam)
    ky = sin(lam)
    out.append(f'<circle class="parallel-circle" cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}"/>')
    out.append(f'<line class="radius" x1="{cx:.1f}" y1="{cy:.1f}" x2="{sx:.1f}" y2="{sy:.1f}"/>')
    out.append(
        f'<line class="tangent" x1="{sx - tx * tangent_len / 2:.1f}" y1="{sy - ty * tangent_len / 2:.1f}" x2="{sx + tx * tangent_len / 2:.1f}" y2="{sy + ty * tangent_len / 2:.1f}"/>'
    )
    out.append(f'<line class="kcircle" x1="{sx:.1f}" y1="{sy:.1f}" x2="{sx + kx * k_len:.1f}" y2="{sy + ky * k_len:.1f}"/>')
    out.append(f'<circle class="surface" cx="{sx:.1f}" cy="{sy:.1f}" r="6.2"/>')
    out.append(f'<circle class="point" cx="{cx:.1f}" cy="{cy:.1f}" r="4.8"/>')
    out.append(f'<text class="small" x="{cx - 20:.1f}" y="{cy + 24:.1f}">Z axis</text>')
    out.append(f'<text class="orange" x="{sx + 10:.1f}" y="{sy - 6:.1f}">S</text>')
    out.append(f'<text class="indigo" x="{(cx + sx) / 2 + 4:.1f}" y="{(cy + sy) / 2 + 26:.1f}">ρ = ν cosφ</text>')
    out.append(f'<text class="teal" x="{sx + kx * k_len - 112:.1f}" y="{sy + ky * k_len - 12:.1f}">κ_circle = 1/ρ</text>')
    out.append(f'<text class="small" x="{sx - tx * tangent_len / 2 + 6:.1f}" y="{sy - ty * tangent_len / 2 + 8:.1f}">east tangent</text>')

    # Right panel: local vector projection in the plane spanned by the inward radial direction and Z.
    ox, oy = 735.0, 310.0
    vec_len = 172.0
    q_end = (ox - vec_len, oy)
    normal_end = (ox - vec_len * cos_phi, oy + vec_len * sin_phi)
    proj_len = vec_len * cos_phi
    proj_end = (ox - proj_len * cos_phi, oy + proj_len * sin_phi)
    out.append(f'<line class="helper" x1="{ox - 205:.1f}" y1="{oy:.1f}" x2="{ox + 22:.1f}" y2="{oy:.1f}"/>')
    out.append(f'<line class="kcircle" x1="{ox:.1f}" y1="{oy:.1f}" x2="{q_end[0]:.1f}" y2="{q_end[1]:.1f}"/>')
    out.append(f'<line class="normal-line" x1="{ox:.1f}" y1="{oy:.1f}" x2="{normal_end[0]:.1f}" y2="{normal_end[1]:.1f}"/>')
    out.append(f'<line class="projection" x1="{ox:.1f}" y1="{oy:.1f}" x2="{proj_end[0]:.1f}" y2="{proj_end[1]:.1f}"/>')
    out.append(f'<line class="helper" x1="{q_end[0]:.1f}" y1="{q_end[1]:.1f}" x2="{proj_end[0]:.1f}" y2="{proj_end[1]:.1f}"/>')
    out.append(f'<circle class="surface" cx="{ox:.1f}" cy="{oy:.1f}" r="6.2"/>')
    out.append(f'<circle class="point" cx="{proj_end[0]:.1f}" cy="{proj_end[1]:.1f}" r="4.8"/>')
    arc_r = 44.0
    start = (ox - arc_r, oy)
    end = (ox - arc_r * cos_phi, oy + arc_r * sin_phi)
    out.append(f'<path class="arc" d="M {start[0]:.1f},{start[1]:.1f} A {arc_r:.1f},{arc_r:.1f} 0 0 0 {end[0]:.1f},{end[1]:.1f}"/>')
    out.append(f'<text class="indigo" x="{ox - 62:.1f}" y="{oy + 36:.1f}">φ</text>')
    out.append(f'<text class="teal" x="{q_end[0] - 4:.1f}" y="{q_end[1] - 14:.1f}">horizontal inward q̂</text>')
    out.append(f'<text class="orange" x="{normal_end[0] - 32:.1f}" y="{normal_end[1] + 22:.1f}">inward normal −û</text>')
    out.append(f'<text class="orange" x="{proj_end[0] + 14:.1f}" y="{proj_end[1] + 4:.1f}">projection = |κ_circle| cosφ</text>')
    out.append(f'<text class="small" x="{ox + 10:.1f}" y="{oy + 22:.1f}">S</text>')

    out.append("</svg>")
    return "\n".join(out)


def diagram_forward() -> str:
    s = geodetic_surface_unit(REFERENCE_PHI, REFERENCE_LAMBDA)
    n = normal_unit(REFERENCE_PHI, REFERENCE_LAMBDA)
    p = add(s, mul(VISUAL_H_OVER_A, n))
    q = (p[0], p[1], 0.0)
    x_comp = (p[0], 0.0, 0.0)
    y_comp = (0.0, p[1], 0.0)
    pts = wire_points() + [(0, 0, 0), (1.45, 0, 0), (0, 1.45, 0), (0, 0, 1.35), s, p, q, x_comp, y_comp]
    svg = Svg(900, 600, "Forward conversion geodetic to ECEF", pts)
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
    svg.text("h·û", add(s, mul(VISUAL_H_OVER_A * 0.55, n)), -42, 2, "orange")
    svg.text("p=sqrt(X²+Y²)", q, 18, -8, "teal")
    return svg.render()


def diagram_inverse() -> str:
    s = geodetic_surface_unit(REFERENCE_PHI, REFERENCE_LAMBDA)
    n = normal_unit(REFERENCE_PHI, REFERENCE_LAMBDA)
    p = add(s, mul(VISUAL_H_OVER_A, n))
    q = (p[0], p[1], 0.0)
    pts = wire_points() + [(0, 0, 0), (1.45, 0, 0), (0, 1.45, 0), (0, 0, 1.35), s, p, q]
    svg = Svg(900, 600, "Inverse conversion ECEF to geodetic", pts)
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
    svg.text("P(X,Y,Z)", p, 8, -6, "teal")
    svg.text("Q=(X,Y,0)", q, 8, 14, "teal")
    svg.text("S", s, 8, 16, "orange")
    svg.text("h=(P−S)·û", add(s, mul(VISUAL_H_OVER_A * 0.55, n)), -96, 2, "orange")
    svg.text("λ=atan2(Y,X)", q, 20, -18, "teal")
    return svg.render()


def diagram_iterative_inverse_atan2() -> str:
    width, height = 940, 620
    cx, cy = 250, 400
    scale = 270
    phi = REFERENCE_PHI
    b = WGS84_B_OVER_A
    nu = nu_over_a(phi)
    p_s = nu * cos(phi)
    z_s = (1.0 - WGS84_E2) * nu * sin(phi)
    h = VISUAL_H_OVER_A
    p = p_s + h * cos(phi)
    z = z_s + h * sin(phi)
    z_corrected = z + WGS84_E2 * nu * sin(phi)
    phi_initial = atan2(z, p * (1.0 - WGS84_E2))

    def xy(p_value: float, z_value: float) -> Vec2:
        return (cx + p_value * scale, cy - z_value * scale)

    sx, sy = xy(p_s, z_s)
    px, py = xy(p, z)
    pcx, pcy = xy(p, z_corrected)
    init_x, init_y = xy(p * (1.0 - WGS84_E2), z)
    angle_r = 58
    angle_end = (cx + angle_r * cos(phi), cy - angle_r * sin(phi))
    init_angle_end = (cx + 42 * cos(phi_initial), cy - 42 * sin(phi_initial))

    out = svg_2d_header(width, height, "Iterative inverse latitude and atan2")
    # Only the upper-right quadrant carries the construction.
    out.append(f'<path class="ellipse" d="M {cx + scale},{cy} A {scale},{scale * b:.1f} 0 0 0 {cx},{cy - scale * b:.1f} L {cx},{cy} Z"/>')
    out.append(f'<line class="axis" x1="{cx - 20}" y1="{cy}" x2="{cx + 585}" y2="{cy}"/>')
    out.append(f'<line class="axis" x1="{cx}" y1="{cy + 20}" x2="{cx}" y2="{cy - 330}"/>')
    out.append(f'<line class="helper" x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{cy:.1f}"/>')
    out.append(f'<line class="helper" x1="{cx:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{py:.1f}"/>')
    out.append(f'<line class="helper" x1="{cx:.1f}" y1="{pcy:.1f}" x2="{pcx:.1f}" y2="{pcy:.1f}"/>')
    out.append(f'<line class="helper" x1="{pcx:.1f}" y1="{pcy:.1f}" x2="{pcx:.1f}" y2="{cy:.1f}"/>')
    out.append(f'<line class="normal" x1="{sx:.1f}" y1="{sy:.1f}" x2="{px:.1f}" y2="{py:.1f}"/>')
    out.append(f'<line class="radius" x1="{cx}" y1="{cy}" x2="{pcx:.1f}" y2="{pcy:.1f}"/>')
    out.append(f'<line class="helper" x1="{cx}" y1="{cy}" x2="{px:.1f}" y2="{py:.1f}"/>')
    out.append(f'<line class="normal" x1="{px:.1f}" y1="{py:.1f}" x2="{pcx:.1f}" y2="{pcy:.1f}"/>')
    out.append(f'<line class="helper" x1="{cx}" y1="{cy}" x2="{init_x:.1f}" y2="{init_y:.1f}"/>')
    out.append(f'<circle class="surface" cx="{sx:.1f}" cy="{sy:.1f}" r="6"/>')
    out.append(f'<circle class="point" cx="{px:.1f}" cy="{py:.1f}" r="7"/>')
    out.append(f'<circle class="point" cx="{pcx:.1f}" cy="{pcy:.1f}" r="5"/>')
    out.append(f'<circle class="surface" cx="{init_x:.1f}" cy="{init_y:.1f}" r="4.5"/>')
    out.append(f'<path class="helper" d="M {cx + angle_r:.1f},{cy:.1f} A {angle_r:.1f},{angle_r:.1f} 0 0 0 {angle_end[0]:.1f},{angle_end[1]:.1f}"/>')
    out.append(f'<path class="helper" d="M {cx + 42:.1f},{cy:.1f} A 42,42 0 0 0 {init_angle_end[0]:.1f},{init_angle_end[1]:.1f}"/>')
    out.append(f'<text class="label" x="{cx + 592}" y="{cy + 6}">p</text>')
    out.append(f'<text class="label" x="{cx + 10}" y="{cy - 333}">z</text>')
    out.append(f'<text class="small" x="{cx + 8}" y="{cy + 18}">O</text>')
    out.append(f'<text class="orange" x="{sx - 20:.1f}" y="{sy + 30:.1f}">S</text>')
    out.append(f'<text class="label" x="{px + 18:.1f}" y="{py + 38:.1f}">P(p,Z)</text>')
    out.append(f'<text class="indigo" x="{pcx + 22:.1f}" y="{pcy - 34:.1f}">Cₙ=(p, Z+e²νₙ sinφₙ)</text>')
    out.append(f'<text class="orange" x="{px + 42:.1f}" y="{(py + pcy) / 2 - 10:.1f}">vertical correction e²νₙ sinφₙ</text>')
    out.append(f'<text class="small" x="{cx + (p * scale) * 0.45:.1f}" y="{cy + 22:.1f}">horizontal side p</text>')
    out.append(f'<text class="indigo" x="{cx + 110}" y="{cy - 40}">φₙ₊₁ = atan2(z_c,n, p)</text>')
    out.append(f'<text class="small" x="{init_x + 34:.1f}" y="{init_y + 58:.1f}">initial angle: atan2(Z, p(1-e²))</text>')
    out.append("</svg>")
    return "\n".join(out)


def diagram_height() -> str:
    width, height = 900, 520
    cx, cy = 380, 290
    scale = 250
    phi = REFERENCE_PHI
    b = WGS84_B_OVER_A
    p_s = nu_over_a(phi) * cos(phi)
    z_s = (1.0 - WGS84_E2) * nu_over_a(phi) * sin(phi)
    sx, sy = cx + p_s * scale, cy - z_s * scale
    n = (cos(phi), -sin(phi))
    h_px = 120
    px, py = sx + n[0] * h_px, sy + n[1] * h_px
    out = svg_2d_header(width, height, "Ellipsoidal height along the normal")
    out.append(f'<ellipse class="ellipse" cx="{cx}" cy="{cy}" rx="{scale}" ry="{scale * b:.1f}"/>')
    out.append(f'<line class="axis" x1="{cx - 310}" y1="{cy}" x2="{cx + 335}" y2="{cy}"/>')
    out.append(f'<line class="axis" x1="{cx}" y1="{cy + 245}" x2="{cx}" y2="{cy - 280}"/>')
    out.append(f'<line class="radius" x1="{cx}" y1="{cy}" x2="{px:.1f}" y2="{py:.1f}"/>')
    out.append(f'<line class="normal" x1="{sx:.1f}" y1="{sy:.1f}" x2="{px:.1f}" y2="{py:.1f}"/>')
    out.append(f'<circle class="surface" cx="{sx:.1f}" cy="{sy:.1f}" r="6"/>')
    out.append(f'<circle class="point" cx="{px:.1f}" cy="{py:.1f}" r="7"/>')
    out.append(f'<text class="small" x="{cx + 8}" y="{cy + 18}">O</text>')
    out.append(f'<text class="orange" x="{sx + 14:.1f}" y="{sy + 4:.1f}">S</text>')
    out.append(f'<text class="label" x="{px + 10:.1f}" y="{py - 6:.1f}">P</text>')
    out.append(f'<text class="orange" x="{(sx + px) / 2 + 16:.1f}" y="{(sy + py) / 2 + 18:.1f}">h = |P−S| along û</text>')
    out.append(f'<text class="indigo" x="{cx + 0.4 * (px - cx) + 16:.1f}" y="{cy + 0.4 * (py - cy) + 20:.1f}">r = ||P||, not h</text>')
    out.append("</svg>")
    return "\n".join(out)


def diagram_local_tangent_enu() -> str:
    phi0 = REFERENCE_PHI
    lam0 = REFERENCE_LAMBDA
    origin = geodetic_surface_unit(phi0, lam0)
    east = east_unit(lam0)
    north = north_unit(phi0, lam0)
    up = normal_unit(phi0, lam0)

    # Visual offsets are intentionally larger than real local engineering
    # offsets so the ENU decomposition remains readable in a page-sized SVG.
    east_len = 0.45
    north_len = 0.30
    up_len = 0.22
    p_e = add(origin, mul(east_len, east))
    p_en = add(p_e, mul(north_len, north))
    target = add(p_en, mul(up_len, up))
    delta_mid = add(origin, mul(0.48, sub(target, origin)))
    plane = tangent_plane_polygon(origin, phi0, lam0, size=0.42)

    pts = (
        wire_points()
        + plane
        + [
            (0, 0, 0),
            (1.45, 0, 0),
            (0, 1.45, 0),
            (0, 0, 1.35),
            origin,
            p_e,
            p_en,
            target,
        ]
    )
    svg = Svg(900, 600, "WGS 84 geodetic to local tangent ENU", pts, camera=Camera(ENU_CAMERA_EYE))
    draw_wire(svg)
    for end, label in [((1.45, 0, 0), "X"), ((0, 1.45, 0), "Y"), ((0, 0, 1.35), "Z")]:
        svg.line((0, 0, 0), end, "axis", True)
        svg.text(label, end, 8, 4)
    polygon_points(svg, plane, "plane")
    svg.line((0, 0, 0), origin, "helper")
    svg.line((0, 0, 0), target, "helper")
    svg.line(origin, target, "vector", True)
    svg.line(origin, p_e, "east", True)
    svg.line(p_e, p_en, "north", True)
    svg.line(p_en, target, "normal", True)
    svg.circle(origin, 6.5, "surface")
    svg.circle(p_e, 4.8, "surface")
    svg.circle(p_en, 4.8, "surface")
    svg.circle(target, 7.0, "point")
    svg.text("O_L(φ₀,λ₀,h₀)", origin, 10, 16, "orange")
    svg.text("P(φ,λ,h)", target, 10, -8, "teal")
    svg.text("Δr", delta_mid, -30, -4, "indigo")
    svg.text("E ê₀", add(origin, mul(0.5, sub(p_e, origin))), -14, -10, "teal")
    svg.text("N n̂₀", add(p_e, mul(0.5, sub(p_en, p_e))), 10, 6, "indigo")
    svg.text("U û₀", add(p_en, mul(0.5, sub(target, p_en))), 10, 4, "orange")
    return svg.render()


def diagram_enu_basis_derivation() -> str:
    phi0 = REFERENCE_PHI
    lam0 = REFERENCE_LAMBDA
    origin = geodetic_surface_unit(phi0, lam0)
    east = east_unit(lam0)
    north = north_unit(phi0, lam0)
    up = normal_unit(phi0, lam0)
    e_end = add(origin, mul(0.42, east))
    n_end = add(origin, mul(0.42, north))
    u_end = add(origin, mul(0.42, up))
    plane = tangent_plane_polygon(origin, phi0, lam0, size=0.38)
    pts = (
        wire_points()
        + plane
        + [
            (0, 0, 0),
            (1.45, 0, 0),
            (0, 1.45, 0),
            (0, 0, 1.35),
            origin,
            e_end,
            n_end,
            u_end,
        ]
    )
    svg = Svg(900, 600, "ENU basis vectors derived at a WGS 84 local origin", pts, camera=Camera(ENU_CAMERA_EYE))
    draw_wire(svg)
    for end, label in [((1.45, 0, 0), "X"), ((0, 1.45, 0), "Y"), ((0, 0, 1.35), "Z")]:
        svg.line((0, 0, 0), end, "axis", True)
        svg.text(label, end, 8, 4)
    polygon_points(svg, plane, "plane")
    svg.line((0, 0, 0), origin, "helper")
    svg.line(origin, e_end, "east", True)
    svg.line(origin, n_end, "north", True)
    svg.line(origin, u_end, "normal", True)
    svg.circle(origin, 6.6, "surface")
    svg.text("O_L(φ₀,λ₀,h₀)", origin, 10, 16, "orange")
    svg.text("ê₀ East", e_end, 8, 16, "teal")
    svg.text("n̂₀ North", n_end, -40, -12, "indigo")
    svg.text("û₀ Up / normal", u_end, 8, -8, "orange")
    svg.text("local tangent plane", plane[3], 8, 22, "small")
    return svg.render()


# The three frame figures (00a/00b/00c) put one point P, with height, in each frame.
OVERVIEW_P_DPHI_DEG = 14.0
OVERVIEW_P_DLAMBDA_DEG = 26.0
OVERVIEW_P_H_OVER_A = 0.30
# One Earth for all three figures: an oblate ellipsoid (short polar axis b along Z)
# with the flattening exaggerated from the true b/a ≈ 0.9966 so the shape is visible.
OVERVIEW_B_OVER_A = 0.90
OVERVIEW_E2 = 1.0 - OVERVIEW_B_OVER_A**2
# 00b and 00c share this low-elevation camera, so the same Earth looks identical in both.
OVERVIEW_GLOBE_EYE = (cos(radians(12)) * cos(radians(-15)), cos(radians(12)) * sin(radians(-15)), sin(radians(12)))


def overview_point() -> tuple[float, float, float]:
    return (
        REFERENCE_PHI + OVERVIEW_P_DPHI_DEG * pi / 180.0,
        REFERENCE_LAMBDA + OVERVIEW_P_DLAMBDA_DEG * pi / 180.0,
        OVERVIEW_P_H_OVER_A,
    )


def geodetic_to_ecef_unit(phi: float, lam: float, h: float, e2: float = WGS84_E2) -> Vec3:
    nu = 1.0 / sqrt(1.0 - e2 * sin(phi) ** 2)
    return (
        (nu + h) * cos(phi) * cos(lam),
        (nu + h) * cos(phi) * sin(lam),
        ((1.0 - e2) * nu + h) * sin(phi),
    )


def overview_surface(phi: float, lam: float) -> Vec3:
    return geodetic_to_ecef_unit(phi, lam, 0.0, OVERVIEW_E2)


def arc3d(center: Vec3, radius: float, u: Vec3, v: Vec3, angle: float, samples: int = 40) -> list[Vec3]:
    """Arc from direction u towards v (orthonormal) through `angle` radians."""
    return [
        add(center, add(mul(radius * cos(angle * i / (samples - 1)), u), mul(radius * sin(angle * i / (samples - 1)), v)))
        for i in range(samples)
    ]


def facing_viewer(p: Vec3, camera: Camera, e2: float) -> bool:
    """True where the ellipsoid surface at p faces the camera (outward normal ∝ (x, y, z/b²))."""
    return dot((p[0], p[1], p[2] / (1.0 - e2)), camera.view) >= 0.0


def surface_polyline(
    svg: Svg, pts: Sequence[Vec3], cls: str, hidden_cls: str | None = None, e2: float = WGS84_E2
) -> None:
    """Draw a curve on the ellipsoid: the far-side runs are dropped, or drawn as hidden_cls."""
    runs: list[tuple[bool, list[Vec3]]] = []
    for p in pts:
        vis = facing_viewer(p, svg.camera, e2)
        if runs and runs[-1][0] == vis:
            runs[-1][1].append(p)
        else:
            # Start the new run at the previous point so the two runs join up.
            runs.append((vis, [runs[-1][1][-1], p] if runs else [p]))
    for vis, run in runs:
        if vis:
            svg.polyline(run, cls)
        elif hidden_cls:
            svg.polyline(run, hidden_cls)


def ellipsoid_outline(camera: Camera, e2: float = WGS84_E2, samples: int = 241) -> list[Vec3]:
    """Silhouette under orthographic projection: for each screen direction d, the surface
    point that extends furthest along d is diag(1, 1, b²)·d / sqrt(d·diag(1, 1, b²)·d)."""
    b2 = 1.0 - e2
    out = []
    for i in range(samples):
        t = 2.0 * pi * i / (samples - 1)
        d = add(mul(cos(t), camera.x_axis), mul(sin(t), camera.y_axis))
        md = (d[0], d[1], b2 * d[2])
        out.append(mul(1.0 / sqrt(dot(d, md)), md))
    return out


def overview_globe(title: str) -> tuple[Svg, list[Vec3]]:
    """The shared Earth of 00b/00c: front-side graticule, outline, equator, P's meridian."""
    lam_p = overview_point()[1]
    lat_lines = [[overview_surface(f * pi / 180.0, 2.0 * pi * i / 144) for i in range(145)] for f in (-60, -30, 30, 60)]
    lon_lines = [[overview_surface(-pi / 2.0 + pi * i / 120, l * pi / 180.0) for i in range(121)] for l in range(0, 360, 30)]
    equator = [overview_surface(0.0, 2.0 * pi * i / 144) for i in range(145)]
    meridian = [overview_surface(-pi / 2.0 + pi * i / 120, lam_p) for i in range(121)]
    camera = Camera(OVERVIEW_GLOBE_EYE)
    outline = ellipsoid_outline(camera, OVERVIEW_E2)
    ends = [(1.4, 0.0, 0.0), (0.0, 1.4, 0.0), (0.0, 0.0, 1.3)]
    target = geodetic_to_ecef_unit(*overview_point(), OVERVIEW_E2)
    svg = Svg(560, 460, title, outline + ends + [target], camera=camera)
    for line in lat_lines + lon_lines:
        surface_polyline(svg, line, "wire", e2=OVERVIEW_E2)
    surface_polyline(svg, equator, "equator", "hidden", OVERVIEW_E2)
    surface_polyline(svg, meridian, "meridian", "hidden", OVERVIEW_E2)
    svg.polyline(outline, "outline")
    for end, label in zip(ends, "XYZ"):
        svg.line((0.0, 0.0, 0.0), end, "axis", True)
        svg.text(label, end, 8, 4)
    svg.text("O", (0.0, 0.0, 0.0), -20, 6)
    return svg, ends


def overview_enu_frame() -> str:
    phi0, lam0 = REFERENCE_PHI, REFERENCE_LAMBDA
    origin = overview_surface(phi0, lam0)
    east, north, up = east_unit(lam0), north_unit(phi0, lam0), normal_unit(phi0, lam0)
    target = geodetic_to_ecef_unit(*overview_point(), OVERVIEW_E2)
    delta = sub(target, origin)
    e_c, n_c, u_c = dot(delta, east), dot(delta, north), dot(delta, up)
    assert min(e_c, n_c, u_c) > 0.0, "P must lie east, north and above O_L to draw the coordinate box"
    # Coordinate box: P' is P dropped onto the tangent plane; its feet on the E and N axes.
    foot_e = add(origin, mul(e_c, east))
    foot_n = add(origin, mul(n_c, north))
    p_plane = add(foot_e, mul(n_c, north))
    foot_u = add(origin, mul(u_c, up))
    axis_len = 1.35 * max(e_c, n_c, u_c)

    # A quarter of the northern hemisphere: 90° of longitude centred on the origin.
    lon_lo, lon_hi = lam0 - pi / 4.0, lam0 + pi / 4.0
    meridians = [
        [overview_surface(pi / 2.0 * i / 60, lon_lo + (lon_hi - lon_lo) * k / 6) for i in range(61)]
        for k in range(7)
    ]
    parallels = [
        [overview_surface(pi / 2.0 * j / 6, lon_lo + (lon_hi - lon_lo) * i / 60) for i in range(61)]
        for j in range(6)
    ]
    # Large enough that P' falls inside the drawn plane (its N half-extent is 0.7 × size).
    plane = tangent_plane_polygon(origin, phi0, lam0, size=1.08 * max(e_c, n_c / 0.7))
    axes = [add(origin, mul(axis_len, v)) for v in (east, north, up)]
    pts = [p for line in meridians + parallels for p in line] + plane + [target] + axes

    svg = Svg(560, 460, "Local ENU frame on a quarter of the northern hemisphere", pts, camera=Camera(ENU_CAMERA_EYE))
    for k, line in enumerate(meridians):
        svg.polyline(line, "outline" if k in (0, 6) else "wire")
    for j, line in enumerate(parallels):
        svg.polyline(line, "outline" if j == 0 else "wire")
    polygon_points(svg, plane, "plane")
    svg.line(origin, axes[0], "east", True)
    svg.line(origin, axes[1], "north", True)
    svg.line(origin, axes[2], "normal", True)
    svg.line(target, p_plane, "helper")
    svg.line(p_plane, foot_e, "helper")
    svg.line(p_plane, foot_n, "helper")
    svg.line(target, foot_u, "helper")
    for foot in (foot_e, foot_n, foot_u, p_plane):
        svg.circle(foot, 3.2, "surface")
    svg.circle(origin, 5.6, "surface")
    svg.circle(target, 6.2, "point")
    svg.text("E axis ê₀", axes[0], 8, 14, "teal")
    svg.text("N axis n̂₀", axes[1], -34, -10, "indigo")
    svg.text("U axis û₀", axes[2], -30, -10, "orange")
    svg.text("O_L", origin, -32, 16, "orange")
    svg.text("P(E, N, U)", target, 10, -8, "teal")
    svg.text("P′", p_plane, 8, 16, "small")
    svg.text("E", foot_e, 2, 20, "teal")
    svg.text("N", foot_n, 8, 16, "indigo")
    svg.text("U", foot_u, -18, 4, "orange")
    return svg.render()


def overview_ecef_frame() -> str:
    svg, _ = overview_globe("ECEF frame centred on the Earth")
    target = geodetic_to_ecef_unit(*overview_point(), OVERVIEW_E2)
    q = (target[0], target[1], 0.0)
    svg.line((0.0, 0.0, 0.0), target, "vector", True)
    svg.line((target[0], 0.0, 0.0), q, "helper")
    svg.line((0.0, target[1], 0.0), q, "helper")
    svg.line(q, target, "helper")
    svg.circle(q, 4.4, "surface")
    svg.circle(target, 6.2, "point")
    svg.text("P(X, Y, Z)", target, 10, -8, "teal")
    svg.text("(X, Y, 0)", q, 8, 16, "small")
    return svg.render()


def overview_geodetic_frame() -> str:
    phi, lam, h = overview_point()
    nu = 1.0 / sqrt(1.0 - OVERVIEW_E2 * sin(phi) ** 2)
    s = overview_surface(phi, lam)
    n = normal_unit(phi, lam)
    target = add(s, mul(h, n))
    radial = (cos(lam), sin(lam), 0.0)
    # The normal through S meets the equatorial plane at F = e²ν cosφ, not at the centre O.
    f_point = mul(OVERVIEW_E2 * nu * cos(phi), radial)
    b = OVERVIEW_B_OVER_A

    svg, _ = overview_globe("WGS 84 geodetic coordinates on an oblate ellipsoid")
    svg.line((0.0, 0.0, 0.0), radial, "helper")
    svg.line(f_point, s, "helper")
    svg.line(s, target, "normal", True)
    svg.polyline(arc3d((0.0, 0.0, 0.0), 0.32, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), lam), "arc")
    svg.polyline(arc3d(f_point, 0.24, radial, (0.0, 0.0, 1.0), phi), "arc")
    svg.circle(s, 4.8, "surface")
    svg.circle(f_point, 3.6, "surface")
    svg.circle(target, 6.2, "point")
    # Semi-axes: a along X in the equatorial plane, b along Z.
    svg.circle((1.0, 0.0, 0.0), 3.2, "surface")
    svg.circle((0.0, 0.0, b), 3.2, "surface")
    svg.text("a", (0.8, 0.0, 0.0), -20, 2)
    svg.text("b", (0.0, 0.0, b * 0.5), -18, 4)
    svg.text("λ", arc3d((0.0, 0.0, 0.0), 0.4, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), lam / 2.0)[-1], -4, 6, "indigo")
    svg.text("φ", arc3d(f_point, 0.3, radial, (0.0, 0.0, 1.0), phi / 2.0)[-1], 4, 4, "indigo")
    svg.text("S", s, -20, 4, "orange")
    svg.text("h", add(s, mul(0.5 * h, n)), -18, -2, "orange")
    svg.text("P(φ, λ, h)", target, 10, -8, "teal")
    return svg.render()


def diagram_values_readme() -> str:
    phi = REFERENCE_PHI
    lam = REFERENCE_LAMBDA
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
- Reference latitude phi: {REFERENCE_PHI_DEG:.9f} deg
- Reference longitude lambda: {REFERENCE_LAMBDA_DEG:.9f} deg
- Reference surface point S/a: ({s[0]:.9f}, {s[1]:.9f}, {s[2]:.9f})
- Reference normal u-hat: ({n[0]:.9f}, {n[1]:.9f}, {n[2]:.9f})
- Geocentric latitude psi at reference latitude: {degrees(psi):.9f} deg
- Meridian radius M at reference latitude: {meridian_radius_over_a(phi) * WGS84_A_M:.3f} m
- Prime vertical radius nu at reference latitude: {nu_over_a(phi) * WGS84_A_M:.3f} m
- p-z section diagram: fixed lambda meridian half-section, z=z_S parallel circle, and p=p_S cylinder surface.
- Prime vertical projection diagram: parallel-circle radius rho=nu*cos(phi), circle curvature, and normal-curvature projection factor cos(phi).
- Iterative inverse atan2 diagram: fixed-longitude p-z section, corrected vertical leg Z+e^2*nu_n*sin(phi_n), and fixed-point latitude update.
- ENU basis derivation diagram: east from parallel tangent, up from ellipsoid normal, north from u-hat cross e-hat.
- Visual height arrow: {VISUAL_H_OVER_A:.3f} a, used only to make the normal direction visible.
- Tangent slope diagram delta p: 0.12 a, used only to show the limiting secant visually.
- Local tangent ENU diagram visual offsets: E=0.45 a, N=0.30 a, U=0.22 a, used only to show the decomposition.
- Frame figures 00a-00c: P at reference phi+{OVERVIEW_P_DPHI_DEG:g} deg, lambda+{OVERVIEW_P_DLAMBDA_DEG:g} deg, h={OVERVIEW_P_H_OVER_A:g} a (visual); all three use one ellipsoid with b/a={OVERVIEW_B_OVER_A:g} (flattening exaggerated).
- EPSG example radial distance ||P|| for h=73.0 m: {epsg_r:.3f} m. This is not the ellipsoidal height.
"""


# Font size (px) of every text class; widths are estimated from it to fit the viewBox.
TEXT_CLASS_PX = {"label": 14.0, "small": 12.5, "teal": 14.0, "orange": 14.0, "indigo": 14.0, "panel-title": 16.0}
AVG_CHAR_EM = 0.62  # bold sans-serif average advance; generous so labels never clip
# A combining circumflex after a letter with no precomposed glyph (n̂, q̂) renders
# shifted right in most fonts, so the hat is drawn as its own glyph centred on the
# letter. Advances in em, measured in Chrome for the -apple-system labels: lowercase
# n/q ≈ 0.60, modifier circumflex U+02C6 ≈ 0.49 (its ink already sits above x-height).
HAT_BASE_EM = 0.60
HAT_GLYPH_EM = 0.49

_TEXT_RE = re.compile(r'<text class="([\w-]+)" x="([-\d.]+)" y="([-\d.]+)">(.*?)</text>')
_NUM = r"(-?\d+(?:\.\d+)?)"


def _hat_markup(content: str) -> str:
    pieces = content.split("\u0302")
    markup = pieces[0]
    to_centre = -(HAT_BASE_EM + HAT_GLYPH_EM) / 2.0
    back = (HAT_BASE_EM - HAT_GLYPH_EM) / 2.0
    for rest in pieces[1:]:
        markup += f'<tspan dx="{to_centre:.3f}em">\u02c6</tspan>'
        if rest:
            markup += f'<tspan dx="{back:.3f}em">{rest}</tspan>'
    return markup


def typeset_hats(svg: str) -> str:
    def one(m: re.Match[str]) -> str:
        content = unicodedata.normalize("NFC", m.group(4))
        return f'<text class="{m.group(1)}" x="{m.group(2)}" y="{m.group(3)}">{_hat_markup(content)}</text>'

    return _TEXT_RE.sub(one, svg)


def content_bbox(svg: str) -> tuple[float, float, float, float]:
    """(x, y, width, height) of the drawn geometry and (estimated) label extents, plus margin."""
    xs: list[float] = []
    ys: list[float] = []
    for x1, y1, x2, y2 in re.findall(rf'x1="{_NUM}" y1="{_NUM}" x2="{_NUM}" y2="{_NUM}"', svg):
        xs += [float(x1), float(x2)]
        ys += [float(y1), float(y2)]
    for pts in re.findall(r'points="([^"]+)"', svg):
        for pair in pts.split():
            x, y = pair.split(",")
            xs.append(float(x))
            ys.append(float(y))
    for cx, cy, r in re.findall(rf'cx="{_NUM}" cy="{_NUM}" r="{_NUM}"', svg):
        xs += [float(cx) - float(r), float(cx) + float(r)]
        ys += [float(cy) - float(r), float(cy) + float(r)]
    for cx, cy, rx, ry in re.findall(rf'cx="{_NUM}" cy="{_NUM}" rx="{_NUM}" ry="{_NUM}"', svg):
        xs += [float(cx) - float(rx), float(cx) + float(rx)]
        ys += [float(cy) - float(ry), float(cy) + float(ry)]
    for x, y, w, h in re.findall(rf'<rect[^>]* x="{_NUM}" y="{_NUM}" width="{_NUM}" height="{_NUM}"', svg):
        xs += [float(x), float(x) + float(w)]
        ys += [float(y), float(y) + float(h)]
    for cls, x, y, content in _TEXT_RE.findall(svg):
        px = TEXT_CLASS_PX[cls]
        chars = sum(1 for c in content if not unicodedata.combining(c))
        xs += [float(x), float(x) + chars * AVG_CHAR_EM * px]
        ys += [float(y) - 0.95 * px, float(y) + 0.3 * px]
    m = FIT_MARGIN_PX
    return min(xs) - m, min(ys) - m, max(xs) - min(xs) + 2 * m, max(ys) - min(ys) + 2 * m


def fit_viewbox(svg: str) -> str:
    """Crop the viewBox to the drawn geometry and (estimated) label extents."""
    x0, y0, w, h = content_bbox(svg)
    return re.sub(r'viewBox="[^"]+"', f'viewBox="{x0:.1f} {y0:.1f} {w:.1f} {h:.1f}"', svg, count=1)


def write(path: Path, content: str) -> None:
    if path.suffix == ".svg":
        content = typeset_hats(fit_viewbox(content))
    path.write_text(content, encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    write(ASSET_DIR / "01-coordinate-systems.svg", diagram_coordinate_system())
    write(ASSET_DIR / "00a-frame-local-enu.svg", overview_enu_frame())
    write(ASSET_DIR / "00b-frame-ecef.svg", overview_ecef_frame())
    write(ASSET_DIR / "00c-frame-wgs84-geodetic.svg", overview_geodetic_frame())
    write(ASSET_DIR / "02-pz-section-geometry.svg", diagram_pz_section_geometry())
    write(ASSET_DIR / "02-geodetic-vs-geocentric-latitude.svg", diagram_latitudes())
    write(ASSET_DIR / "03-curvature-radii.svg", diagram_curvature_radii())
    write(ASSET_DIR / "03-prime-vertical-projection.svg", diagram_prime_vertical_projection())
    write(ASSET_DIR / "03-tangent-slope-dp.svg", diagram_tangent_slope_dp())
    write(ASSET_DIR / "04-forward-geodetic-to-ecef.svg", diagram_forward())
    write(ASSET_DIR / "05-inverse-ecef-to-geodetic.svg", diagram_inverse())
    write(ASSET_DIR / "05-iterative-inverse-atan2.svg", diagram_iterative_inverse_atan2())
    write(ASSET_DIR / "06-height-normal-vs-radial.svg", diagram_height())
    write(ASSET_DIR / "07-enu-basis-derivation.svg", diagram_enu_basis_derivation())
    write(ASSET_DIR / "07-local-tangent-enu.svg", diagram_local_tangent_enu())
    write(ASSET_DIR / "README.md", diagram_values_readme())


if __name__ == "__main__":
    main()
