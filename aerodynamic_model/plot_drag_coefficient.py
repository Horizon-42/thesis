import argparse
from pathlib import Path


DEFAULT_CD0 = 0.02
DEFAULT_K = 0.04
DEFAULT_OUTPUT = "drag_coefficient_curve.svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw Cd = Cd0 + k * Cl^2 for the aerodynamic simulator."
    )
    parser.add_argument("--cd0", type=float, default=DEFAULT_CD0, help="Zero-lift drag coefficient.")
    parser.add_argument("--k", type=float, default=DEFAULT_K, help="Induced drag factor.")
    parser.add_argument("--cl-min", type=float, default=-1.0, help="Minimum lift coefficient to plot.")
    parser.add_argument("--cl-max", type=float, default=2.0, help="Maximum lift coefficient to plot.")
    parser.add_argument("--samples", type=int, default=300, help="Number of curve samples.")
    parser.add_argument(
        "--output",
        help=(
            "Optional output path. Use .svg for a dependency-free file, or another "
            "extension such as .png when matplotlib is installed."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive matplotlib plot window.",
    )
    return parser.parse_args()


def get_drag_coefficient(lift_coefficient: float, cd0: float, k: float) -> float:
    """Same drag polar as Simulator._get_drag_coefficient."""
    return cd0 + k * lift_coefficient**2


def build_curve(
    cd0: float,
    k: float,
    cl_min: float,
    cl_max: float,
    samples: int,
) -> tuple[list[float], list[float]]:
    if samples < 2:
        raise ValueError("--samples must be at least 2.")
    if cl_min >= cl_max:
        raise ValueError("--cl-min must be smaller than --cl-max.")

    lift_coefficients = [
        cl_min + (cl_max - cl_min) * index / (samples - 1)
        for index in range(samples)
    ]
    drag_coefficients = [
        get_drag_coefficient(cl, cd0, k)
        for cl in lift_coefficients
    ]
    return lift_coefficients, drag_coefficients


def svg_polyline(
    x_values: list[float],
    y_values: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    plot_left: float,
    plot_top: float,
    plot_width: float,
    plot_height: float,
) -> str:
    def scale_x(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * plot_width

    def scale_y(value: float) -> float:
        return plot_top + (y_max - value) / (y_max - y_min) * plot_height

    return " ".join(
        f"{scale_x(x):.2f},{scale_y(y):.2f}"
        for x, y in zip(x_values, y_values)
    )


def write_svg(
    output_path: str,
    lift_coefficients: list[float],
    drag_coefficients: list[float],
    cd0: float,
    k: float,
) -> None:
    width = 900
    height = 560
    plot_left = 92
    plot_top = 54
    plot_width = 740
    plot_height = 410

    x_min = min(lift_coefficients)
    x_max = max(lift_coefficients)
    y_min = min(0.0, min(drag_coefficients))
    y_max = max(drag_coefficients)
    y_padding = max((y_max - y_min) * 0.08, 0.005)
    y_min -= y_padding
    y_max += y_padding

    curve_points = svg_polyline(
        lift_coefficients,
        drag_coefficients,
        x_min,
        x_max,
        y_min,
        y_max,
        plot_left,
        plot_top,
        plot_width,
        plot_height,
    )
    cd0_point = svg_polyline(
        [0.0],
        [cd0],
        x_min,
        x_max,
        y_min,
        y_max,
        plot_left,
        plot_top,
        plot_width,
        plot_height,
    )

    x_ticks = [x_min + (x_max - x_min) * index / 6 for index in range(7)]
    y_ticks = [y_min + (y_max - y_min) * index / 6 for index in range(7)]

    def x_pos(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * plot_width

    def y_pos(value: float) -> float:
        return plot_top + (y_max - value) / (y_max - y_min) * plot_height

    x_grid = "\n".join(
        f'<line class="grid" x1="{x_pos(tick):.2f}" y1="{plot_top}" '
        f'x2="{x_pos(tick):.2f}" y2="{plot_top + plot_height}" />'
        for tick in x_ticks
    )
    y_grid = "\n".join(
        f'<line class="grid" x1="{plot_left}" y1="{y_pos(tick):.2f}" '
        f'x2="{plot_left + plot_width}" y2="{y_pos(tick):.2f}" />'
        for tick in y_ticks
    )
    x_labels = "\n".join(
        f'<text class="tick" x="{x_pos(tick):.2f}" y="{plot_top + plot_height + 28}" '
        f'text-anchor="middle">{tick:.2f}</text>'
        for tick in x_ticks
    )
    y_labels = "\n".join(
        f'<text class="tick" x="{plot_left - 14}" y="{y_pos(tick) + 4:.2f}" '
        f'text-anchor="end">{tick:.3f}</text>'
        for tick in y_ticks
    )

    cd0_x, cd0_y = cd0_point.split(",")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Drag coefficient curve">
  <style>
    .title {{ font: 700 24px Arial, sans-serif; fill: #111827; }}
    .subtitle {{ font: 14px Arial, sans-serif; fill: #4b5563; }}
    .label {{ font: 15px Arial, sans-serif; fill: #111827; }}
    .tick {{ font: 12px Arial, sans-serif; fill: #4b5563; }}
    .grid {{ stroke: #d1d5db; stroke-width: 1; opacity: 0.75; }}
    .axis {{ stroke: #111827; stroke-width: 1.6; }}
    .curve {{ fill: none; stroke: #2563eb; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text class="title" x="{plot_left}" y="32">Drag Coefficient Curve</text>
  <text class="subtitle" x="{plot_left}" y="52">Cd = Cd0 + k * Cl^2, Cd0={cd0:g}, k={k:g}</text>
  {x_grid}
  {y_grid}
  <line class="axis" x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" />
  <line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" />
  <polyline class="curve" points="{curve_points}" />
  <circle cx="{cd0_x}" cy="{cd0_y}" r="5" fill="#dc2626" />
  <text class="label" x="{float(cd0_x) + 10:.2f}" y="{float(cd0_y) - 10:.2f}">Cd0={cd0:g}</text>
  {x_labels}
  {y_labels}
  <text class="label" x="{plot_left + plot_width / 2}" y="{height - 32}" text-anchor="middle">Lift coefficient, Cl</text>
  <text class="label" x="24" y="{plot_top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 24 {plot_top + plot_height / 2})">Drag coefficient, Cd</text>
</svg>
"""
    Path(output_path).write_text(svg, encoding="utf-8")


def plot_with_matplotlib(
    output_path: str | None,
    show_plot: bool,
    lift_coefficients: list[float],
    drag_coefficients: list[float],
    cd0: float,
    k: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lift_coefficients, drag_coefficients, linewidth=2, color="#2563eb")
    ax.scatter([0.0], [cd0], color="#dc2626", zorder=3, label=f"Cd0 = {cd0:g}")
    ax.set_title(f"Drag Coefficient Curve (Cd0={cd0:g}, k={k:g})")
    ax.set_xlabel("Lift coefficient, Cl")
    ax.set_ylabel("Drag coefficient, Cd")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=160)
        print(f"Saved drag coefficient curve to {output_path}")

    if show_plot:
        plt.show()


def main() -> None:
    args = parse_args()
    lift_coefficients, drag_coefficients = build_curve(
        args.cd0,
        args.k,
        args.cl_min,
        args.cl_max,
        args.samples,
    )

    output_path = args.output or (DEFAULT_OUTPUT if args.no_show else None)
    output_suffix = Path(output_path).suffix.lower() if output_path else ""

    if output_suffix == ".svg":
        write_svg(output_path, lift_coefficients, drag_coefficients, args.cd0, args.k)
        print(f"Saved drag coefficient curve to {output_path}")
        return

    try:
        plot_with_matplotlib(
            output_path,
            show_plot=not args.no_show,
            lift_coefficients=lift_coefficients,
            drag_coefficients=drag_coefficients,
            cd0=args.cd0,
            k=args.k,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        if output_path:
            raise RuntimeError(
                "matplotlib is required for non-SVG output. Use --output curve.svg "
                "or install matplotlib."
            ) from exc
        write_svg(DEFAULT_OUTPUT, lift_coefficients, drag_coefficients, args.cd0, args.k)
        print(
            "matplotlib is not installed, so an SVG was written instead: "
            f"{DEFAULT_OUTPUT}"
        )


if __name__ == "__main__":
    main()
