#!/usr/bin/env python3
"""
Run the OpenSky fetch stage and CZML generation stage as two decoupled steps.

This script intentionally orchestrates through CLI boundaries:
1) opensky_data_query/fetch_cylw_opensky.py
2) aeroviz-4d/python/generate_czml.py

All unknown CLI arguments are forwarded to the fetch script so this wrapper
stays low-coupling when fetch options evolve.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _has_flag(args: list[str], flag: str) -> bool:
    """检查参数列表中是否已经显式提供某个 CLI 标志。"""
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _extract_option(args: list[str], flag: str, default: str) -> str:
    """从 CLI 参数列表中读取选项值，兼容 ``--flag value`` 和 ``--flag=value`` 两种写法。"""
    value = default
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            value = args[idx + 1]
        elif arg.startswith(f"{flag}="):
            value = arg.split("=", 1)[1]
    return value


def _latest_matching_file(folder: Path, pattern: str, before: set[Path]) -> Path | None:
    """在指定目录中按通配模式查找最新文件，优先返回本次流程新生成的文件。"""
    candidates = sorted(
        folder.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    created_now = [p for p in candidates if p.resolve() not in before]
    if created_now:
        return created_now[-1]
    if candidates:
        return candidates[-1]
    return None


def _airport_output_dir(output_root: Path, airport: str) -> Path:
    """生成某个机场在 fetch 输出根目录下对应的子目录路径。"""
    return output_root / airport.lower()


def _default_czml_output(aeroviz_root: Path, airport: str) -> Path:
    """生成 AeroViz 前端默认读取的机场轨迹 CZML 输出路径。"""
    return aeroviz_root / "public" / "data" / "airports" / airport.upper() / "trajectories.czml"


def _default_procedure_output(aeroviz_root: Path, airport: str) -> Path:
    """生成 AeroViz 前端默认读取的机场程序 GeoJSON 输出路径。"""
    return aeroviz_root / "public" / "data" / "airports" / airport.upper() / "procedures.geojson"


def _default_cifp_root(repo_root: Path) -> Path:
    """返回仓库内默认使用的 CIFP 周期数据目录。"""
    return repo_root / "data" / "CIFP" / "CIFP_260319"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """解析本包装脚本参数，并保留未知参数以继续转发给 OpenSky fetch 脚本。"""
    parser = argparse.ArgumentParser(
        description="Run fetch/normalization and CZML generation in one command",
        epilog=(
            "Any unknown arguments are forwarded to opensky_data_query/fetch_cylw_opensky.py.\n"
            "Common forwarded options include: --mode, --allow-partial, --approach-window-min, --altitude-mode."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--airport",
        default="CYYC",
        help="Airport ICAO for output file matching (default: CYYC)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output folder for fetch stage (defaults to opensky_data_query/outputs)",
    )
    parser.add_argument(
        "--aeroviz-root",
        default=None,
        help="Path to aeroviz-4d folder (defaults to ./aeroviz-4d)",
    )
    parser.add_argument(
        "--czml-output",
        default=None,
        help="Output CZML path (defaults to <aeroviz-root>/public/data/airports/<AIRPORT>/trajectories.czml)",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=None,
        help="Optional CZML clock multiplier passed to generate_czml.py",
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help=(
            "Existing JSON to bypass live fetch. Accepts either:\n"
            "- *_raw_*.json: runs normalization/conversion then CZML generation\n"
            "- *_czml_input_*.json: skips straight to CZML generation"
        ),
    )
    parser.add_argument(
        "--disable-normalization",
        action="store_true",
        help="Forwarded to fetch stage: export raw track waypoints without normalization/filtering",
    )
    parser.add_argument(
        "--hours-ago",
        type=float,
        default=None,
        help=(
            "Historical window start offset in hours before now (decimal, e.g. 2.5). "
            "Must be paired with --range-hours. Triggers --mode historical on the fetch stage."
        ),
    )
    parser.add_argument(
        "--range-hours",
        type=float,
        default=None,
        help="Historical window duration in hours (decimal, e.g. 1.5). Must be paired with --hours-ago.",
    )
    parser.add_argument(
        "--generate-procedures",
        action="store_true",
        help=(
            "Also regenerate AeroViz RNAV/RNP procedure assets after CZML generation. "
            "This runs aeroviz-4d/python/preprocess_procedures.py with --include-all-rnav."
        ),
    )
    parser.add_argument(
        "--cifp-root",
        default=None,
        help="CIFP cycle directory for --generate-procedures (defaults to ./data/CIFP/CIFP_260319).",
    )
    parser.add_argument(
        "--procedure-type",
        default="SIAP",
        help="CIFP procedure type for --generate-procedures (default: SIAP).",
    )
    parser.add_argument(
        "--procedure-output",
        default=None,
        help=(
            "Output GeoJSON path for --generate-procedures "
            "(defaults to <aeroviz-root>/public/data/airports/<AIRPORT>/procedures.geojson)."
        ),
    )
    parser.add_argument(
        "--include-procedure-transitions",
        action="store_true",
        help="Pass --include-transitions to procedure preprocessing.",
    )
    parser.add_argument(
        "--procedure-charts-root",
        default=None,
        help="Optional charts root forwarded to procedure preprocessing as --charts-root.",
    )

    # Everything not recognized here is forwarded to fetch_cylw_opensky.py.
    args, fetch_passthrough = parser.parse_known_args()
    return args, fetch_passthrough


def _resolve_historical_window(args: argparse.Namespace) -> tuple[int, int] | None:
    """把 --hours-ago/--range-hours 转换为历史查询的 begin/end Unix 秒时间窗。"""
    if args.hours_ago is None and args.range_hours is None:
        return None
    if args.hours_ago is None or args.range_hours is None:
        raise RuntimeError("--hours-ago and --range-hours must be provided together.")
    if args.hours_ago <= 0 or args.range_hours <= 0:
        raise RuntimeError("--hours-ago and --range-hours must be positive decimals.")

    now = int(time.time())
    begin = now - int(round(args.hours_ago * 3600))
    end = begin + int(round(args.range_hours * 3600))
    if end <= begin:
        raise RuntimeError("Computed historical window is empty; check --range-hours.")
    return begin, end


def main() -> None:
    """编排完整流水线：获取或复用轨迹 JSON，生成 CZML，并按需生成程序资产。"""
    args, fetch_passthrough = parse_args()

    historical_window = _resolve_historical_window(args)
    if historical_window is not None:
        if args.input_json:
            raise RuntimeError(
                "--hours-ago/--range-hours cannot be combined with --input-json (fetch stage is bypassed)."
            )
        for conflicting in ("--mode", "--begin", "--end"):
            if _has_flag(fetch_passthrough, conflicting):
                raise RuntimeError(
                    f"--hours-ago/--range-hours conflicts with forwarded {conflicting}; "
                    "remove one of them."
                )

    repo_root = Path(__file__).resolve().parent
    fetch_script = repo_root / "opensky_data_query" / "fetch_cylw_opensky.py"

    aeroviz_root = Path(args.aeroviz_root) if args.aeroviz_root else repo_root / "aeroviz-4d"
    output_root = Path(args.output_root) if args.output_root else repo_root / "opensky_data_query" / "outputs"

    generator_script = aeroviz_root / "python" / "generate_czml.py"

    if not fetch_script.exists():
        raise RuntimeError(f"Fetch script not found: {fetch_script}")
    if not generator_script.exists():
        raise RuntimeError(f"CZML generator script not found: {generator_script}")
    procedure_script = aeroviz_root / "python" / "preprocess_procedures.py"
    if args.generate_procedures and not procedure_script.exists():
        raise RuntimeError(f"Procedure preprocessing script not found: {procedure_script}")

    output_root.mkdir(parents=True, exist_ok=True)

    czml_input_path: Path
    effective_airport = args.airport.upper()

    if args.input_json:
        input_json_path = Path(args.input_json)
        if not input_json_path.exists():
            raise RuntimeError(f"Input JSON not found: {input_json_path}")

        loaded = json.loads(input_json_path.read_text(encoding="utf-8"))

        if isinstance(loaded, list):
            # Already in CZML-input schema, so go directly to generator.
            czml_input_path = input_json_path
            print("[Pipeline] Fetch stage bypassed: using existing CZML-input JSON.")
        elif isinstance(loaded, dict) and isinstance(loaded.get("tracks"), list):
            # Reuse existing raw payload and run conversion/normalization only.
            effective_airport = str(loaded.get("airport") or args.airport).upper()
            airport_tag = effective_airport.lower()
            pattern = f"{airport_tag}_czml_input_*.json"
            airport_output_dir = _airport_output_dir(output_root, effective_airport)
            existing = {p.resolve() for p in airport_output_dir.glob(pattern)}

            fetch_cmd = [
                sys.executable,
                str(fetch_script),
                "--input-raw-json",
                str(input_json_path),
            ]
            if not _has_flag(fetch_passthrough, "--airport"):
                fetch_cmd.extend(["--airport", effective_airport])
            if not _has_flag(fetch_passthrough, "--output-root"):
                fetch_cmd.extend(["--output-root", str(output_root)])
            if not _has_flag(fetch_passthrough, "--aeroviz-root"):
                fetch_cmd.extend(["--aeroviz-root", str(aeroviz_root)])
            if args.disable_normalization and not _has_flag(fetch_passthrough, "--disable-normalization"):
                fetch_cmd.append("--disable-normalization")
            fetch_cmd.extend(fetch_passthrough)

            print("[Pipeline] Running normalization stage from existing raw JSON...")
            subprocess.run(fetch_cmd, check=True)

            czml_input_path = _latest_matching_file(airport_output_dir, pattern, existing)
            if not czml_input_path:
                raise RuntimeError(
                    f"No CZML input JSON produced for airport {effective_airport}. "
                    f"Checked pattern {pattern} under {airport_output_dir}."
                )
        else:
            raise RuntimeError(
                "--input-json must be either a raw payload object (with key 'tracks') "
                "or a CZML-input list of flight records."
            )
    else:
        fetch_cmd = [sys.executable, str(fetch_script)]
        if not _has_flag(fetch_passthrough, "--airport"):
            fetch_cmd.extend(["--airport", args.airport])
        if not _has_flag(fetch_passthrough, "--output-root"):
            fetch_cmd.extend(["--output-root", str(output_root)])
        if not _has_flag(fetch_passthrough, "--aeroviz-root"):
            fetch_cmd.extend(["--aeroviz-root", str(aeroviz_root)])
        if args.disable_normalization and not _has_flag(fetch_passthrough, "--disable-normalization"):
            fetch_cmd.append("--disable-normalization")
        if historical_window is not None:
            begin_unix, end_unix = historical_window
            fetch_cmd.extend([
                "--mode", "historical",
                "--begin", str(begin_unix),
                "--end", str(end_unix),
            ])
            print(
                f"[Pipeline] Historical window: {args.hours_ago}h ago for {args.range_hours}h "
                f"(begin={begin_unix}, end={end_unix}).",
                flush=True,
            )
        fetch_cmd.extend(fetch_passthrough)

        effective_airport = _extract_option(fetch_cmd, "--airport", args.airport).upper()
        airport_tag = effective_airport.lower()
        pattern = f"{airport_tag}_czml_input_*.json"
        airport_output_dir = _airport_output_dir(output_root, effective_airport)
        existing = {p.resolve() for p in airport_output_dir.glob(pattern)}

        print("[Pipeline] Running fetch stage...", flush=True)
        subprocess.run(fetch_cmd, check=True)

        czml_input_path = _latest_matching_file(airport_output_dir, pattern, existing)
        if not czml_input_path:
            raise RuntimeError(
                f"No CZML input JSON produced for airport {effective_airport}. "
                f"Checked pattern {pattern} under {airport_output_dir}."
            )

    czml_output_path = (
        Path(args.czml_output)
        if args.czml_output
        else _default_czml_output(aeroviz_root, effective_airport)
    )

    generate_cmd = [
        sys.executable,
        str(generator_script),
        "--airport",
        effective_airport,
        "--input",
        str(czml_input_path),
        "--output",
        str(czml_output_path),
    ]
    if args.multiplier is not None:
        generate_cmd.extend(["--multiplier", str(args.multiplier)])

    print("[Pipeline] Running CZML generation stage...", flush=True)
    subprocess.run(generate_cmd, check=True)

    if args.generate_procedures:
        procedure_output_path = (
            Path(args.procedure_output)
            if args.procedure_output
            else _default_procedure_output(aeroviz_root, effective_airport)
        )
        procedure_cmd = [
            sys.executable,
            str(procedure_script),
            "--cifp-root",
            str(Path(args.cifp_root) if args.cifp_root else _default_cifp_root(repo_root)),
            "--airport",
            effective_airport,
            "--procedure-type",
            args.procedure_type,
            "--include-all-rnav",
            "--output",
            str(procedure_output_path),
        ]
        if args.include_procedure_transitions:
            procedure_cmd.append("--include-transitions")
        if args.procedure_charts_root:
            procedure_cmd.extend(["--charts-root", args.procedure_charts_root])

        print("[Pipeline] Running procedure asset generation stage...", flush=True)
        subprocess.run(procedure_cmd, check=True)

    print(f"[Pipeline] airport:      {effective_airport}")
    print(f"[Pipeline] czml input:   {czml_input_path}")
    print(f"[Pipeline] czml output:  {czml_output_path}")
    if args.generate_procedures:
        print(f"[Pipeline] procedures:   {procedure_output_path}")


if __name__ == "__main__":
    main()
