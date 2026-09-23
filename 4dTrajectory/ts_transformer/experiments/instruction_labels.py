"""Instruction labeller, step 3: read every train and val flight into its sentence with the
spec of step 2, and write the sentences and the readout (vocabulary design §7).

Writes ``sentences_{train,val}.npz``, ``labels.json``, ``readout.json`` and ``readout.md``
into the signals directory (never over an existing file). The candidate runways are the
artefact's own ``candidates.json`` (written with the signals).

    python run_ts.py instruction_labels --dir 4dTrajectory/outputs/POOLED/instruction_language/<name>
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import (
    SPLITS, load_candidates, load_signals, load_spec, require_current_labeller, write_sentences,
)
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.readout import class_usage, flight_record, summarise
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT

CHUNK = 1000


def _label(flights: list[Any], spec_data: dict[str, Any], geometry_data: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    spec = VocabularySpec.from_dict(spec_data)
    words = Words(spec)
    geometries = {code: AirportGeometry.from_dict(data) for code, data in geometry_data.items()}
    results = []
    for flight in flights:
        geometry = geometries[flight.airport]
        try:
            reading = read_flight(flight, geometry, spec, words)
        except Refused as refusal:
            results.append(("refused", None, {"dataset_id": flight.dataset_id, "airport": flight.airport,
                                              "status": "refused", "reason": refusal.reason, "detail": refusal.detail}))
            continue
        record = flight_record(reading)
        reading.checks = {}
        results.append(("labelled", reading, record))
    return results


def _fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f} %"


def render(summary: dict[str, Any], spec: VocabularySpec) -> str:
    lines = [f"# 指令标注读数（spec `{spec.sha256[:12]}`，{spec.reading_rule}）", ""]
    lines += ["## 1 完整性", "", "| 划分 | 成功 | 拒绝 |", "|---|---|---|"]
    for split in SPLITS:
        s = summary[split]
        lines.append(f"| {split} | {s['labelled']} | {s['refused']} |")
    lines += ["", "拒绝原因（train + val）：", ""]
    reasons: dict[str, int] = {}
    for split in SPLITS:
        for reason, count in summary[split]["refusal_reasons"].items():
            reasons[reason] = reasons.get(reason, 0) + count
    lines += [f"- {reason}：{count}" for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])] or ["- 无"]

    def group_rows(title: str, groups: dict[str, dict[str, Any]]) -> list[str]:
        out = ["", f"### {title}", "",
               "| 组 | 架次 | 每架指令 均值 / p95 | 航向 | 高度 | 下降角 | 速度 | 进近 | 非沉默步 | 大转弯拆分/架 | 插入切入航向 | "
               f"航向词中 < {spec.turn_bank_min_from_deg:g}° 的小修正 | 在入口处截断 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, g in groups.items():
            w = g["words_per_flight"]
            out.append(f"| {name} | {g['flights']} | {_fmt(g['instructions_per_flight'].get('mean'))} / "
                       f"{_fmt(g['instructions_per_flight'].get('p95'), 0)} | "
                       + " | ".join(_fmt(w[c].get("mean"), 2) for c in ("heading", "altitude", "angle", "speed", "approach"))
                       + f" | {_pct(g['non_silent_step_share'])} | {_fmt(g['heading_splits_per_flight'].get('mean'), 2)} | "
                       f"{_pct(g['intercept_inserted_share'])} | {_pct(g['small_heading_turn_share'])} | "
                       f"{_pct(g['cut_at_crossing_share'])} |")
        return out

    def envelope_rows(title: str, groups: dict[str, dict[str, Any]]) -> list[str]:
        out = ["", f"### {title}", "",
               "| 组 | 转弯单调 | 转弯坡度内 | 插入的切入航向单调 | 截获转弯单调 / 坡度内 | "
               "保持段末漏斗半宽 p50 / p95 (m) | 截获时离入口 p5 / p50 (km) | 高度词包含 | 高度逐行 | "
               "管子末宽 p50 / p95 (m) | 下降至落地末宽 p50 / p95 (m) | 速度词包含 | 速度带内逐行 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, g in groups.items():
            h, a, v, c = g["hold_funnel_half_width_end_m"], g["altitude"], g["speed"], g["capture_turns"]
            d = g["capture_before_threshold_m"]
            out.append(f"| {name} | {_pct(g['turns']['progress_ok'])} | {_pct(g['turns']['bank_ok'])} | "
                       f"{_pct(g['turns']['intercept_progress_ok'])} | "
                       f"{_pct(c['progress_ok'])} / {_pct(c['bank_ok'])} | "
                       f"{_fmt(h.get('p50'), 0)} / {_fmt(h.get('p95'), 0)} | "
                       f"{_fmt(d['p5'] / 1000, 1)} / {_fmt(d['p50'] / 1000, 1)} | "
                       f"{_pct(a['contained_share'])} | {_pct(a['row_share'])} | "
                       f"{_fmt(a['tube_width_end_m'].get('p50'), 0)} / {_fmt(a['tube_width_end_m'].get('p95'), 0)} | "
                       f"{_fmt(a['land_tube_width_end_m'].get('p50'), 0)} / {_fmt(a['land_tube_width_end_m'].get('p95'), 0)} | "
                       f"{_pct(v['contained_share'])} | {_pct(v['band_row_share'])} |")
        return out

    for split in SPLITS:
        s = summary[split]
        lines += ["", f"## {split}", ""]
        lines += group_rows("句子长度（第 0 步之后的指令数，每架）",
                            {"全部": s["all"], **s["by_stratum"], **s["by_airport"]})
        lines += envelope_rows("包络：包含率与宽度", {"全部": s["all"], **s["by_stratum"], **s["by_airport"]})
    lines += ["", "## 类别使用（train，含第 0 步）", "", "| 列 | 类别数 | 用到 | 少于 10 次 | 最少一类的次数 |", "|---|---|---|---|---|"]
    for column, u in summary["train"]["class_usage"].items():
        lines.append(f"| {column} | {u['classes']} | {u['used']} | {u['rare_under_10']} | {u['min_used']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--dir", type=Path, required=True, help="the directory instruction_signals and instruction_spec wrote")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    directory = args.dir if args.dir.is_absolute() else REPO_ROOT / args.dir
    for name in ("sentences_train.npz", "sentences_val.npz", "labels.json", "readout.json", "readout.md"):
        if (directory / name).exists():
            parser.error(f"{directory / name} exists; an instruction artefact is never overwritten")
    started = time.perf_counter()
    require_current_labeller(directory)
    spec = load_spec(directory)
    words = Words(spec)
    geometry_data = {code: geometry.to_dict() for code, geometry in load_candidates(directory).items()}

    summary: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        for split in SPLITS:
            flights = load_signals(directory, split)
            futures = [pool.submit(_label, flights[i: i + CHUNK], spec.to_dict(), geometry_data) for i in range(0, len(flights), CHUNK)]
            readings, indices, records, refusals = [], [], [], []
            for chunk, future in enumerate(futures):
                for offset, (status, reading, record) in enumerate(future.result()):
                    if status == "labelled":
                        readings.append(reading)
                        indices.append(chunk * CHUNK + offset)
                        records.append(record)
                    else:
                        refusals.append(record)
            print(f"  {split}: {len(readings)} labelled, {len(refusals)} refused, "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
            if not readings:
                raise SystemExit(f"no {split} flight was labelled ({len(refusals)} refused): nothing to write")
            require_current_labeller(directory)      # the code did not change while the workers read
            write_sentences(directory, split, spec, readings, indices)
            summary[split] = {**summarise(records, refusals),
                              "class_usage": class_usage(np.concatenate([r.words for r in readings]), words)}
            labels[split] = {"labelled": records, "refused": refusals}
    write_json_atomic(directory / "labels.json", {"spec_sha256": spec.sha256, "written_utc": utc_now(), **labels})
    write_json_atomic(directory / "readout.json", {"spec_sha256": spec.sha256, "written_utc": utc_now(),
                                                   "columns": list(COLUMNS), "elapsed_s": time.perf_counter() - started,
                                                   **summary})
    text = render(summary, spec)
    (directory / "readout.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
