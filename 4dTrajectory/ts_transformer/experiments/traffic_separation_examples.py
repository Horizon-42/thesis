"""Draw recorded examples of the separation losses the census finds under the IFR reading (readout
`docs/two_tier/readouts/2026-09-27_parallel_runway_separation.md`).

From a traffic census (`experiments/traffic_census.py`) and the instruction artefact it read, one episode per category is
drawn from the real tracks of its two flights:

- ``turn_on_independent`` — KSMF, independent parallels: one aircraft established, the other turning onto the next final;
- ``close_parallel_pair`` — KSJC, a pair under 2,500 ft separated as one runway: in trail across the two runways;
- ``dependent_aligned`` — KRDU, dependent parallels: both headings within 10° of their courses, not both captured yet;
- ``crossing_established`` — KRDU, runways of other directions, both established on their finals;
- ``same_runway_in_trail`` — KSTL, one runway, in trail under the minimum (a loss under both readings).

The episode drawn is the category's TYPICAL one: its closest approach (distance over the minimum) nearest the category's
median, the earliest of equals — chosen by rule, not by eye. Each figure has a plan view (airport frame, north up; the
flights' full tracks faint, three minutes either side of the episode solid, the episode itself thick; runways and
their extended centrelines; where each is captured) and the pair's horizontal distance and height difference over
time against 3 NM and 1,000 ft. For each airport and relation a category is drawn from, ``examples.json`` also summarises
every IFR episode of that relation at its start: how many had both, one or neither aircraft established, and the larger
heading off the course, the larger distance off the centreline and the height difference between the two. Writes
``<category>.svg`` and ``examples.json`` into ``--out``.

    python run_ts.py traffic_separation_examples --census 4dTrajectory/outputs/POOLED/traffic/census_<date> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --out 4dTrajectory/ts_transformer/docs/two_tier/readouts/figures/parallel_runway_separation
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from geokit import FT_M, NM_M

from ts_transformer.experiments.traffic_census import APPROACH_SPEED_MPS, Track, track
from ts_transformer.inference.runway_schedule import FAA_RADAR_NM, FAA_VERTICAL_FT, faa_separation
from ts_transformer.inference.separation import IFR, VISUAL
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals, load_spec
from ts_transformer.io_utils import write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, repo_relative

#: Seconds of track drawn solid either side of the episode.
AROUND_S = 180.0
#: Both headings within this of their courses: aligned with the final (``dependent_aligned``).
ALIGNED_DEG = 10.0


def _in_visual(census: dict[str, Any], airport: str, pair: list[str], first_s: float) -> bool:
    return any(e["pair"] == pair and e["first_s"] <= first_s <= e["last_s"]
               for e in census["airports"][airport]["readings"][VISUAL]["episodes"])


#: category → (airport, which IFR episodes belong to it, given the episode and the heading offsets at its start).
CATEGORIES: dict[str, tuple[str, Callable[[dict[str, Any], tuple[float, float], tuple[bool, bool]], bool]]] = {
    "turn_on_independent": ("KSMF", lambda e, h, est: e["relation"] == "independent" and est[0] != est[1]),
    "close_parallel_pair": ("KSJC", lambda e, h, est: e["relation"] == "single" and "in_trail" in e["kinds"]),
    "dependent_aligned": ("KRDU", lambda e, h, est: e["relation"] == "dependent" and not all(est)
                          and max(h) <= ALIGNED_DEG),
    "crossing_established": ("KRDU", lambda e, h, est: e["relation"] == "unrelated" and all(est)),
    "same_runway_in_trail": ("KSTL", lambda e, h, est: e["relation"] == "same" and e["kinds"] == ["in_trail"]),
}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def p50(name: str) -> float:
        return float(np.median([r[name] for r in rows]))

    return {"episodes": len(rows), "both_established": sum(all(r["established"]) for r in rows),
            "one_established": sum(r["established"][0] != r["established"][1] for r in rows),
            "neither_established": sum(not any(r["established"]) for r in rows),
            "both_headings_within_10_deg_share": float(np.mean([max(r["headings"]) <= ALIGNED_DEG for r in rows])),
            "larger_heading_off_course_deg_p50": p50("max_heading"),
            "larger_off_centreline_m_p50": p50("max_lateral"), "height_difference_m_p50": p50("vertical"),
            "min_distance_over_required_p50": p50("min_ratio")}


def typical(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """The episode whose closest approach (distance over the minimum) is nearest the median, the earliest of equals."""
    median = float(np.median([e["min_ratio"] for e in episodes]))
    return min(episodes, key=lambda e: (abs(e["min_ratio"] - median), e["first_s"]))


def _heading_off(item: Track, flight, geometry: AirportGeometry, t_s: float) -> float:
    candidate = geometry.candidates[geometry.candidate_index(flight.runway)]
    rows = len(item.presence.times_s)
    offset = (flight.track_deg[:rows] - candidate.course_deg + 180.0) % 360.0 - 180.0
    return abs(float(np.interp(t_s, item.presence.times_s, offset)))


def draw(path: Path, title: str, verdict: str, pair: tuple[tuple[Track, Any], tuple[Track, Any]],
         geometry: AirportGeometry, episode: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.hashsalt": "traffic-separation-examples", "svg.fonttype": "none", "font.size": 9})
    figure, (plan, series) = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    first, last = episode["first_s"], episode["last_s"]
    colours = ("#2563eb", "#ea580c")
    # the runways of the two flights, their extended centrelines, the others faint
    runways = sorted({flight.runway for _, flight in pair})
    for candidate in geometry.candidates:
        ue, un = math.sin(math.radians(candidate.course_deg)), math.cos(math.radians(candidate.course_deg))
        e0, n0 = candidate.threshold_e_m / 1000.0, candidate.threshold_n_m / 1000.0
        mine = candidate.ident in runways
        plan.plot([e0, e0 + ue * candidate.length_m / 1000.0], [n0, n0 + un * candidate.length_m / 1000.0],
                  color="#111827" if mine else "#d1d5db", linewidth=3.0 if mine else 2.0, solid_capstyle="butt")
        if mine:
            plan.plot([e0, e0 - ue * 20.0], [n0, n0 - un * 20.0], color="#9ca3af", linewidth=0.8, linestyle="--")
            # the far end of the runway, one label each side, so a close pair's two do not overlap
            side = 1 if runways.index(candidate.ident) else -1
            plan.annotate(candidate.ident, (e0 + ue * candidate.length_m / 1000.0, n0 + un * candidate.length_m / 1000.0),
                          textcoords="offset points", xytext=(6 * side, 8 * side), color="#111827",
                          ha="left" if side > 0 else "right")
    for (item, flight), colour in zip(pair, colours):
        t = item.presence.times_s
        e, n = item.e_m / 1000.0, item.n_m / 1000.0
        plan.plot(e, n, color=colour, linewidth=0.8, alpha=0.3)
        near = (t >= first - AROUND_S) & (t <= last + AROUND_S)
        plan.plot(e[near], n[near], color=colour, linewidth=1.4,
                  label=f"{flight.dataset_id.split(':')[1].split('_')[0]} → {flight.runway}")
        during = (t >= first) & (t <= last)
        plan.plot(e[during], n[during], color=colour, linewidth=4.0, alpha=0.8)
        start = np.searchsorted(t, first - AROUND_S)
        plan.annotate("", xy=(e[min(start + 3, len(e) - 1)], n[min(start + 3, len(n) - 1)]), xytext=(e[start], n[start]),
                      arrowprops={"arrowstyle": "->", "color": colour})
        if math.isfinite(item.captured_s):
            plan.plot(np.interp(item.captured_s, t, e), np.interp(item.captured_s, t, n), "o", markersize=6,
                      markerfacecolor="white", markeredgecolor=colour)
    shown = np.concatenate([np.stack([it.e_m, it.n_m], axis=1)[(it.presence.times_s >= first - AROUND_S)
                                                               & (it.presence.times_s <= last + AROUND_S)]
                            for it, _ in pair]) / 1000.0
    low, high = shown.min(axis=0) - 2.0, shown.max(axis=0) + 2.0
    plan.set_xlim(low[0], high[0])
    plan.set_ylim(low[1], high[1])
    plan.set_aspect("equal")
    plan.set_xlabel("east (km, airport frame)")
    plan.set_ylabel("north (km)")
    plan.legend(loc="best", fontsize=8, title="thick: the episode; ○ captured", title_fontsize=8)
    plan.set_title("plan view", fontsize=9)

    # the pair over time
    (a, _), (b, _) = pair
    t = np.arange(max(a.presence.times_s[0], b.presence.times_s[0], first - AROUND_S),
                  min(a.presence.times_s[-1], b.presence.times_s[-1], last + AROUND_S) + 1e-9, 2.0)
    distance = np.hypot(np.interp(t, a.presence.times_s, a.e_m) - np.interp(t, b.presence.times_s, b.e_m),
                        np.interp(t, a.presence.times_s, a.n_m) - np.interp(t, b.presence.times_s, b.n_m))
    vertical = np.abs(np.interp(t, a.presence.times_s, a.height_m) - np.interp(t, b.presence.times_s, b.height_m))
    series.plot(t - first, distance / 1000.0, color="#111827", label="horizontal distance")
    series.axhline(FAA_RADAR_NM * NM_M / 1000.0, color="#111827", linewidth=0.8, linestyle="--")
    series.annotate("3 NM", (t[0] - first, FAA_RADAR_NM * NM_M / 1000.0), textcoords="offset points", xytext=(2, 3))
    series.axvspan(0.0, last - first, color="#fecaca", alpha=0.5, label="the episode (IFR loss)")
    series.set_xlabel("seconds from the episode's start")
    series.set_ylabel("horizontal distance (km)")
    twin = series.twinx()
    twin.plot(t - first, vertical, color="#16a34a", label="height difference")
    twin.axhline(FAA_VERTICAL_FT * FT_M, color="#16a34a", linewidth=0.8, linestyle="--")
    twin.set_ylabel("height difference (m); dashed: 1,000 ft", color="#16a34a")
    lines = series.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = series.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    series.legend(lines, labels, loc="best", fontsize=8, framealpha=0.9)
    series.set_title(verdict, fontsize=9)
    figure.suptitle(title, fontsize=10, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, format="svg", metadata={"Date": None})
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--census", type=Path, required=True, help="a traffic census directory")
    parser.add_argument("--instructions", type=Path, required=True, help="the instruction artefact it read")
    parser.add_argument("--out", type=Path, required=True, help="where the figures go")
    args = parser.parse_args(argv)
    resolve = lambda p: p if p.is_absolute() else REPO_ROOT / p  # noqa: E731
    census_dir, directory, out = resolve(args.census), resolve(args.instructions), resolve(args.out)
    census = json.loads((census_dir / "census.json").read_text(encoding="utf-8"))
    if repo_relative(directory) != census["instructions"]:
        parser.error(f"the census read {census['instructions']}, not {repo_relative(directory)}")
    spec, candidates = load_spec(directory), load_candidates(directory)
    flights = load_signals(directory, census["split"])
    sentences = load_sentences(directory, census["split"], spec)
    offsets = sentences["offsets"]
    spoken = {int(s): (int(offsets[i + 1] - offsets[i]), int(sentences["capture_row"][i]))
              for i, s in enumerate(sentences["signal_index"])}
    by_key = {f.dataset_id: (i, f) for i, f in enumerate(flights)}
    out.mkdir(parents=True, exist_ok=True)
    chosen: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    census_relation = {"turn_on_independent": "independent", "close_parallel_pair": "single",
                       "dependent_aligned": "dependent", "crossing_established": "unrelated",
                       "same_runway_in_trail": "same"}
    for name, (airport, belongs) in CATEGORIES.items():
        geometry = candidates[airport]
        manifest = json.loads(arrival_manifest_path(airport).read_text(encoding="utf-8"))
        separation = faa_separation(manifest["runway_targets"], speed_mps=APPROACH_SPEED_MPS)
        tracks: dict[str, tuple[Track, Any]] = {}

        def flown(key: str) -> tuple[Track, Any]:
            if key not in tracks:
                i, flight = by_key[key]
                tracks[key] = (track(flight, *spoken.get(i, (None, None)), geometry,
                                     separation.along_nm[flight.runway] * NM_M, spec, spec.step_s), flight)
            return tracks[key]

        def off_centreline(item: Track, flight, t_s: float) -> float:
            candidate = geometry.candidates[geometry.candidate_index(flight.runway)]
            relative = relative_to_runway(item.e_m, item.n_m, flight.track_deg[:len(item.e_m)], item.height_m, candidate)
            return abs(float(np.interp(t_s, item.presence.times_s, relative.right_of_course_m)))

        members, by_relation = [], {}
        for episode in census["airports"][airport]["readings"][IFR]["episodes"]:
            pair = [flown(key) for key in episode["pair"]]
            t_s = episode["first_s"]
            headings = tuple(_heading_off(item, flight, geometry, t_s) for item, flight in pair)
            established = tuple(t_s >= item.captured_s for item, _ in pair)
            if belongs(episode, headings, established):
                members.append(episode)
            by_relation.setdefault(episode["relation"], []).append({
                "established": established, "headings": headings, "max_heading": max(headings),
                "max_lateral": max(off_centreline(item, flight, t_s) for item, flight in pair),
                "vertical": abs(float(np.interp(t_s, pair[0][0].presence.times_s, pair[0][0].height_m)
                                      - np.interp(t_s, pair[1][0].presence.times_s, pair[1][0].height_m))),
                "min_ratio": episode["min_ratio"]})
        summaries[f"{airport} {census_relation[name]}"] = _summary(by_relation[census_relation[name]])
        episode = typical(members)
        pair = tuple(flown(key) for key in episode["pair"])
        visual = _in_visual(census, airport, episode["pair"], episode["first_s"])
        a, b = pair
        rows = [relative_to_runway(f.e_m, f.n_m, f.track_deg, f.altitude_m,
                                   geometry.candidates[geometry.candidate_index(f.runway)]) for _, f in pair]
        chosen[name] = {"airport": airport, "episodes_in_category": len(members), "pair": episode["pair"],
                        "runways": [f.runway for _, f in pair], "first_s": episode["first_s"],
                        "last_s": episode["last_s"], "seconds": episode["last_s"] - episode["first_s"] + spec.step_s,
                        "kinds": episode["kinds"], "min_distance_over_required": episode["min_ratio"],
                        "loss_under_visual": visual,
                        "at_start": [{"established": episode["first_s"] >= item.captured_s,
                                      "heading_off_course_deg": _heading_off(item, f, geometry, episode["first_s"]),
                                      "off_centreline_m": abs(float(np.interp(episode["first_s"], item.presence.times_s,
                                                                              rel.right_of_course_m[:len(item.e_m)]))),
                                      "before_threshold_km": float(np.interp(
                                          episode["first_s"], item.presence.times_s,
                                          rel.before_threshold_m[:len(item.e_m)])) / 1000.0}
                                     for (item, f), rel in zip(pair, rows)]}
        verdict = (f"IFR reading: loss ({', '.join(episode['kinds'])}); "
                   f"visual reading: {'loss' if visual else 'no loss'}")
        draw(out / f"{name}.svg", f"{airport} — {name.replace('_', ' ')}", verdict, pair, geometry, episode)
        print(f"{name}: {len(members)} episodes, drew {episode['pair']} ({chosen[name]['seconds']:.0f} s)", flush=True)
    write_json_atomic(out / "examples.json", {"census": repo_relative(census_dir), "rule": "closest approach nearest "
                                              "the category's median, the earliest of equals", "examples": chosen,
                                              "episodes_at_their_start": summaries})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
