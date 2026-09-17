"""The segment features (two-tier v2 §4): what the plan layer attends over.

The observed window is cut into K coarse segments of `PLAN_WAYPOINT_SEGMENT_S` and every
segment becomes one row of features — the three groups the requirement names (the segment's
START and END, its OVERALL DIRECTION, the PATH itself) and the final-approach context —
all in RUNWAY AXES about the anchor (the window's last row), so the same manoeuvre reads
the same on every runway. Computed in torch from the physical window inside the model
(`SegmentPlanModel`), never by the dataset: the window contract stays the six channels.

Under ``segment_plan_attention="channels"`` each FEATURE's K-long series is a token
(iTransformer's inversion — attention between the channels the requirement names); under
``"segments"`` each SEGMENT's feature vector is a token (attention between segments).
"""

from __future__ import annotations

import math

import torch

from ts_transformer.data.channels import IDX

#: Sub-samples of the path inside a segment (equally spaced, the segment's end included).
PATH_SUBSAMPLES = 5
#: The scales that put every feature at O(1): a segment reaches a few km, a window a few
#: tens; heights hundreds of metres; approach speeds ~100 m/s.
WINDOW_SCALE_M = 10_000.0
SEGMENT_SCALE_M = 5_000.0
HEIGHT_SCALE_M = 1_000.0
SPEED_SCALE_MPS = 100.0

SEGMENT_FEATURES: tuple[str, ...] = (
    "start_along", "start_across", "start_up",
    "end_along", "end_across", "end_up",
    "direction_cos", "direction_sin", "descent_ratio",
    *(f"path{i}_{axis}" for i in range(1, PATH_SUBSAMPLES + 1) for axis in ("along", "across")),
    "path_length", "mean_ground_speed", "heading_change",
    "anchor_distance", "threshold_bearing_cos", "threshold_bearing_sin",
    "course_cos", "course_sin",
)


def rotate(de: torch.Tensor, dn: torch.Tensor, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Chart deltas ``[B, ...]`` → (along, across) the course ``psi`` ``[B]`` (`labels.runway_deltas`)."""
    shape = (-1,) + (1,) * (de.dim() - 1)
    c, s = torch.cos(psi).view(shape), torch.sin(psi).view(shape)
    return de * c + dn * s, -de * s + dn * c


def segment_features(
    window: torch.Tensor, psi: torch.Tensor, target_chart: torch.Tensor, *, segment_samples: int,
) -> torch.Tensor:
    """``[B, L, C]`` physical window (the six channels, the anchor LAST) → ``[B, K, F]``
    features, K = (L − 1) // segment_samples segments of ``segment_samples`` intervals
    each, read from the window's END backwards so the last segment ends at the anchor.
    ``psi`` ``[B]`` is the runway course, ``target_chart`` ``[B, 3]`` the threshold's chart
    position — what the approach context is measured FROM (the chart origin only under the
    threshold-anchored frames, `data.channels.target_chart_position`)."""
    batch, length, _channels = window.shape
    count = (length - 1) // segment_samples
    if count < 1:
        raise ValueError(f"a {length}-row window holds no {segment_samples}-interval segment")
    first = length - 1 - count * segment_samples
    e, n, u = window[..., IDX["e"]], window[..., IDX["n"]], window[..., IDX["u"]]
    edot, ndot = window[..., IDX["edot"]], window[..., IDX["ndot"]]
    anchor_e, anchor_n, anchor_u = e[:, -1:], n[:, -1:], u[:, -1:]
    along, across = rotate(e - anchor_e, n - anchor_n, psi)
    up = u - anchor_u
    speed = torch.sqrt(edot.square() + ndot.square())
    heading = torch.atan2(ndot, edot)
    sub = torch.linspace(0, segment_samples, PATH_SUBSAMPLES + 1, device=window.device)[1:].round().long()
    rows = []
    for j in range(count):
        a, b = first + j * segment_samples, first + (j + 1) * segment_samples
        seg_along, seg_across, seg_up = along[:, a:b + 1], across[:, a:b + 1], up[:, a:b + 1]
        d_along, d_across = seg_along[:, -1] - seg_along[:, 0], seg_across[:, -1] - seg_across[:, 0]
        horizontal = torch.sqrt(d_along.square() + d_across.square()).clamp(min=1.0)
        chords = torch.sqrt(
            (seg_along[:, 1:] - seg_along[:, :-1]).square() + (seg_across[:, 1:] - seg_across[:, :-1]).square()
        ).sum(dim=1)
        turn = torch.remainder(heading[:, b] - heading[:, a] + math.pi, 2.0 * math.pi) - math.pi
        path = torch.stack([
            torch.stack((seg_along[:, i] - seg_along[:, 0], seg_across[:, i] - seg_across[:, 0]), dim=1) for i in sub.tolist()
        ], dim=1).reshape(batch, -1) / SEGMENT_SCALE_M
        rows.append(torch.cat([
            torch.stack((seg_along[:, 0] / WINDOW_SCALE_M, seg_across[:, 0] / WINDOW_SCALE_M, seg_up[:, 0] / HEIGHT_SCALE_M), dim=1),
            torch.stack((seg_along[:, -1] / WINDOW_SCALE_M, seg_across[:, -1] / WINDOW_SCALE_M, seg_up[:, -1] / HEIGHT_SCALE_M), dim=1),
            torch.stack((d_along / horizontal, d_across / horizontal, (seg_up[:, -1] - seg_up[:, 0]) / horizontal), dim=1),
            path,
            torch.stack((
                chords / SEGMENT_SCALE_M, speed[:, a:b + 1].mean(dim=1) / SPEED_SCALE_MPS, turn / math.pi,
            ), dim=1),
        ], dim=1))
    # the final-approach context, the same on every segment of a flight: the anchor's distance
    # to the threshold, the threshold's bearing in the course's axes, the course
    to_e, to_n = target_chart[:, 0] - anchor_e[:, 0], target_chart[:, 1] - anchor_n[:, 0]
    to_threshold_along, to_threshold_across = rotate(to_e, to_n, psi)
    distance = torch.sqrt(to_e.square() + to_n.square()).clamp(min=1.0)
    context = torch.stack((
        distance / WINDOW_SCALE_M, to_threshold_along / distance, to_threshold_across / distance,
        torch.cos(psi), torch.sin(psi),
    ), dim=1)
    features = torch.stack(rows, dim=1)                                    # [B, K, F_segment]
    features = torch.cat([features, context.unsqueeze(1).expand(-1, count, -1)], dim=2)
    if features.shape[2] != len(SEGMENT_FEATURES):
        raise RuntimeError(f"{features.shape[2]} features built, {len(SEGMENT_FEATURES)} named")
    return features


__all__ = [
    "HEIGHT_SCALE_M", "PATH_SUBSAMPLES", "SEGMENT_FEATURES", "SEGMENT_SCALE_M", "SPEED_SCALE_MPS", "WINDOW_SCALE_M",
    "rotate", "segment_features",
]
