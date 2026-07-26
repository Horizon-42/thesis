"""The single configuration object for a ts_transformer run.

Both vendored architectures take one ``configs`` object and read attributes off it —
that is upstream's contract (they were driven by an argparse namespace), and this
dataclass is the drop-in. Keeping data, architecture and training knobs in ONE frozen
object is deliberate: the whole thing is serialised into every checkpoint, so a trained
artifact carries the exact recipe that produced it and inference never has to guess the
resample step, the channel order, or the normalized prediction grid.

Fields are grouped by who reads them. The "read by both" and per-model groups are named
exactly as upstream expects — do not rename them without editing the vendored code, which
would break the byte-identical property PROVENANCE.md promises.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Channel order is a hard contract between the data build, the model, and the export.
# It lives in channels.py; imported here so the default cannot drift from it.
from channels import CHANNELS

MODELS = ("itransformer", "patchtst")
SAMPLING_ALL_WINDOWS = "all-windows"
SAMPLING_AIRPORT_FLIGHT_BALANCED = "airport-flight-balanced"
SAMPLING_STRATEGIES = (SAMPLING_ALL_WINDOWS, SAMPLING_AIRPORT_FLIGHT_BALANCED)
EVAL_ANCHOR_POLICIES = ("all", "first")
COORDINATE_FRAMES = ("enu", "runway-aligned")

# Time grid. ADS-B arrives at ~1 Hz but irregularly; 2 s is the resample step — fine enough
# not to smooth away the turn onto final, coarse enough not to invent samples between
# reports.
#
# Everything below is sized from the MEASURED duration distribution of the 3747 harvested
# arrivals (5 airports, rostered by `arrivals/manifest.json`, truncated at the 25 km ring):
#
#     p5  235 s | p25 271 s | p50 328 s | p75 533 s | p90 607 s | p95 651 s | p99 920 s
#
# i.e. a median arrival is ~5.5 min and the long tail reaches ~15 min. Note this is much
# longer than the naive "25 km at 120 m/s = 3.5 min" estimate the first draft used — real
# arrivals are vectored (downwind legs, base turns, the occasional hold), so the flown path
# is far longer than the straight-line distance to the ring. Sizing off the straight-line
# guess made `full` mode cover barely half an approach.
DEFAULT_DT_S = 2.0

# Lookback. 60 steps x 2 s = 120 s of observed track — long enough to contain a vectoring
# turn rather than just the straight segment before it. Raising it costs anchors twice
# over: fewer per flight, AND whole short flights dropped (p5 is only 235 s).
DEFAULT_SEQ_LEN = 60

# Every remaining approach is mapped onto the same normalized progress domain [0, 1].
# N is the number of equal progress segments (and therefore the number of future state
# endpoints).  It is deliberately independent of ``dt_s`` and is tuned by cross-validation.
DEFAULT_N_SEGMENTS = 128

# ``final_time_s`` is emitted in physical seconds.  The scale only nondimensionalizes its
# loss; it is not a duration cap and does not change the value returned at inference.
DEFAULT_FINAL_TIME_SCALE_S = 600.0

# Fallback aircraft when a flight dict has no resolvable type. Every harvested arrival is
# "UNK" today, so in practice this applies to ALL of them. Not cosmetic: it sets the target
# state's Vref and threshold-crossing height — the ENU frame and the state the evaluation
# gates judge — which is why the resolved value is a config field (serialised into every
# checkpoint) and predict defaults to the checkpoint's value, not to this constant.
DEFAULT_AIRCRAFT_TYPE = "A320"


@dataclass(frozen=True)
class TSConfig:
    """Everything that defines a run. Serialised whole into each checkpoint."""

    # ── what to train ────────────────────────────────────────────────────────
    model: str = MODELS[0]
    # ── the time grid + windowing (read by the data build AND both models) ──
    dt_s: float = DEFAULT_DT_S
    seq_len: int = DEFAULT_SEQ_LEN                  # L
    n_segments: int = DEFAULT_N_SEGMENTS            # N normalized progress segments
    channels: tuple[str, ...] = CHANNELS
    # The aircraft-type fallback the series were built with (target Vref / TCH -> the ENU
    # frame and the gate target). In the config so the checkpoint records it and predict
    # rebuilds series with the SAME frames the normalizer stats were fit under.
    aircraft_type: str = DEFAULT_AIRCRAFT_TYPE
    # ``runway-aligned`` rotates the horizontal plane so every threshold course points
    # along the first axis. It keeps the six-channel tensor shape while removing a major
    # source of cross-airport orientation variance.
    coordinate_frame: str = "enu"

    # ── architecture, shared by both models ─────────────────────────────────
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 256 # dimension of feed-forward net; need to do ablate experiments
    e_layers: int = 3 #number of encoder layers; why 3? 
    dropout: float = 0.1
    activation: str = "gelu"

    # ── iTransformer only ───────────────────────────────────────────────────
    # use_norm / PatchTST's revin below are the per-window instance normalisations, ON
    # upstream and OFF here. Both exist to strip a window's absolute level as nuisance and
    # keep its shape as signal. In a threshold-anchored ENU frame that is backwards:
    # absolute position IS the signal — it determines where the turn onto final happens,
    # when the descent starts, and where the approach ends. Measured cost of leaving them
    # on (synthetic KRDU, window mode): iTransformer ADE 288 -> 743 m, PatchTST 698 -> 910 m.
    # Re-ablate on real data with --instance-norm; see README "Instance normalisation".
    use_norm: bool = False
    output_attention: bool = False
    # Read by the vendored code but inert on this path — kept so the object stays a drop-in
    # for upstream's run.py. See vendor/itransformer/PROVENANCE.md "Config contract".
    embed: str = "timeF" # unused, ignored by iTransformer
    freq: str = "h"
    factor: int = 1
    class_strategy: str = "projection"

    # ── PatchTST only ───────────────────────────────────────────────────────
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"
    revin: bool = False             # per-window instance norm (RevIN); see use_norm above
    affine: bool = False
    subtract_last: bool = False
    decomposition: bool = False
    kernel_size: int = 25
    individual: bool = False
    fc_dropout: float = 0.1
    head_dropout: float = 0.0

    # ── training ────────────────────────────────────────────────────────────
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    patience: int = 8               # early-stopping patience, in epochs without val improvement
    seed: int = 1337
    device: str = "auto"            # "auto" -> cuda when available, else cpu
    val_fraction: float = 0.15      # split is BY FLIGHT, never by window — see dataset.py
    test_fraction: float = 0.15
    sampling_strategy: str = SAMPLING_ALL_WINDOWS
    train_samples_per_epoch: int | None = None
    # Full prediction always starts at the earliest anchor. Pooled validation can mirror
    # that contract instead of materialising every highly-overlapping sliding anchor.
    eval_anchor_policy: str = "all"
    # Inferred final-approach geometry is weaker supervision than an observed ADS-B row.
    # These weights apply to POSITION channels only; fitted velocity channels are always
    # masked.  The terminal weight is added on the fitted crossing row so the endpoint is
    # not diluted by the rest of the short extrapolated tail.
    fitted_tail_position_weight: float = 0.25
    fitted_terminal_position_weight: float = 1.0
    final_time_loss_weight: float = 1.0
    final_time_scale_s: float = DEFAULT_FINAL_TIME_SCALE_S

    # ── provenance (free-form; recorded in the checkpoint) ──────────────────
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; expected one of {MODELS}")
        if self.coordinate_frame not in COORDINATE_FRAMES:
            raise ValueError(
                f"unknown coordinate_frame {self.coordinate_frame!r}; "
                f"expected one of {COORDINATE_FRAMES}"
            )
        if self.sampling_strategy not in SAMPLING_STRATEGIES:
            raise ValueError(
                f"unknown sampling_strategy {self.sampling_strategy!r}; "
                f"expected one of {SAMPLING_STRATEGIES}"
            )
        if self.eval_anchor_policy not in EVAL_ANCHOR_POLICIES:
            raise ValueError(
                f"unknown eval_anchor_policy {self.eval_anchor_policy!r}; "
                f"expected one of {EVAL_ANCHOR_POLICIES}"
            )
        for name in (
            "seq_len",
            "n_segments",
            "d_model",
            "n_heads",
            "d_ff",
            "e_layers",
            "patch_len",
            "stride",
            "kernel_size",
            "batch_size",
            "epochs",
            "patience",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        for name in ("dt_s", "learning_rate", "final_time_scale_s"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} must divide evenly by n_heads={self.n_heads}"
            )
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                f"val_fraction + test_fraction must leave a training split "
                f"(got {self.val_fraction} + {self.test_fraction})"
            )
        if self.fitted_tail_position_weight < 0.0:
            raise ValueError("fitted_tail_position_weight must be non-negative")
        if self.fitted_terminal_position_weight < 0.0:
            raise ValueError("fitted_terminal_position_weight must be non-negative")
        if self.final_time_loss_weight < 0.0:
            raise ValueError("final_time_loss_weight must be non-negative")
        if self.train_samples_per_epoch is not None and self.train_samples_per_epoch <= 0:
            raise ValueError("train_samples_per_epoch must be positive when supplied")

    # PatchTST reads configs.enc_in; iTransformer infers the count from the tensor.
    @property
    def enc_in(self) -> int:
        return len(self.channels)

    @property
    def pred_len(self) -> int:
        """Vendored model name for the normalized output length ``N``."""
        return self.n_segments

    @property
    def lookback_s(self) -> float:
        """Wall-clock seconds of observed track the model is shown."""
        return self.seq_len * self.dt_s

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["channels"] = list(self.channels)  # JSON has no tuple
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TSConfig:
        data = dict(data)
        data["channels"] = tuple(data["channels"])
        return cls(**data)
