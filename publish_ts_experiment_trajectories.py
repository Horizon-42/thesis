#!/usr/bin/env python
"""Publish train/validation (and a day partition's held-out-day) trajectories for indexed TS checkpoints.

The script is intentionally an orchestration layer.  It does not define another trajectory
or evaluation format; every job calls the existing predictor, the shared ``evaluation`` package,
and the existing comparison-CZML publisher in that order.  Publications are resumable and kept
under a checkpoint-specific raw-output directory.  ``--reuse-prediction-dir`` publishes the
records a campaign already wrote (its extra predict flags — latent sampling, a given CTA,
a quantile-decoded CTA — are not reconstructable from the checkpoint alone) instead of predicting again;
that directory is read-only here. ``--result-source prediction`` publishes a
primary result under Prediction; the default ``experiment`` source includes checkpoint metadata
for the Experiments picker.

**One checkpoint, several prediction directories.** Everything a category is named from —
the key, the picker's experiment id, the label — is derived from the CHECKPOINT, so two
directories of one checkpoint collide unless something tells them apart. That something is a
``CategoryVariant``, and it has exactly two sources: a directory written by
``run_ts.py anytime_curve --write-records`` carries an ``anytime`` block naming the
remaining-path bin its forecasts were anchored in (its label then states the bin and how many
flights it holds, so a re-anchored SUBSET's error can never be read as the split's, and its
picker heading is the record campaign); anything else — a predict-time projection or command
hook published beside the baseline whose checkpoint it reuses — names one with
``--category-variant SLUG[=LABEL]``. Giving both is refused, and so is publishing a second
directory onto a category some other directory already holds: they must coexist, never
overwrite.

**Every experiment states its intent.** The picker shows what a campaign asks and what each
run changes, read from the tracked registry ``INTENT_REGISTRY`` (``campaigns[<picker group>]``
title + intent, ``campaigns[<training campaign>].runs[<run id>]``, optionally
``variants[<run>@<variant>]``) and stamped into the category's ``experiment.intent`` beside the
structured ``experiment.parameters`` rows (``run_naming.run_parameter_rows``). A publication
whose group or run has no entry is BLOCKED before any work, and ``--refresh-labels-only``
refuses to write anything while a listed category lacks one — write the entries when the
campaign is designed, with its arm declaration.

**Held-out days (``dayval``).** A later campaign's runner may fly an earlier checkpoint over the
validation days of a day partition it was trained beside (runway-intent R3's schedule, flown by R2b's
day_a experts). No predict step writes that split, so it is published only from the runner's own
directory (``--reuse-prediction-dir``), only under Experiments, and every record's flight is checked
before anything is written: not in the checkpoint's locked outer-test hash, not among its own
training / validation flights. ``--category-group``
files such a variant under the campaign that WROTE the records (its registry entry gives the heading
and question) rather than the one that trained the checkpoint.

Outer-test is not a valid option here.  This command is for development train/validation
inspection (and a day partition's held-out days) only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent
_TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(_TS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_TS_DIR.parent))

from ts_transformer.run_naming import (  # noqa: E402
    SPLIT_DAYVAL,
    category_display_label,
    run_display_name,
    run_parameter_rows,
)

EXPERIMENT_ROOT = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiments"
EXPERIMENT_INDEX = EXPERIMENT_ROOT / "index.json"
RAW_OUTPUT_ROOT = (
    REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiment_predictions"
)
HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
FRONTEND_AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
TS_SCRIPT = REPO_ROOT / "4dTrajectory" / "ts_transformer" / "__main__.py"
CZML_SCRIPT = REPO_ROOT / "aeroviz-4d" / "python" / "build_scenario_comparison_czml.py"

#: Stamped into every manifest written from here on. v2 means "a category may be one of
#: several publications of its checkpoint" — a v1 reader would take a variant's category and
#: write the BASELINE's experiment id onto it, so it must skip these rather than refresh them.
PUBLICATION_SCHEMA = "ts-experiment-publication-v2"
#: What this reader accepts: v1 manifests predate the variant and describe a lone publication.
PUBLICATION_SCHEMAS_READ = ("ts-experiment-publication-v1", PUBLICATION_SCHEMA)
PUBLICATION_INDEX_SCHEMA = "ts-experiment-publication-index-v1"
PUBLICATION_MANIFEST = "publication.json"
DEVELOPMENT_SPLITS = ("train", "val")
#: Record splits a REUSED prediction directory may carry beyond the development ones — no predict step
#: writes them: `SPLIT_DAYVAL`, the validation days of a day partition flown by a checkpoint trained on
#: that partition's training days (runway-intent R3's schedule, `runway_intent_r3 --write-records`).
REUSE_ONLY_SPLITS = (SPLIT_DAYVAL,)
PUBLISHABLE_SPLITS = DEVELOPMENT_SPLITS + REUSE_ONLY_SPLITS

#: What each campaign asks and what each run changes — tracked source, written when a campaign
#: is designed and read (never written) here. Read at CALL time, so a test can point it elsewhere.
INTENT_REGISTRY = _TS_DIR / "docs" / "experiments" / "intents.json"
INTENT_REGISTRY_SCHEMA = "ts-experiment-intents-v1"

#: A MIRROR of ``experiments.anytime_curve.RECORDS_SCHEMA`` — the block that runner writes into
#: each record directory's ``summary.json``. Not imported: that module pulls in torch and the
#: whole ts package, and this script is a subprocess orchestrator that must stay importable
#: without them. Change the two together.
ANYTIME_RECORDS_SCHEMA = "ts-anytime-records-v1"
#: The summary blocks the REPLAY runners write beside a checkpoint's own forecasts — MIRRORS of
#: ``experiments.chain_sensitivity`` (``chain``: the one-shot and the chained re-ask of one
#: checkpoint) and the archived two-tier lockstep's ``lockstep`` block (its published
#: categories keep it readable; `archive/two_tier_v2_2026_09/tracker_lockstep.py`), for the same
#: reason the anytime schema above is one. The manoeuvre-token readouts register their own
#: blocks here when they publish (plan §4.2). A directory carrying one is a VARIANT of that
#: checkpoint's prediction and has no category of its own without ``--category-variant``:
#: published bare, it would be filed as the checkpoint's L−1 prediction.
VARIANT_RECORD_BLOCKS = ("chain", "lockstep")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@lru_cache(maxsize=None)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(value: str, *, limit: int = 56) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return (normalized or "experiment")[:limit]


def _path_for_manifest(path: Path) -> str:
    """A repository-relative path when possible, otherwise an absolute one.

    Tried BEFORE resolving as well as after: in a worktree, `4dTrajectory/outputs` is a
    symlink into the main tree, so resolving first turns a path that is lexically inside the
    repository into an absolute one outside it, and the manifest stops being portable.
    """
    for candidate, root in ((path, REPO_ROOT), (path.resolve(), REPO_ROOT.resolve())):
        try:
            return str(candidate.relative_to(root))
        except ValueError:
            continue
    return str(path.resolve())


def _same_prediction_dir(claimed: str | None, ours: Path | None) -> bool:
    """Do a manifest's stored prediction directory and this plan's name the same place?

    Compared as PATHS, never as strings. Every manifest written before this fix stored an
    absolute path (a worktree resolves `4dTrajectory/outputs` into the main tree, so the
    relative attempt failed), while the same publication run from the main tree now renders
    it relative — string equality would call a byte-identical republish a collision. So a
    relative value is resolved against the repository and both sides are resolved through
    their symlinks; `samefile` decides whenever both exist, since two different paths can be
    one directory.
    """
    if claimed is None or ours is None:
        return claimed is None and ours is None
    theirs = Path(claimed)
    if not theirs.is_absolute():
        theirs = REPO_ROOT / theirs
    if theirs.exists() and ours.exists():
        return theirs.samefile(ours)
    return theirs.resolve() == ours.resolve()


@dataclass(frozen=True)
class CategoryVariant:
    """What tells two publications of the SAME checkpoint apart.

    A checkpoint is published more than once whenever the prediction directory — not the
    training — is what differs: one directory per remaining-path bin
    (``run_ts.py anytime_curve --write-records``), or one per predict-time variant (an
    inference projection, a command hook applied at predict time only). Every part of a
    category's identity is derived from the CHECKPOINT — the key, the picker's experiment id,
    the label — so without something to tell those publications apart the second silently
    overwrites the first. This is that something, and a publication has at most one.

    ``id_suffix`` is separate from ``key_suffix`` because the two live in different alphabets:
    a category key is a filesystem directory name, an experiment id is a display/lookup
    identity that may carry ``@``.
    """

    #: Appended to the category key (hence to the frontend directory name).
    key_suffix: str
    #: Appended to ``experiment.id`` after ``@``. The picker DEDUPES by that id.
    id_suffix: str
    #: Appended to the canonical run display name, in the picker and the category label.
    label_suffix: str
    #: Picker heading override, or None to keep the checkpoint's own training campaign.
    group: str | None = None


#: A ``--category-variant`` slug: a directory-name fragment that must not open with a digit
#: (the category key reads as `<source>_<run>_<token>_<slug>_<split>`) and must not contain a
#: path separator. Lower-cased and length-capped like the rest of the key (`_safe_stem`), so
#: what is typed and what appears on disk cannot differ by case alone.
class MissingIntentError(LookupError):
    """An experiment publication whose campaign or run has no entry in the intent registry."""


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_intent_campaigns(path: Path) -> dict[str, dict[str, Any]]:
    """The registry's campaigns, validated once here: ``title`` and ``intent`` required,
    ``design`` optional, ``runs`` / ``variants`` optional maps of non-empty strings (read as
    empty when absent — a record campaign that trained nothing has no runs)."""
    document = _load_object(path)
    if document.get("schemaVersion") != INTENT_REGISTRY_SCHEMA:
        raise ValueError(
            f"{path}: schemaVersion {document.get('schemaVersion')!r} is not "
            f"{INTENT_REGISTRY_SCHEMA!r}"
        )
    campaigns = document.get("campaigns")
    if not isinstance(campaigns, dict):
        raise ValueError(f"{path}: `campaigns` must be an object")
    validated: dict[str, dict[str, Any]] = {}
    for key, entry in campaigns.items():
        where = f"{path}: campaigns[{key!r}]"
        if not isinstance(entry, dict) or not _text(entry.get("title")) or not _text(
            entry.get("intent")
        ):
            raise ValueError(f"{where} needs a non-empty `title` and `intent`")
        if "design" in entry and not _text(entry["design"]):
            raise ValueError(f"{where}.design must be a non-empty string when present")
        maps: dict[str, dict[str, str]] = {}
        for name in ("runs", "variants"):
            value = entry.get(name, {})
            if not isinstance(value, dict) or not all(
                isinstance(k, str) and _text(v) for k, v in value.items()
            ):
                raise ValueError(f"{where}.{name} must map ids to non-empty strings")
            maps[name] = value
        validated[key] = {**entry, **maps}
    return validated


def experiment_intent(
    *,
    group: str,
    training_campaign: str,
    run_id: str,
    variant: CategoryVariant | None,
) -> dict[str, str]:
    """The picker's intent block: the heading campaign's title + question, the run's intent,
    and — when the registry has one — the variant's.

    ``group`` is the picker heading (an anytime bin's RECORD campaign), ``training_campaign``
    the campaign the checkpoint was trained in: a run's intent belongs to where it was designed.
    """
    campaigns = load_intent_campaigns(INTENT_REGISTRY)
    heading = campaigns.get(group)
    trained = campaigns.get(training_campaign)
    run_intent = None if trained is None else trained["runs"].get(run_id)
    missing = []
    if heading is None:
        missing.append(f"campaigns[{group!r}] (title + intent)")
    if run_intent is None:
        missing.append(f"campaigns[{training_campaign!r}].runs[{run_id!r}]")
    if missing:
        raise MissingIntentError(
            f"{_path_for_manifest(INTENT_REGISTRY)} has no " + " and no ".join(missing)
            + " — state what the campaign asks and what this run changes before publishing it"
        )
    intent = {"groupTitle": heading["title"], "group": heading["intent"], "run": run_intent}
    if "design" in heading:
        intent["design"] = heading["design"]
    if variant is not None:
        variant_intent = trained["variants"].get(f"{run_id}@{variant.id_suffix}")
        if variant_intent is not None:
            intent["variant"] = variant_intent
    return intent


_VARIANT_SLUG = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
#: The anytime bins' own key shape (`a12km`). A flag slug must not be able to impersonate one:
#: the two would be indistinguishable in a key while meaning different things.
_ANYTIME_KEY_SHAPE = re.compile(r"^a\d+(\.\d+)?km$")
#: `references/` is the record contract's own subdirectory inside a prediction batch.
_RESERVED_VARIANT_SLUGS = frozenset({"references"})


def _parse_category_variant(value: str) -> CategoryVariant:
    """``SLUG`` or ``SLUG=LABEL`` → the variant every derived name is suffixed with."""
    slug, separator, label = value.partition("=")
    slug, label = slug.strip(), label.strip()
    if not _VARIANT_SLUG.match(slug):
        raise ValueError(
            f"--category-variant slug {slug!r} must start with a letter and contain only "
            "letters, digits, '_' and '-' — it becomes part of a category key"
        )
    normalized = _safe_stem(slug, limit=24).lower()
    if normalized in _RESERVED_VARIANT_SLUGS:
        raise ValueError(
            f"--category-variant slug {slug!r} is reserved by the record contract"
        )
    if _ANYTIME_KEY_SHAPE.match(normalized):
        raise ValueError(
            f"--category-variant slug {slug!r} has the shape an anytime bin's key takes "
            "(a12km); pick one that cannot be mistaken for a re-anchored bin"
        )
    if separator and not label:
        raise ValueError(f"--category-variant {value!r} has an empty label after '='")
    slug = normalized
    return CategoryVariant(
        key_suffix=slug, id_suffix=slug, label_suffix=label or slug,
    )


@dataclass(frozen=True)
class AnytimeBin:
    """One re-anchored bin of an anytime record campaign: what a category must SAY about it.

    Read off the reused prediction directory's own ``summary.json``; absent for every other
    prediction directory, and a publication is then exactly what it was before. The bin is
    the single source of the category's identity — key suffix, picker group and label — so
    nothing downstream has to parse a directory name to learn which anchor it is looking at.
    """

    campaign: str
    #: The runner's own spelling of the bin (``12km``) — one spelling for the directory it
    #: wrote, the category key and the label, so none of the three can drift.
    bin_label: str
    records: int
    measured_flights: int
    split_flights: int
    limit: int

    @property
    def variant(self) -> CategoryVariant:
        """The bin, as the one thing that tells its publication from the checkpoint's others.

        ``a12km`` for the key — the ``a`` keeps the fragment from opening with a digit — and
        the bare ``12km`` for the experiment id, which has no such constraint. The group is
        the RECORD campaign, so a campaign's bins sit under one picker heading instead of
        scattering across the training campaigns they were replayed from.
        """
        return CategoryVariant(
            key_suffix=f"a{self.bin_label}",
            id_suffix=self.bin_label,
            label_suffix=self.label_suffix,
            group=self.campaign,
        )

    @property
    def label_suffix(self) -> str:
        """The bin AND its population: a subset's numbers must never read as the split's.

        Both denominators, because they answer different questions: ``records`` of
        ``measured_flights`` is how much of the replayed cohort reached this bin (the rest
        never flew a sample near it), and ``--limit`` of ``split_flights`` is how much of the
        split was replayed at all.
        """
        counts = f"{self.records} of {self.measured_flights} flights"
        if self.limit:
            counts += f", --limit {self.limit} of {self.split_flights} in the split"
        return f"@ {self.bin_label} remaining ({counts})"


def _anytime_from_block(block: dict[str, Any]) -> AnytimeBin:
    """The fields this publisher uses, named once.

    Both sources are read through here: the record directory's summary block (which carries
    more than a category needs — arm, anchor rule, coverage) and the publication manifest's
    own copy. Selecting rather than splatting is what lets the two shapes differ, and what
    keeps a field this publisher stops using from breaking a refresh of an older manifest.
    """
    return AnytimeBin(
        campaign=str(block["campaign"]),
        bin_label=str(block["bin_label"]),
        records=int(block["records"]),
        measured_flights=int(block["measured_flights"]),
        split_flights=int(block["split_flights"]),
        limit=int(block["limit"]),
    )


def _anytime_bin(summary_path: Path) -> AnytimeBin | None:
    block = _load_object(summary_path).get("anytime")
    if block is None:
        return None
    if block.get("schema") != ANYTIME_RECORDS_SCHEMA:
        raise ValueError(
            f"{summary_path} carries an anytime block of unknown schema "
            f"{block.get('schema')!r} (this publisher speaks {ANYTIME_RECORDS_SCHEMA})"
        )
    return _anytime_from_block(block)


@dataclass(frozen=True)
class ExperimentCheckpoint:
    experiment_id: str
    campaign: str
    run_id: str
    checkpoint: Path
    checkpoint_sha256: str
    arrival_manifests: dict[str, str]
    #: airport -> eligible-set digest. Empty for metadata written before 2026-09-08, whose
    #: only eligibility identity was the roster FILE's digest; those are verified through
    #: the checkpoint's own provenance instead (`preflight_error`).
    eligible_sets: dict[str, str]
    #: airports this run applied a pre-split roster to — which is what decides whether
    #: `predict` is handed one, in either metadata generation.
    eligibility_airports: tuple[str, ...]
    config: dict[str, Any]

    @property
    def token(self) -> str:
        return hashlib.sha256(self.experiment_id.encode("utf-8")).hexdigest()[:12]

    @property
    def directory_name(self) -> str:
        return f"{_safe_stem(self.run_id)}_{self.token}"

    @property
    def checkpoint_relative(self) -> str:
        return _path_for_manifest(self.checkpoint)


def _checkpoint_config(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = _load_object(manifest_path)
        config = manifest.get("config")
        if isinstance(config, dict):
            return config
    history_path = directory / "history.json"
    if history_path.is_file():
        history = _load_object(history_path)
        config = history.get("config")
        if isinstance(config, dict):
            return config
    return {}


def discover_checkpoints(
    index_path: Path = EXPERIMENT_INDEX,
    *,
    selected_ids: set[str] | None = None,
    campaigns: set[str] | None = None,
) -> list[ExperimentCheckpoint]:
    """Return completed, indexed training checkpoints in stable path order."""
    document = _load_object(index_path)
    root = Path(document.get("root") or index_path.parent).resolve()
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{index_path} has no entries list")

    checkpoints: list[ExperimentCheckpoint] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "training" or entry.get("status") != "completed":
            continue
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, list) or "checkpoint.pt" not in artifacts:
            continue
        experiment_id = str(entry.get("path") or "")
        if not experiment_id:
            continue
        if selected_ids is not None and experiment_id not in selected_ids:
            continue
        campaign = str(entry.get("campaign_id") or Path(experiment_id).parts[0])
        if campaigns is not None and campaign not in campaigns:
            continue

        directory = root / experiment_id
        checkpoint = directory / "checkpoint.pt"
        metadata_path = directory / "checkpoint_metadata.json"
        if not checkpoint.is_file() or not metadata_path.is_file():
            continue
        metadata = _load_object(metadata_path)
        manifests = metadata.get("arrival_manifests")
        if not isinstance(manifests, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in manifests.items()
        ):
            raise ValueError(f"{metadata_path} has no valid arrival_manifests map")
        eligible_sets = metadata.get("eligible_sets")
        legacy_rosters = metadata.get("eligibility_rosters")
        rosters = eligible_sets if eligible_sets is not None else legacy_rosters
        if rosters is None:
            rosters = {}
        if not isinstance(rosters, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in rosters.items()
        ):
            key = "eligible_sets" if eligible_sets is not None else "eligibility_rosters"
            raise ValueError(f"{metadata_path} has no valid {key} map")
        declared_sha = metadata.get("checkpoint_sha256")
        checkpoint_sha = declared_sha if isinstance(declared_sha, str) else _sha256(checkpoint)
        checkpoints.append(ExperimentCheckpoint(
            experiment_id=experiment_id,
            campaign=campaign,
            run_id=str(entry.get("run_id") or directory.name),
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            arrival_manifests=dict(sorted(manifests.items())),
            eligible_sets=dict(sorted((eligible_sets or {}).items())),
            eligibility_airports=tuple(sorted(rosters)),
            config=_checkpoint_config(directory),
        ))
    return checkpoints


@dataclass(frozen=True)
class PublicationPlan:
    experiment: ExperimentCheckpoint
    airport: str
    split: str
    result_source: str = "experiment"
    raw_output_root: Path = RAW_OUTPUT_ROOT
    harvest_root: Path = HARVEST_ROOT
    frontend_airports_root: Path = FRONTEND_AIRPORTS_ROOT
    device: str = "auto"
    record_retention: str = "archive"
    prediction_dir: Path | None = None
    #: ``--category-variant``. The other source is the reused directory's own anytime block;
    #: a publication takes at most one of the two (see ``category_variant``).
    variant: CategoryVariant | None = None

    def __post_init__(self) -> None:
        if self.split not in PUBLISHABLE_SPLITS:
            raise ValueError(
                f"experiment publication accepts development splits {DEVELOPMENT_SPLITS} "
                f"(and {REUSE_ONLY_SPLITS} from a reused directory), got {self.split!r}"
            )
        if self.split in REUSE_ONLY_SPLITS and self.prediction_dir is None:
            raise ValueError(
                f"the {self.split!r} split needs --reuse-prediction-dir: no predict step writes it "
                "(its records come from the runner that flew them)"
            )
        if self.split in REUSE_ONLY_SPLITS and self.result_source != "experiment":
            raise ValueError(
                f"the {self.split!r} split is an experiment's records: it is published under "
                "Experiments (--result-source experiment), never as a primary Prediction"
            )
        if self.result_source not in {"prediction", "experiment"}:
            raise ValueError(f"unknown result source {self.result_source!r}")
        if self.record_retention not in {"loose", "archive"}:
            raise ValueError(f"unknown record retention {self.record_retention!r}")
        if self.prediction_dir is not None and not (self.prediction_dir / "summary.json").is_file():
            raise ValueError(
                f"reused prediction directory {self.prediction_dir} has no summary.json"
            )
        if self.prediction_dir is not None and self.variant is None:
            carried = [
                name for name in VARIANT_RECORD_BLOCKS
                if name in _load_object(self.prediction_dir / "summary.json")
            ]
            if carried:
                raise ValueError(
                    f"{self.prediction_dir} carries a {carried[0]!r} block: its records were re-flown "
                    "by a replay runner from the checkpoint, not predicted by it — give "
                    "--category-variant, or they would be filed as the checkpoint's own prediction"
                )
        if self.variant is not None and self.prediction_dir is None:
            # Without a reused directory the predict step would run again and differ only in
            # --output-dir: the same checkpoint, the same flags, the same forecasts, published
            # a second time under a relabelled key. A variant distinguishes DIRECTORIES.
            raise ValueError(
                f"--category-variant {self.variant.key_suffix!r} needs "
                "--reuse-prediction-dir: without one it would republish the checkpoint's own "
                "prediction under a second name, not a different prediction"
            )
        if self.variant is not None and self.anytime is not None:
            # Checked HERE, at construction, so a batch cannot die halfway through with a
            # traceback: two answers to "which publication is this" is the ambiguity the
            # variant exists to remove.
            raise ValueError(
                f"{self.prediction_dir} describes its own anytime bin "
                f"({self.anytime.bin_label}) and --category-variant "
                f"{self.variant.key_suffix!r} names another; a publication has one identity"
            )

    @cached_property
    def anytime(self) -> AnytimeBin | None:
        """The re-anchored bin these records are, or None for an ordinary L−1 batch.

        Reads the reused directory's summary DIRECTLY rather than through ``self.summary``:
        that property goes via ``records_dir`` → ``output_dir``, and ``output_dir`` now reads
        this one. Today the cycle is broken only by ``or`` short-circuiting; spelling the
        path here means it cannot come back as a ``RecursionError``.
        """
        if self.prediction_dir is None:
            return None
        return _anytime_bin(self.prediction_dir / "summary.json")

    @cached_property
    def category_variant(self) -> CategoryVariant | None:
        """The ONE thing that tells this publication from the checkpoint's others.

        Two sources can supply it and a publication takes at most one: a record directory
        that describes its own bin, and ``--category-variant`` on the command line. Both at
        once is refused at construction (``__post_init__``), so by the time anything reads
        this there is one answer.
        """
        return self.variant if self.anytime is None else self.anytime.variant

    @property
    def data_manifest(self) -> Path:
        return self.harvest_root / self.airport / "arrivals" / "manifest.json"

    @property
    def eligibility_roster(self) -> Path:
        return self.data_manifest.parent / "lateral_pass_eligibility.json"

    @property
    def output_dir(self) -> Path:
        base = (
            self.raw_output_root
            / self.experiment.directory_name
            / self.result_source
            / self.airport
            / self.split
        )
        # One checkpoint publishes once per VARIANT, so each owns its evaluation report and
        # publication manifest — sharing the split's directory would have them overwrite each
        # other's verdicts.
        variant = self.category_variant
        return base if variant is None else base / variant.key_suffix

    @property
    def records_dir(self) -> Path:
        """Where the per-flight records and their summary.json live.

        The predict step writes them into ``output_dir``; ``--reuse-prediction-dir`` instead
        points at a directory an experiment campaign already produced (which this command then
        only ever reads — it is not ours to archive or delete).
        """
        return self.prediction_dir or self.output_dir

    @property
    def summary(self) -> Path:
        return self.records_dir / "summary.json"

    @property
    def evaluation_report(self) -> Path:
        return self.output_dir / "evaluation_report.json"

    @property
    def publication_manifest(self) -> Path:
        return self.output_dir / PUBLICATION_MANIFEST

    @property
    def records_archive(self) -> Path:
        return self.output_dir / "prediction_records.tar.gz"

    @property
    def category(self) -> str:
        run = _safe_stem(self.experiment.run_id, limit=42).lower()
        variant = self.category_variant
        suffix = "" if variant is None else f"_{variant.key_suffix}"
        return f"{self.result_source}_{run}_{self.experiment.token}{suffix}_{self.split}"

    @property
    def experiment_group(self) -> str | None:
        return _experiment_group(self.experiment.campaign, self.category_variant)

    @property
    def comparison_dir(self) -> Path:
        return self.frontend_airports_root / self.airport / "comparison" / self.category

    @property
    def comparison_index(self) -> Path:
        return self.comparison_dir / "comparison_index.json"

    @property
    def category_label(self) -> str:
        return _publication_label(
            self.split, self.result_source, self.experiment.config, self.experiment.run_id,
            variant=self.category_variant,
        )

    @property
    def picker_id(self) -> str:
        return _picker_id(self.experiment.experiment_id, self.category_variant)

    @property
    def experiment_intent(self) -> dict[str, str]:
        """Raises ``MissingIntentError`` when the registry has no entry for this publication."""
        return experiment_intent(
            group=self.experiment_group,
            training_campaign=self.experiment.campaign,
            run_id=self.experiment.run_id,
            variant=self.category_variant,
        )

    @property
    def experiment_metadata(self) -> dict[str, Any]:
        return _publication_experiment_metadata(
            experiment_id=self.experiment.experiment_id,
            campaign=self.experiment_group,
            training_campaign=self.experiment.campaign,
            checkpoint=self.experiment.checkpoint_relative,
            config=self.experiment.config,
            run_id=self.experiment.run_id,
            variant=self.category_variant,
        )

    def commands(self) -> list[tuple[str, list[str]]]:
        py = sys.executable
        predict = [
            py, str(TS_SCRIPT), "predict",
            "--checkpoint", str(self.experiment.checkpoint),
            "--data", str(self.data_manifest),
        ]
        if self.airport in self.experiment.eligibility_airports:
            predict += ["--eligibility-roster", str(self.eligibility_roster)]
        predict += [
            "--output-dir", str(self.output_dir),
            "--split", self.split,
            "--device", self.device,
        ]
        publish = [
            py, str(CZML_SCRIPT),
            "--summary", str(self.summary),
            "--output-dir", str(self.comparison_dir),
            "--airport", self.airport,
            "--category", self.category,
            "--category-label", self.category_label,
            "--dataset-split", self.split,
            "--evaluation-report", str(self.evaluation_report),
            "--result-source", self.result_source,
        ]
        if self.result_source == "experiment":
            publish += [
                "--experiment-id", self.picker_id,
                "--experiment-group", self.experiment_group,
                "--experiment-checkpoint", self.experiment.checkpoint_relative,
            ]
        steps = [
            ("evaluate", [
                py, "-m", "evaluation",
                "--input", str(self.records_dir),
                "--output", str(self.evaluation_report),
            ]),
            ("publish-czml", publish),
        ]
        if self.prediction_dir is None:
            steps.insert(0, ("predict", predict))
        return steps

    def _held_out_error(self, rows: Iterable[dict[str, Any]]) -> str | None:
        """Refuse records that are not held out from this checkpoint: a flight of its locked
        outer-test split, or one of its OWN training / validation flights.

        A development split's records come from `predict`, which reads only that split; a
        reuse-only split's come from a runner, so both halves of what the label promises are
        CHECKED here — the package's per-flight hash on the checkpoint's split contract, and the
        split the checkpoint persisted — not taken on the runner's word.
        """
        from flight_scenarios.identity import flight_key   # the package's rules, imported where they are needed
        from ts_transformer.config import TSConfig
        from ts_transformer.data.splits import split_name_for_dataset_id
        from ts_transformer.training.train import load_checkpoint_payload
        try:
            config = TSConfig.from_dict(self.experiment.config)
        except ValueError as exc:
            return f"cannot read the checkpoint's locked split to check the outer-test seal: {exc}"
        dataset_ids = [f"{self.airport}:{flight_key(row, index)}" for index, row in enumerate(rows)]
        sealed = [key for key in dataset_ids if split_name_for_dataset_id(key, config) == "test"]
        if sealed:
            return (
                f"reused predictions in {self.prediction_dir} hold {len(sealed)} flight(s) of the "
                f"locked outer-test split (first {sealed[0]}); outer-test is never published"
            )
        own = load_checkpoint_payload(self.experiment.checkpoint)["split"]
        seen = [key for key in dataset_ids if key in set(own["train"]) | set(own["val"])]
        if seen:
            return (
                f"reused predictions in {self.prediction_dir} hold {len(seen)} flight(s) the checkpoint "
                f"trained or selected on (first {seen[0]}); a {self.split!r} publication shows only "
                "flights it never saw"
            )
        return None

    def _eligibility_error(self) -> str | None:
        """Refuse a roster that no longer selects the flights the checkpoint was trained on.

        The identity is the eligible SET — the package's own `eligible_set_digest`, not a
        second comparison written here. This preflight held its own byte comparison until
        2026-09-08, and blocked every publication the morning the observed evaluation was
        regenerated (v6 -> v9) over eligible sets that had not changed at all.
        """
        if self.airport not in self.experiment.eligibility_airports:
            return None
        if not self.eligibility_roster.is_file():
            return f"missing eligibility roster {self.eligibility_roster}"
        from ts_transformer.data.data_provenance import (   # the package's rule, imported where it is needed
            checkpoint_data_provenance,
            require_matching_data_provenance,
            roster_eligible_set_digest,
        )
        expected = self.experiment.eligible_sets.get(self.airport)
        if expected is not None:
            actual = roster_eligible_set_digest(self.eligibility_roster)
            if actual != expected:
                return (
                    f"eligible set changed for {self.airport}: "
                    f"checkpoint={expected}, current={actual}"
                )
            return None
        # Metadata that predates `eligible_sets` names only the roster's bytes, so the
        # answer comes from the checkpoint payload — one path, torch load included.
        from ts_transformer.training.train import load_checkpoint_payload
        try:
            payload = load_checkpoint_payload(self.experiment.checkpoint)
            require_matching_data_provenance(
                payload,
                checkpoint_data_provenance(payload, [self.data_manifest]),
                allow_subset=True,
            )
        except (OSError, ValueError, KeyError) as exc:
            return f"checkpoint predates the eligible-set identity and {exc}"
        return None

    def preflight_error(self) -> str | None:
        if not self.experiment.checkpoint.is_file():
            return f"missing checkpoint {self.experiment.checkpoint}"
        if _sha256(self.experiment.checkpoint) != self.experiment.checkpoint_sha256:
            return "checkpoint SHA-256 differs from checkpoint_metadata.json"
        if not self.data_manifest.is_file():
            return f"missing arrival manifest {self.data_manifest}"
        expected = self.experiment.arrival_manifests.get(self.airport)
        if expected is None:
            return f"checkpoint provenance does not include airport {self.airport}"
        actual = _sha256(self.data_manifest)
        if actual != expected:
            return (
                f"arrival manifest SHA-256 mismatch for {self.airport}: "
                f"checkpoint={expected}, current={actual}"
            )
        eligibility_error = self._eligibility_error()
        if eligibility_error is not None:
            return eligibility_error
        if self.prediction_dir is not None:
            reused = _load_object(self.summary)
            produced_by = Path(str(reused.get("checkpoint") or ""))
            if produced_by.resolve() != self.experiment.checkpoint.resolve():
                return (
                    f"reused predictions in {self.prediction_dir} were produced by "
                    f"{produced_by}, not {self.experiment.checkpoint}"
                )
            if reused.get("split") != self.split:
                return (
                    f"reused predictions in {self.prediction_dir} are the "
                    f"{reused.get('split')!r} split, not {self.split!r}"
                )
            # ANY reused directory, not only an anytime one: the publisher fans out a plan
            # per airport of the checkpoint's provenance, so a directory holding several
            # airports' flights would be filed whole under each airport's category with the
            # pooled count printed as that airport's. (The anytime runner is simply the one
            # that always produces such a directory — it replays the cohort its provenance
            # names and offers no airport narrowing.) Refuse rather than mislabel.
            airports = {
                str(row.get("arr_airport") or "").upper()
                for row in reused.get("results") or ()
            }
            if airports and airports != {self.airport}:
                return (
                    f"reused predictions in {self.prediction_dir} cover {sorted(airports)}, "
                    f"not just {self.airport}: this publication is one airport's category, "
                    "and a directory spanning several would be filed whole under each"
                )
            if self.split in REUSE_ONLY_SPLITS:
                held_out_error = self._held_out_error(reused.get("results") or ())
                if held_out_error is not None:
                    return held_out_error
        # Two prediction directories must never land on ONE category. Everything a category
        # is named from comes from the checkpoint, so publishing a second directory of the
        # same checkpoint without a variant would silently replace the first — its CZML, its
        # evaluation report and its picker entry. Refuse and name the variant flag; the
        # variant is what makes them coexist.
        # Only a COMPLETED manifest holds a category: a failed or blocked attempt left its
        # document behind without owning anything, and treating it as an owner would wedge the
        # category permanently. The comparison is by PATH, not by string — see
        # `_same_prediction_dir`, without which a byte-identical republish from the other tree
        # reads as a collision.
        if self.publication_manifest.is_file():
            existing = _load_object(self.publication_manifest)
            claimed = existing.get("predictionDir")
            if (
                existing.get("status") == "completed"
                and not _same_prediction_dir(claimed, self.prediction_dir)
            ):
                ours = (
                    "a predict run" if self.prediction_dir is None
                    else _path_for_manifest(self.prediction_dir)
                )
                return (
                    f"category {self.category!r} was already published from "
                    f"{claimed or 'a predict run'} and this would republish it from {ours}; "
                    "give one of them a --category-variant so the two can coexist instead of "
                    "overwriting each other"
                )
        return None

    def is_complete(self) -> bool:
        if not self.publication_manifest.is_file():
            return False
        try:
            manifest = _load_object(self.publication_manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            manifest.get("status") == "completed" and
            manifest.get("checkpointSha256") == self.experiment.checkpoint_sha256 and
            manifest.get("resultSource") == self.result_source and
            self.summary.is_file() and
            self.evaluation_report.is_file() and
            self.comparison_index.is_file()
        )


def _manifest_is_completed(path: Path) -> bool:
    """Does a publication manifest on disk record a COMPLETED publication?

    Unreadable or absent counts as "no": the question is only ever asked to decide whether
    something worth keeping is already there.
    """
    if not path.is_file():
        return False
    try:
        return _load_object(path).get("status") == "completed"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _variant_from_document(document: dict[str, Any]) -> CategoryVariant | None:
    """The variant a completed publication was made under, straight from its manifest.

    It travels in the manifest so a label refresh never has to reopen the (read-only, possibly
    moved) prediction directory to recompute the same name. Manifests written before the
    variant became a first-class thing carry only the anytime block, which describes exactly
    one variant — so those are read through the bin.
    """
    block = document.get("anytime")
    if block:
        # DERIVED, never the stored rendering: `AnytimeBin.label_suffix` owns how a bin reads,
        # so a wording change there must reach every anytime publication on the next refresh.
        return _anytime_from_block(block).variant
    stored = document.get("variant")
    if not stored:
        return None
    group = stored.get("group")
    if group is not None and not isinstance(group, str):
        # The picker groups by this value; a non-string silently drops the whole category.
        raise ValueError(
            f"stored variant group must be a string or null, got {group!r}"
        )
    return CategoryVariant(
        key_suffix=str(stored["key_suffix"]),
        id_suffix=str(stored["id_suffix"]),
        label_suffix=str(stored["label_suffix"]),
        group=group,
    )


def _experiment_group(campaign: str | None, variant: CategoryVariant | None) -> str | None:
    """The heading the picker files a model under.

    A variant may override it: an anytime bin belongs to its RECORD campaign, not to the
    campaign its checkpoint was trained in, which is what puts one campaign's bins under one
    heading instead of scattering them across the training campaigns they were replayed from.
    """
    if variant is None or variant.group is None:
        return campaign
    return variant.group


def _run_label(
    config: dict[str, Any], run_id: str, variant: CategoryVariant | None
) -> str:
    """The canonical run name, plus which anchor these particular records were taken at.

    ONE definition, used by the category label and by the picker entry — a bin whose picker
    entry did not say "@ 12 km" would be indistinguishable from the L−1 publication of the
    same checkpoint.
    """
    name = run_display_name(config, extra=(run_id,))
    return name if variant is None else f"{name} {variant.label_suffix}"


def _publication_label(
    split: str, result_source: str, config: dict[str, Any], run_id: str,
    variant: CategoryVariant | None = None,
) -> str:
    kind = "Experiment" if result_source == "experiment" else "Predicted"
    return category_display_label(split, _run_label(config, run_id, variant), kind=kind)


def _picker_id(experiment_id: str, variant: CategoryVariant | None) -> str:
    # The picker DEDUPES by id, so each bin of a checkpoint needs its own — otherwise only
    # the first bin published would ever appear in the Experiments list.
    return experiment_id if variant is None else f"{experiment_id}@{variant.id_suffix}"


def _publication_experiment_metadata(
    *,
    experiment_id: str,
    campaign: str,
    training_campaign: str,
    checkpoint: str,
    config: dict[str, Any],
    run_id: str,
    variant: CategoryVariant | None = None,
) -> dict[str, Any]:
    """The category's ``experiment`` block. Raises ``MissingIntentError`` (see
    ``experiment_intent``) — a run is not published without a stated intent."""
    return {
        "id": _picker_id(experiment_id, variant),
        "group": campaign,
        "checkpoint": checkpoint,
        "label": _run_label(config, run_id, variant),
        # The same run, structured for the picker: its name, which records these are, and
        # every parameter as a named row.
        "runName": run_id,
        "variantLabel": None if variant is None else variant.label_suffix,
        "parameters": run_parameter_rows(config),
        "intent": experiment_intent(
            group=campaign, training_campaign=training_campaign, run_id=run_id, variant=variant,
        ),
        "model": config.get("model"),
        "predictionOutput": config.get("prediction_output", "state"),
        "horizonMode": config.get("horizon_mode", "normalized"),
        "seed": config.get("seed"),
    }


def _apply_category_refresh(
    manifest_path: Path,
    category_key: str,
    label: str,
    result_source: str,
    experiment_metadata: dict[str, Any] | None,
) -> bool:
    """Patch one category's derived label/metadata in a categories.json; True if found."""
    if not manifest_path.is_file():
        return False
    document = _load_object(manifest_path)
    categories = document.get("categories")
    if not isinstance(categories, list):
        raise ValueError(f"{manifest_path} must contain a categories array")
    changed = False
    found = False
    updated_categories: list[Any] = []
    for value in categories:
        if not isinstance(value, dict) or value.get("key") != category_key:
            updated_categories.append(value)
            continue
        found = True
        updated = dict(value)
        if updated.get("label") != label:
            updated["label"] = label
            changed = True
        if updated.get("resultSource") != result_source:
            updated["resultSource"] = result_source
            changed = True
        if experiment_metadata is not None:
            if updated.get("experiment") != experiment_metadata:
                updated["experiment"] = experiment_metadata
                changed = True
        elif "experiment" in updated:
            del updated["experiment"]
            changed = True
        updated_categories.append(updated)
    if found and changed:
        document["categories"] = updated_categories
        _write_json_atomic(manifest_path, document)
    return found


def refresh_category_metadata(plan: PublicationPlan) -> bool:
    """Refresh derived labels/metadata without regenerating archived trajectories."""
    return _apply_category_refresh(
        plan.comparison_dir.parent / "categories.json",
        plan.category,
        plan.category_label,
        plan.result_source,
        plan.experiment_metadata if plan.result_source == "experiment" else None,
    )


def _listed_category_keys(frontend_airports_root: Path, airport: str) -> set[str]:
    manifest_path = frontend_airports_root / airport / "comparison" / "categories.json"
    if not manifest_path.is_file():
        return set()
    return {
        value.get("key")
        for value in _load_object(manifest_path).get("categories") or ()
        if isinstance(value, dict)
    }


def refresh_labels_from_manifests(
    output_root: Path, frontend_airports_root: Path
) -> tuple[int, int]:
    """Recompute labels/metadata for every completed publication under ``output_root``.

    Reads only the stored publication manifests (which carry the run's exact config), so
    it needs neither the experiment index nor the checkpoints and never regenerates
    trajectories — a pure metadata refresh for already-published categories. ALL OR NOTHING
    on intents: if any listed experiment category has no registry entry, nothing is written
    and every missing entry is named (``MissingIntentError``).
    """
    seen = 0
    listed: dict[str, set[str]] = {}
    updates: list[tuple[dict[str, Any], str, dict[str, Any] | None]] = []
    unlisted: list[tuple[dict[str, Any], Path]] = []
    missing: list[str] = []
    for manifest_path in sorted(output_root.rglob(PUBLICATION_MANIFEST)):
        document = _load_object(manifest_path)
        if (
            document.get("schemaVersion") not in PUBLICATION_SCHEMAS_READ
            or document.get("status") != "completed"
        ):
            continue
        seen += 1
        airport = document["airport"]
        if airport not in listed:
            listed[airport] = _listed_category_keys(frontend_airports_root, airport)
        if document["category"] not in listed[airport]:
            unlisted.append((document, manifest_path))
            continue
        config = document.get("config") or {}
        run_id = document.get("runId") or ""
        result_source = document.get("resultSource") or "experiment"
        variant = _variant_from_document(document)
        metadata = None
        if result_source == "experiment":
            try:
                metadata = _publication_experiment_metadata(
                    experiment_id=document.get("experimentId") or run_id,
                    campaign=_experiment_group(document.get("campaign"), variant),
                    training_campaign=document.get("campaign") or "",
                    checkpoint=document.get("checkpoint") or "",
                    config=config,
                    run_id=run_id,
                    variant=variant,
                )
            except MissingIntentError as error:
                missing.append(f"{airport}/{document['category']}: {error}")
                continue
        label = _publication_label(
            document.get("split") or "", result_source, config, run_id, variant
        )
        updates.append((document, label, metadata))
    if missing:
        raise MissingIntentError(
            f"refusing to refresh: {len(missing)} published experiment categories have no "
            "registered intent (nothing was written):\n  " + "\n  ".join(missing)
        )
    patched = 0
    for document, label, metadata in updates:
        _apply_category_refresh(
            frontend_airports_root / document["airport"] / "comparison" / "categories.json",
            document["category"],
            label,
            document.get("resultSource") or "experiment",
            metadata,
        )
        patched += 1
        print(f"  ✓ refreshed {document['airport']}/{document['category']}")
    for document, manifest_path in unlisted:
        print(
            f"  ⚠ no category {document['category']} at "
            f"{document['airport']} (publication {manifest_path})"
        )
    return seen, patched


def _publication_document(
    plan: PublicationPlan,
    *,
    status: str,
    failure: str | None = None,
    completed_steps: Iterable[str] = (),
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schemaVersion": PUBLICATION_SCHEMA,
        "updatedAtUtc": _utc_now(),
        "status": status,
        "failure": failure,
        "experimentId": plan.experiment.experiment_id,
        "campaign": plan.experiment.campaign,
        "runId": plan.experiment.run_id,
        "checkpoint": plan.experiment.checkpoint_relative,
        "checkpointSha256": plan.experiment.checkpoint_sha256,
        "airport": plan.airport,
        "split": plan.split,
        "resultSource": plan.result_source,
        "category": plan.category,
        "rawOutputDir": _path_for_manifest(plan.output_dir),
        "predictionDir": (
            _path_for_manifest(plan.prediction_dir)
            if plan.prediction_dir is not None else None
        ),
        "frontendDir": _path_for_manifest(plan.comparison_dir),
        "completedSteps": list(completed_steps),
        "config": plan.experiment.config,
        "recordRetention": plan.record_retention,
    }
    if plan.anytime is not None:
        # The domain fact, and the ONLY thing stored for an anytime publication: its variant
        # is a rendering of this block, and re-deriving it on read is what lets the label
        # wording change without rewriting every manifest.
        document["anytime"] = asdict(plan.anytime)
    elif plan.category_variant is not None:
        # A flag slug has no domain object behind it, so the rendering IS the record.
        document["variant"] = asdict(plan.category_variant)
    if plan.records_archive.is_file():
        document["recordsArchive"] = {
            "file": plan.records_archive.name,
            "bytes": plan.records_archive.stat().st_size,
            "sha256": _sha256(plan.records_archive),
        }
    if status == "completed":
        summary = _load_object(plan.summary)
        report = _load_object(plan.evaluation_report)
        document["accuracy"] = summary.get("accuracy")
        document["evaluation"] = {
            key: value
            for key, value in report.items()
            if key not in {"trajectories", "reference"}
        }
    return document


def _loose_prediction_records(output_dir: Path) -> list[Path]:
    records = sorted(output_dir.glob("*_states.json"))
    records.extend(sorted(output_dir.glob("*_eval.json")))
    references = output_dir / "references"
    if references.is_dir():
        records.extend(sorted(references.glob("*_reference_eval.json")))
    return records


def archive_prediction_records(plan: PublicationPlan, *, replace: bool = False) -> int:
    """Atomically archive regenerable per-flight records after CZML/evaluation publication.

    ``summary.json``, aggregate evaluation, flyability, publication metadata and all frontend
    assets remain directly readable.  The archive retains the exact per-flight JSON contract for
    later inspection while avoiding tens of thousands of loose files and substantially reducing
    disk use.  Source files are removed only after the completed archive has been reopened and its
    member roster exactly matches the intended set.
    """
    records = _loose_prediction_records(plan.output_dir)
    if not records:
        return 0
    relative_names = [str(path.relative_to(plan.output_dir)) for path in records]
    if plan.records_archive.is_file() and not replace:
        # Recovery path for an interrupted post-archive cleanup: never replace the already
        # complete archive with only the loose subset that happened not to be deleted yet.
        with tarfile.open(plan.records_archive, "r:gz") as archive:
            archived_names = {
                member.name for member in archive.getmembers() if member.isfile()
            }
        missing = set(relative_names) - archived_names
        if missing:
            raise RuntimeError(
                f"existing prediction archive is missing loose record {sorted(missing)[0]}"
            )
    else:
        temporary = plan.records_archive.with_suffix(plan.records_archive.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        try:
            with tarfile.open(temporary, "w:gz") as archive:
                for path, relative_name in zip(records, relative_names):
                    archive.add(path, arcname=relative_name, recursive=False)
            with tarfile.open(temporary, "r:gz") as archive:
                archived_names = [
                    member.name for member in archive.getmembers() if member.isfile()
                ]
            if archived_names != relative_names:
                raise RuntimeError("prediction record archive roster verification failed")
            temporary.replace(plan.records_archive)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    for path in records:
        path.unlink()
    references = plan.output_dir / "references"
    if references.is_dir() and not any(references.iterdir()):
        references.rmdir()
    _sha256.cache_clear()
    return len(records)


def rebuild_publication_index(output_root: Path = RAW_OUTPUT_ROOT) -> dict[str, Any]:
    publications: list[dict[str, Any]] = []
    if output_root.exists():
        for path in sorted(output_root.rglob(PUBLICATION_MANIFEST)):
            try:
                document = _load_object(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if document.get("schemaVersion") in PUBLICATION_SCHEMAS_READ:
                publications.append(document)
    result = {
        "schemaVersion": PUBLICATION_INDEX_SCHEMA,
        "generatedAtUtc": _utc_now(),
        "publications": publications,
        "counts": {
            status: sum(item.get("status") == status for item in publications)
            for status in ("completed", "failed", "blocked", "running")
        },
    }
    _write_json_atomic(output_root / "index.json", result)
    return result


def run_publication(
    plan: PublicationPlan,
    *,
    dry_run: bool,
    force: bool,
    fail_fast: bool,
) -> str:
    context = f"{plan.experiment.experiment_id} · {plan.airport} · {plan.split}"
    if plan.result_source == "experiment":
        # First, before a reuse refresh or any predict/evaluate/CZML work: the category's
        # metadata cannot be written without it, so nothing else is worth starting.
        try:
            plan.experiment_intent
        except MissingIntentError as error:
            print(f"  ⚠ blocked {context}: {error}")
            return "blocked"
    if not force and plan.is_complete():
        if not dry_run:
            refresh_category_metadata(plan)
            if plan.record_retention == "archive" and plan.prediction_dir is None:
                archived = archive_prediction_records(plan)
            else:
                archived = 0
            if archived:
                _write_json_atomic(
                    plan.publication_manifest,
                    _publication_document(
                        plan,
                        status="completed",
                        completed_steps=("predict", "evaluate", "publish-czml", "archive-records"),
                    ),
                )
                rebuild_publication_index(plan.raw_output_root)
                print(f"  ✓ archived {archived} per-flight records for {context}")
        print(f"  ✓ reuse {context}")
        return "completed"

    error = plan.preflight_error()
    if error:
        print(f"  ⚠ blocked {context}: {error}")
        # A blocked attempt must never overwrite a COMPLETED manifest. It publishes nothing,
        # so the previous publication still stands on disk — but its document carries the
        # accuracy and evaluation blocks, and replacing it with a blocked one loses them,
        # flips the publication index, and makes `is_complete()` false forever, so the next
        # run blocks again on the same stale reason. Refuse loudly, write nothing.
        if not dry_run and not _manifest_is_completed(plan.publication_manifest):
            _write_json_atomic(
                plan.publication_manifest,
                _publication_document(plan, status="blocked", failure=error),
            )
            rebuild_publication_index(plan.raw_output_root)
        elif not dry_run:
            print(f"     (kept the completed manifest at {plan.publication_manifest})")
        return "blocked"

    commands = plan.commands()
    if dry_run:
        print(f"\n━━ {context}")
        for label, command in commands:
            print(f"  [{label}] {' '.join(command)}")
        return "planned"

    completed: list[str] = []
    _write_json_atomic(
        plan.publication_manifest,
        _publication_document(plan, status="running"),
    )
    rebuild_publication_index(plan.raw_output_root)
    try:
        for label, command in commands:
            print(f"\n=== [{context} · {label}] ===\n{' '.join(command)}", flush=True)
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            completed.append(label)
        if plan.record_retention == "archive" and plan.prediction_dir is None:
            archived = archive_prediction_records(plan, replace=True)
            completed.append("archive-records")
            print(f"  ✓ archived {archived} per-flight records -> {plan.records_archive}")
        refresh_category_metadata(plan)
        _write_json_atomic(
            plan.publication_manifest,
            _publication_document(plan, status="completed", completed_steps=completed),
        )
        rebuild_publication_index(plan.raw_output_root)
        return "completed"
    except KeyboardInterrupt as exc:
        failure = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(
            plan.publication_manifest,
            _publication_document(
                plan,
                status="failed",
                failure=failure,
                completed_steps=completed,
            ),
        )
        rebuild_publication_index(plan.raw_output_root)
        print(f"  ✗ interrupted {context}: {failure}", file=sys.stderr)
        raise
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(
            plan.publication_manifest,
            _publication_document(
                plan,
                status="failed",
                failure=failure,
                completed_steps=completed,
            ),
        )
        rebuild_publication_index(plan.raw_output_root)
        print(f"  ✗ failed {context}: {failure}", file=sys.stderr)
        if fail_fast:
            raise
        return "failed"


def _experiment_root(index_path: Path) -> Path:
    document = _load_object(index_path)
    return Path(document.get("root") or index_path.parent).resolve()


def _normalize_checkpoint_id(
    value: str,
    *,
    experiment_root: Path = EXPERIMENT_ROOT,
) -> str:
    normalized = value.strip().replace("\\", "/").strip("/")
    suffix = "/checkpoint.pt"
    if normalized.endswith(suffix):
        normalized = normalized[:-len(suffix)]
    candidate = Path(normalized)
    candidate_paths = (
        (candidate,) if candidate.is_absolute() else (REPO_ROOT / candidate,)
    )
    for candidate_path in candidate_paths:
        try:
            return candidate_path.resolve().relative_to(experiment_root.resolve()).as_posix()
        except ValueError:
            continue
    return normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-index", type=Path, default=EXPERIMENT_INDEX)
    parser.add_argument("--output-root", type=Path, default=RAW_OUTPUT_ROOT)
    parser.add_argument("--harvest-root", type=Path, default=HARVEST_ROOT)
    parser.add_argument("--frontend-airports-root", type=Path, default=FRONTEND_AIRPORTS_ROOT)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help=(
            "indexed experiment run ID or repository-relative checkpoint.pt path; "
            "repeat to publish selected checkpoints"
        ),
    )
    parser.add_argument("--campaign", action="append", default=None)
    parser.add_argument("--airport", action="append", default=None)
    parser.add_argument(
        "--reuse-prediction-dir",
        action="append",
        default=None,
        metavar="EXPERIMENT_ID=DIR",
        help=(
            "publish the records a campaign already wrote instead of predicting again; "
            "DIR is read-only here (never archived or deleted) and its summary.json must "
            "name this checkpoint and split. Repeat per checkpoint"
        ),
    )
    parser.add_argument(
        "--category-variant",
        default=None,
        metavar="SLUG[=LABEL]",
        help=(
            "publish this run as a distinct category of the SAME checkpoint: SLUG is "
            "appended to the category key and to the picker's experiment id, and LABEL (or "
            "SLUG) to the display label. Needed whenever two prediction directories share a "
            "checkpoint — a predict-time projection or hook beside its own baseline — "
            "because everything else in a category's identity comes from the checkpoint. A "
            "directory written by `run_ts.py anytime_curve --write-records` names its own "
            "bin instead, and giving both is refused"
        ),
    )
    parser.add_argument(
        "--category-group",
        default=None,
        metavar="CAMPAIGN",
        help=(
            "file the --category-variant under CAMPAIGN's picker heading (its registry entry "
            "gives the title and question) instead of the checkpoint's training campaign: for "
            "records a LATER campaign wrote with an earlier checkpoint"
        ),
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=PUBLISHABLE_SPLITS,
        default=None,
        help=(
            "development split; repeat as needed (default: train and val); "
            f"{', '.join(REUSE_ONLY_SPLITS)} (a day partition's held-out days) only with "
            "--reuse-prediction-dir and the experiment source"
        ),
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--result-source",
        choices=("prediction", "experiment"),
        default="experiment",
        help="frontend result category (default: experiment)",
    )
    parser.add_argument(
        "--record-retention",
        choices=("archive", "loose"),
        default="archive",
        help="archive per-flight JSON after successful publication (default), or keep loose files",
    )
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--refresh-labels-only",
        action="store_true",
        help=(
            "recompute frontend labels/metadata for every completed publication under "
            "--output-root from its stored manifest; no prediction, no CZML, no archive"
        ),
    )
    args = parser.parse_args(argv)

    if args.refresh_labels_only:
        try:
            seen, patched = refresh_labels_from_manifests(
                args.output_root, args.frontend_airports_root
            )
        except MissingIntentError as error:
            print(f"✗ {error}", file=sys.stderr)
            return 1
        print(f"refreshed {patched} of {seen} completed publications under {args.output_root}")
        return 0

    if args.max_checkpoints is not None and args.max_checkpoints <= 0:
        parser.error("--max-checkpoints must be positive")
    experiment_root = _experiment_root(args.experiment_index)
    selected_ids = (
        {
            _normalize_checkpoint_id(value, experiment_root=experiment_root)
            for value in args.checkpoint
        }
        if args.checkpoint else None
    )
    checkpoints = discover_checkpoints(
        args.experiment_index,
        selected_ids=selected_ids,
        campaigns=set(args.campaign) if args.campaign else None,
    )
    if selected_ids is not None:
        missing = selected_ids - {checkpoint.experiment_id for checkpoint in checkpoints}
        if missing:
            parser.error(f"checkpoint is not a completed indexed run: {sorted(missing)[0]}")
    if args.max_checkpoints is not None:
        checkpoints = checkpoints[:args.max_checkpoints]
    if not checkpoints:
        parser.error("no completed indexed checkpoints matched the selection")

    reused_dirs: dict[str, Path] = {}
    for assignment in args.reuse_prediction_dir or ():
        experiment_id, separator, directory = assignment.partition("=")
        if not separator:
            parser.error(f"--reuse-prediction-dir expects EXPERIMENT_ID=DIR, got {assignment!r}")
        normalized = _normalize_checkpoint_id(experiment_id, experiment_root=experiment_root)
        # One invocation publishes ONE directory per checkpoint. Silently keeping the last
        # would turn the obvious way to ask for several anytime bins at once into a run that
        # publishes only the last of them and says nothing.
        if normalized in reused_dirs:
            parser.error(
                f"--reuse-prediction-dir names {normalized!r} twice ({reused_dirs[normalized]} "
                f"and {directory}); one invocation publishes one directory per checkpoint — "
                "run it once per anytime bin"
            )
        reused_dirs[normalized] = Path(directory).resolve()
    unknown = set(reused_dirs) - {checkpoint.experiment_id for checkpoint in checkpoints}
    if unknown:
        parser.error(f"--reuse-prediction-dir names an unselected run: {sorted(unknown)[0]}")

    variant = None
    if args.category_variant is not None:
        try:
            variant = _parse_category_variant(args.category_variant)
        except ValueError as error:
            parser.error(str(error))
    if args.category_group is not None:
        if variant is None:
            parser.error("--category-group files a --category-variant; give one")
        variant = replace(variant, group=args.category_group)

    requested_airports = {value.strip().upper() for value in args.airport} if args.airport else None
    splits = tuple(args.split or DEVELOPMENT_SPLITS)
    plans: list[PublicationPlan] = []
    for checkpoint in checkpoints:
        airports = sorted(checkpoint.arrival_manifests)
        if requested_airports is not None:
            airports = [airport for airport in airports if airport in requested_airports]
        for airport in airports:
            for split in splits:
                try:
                    plans.append(PublicationPlan(
                        checkpoint,
                        airport,
                        split,
                        result_source=args.result_source,
                        raw_output_root=args.output_root.resolve(),
                        harvest_root=args.harvest_root.resolve(),
                        frontend_airports_root=args.frontend_airports_root.resolve(),
                        device=args.device,
                        record_retention=args.record_retention,
                        prediction_dir=reused_dirs.get(checkpoint.experiment_id),
                        variant=variant,
                    ))
                except ValueError as error:   # a refused combination of flags, before any work
                    parser.error(f"{checkpoint.experiment_id} {airport}/{split}: {error}")
    if not plans:
        parser.error("no checkpoint provenance contains the requested airport(s)")

    print(
        f"{len(checkpoints)} checkpoint(s), {len(plans)} airport/split publication job(s); "
        f"source={args.result_source}; outer-test is sealed"
    )
    counts: dict[str, int] = {}
    for plan in plans:
        status = run_publication(
            plan,
            dry_run=args.dry_run,
            force=args.force,
            fail_fast=args.fail_fast,
        )
        counts[status] = counts.get(status, 0) + 1
    print(f"\n✓ publication batch finished: {counts}")
    return 1 if counts.get("failed", 0) or counts.get("blocked", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
