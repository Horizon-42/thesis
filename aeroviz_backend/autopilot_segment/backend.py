"""``POST /autopilot/segment``: which set, which spec, the rebuilt flights kept, one flight flown at a time — and only
each page's latest request: a page (``clientId``) numbers its requests (``seq``, rising), and a request of a higher number
supersedes the page's lower ones (`Superseded`) — one still waiting is not flown, one flying stops before its next cycle,
one that arrives after it is refused at once — so clicking through bands never queues segments nobody is looking at.
The page's own numbers decide, not the order requests happen to arrive in.

Which flight: the request names a Training set (``airport``, ``setId``) and a flight of it (``flightKey``); the set's
sample (under the frontend's airports root) names the artefact it was exported from and the split it was drawn from.
Which sentence: ``sentence`` is null for the flight's labelled sentence (the truth), or a model's own sentence of it
(`segment.model_sentence`: the prior's free generation, one sample, as the Training view read it from its overlay) —
the words are the request's; nothing precomputed is read. It names the procedure's masks it was spoken under, each with
the digest of the data the set read; the backend builds them on its own code and data (`prior.masks`) and refuses the
request when a set is unknown or its data moved — the glidepath lower edge must stop the flight where it stopped the
sample.
Its spec is the ONE executor spec this code accepts for the artefact (`replay.open_executor`), looked up again whenever
a spec is added, moved or rewritten.

WARMED UP AT START (`warm_up`, which the server runs in a thread): every Training set this code can fly is opened ahead
of the first request — its flights rebuilt, its spec chosen, the procedure's masks built — which the first request of a
set would otherwise do (2–4 s a set, the first masks 2 s more). It reads each split's files once for every set drawn from
it and lets them go when it ends (the val split's signals are ~0.8 s to read and a few hundred MB held); a set opened by a
request reads its own and lets them go. The airports' published vertical paths are kept.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.runway_data import VerticalPath, published_vertical_paths
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions import training_files
from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.instructions.artefact import load_candidates
from ts_transformer.io_utils import utc_now
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.prior.masks import SETS, ProcedureMasks
from ts_transformer.repo_layout import COMPARISON_AIRPORTS_ROOT, OPT_OUTPUTS_ROOT, REPO_ROOT

from aeroviz_backend.autopilot_segment import single
from aeroviz_backend.autopilot_segment.errors import NotFlyable, NotListed, RequestRefused, Superseded
from aeroviz_backend.autopilot_segment.fly import (
    MODEL_WORD_CLOCK, FlightContext, SplitFiles, fly_segment, open_flights, read_split,
)
from aeroviz_backend.autopilot_segment.payload import SCHEMA, segment_payload
from aeroviz_backend.autopilot_segment.segment import model_sentence

#: Where executor specs are written (`run_ts.py executor_spec --dir`): each is a directory holding ``spec.json``.
DEFAULT_EXECUTOR_ROOT = OPT_OUTPUTS_ROOT / "POOLED" / "executor"
#: An airport as the frontend's directories name it; the request's airport is a path segment.
AIRPORT_CODE = re.compile(r"[A-Z0-9]{3,4}")


def _field(record: dict[str, Any], name: str) -> Any:
    if name not in record:
        raise RequestRefused(f"the request has no {name!r}")
    return record[name]


class AutopilotSegmentBackend:
    """``fly(payload)`` for ``POST /autopilot/segment``: ``{clientId, seq, airport, setId, flightKey, column, row,
    sentence}``."""

    def __init__(self, *, airports_root: Path = COMPARISON_AIRPORTS_ROOT,
                 executor_root: Path = DEFAULT_EXECUTOR_ROOT) -> None:
        self.airports_root = Path(airports_root)
        self.executor_root = Path(executor_root)
        self._lock = threading.Lock()
        # each page's highest request number (`clientId` → `seq`): a lower one waiting or flying is superseded by it
        self._latest: dict[str, int] = {}
        self._claims = threading.Lock()
        # an artefact's spec, with the spec files it was chosen among (their paths and times of writing)
        self._executors: dict[Path, tuple[tuple[tuple[Path, int], ...], tuple[Path, ExecutorParams, dict[str, Any], Words]]] = {}
        # each Training set's flights, rebuilt together the first time the set is asked for (`open_flights`: one read of
        # the split's signals, the airports' procedure files and arrival manifests, ~2 s for a set of 40 where one flight
        # alone took 1.5–2 s): keyed by the artefact, the split and the set's flights
        self._sets: dict[tuple[Path, str, tuple[str, ...]], dict[str, FlightContext | NotFlyable | ValueError]] = {}
        self._files: dict[Path, tuple[int, dict[str, Any]]] = {}
        # the procedure's masks built for an artefact's airports (the published finals are read once)
        self._masks: dict[tuple[Path, tuple[str, ...]], ProcedureMasks] = {}
        # each airport's published vertical paths for an artefact's candidates (the harvest's runway data, read once)
        self._vertical_paths: dict[tuple[Path, str], tuple[VerticalPath, ...]] = {}

    def procedure_masks(self, artefact: Path, asked: tuple[tuple[str, str], ...]) -> ProcedureMasks:
        """The sets ``asked`` (name, data digest) built for ``artefact``'s airports on this code and data — refused
        unless this code implements each and it reads the data the sample was spoken under."""
        names = tuple(name for name, _ in asked)
        if not names:
            return ProcedureMasks.none()
        try:
            masks = self._built_masks(artefact, names)
        except ValueError as error:
            raise RequestRefused(f"the sentence's procedure's masks: {error}") from None
        built = masks.data_sha256()
        moved = [name for name, digest in asked if built[name] != digest]
        if moved:
            raise RequestRefused(f"the data the procedure's masks {moved} read differs from the sample's (its overlay's "
                                 f"digests differ from this backend's, built once when it first flew them — restart the "
                                 f"backend if the procedure data changed since): it cannot be flown as it was said")
        return masks

    def _built_masks(self, artefact: Path, names: tuple[str, ...]) -> ProcedureMasks:
        """The sets ``names`` built for ``artefact``'s airports, once (`ProcedureMasks.build`)."""
        key = (artefact, names)
        if key not in self._masks:
            self._masks[key] = ProcedureMasks.build(names, load_candidates(artefact))
        return self._masks[key]

    def _json(self, path: Path) -> dict[str, Any]:
        """A Training file, parsed once per version on disk (a sample is several MB)."""
        version = path.stat().st_mtime_ns
        if path not in self._files or self._files[path][0] != version:
            self._files[path] = (version, json.loads(path.read_text(encoding="utf-8")))
        return self._files[path][1]

    def training_set(self, airport: str, set_id: str) -> tuple[Path, str, dict[str, Any]]:
        """The set's artefact (absolute), the split its flights were drawn from, and its sample — refused by name
        unless it is a read-back set the Training export writes (`training_files.check_readback`, the exporters' own
        check, on this backend's cached copies of the files)."""
        if not AIRPORT_CODE.fullmatch(airport):
            raise RequestRefused(f"airport {airport!r} is not an airport code")
        training = self.airports_root / airport / "training"
        index = training / training_files.INDEX_FILE
        if not index.is_file():
            raise NotListed(f"{airport} has no Training export")
        try:
            entry = training_files.listed_set(self._json(index), index, airport, set_id)
        except training_files.NotListed as error:
            raise NotListed(str(error)) from None
        sample = self._json(training / entry["file"])
        training_files.check_readback(entry, sample, training / entry["file"], airport)
        return REPO_ROOT / sample["producedBy"]["artefact"], training_files.SPLIT, sample

    def executor_for(self, artefact: Path) -> tuple[Path, ExecutorParams, dict[str, Any], Words]:
        """The one executor spec written by this code for ``artefact``'s vocabulary (`replay.open_executor`); refused,
        naming every spec and why, when there is none or more than one. Chosen again when a spec is added, moved or
        rewritten — and only while the single-flight executor that flies it mirrors this code
        (`single.require_mirrored_source`)."""
        listing = tuple((path.parent, path.stat().st_mtime_ns) for path in sorted(self.executor_root.glob("*/spec.json")))
        if artefact not in self._executors or self._executors[artefact][0] != listing:
            single.require_mirrored_source()
            usable, refused = [], []
            for directory, _ in listing:
                try:
                    usable.append((directory, *replay.open_executor(directory, artefact)))
                except ValueError as error:
                    refused.append(f"{directory.name}: {error}")
            if len(usable) != 1:
                named = [directory.name for directory, *_ in usable]
                raise ValueError(f"{len(usable)} executor specs under {self.executor_root} fly {artefact.name} with this "
                                 f"code ({named}); one is needed. Refused: {refused}")
            self._executors[artefact] = (listing, usable[0])
        return self._executors[artefact][1]

    def vertical_paths(self, artefact: Path, geometry: AirportGeometry) -> tuple[VerticalPath, ...]:
        """``geometry``'s airport's published vertical paths (`published_vertical_paths`, ~0.5 s: the harvest's runway
        data), read once for ``artefact``'s candidates."""
        key = (artefact, geometry.code)
        if key not in self._vertical_paths:
            self._vertical_paths[key] = published_vertical_paths(geometry)
        return self._vertical_paths[key]

    def open_set(self, artefact: Path, split: str, sample: dict[str, Any], words: Words,
                 files: Callable[[], SplitFiles] | None = None
                 ) -> tuple[dict[str, FlightContext | NotFlyable | ValueError], bool]:
        """The set's flights, rebuilt together the first time the set is asked for (`open_flights`), and whether they
        were kept from before; ``files``: the split's files the warm-up holds (None: read for this set, then let go)."""
        key = (artefact, split, tuple(item["datasetId"] for item in sample["flights"]))
        kept = key in self._sets
        if not kept:
            split_files = read_split(artefact, split, words) if files is None else files()
            self._sets[key] = open_flights(artefact, split_files, key[2], words,
                                           lambda geometry: self.vertical_paths(artefact, geometry))
        return self._sets[key], kept

    def flight(self, artefact: Path, split: str, sample: dict[str, Any], dataset_id: str, words: Words
               ) -> tuple[FlightContext, bool]:
        """The flight rebuilt — with every other flight of its set, the first time the set is asked for — and whether
        it was kept from an earlier request; a flight the data cannot fly is refused (`NotFlyable`), one a check refused
        with that check's reason (`ValueError`, as when it was opened alone)."""
        contexts, kept = self.open_set(artefact, split, sample, words)
        context = contexts[dataset_id]
        if isinstance(context, (NotFlyable, ValueError)):
            raise type(context)(str(context))
        return context, kept

    def warm_up(self, log: Callable[[str], None] = print) -> None:
        """Every Training set this code can fly opened ahead of its first request (the module docstring): for each
        airport's index, each read-back set of the current reading rule (`READING_RULE`) — its sample and spec
        (`training_set`, `executor_for`), its flights (`open_set`, each split's files read once for all the sets drawn
        from it and let go at the end) and the procedure's masks (`prior.masks.SETS`: each set alone and all of them, as
        a request names them). Each set is opened under the request lock, so a request waits at most for the set being
        opened. A set it cannot open — no spec for its artefact, a refused or missing sample, an unreadable index — is
        skipped with its reason's first sentence (a request for it gets all of it) and its sample let go."""
        started, opened = time.perf_counter(), 0
        splits: dict[tuple[Path, str], SplitFiles] = {}

        def split_files(artefact: Path, split: str, words: Words) -> Callable[[], SplitFiles]:
            def files() -> SplitFiles:
                if (artefact, split) not in splits:
                    splits[(artefact, split)] = read_split(artefact, split, words)
                return splits[(artefact, split)]
            return files

        mask_names = {(name,) for name in SETS} | {tuple(SETS)}
        for index in sorted(self.airports_root.glob(f"*/training/{training_files.INDEX_FILE}")):
            airport = index.parent.parent.name
            try:
                with self._lock:                                    # the files read are the requests' cached copies
                    entries = list(training_files.index_sets(self._json(index), index, airport))
            except (ValueError, OSError) as error:
                log(f"autopilot warm-up: {airport} skipped — {index}: {error}")
                continue
            for entry in entries:
                if entry["kind"] != training_files.KIND_READBACK or entry["readingRule"] != READING_RULE:
                    continue
                began = time.perf_counter()
                with self._lock:
                    try:
                        artefact, split, sample = self.training_set(airport, entry["id"])
                        _, _, _, words = self.executor_for(artefact)
                        contexts, _ = self.open_set(artefact, split, sample, words, split_files(artefact, split, words))
                        for names in mask_names:
                            self._built_masks(artefact, names)
                    except (NotListed, ValueError, OSError, KeyError) as error:
                        self._files.pop(index.parent / entry["file"], None)
                        # the reason's first sentence: a spec refusal goes on to list every spec it read (~2 kB)
                        log(f"autopilot warm-up: {airport} {entry['id']} skipped — {type(error).__name__}: "
                            f"{str(error).split('; ')[0].split('. ')[0]}")
                        continue
                flyable = sum(isinstance(context, FlightContext) for context in contexts.values())
                opened += 1
                log(f"autopilot warm-up: {airport} {entry['id']}: {flyable} of {len(contexts)} flights opened in "
                    f"{time.perf_counter() - began:.1f} s")
        log(f"autopilot warm-up: {opened} sets ready in {time.perf_counter() - started:.1f} s")

    def _claim(self, client: str, seq: int) -> None:
        """Make ``seq`` the page's latest request — unless the page already sent a later one."""
        with self._claims:
            if client in self._latest and seq <= self._latest[client]:
                raise Superseded(f"this page's request {self._latest[client]} came in before its request {seq}")
            self._latest[client] = seq

    def fly(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = str(_field(payload, "clientId"))
        seq = _field(payload, "seq")
        airport = str(_field(payload, "airport"))
        set_id = str(_field(payload, "setId"))
        flight_key = str(_field(payload, "flightKey"))
        column_name = _field(payload, "column")
        row = _field(payload, "row")
        sentence = _field(payload, "sentence")
        if column_name not in COLUMNS:
            raise RequestRefused(f"column {column_name!r} is none of {COLUMNS}")
        if not isinstance(row, int) or isinstance(row, bool):
            raise RequestRefused(f"row must be an integer step, got {row!r}")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise RequestRefused(f"seq must be the page's request number, a whole number, got {seq!r}")
        self._claim(client, seq)
        superseded = lambda: self._latest[client] != seq  # noqa: E731 — read each cycle
        asked = time.perf_counter()
        with self._lock:
            if superseded():
                raise Superseded("a newer request from this page came in while this one waited")
            started = time.perf_counter()
            artefact, split, sample = self.training_set(airport, set_id)
            flights = [item for item in sample["flights"] if item["flightKey"] == flight_key]
            if len(flights) != 1:
                raise NotListed(f"Training set {set_id} at {airport} has no flight {flight_key}")
            dataset_id = flights[0]["datasetId"]
            directory, params, record, words = self.executor_for(artefact)
            opening = time.perf_counter()
            context, kept = self.flight(artefact, split, sample, dataset_id, words)
            model = None if sentence is None else model_sentence(sentence, words, len(context.geometry.candidates),
                                                                 len(context.reading.words), params.timeout_factor)
            opened = time.perf_counter()
            masks = None if model is None else self.procedure_masks(artefact, model.procedure_masks)
            result = fly_segment(context, params, words, COLUMNS.index(column_name), row, superseded, model, masks)
            answering = time.perf_counter()
            body = segment_payload(result, context, words)
            finished = time.perf_counter()
        return {
            "ok": True, "schema": SCHEMA, "airport": airport, "setId": set_id, "flightKey": flight_key,
            "datasetId": dataset_id, "computedUtc": utc_now(),
            # wall-clock seconds on the backend: waiting for the flight before it (one at a time); then, adding up to
            # ``computeS``: the set and the executor spec found, the flight rebuilt (or kept), the segment set up (its
            # sentence, clock and the executor's physics), the executor's cycles (``cycles`` of them: what was
            # computed, which may run past the judged outcome), the judge, and the answer written
            "timing": {"waitS": round(started - asked, 3), "setupS": round(opening - started, 3),
                       "openS": round(opened - opening, 3), "flightKept": kept,
                       "prepareS": round(answering - opened - result.fly_s - result.judge_s, 3),
                       "flyS": round(result.fly_s, 3), "cycles": int(result.flown.commands.shape[1]),
                       "judgeS": round(result.judge_s, 3), "answerS": round(finished - answering, 3),
                       "computeS": round(finished - started, 3)},
            # the spec and the artefact by name: their shas say which (the spec's is its record's); the word clock
            # this flight was flown on — the spec's for the truth, the time clock for a model's sentence
            "executor": {"spec": directory.name, "specSha256": record["sha256"],
                         "sourceSha256": record["source"]["executor_source_sha256"],
                         "wordClock": params.word_clock if model is None else MODEL_WORD_CLOCK, "cycleS": params.cycle_s,
                         "timeoutFactor": params.timeout_factor},
            "artefact": artefact.name,
            "vocabularySpecSha256": words.spec.sha256,
            "group": context.group,
            **body,
        }
