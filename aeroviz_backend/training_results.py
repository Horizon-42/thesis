"""`GET /training/results?stage=<A|B|C|D>&airport=<ICAO>&set=<id>` (outline §6.2 items 3, 4; D134): the results of the
experiment that made a Training set, for its details page.

The set is found in its stage's index of the airport (`TrainingFiles.listed_set`); only the files its ``source`` and
``model`` name are read, each under `4dTrajectory/outputs/`; the answer holds the NAMED fields of each section, never a
whole file. A section whose file lies elsewhere (a smoke set in a scratch directory) or is missing answers by name, and the
page says so. It reads at each request and writes nothing.

WHAT IS NEVER ANSWERED (outline §6.2 item 3). Stage A's val block of the labelling readout (the replay of the val days
waits for the user, vocabulary D85): the labelling is read for train and select only, the closed loop for the set's own
splits, which are train and select. Stage B's validation and the free generation of the val days: only for the set
exported from the base's claimed, written validation readout (`source.validationClaim`, D109; prior D119), and never
the val counts of a run's identity (prior D120: no identity is read). Stage C's validation readout: not read.

Sections — A: ``labelling``, ``closedLoop``; B: ``freeGeneration``, ``training``, ``validation``, ``speed``,
``choice``; C and D: ``rounds``, ``speed``, ``checks`` (D's rounds by its readout's ``all`` cells, with its losses by pair). An airport that is no airport code is refused (400), a set of another
format answered by name (409). Each is ``{"ok": true, ...}`` or ``{"ok": false, "problem": "..."}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from aeroviz_backend.autopilot_segment.backend import AIRPORT_CODE      # the live segments' own: never a path
from ts_transformer.instructions import training_files as stage_a_files
from ts_transformer.post import training_files as post_files
from ts_transformer.prior import training_files as prior_files
from ts_transformer.prior.checkpoint import holds_written_claim
from ts_transformer.repo_layout import REPO_ROOT

#: Where every file a section reads must lie.
OUTPUTS = REPO_ROOT / "4dTrajectory" / "outputs"
#: Each stage's Training files (its index and sets).
STAGES = {"A": stage_a_files.FILES, "B": prior_files.FILES, "C": post_files.FILES, "D": post_files.MULTI_FILES}
#: Stage A's labelling and closed loop: these splits only (never val, vocabulary D85).
STAGE_A_SPLITS = ("train", "select")
#: The variants whose scores a variant choice holds: MIRROR of `experiments/prior_select.py` `choose_variant`, which
#: writes them by these names (a runner the backend does not import).
CHOICE_VARIANTS = ("full", "constants")
#: The reader of the base's teacher-forced validation readout (`experiments/prior_validation.py`'s claim).
VALIDATION_READER = "prior_validation"


class Unreadable(ValueError):
    """A section's file lies outside the outputs root or is missing: the section answers this by name."""


FREE_GENERATION_FIELDS = {"sentences": "sentences", "outcomes": "outcomes", "timedOut": "timed_out",
                          "goArounds": "go_arounds", "atTheBound": "at_the_bound",
                          "wordsPerSentence": "words_per_sentence", "labelledWordsPerSentence": "labelled_words_per_sentence"}
SPEED_FIELDS = ("device", "batch", "loops", "loopSizes", "rowsTimed", "sentences", "priorStepMs", "executorStepsMs",
                "rowMs", "sentenceS", "flightRowsPerS", "shareOfInterval")
#: What a speed setting says of its host (a GPU's name too; the load and memory are the runner's own information).
SPEED_HOST = ("device", "torch", "threads", "name")


class TrainingResults:
    """The route over the Training sets of ``airports_root``, reading only under ``outputs`` (module docstring; a test
    gives both its own)."""

    def __init__(self, airports_root: Path, outputs: Path = OUTPUTS) -> None:
        self.airports_root, self.outputs = Path(airports_root), Path(outputs)

    def answer(self, stage: str, airport: str, set_id: str) -> tuple[int, dict[str, Any]]:
        """400 for a stage that is none of A, B, C, D; 404 for a set the stage's index of the airport does not list (or no
        index); else 200 with each section's fields or reason, and the one line of provenance."""
        if stage not in STAGES:
            return 400, {"ok": False, "error": f"stage {stage!r} is none of {sorted(STAGES)}"}
        if not AIRPORT_CODE.fullmatch(airport):
            return 400, {"ok": False, "error": f"airport {airport!r} is not an airport code"}
        try:
            entry, sample = STAGES[stage].listed_set(self.airports_root / airport / "training", airport, set_id)
        except stage_a_files.NotListed as error:
            return 404, {"ok": False, "error": str(error)}
        except (ValueError, KeyError) as error:            # an index or a set of another format: refused by name
            return 409, {"ok": False, "error": f"set {set_id} at {airport} cannot be read: {error!r}"}
        source = entry["source"]
        if stage == "A":
            provenance = source["instructions"]
            sections = {"labelling": self.section(lambda: self.labelling(source["instructions"])),
                        "closedLoop": self.section(lambda: self.closed_loop(
                            source["executor"], list(entry["cohort"]["splits"]), list(sample["vocabulary"]["rowIntervalsS"]),
                            airport))}
        elif stage == "B":
            model = entry["model"]
            provenance = source["readout"]
            sections = {"freeGeneration": self.section(lambda: self.free_generation(source, model)),
                        "training": self.section(lambda: self.training(model)),
                        "validation": self.section(lambda: self.validation(source, model)),
                        "speed": self.section(lambda: self.speed(source)),
                        "choice": self.section(lambda: self.choice(model))}
        else:
            model = entry["model"]
            provenance = model["campaign"]
            read = self.rounds if stage == "C" else self.multi_rounds
            sections = {"rounds": self.section(lambda: read(model["campaign"])),
                        "speed": self.section(lambda: self.speed(source)),
                        "checks": self.section(lambda: self.checks(model["campaign"]))}
        return 200, {"ok": True, "stage": stage, "airport": airport, "set": set_id, "provenance": provenance,
                     "sections": sections}

    # ---- the files
    def path(self, name: str | Path) -> Path:
        """A path a set names (a relative one is the repository's), refused by name unless it lies under the outputs
        root."""
        path = Path(name) if Path(name).is_absolute() else REPO_ROOT / name
        if not path.resolve().is_relative_to(self.outputs.resolve()):
            raise Unreadable(f"{name} is not under 4dTrajectory/outputs/ (a smoke or scratch output): not read")
        return path

    def read(self, name: str | Path) -> Any:
        path = self.path(name)
        if not path.is_file():
            raise Unreadable(f"{name} does not exist")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def section(build: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """A section's answer: its fields, or the reason it has none — a file elsewhere or missing, or one of another
        format (a field it lacks, a value it cannot read), named."""
        try:
            return {"ok": True, **build()}
        except Unreadable as error:
            return {"ok": False, "problem": str(error)}
        except (KeyError, TypeError, ValueError) as error:
            return {"ok": False, "problem": f"a file of another format: {error!r}"}

    # ---- stage A
    def labelling(self, instructions: str) -> dict[str, Any]:
        """Labelled and refused, the reasons, by airport — train and select only (never the val block, D85)."""
        readout = self.read(f"{instructions}/readout.json")
        return {"splits": {split: {"labelled": readout[split]["labelled"], "refused": readout[split]["refused"],
                                   "refusalReasons": readout[split]["refusal_reasons"],
                                   "refusedByAirport": readout[split]["refused_by_airport"],
                                   "labelledByAirport": {code: block["flights"]
                                                         for code, block in readout[split]["by_airport"].items()}}
                           for split in STAGE_A_SPLITS}}

    def closed_loop(self, executor: str, splits: list[str], intervals: list[float], airport: str) -> dict[str, Any]:
        """The executor's replay of each split's closed-loop sentences at each Δ: by group of dynamics, for the set's
        airport and for all airports, the flights, their outcomes and the share landed."""
        out = {}
        for split in splits:
            if split not in STAGE_A_SPLITS:
                raise Unreadable(f"split {split} is not read (vocabulary D85)")
            for interval in intervals:
                replay = self.read(f"{executor}/replay-closed-{split}-{interval:g}s/replay.json")
                out[f"{split} {interval:g}"] = {
                    "split": split, "intervalS": interval, "flights": replay["drawn"]["flights"],
                    "groups": {group: {where: {key: by[where]["all"]["all"][key] for key in ("flights", "outcomes", "landed")}
                                       for where in (airport, "all")}
                               for group, by in replay["readout"].items()}}
        return {"replays": out}

    # ---- stage B
    def prior_dir(self, model: Mapping[str, Any]) -> Path:
        directory = self.path(model["prior"])
        if not (directory / "config.json").is_file():
            raise Unreadable(f"{model['prior']} is no prior run (no config.json)")
        return directory

    def free_generation(self, source: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
        """The free-generation readout the set was made from; one of the val days only under the base's written claim
        (D109)."""
        config = self.read(f"{source['readout']}/config.json")
        if config["split"] == "val" and (source["validationClaim"] is None or not holds_written_claim(
                self.prior_dir(model), prior_files.CLAIM_READER, self.path(source["readout"]))):
            raise Unreadable(f"{source['readout']} reads the val days without the base's written claim (D109): not read")
        readout = self.read(f"{source['readout']}/readout.json")
        return {"split": config["split"], "selection": readout["selection"],
                "sides": {side: {code: {stratum: {name: cell[key] for name, key in FREE_GENERATION_FIELDS.items()}
                                        for stratum, cell in strata.items()}
                                 for code, strata in readout[side].items()}
                          for side in ("inside", "outside_fault", "outside_outcome")}}

    def training(self, model: Mapping[str, Any]) -> dict[str, Any]:
        history = self.read(f"{model['prior']}/history.json")
        return {"bestEpoch": history["best_epoch"],
                "epochs": [{"epoch": e["epoch"], "trainLossPerStep": e["train_loss_per_step"],
                            "selectLossPerStep": e["select_loss_per_step"]} for e in history["epochs"]]}

    def validation(self, source: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
        """The base's teacher-forced validation readout, beside the base — for the set of its claimed readout only."""
        prior = self.prior_dir(model)
        if source["validationClaim"] is None or not holds_written_claim(prior, prior_files.CLAIM_READER,
                                                                         self.path(source["readout"])):
            raise Unreadable("shown only for the set exported from the base's claimed validation readout (D109)")
        directory = prior.parent / "validation"
        if not holds_written_claim(prior, VALIDATION_READER, directory):
            raise Unreadable(f"{directory} is not the validation readout the base's written claim names")
        readout = self.read(directory / "readout.json")
        pooled = readout["teacher_forced"]["pooled"]
        return {"teacherForced": {"lossPerStep": pooled["loss_per_step"], "perColumn": pooled["per_column"]},
                "masksOnLabelledWords": {side: {code: {column: cell["share"] for column, cell in columns.items()}
                                                for code, columns in airports.items()}
                                         for side, airports in readout["masks_on_labelled_words"].items()}}

    def choice(self, model: Mapping[str, Any]) -> dict[str, Any]:
        """The campaign's two choices (the configuration, the variant): the campaign is the nearest directory above the
        prior run that holds ``campaign.json``."""
        prior = self.prior_dir(model)
        outputs = self.outputs.resolve()
        campaign = next((d for d in prior.resolve().parents if d.is_relative_to(outputs) and (d / "campaign.json").is_file()),
                        None)
        if campaign is None:
            raise Unreadable(f"{model['prior']} belongs to no campaign under the outputs (no campaign.json above it)")
        def step(made: Mapping[str, Any]) -> dict[str, Any]:
            return {"arms": {name: {"score": arm["score"], "folds": arm["folds"]} for name, arm in made["arms"].items()},
                    "seedScale": made["seed_scale"], "chosen": made["chosen"]}

        configuration = self.read(campaign / "choice_configuration.json")
        variant = self.read(campaign / "choice_variant.json")
        return {"configuration": {**step(configuration), "bestScore": configuration["best_score"],
                                  "within": configuration["within"]},
                "variant": {**step(variant), "scores": {name: variant[name] for name in CHOICE_VARIANTS}}}

    def speed(self, source: Mapping[str, Any]) -> dict[str, Any]:
        """The model's speed readout the set names (D136): the model it timed (stage B: the prior; C: the campaign's
        round), each setting's named statistics and what it ran on."""
        record = self.read(f"{source['speed']['readout']}/speed.json")
        timed = record["model"]
        model = timed["prior"] if timed["stage"] == "B" else f"{timed['campaign']} round {timed['round']}"
        return {"model": model, "split": timed["split"], "warmupRows": record["warmupRows"],
                "smoke": record["smoke"],
                "settings": [{**{key: setting[key] for key in SPEED_FIELDS},
                              "host": {key: setting["host"][key] for key in SPEED_HOST if key in setting["host"]}}
                             for setting in record["settings"]]}

    # ---- stage C
    def rounds(self, campaign: str) -> dict[str, Any]:
        """Each round the campaign holds (``round_<n>/round.json``, n from 0 while one exists): what it spoke and its
        selection readout by airport; and the start's (`start`)."""
        record = self.read(f"{campaign}/campaign.json")
        out = []
        while self.path(f"{campaign}/round_{len(out)}/round.json").is_file():
            made = self.read(f"{campaign}/round_{len(out)}/round.json")
            out.append({"round": made["round"],
                        "speaking": {"windows": made["speaking"]["windows"], "rewardSum": made["speaking"]["reward_sum"],
                                     "outcomes": made["speaking"]["outcomes"]},
                        "selection": self.selection(made)})
        return {"started": record["started_utc"], "rounds": out, "start": self.start(record["inputs"]["settings"])}

    @staticmethod
    def selection(made: Mapping[str, Any]) -> dict[str, Any]:
        """A round's selection readout by airport (its ``round.json``)."""
        return {code: {"windows": cell["windows"], "rewardMean": cell["reward_mean"], "outcomes": cell["outcomes"]}
                for code, cell in made["selection_readout"].items()}

    def start(self, settings: Mapping[str, Any]) -> dict[str, Any] | None:
        """The selection readout of the model a campaign starts from (frontend §4.3): None from the base (it has none);
        from another campaign's round (post-training D162), that round's own readout — read on the same select windows
        with the same numbers only when the two campaigns' ``select_seed`` and ``select_per_airport`` are equal — a MIRROR
        of the condition of the value method's warm-up check (`post_train.close_value_round`, a runner the backend does
        not import) — else ``selection`` None and ``why``."""
        start = settings["start"]
        if start is None:
            return None
        named = {"campaign": start["campaign"], "round": start["round"]}
        source = self.read(f"{start['campaign']}/campaign.json")["inputs"]["settings"]
        if (source["select_seed"], source["select_per_airport"]) != (settings["select_seed"], settings["select_per_airport"]):
            return {**named, "selection": None,
                    "why": "the source campaign read other select windows (its select seed or windows per airport differ)"}
        return {**named, "selection": self.selection(self.read(f"{start['campaign']}/round_{start['round']}/round.json")),
                "why": None}

    # ---- stage D
    def multi_rounds(self, campaign: str) -> dict[str, Any]:
        """Stage D's rounds (frontend §5.6): what each spoke (its windows, the commanded aircraft's reward sum and
        outcomes) and its readout by airport — the airport's ``all`` span and ``all`` kind: its windows and commanded
        aircraft, their reward sum, outcomes, go-arounds and silent ones, and the windows with a loss of each pair
        (`multi.separation.PAIRS`, once a pair a window). The start (stage C's round, D164) is read on stage C's windows,
        never stage D's: no readout of it, and why."""
        record = self.read(f"{campaign}/campaign.json")
        out = []
        while self.path(f"{campaign}/round_{len(out)}/round.json").is_file():
            made = self.read(f"{campaign}/round_{len(out)}/round.json")
            out.append({"round": made["round"],
                        "speaking": {"windows": made["speaking"]["windows"], "rewardSum": made["speaking"]["reward_sum"],
                                     "outcomes": made["speaking"]["outcomes"]},
                        "selection": {code: self.multi_cell(spans["all"]["all"])
                                      for code, spans in made["selection_readout"].items()}})
        start = record["inputs"]["settings"]["start"]
        return {"started": record["started_utc"], "rounds": out,
                "start": {"selection": None, "why": f"the start, round {start['round']} of stage C's {start['campaign']}, "
                                                    f"was read on stage C's windows, not stage D's"}}

    @staticmethod
    def multi_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
        """A cell of stage D's readout as the route answers it."""
        return {"windows": cell["windows"], "aircraft": cell["aircraft"], "rewardSum": cell["reward_sum"],
                "outcomes": cell["outcomes"], "goArounds": cell["go_arounds"], "silent": cell["silent"],
                "lossWindows": cell["loss_windows"]}

    def checks(self, campaign: str) -> dict[str, Any]:
        """The labeller's and the executor's checks at the campaign's start."""
        return {"checks": self.read(f"{campaign}/campaign.json")["checks"]}
