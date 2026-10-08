"""`GET /training/results` (outline §6.2 items 3, 4; D134): each stage's sections from the files its set names under the
outputs root, named fields only; a file elsewhere or missing answered by name; stage A's val block and stage B's val
results never answered but for the base's claimed set — on the frontend's fixtures (written by the exports), copied into
a tmp airports root whose sets name a tmp outputs root."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from aeroviz_backend.http_server import AeroVizBackendApp
from aeroviz_backend.training_results import TrainingResults
from ts_transformer.experiments.model_speed import TimedLoop, summary
from ts_transformer.instructions import training_files as stage_a_files
from ts_transformer.post import training_files as post_files
from ts_transformer.prior import training_files as prior_files

FIXTURES = Path(__file__).resolve().parents[2] / "aeroviz-4d" / "src" / "data" / "__tests__" / "fixtures"
#: A value no answer may hold: written into every block the route must never read.
SEALED = 987654321


def write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def listed(root: Path, stage_dir: str, files, set_dirs: list[str], change) -> str:
    """The stage's fixture index and samples copied under ``root`` (its airport), each entry changed by ``change``."""
    index = json.loads((FIXTURES / stage_dir / files.INDEX_FILE).read_text(encoding="utf-8"))
    training = root / index["airport"] / "training"
    training.mkdir(parents=True, exist_ok=True)
    for name in set_dirs:
        shutil.copytree(FIXTURES / stage_dir / name, training / name)
    index["sets"] = [change(entry) for entry in index["sets"]]
    write(training / files.INDEX_FILE, index)
    return index["airport"]


@pytest.fixture()
def world(tmp_path):
    outputs = tmp_path / "outputs"
    roots = {stage: tmp_path / f"airports_{stage}" for stage in "ABC"}        # the fixtures share an airport and an id
    # stage A: the labelling readout (its val block sealed) and the executor's replays at each Δ of the set
    instructions, executor = outputs / "instruction_language" / "v", outputs / "executor" / "v"
    split = {"labelled": 10, "refused": 2, "refusal_reasons": {"impossible ground speed": 2},
             "refused_by_airport": {"KXXX": {"impossible ground speed": 2}}, "by_airport": {"KXXX": {"flights": 10}}}
    write(instructions / "readout.json", {"train": split, "select": split, "val": {**split, "labelled": SEALED}})
    cell = {"all": {"all": {"flights": 4, "outcomes": {"landed": 3, "timeout": 1}, "landed": 0.75, "words_judged": SEALED}}}
    for name in ("train", "select"):
        for interval in (2, 4, 8):
            write(executor / f"replay-closed-{name}-{interval}s" / "replay.json",
                  {"drawn": {"flights": 4}, "readout": {"own dynamics": {"KXXX": cell, "all": cell}}})
    airport_a = listed(roots["A"], "stage_a", stage_a_files, ["fixture_set"], lambda e: {
        **e, "source": {**e["source"], "instructions": str(instructions), "executor": str(executor)}})
    # stage B: a campaign with its choices, the base and its readouts; the set of the base's claimed val readout
    campaign = outputs / "prior" / "campaign"
    write(campaign / "campaign.json", {})
    arms = {"arms": {"C_full": {"score": 1.36, "folds": {"KXXX": 1.3}}}, "seed_scale": 0.002, "best_score": 1.36,
            "within": ["C"], "chosen": "C"}
    write(campaign / "choice_configuration.json", arms)
    # the variant step's own fields (prior_select): each variant's score, no best score
    write(campaign / "choice_variant.json", {"arms": arms["arms"], "seed_scale": 0.002, "full": 1.36, "constants": 1.37,
                                             "chosen": "full"})
    prior = campaign / "base" / "run"
    write(prior / "config.json", {})
    write(prior / "history.json", {"best_epoch": 2, "epochs": [
        {"epoch": 1, "train_loss_per_step": 2.0, "select_loss_per_step": 1.5, "select_per_column": {}, "seconds": SEALED}]})
    cell = {"sentences": 3, "outcomes": {"landed": 3}, "timed_out": 0, "go_arounds": 1, "at_the_bound": 0,
            "words_per_sentence": {"heading": 9.0}, "labelled_words_per_sentence": {"heading": 8.0},
            "go_around_probability_on_final": {"rows": SEALED}}
    readout = {"selection": "landed", "inside": {"KXXX": {"vectored": cell}}, "outside_fault": {}, "outside_outcome": {}}
    select_readout, val_readout = campaign / "base" / "free_generation_select", campaign / "base" / "free_generation"
    write(select_readout / "config.json", {"split": "select"})
    write(select_readout / "readout.json", readout)
    write(val_readout / "config.json", {"split": "val"})
    write(val_readout / "readout.json", readout)
    validation = campaign / "base" / "validation"
    write(validation / "readout.json", {"teacher_forced": {"pooled": {"loss_per_step": 1.08, "per_column": {"heading": 0.4}},
                                                           "by_airport": SEALED},
                                        "masks_on_labelled_words": {"inside": {"KXXX": {"angle": {"said": 5, "blocked": 1,
                                                                                                  "share": 0.2}}}}})
    speed = write(outputs / "speed" / "base" / "speed.json", {
        "model": {"stage": "B", "prior": str(prior), "split": "select"}, "warmupRows": 20, "smoke": False,
        # a setting as the runner writes it: its statistics by the runner's own summary, of fixed times
        "settings": [{"device": "cpu", "batch": 1, "loops": 1, "loopSizes": [1],
                      **summary([TimedLoop(None, lambda: None, prior_s=[0.005, 0.007], executor_s=[0.001, 0.001],
                                           flying=[np.array([True]), np.array([True])])], 4.0),
                      "host": {"device": "cpu", "torch": "2", "threads": 1, "loadAverage": [SEALED]}}]}).parent

    def prior_entry(entry):
        claimed = entry["source"]["validationClaim"] is not None
        return {**entry, "model": {**entry["model"], "prior": str(prior)},
                "source": {**entry["source"], "readout": str(val_readout if claimed else select_readout),
                           "speed": {**entry["source"]["speed"], "readout": str(speed)}}}

    airport_b = listed(roots["B"], "stage_b", prior_files, ["fixture_set", "fixture_val"], prior_entry)
    write(prior / f"val_read_{prior_files.CLAIM_READER}.json", {"out": str(val_readout)})
    write(prior / "val_read_prior_validation.json", {"out": str(validation)})
    # stage C: a campaign of one round, its checks, started from round 1 of another campaign (D162) on the same windows
    source = outputs / "post" / "source"
    write(source / "campaign.json", {"inputs": {"settings": {"start": None, "select_seed": 1337, "select_per_airport": 200}}})
    write(source / "round_0" / "round.json", {
        "round": 0, "selection_readout": {"KXXX": {"windows": 1, "reward_mean": 0.5, "outcomes": {"landed": 1}}}})
    write(source / "round_1" / "round.json", {
        "round": 1, "selection_readout": {"KXXX": {"windows": 1, "reward_mean": 0.0, "outcomes": {"lost_separation": 1},
                                                   "faulty_steps": SEALED}}})
    post = outputs / "post" / "campaign"
    write(post / "campaign.json", {"started_utc": "2026-10-06T00:00:00Z", "checks": {"labeller": {"flights": 3}},
                                   "inputs": {"settings": {"start": {"campaign": str(source), "round": 1,
                                                                     "checkpoint_sha256": SEALED},
                                                           "select_seed": 1337, "select_per_airport": 200}}})
    write(post / "round_0" / "round.json", {
        "round": 0, "speaking": {"windows": 5, "reward_sum": 4.0, "outcomes": {"landed": 4, "lost_separation": 1}},
        "selection_readout": {"KXXX": {"windows": 1, "reward_mean": 1.0, "outcomes": {"landed": 1}, "faulty_steps": SEALED}}})
    c_speed = write(outputs / "speed" / "round" / "speed.json", {
        "model": {"stage": "C", "campaign": str(post), "round": "0", "split": "select"}, "warmupRows": 20, "smoke": False,
        "settings": []}).parent
    airport_c = listed(roots["C"], "stage_c", post_files, ["fixture-windows"], lambda e: {
        **e, "model": {**e["model"], "campaign": str(post)},
        "source": {**e["source"], "speed": {**e["source"]["speed"], "readout": str(c_speed)}}})
    return {"results": {stage: TrainingResults(roots[stage], outputs) for stage in "ABC"},
            "airports": dict(zip("ABC", (airport_a, airport_b, airport_c))), "outputs": outputs, "roots": roots,
            "post": post, "source": source}


def answered(world, stage, set_id):
    status, answer = world["results"][stage].answer(stage, world["airports"][stage], set_id)
    assert status == 200, answer
    assert str(SEALED) not in json.dumps(answer), "a sealed or unnamed field reached the answer"
    return answer["sections"]


def test_stage_a_answers_the_labelling_of_train_and_select_and_the_replays_of_its_splits(world):
    sections = answered(world, "A", "fixture_set")
    assert sorted(sections["labelling"]["splits"]) == ["select", "train"]           # never val (D85)
    replays = sections["closedLoop"]["replays"]
    assert sorted(replays) == [f"{split} {d}" for split in ("select", "train") for d in (2, 4, 8)]   # its splits × Δs
    assert replays["train 4"]["groups"]["own dynamics"]["KXXX"]["landed"] == 0.75


def test_stage_b_answers_its_results_and_the_val_days_only_for_the_claimed_set(world):
    sections = answered(world, "B", "fixture_set")
    assert sections["freeGeneration"]["sides"]["inside"]["KXXX"]["vectored"]["goArounds"] == 1
    assert sections["training"]["bestEpoch"] == 2 and sections["choice"]["configuration"]["chosen"] == "C"
    assert sections["choice"]["variant"]["scores"] == {"full": 1.36, "constants": 1.37}
    assert sections["speed"]["settings"][0]["rowMs"]["max"] == pytest.approx(8.0)
    assert sections["validation"] == {"ok": False, "problem": sections["validation"]["problem"]}
    assert "claimed validation readout" in sections["validation"]["problem"]
    val = answered(world, "B", "fixture_val")
    assert val["freeGeneration"]["split"] == "val" and val["validation"]["teacherForced"]["lossPerStep"] == 1.08


def test_a_val_readout_without_the_written_claim_is_not_read(world):
    claim = world["outputs"] / "prior" / "campaign" / "base" / "run" / f"val_read_{prior_files.CLAIM_READER}.json"
    claim.write_text(json.dumps({"out": "another"}))
    sections = answered(world, "B", "fixture_val")
    assert sections["freeGeneration"]["ok"] is False and "D109" in sections["freeGeneration"]["problem"]
    assert sections["validation"]["ok"] is False                     # the validation rests on the same written claim


def test_stage_c_answers_its_rounds_and_checks(world):
    sections = answered(world, "C", "fixture-windows")
    assert [r["round"] for r in sections["rounds"]["rounds"]] == [0]
    assert sections["checks"]["checks"] == {"labeller": {"flights": 3}}
    assert sections["speed"]["model"].endswith("campaign round 0")
    # the start's readout: the source round's own, on the same select windows (frontend §4.3)
    start = sections["rounds"]["start"]
    assert start == {"campaign": str(world["source"]), "round": 1, "why": None,
                     "selection": {"KXXX": {"windows": 1, "rewardMean": 0.0, "outcomes": {"lost_separation": 1}}}}


def test_stage_c_answers_no_start_readout_from_the_base_and_none_on_other_windows(world):
    record = json.loads((world["post"] / "campaign.json").read_text())
    settings = record["inputs"]["settings"]
    (world["post"] / "campaign.json").write_text(json.dumps({**record, "inputs": {"settings": {**settings, "start": None}}}))
    assert answered(world, "C", "fixture-windows")["rounds"]["start"] is None
    for other in ({"select_seed": 2024}, {"select_per_airport": 100}):
        (world["post"] / "campaign.json").write_text(json.dumps({**record, "inputs": {"settings": {**settings, **other}}}))
        start = answered(world, "C", "fixture-windows")["rounds"]["start"]
        assert start["selection"] is None and "other select windows" in start["why"], other


def test_a_file_elsewhere_or_missing_is_answered_by_name(world, tmp_path):
    (world["post"] / "campaign.json").unlink()
    sections = answered(world, "C", "fixture-windows")
    assert sections["rounds"]["ok"] is False and "does not exist" in sections["rounds"]["problem"]
    elsewhere = TrainingResults(world["roots"]["A"], tmp_path / "another_outputs")
    status, answer = elsewhere.answer("A", world["airports"]["A"], "fixture_set")
    assert status == 200 and "not under 4dTrajectory/outputs/" in answer["sections"]["labelling"]["problem"]


def test_the_route_refuses_a_bad_stage_and_names_a_set_not_listed(world):
    results = world["results"]["A"]
    assert results.answer("D", "KXXX", "x")[0] == 400
    status, answer = results.answer("A", world["airports"]["A"], "no_such_set")
    assert status == 404 and "no_such_set" in answer["error"]
    app = AeroVizBackendApp(training_results=results)
    assert app.handle_get("/training/results?stage=A&airport=KXXX")[0] == 400        # no set asked
    assert app.handle_get(f"/training/results?stage=A&airport={world['airports']['A']}&set=fixture_set")[0] == 200


#: The frontend's fixture of the route's answers (`aeroviz-4d/src/data/__tests__/fixtures/training_results/answers.json`):
#: written by this test (`AEROVIZ_WRITE_FIXTURES=1`), read by the frontend's reader.
RESULTS_FIXTURE = FIXTURES / "training_results" / "answers.json"


def test_the_frontend_fixture_of_the_answers_is_what_the_route_answers(world):
    """Each stage's answer on the fixture sets, its outputs' paths made fixed names; B's for the set and the claimed val
    set; C's, and C's with its campaign's file missing."""
    import os

    answers = {}
    for name, stage, set_id in (("A", "A", "fixture_set"), ("B", "B", "fixture_set"), ("Bval", "B", "fixture_val"),
                                ("C", "C", "fixture-windows")):
        answers[name] = world["results"][stage].answer(stage, world["airports"][stage], set_id)[1]
    (world["post"] / "campaign.json").unlink()
    answers["Cmissing"] = world["results"]["C"].answer("C", world["airports"]["C"], "fixture-windows")[1]
    text = json.dumps(answers, indent=1, sort_keys=True).replace(str(world["outputs"]), "fixture/outputs") + "\n"
    if os.environ.get("AEROVIZ_WRITE_FIXTURES") == "1":
        RESULTS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_FIXTURE.write_text(text, encoding="utf-8")
    assert RESULTS_FIXTURE.read_text(encoding="utf-8") == text, (
        f"{RESULTS_FIXTURE} is not what the route answers now: AEROVIZ_WRITE_FIXTURES=1 writes it again")


def test_a_bad_airport_an_index_or_a_file_of_another_format_and_a_path_out_of_the_outputs_are_refused_by_name(world, tmp_path):
    results = world["results"]["A"]
    for airport in ("../KXXX", "/tmp", "kxxx", "KXXXX"):
        assert results.answer("A", airport, "fixture_set")[0] == 400, airport
    index = world["roots"]["A"] / world["airports"]["A"] / "training" / stage_a_files.INDEX_FILE
    payload = json.loads(index.read_text())
    index.write_text(json.dumps({**payload, "schema": "aeroviz-training-index-v0"}))
    status, answer = results.answer("A", world["airports"]["A"], "fixture_set")
    assert status == 409 and "aeroviz-training-index-v0" in answer["error"]
    index.write_text(json.dumps(payload))
    readout = world["outputs"] / "instruction_language" / "v" / "readout.json"
    readout.write_text(json.dumps({"train": {}}))                                   # a readout of another format
    sections = answered(world, "A", "fixture_set")
    assert sections["labelling"]["ok"] is False and "another format" in sections["labelling"]["problem"]
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "readout.json").write_text("{}")
    instructions = world["outputs"] / "instruction_language" / "v"
    shutil.rmtree(instructions)
    instructions.symlink_to(outside)                                                # a link out of the outputs
    assert "not under" in answered(world, "A", "fixture_set")["labelling"]["problem"]
    assert "not under" in results.section(lambda: results.read(f"{world['outputs']}/../outside/readout.json"))["problem"]
