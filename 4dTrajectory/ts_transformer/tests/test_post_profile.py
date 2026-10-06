"""Stage C, C8: the profile of a round (post-training §8 C8) — on A26's one-flight synthetic artefact."""

from __future__ import annotations

from ts_transformer.experiments import post_profile
from ts_transformer.experiments.post_profile import APPROXIMATION, PARTS, memory, part_times, profile
from ts_transformer.tests.test_post_train import _context, _settings, select_is_train, short_round  # noqa: F401
from ts_transformer.tests.test_post_window_loop import CPU, setup  # noqa: F401


def test_a_profile_times_each_part_of_a_batch_and_the_round(setup, tmp_path, monkeypatch, select_is_train):  # noqa: F811
    window = short_round(monkeypatch, setup)
    monkeypatch.setattr(post_profile, "draw_round", lambda context, per_kind, rng: ([window], {}))
    record = profile(_context(setup), _settings(rounds=1, continuations=2), tmp_path / "profile")
    batch = record["one_batch"]
    assert set(batch["parts_s"]) == set(PARTS) and batch["approximation"] == APPROXIMATION
    for part in ("prior_step", "executor_step", "speaker_masks", "edge_features", "separation_scene"):
        assert 0.0 < batch["parts_s"][part] <= batch["wall_s"], part
    assert record["batches"] == [1] and batch["windows"] == 1 and batch["groups"] == 1
    rounds = record["round"]
    assert rounds["speaking"]["windows"] == 1 and rounds["selection_windows"] == 1
    assert min(rounds["speaking_s"], rounds["pass_s"], rounds["selection_readout_s"]) >= 0.0
    assert (tmp_path / "profile" / "round_0").is_dir() and rounds["groups_bytes"] > 0
    assert {"host_peak_rss_gib", "host_available_gib"} <= set(record["memory_before"])
    assert "gpu_peak_allocated_gib" not in memory(CPU)


def test_a_part_is_the_largest_of_its_functions_never_their_sum():
    class Stats:
        stats = {("/x/ts_transformer/prior/model.py", 1, "extend"): (1, 1, 0.1, 2.0, {}),
                 ("/x/ts_transformer/prior/model.py", 9, "extend"): (1, 1, 0.1, 3.0, {}),
                 ("/x/other/prior/model.py", 1, "extend"): (1, 1, 0.1, 9.0, {})}
    times = part_times(Stats())
    assert times["prior_step"] == 3.0 and times["executor_step"] == 0.0
