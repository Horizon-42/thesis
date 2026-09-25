"""The prior's rows in a closed loop: the speaker's inputs are the rows a closed-loop sentence trains on
(`data.chain_record`), the loss counts only the asked columns, and the model encodes them row by row (`Prior.extend`)
as it would all at once."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from ts_transformer.experiments.prior_free_generation import speak_and_fly
from ts_transformer.instructions.words import ALTITUDE, SPEED, Words
from ts_transformer.prior import data as prior_data
from ts_transformer.prior.data import Split, chain_record, column_classes
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.prior.train import column_nll, to_batch
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec as spec
from ts_transformer.tests.test_autopilot import DOWNWIND_BASE_FINAL, _params, _physics

CPU = torch.device("cpu")


def _repeated(table, index: torch.Tensor):
    """A per-flight table of the executor's (`FlightInputs`, `Runways`, `AirportCharts`) at the flights ``index``."""
    return type(table)(**{field.name: getattr(table, field.name)[index] for field in fields(table)})


def _model(words, slots=1, variant="no-context"):
    torch.manual_seed(0)
    return Prior(PriorConfig(classes=column_classes(words, slots), airports=("KXXX",), candidate_slots=slots,
                             variant=variant, d_model=32, layers=2, heads=4, feedforward=64, dropout=0.0),
                 torch.as_tensor(prior_data.candidate_table({"KXXX": instruction_airport()}, ("KXXX",), slots))).eval()


def _flight():
    one, geometry = spec(), instruction_airport()
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    start = replace(signals, **{name: getattr(signals, name)[N_LOOK:] for name in
                                ("time_s", "e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps",
                                 "vertical_rate_mps")})
    return one, geometry, signals, _physics(start, geometry)


def test_a_chain_s_training_rows_are_the_rows_the_prior_read_on_it():
    one, geometry, signals, (inputs, runways, charts, approach) = _flight()
    words, params = Words(one), _params()
    model = _model(words)
    _, said, _, speaker = speak_and_fly(model, [signals], [geometry], inputs, runways, charts, approach, [60.0], words,
                                        params, None, generator=torch.Generator().manual_seed(2), temperature=1.0)
    said = said[0]
    rows = N_LOOK + len(said)
    classes = np.ones((len(said), 6), dtype=np.int64)
    asked = np.ones((len(said), 6), dtype=bool)
    asked[3, ALTITUDE] = False
    flight = chain_record(signals, speaker.e[0, :rows], speaker.n[0, :rows], speaker.h[0, :rows], said, classes, asked,
                          geometry, None, 0, 0, one.step_s)
    assert np.allclose(flight.features, speaker.features[0, 0, :rows].numpy(), atol=1e-6)
    assert np.allclose(flight.relative, speaker.relative[0, 0, :rows].numpy(), atol=1e-6)
    assert np.array_equal(flight.in_force, speaker.in_force[0, 0, :rows].numpy())
    assert np.allclose(flight.since, speaker.since[0, 0, :rows].numpy(), atol=1e-6)
    assert (flight.targets[:N_LOOK] == 0).all() and (flight.targets[N_LOOK:] == 1).all()
    assert not flight.asked[:N_LOOK].any() and not flight.asked[N_LOOK + 3, ALTITUDE]
    assert flight.asked[N_LOOK:].sum() == 6 * len(said) - 1


def test_the_loss_counts_only_the_asked_columns():
    one, geometry, signals, (inputs, runways, charts, approach) = _flight()
    words, params = Words(one), _params()
    _, said, _, speaker = speak_and_fly(_model(words), [signals], [geometry], inputs, runways, charts, approach, [40.0],
                                        words, params, None, generator=torch.Generator().manual_seed(2),
                                        temperature=1.0)
    said = said[0]
    rows = N_LOOK + len(said)
    asked = np.ones((len(said), 6), dtype=bool)
    asked[:, SPEED] = False

    def flight(speed_class: int):
        classes = np.ones((len(said), 6), dtype=np.int64)
        classes[:, SPEED] = speed_class
        return chain_record(signals, speaker.e[0, :rows], speaker.n[0, :rows], speaker.h[0, :rows], said, classes,
                            asked, geometry, None, 0, 0, one.step_s)

    table = prior_data.candidate_table({"KXXX": geometry}, ("KXXX",), 1)
    splits = [Split([flight(c)], ("KXXX",), table, (("09",),), ((90.0,),), column_classes(words, 1), "no-context")
              for c in (0, 3)]
    model = _model(words)
    batches = [to_batch(s, [0], CPU) for s in splits]
    nll = [column_nll(model(b["features"], b["relative"], b["static"], b["in_force"], b["since"], b["airport"],
                            b["present"], b["edges"], b["targets"]), b["targets"], b["present"], b["asked"])
           for b in batches]
    assert float(nll[0][SPEED].detach()) == 0.0 and torch.allclose(nll[0][:SPEED], nll[1][:SPEED])


def test_rows_encoded_one_at_a_time_are_the_rows_encoded_together():
    """`Prior.extend` (a speaker's row by row, each layer's keys and values kept) gives what `encode` gives, absent
    rows included."""
    from ts_transformer.prior.model import self_edges

    words = Words(spec())
    torch.manual_seed(3)
    model = _model(words, slots=2, variant="full")
    batch, rows = 3, N_LOOK + 6
    features = torch.randn(batch, 1, rows, len(prior_data.STEP_FEATURES))
    relative = torch.randn(batch, 1, rows, 2, len(prior_data.VARIANTS["full"].relative_features))
    in_force = torch.randint(0, 3, (batch, 1, rows, 6))
    since = torch.rand(batch, 1, rows, 6)
    airport, static = torch.zeros(batch, dtype=torch.long), torch.zeros(batch, 1, 0)
    present = torch.ones(batch, 1, rows, dtype=torch.bool)
    present[1, 0, 3] = False
    whole, tokens, valid = model.encode(features, relative, static, in_force, since, airport, present,
                                        self_edges(batch, 1, rows, CPU))
    past, pieces = model.no_past(batch, rows), []
    for low, high in ((0, N_LOOK + 1), *((t, t + 1) for t in range(N_LOOK + 1, rows))):
        h, piece_tokens, _, past = model.extend(features[:, :, low:high], relative[:, :, low:high], static,
                                                in_force[:, :, low:high], since[:, :, low:high], airport,
                                                present[:, :, low:high], self_edges(batch, 1, high - low, CPU), past)
        pieces.append(h)
        assert torch.allclose(piece_tokens, tokens[:, :, low:high])
    assert torch.allclose(torch.cat(pieces, dim=2), whole, atol=1e-5)
    with pytest.raises(ValueError, match="the past holds"):
        model.extend(features[:, :, :1], relative[:, :, :1], static, in_force[:, :, :1], since[:, :, :1], airport,
                     present[:, :, :1], self_edges(batch, 1, 1, CPU), past)
