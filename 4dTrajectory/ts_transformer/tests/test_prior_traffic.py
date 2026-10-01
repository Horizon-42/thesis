"""The traffic attention (`prior.model.with_traffic`, multi-aircraft design §2.5, §6.2): a single-aircraft prior that
gains it says, until it learns, what it said alone — bit for bit — and reads the other aircraft once it has weights."""

import pytest
import torch

from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.instructions.words import HEADING
from ts_transformer.prior.model import Prior, with_traffic
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.test_prior import ROWS, _inputs, _model, _run

NAMES = ("features", "relative", "static", "in_force", "since", "airport", "present", "rows", "edges", "targets")


def _scene(model, aircraft=3, enters=4):
    """Random inputs for ``aircraft`` aircraft, the last entering at step ``enters`` (its own rows from 0 there), and
    random traffic edge features with "self" on the diagonal; in the model's dtype."""
    batch = _inputs(model, rows=ROWS + enters, aircraft=aircraft)
    generator = torch.Generator().manual_seed(5)
    batch["features"] = torch.randn(batch["features"].shape, generator=generator)
    batch["present"][0, -1, :enters] = False
    batch["rows"] = batch["rows"].clone()
    batch["rows"][0, -1] = torch.clamp(torch.arange(ROWS + enters) - enters, min=0)
    steps = ROWS + enters
    edges = torch.randn(1, steps, aircraft, aircraft, len(EDGE_FEATURES), generator=generator)
    edges[..., 0] = torch.eye(aircraft)
    batch["edges"] = edges
    dtype = next(model.parameters()).dtype
    return {name: value.to(dtype) if value.is_floating_point() else value for name, value in batch.items()}


def _alone(batch, a, start):
    """Aircraft ``a`` of ``batch`` alone, from step ``start`` (its row 0), with only its "self" edge."""
    out = {name: batch[name][:, a: a + 1, start:] for name in ("features", "relative", "in_force", "since", "present",
                                                                "rows", "targets")}
    out.update(static=batch["static"][:, a: a + 1], airport=batch["airport"],
               edges=batch["edges"][:, start:, a: a + 1, a: a + 1, :1])
    return out


def test_a_traffic_attention_at_zero_says_what_the_single_prior_says_alone():
    """It adds exactly 0; the rest is the single prior's arithmetic, which a batch holding several aircraft sums in
    another order than one holding one: equal to rounding (1e-15 in double — the single prior itself differs as much
    between the two layouts)."""
    single = _model().double()
    traffic = with_traffic(single, EDGE_FEATURES).eval()
    added = []
    for layer in traffic.layers:
        layer.traffic.register_forward_hook(lambda module, inputs, output: added.append(output))
    batch = _scene(traffic)
    together = _run(traffic, batch)
    assert len(added) == len(traffic.layers) and all(torch.equal(out, torch.zeros_like(out)) for out in added)
    for a, start in ((0, 0), (1, 0), (2, 4)):
        alone = _run(single, _alone(batch, a, start))
        for x, y in zip(together, alone):
            assert torch.equal(torch.isinf(x[0, a, start:]), torch.isinf(y[0, 0]))
            finite = torch.isfinite(y[0, 0])
            torch.testing.assert_close(x[0, a, start:][finite], y[0, 0][finite], rtol=0, atol=1e-12)


def test_the_traffic_attention_reads_the_others_once_it_has_weights_and_never_an_aircraft_alone():
    single = _model().double()
    traffic = with_traffic(single, EDGE_FEATURES).eval()
    generator = torch.Generator().manual_seed(7)
    for layer in traffic.layers:
        layer.traffic.out.weight.data = 0.1 * torch.randn(layer.traffic.out.weight.shape, generator=generator,
                                                         dtype=torch.float64)
    batch = _scene(traffic)
    moved = _run(traffic, batch)
    alone = _run(single, _alone(batch, 0, 0))
    assert not torch.allclose(moved[HEADING][0, 0, N_LOOK:], alone[HEADING][0, 0, N_LOOK:])
    # an aircraft with no other present at any step reads nothing: its answers are its own — whatever the traffic
    # attention has learned (every weight moved, not only the output layer's)
    for layer in traffic.layers:
        for name, parameter in layer.traffic.named_parameters():
            if name != "out.weight":
                parameter.data += 0.05 * torch.randn(parameter.shape, generator=generator, dtype=torch.float64)
    moved = _run(traffic, batch)
    lonely = {name: batch[name][:, :1] if name not in ("airport", "edges") else batch[name] for name in NAMES}
    lonely["edges"] = batch["edges"][:, :, :1, :1]
    for x, y in zip(_run(traffic, lonely), alone):
        assert torch.equal(x, y)
    # no NaN anywhere a step has no other aircraft, forward or backward
    traffic.train()
    out = traffic(*(batch[name] for name in NAMES))
    loss = sum(logit[torch.isfinite(logit)].sum() for logit in out)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in traffic.parameters() if p.grad is not None)


def test_the_zero_output_layer_learns_first():
    """At zero only the traffic attention's output layer has a gradient (what it reads times the gradient above), the
    layers inside it none yet — the zero start of a gated adapter."""
    traffic = with_traffic(_model().double(), EDGE_FEATURES).train()
    batch = _scene(traffic)
    out = traffic(*(batch[name] for name in NAMES))
    sum(logit[torch.isfinite(logit)].sum() for logit in out).backward()
    first = traffic.layers[0].traffic
    assert first.out.weight.grad.abs().sum() > 0
    assert first.qkv.weight.grad.abs().sum() == 0 and first.value[0].weight.grad.abs().sum() == 0


def test_row_by_row_is_the_whole_scene_with_a_traffic_attention():
    traffic = with_traffic(_model().double(), EDGE_FEATURES).eval()
    for layer in traffic.layers:
        layer.traffic.out.weight.data.normal_(0.0, 0.1)
    batch = _scene(traffic)
    steps = batch["present"].shape[2]
    with torch.no_grad():
        whole = traffic.encode(*(batch[name] for name in NAMES[:-1]))[0]
        past, rows = traffic.no_past(3, steps), []
        for t in range(steps):
            step = {name: batch[name][:, :, t: t + 1] for name in ("features", "relative", "in_force", "since",
                                                                  "present", "rows")}
            h, _, _, past = traffic.extend(step["features"], step["relative"], batch["static"], step["in_force"],
                                           step["since"], batch["airport"], step["present"], step["rows"],
                                           batch["edges"][:, t: t + 1], past)
            rows.append(h)
    present = batch["present"][..., None]
    torch.testing.assert_close(torch.cat(rows, dim=2) * present, whole * present, rtol=0, atol=1e-10)


def test_blocks_of_steps_are_the_whole_encoding_values_and_gradients():
    """`Prior.encode`'s ``pairs_per_block``: the step-wise part a block of steps at a time — one step, blocks that do not
    divide the steps, one block — with and without the layers recomputed: the whole encoding's values and gradients to
    rounding (1e-12 in double; a block's products are other shapes)."""
    traffic = with_traffic(_model().double(), EDGE_FEATURES).eval()
    generator = torch.Generator().manual_seed(9)
    for layer in traffic.layers:
        for parameter in layer.traffic.parameters():
            parameter.data += 0.1 * torch.randn(parameter.shape, generator=generator, dtype=torch.float64)
    batch = _scene(traffic)
    steps = batch["present"].shape[2]
    assert steps % 5 != 0

    def encoded(**options):
        traffic.zero_grad(set_to_none=True)
        h = traffic.encode(*(batch[name] for name in NAMES[:-1]), **options)[0]
        (h * batch["present"][..., None]).square().sum().backward()
        return h.detach(), {name: p.grad.clone() for name, p in traffic.named_parameters() if p.grad is not None}

    whole, gradients = encoded()
    for pairs in (1, 5 * 9, 10 ** 6):                            # 1 step a block; 5 (3 aircraft: 9 pairs a step); one
        for checkpoint in (False, True):
            h, grads = encoded(checkpoint=checkpoint, pairs_per_block=pairs)
            torch.testing.assert_close(h, whole, rtol=0, atol=1e-12)
            assert grads.keys() == gradients.keys() and any(".traffic." in name for name in grads)
            for name, g in grads.items():
                torch.testing.assert_close(g, gradients[name], rtol=0, atol=1e-12)


def test_step_blocks_cover_every_step_in_order_at_most_the_pairs_given():
    from ts_transformer.prior.model import step_blocks

    assert step_blocks(1, 3, 23, 5 * 9) == [slice(0, 5), slice(5, 10), slice(10, 15), slice(15, 20), slice(20, 23)]
    assert step_blocks(2, 3, 4, 1) == [slice(k, k + 1) for k in range(4)]           # one step at least
    assert step_blocks(1, 3, 23, 10 ** 6) == [slice(0, 23)]


def test_every_weight_of_the_single_prior_is_kept_and_misfits_are_refused():
    single = _model()
    traffic = with_traffic(single, EDGE_FEATURES)
    kept = traffic.state_dict()
    assert all(torch.equal(value, kept[name]) for name, value in single.state_dict().items())
    assert {name for name in kept if name not in single.state_dict()} == {
        name for name in kept if ".traffic." in name} != set()
    assert traffic.traffic_features == EDGE_FEATURES and traffic.edge_features == single.edge_features
    with pytest.raises(ValueError, match="has a traffic attention already"):
        with_traffic(traffic, EDGE_FEATURES)
    with pytest.raises(ValueError, match="do not start with"):
        Prior(single.config, single.candidates, ("self",), EDGE_FEATURES[1:])
    with pytest.raises(ValueError, match="single-aircraft prior"):
        with_traffic(Prior(single.config, single.candidates, EDGE_FEATURES), EDGE_FEATURES)
    with pytest.raises(ValueError, match="single-aircraft prior"):
        Prior(single.config, single.candidates, EDGE_FEATURES, EDGE_FEATURES)
    assert all(layer.traffic.out.bias is None for layer in traffic.layers)
