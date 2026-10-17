"""The loss of the post-training (post-training §2 item 5, §8 C7; D36, D76): one update and one pass.

    loss = the clipped-ratio surrogate (ε = `CLIP`) on the words said, with the advantage of each row
         + `KL_WEIGHT` × the pull to the base (the KL on the sampled words, under the masks they were said under)
         + `DATA_WEIGHT` × the teacher-forced data term (single-aircraft samples of the closed-loop sentences in the
           base's selection ``landed``, D36, D76)

A caller may give the data term another weight (``data_weight``; stage D's `MultiSettings.data_weight`, the user,
2026-10-10, R2 of stage D's plan); `DATA_WEIGHT` is the default.

**The words said** are scored under the speaker's records of the permitted words (`prior.train.masked_log_probability`,
prior §7 item 4, D96): the masked distribution that the speaker drew from, with the traffic module's input. A word that
a record blocks has no probability.

**The ratio** of a word is its probability under the model over its probability under the model that spoke it: a frozen
copy of the model at the start of the pass (`PassStart`), every sample of a pass spoken by it (one pass over the samples
of a round). At the parameters that spoke the ratio is 1, within the float tolerance of post-training §6.4. Per word
(each column of each row is one word): ``−min(r·A, clip(r, 1 − ε, 1 + ε)·A)``, A the advantage of its row — the
recipe of the archived post-training (`archive/two_tier_v3_2026_10/prior/train.py` `flight_surrogate`).

**The pull to the base** is the base's log-probability b and the model's p of each word said, under the same records:
per word ``exp(b − p) − (b − p) − 1`` (≥ 0, an estimate of the KL from the samples; the archived `flight_kl`). The base
has no traffic module.

**Which words count.** A sample's rows carry an advantage and a mark of the rows it applies to (``counted``: in a branch
group, the rows after the branch point, §2 item 9; branch training C6 makes them). The surrogate and the pull read the
counted rows only; a batch's sum over its counted words is divided by its counted rows, so every counted row weighs the
same (D115; a continuation from a late branch point weighs no more for each of its words than a whole first sentence).

**Dropout** (the user, 2026-10-05): off in the ratio and the pull — the words are scored in eval mode, the distribution
they were drawn from — and on in the data term (training mode).

**One update in pieces** (`update_step`, the user, 2026-10-06, after C10's round 4 ran out of the GPU's memory): an
update's samples are given as pieces (a branch group each); each piece's surrogate and pull are summed over its counted
words and divided by the counted rows of the WHOLE update, and its backward taken at once, so the gradient is the
whole update's (to float rounding: the sums are added in another order) while the memory is a piece's, whatever the
longest group of the round. The data term is one piece (its dropout would draw other masks if cut). `update_loss`, the
whole update at once, is the readable reference `update_step` is checked against. Of window samples, a piece is one
window (`Samples.pieces`, MC9: a group of whole windows did not fit the GPU).

**Gradient-norm clipping** (D168, `one_pass`'s ``clip_norm``): each update's gradient, over the optimizer's
parameters, is scaled to that norm where its norm is above it, before the optimizer's step
(`torch.nn.utils.clip_grads_with_norm_`, as `torch.nn.utils.clip_grad_norm_`). Every update's norm before the clip is
kept (`LossParts.grad_norm`), clipped or not; with ``clip_norm`` None no gradient is touched (the pass before D168).

**The dual clip** (multi-aircraft control D219 (2), Ye et al. 2020; `update_step`'s ``dual_clip`` c > 1): of a word of
a negative advantage, the loss is at most −c·A: ``min(−min(r·A, clip(r, 1 − ε, 1 + ε)·A), −c·A)`` where A < 0. A word
whose ratio is above c then adds a constant and no gradient (without it, its loss r·|A| and its gradient grow with r,
unbounded); every other word is as before. With ``dual_clip`` None no word is touched (the loss before D219).

**An update's extreme words** (D219 (4), `Extremes`, `update_step`; where a large gradient comes from): over its counted
words, the largest ratio, the largest weight of a word in the surrogate's gradient (r·|A| where the word's gradient flows;
0 where a clip holds it), the largest |A| of a counted row, and the words the dual clip holds. Read under no gradient;
the loss and its gradient do not change.

**Passes per round** (D169, `passes`): E passes over a round's samples, each drawn in its own order by the caller, the
ratio of every pass against the same `PassStart` (the model that spoke the samples, at the round's start), the clip of
the ratio unchanged; one pass is `one_pass`.

**Window samples** (multi-aircraft control D153, D212; `post.branches.samples`): a sample is a window, several rows of
the batch of which only one counts rows; every window counts some. The model and the model that spoke read every row
(each row reads the others of its window); the base reads only the rows of the sentences with counted rows (it has no
module that reads another row), the others get the model's own log-probability (no pull). Samples alone (stage C's,
version 1's) count rows in every sentence: the base reads them all, as before.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from typing import Callable, Iterable, Sequence

import torch

from ts_transformer.post.traffic_attention import Traffic
from ts_transformer.prior.batch import RowTensors
from ts_transformer.prior.model import Prior
from ts_transformer.prior.speaker import Permitted
from ts_transformer.prior.train import batch_nll, masked_log_probability

#: Post-training §2 item 5.
CLIP = 0.2
KL_WEIGHT = 0.04
DATA_WEIGHT = 1.0


@dataclass(frozen=True)
class Samples:
    """A batch of spoken samples: their rows (``rows.targets`` the words said, ``rows.asked`` the rows said), the
    speaker's records of them, the traffic module's input, each row's advantage (``advantage`` [B, R] float) and the
    rows it applies to (``counted`` [B, R] bool, within ``rows.asked``)."""

    rows: RowTensors
    permitted: Permitted
    traffic: Traffic
    advantage: torch.Tensor
    counted: torch.Tensor

    def select(self, indices: Sequence[int]) -> Samples:
        """The samples ``indices``, in that order (each tensor's first axis), padded as here; of window samples, whole
        windows only (`Traffic.select`)."""
        chosen = torch.as_tensor(list(indices), device=self.advantage.device)
        return Samples(RowTensors(*(tensor[chosen] for tensor in self.rows)), self.permitted.select(indices),
                       self.traffic.select(chosen), self.advantage[chosen],
                       self.counted[chosen])

    def pieces(self) -> list[Samples]:
        """The pieces of these samples for `update_step` (module docstring, "One update in pieces"): of window samples,
        one piece for each window (a window reads no row of another window, so the gradient does not change, and the
        memory is a window's); samples alone, one piece."""
        if self.traffic.windows is None:
            return [self]
        return [self.select(window[window >= 0].tolist()) for window in self.traffic.windows]

    def __post_init__(self) -> None:
        shape = tuple(self.rows.asked.shape)
        if tuple(self.advantage.shape) != shape or tuple(self.counted.shape) != shape or self.counted.dtype != torch.bool:
            raise ValueError(f"the advantage and the counted rows are [B, R] = {shape} beside the rows")
        if (self.counted & ~self.rows.asked).any():
            raise ValueError("a counted row is a row the speaker said")
        counts = self.counted.any(dim=1)
        if self.traffic.windows is not None:              # window samples: every window counts some rows
            windows = self.traffic.windows
            real = windows >= 0
            if not (counts[windows.clamp(min=0)] & real).any(dim=1).all():
                raise ValueError("every window sample counts some rows: one without carries no gradient")
            in_window = torch.zeros_like(counts)
            in_window[windows[real]] = True
            counts = counts | in_window                   # the other rows of a window count none
        if not counts.all():
            raise ValueError("every sample counts some rows: a sample without one carries no gradient")


@dataclass(frozen=True)
class Extremes:
    """An update's extreme words (module docstring; D219 (4)): over its counted words, the largest ratio, the largest
    weight in the surrogate's gradient (r·|A| where the word's gradient flows, 0 where a clip holds it), the largest |A|
    of a counted row, and the count of words the dual clip holds (0 without it)."""

    ratio_max: float
    weight_max: float
    advantage_max: float
    dual_clipped: int

    def joined(self, other: Extremes) -> Extremes:
        """The extremes of the two sets of words together."""
        return Extremes(max(self.ratio_max, other.ratio_max), max(self.weight_max, other.weight_max),
                        max(self.advantage_max, other.advantage_max), self.dual_clipped + other.dual_clipped)


@dataclass(frozen=True)
class LossParts:
    """One update's loss and its parts (each a scalar tensor), the words counted and those whose ratio left the clip;
    after `one_pass`'s step, the norm of its gradient before the clip of D168 and whether it was clipped (None and False
    from a step alone); its extreme words (`update_step`; None from `update_loss` and `landed_step`)."""

    loss: torch.Tensor
    surrogate: torch.Tensor
    kl: torch.Tensor
    data: torch.Tensor
    words: int
    clipped: int
    grad_norm: torch.Tensor | None = None
    grad_clipped: bool = False
    extremes: Extremes | None = None


class PassStart:
    """A frozen copy of the model at the start of a pass: the model that spoke the pass's samples (module docstring)."""

    def __init__(self, model: Prior) -> None:
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)


def _counted_sum(values: torch.Tensor, counted: torch.Tensor) -> torch.Tensor:
    """The batch's sum over its counted words ``values`` [B, R, 5] (a row not counted adds nothing)."""
    return torch.where(counted[..., None], values, torch.zeros_like(values)).sum()


def _per_row(values: torch.Tensor, counted: torch.Tensor) -> torch.Tensor:
    """The batch's sum over its counted words ``values`` [B, R, 5] divided by its counted rows: every counted row weighs
    the same (D115)."""
    return _counted_sum(values, counted) / counted.sum().to(values.dtype)


def _surrogate_words(log_p: torch.Tensor, start_log_p: torch.Tensor, advantage: torch.Tensor, counted: torch.Tensor,
                     clip: float, dual_clip: float | None = None) -> tuple[torch.Tensor, int, int, Extremes]:
    """Each word's clipped-ratio loss [B, R, 5] (with ``dual_clip``, the dual clip's), the words counted, those clipped
    and their extremes (module docstring)."""
    ratio = torch.exp(log_p - start_log_p)
    a = advantage[..., None]
    unclipped, clamped = ratio * a, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * a
    loss = -torch.minimum(unclipped, clamped)
    flows = unclipped <= clamped                          # the term the minimum takes carries the word's gradient
    held = torch.zeros_like(flows)
    if dual_clip is not None:
        held = (a < 0) & (loss > -dual_clip * a)
        loss = torch.where(held, -dual_clip * a, loss)
        flows = flows & ~held
    words = counted[..., None].expand_as(ratio)
    with torch.no_grad():
        weight = torch.where(flows, ratio * a.abs(), torch.zeros_like(ratio))
        extremes = Extremes(float(ratio[words].max()), float(weight[words].max()), float(advantage[counted].abs().max()),
                            int(held[words].sum()))
    return loss, int(words.sum()), int(((ratio - 1.0).abs() > clip)[words].sum()), extremes


def _pull_words(log_p: torch.Tensor, base_log_p: torch.Tensor) -> torch.Tensor:
    """Each word's pull to the base [B, R, 5] (module docstring)."""
    gap = base_log_p - log_p
    return torch.exp(gap) - gap - 1.0


def surrogate(log_p: torch.Tensor, start_log_p: torch.Tensor, advantage: torch.Tensor, counted: torch.Tensor,
              clip: float = CLIP, dual_clip: float | None = None) -> tuple[torch.Tensor, int, int]:
    """``(the clipped-ratio surrogate's loss, the words counted, the words clipped)`` (module docstring; with
    ``dual_clip``, the dual clip's); the log-probabilities [B, R, 5] of the words said under the model and under the
    model that spoke them."""
    loss, words, clipped, _ = _surrogate_words(log_p, start_log_p, advantage, counted, clip, dual_clip)
    return _per_row(loss, counted), words, clipped


def pull_to_base(log_p: torch.Tensor, base_log_p: torch.Tensor, counted: torch.Tensor) -> torch.Tensor:
    """The pull to the base on the counted words (module docstring)."""
    return _per_row(_pull_words(log_p, base_log_p), counted)


def base_log_probability(base: Prior, samples: Samples, model_log_p: torch.Tensor) -> torch.Tensor:
    """The base's log-probability of ``samples``' words (`masked_log_probability`, no traffic input), read on the
    sentences with counted rows only (module docstring, window samples); every other sentence gets ``model_log_p``
    (the model's, detached), so its pull is zero. With every sentence counted (samples alone), every one is read, as
    before."""
    owners = samples.counted.any(dim=1)
    if bool(owners.all()):
        return masked_log_probability(base, samples.rows, samples.permitted)
    index = torch.nonzero(owners)[:, 0]
    out = model_log_p.detach().clone()
    out[index] = masked_log_probability(base, RowTensors(*(tensor[index] for tensor in samples.rows)),
                                        samples.permitted.select(index.tolist()))
    return out


def data_term(model: Prior, rows: RowTensors) -> torch.Tensor:
    """The teacher-forced loss per step of a batch of single-aircraft samples (prior §7 item 4, `batch_nll`), in
    training mode (dropout on); the model is put back in eval mode after it."""
    model.train()
    try:
        nll, speaks = batch_nll(model, rows)
    finally:
        model.eval()
    return nll.sum() / speaks


def update_loss(model: Prior, start: PassStart, base: Prior, samples: Samples, data: RowTensors,
                dual_clip: float | None = None) -> LossParts:
    """One update's loss, the whole update at once (module docstring; the readable reference of `update_step`):
    ``model`` the model trained (its traffic modules added), ``start`` the model that spoke the samples, ``base`` the
    base checkpoint, ``data`` a batch of single-aircraft samples. The words are scored in eval mode (``model`` is put
    there; a base or a pass-start copy with any module in training mode is refused by `masked_log_probability`,
    D107). ``dual_clip``: as `update_step`'s."""
    model.eval()
    log_p = masked_log_probability(model, samples.rows, samples.permitted, samples.traffic)
    with torch.no_grad():
        start_log_p = masked_log_probability(start.model, samples.rows, samples.permitted, samples.traffic)
        base_log_p = base_log_probability(base, samples, log_p)
    reward, words, clipped = surrogate(log_p, start_log_p, samples.advantage, samples.counted, dual_clip=dual_clip)
    kl = pull_to_base(log_p, base_log_p, samples.counted)
    teacher = data_term(model, data)
    object.__setattr__(samples.traffic, "embedded", None)        # the input keeps no embedded tokens and their graph
    return LossParts(reward + KL_WEIGHT * kl + DATA_WEIGHT * teacher, reward, kl, teacher, words, clipped)


def update_step(model: Prior, start: PassStart, base: Prior, pieces: Sequence[Samples], data: RowTensors,
                data_weight: float = DATA_WEIGHT, dual_clip: float | None = None) -> LossParts:
    """One update's loss in pieces (module docstring, "One update in pieces"): for each piece of the update's samples
    its surrogate and pull, summed over its counted words and divided by the counted rows of all the pieces, and their
    backward; then the data term on ``data`` and its backward. The gradients are left in the model's parameters (the
    caller zeroes them before and steps after); the parts, detached, are `update_loss`'s on the pieces joined (to float
    rounding; at `DATA_WEIGHT`, the weight `update_loss` takes). The words are scored in eval mode, the data term in
    training mode (as `update_loss`). ``data_weight``: the data term's weight; ``dual_clip``: the dual clip's c (None:
    none, D219 (2)); the parts carry the update's extreme words (module docstring)."""
    model.eval()
    rows = sum(piece.counted.sum() for piece in pieces).to(torch.float32)
    surrogate_sum = torch.zeros((), device=rows.device)
    kl_sum = torch.zeros((), device=rows.device)
    words = clipped = 0
    extremes = None
    for piece in pieces:
        log_p = masked_log_probability(model, piece.rows, piece.permitted, piece.traffic)
        with torch.no_grad():
            start_log_p = masked_log_probability(start.model, piece.rows, piece.permitted, piece.traffic)
            base_log_p = base_log_probability(base, piece, log_p)
        loss, counted_words, counted_clipped, piece_extremes = _surrogate_words(
            log_p, start_log_p, piece.advantage, piece.counted, CLIP, dual_clip)
        extremes = piece_extremes if extremes is None else extremes.joined(piece_extremes)
        reward = _counted_sum(loss, piece.counted)
        kl = _counted_sum(_pull_words(log_p, base_log_p), piece.counted)
        ((reward + KL_WEIGHT * kl) / rows).backward()
        object.__setattr__(piece.traffic, "embedded", None)      # the input keeps no embedded tokens and their graph
        surrogate_sum += reward.detach()
        kl_sum += kl.detach()
        words += counted_words
        clipped += counted_clipped
    teacher = data_term(model, data)
    (data_weight * teacher).backward()
    reward, kl, teacher = surrogate_sum / rows, kl_sum / rows, teacher.detach()
    return LossParts(reward + KL_WEIGHT * kl + data_weight * teacher, reward, kl, teacher, words, clipped,
                     extremes=extremes)


def update_gradients(model: Prior, start: PassStart, base: Prior, pieces: Sequence[Samples], data: RowTensors,
                     parameters: Sequence[torch.nn.Parameter], data_weight: float = DATA_WEIGHT
                     ) -> dict[str, list[torch.Tensor | None]]:
    """One update's gradient part by part (the user, 2026-10-10: which part sets a pass's direction): of each of
    ``parameters``, the surrogate's, the pull's (times `KL_WEIGHT`) and the data term's (times ``data_weight``), each
    summed over the pieces as `update_step` sums them, so that the three add up to `update_step`'s gradient (to float
    rounding); None where a part does not reach a parameter. No gradient is left in the model; the words are scored in
    eval mode, the data term in training mode (as `update_step`)."""
    model.eval()
    rows = sum(piece.counted.sum() for piece in pieces).to(torch.float32)
    out: dict[str, list[torch.Tensor | None]] = {part: [None] * len(parameters)
                                                 for part in ("surrogate", "pull", "data")}

    def add(part: str, grads: Sequence[torch.Tensor | None]) -> None:
        out[part] = [g if held is None else held if g is None else held + g for held, g in zip(out[part], grads)]

    for piece in pieces:
        log_p = masked_log_probability(model, piece.rows, piece.permitted, piece.traffic)
        with torch.no_grad():
            start_log_p = masked_log_probability(start.model, piece.rows, piece.permitted, piece.traffic)
            base_log_p = base_log_probability(base, piece, log_p)
        loss, _, _, _ = _surrogate_words(log_p, start_log_p, piece.advantage, piece.counted, CLIP)
        add("surrogate", torch.autograd.grad(_counted_sum(loss, piece.counted) / rows, parameters, retain_graph=True,
                                             allow_unused=True))
        add("pull", torch.autograd.grad(KL_WEIGHT * _counted_sum(_pull_words(log_p, base_log_p), piece.counted) / rows,
                                        parameters, allow_unused=True))
        object.__setattr__(piece.traffic, "embedded", None)      # the input keeps no embedded tokens and their graph
    add("data", torch.autograd.grad(data_weight * data_term(model, data), parameters, allow_unused=True))
    return out


def landed_step(model: Prior, start: PassStart, base: Prior, pieces: Sequence[Samples], data: RowTensors) -> LossParts:
    """One update of a campaign that trains on the landed sentences (P49), in pieces as `update_step`: each kept
    sentence's words, teacher-forced under its traffic and the speaker's records, their negative log-likelihood (in
    ``surrogate``) and the pull to the base, summed over the counted words (every row said) and divided by the counted
    rows of all the pieces, and their backward; then the data term and its backward. No ratio and no advantage
    (``start`` is not read; ``clipped`` is 0). The words are scored in eval mode, the data term in training mode."""
    model.eval()
    rows = sum(piece.counted.sum() for piece in pieces).to(torch.float32)
    nll_sum = torch.zeros((), device=rows.device)
    kl_sum = torch.zeros((), device=rows.device)
    words = 0
    for piece in pieces:
        log_p = masked_log_probability(model, piece.rows, piece.permitted, piece.traffic)
        with torch.no_grad():
            base_log_p = masked_log_probability(base, piece.rows, piece.permitted)
        nll = -_counted_sum(log_p, piece.counted)
        kl = _counted_sum(_pull_words(log_p, base_log_p), piece.counted)
        ((nll + KL_WEIGHT * kl) / rows).backward()
        object.__setattr__(piece.traffic, "embedded", None)      # the input keeps no embedded tokens and their graph
        nll_sum += nll.detach()
        kl_sum += kl.detach()
        words += int(piece.counted[..., None].expand_as(log_p).sum())
    teacher = data_term(model, data)
    (DATA_WEIGHT * teacher).backward()
    nll, kl, teacher = nll_sum / rows, kl_sum / rows, teacher.detach()
    return LossParts(nll + KL_WEIGHT * kl + DATA_WEIGHT * teacher, nll, kl, teacher, words, 0)


def one_pass(model: Prior, base: Prior, optimizer: torch.optim.Optimizer,
             pairs: Iterable[tuple[Sequence[Samples], RowTensors]], step: Callable[..., LossParts] = update_step,
             clip_norm: float | None = None) -> list[LossParts]:
    """One pass over a round's samples (§2 item 5): the model at the start of the pass is the one that spoke them;
    each update's samples, in pieces (``step``: `update_step`, or `landed_step` for P49), paired with a batch of the
    data term, drawn from ``pairs`` one at a time (a round's samples need not be held at once); one optimizer step for
    each, its gradient clipped to ``clip_norm`` first (D168; None: not clipped). The parts of each update, detached,
    with its gradient's norm before the clip."""
    return passes(model, base, optimizer, [pairs], step, clip_norm)[0]


def passes(model: Prior, base: Prior, optimizer: torch.optim.Optimizer,
           each: Sequence[Iterable[tuple[Sequence[Samples], RowTensors]]], step: Callable[..., LossParts] = update_step,
           clip_norm: float | None = None) -> list[list[LossParts]]:
    """E passes over a round's samples (D169; E = ``len(each)``): the model at the start of the first is the one that
    spoke them, and every pass's ratio is against it (one `PassStart`); pass e's updates drawn from ``each[e]`` (its own
    order), each as `one_pass`'s. The parts of each pass's updates."""
    start = PassStart(model)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    return [_updates(model, start, base, optimizer, parameters, pairs, step, clip_norm) for pairs in each]


def _updates(model: Prior, start: PassStart, base: Prior, optimizer: torch.optim.Optimizer,
             parameters: Sequence[torch.nn.Parameter], pairs: Iterable[tuple[Sequence[Samples], RowTensors]],
             step: Callable[..., LossParts], clip_norm: float | None) -> list[LossParts]:
    """One pass's updates against ``start`` (`one_pass`)."""
    out = []
    for pieces, rows in pairs:
        optimizer.zero_grad(set_to_none=True)
        parts = step(model, start, base, pieces, rows)
        norm, clipped = clipped_step(optimizer, parameters, clip_norm)
        out.append(replace(parts, grad_norm=norm, grad_clipped=clipped))
    return out


def clipped_step(optimizer: torch.optim.Optimizer, parameters: Sequence[torch.nn.Parameter], clip_norm: float | None
                 ) -> tuple[torch.Tensor, bool]:
    """One step of ``optimizer`` on the gradients of ``parameters``, clipped first to ``clip_norm`` (D168; None: not
    clipped): the gradients' norm before the clip (detached) and whether they were clipped."""
    norm = torch.nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None])
    clipped = clip_norm is not None and bool(norm > clip_norm)
    if clip_norm is not None:
        torch.nn.utils.clip_grads_with_norm_(parameters, clip_norm, norm)
    optimizer.step()
    return norm.detach(), clipped


def update_row(parts: LossParts) -> dict[str, float | int]:
    """One update of a pass of `update_step` for its log (D219 (4)): its gradient's norm before the clip, its counted
    words, its surrogate and pull, and its extreme words (`Extremes`)."""
    return {"grad_norm": float(parts.grad_norm), "words": parts.words, "surrogate": float(parts.surrogate),
            "kl": float(parts.kl), **asdict(parts.extremes)}


def stacked(parts: Sequence[LossParts]) -> dict[str, float]:
    """The means of a pass's parts, its share of clipped words, its updates' gradient norms before the clip (mean and
    largest) and the share of its updates clipped (D168; for its log)."""
    words = sum(p.words for p in parts)
    norms = torch.stack([p.grad_norm for p in parts])
    return {"loss": float(torch.stack([p.loss for p in parts]).mean()),
            "surrogate": float(torch.stack([p.surrogate for p in parts]).mean()),
            "kl": float(torch.stack([p.kl for p in parts]).mean()),
            "data": float(torch.stack([p.data for p in parts]).mean()),
            "words": words, "clipped_share": sum(p.clipped for p in parts) / words,
            "grad_norm_mean": float(norms.mean()), "grad_norm_max": float(norms.max()),
            "grad_clipped_share": sum(p.grad_clipped for p in parts) / len(parts)}
