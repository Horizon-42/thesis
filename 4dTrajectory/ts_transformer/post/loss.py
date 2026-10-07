"""The loss of the post-training (post-training §2 item 5, §8 C7; D36, D76): one update and one pass.

    loss = the clipped-ratio surrogate (ε = `CLIP`) on the words said, with the advantage of each row
         + `KL_WEIGHT` × the pull to the base (the KL on the sampled words, under the masks they were said under)
         + `DATA_WEIGHT` × the teacher-forced data term (single-aircraft samples of the closed-loop sentences in the
           base's selection ``landed``, D36, D76)

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
whole update at once, is the readable reference `update_step` is checked against.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
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
        """The samples ``indices``, in that order (each tensor's first axis), padded as here."""
        chosen = torch.as_tensor(list(indices), device=self.advantage.device)
        return Samples(RowTensors(*(tensor[chosen] for tensor in self.rows)), self.permitted.select(indices),
                       self.traffic.select(chosen), self.advantage[chosen],
                       self.counted[chosen])

    def __post_init__(self) -> None:
        shape = tuple(self.rows.asked.shape)
        if tuple(self.advantage.shape) != shape or tuple(self.counted.shape) != shape or self.counted.dtype != torch.bool:
            raise ValueError(f"the advantage and the counted rows are [B, R] = {shape} beside the rows")
        if (self.counted & ~self.rows.asked).any():
            raise ValueError("a counted row is a row the speaker said")
        if not self.counted.any(dim=1).all():
            raise ValueError("every sample counts some rows: a sample without one carries no gradient")


@dataclass(frozen=True)
class LossParts:
    """One update's loss and its parts (each a scalar tensor), the words counted and those whose ratio left the clip."""

    loss: torch.Tensor
    surrogate: torch.Tensor
    kl: torch.Tensor
    data: torch.Tensor
    words: int
    clipped: int


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
                     clip: float) -> tuple[torch.Tensor, int, int]:
    """Each word's clipped-ratio loss [B, R, 5], the words counted and those clipped (module docstring)."""
    ratio = torch.exp(log_p - start_log_p)
    a = advantage[..., None]
    loss = -torch.minimum(ratio * a, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * a)
    words = counted[..., None].expand_as(ratio)
    return loss, int(words.sum()), int(((ratio - 1.0).abs() > clip)[words].sum())


def _pull_words(log_p: torch.Tensor, base_log_p: torch.Tensor) -> torch.Tensor:
    """Each word's pull to the base [B, R, 5] (module docstring)."""
    gap = base_log_p - log_p
    return torch.exp(gap) - gap - 1.0


def surrogate(log_p: torch.Tensor, start_log_p: torch.Tensor, advantage: torch.Tensor, counted: torch.Tensor,
              clip: float = CLIP) -> tuple[torch.Tensor, int, int]:
    """``(the clipped-ratio surrogate's loss, the words counted, the words clipped)`` (module docstring); the log-
    probabilities [B, R, 5] of the words said under the model and under the model that spoke them."""
    loss, words, clipped = _surrogate_words(log_p, start_log_p, advantage, counted, clip)
    return _per_row(loss, counted), words, clipped


def pull_to_base(log_p: torch.Tensor, base_log_p: torch.Tensor, counted: torch.Tensor) -> torch.Tensor:
    """The pull to the base on the counted words (module docstring)."""
    return _per_row(_pull_words(log_p, base_log_p), counted)


def data_term(model: Prior, rows: RowTensors) -> torch.Tensor:
    """The teacher-forced loss per step of a batch of single-aircraft samples (prior §7 item 4, `batch_nll`), in
    training mode (dropout on); the model is put back in eval mode after it."""
    model.train()
    try:
        nll, speaks = batch_nll(model, rows)
    finally:
        model.eval()
    return nll.sum() / speaks


def update_loss(model: Prior, start: PassStart, base: Prior, samples: Samples, data: RowTensors) -> LossParts:
    """One update's loss, the whole update at once (module docstring; the readable reference of `update_step`):
    ``model`` the model trained (its traffic modules added), ``start`` the model that spoke the samples, ``base`` the
    base checkpoint, ``data`` a batch of single-aircraft samples. The words are scored in eval mode (``model`` is put
    there; a base or a pass-start copy with any module in training mode is refused by `masked_log_probability`,
    D107)."""
    model.eval()
    log_p = masked_log_probability(model, samples.rows, samples.permitted, samples.traffic)
    with torch.no_grad():
        start_log_p = masked_log_probability(start.model, samples.rows, samples.permitted, samples.traffic)
        base_log_p = masked_log_probability(base, samples.rows, samples.permitted)
    reward, words, clipped = surrogate(log_p, start_log_p, samples.advantage, samples.counted)
    kl = pull_to_base(log_p, base_log_p, samples.counted)
    teacher = data_term(model, data)
    object.__setattr__(samples.traffic, "embedded", None)        # the input keeps no embedded tokens and their graph
    return LossParts(reward + KL_WEIGHT * kl + DATA_WEIGHT * teacher, reward, kl, teacher, words, clipped)


def update_step(model: Prior, start: PassStart, base: Prior, pieces: Sequence[Samples], data: RowTensors) -> LossParts:
    """One update's loss in pieces (module docstring, "One update in pieces"): for each piece of the update's samples
    its surrogate and pull, summed over its counted words and divided by the counted rows of all the pieces, and their
    backward; then the data term on ``data`` and its backward. The gradients are left in the model's parameters (the
    caller zeroes them before and steps after); the parts, detached, are `update_loss`'s on the pieces joined (to float
    rounding). The words are scored in eval mode, the data term in training mode (as `update_loss`)."""
    model.eval()
    rows = sum(piece.counted.sum() for piece in pieces).to(torch.float32)
    surrogate_sum = torch.zeros((), device=rows.device)
    kl_sum = torch.zeros((), device=rows.device)
    words = clipped = 0
    for piece in pieces:
        log_p = masked_log_probability(model, piece.rows, piece.permitted, piece.traffic)
        with torch.no_grad():
            start_log_p = masked_log_probability(start.model, piece.rows, piece.permitted, piece.traffic)
            base_log_p = masked_log_probability(base, piece.rows, piece.permitted)
        loss, counted_words, counted_clipped = _surrogate_words(log_p, start_log_p, piece.advantage, piece.counted,
                                                                CLIP)
        reward = _counted_sum(loss, piece.counted)
        kl = _counted_sum(_pull_words(log_p, base_log_p), piece.counted)
        ((reward + KL_WEIGHT * kl) / rows).backward()
        object.__setattr__(piece.traffic, "embedded", None)      # the input keeps no embedded tokens and their graph
        surrogate_sum += reward.detach()
        kl_sum += kl.detach()
        words += counted_words
        clipped += counted_clipped
    teacher = data_term(model, data)
    (DATA_WEIGHT * teacher).backward()
    reward, kl, teacher = surrogate_sum / rows, kl_sum / rows, teacher.detach()
    return LossParts(reward + KL_WEIGHT * kl + DATA_WEIGHT * teacher, reward, kl, teacher, words, clipped)


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
             pairs: Iterable[tuple[Sequence[Samples], RowTensors]], step: Callable[..., LossParts] = update_step
             ) -> list[LossParts]:
    """One pass over a round's samples (§2 item 5): the model at the start of the pass is the one that spoke them;
    each update's samples, in pieces (``step``: `update_step`, or `landed_step` for P49), paired with a batch of the
    data term, drawn from ``pairs`` one at a time (a round's samples need not be held at once); one optimizer step for
    each. The parts of each update, detached."""
    start = PassStart(model)
    out = []
    for pieces, rows in pairs:
        optimizer.zero_grad(set_to_none=True)
        out.append(step(model, start, base, pieces, rows))
        optimizer.step()
    return out


def stacked(parts: Sequence[LossParts]) -> dict[str, float]:
    """The means of a pass's parts and its share of clipped words (for its log)."""
    words = sum(p.words for p in parts)
    return {"loss": float(torch.stack([p.loss for p in parts]).mean()),
            "surrogate": float(torch.stack([p.surrogate for p in parts]).mean()),
            "kl": float(torch.stack([p.kl for p in parts]).mean()),
            "data": float(torch.stack([p.data for p in parts]).mean()),
            "words": words, "clipped_share": sum(p.clipped for p in parts) / words}
