"""Closed-loop targets (prior design §9.2): at each row of a chain — flown by the executor from the words the prior
said — what the prior should say there, read off the same flight's labelled sentence.

The target is still the labelled words (design §7), aligned to the chain by the label's own events — the rows at which
the label says a word:

- the first predicted step says every column's labelled word in force there, as in pretraining (the chain is at the
  observed state there);
- a column the label has not spoken in since the first predicted step says nothing: where the chain's words from its
  first step differ from the label's, it is not asked to change them — those words describe what was in force before
  the prior spoke, which the label partly reads from the flight's future (design §11), and asking for them again would
  teach the prior to say them again and again. Except a clearance: one the label has in force is asked for until the
  chain gives it, however early the label gave it (below);
- a word the label says after the first predicted step, at row ``u``, is the chain's to answer from ``u − window``
  rows on: it has answered when its word in that column is the label's, or — heading, altitude, angle, speed, whose
  values the chain may pick for itself — when it said any word in that column since ``u − window`` (after its first
  step, which says every column anyway); the runway and the approach must be the label's word. Answered, the column
  says nothing (a word said early is not taken back); unanswered, it says the label's word from ``u`` to
  ``u + window``. Later than that the label no longer says what the chain should say, and the column is not asked —
  except a clearance, which is asked for until the chain gives it — the label's clearance given at the handover as
  well (a flight not cleared is the free generation's commonest failure, readouts §5; on the select readout 15.6 % of
  the sentences whose label was cleared at the first predicted step did not say so there);
- past the sentence's last row the label's words are that row's.

A label word the executor would not act on is withheld ("unchanged"): a runway once the executor has cleared or captured
(`Executor.runway_locked`), "not cleared" once it has cleared (only a go-around undoes a clearance), a heading once it
has captured the line (it flies the line) — words the labeller never writes in those states either. A step that breaks
a vocabulary compatibility rule at the chain's altitude (`instructions.grammar`, the decoding mask's own check) leaves
the columns of the broken rule out of the loss, as "unchanged" for the heads that read them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ts_transformer.instructions.grammar import step_allowed
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, APPROACH, APPROACH_CLEARED, APPROACH_NOT_CLEARED, HEADING, RUNWAY, UNCHANGED, Words,
)
from ts_transformer.prior.data import in_force_words, issued_rows
from ts_transformer.prior.scene import N_LOOK

#: The columns each compatibility rule reads (vocabulary design §2.1, §2.5).
FAMILIES = {"runway": (RUNWAY, APPROACH), "vertical": (ALTITUDE, ANGLE)}
#: The columns whose label word the chain answers only by saying that word.
EXACT = (RUNWAY, APPROACH)


@dataclass(frozen=True)
class Targets:
    classes: np.ndarray        # [steps, 6] int64: 0 "unchanged", else the word + 1 — taught, and read by later heads
    asked: np.ndarray          # [steps, 6] bool: the columns the loss counts
    counts: dict[str, int]     # per reason, the column-steps it decided (`relabel`)


def relabel(labelled: np.ndarray, said: np.ndarray, cleared: np.ndarray, captured: np.ndarray, height_m: np.ndarray,
            spec: VocabularySpec, words: Words, *, window: int) -> Targets:
    """The targets of a chain's steps. ``labelled``: the flight's sentence (``[N, 6]`` from row 0, `UNCHANGED` where a
    column says nothing); ``said``: the chain's steps from row `N_LOOK` (``[steps, 6]``, step 0 every column);
    ``cleared``, ``captured``: the executor's state when each step was said (``[steps]``); ``height_m``: the
    aircraft's height at each step's row; ``window``: rows either side of a label word within which the chain's
    answer counts, and after which an unanswered word is no longer asked for. ``counts``: column-steps asked to say
    the label's word (``"said on time"`` within the window, ``"clearance late"`` beyond it, ``"clearance at the
    handover"`` for one the label had in force at the first predicted step), answered by the chain
    (``"answered"``), left out because the chain missed the word (``"missed"``), withheld because the executor would
    not act on the word (per reason), and left out because the step broke a compatibility rule (per rule)."""
    steps = len(said)
    if not (len(cleared) == len(captured) == len(height_m) == steps) or (said[0] == UNCHANGED).any():
        raise ValueError("one state per step said, and a first step that says every column")
    labelled = np.asarray(labelled, dtype=np.int64)
    label, spoken = in_force_words(labelled), issued_rows(labelled)
    last = len(label) - 1
    classes = np.zeros((steps, 6), dtype=np.int64)
    asked = np.ones((steps, 6), dtype=bool)
    classes[0] = label[min(N_LOOK, last)] + 1
    counts = dict.fromkeys(("said on time", "clearance late", "clearance at the handover", "answered", "missed",
                            "runway locked", "not cleared after a clearance", "heading after the capture",
                            *(f"broke the {name} rule" for name in FAMILIES)), 0)
    chain = said[0].astype(np.int64)
    last_said = np.full(6, -np.inf)                          # the row each column last said a word after step 0
    for j in range(1, steps):
        row = N_LOOK + j
        wanted, at = label[min(row, last)], spoken[min(row, last)]
        deaf = {RUNWAY: ("runway locked", cleared[j] or captured[j]),
                APPROACH: ("not cleared after a clearance", cleared[j] and wanted[APPROACH] == APPROACH_NOT_CLEARED),
                HEADING: ("heading after the capture", captured[j])}
        target = np.full(6, UNCHANGED, dtype=np.int64)
        for c in range(6):
            if chain[c] == wanted[c]:
                continue
            if at[c] <= N_LOOK:
                if c == APPROACH and wanted[c] == APPROACH_CLEARED:
                    target[c] = wanted[c]
                    counts["clearance at the handover"] += 1
                continue
            if c not in EXACT and last_said[c] >= at[c] - window:
                counts["answered"] += 1
            elif c in deaf and deaf[c][1]:
                counts[deaf[c][0]] += 1
            elif row <= at[c] + window:
                target[c] = wanted[c]
                counts["said on time"] += 1
            elif c == APPROACH and wanted[c] == APPROACH_CLEARED:
                target[c] = wanted[c]
                counts["clearance late"] += 1
            else:
                asked[j, c] = False
                counts["missed"] += 1
        for name, columns in FAMILIES.items():
            cols = list(columns)
            if (target[cols] == UNCHANGED).all():
                continue
            step = np.full(6, UNCHANGED, dtype=np.int64)
            step[cols] = target[cols]
            if not step_allowed(chain, step, float(height_m[j]), spec, words):
                target[cols] = UNCHANGED
                asked[j, cols] = False
                counts[f"broke the {name} rule"] += 1
        classes[j] = np.where(target == UNCHANGED, 0, target + 1)
        written = said[j] != UNCHANGED
        chain = np.where(written, said[j], chain)
        last_said = np.where(written, row, last_said)
    return Targets(classes, asked, counts)
