"""Stage 5 of the two-tier framework: the prior, an autoregressive model of the instruction words
(``docs/2026-09-24_prior_design.zh.md``).

At each 2 s step it reads the flight's context (its airport and candidate runways), the state and the words in
force, and writes the step's six word columns — each "unchanged" or a new word; at step 0 only the runway, the
other columns being the hand-over's, given. It learns from the sentence artefact alone (teacher forcing); it knows
nothing of the executor.

- ``data``    the sentence artefact → per-step inputs (positions, every candidate runway's place; the input sets
              of §8.3) and targets, the train-internal selection set, batches by length
- ``model``   the causal transformer, candidate-runway tokens, the five heads and the runway scorer
- ``train``   the training loop: early stopping on val, the best state kept
- ``readout`` per-step likelihood, word-change metrics, the two baselines counted from train
"""
