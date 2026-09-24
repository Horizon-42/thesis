"""Stage 5 of the two-tier framework: the prior, an autoregressive model of the instruction words
(``docs/2026-09-24_prior_design.zh.md``).

At each 2 s step it reads the flight's context (its airport and candidate runways), the state and the words in
force, and writes the step's six word columns — each "unchanged" or a new word. It learns from the sentence
artefact alone (teacher forcing); it knows nothing of the executor.

- ``data``    the sentence artefact → per-step inputs and targets, batched by length
- ``model``   the causal transformer and its six heads
- ``train``   the training loop: early stopping on val, the best state kept
- ``readout`` per-step likelihood, word-change metrics, the two baselines counted from train
"""
