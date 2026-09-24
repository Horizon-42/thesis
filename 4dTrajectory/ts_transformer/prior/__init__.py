"""Stage 5 of the two-tier framework: the prior, an autoregressive model of the instruction words
(``docs/2026-09-24_prior_design.zh.md``).

At each 2 s step of a scene (the aircraft at one airport at one time; one aircraft in design §9 step 1) it reads
only what is known before the step — positions, the airport's earlier landings, the words it has said — and writes
each aircraft's six word columns, in order, each "unchanged" or a new word; the first `scene.N_LOOK` rows are only
observed, and the first predicted step says every column. It learns from the sentence artefact alone (teacher
forcing); it knows nothing of the executor.

- ``scene``   the scene index and the airport's landing context (the sealed test days left out)
- ``data``    the sentence artefact → per-step inputs and targets, the variants compared, batches by length
- ``model``   time attention → aircraft attention → feed-forward, candidate-runway tokens, the ordered heads
- ``train``   the training loop: early stopping on val, the best state kept
- ``readout`` per-step likelihood, word-change metrics, the two baselines, the first-step runway against B0 / B1 / B3
"""
