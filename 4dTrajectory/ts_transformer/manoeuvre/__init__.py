"""The intent-token model (`docs/2026-09-18_manoeuvre_token_plan.zh.md`): a small discrete
intent space per segment whose decoder IS the control-path executor, a causal prior over the
code sequence, and (later) the multi-aircraft graph.

Grouped by role, one module each (plan §5.3):

    segments     how an approach is cut into `segment_s` pieces from an anchor, the
                 segment-start frame every piece is read in, the rows the encoder sees (P1.1)
    tokenizer    the encoder + FSQ, the codebook artefact, the command-vocabulary baseline (P1.2)
    sequences    a flight as its [code, state] sequence; the operating-day split (P2.1)
    context      the runway-end, procedure-fix and aircraft-type context tokens (P2.1 / P4.1)
    prior        the causal transformer over codes, its loss and decoding (P2.1)
    lockstep     protocols C / A / A-truth, one round = `segment_s` (P3.1)
    readout      the L1 gain, code usage, open-loop displacement, flip rate, Markov baseline (P1.4 …)
    gates        gates T / P / S / X / E / R / G (P3.1)
    scene, graph, decode   the multi-aircraft layer (P5)

**Layering** (`tests/test_architecture.py`): `segments` and `tokenizer` are LEAVES — they import
the data plane, `config`, `io_utils` and torch only — because the control path builds the
truth segment rows into its context rows and holds the tokenizer as a submodule of the executor
(`outputs/control/plan_token.py`, `outputs/control/heads.py`). Every other module here imports
the control path, the guidance layer, the data plane and the inference helpers, never the
reverse; runners under `experiments/` are consumers of this package. Nothing is re-exported
here on purpose (layout rule L2).
"""
