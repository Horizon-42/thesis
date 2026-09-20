"""The second layer's live modules, and the no-token closed loop everything is read against
(`docs/2026-09-18_two_tier_plan_v3.zh.md`).

The INTENT-CODE layer this package was built for — the learned FSQ tokenizer and its codebook,
the code sequences, the causal prior over codes, the gate-T readout and the code atlas, and the
executor conditioned on a code — is **ARCHIVED 2026-09-20**
(`archive/manoeuvre_codes_2026_09/`, README there): plan v3 §10's audit found both layers
trained on truth and evaluated closed-loop, and stage B was rewritten (2026-09-20) around an
INSTRUCTION vocabulary instead. Its readings stay citable in
`docs/2026-09-18_manoeuvre_token_results.zh.md` §9–§11.

What is here, one module each:

    segments     how an approach is cut into pieces from an anchor, the segment-start frame
                 every piece is read in, the rows an encoder sees — kept for the instruction
                 labeller, which reads the same frame
    context      the runway-end and aircraft-type context tokens — the instruction prior's
    instructions the instruction vocabulary and the labeller (stage B, 2026-09-20): a track read
                 back as the controller's words — a LEAF the control path reads for its token
    instruction_sequences / instruction_prior   the sentence with states, and the causal prior
                 over it (four factorised heads + landed)
    lockstep     the closed loop: one round = `executed_step_s`; protocol ``none`` (a no-token
                 executor) or ``truth-instruction`` (an instruction executor, the truth's words
                 by flown position)
    gates        the grid gate (stage A1) and the relative gate (stage A3 / B)
    failure_modes  A2's six non-crossing modes, in the course frame

**Layering** (`tests/test_architecture.py`): `segments` and `instructions` are LEAVES — they
import the data plane, `config`, `io_utils`, geokit and torch only, and the control path may
import THEM (`outputs/control/instruction_token.py` reads the vocabulary for the executor's
token). Every other module here imports the control path, the guidance layer, the data plane
and the inference helpers, never the reverse; runners under `experiments/` are consumers of this
package. Nothing is re-exported here on purpose (layout rule L2).
"""
