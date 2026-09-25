"""The executor flies ONE SEGMENT of a Training flight's sentence, live, for the frontend's Training view
(aeroviz-4d `docs/36-2026-09-20-training-module.zh.md` §4.7; the executor: `4dTrajectory/ts_transformer/autopilot/`).

A SEGMENT is one word of one column in force, flown from the observed aircraft's state where the word was said to where
its own envelope ends (`segment`): the executor is told the six words in force there as its step 0 (a sentence's step 0
is what the aircraft is already doing, executor design §2.4), then the sentence's words of the steps after it, each
where the observed aircraft heard it (the spec's word clock, design §11), and is flown with its own stepper, stopped at
the segment's stop (`fly`). Only the selected word is judged, by the executor's own judge on what was flown
(`verdict`); the answer (`payload`) says how long each part took. NOTHING IS PRECOMPUTED: no replay record and no
Training overlay is read. The service — which set, which spec, the flights kept rebuilt — is `backend`.

    segment.py   which segment a word is, what it is told (pure)       verdict.py  the selected word's verdict
    fly.py       the flight rebuilt; the segment flown and stopped     payload.py  the answer
    backend.py   ``POST /autopilot/segment``
"""

# `ts_transformer` lives under `4dTrajectory/`: the backend's import paths put it on `sys.path` (cheap: no import of it)
import aeroviz_backend.paths  # noqa: F401,E402
