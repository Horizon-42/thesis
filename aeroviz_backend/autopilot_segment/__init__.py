"""The executor flies ONE SEGMENT of a Training flight's closed-loop sentence, live, for the frontend's Training view of
stage A (two-tier vocabulary §12.1 A23, outline §6; the executor: `4dTrajectory/ts_transformer/autopilot/`).

A SEGMENT is one word of one column of the closed-loop sentence at a row interval Δ, flown with the single-flight
executor from the sentence's first predicted step — set up as the export and the formal replay set it up
(`ts_transformer.experiments.training_flights`) — to where the next word of its column is heard (a heading word a lead
later), or to its outcome (`fly`). The answer (`payload`) returns the flight from the word on, the judge's crossing and
decision-altitude check when flown to the outcome, and how far it lies from the artefact's stored flown states. NOTHING
IS PRECOMPUTED: no exported track is read. The service — which set, which spec, the flights kept set up — is `backend`.

    fly.py       the segment, flown and stopped         payload.py  the answer
    backend.py   ``POST /autopilot/segment``
"""

# `ts_transformer` lives under `4dTrajectory/`: the backend's import paths put it on `sys.path` (cheap: no import of it)
import aeroviz_backend.paths  # noqa: F401,E402
