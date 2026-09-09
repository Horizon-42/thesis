"""The scene data plane's tensor side (scene design §五 P2.c): the neighbours' ENTITY rows
and the runway-use scalars as arrays in the ego's chart, from
``flight_scenarios.scene_context``.

The per-neighbour SEQUENCE half (``[N_MAX, L, 6]`` tracks on the ego's lookback grid and
their mask) was deleted 2026-09-07 (package audit T2): the L4 gate it was built for did not
pass, and the one consumer — ``run_ts_scene_explainability.py`` — reads only the entity and
scalar halves. Nothing here resamples a neighbour onto a grid any more."""
