# Fronter's requests to the designer

Fronter writes this file (outline §5 rule 10, frontend §0.3); it is rewritten in full each time. Each item is a reading
fronter made where the design says nothing, or a gap; a reading is a proposal until the user decides, and a decided item
is removed. Paths are relative to `4dTrajectory/ts_transformer/`; frontend paths to `aeroviz-4d/src/`.

State: 2026-10-08. Items 1–8 were decided (frontend D178) and are removed. All three sets of stage C are published
(`windows_seg60_r5_20261008`, `loss_windows_seg60_20261008`, `loss_windows_c10_20261008`). D178's two mirrors wait for
stage C's `post.window_lists.identity_of` and stage D's positions helper on `dev-two-tier`.

1. **`speed.json` keeps its name `ts-model-speed-v1`.** Stage C's record changed shape with the sample (`model.windows`
   a count by batch, `perAirport`, `seed`, `selectionWindows`), but the one stage C readout written before it
   (`outputs/POOLED/speed/post_seg60_r5_20261008`) was deleted at the user's word (2026-10-08) and nothing reads the
   file, so no file of the old shape is left; stage B's record is unchanged. The name changes the next time a reader
   is written for it.
