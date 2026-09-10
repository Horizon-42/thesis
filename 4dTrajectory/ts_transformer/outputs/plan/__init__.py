"""The plan-and-guidance path (`docs/2026-09-09_plan_and_guidance_design.md`).

The published approach procedure is the skeleton; the network predicts only what the
procedure leaves free (the plan — five operating parameters, three route parameters); a
deterministic guidance layer flies the plan inside the skeleton through the shared
point-mass rollout, so the trajectory conforms to the procedure and the envelope by
construction. Built in the design's §9 order:

    skeleton       the coded approach in a flight's own chart (step 1; reads
                   `flight_scenarios.procedure_final.procedure_skeleton`)
    extractors     the eight plan parameters read off an observed track, with their
                   ranges (step 1; the supervision, and the oracle test's input)
    guidance/      the controller that flies a plan on the skeleton (step 2)
    forecast       the batch entry point: plans flown through the guidance as `Forecast`s
                   (the strategy seam; the shape the control path's forecast has)

Nothing is re-exported here; import the submodule that owns what you need.
"""
