"""Make every figure of the tutorial (PNG 300 dpi and PDF) into docs/two_tier/tutorial/figures/.

    node scripts/two_tier_tutorial/figure_data.js > scripts/two_tier_tutorial/data/figure_sim.json   # simulation data
    conda run -n aeroviz python scripts/two_tier_tutorial/make_figures.py
"""
import figs_data
import figs_diagrams

for name in ("fig02_vocabulary_grids", "fig03_sentence_example", "fig04_executor_laws", "fig05_closed_loop_real",
             "fig06_corrections_effect", "fig07_delta_grid", "fig09_procedure_masks"):
    getattr(figs_data, name)()
for name in ("fig01_architecture", "fig08_prior_architecture", "fig10_decoding", "fig11_cv_design", "fig12_post_training_loop",
             "fig13_branch_training", "fig14_traffic_features", "fig15_judge_outcomes", "fig16_window_kinds", "fig17_landed_training", "fig18_prior_layers", "fig19_traffic_attention", "fig20_multi_round", "fig21_random_number", "fig22_window_branches", "fig23_window_tree"):
    getattr(figs_diagrams, name)()
