#!/usr/bin/env bash
# Re-fetch every paper indexed in README.md §1 (2026-09-16 URLs). 32 PDFs.
# Not fetchable by script (see README §4): the four paywalled items (B-STAR / Knowledge-Based
# Systems, the JAIS S-STGCNN paper, FPG-SLSTM / Eng. Appl. AI, and the Neurocomputing
# local-history-intent paper). Everything else below is open access or an author/CVF/PMLR mirror.
#
# Two notes on the mirrors:
#  * OpenReview returns HTTP 403 to any script (Crossformer, Pyraformer). Pyraformer is taken
#    from the TU Wien co-author's copy; Crossformer from the Wayback snapshot of the OpenReview
#    PDF (`id_` = raw bytes, no toolbar rewrite). Both were checked against the ICLR header line.
#  * Marcellino/Stock/Watson is the author's Princeton copy of the Feb-2004 draft of the
#    J. Econometrics (2006) paper. NBER TWP 285 is a DIFFERENT paper (Imbens & Newey) — do not
#    "fix" this URL to point there.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p papers
UA="Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
get() { # get <destination> <url>
  if [ -s "papers/$1.pdf" ]; then echo "have $1"; return; fi
  if curl -sL -f --retry 2 -A "$UA" -o "papers/$1.pdf" "$2" && [ "$(head -c 5 "papers/$1.pdf")" = "%PDF-" ]; then
    echo "OK   $1"
  else
    rm -f "papers/$1.pdf"; echo "FAIL $1  $2"
  fi
}
arxiv() { # arxiv <destination> <arxiv id>   (2 s pause between arXiv requests)
  if [ -s "papers/$1.pdf" ]; then echo "have $1"; return; fi
  get "$1" "https://arxiv.org/pdf/$2"; sleep 2
}

# --- Cluster A: hierarchical / two-stage trajectory prediction -------------------------------
arxiv TNT_Zhao2020_target-driven_trajectory_prediction 2008.08294
arxiv DenseTNT_Gu2021_end-to-end_trajectory_prediction_from_dense_goal_sets 2108.09640
arxiv MTR_Shi2022_motion_transformer_with_global_intention_localization_and_local_movement_refinement 2209.13508
arxiv MTRpp_Shi2023_multi-agent_motion_prediction_with_symmetric_scene_modeling_and_guided_intention_querying 2306.17770
get QCNet_Zhou2023_query-centric_trajectory_prediction "https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf"
arxiv Wayformer_Nayakanti2022_motion_forecasting_via_simple_and_efficient_attention_networks 2207.05844
arxiv MultiPathPP_Varadarajan2021_efficient_information_fusion_and_trajectory_aggregation_for_behavior_prediction 2111.14973
arxiv C2F-TP_Zhang2024_coarse-to-fine_denoising_framework_for_uncertainty-aware_trajectory_prediction 2412.13231
arxiv ThreeStepHT_2026_three-step_hierarchical_transformer_for_multi-pedestrian_trajectory_prediction 2606.23058
arxiv HierarchicalIL_Wang2023_interpretable_motion_planner_for_urban_driving_via_hierarchical_imitation_learning 2303.13986
arxiv PlanT_Renz2022_explainable_planning_transformers_via_object-level_representations 2210.14222
# --- Cluster B: multi-scale / segment-token time-series transformers -------------------------
get Crossformer_Zhang2023_transformer_utilizing_cross-dimension_dependency_for_multivariate_time_series_forecasting "https://web.archive.org/web/2023id_/https://openreview.net/pdf?id=vSVLM2j9eie"
arxiv Pathformer_Chen2024_multi-scale_transformers_with_adaptive_pathways_for_time_series_forecasting 2402.05956
arxiv TimeMixer_Wang2024_decomposable_multiscale_mixing_for_time_series_forecasting 2405.14616
arxiv Scaleformer_Shabani2022_iterative_multi-scale_refining_transformers_for_time_series_forecasting 2206.04038
get Pyraformer_Liu2022_low-complexity_pyramidal_attention_for_long-range_time_series_modeling_and_forecasting "https://dsg.tuwien.ac.at/team/sd/papers/ICLR_2022_SD_Pyraformer.pdf"
# --- Cluster C: compounding error, iterated vs direct multi-step -----------------------------
get RossBagnell2010_efficient_reductions_for_imitation_learning "http://proceedings.mlr.press/v9/ross10a/ross10a.pdf"
arxiv DAgger_Ross2011_reduction_of_imitation_learning_and_structured_prediction_to_no-regret_online_learning 1011.0686
get DaD_Venkatraman2015_improving_multi-step_prediction_of_learned_time_series_models "https://www.ri.cmu.edu/pub_files/2015/1/Venkatraman.pdf"
arxiv BenTaieb2012_review_and_comparison_of_strategies_for_multi-step_ahead_forecasting_NN5 1108.3259
get MarcellinoStockWatson2006_comparison_of_direct_and_iterated_multistep_AR_methods_for_forecasting_macroeconomic_time_series "https://www.princeton.edu/~mwatson/papers/hstep_3.pdf"
get Chevillon2007_direct_multi-step_estimation_and_forecasting "https://legacy.econ.tuwien.ac.at/hanappi/AgeSo/rp/Chevillon_2007.pdf"
arxiv MultiStepMitigates_2025_learning_with_imperfect_models_when_multi-step_prediction_mitigates_compounding_error 2504.01766
arxiv StatEfficiency_2026_statistical_efficiency_of_single-_and_multi-step_models_for_forecasting_and_control 2603.23465
arxiv Stratify_2024_unifying_multi-step_forecasting_strategies 2412.20510
arxiv FlightBERTpp_Guo2023_non-autoregressive_multi-horizon_flight_trajectory_prediction_framework 2305.01658
# --- Cluster D: multi-aircraft interaction and intent inputs ---------------------------------
arxiv TrajAirNet_Patrikar2021_predicting_like_a_pilot_dataset_and_method_to_predict_socially-aware_aircraft_trajectories 2109.15158
arxiv SpokenInstructions_Guo2023_flightbert_with_spoken_instructions_for_flight_trajectory_prediction 2305.01661
get WTFTPplus_Guo2025_multi-horizon_flight_trajectory_prediction_enabled_by_time-frequency_wavelet_transform "https://www.nature.com/articles/s41467-025-67399-9.pdf"
get WTFTP_Guo2023_flight_trajectory_prediction_enabled_by_time-frequency_wavelet_transform "https://www.nature.com/articles/s41467-023-40903-9.pdf"
arxiv AgentFormer_Yuan2021_agent-aware_transformers_for_socio-temporal_multi-agent_forecasting 2103.14023
arxiv SceneTransformer_Ngiam2021_unified_architecture_for_predicting_multiple_agent_trajectories 2106.08417
# Trajectron++ (arXiv 2001.03093) is NOT re-downloaded here: it already lives at
# docs/literature/control_normalization/papers/Trajectron++_Salzmann2020_…pdf (README §1.4 cites
# that path). Run that folder's download.sh if it is missing.
