#!/usr/bin/env bash
# Re-fetch every paper indexed in README.md §1.1 (2026-09-18 URLs). 9 PDFs.
#
# NOT fetched here on purpose: nine sources in §1.2 already live in sibling literature folders and
# are cited there by path (same rule as manoeuvre_tokens/download.sh uses for DAgger). Run that
# folder's download.sh if one is missing:
#   ../hierarchical_prediction/  TNT, DenseTNT, MTR, MTR++, MultiPath++, TrajAirNet
#   ../control_normalization/    Trajectron++
#   ../procedure_hard_constraints/  CoverNet, annealed-WTA (Xu 2025), ASCENT (Prutsch 2026)
#   ../manoeuvre_tokens/         MotionLM, Trajeglish, SMART
#
# Two non-arXiv mirrors:
#  * Multiple Choice Learning (Guzmán-Rivera, Batra & Kohli, NIPS 2012) predates arXiv posting for
#    this line of work; it comes from the NeurIPS proceedings server (open access).
#  * Pang, Xu & Liu (2019) is an open-access (CC-BY 3.0 US) PHM Society conference paper; the OJS
#    "download" endpoint serves the PDF directly.
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

# --- Cluster A: where the min-over-K loss comes from -----------------------------------------
get MCL_GuzmanRivera2012_multiple_choice_learning_learning_to_produce_multiple_structured_outputs \
    "https://proceedings.neurips.cc/paper_files/paper/2012/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf"
arxiv MHP_Rupprecht2017_learning_in_an_uncertain_world_representing_ambiguity_through_multiple_hypotheses 1612.00197
arxiv EWTA_Makansi2019_overcoming_limitations_of_mixture_density_networks_sampling_and_fitting_framework 1906.03631
# --- Cluster B: K hypotheses in motion forecasting -------------------------------------------
arxiv MultiPath_Chai2019_multiple_probabilistic_anchor_trajectory_hypotheses_for_behavior_prediction 1910.05449
arxiv DESIRE_Lee2017_distant_future_prediction_in_dynamic_scenes_with_interacting_agents 1704.04394
arxiv mmTransformer_Liu2021_multimodal_motion_prediction_with_stacked_transformers 2103.11624
# --- Cluster C: tree-/branch-structured multi-step prediction --------------------------------
arxiv TPP_Chen2023_tree-structured_policy_planning_with_learned_behavior_models 2301.11902
# --- Cluster D: aviation ----------------------------------------------------------------------
arxiv DeepTP_LiuHansen2018_predicting_aircraft_trajectories_deep_generative_convolutional_recurrent_neural_networks 1812.11670
get PangXuLiu2019_aircraft_trajectory_prediction_using_lstm_neural_network_with_embedded_convolutional_layer \
    "https://papers.phmsociety.org/index.php/phmconf/article/download/849/phmc_19_849"
