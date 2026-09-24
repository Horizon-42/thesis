#!/usr/bin/env bash
# Re-fetch every paper this folder cites that is NOT already in a sibling literature folder
# (2026-09-24 URLs). 14 PDFs, one curl per file, each verified to start with "%PDF-".
# Existing files are skipped, so the script is safe to re-run.
#
# NOT fetched here on purpose: these sources already live in sibling folders and are cited in
# README.md by path (the rule trajectory_as_language/download.sh uses). Run that folder's
# download.sh if one is missing:
#   ../manoeuvre_tokens/          MotionLM (Seff 2023), Trajeglish (Philion 2024), SMART (Wu 2024),
#                                 Graphormer (Ying 2021), grammar-constrained decoding (Geng 2023)
#   ../hierarchical_prediction/   SceneTransformer (Ngiam 2022), AgentFormer (Yuan 2021),
#                                 Wayformer (Nayakanti 2023), MTR (Shi 2022), MTR++ (Shi 2024),
#                                 QCNet (Zhou 2023), TrajAirNet (Patrikar 2022), DAgger (Ross 2011),
#                                 FlightBERT + spoken instructions (Guo 2023)
#   ../multimodal_intent/         DESIRE (Lee 2017)
#   ../control_normalization/     Trajectron++ (Salzmann 2020)  <- asked for in the brief, already here
#   ../trajectory_as_language/    CAT-K (Zhang 2025)            <- asked for in the brief, already here
#                                 R1Sim (Wang 2026)
#   ../prediction_horizons/       MAIFormer (Yoon & Lee 2026)
#   ../procedure_hard_constraints/ ASCENT (Prutsch 2026), phased flight TP (Zhang & Chen 2022)
# (21 sources in total; README.md section 1.2 gives each file name.)
#
# Two sources have no arXiv record (see README section 1.1):
#  * Social LSTM has no arXiv record; the CVF open-access copy is the authors' CVPR 2016 paper.
#  * Groot, Ellerbroek & Hoekstra (SESAR Innovation Days 2022) has no arXiv record; the TU Delft
#    repository serves the final published version (cover page + 9 pages).
set -u
cd "$(dirname "$0")"
mkdir -p papers
UA='Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0'

get() { # get <destination stem> <url>
  if [ -s "papers/$1.pdf" ]; then echo "have $1"; return; fi
  if curl -sL -f --retry 2 -A "$UA" -m 180 -o "papers/$1.pdf" "$2" &&
     [ "$(head -c 5 "papers/$1.pdf")" = "%PDF-" ]; then
    echo "OK   $1"
  else
    rm -f "papers/$1.pdf"; echo "FAIL $1  $2"
  fi
}
arxiv() { # arxiv <destination stem> <arxiv id>   (2 s pause between arXiv requests)
  if [ -s "papers/$1.pdf" ]; then echo "have $1"; return; fi
  get "$1" "https://arxiv.org/pdf/$2"; sleep 2
}

# --- A. how interaction enters the model ---------------------------------------------------------
# Alahi, Goel, Ramanathan, Robicquet, Fei-Fei & Savarese, CVPR 2016 (no arXiv record)
get SocialLSTM_Alahi2016_social_lstm_human_trajectory_prediction_in_crowded_spaces "https://openaccess.thecvf.com/content_cvpr_2016/papers/Alahi_Social_LSTM_Human_CVPR_2016_paper.pdf"
# Velickovic, Cucurull, Casanova, Romero, Lio & Bengio, ICLR 2018
arxiv GAT_Velickovic2018_graph_attention_networks 1710.10903
# Makansi, von Kuegelgen, Locatello, Gehler, Janzing, Brox & Schoelkopf, ICLR 2022
arxiv YouMostlyWalkAlone_Makansi2022_analyzing_feature_attribution_in_trajectory_prediction 2110.05304

# --- B. how interaction is learned: closed-loop training and its benchmark ------------------------
# Suo, Regalado, Casas & Urtasun, CVPR 2021
arxiv TrafficSim_Suo2021_learning_to_simulate_realistic_multi-agent_behaviors 2101.06557
# Montali, Lambert, Mougin, Kuefler, Rhinehart, Li, Gulino, Emrich, Yang, Whiteson, White & Anguelov, NeurIPS 2023 D&B
arxiv WOSAC_Montali2023_the_waymo_open_sim_agents_challenge 2305.12032
# Peng, Luo, Lu, Shen, Gulino, Seff & Fu, ECCV 2024
arxiv RLFT_Peng2024_improving_agent_behaviors_with_rl_fine-tuning_for_autonomous_driving 2409.18343
# Pei, Shi & Shen, ICLR 2026
arxiv SMART-R1_Pei2025_advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning 2509.23993
# Ahmadi, Schofield, Khamidehi, Arasteh, Shan, Mou, Bai & Rezaee, CVPR 2026
arxiv RLFTSim_Ahmadi2026_realistic_and_controllable_multi-agent_traffic_simulation_via_rl_fine-tuning 2605.19033

# --- C. aviation: multi-aircraft interaction and controller actions with traffic context ----------
# Kim, Yoon & Lee, AIAA SciTech 2026
arxiv MALTP_Kim2025_probabilistic_multi-agent_aircraft_landing_time_prediction 2512.08281
# Kuang, Wang, Zhang, Shi & Zhang, arXiv 2025
arxiv DA-STGCN_Kuang2025_4d_trajectory_prediction_based_on_spatiotemporal_feature_extraction 2503.04823
# Jung, Hardy & Kochenderfer, arXiv 2023
arxiv PairwiseTerminal_Jung2023_inferring_traffic_models_in_terminal_airspace_from_flight_tracks_and_procedures 2303.09981
# Brittain, Yang & Wei, arXiv 2020 (a JAIS 2021 article by the same authors exists; identity not verified)
arxiv D2MAV-A_Brittain2020_deep_multi-agent_rl_approach_to_autonomous_separation_assurance 2003.08353
# Groot, Ellerbroek & Hoekstra, SESAR Innovation Days 2022 (TU Delft repository, no arXiv record)
get RelStateTransformer_Groot2022_relative_state_transformer_models_for_marl_in_air_traffic_control "https://repository.tudelft.nl/file/File_7ce604b6-29f7-4a7b-8a4e-bfeda7d0c8de"
# Carvell, De Ath, Benjamin & Everson, AIAA 2026 conference paper, DOI 10.2514/6.2026-2746
arxiv ActionStacking_Carvell2026_online_action-stacking_improves_rl_performance_for_air_traffic_control 2601.04287
