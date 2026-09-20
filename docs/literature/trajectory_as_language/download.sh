#!/usr/bin/env bash
# Re-fetch every paper this folder cites that is NOT already in a sibling literature folder
# (2026-09-20 URLs). 12 PDFs, one curl per file, each verified to start with "%PDF-".
#
# NOT fetched here on purpose: eight sources already live in sibling folders and are cited there
# by path (the rule multimodal_intent/download.sh and prediction_horizons/download.sh use). Run
# that folder's download.sh if one is missing:
#   ../manoeuvre_tokens/          MotionLM (Seff 2023), Trajeglish (Philion 2024), SMART (Wu 2024),
#                                 grammar-constrained decoding (Geng 2023),
#                                 scheduled sampling (Bengio 2015), FTP-LLM (Luo & Zhou 2025)
#   ../hierarchical_prediction/   DAgger (Ross, Gordon & Bagnell 2011),
#                                 FlightBERT++ (Guo 2023), FlightBERT + spoken instructions (Guo 2023),
#                                 TNT (Zhao 2020), DenseTNT (Gu 2021), MTR (Shi 2022)
#   ../multimodal_intent/         TPP — tree-structured policy planning (Chen 2023)
#
# NOT reachable and NOT fetched (see README §5 "Not verified"): FlightBERT (the original,
# IEEE T-ITS) has no arXiv record at all — `ti:"FlightBERT"` on the arXiv API returns zero hits.
set -u
cd "$(dirname "$0")"
mkdir -p papers
UA='Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0'

get() { # get <destination stem> <url>
  if [ -s "papers/$1.pdf" ]; then echo "have $1"; return; fi
  if curl -sL -f --retry 2 -A "$UA" -m 120 -o "papers/$1.pdf" "$2" &&
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

# --- A. tokenised motion models trained by next-token prediction (2024-2025 successors) -------
# Hu, Chai, Yang, Qian, Li, Shao, Zhang, Xu & Liu, ECCV 2024 (GUMP)
arxiv GUMP_Hu2024_solving_motion_planning_tasks_with_a_scalable_generative_model 2407.02797
# Zhou, Hu, Chen, Wang, Guan, Wu, Li, Huang & Xue, NeurIPS 2024 (BehaviorGPT)
arxiv BehaviorGPT_Zhou2024_smart_agent_simulation_for_autonomous_driving_with_next-patch_prediction 2405.17372
# Yang, Tan & Kraehenbuehl, ICCV 2025 (InfGen)
arxiv InfGen_Yang2025_long-term_traffic_simulation_with_interleaved_autoregressive_motion_and_scenario_generation 2506.17213

# --- B. closed-loop / on-policy fine-tuning of a tokenised motion model (the central item) ----
# Zhang, Karkus, Igl, Ding, Chen, Ivanovic & Pavone, CVPR 2025 (CAT-K)
arxiv CATK_Zhang2025_closed-loop_supervised_fine-tuning_of_tokenized_traffic_models 2412.05334

# --- C. RL / RLHF fine-tuning of trajectory-token models --------------------------------------
# Cao, Ivanovic, Xiao & Pavone, arXiv 2023 (TrafficRLHF)
arxiv TrafficRLHF_Cao2023_reinforcement_learning_with_human_feedback_for_realistic_traffic_simulation 2309.00709
# Rowe, Girgis, Gosselin, Carrez, Golemo, Heide, Paull & Pal, CoRL 2024 (CtRL-Sim)
arxiv CtRLSim_Rowe2024_reactive_and_controllable_driving_agents_with_offline_reinforcement_learning 2403.19918
# Wang, Chen, Li, Li, Zhang, Xia & Yu, IEEE RA-L 2026, DOI 10.1109/LRA.2026.3678842 (R1Sim)
arxiv R1Sim_Wang2026_learning_rollout_from_sampling_an_r1-style_tokenized_traffic_simulation_model 2603.24989

# --- D. goal-directed decoding: a tree over a learned behaviour model -------------------------
# Huang, Karkus, Ivanovic, Chen, Pavone & Lv, ICRA 2024 (DTPP)
arxiv DTPP_Huang2024_differentiable_joint_conditional_prediction_and_cost_evaluation_for_tree_policy_planning 2310.05885

# --- E. constrained decoding as feasibility: masking tokens in a motion model ------------------
# Cui, Liang, Yang, Tang & Cui, arXiv 2026 (SaFeR)
arxiv SaFeR_Cui2026_safety-critical_scenario_generation_via_feasibility-constrained_token_resampling 2603.04071

# --- F. exposure-bias background --------------------------------------------------------------
# Ranzato, Chopra, Auli & Zaremba, ICLR 2016 (MIXER) — the paper that names "exposure bias"
arxiv MIXER_Ranzato2016_sequence_level_training_with_recurrent_neural_networks 1511.06732

# --- G. aviation: LLM / token flight-trajectory work 2024-2026 --------------------------------
# Wu, Fang, Wang, Liu, Yang, Yang, Guo, Li & Wang, COLM 2026 (FLY-EVAL++)
arxiv FLYEVALpp_Wu2026_evidence-driven_evaluation_protocol_for_safety-constrained_flight_prediction_with_llms 2609.04021
# Phisannupawong, Damanik & Choi, arXiv 2025 (LLM4Delay)
arxiv LLM4Delay_Phisannupawong2025_flight_delay_prediction_via_cross-modality_adaptation_and_aircraft_trajectory_representation 2510.23636
