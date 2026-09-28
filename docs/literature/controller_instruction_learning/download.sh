#!/usr/bin/env bash
# Re-fetch every paper this folder cites that is NOT already in a sibling literature folder
# (2026-09-28 URLs). 21 PDFs, one curl per file, each verified to start with "%PDF-".
# Existing files are skipped, so the script is safe to re-run.
#
# NOT fetched here on purpose: these sources already live in sibling folders and are cited in
# README.md by path (the rule multi_agent_interaction/download.sh uses). Run that folder's
# download.sh if one is missing:
#   ../manoeuvre_tokens/          MotionLM (Seff 2023), Trajeglish (Philion 2023), SMART (Wu 2024),
#                                 TimeVQVAE-ATM (Murad 2025), LLM-FTP (Luo 2025)
#   ../trajectory_as_language/    CAT-K (Zhang 2025), R1Sim (Wang 2026)
#   ../multi_agent_interaction/   SMART-R1 (Pei 2025), RLFTSim (Ahmadi 2026), RLFT (Peng 2024),
#                                 TrafficSim (Suo 2021), WOSAC (Montali 2023), MALTP (Kim 2025),
#                                 D2MAV-A (Brittain 2020), Relative-state transformer (Groot 2022),
#                                 Action stacking (Carvell 2026)
#   ../prediction_horizons/       MAIFormer (Yoon & Lee 2026), Xiang & Chen 2024 (arXiv 2409.17359,
#                                 item 17 of the brief)
#   ../hierarchical_prediction/   TrajAirNet (Patrikar 2021), FlightBERT + spoken instructions (Guo 2023)
#   ../control_normalization/     Hodgkin, Pepper & Thomas 2025 (arXiv 2504.02529, item 12 of the brief)
# (21 sources in total; README.md section 3 gives each file name.)
#
# Three sources have no arXiv record; README section 1 says why each mirror is the right file:
#  * Pham, Alam & Duong, Complexity 2020 (CC BY): the publisher host downloads.hindawi.com sits behind a
#    Cloudflare challenge for curl; NTU's DR-NTU repository serves the same publisher PDF (Hindawi header).
#  * Perez-Castan et al., Applied Sciences 2026 (CC BY): mdpi.com returns "Access Denied" to curl;
#    the UPM institutional repository (oa.upm.es/96635) serves the published article.
#  * Kleinert et al., SESAR Innovation Days 2017: served by sesarju.eu as SIDs_2017_paper_27.pdf.
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

# --- A. controller actions / instructions recovered from recorded traffic -------------------------
# Pham, Alam & Duong, Complexity 2020, Article ID 1659103 (no arXiv record; DR-NTU copy of the publisher PDF)
get ATCActionExtraction_Pham2020_air_traffic_controller_action_extraction-prediction_model_using_ml "https://dr.ntu.edu.sg/server/api/core/bitstreams/2057a1ac-1dbd-4f6c-a381-e53608acf817/content"
# Perez-Castan, Perez Navarro, Serrano-Mira, Barcena Martin, Ortega Cuevas & Perez Sanz, Appl. Sci. 16(12):6200, 2026
get SeparationIntent_PerezCastan2026_data-driven_inference_of_atco_separation_intent "https://oa.upm.es/96635/1/10529366.pdf"
# Bastas & Vouros, arXiv 2022
arxiv ATCoReactions_Bastas2022_data-driven_prediction_of_atco_reactions_to_resolving_conflicts 2205.09539
# Vouros, Papadopoulos, Bastas, Cordero & Rodrigez, arXiv 2022 (submitted to PAIS 2022)
arxiv DRLConflicts_Vouros2022_automating_the_resolution_of_flight_conflicts_drl_in_service_of_atcos 2206.07403
# Tolstaya, Ribeiro, Kumar & Kapoor, IROS 2019
arxiv InverseOptimalPlanning_Tolstaya2019_inverse_optimal_planning_for_air_traffic_control 1903.10525

# --- B. speech + surveillance: the voice side of the same instruction ------------------------------
# Kleinert, Helmke, Siol, Ehr, Finke, Oualil & Srinivasamurthy, 7th SESAR Innovation Days 2017 (no arXiv record)
get CommandPrediction_Kleinert2017_ml_of_controller_command_prediction_models_from_radar_data_and_speech "https://www.sesarju.eu/sites/default/files/documents/sid/2017/SIDs_2017_paper_27.pdf"
# Nigmatulina, Braun, Zuluaga-Gomez & Motlicek, arXiv 2021 (submitted to Interspeech 2021)
arxiv CallsignSurveillance_Nigmatulina2021_improving_callsign_recognition_with_air-surveillance_data 2108.12156
# Nigmatulina, Zuluaga-Gomez, Prasad, Sarfjoo & Motlicek, ICASSP 2022
arxiv TwoStepContextASR_Nigmatulina2022_two-step_approach_to_leverage_contextual_data_asr_in_atc 2202.03725
# Patrikar, Dantas, Moon, Hamidi, Ghosh, Keetha, Higgins, Chandak, Yoneyama & Scherer, Scientific Data 12:468 (2025)
arxiv TartanAviation_Patrikar2024_image_speech_and_adsb_trajectory_datasets_for_terminal_airspace 2403.03372

# --- C. Project Bluebird (NATS / Turing / Exeter): digital twin with recorded clearances -----------
# Pepper, Keane, Hodgkin, Gould, Henderson, Lauritsen, Vlahos, De Ath, Everson, Cannon, Sierra Castro, Korna,
# Carvell & Thomas, AIAA SciTech 2026, DOI 10.2514/6.2026-1794
arxiv ProbDigitalTwin_Pepper2026_probabilistic_digital_twin_of_uk_en_route_airspace_for_ai_atc_agents 2601.03113
# Keane, Pepper, Burr, Hodgkin, Gould, Korna & Thomas, AIAA SciTech 2026
arxiv DTAssurance_Keane2026_framework_for_assuring_accuracy_and_fidelity_of_ai-enabled_digital_twin 2601.03120
# Kent, De Ath, Layton, Hart, Everson & Carvell, AIAA SciTech 2026, DOI 10.2514/6.2026-1203
arxiv FutureCapabilitiesAgent_Kent2026_future_capabilities_agent_for_tactical_atc 2601.04285
# Carvell, Thomas, Pace, Dorney, De Ath, Everson, Pepper, Keane, Tomlinson & Cannon, AIAA SciTech 2026, DOI 10.2514/6.2026-2558
arxiv HITLTesting_Carvell2026_human-in-the-loop_testing_of_ai_agents_for_atc_regulated_assessment_framework 2601.04288

# --- D. language-model controllers ---------------------------------------------------------------
# Andriuskevicius & Sun, arXiv 2024
arxiv LLMAirTrafficAgents_Andriuskevicius2024_automatic_control_with_human-like_reasoning_llm_embodied_atc_agents 2409.09717
# Ghazanfari, Casanova, Kam, Zongo, Wei, Darrell & Bayen, arXiv 2026
arxiv LLMATC_Ghazanfari2026_air_traffic_control_using_llms_prompt_engineering_architecture_evaluation 2608.19299

# --- E. generative trajectory models (no instruction layer) ----------------------------------------
# Petit, Torun, Brusset, Kam & Bayen, arXiv 2026
arxiv FlowATC_Petit2026_aircraft_trajectory_prediction_via_flow_matching 2609.16528
# Larsen, Ruocco, Spitieris, Murad & Ragosta, arXiv 2025
arxiv LandAnywhere_Larsen2025_transferable_generative_models_for_aircraft_trajectories 2511.04155

# --- F. RL post-training of a pretrained trajectory model (non-aviation) ----------------------------
# Zhan, Li, Zhang, Lu & Li, PAKDD 2026
arxiv ShipTraj-R1_Zhan2026_reinforcing_ship_trajectory_prediction_in_llms_via_grpo 2603.02939
# Tang, Kan, Shan & Chen, ICLR 2026
arxiv Plan-R1_Tang2025_safe_and_feasible_trajectory_planning_as_language_modeling 2505.17659

# --- G. aviation closed-loop control without learned instructions ----------------------------------
# Li, Liu, Zeng & Jiang, arXiv 2025
arxiv DiffusionRL-CDR_Li2025_diffusion-rl_based_air_traffic_conflict_detection_and_resolution 2509.03550
# Zhang, Long, Huang, Zhang, Zhang & Yin, AI4AT workshop @ AAAI 2026
arxiv DataDrivenMPC-TMA_Zhang2025_data-driven_mpc_for_multi-aircraft_tma_routing_under_travel_time_uncertainty 2511.19452
