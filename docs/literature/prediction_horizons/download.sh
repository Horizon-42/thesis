#!/usr/bin/env bash
# Re-fetch every paper this folder cites that is NOT already in a sibling literature folder
# (2026-09-19 URLs). 6 PDFs. One curl per file.
#
# NOT fetched here on purpose: eight sources already live in sibling folders and are cited there
# by path (the rule multimodal_intent/download.sh uses). Run that folder's download.sh if one is
# missing:
#   ../hierarchical_prediction/  FlightBERT++ (Guo 2023), WTFTP (Zhang 2023), WTFTP+ (Guo 2026),
#                                TrajAirNet (Patrikar 2022)
#   ../multimodal_intent/        DeepTP (Liu & Hansen 2018), Pang, Xu & Liu (2019)
#   ../manoeuvre_tokens/         FTP-LLM (Luo & Zhou 2025)
#   ../procedure_hard_constraints/  Phased Flight TP (Zhang & Chen 2022)
#
# Two IEEE Access papers are open access; ieeexplore.ieee.org serves their PDFs from the
# stampPDF endpoint (the /iel7/ path returns HTML). The other four are arXiv.
# NOT reachable by curl (see README section "Not verified"): MDPI (Access Denied to curl),
# IEEE conference/TITS paywall, AIAA ARC, Elsevier.
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
ieee() { # ieee <destination stem> <arnumber>     (IEEE Access, open access)
  get "$1" "https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=$2&ref="
}

# --- terminal airspace / approach -------------------------------------------------------------
# Zeng, Quan, Zhao, Xie & Lu, IEEE Access 8 (2020) 151250-151266, 10.1109/ACCESS.2020.3016289
ieee  Zeng2020_deep_learning_approach_for_aircraft_trajectory_prediction_in_terminal_airspace 9166485
# Yoon & Lee, IEEE T-ITS 2026, 10.1109/TITS.2026.3689290 (MAIFormer)
arxiv MAIFormer_YoonLee2026_multi_agent_inverted_transformer_for_flight_trajectory_prediction 2509.21004
# Xiang & Chen, arXiv 2024 ("Submitted to AIAA-JAIS")
arxiv XiangChen2024_data_driven_probabilistic_trajectory_learning_high_temporal_resolution_terminal_airspace 2409.17359
# Huang, Zhang, Zhang, Zhang & Yin, SESAR Innovation Days 2023 (aircraft landing time)
arxiv Huang2023_aircraft_landing_time_prediction_with_deep_learning_on_trajectory_images 2401.01083

# --- en route ---------------------------------------------------------------------------------
# Ma & Tian, IEEE Access 8 (2020) 134668-134680, 10.1109/ACCESS.2020.3010963
ieee  MaTian2020_hybrid_cnn_lstm_model_for_aircraft_4d_trajectory_prediction 9145522
# Wu, Wang, Chu, Liu, Zhang & Wang, UAI 2025 (FlightPatchNet)
arxiv FlightPatchNet_Wu2025_multi_scale_patch_network_with_differential_coding 2405.16200
