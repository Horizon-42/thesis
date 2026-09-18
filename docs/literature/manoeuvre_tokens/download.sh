#!/usr/bin/env bash
# Re-fetch every paper indexed in README.md §1 (2026-09-18 URLs). 13 PDFs.
# Not fetchable by script (see README §4): nothing — every source in §1 came back. The two
# items that needed a mirror are noted below.
#
# Three notes on the mirrors:
#  * Olive & Morio (Aerospace Sci. Tech. 84, 2019) is paywalled at Elsevier and the ONERA
#    postprint on hal.science now sits behind an Anubis proof-of-work bot wall (every script
#    gets an HTML challenge page with HTTP 200, never the PDF). We take the Wayback snapshot
#    of the SAME HAL file under the old hal.archives-ouvertes.fr host (`id_` = raw bytes).
#    Verified against the Elsevier galley header "JID:AESCTE AID:4857" and the author line.
#  * MDPI returns HTTP 403 to scripts for /2504-3900/59/1/7/pdf, so Corrado et al. comes from
#    the Semantic Scholar PDF mirror. Verified against the "Proceedings 2020, 59, 7;
#    doi:10.3390/proceedings2020059007" footer on page 1.
#  * DAgger (Ross et al. 2011, arXiv 1011.0686) is NOT re-downloaded here: it already lives at
#    docs/literature/hierarchical_prediction/papers/DAgger_Ross2011_….pdf, which README §3
#    cites by that path. Run that folder's download.sh if it is missing.
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

# --- Cluster A: motion tokens + next-token prediction over multi-agent trajectories ----------
arxiv MotionLM_Seff2023_multi-agent_motion_forecasting_as_language_modeling 2309.16534
arxiv Trajeglish_Philion2023_traffic_modeling_as_next-token_prediction 2312.04535
arxiv SMART_Wu2024_scalable_multi-agent_real-time_motion_generation_via_next-token_prediction 2405.15677
# --- Cluster B: how the codebook is learned --------------------------------------------------
arxiv VQVAE_vandenOord2017_neural_discrete_representation_learning 1711.00937
arxiv FSQ_Mentzer2023_finite_scalar_quantization_vq-vae_made_simple 2309.15505
arxiv VQBeT_Lee2024_behavior_generation_with_latent_actions 2403.03181
# --- Cluster C: cross-agent edges, rollout exposure, constrained decoding --------------------
arxiv Graphormer_Ying2021_do_transformers_really_perform_bad_for_graph_representation 2106.05234
arxiv ScheduledSampling_Bengio2015_scheduled_sampling_for_sequence_prediction_with_recurrent_neural_networks 1506.03099
arxiv GCD_Geng2023_grammar-constrained_decoding_for_structured_nlp_tasks_without_finetuning 2305.13971
# --- Cluster D: what a "manoeuvre" is in terminal airspace (clustering + aircraft tokens) ----
get OliveMorio2019_trajectory_clustering_of_air_traffic_flows_around_airports "https://web.archive.org/web/2020id_/https://hal.archives-ouvertes.fr/hal-02350789/file/DTIS18300.1571067714_postprint.pdf"
get Corrado2020_trajectory_clustering_within_the_terminal_airspace_utilizing_a_weighted_distance_function "https://pdfs.semanticscholar.org/01bd/a8ba8a504bea822ceb4cf584482067f366eb.pdf"
arxiv TimeVQVAE-ATM_Murad2025_synthetic_aircraft_trajectory_generation_using_time-based_vq-vae 2504.09101
arxiv LLM-FTP_Luo2025_large_language_models_for_single-step_and_multi-step_flight_trajectory_prediction 2501.17459
