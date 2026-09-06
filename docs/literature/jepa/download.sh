#!/usr/bin/env bash
# Re-fetch every paper indexed in README.md from arXiv (2026-09-06 URLs).
# LeCun 2022 "A Path Towards Autonomous Machine Intelligence" is on OpenReview only and is
# behind a browser check; open https://openreview.net/pdf?id=BZ5a1r-kVsf by hand.
set -u
cd "$(dirname "$0")"
mkdir -p papers
get() { # get <destination> <arxiv id>
  if [ -s "papers/$1.pdf" ]; then echo "have $1"; return; fi
  curl -sL -f --retry 2 -o "papers/$1.pdf" "https://arxiv.org/pdf/$2" && echo "OK   $1" || { rm -f "papers/$1.pdf"; echo "FAIL $1  $2"; }
}
get I-JEPA_Assran2023_self-supervised_learning_from_images_with_JEPA 2301.08243
get V-JEPA2_Assran2025_self-supervised_video_models_enable_understanding_prediction_planning 2506.09985
get JointEmbeddingsGoTemporal_Ennadir2025_TS-JEPA 2509.25449
get PLDM_Sobal2025_learning_from_reward-free_offline_data_JEPA_planning 2502.14819
get Phys-JEPA_2026_physics-informed_latent_world_models_for_MTS_forecasting 2606.16076
get SkyJEPA_2026_long-horizon_world_models_for_zero-shot_sim-to-real_quadrotor_control 2606.23444
get Auto-JEPA_2026_latent_world_model_of_continuous_intent_for_end-to-end_driving 2607.29031
