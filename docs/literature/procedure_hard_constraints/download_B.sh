#!/usr/bin/env bash
# download_B.sh — cluster B of the procedure_hard_constraints literature:
# safety layers / action projection, provably-safe RL & shields, predictive safety
# filters and *training through* them, differentiable & discrete-time CBFs,
# safe imitation learning, residual policies, learned certificates.
#
# Never overwrites: a file that already exists is skipped (papers/ is shared with
# another collector).
#
# Usage:  bash download_B.sh
set -u
cd "$(dirname "$0")"
mkdir -p papers

get() {  # get <outfile> <url>
  local out="papers/$1" url="$2"
  if [[ -s "$out" ]]; then echo "skip (exists): $1"; return 0; fi
  echo "get: $1  <-  $url"
  curl -sSL --max-time 120 -A 'Mozilla/5.0 (literature-collector; thesis)' -o "$out.part" "$url" \
    && [[ "$(file -b --mime-type "$out.part")" == "application/pdf" ]] \
    && mv "$out.part" "$out" \
    || { echo "  FAILED: $1"; rm -f "$out.part"; }
}

ax() { get "$1" "https://arxiv.org/pdf/$2"; }   # ax <outfile> <arxiv-id>

# --- 1. Safety layers / action projection --------------------------------------
ax dalal2018_safe_exploration_1801.08757.pdf        1801.08757
ax pham2018_optlayer_1709.07643.pdf                 1709.07643
ax cheng2019_e2e_safe_rl_cbf_1903.08792.pdf         1903.08792

# --- 2. Provably safe RL, shields, safety-filter surveys ------------------------
ax krasowski2023_provably_safe_rl_2205.06750.pdf    2205.06750
ax alshiekh2018_shielding_1708.08611.pdf            1708.08611
ax hsu2023_safety_filter_unified_2309.05837.pdf     2309.05837
ax brunke2022_safe_learning_robotics_2108.06266.pdf 2108.06266

# --- 3. Predictive safety filters + training through a filter -------------------
ax wabersich2021_predictive_safety_filter_1812.05506.pdf 1812.05506
ax tearle2021_psf_racing_2102.11907.pdf                  2102.11907
ax pizarrobejarano2025_filtering_while_training_2410.11671.pdf 2410.11671
ax cbfrl_2510.14959.pdf                                  2510.14959
ax provably_optimal_rl_safety_filtering_2510.18082.pdf   2510.18082

# --- 4. Differentiable / discrete-time CBFs inside the policy ------------------
ax xiao2023_barriernet_2111.11277.pdf               2111.11277
ax xiao2023_safediffuser_2306.00148.pdf             2306.00148
ax zeng2021_mpc_dcbf_2007.11718.pdf                 2007.11718
get agrawal2017_discrete_cbf_rss13_p73.pdf "https://www.roboticsproceedings.org/rss13/p73.pdf"

# --- 5. Safe imitation learning + residual policies ----------------------------
ax cosner2022_e2e_il_cbf_2212.11365.pdf             2212.11365
ax yin2021_il_stability_safety_2012.09293.pdf       2012.09293
ax failsafe_agil_2203.01696.pdf                     2203.01696
ax silver2018_residual_policy_learning_1812.06298.pdf 1812.06298
ax johannink2019_residual_rl_1812.03201.pdf         1812.03201

# --- 6. Learned certificates / reachability filter -----------------------------
ax dawson2023_learned_certificates_2202.11762.pdf   2202.11762
ax fisac2019_general_safety_framework_1705.01292.pdf 1705.01292

echo "--- done ---"
ls -l papers/
