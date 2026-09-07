#!/usr/bin/env bash
# Cluster C — procedure / hard-constraint literature for learned 4D arrival-trajectory prediction.
#
# Re-downloads every PDF referenced by notes_C.md into ./papers/.
# Idempotent: existing files are skipped. Only touches the filenames listed below.
#
#   bash download_C.sh
#
# Two papers in notes_C.md are NOT downloadable (publisher paywall, no preprint found
# on arXiv or an author page as of 2026-09-07). They are listed at the bottom with DOIs.

set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/papers"
mkdir -p "$DIR"
UA="Mozilla/5.0 (literature collection; academic use)"

dl() {
  local id="$1" name="$2"
  if [ -s "$DIR/$name" ]; then echo "skip   $name"; return 0; fi
  local code
  code=$(curl -sSL -m 300 -A "$UA" -o "$DIR/$name" -w "%{http_code}" "https://arxiv.org/pdf/$id")
  if [ "$code" = "200" ] && head -c 4 "$DIR/$name" | grep -q '%PDF'; then
    echo "ok     $name  ($(stat -c%s "$DIR/$name") bytes)"
  else
    rm -f "$DIR/$name"; echo "FAIL   $name  (http=$code)"
  fi
}

# ---------------------------------------------------------------- 1. constrained learning
dl 2006.05487v2 chamon2020_pac_constrained_learning.pdf
dl 2103.05134v5 chamon2023_constrained_learning_nonconvex_losses.pdf
dl 2306.02426v4 hounie2023_resilient_constrained_learning.pdf
dl 2001.09394v2 fioretto2020_lagrangian_duality_constrained_dl.pdf
dl 2007.03964v1 stooke2020_pid_lagrangian.pdf
dl 2406.04558v1 sohrabi2024_nupi_multiplier_controller.pdf
dl 1904.04205v5 kervadec2022_log_barrier_extensions.pdf
dl 1705.10528v1 achiam2017_cpo.pdf
dl 2208.04425v2 gallegoposada2022_controlled_sparsity.pdf
dl 2504.01212v1 gallegoposada2025_cooper_library.pdf
dl 2102.12894v4 sangalli2021_constrained_opt_underrepresented.pdf

# ---------------------------------------------------------------- 2. aviation trajectory prediction
dl 2601.03152v2 hodgkin2026_physics_informed_met_tp.pdf
dl 2309.14941v2 pepper2023_generative_climbing_bada.pdf
dl 2509.23307v1 jarry2025_node_fdm.pdf
dl 2603.16550v1 prutsch2026_ascent_terminal_tp.pdf
dl 2109.15158v2 patrikar2022_predicting_like_a_pilot.pdf
dl 2305.01661v2 guo2023_spoken_instructions_tp.pdf
dl 2509.21004v3 yoon2025_multiagent_itransformer_tp.pdf
dl 2409.17359v1 xiang2024_probabilistic_tp_terminal.pdf

# ---------------------------------------------------------------- 3. predict-then-optimize / amortized
dl 2202.00665v5 amos2023_amortized_optimization_tutorial.pdf
dl 2212.08260v1 sambharya2023_learn_warmstart_qp.pdf
dl 1810.13400v3 amos2018_differentiable_mpc.pdf
dl 1710.08005v5 elmachtoub2022_spo.pdf
dl 2310.13831v3 guffanti2024_art_transformer_trajopt_warmstart.pdf
dl 2312.14336v2 briden2023_constraint_informed_warmstart.pdf

# ---------------------------------------------------------------- 4. constrained generative / diffusion
dl 2205.09991v2 janner2022_diffuser.pdf
dl 2306.03083v1 jiang2023_motiondiffuser.pdf
dl 2402.03559v3 christopher2024_projected_diffusion.pdf
dl 2403.05571v4 li2024_diffusolve.pdf
dl 2306.00148v1 xiao2023_safediffuser.pdf

# ---------------------------------------------------------------- 5. gating / goal / phase
dl 2008.08294v2 zhao2020_tnt.pdf
dl 2108.09640v2 gu2021_densetnt.pdf
dl 2409.11172v3 xu2025_annealed_wta.pdf

cat <<'EOF'

--------------------------------------------------------------------------
NOT DOWNLOADABLE (paywalled; no preprint found 2026-09-07). Cited by DOI:

  Shi, Z., Xu, M., Pan, Q. (2021). "4-D Flight Trajectory Prediction With
  Constrained LSTM Network." IEEE T-ITS 22(11):7242-7255.
  DOI 10.1109/TITS.2020.3004807     (ieeexplore.ieee.org/document/9136843)

  Pang, Y., Zhao, X., Yan, H., Liu, Y. (2021). "Data-driven trajectory
  prediction with weather uncertainties: A Bayesian deep learning approach."
  Transportation Research Part C 130:103326.
  DOI 10.1016/j.trc.2021.103326
--------------------------------------------------------------------------
EOF
