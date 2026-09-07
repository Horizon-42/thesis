#!/usr/bin/env bash
# Cluster A — "hard constraints by construction and differentiable projection layers".
# Re-fetches every PDF stored under papers/ (PDFs are gitignored; this script is the record).
# Usage:  bash download_A.sh          (run from this directory)
set -u
cd "$(dirname "$0")"
mkdir -p papers
cd papers

UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
get() {  # get <url> <outfile>
  echo ">> $2"
  curl -sSL --max-time 120 -A "$UA" -o "$2" "$1" || echo "   FAILED: $1"
  sleep 3   # be polite to arXiv
}

# ---------------------------------------------------------------------------
# A1. Hard-constraint output layers (closed-form / ray / gauge / projection)
# ---------------------------------------------------------------------------
get https://arxiv.org/pdf/2307.08336 rayen_2307.08336.pdf                      # RAYEN
get https://arxiv.org/pdf/2410.10807 hardnet_2410.10807.pdf                    # HardNet
get https://arxiv.org/pdf/2508.10480 pinet_2508.10480.pdf                      # PInet / Pi-net (ICLR 2026)
get https://arxiv.org/pdf/2110.10333 tabas_zhang_gaugemap_2110.10333.pdf       # gauge map, ACC 2022
get https://arxiv.org/pdf/2203.12196 tabas_zhang_mpc_interiorpoint_2203.12196.pdf  # gauge-map MPC, CDC 2022
get http://jmlr.org/papers/volume25/23-1577/23-1577.pdf homeomorphic_projection_jmlr25.pdf  # Liang/Chen/Low

# ---------------------------------------------------------------------------
# A2. Differentiable projection / optimisation layers
# ---------------------------------------------------------------------------
get https://arxiv.org/pdf/2104.12225 dc3_2104.12225.pdf                        # DC3, ICLR 2021
get https://arxiv.org/pdf/1703.00443 optnet_1703.00443.pdf                     # OptNet, ICML 2017
get https://arxiv.org/pdf/1910.12430 cvxpylayers_1910.12430.pdf                # cvxpylayers, NeurIPS 2019
get https://arxiv.org/pdf/2402.07251 kkt_hpinn_2402.07251.pdf                  # KKT-hPINN, CACE 2024

# ---------------------------------------------------------------------------
# A3. Hard-vs-soft evidence, architectural constraints
# ---------------------------------------------------------------------------
get https://arxiv.org/pdf/1909.00912 beucler_analytic_constraints_1909.00912.pdf   # Beucler, PRL 2021
get https://arxiv.org/pdf/1706.02025 marquezneila_hard_constraints_1706.02025.pdf  # Marquez-Neila 2017
get https://arxiv.org/pdf/2211.01340 police_2211.01340.pdf                     # POLICE, Balestriero & LeCun

# ---------------------------------------------------------------------------
# A4. Trajectory-side constructive feasibility (corridors, Frenet, dynamics layers)
# ---------------------------------------------------------------------------
get https://fileadmin.cs.lth.se/ai/Proceedings/ICRA2010/MainConference/data/papers/1650.pdf werling2010_frenet_icra.pdf
get https://arxiv.org/pdf/1907.01531 fastplanner_bspline_1907.01531.pdf        # Zhou/Gao/Shen RA-L 2019 (B-spline hull)
get https://arxiv.org/pdf/2008.08835 egoplanner_2008.08835.pdf                 # EGO-Planner, RA-L 2021
get https://arxiv.org/pdf/1908.00219 cui_deep_kinematic_1908.00219.pdf         # Deep Kinematic Models
get https://arxiv.org/pdf/2001.03093 trajectronpp_2001.03093.pdf               # Trajectron++
get https://arxiv.org/pdf/2103.04027 prime_2103.04027.pdf                      # PRIME
get https://arxiv.org/pdf/1911.10298 covernet_1911.10298.pdf                   # CoverNet
get https://arxiv.org/pdf/2009.04450 map_adaptive_goal_2009.04450.pdf          # Map-Adaptive Goal-Based
get https://arxiv.org/pdf/2309.03750 pbp_pathbased_2309.03750.pdf              # PBP
get https://arxiv.org/pdf/2305.17965 frenet_domain_norm_2305.17965.pdf         # Frenet domain normalisation

# ---------------------------------------------------------------------------
# A5. Aviation
# ---------------------------------------------------------------------------
get https://arxiv.org/pdf/2203.09033 phased_flight_tp_2203.09033.pdf           # Zhang & Chen, phased flight TP

# ---------------------------------------------------------------------------
# NOT DOWNLOADABLE (recorded here so the gap is explicit; see notes_A.md)
#   Shi, Xu & Pan (2021), "4-D Flight Trajectory Prediction With Constrained LSTM
#     Network", IEEE T-ITS 22(11):7242-7255. DOI 10.1109/TITS.2020.3004807
#     -> IEEE paywall. Semantic Scholar reports openAccessPdf status BRONZE that
#        resolves to ieeexplore; no author or repository copy found.
#   Gao, Wu, Lin & Shen (2018), "Online Safe Trajectory Generation for Quadrotors
#     Using Fast Marching Method and Bernstein Basis Polynomial", ICRA 2018.
#     DOI 10.1109/ICRA.2018.8462878
#     -> Semantic Scholar reports openAccessPdf status CLOSED; the HKUST-Aerial-
#        Robotics/Btraj repo hosts code, not the paper. The same Bernstein/convex-
#        hull corridor mechanism is covered by fastplanner_bspline_1907.01531.pdf
#        and egoplanner_2008.08835.pdf, which ARE fetched above.
# ---------------------------------------------------------------------------

echo
echo "Done. Verify with:  file papers/*.pdf | grep -v 'PDF document'"
