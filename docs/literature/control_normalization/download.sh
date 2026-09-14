#!/usr/bin/env bash
# Re-fetch every paper indexed in README.md §1 (2026-09-14 URLs). 31 PDFs.
# Not fetchable by script (see README §4): Alligier & Gianazza 2018 TR-C (HAL bot check; open
# https://enac.hal.science/hal-01878615/file/trc2018.pdf by hand), Sun et al. ICRAT 2018
# (https://research.tudelft.nl/files/103003225/ICRAT_2018_paper_19_1_.pdf returns 403 to scripts),
# and the paywalled items (Mellinger & Kumar 2011, Gallo et al. 2006, Hof 1996, Alexander & Jayes 1983,
# Moisio et al. 2003, Pierrynowski & Galea 2001).
set -u
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
arxiv() { # arxiv <destination> <arxiv id>   (3 s pause between arXiv requests)
  if [ -s "papers/$1.pdf" ]; then echo "have $1"; return; fi
  get "$1" "https://arxiv.org/pdf/$2"; sleep 3
}
# --- 1.1 aviation / ATM --------------------------------------------------------------------
get BADA-overview_Nuic2010_advanced_aircraft_performance_model_for_ATM "https://www.eurocontrol.int/sites/default/files/2019-03/overview-bada-apm.pdf"
get BADA3-UM_EUROCONTROL2009_user_manual_BADA_revision_3.7 "https://www.eurocontrol.int/sites/default/files/library/003_BADA_3_7_User_manual.pdf"
get MassThrustLearning_Alligier2013_learning_aircraft_mass_and_thrust_ground-based_climb_prediction "https://core.ac.uk/download/78384628.pdf"
get ClimbMLMass_Alligier2015_machine_learning_and_mass_estimation_methods_climb_prediction "https://core.ac.uk/download/129780535.pdf"
get MassBayes_Sun2018_aircraft_initial_mass_estimation_using_bayesian_inference "https://repository.tudelft.nl/file/File_68a746e2-b750-4546-8ecf-4f646a4914b1"
get OpenAP_Sun2020_open-source_aircraft_performance_model "https://repository.tudelft.nl/file/File_cf301e95-a762-449e-8511-bd4c55eb2481"
arxiv ClimbGenerative_Pepper2024_learning_generative_models_for_climbing_aircraft_from_radar_data 2309.14941
arxiv DescentPIML_Hodgkin2025_probabilistic_simulation_of_aircraft_descent_physics-informed_ML 2504.02529
arxiv MetConditioning_Hodgkin2026_conditioning_aircraft_TP_on_meteorological_data_physics-informed_ML 2601.03152
# --- 1.2 quadrotors / UAVs -----------------------------------------------------------------
arxiv RotorDragFlatness_Faessler2018_differential_flatness_of_quadrotor_dynamics_subject_to_rotor_drag 1712.02402
arxiv DeepDroneAcrobatics_Kaufmann2020_deep_drone_acrobatics 2006.05768
arxiv QuadBenchmark_Kaufmann2022_benchmark_comparison_of_learned_control_policies_for_agile_quadrotor_flight 2202.10796
get Swift_Kaufmann2023_champion-level_drone_racing_using_deep_RL "https://www.nature.com/articles/s41586-023-06419-4.pdf"
arxiv XAdapt_Zhang2024_learning-based_quadcopter_controller_with_extreme_adaptation 2409.12949
arxiv RAPTOR_Eschmann2025_foundation_policy_for_quadrotor_control 2509.11481
# --- 1.3 robotics cross-embodiment ---------------------------------------------------------
get OSF_Khatib1987_unified_approach_motion_force_control_operational_space_formulation "https://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf"
arxiv UP-OSI_Yu2017_preparing_for_the_unknown_universal_policy_with_online_system_identification 1702.02453
arxiv RMA_Kumar2021_rapid_motor_adaptation_for_legged_robots 2107.04034
arxiv MetaMorph_Gupta2022_learning_universal_controllers_with_transformers 2203.11931
arxiv OXE_OpenXEmbodiment2023_robotic_learning_datasets_and_RT-X_models 2310.08864
arxiv Octo_OctoModelTeam2024_open-source_generalist_robot_policy 2405.12213
arxiv OpenVLA_Kim2024_open-source_vision-language-action_model 2406.09246
arxiv HPT_Wang2024_scaling_proprioceptive-visual_learning_with_heterogeneous_pretrained_transformers 2409.20537
arxiv CrossFormer_Doshi2024_scaling_cross-embodied_learning_manipulation_navigation_locomotion_aviation 2408.11812
arxiv RDT-1B_Liu2024_diffusion_foundation_model_for_bimanual_manipulation 2410.07864
# --- 1.4 driving / multi-agent trajectory prediction ---------------------------------------
arxiv Trajectron++_Salzmann2020_dynamically-feasible_trajectory_forecasting_with_heterogeneous_data 2001.03093
arxiv DKM_Cui2020_deep_kinematic_models_for_kinematically_feasible_vehicle_trajectory_predictions 1908.00219
# --- 1.6 scientific ML (1.5 biomechanics: all paywalled, see README §4) ----------------------
arxiv BuckinghamPi_Bakarji2022_dimensionally_consistent_learning_with_buckingham_pi 2202.04643
get DimensionlessLearning_Xie2022_discovery_of_dimensionless_numbers_and_governing_laws_from_scarce_measurements "https://www.nature.com/articles/s41467-022-35084-w.pdf"
arxiv UnitsEquivariance_Villar2023_dimensionless_machine_learning_imposing_exact_units_equivariance 2204.00887
arxiv ClimateInvariant_Beucler2024_climate-invariant_machine_learning 2112.08440
