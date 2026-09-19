# Input history length and prediction horizon in data-driven aircraft trajectory prediction (2026-09-19)

Why this folder exists. The thesis's related-work paragraph on **how much history a learned
trajectory predictor is given and how far ahead it is asked to predict** has to be written from the
papers' own experimental-setup text, not from memory. Our own setting is unusual on both axes: the
executor is a **closed loop** that flies each arrival from ~25 km to the runway threshold, taking
**60–120 s of observed history** and **re-planning every 20 s**, on ADS-B sampled at **2 s**; and
reading (c) of `4dTrajectory/ts_transformer/docs/2026-09-18_two_tier_v3_results.zh.md` §7 found
**no gain beyond 60 s of history** once every cell starts from the same row ("60 s 历史够用，更长
的历史在同一起点上没有超过 seed 线的收益"). So the questions this folder answers from primary text:

> Across the data-driven trajectory-prediction literature — terminal area / approach first, en
> route second — **what input history length, what prediction horizon, at what sampling interval,
> one-shot or rolling/recursive, with which metrics?** And: **does anyone close the loop and
> evaluate all the way to the runway?**

How to read it:
- **Quotes are copied** from the saved PDF text (`pdftotext`), each kept under about 30 words, with
  "…" marking a cut, and located by section / page / table or figure number. Where a PDF's text
  layer breaks a word across a line ("multihorizon"), the quote reproduces what `pdftotext`
  emitted.
- **"(reading)"** marks anything the text does not state: converting points × interval into
  seconds, classifying a paper's airspace, or calling an output one-shot when the paper only
  describes the architecture.
- **Nothing is filled from memory.** §4 *Not verified* lists every paper and every number that
  could not be confirmed, with its DOI.
- The PDFs are **not tracked in git** (root `.gitignore` has `*.pdf`). `./download.sh` re-fetches
  the **6** papers that are new to this repository (run 2026-09-19: all 6 returned, `%PDF-`
  verified). **Eight further sources already live in sibling folders and are NOT duplicated here**
  — they are cited by path in §1 (the rule `multimodal_intent/download.sh` uses).

---

## At a glance (only what the table below supports)

| | terminal area / approach rows (§1.1) | en route / whole-flight rows (§1.2) |
|---|---|---|
| sampling interval | **1 s** (TrajAir, Changi) → **6 s** (Incheon) → **10 s** (Florida) → 20 s raw-gap cap (Guangzhou, interval itself not stated) | **10 s** (OpenSky) → **20 s** (China ATC) → **1 min** (CAN/PEK/PVG; TFMS ≈ 2 min after down-sampling) |
| history length, stated in time | **11 s** (Patrikar; Xiang & Chen) · **60–90 s** image window (Huang) · **2 min** (Yoon & Lee) · **3 min** (Zhang & Chen) | **3 min** (Guo, FlightBERT++) · **10 min** (Wu, FlightPatchNet) · **16 min** (Luo & Zhou) |
| history length, points only | 48 points landing / 35 takeoff (Zeng — interval not stated, so no seconds) | 6 points (Ma & Tian — interval not stated); none at all in the two flight-plan models (Liu & Hansen; Pang, Xu & Liu) |
| prediction horizon | **90 s** (takeoff) / **150 s** (landing) (Zeng) · **120 s** (Patrikar; Xiang & Chen) · **2 min** (Yoon & Lee) · **5 min** (Zhang & Chen) · **whole arrival, as a scalar landing time** (Huang) | **20 s** single step (Zhang, WTFTP) · **10 s–2.5 min** (Wu) · **1/4/8 min** (Luo & Zhou) · **5 min** (Guo ×2) · **1/3/5 steps, interval unstated** (Ma & Tian) · **rest of the flight** (Liu & Hansen; Pang, Xu & Liu) |
| one-shot vs rolling | one-shot direct: Yoon & Lee, Patrikar, Xiang & Chen, Huang. Recursive: Zeng, Zhang & Chen | one-shot direct: Guo (FlightBERT++, WTFTP+), Wu, Luo & Zhou. Recursive: Zhang (WTFTP), Liu & Hansen. Not stated: Ma & Tian |
| metrics | ADE/FDE in km or m (Patrikar, Xiang & Chen, Zhang & Chen); EE/ATE/CTE/AE (Zeng); MAE/RMSE/MAPE per axis (Yoon & Lee); MAE in **seconds** (Huang) | MAE/RMSE/MAPE/MRE per axis + MDE in NM or km (Guo ×2, Zhang); MAE/RMSE (Wu, Luo & Zhou, Ma & Tian); point- and trajectory-wise horizontal/vertical error in NM/ft (Liu & Hansen) |
| closed loop to the runway | **no row predicts a trajectory to the threshold and scores it there.** The one paper whose target *is* the threshold (Huang) outputs a scalar landing time, not a trajectory | not applicable |

Two further observations the table supports:
- **The longest histories go with the longest sampling intervals, not with the finest data.** Every
  row with ≥ 3 min of history samples at 20 s, 10 s or 1 min; every row sampling at 1 s uses 11 s
  or 60–90 s of history. **(reading)** No paper in this table both samples finer than 10 s and
  looks back more than 2 min, which is the region our own 60–120 s at 2 s sits in.
- **Only one paper in the table states a reason for its horizon**: Yoon & Lee tie 2 minutes to the
  short-term conflict alert look-ahead (§2.2 quote). The rest state the number without a rationale.

---

## 1. The table

Airspace labels: **TMA** = terminal manoeuvring / terminal airspace around one airport;
**approach** = the paper isolates an arrival/approach phase; **en route** = cruise or whole-flight.
"→ s" conversions are **(reading)** unless the paper itself prints the seconds.

### 1.1 Terminal area / approach

| # | citation | data | history length | prediction horizon | metrics | closed loop / to the runway? |
|---|---|---|---|---|---|---|
| T1 | Weili Zeng, Zhibin Quan, Ziyu Zhao, Chao Xie, Xiaobo Lu, *A Deep Learning Approach for Aircraft Trajectory Prediction in Terminal Airspace*, **IEEE Access 8 (2020) 151250–151266**, DOI `10.1109/ACCESS.2020.3016289` | TMA, Guangzhou Baiyun (GBIA), **ADS-B** (VariFlight), 2018-08-01…11-30, 20 km × 20 km box, ≤ 2000 m. Trajectories reconstructed to "equal time intervals"; **the interval value is not stated** (raw gaps > 20 s delete the track) | "the length of encoding sequence is 48" (landing), "is 35" (takeoff) — **points; no seconds derivable** (§V-A) | "we set the lengths of the decoding sequences for takeoff and landing aircraft to 90 seconds and 150 seconds" (§V-A). **Recursive**: the decoder "outputs the 3D position sequence recursively until the end timestamp" (§II) | Euclidean error (EE), along-track (ATE), cross-track (CTE), altitude error (AE) (§V). Errors additionally broken out per 30-s bin of the horizon (Fig. 9) | **no** — fixed 150 s horizon inside a 20 km box; no threshold-referenced evaluation |
| T2 | Seokbin Yoon, Keumjin Lee, *Multi-Agent Inverted Transformer for Flight Trajectory Prediction* (MAIFormer), **IEEE T-ITS, 2026, pp. 1–11**, DOI `10.1109/TITS.2026.3689290`; arXiv:2509.21004 | TMA, Incheon, **within 70 NM**, **ADS-B** (FlightRadar24), **arrivals only**; "resampled all trajectories at a uniform 6-second interval" (§III-A) | "past air traffic scene X … over 2-minute intervals (i.e., 20 time steps at 6-second intervals)" (§III-B) | same 2 min / 20 steps ahead; **one-shot**: encoder-only, "rather than … autoregressive models" (§II). Window slides "with one time step (6 seconds)" | MAE, RMSE, MAPE, "applied separately to the latitude, longitude, and altitude dimensions" (§III-C) | **no** — but the horizon has a stated operational reason (see §2.2) |
| T3 | Jay Patrikar, Brady Moon, Jean Oh, Sebastian Scherer, *Predicting Like A Pilot: Dataset and Method to Predict Socially-Aware Aircraft Trajectories in Non-Towered Terminal Airspace* (TrajAir / TrajAirNet), **ICRA 2022**, DOI `10.1109/ICRA46639.2022.9811972`; arXiv:2109.15158 | Non-towered GA terminal airspace, **KBTP** (Pittsburgh-Butler), **ADS-B** (1090 + 978 MHz), 111 days; "Interpolating trajectory data every second for all agents" (§III-D) | "we use tobs = 11 sec" (§V) → **11 points × 1 s** (reading) | "tpred = 120 sec" (§V), "In order to focus on long-horizon predictions". **One-shot** CVAE decode of the whole 120 s | ADE / FDE **in km** (Table I); "Results for the best of N trajectories … We nominally use N = 5" (§V) | **no** — 120 s window inside the traffic pattern; no landing-referenced score |
| T4 | Jun Xiang, Jun Chen, *Data-driven Probabilistic Trajectory Learning with High Temporal Resolution in Terminal Airspace*, **arXiv:2409.17359** (2024-09-25; arXiv comment "Submitted to AIAA-JAIS") | Same **TrajAir** dataset (KBTP), 28-day subset, ADS-B at 1 s | "the length of past trajectory n = 11 sec" (§V) | "predicting length k = 120 sec … with 12 guided points", "guide time interval Δt = 10 sec" (§V); output at **1 timestep per second vs 0.1 timestep per second** (abstract). One-shot per guided segment | ADE, FDE (defined at the end of §IV); "We report the best ADE and FDE values of 5" runs (§V) | **no** |
| T5 | Liping Huang, Sheng Zhang, Yicheng Zhang, Yi Zhang, Yifang Yin, *Aircraft Landing Time Prediction with Deep Learning on Trajectory Images*, **13th SESAR Innovation Days (SIDS 2023)**; arXiv:2401.01083 | TMA, **Singapore Changi**, **ADS-B**, Nov 2022, 8396 arrivals; "the data sampling granularity is typically one positioning point per second" (§III-C); TMA Research Circle **radius 50 NM** | Image capture window "**τ = 60, δ = 10**" (Fig. 7 caption); ablation over "τ = 60, 90 (in seconds)" (§V). The image holds all aircraft inside the TRC in `[T_TRC − τ, T_TRC]` (§IV) | **The whole remaining arrival**: the target is `t_j = T_THR − T_TRC`, "the timestamps when the aircraft j arrives at the runway threshold and the TRC (50NM to the airport)" (§IV). **One-shot scalar**, predicted once at TRC entry | "mean absolute error (MAE) from 82.23 seconds to 43.96 seconds" (abstract); bad-prediction ratio, % of errors < 60 s | **Partly — the only row whose target is the runway threshold.** But the output is a landing *time*, not a trajectory, so nothing is flown or scored spatially |
| T6 | Kai Zhang, Bowen Chen, *Phased Flight Trajectory Prediction with Deep Learning*, **arXiv:2203.09033** (2022-03-17) | ADS-B from the authors' receivers, southeastern Florida, 2019; split into **climbing / cruising / approaching** sub-datasets. "The ADS-B system records data per 0.5 second … we scale the sampling rate to let one time-step corresponds to 10 seconds" (§6.2) | "we observe a trajectory for 3 minutes (T_obs = 18)" (§6.2) → 18 × 10 s | "predict is future trajectory for the next 5 minutes (T_pred = 48)" (§6.2). **Recursive**: the inference algorithm loops "for t = T_obs+1 to T_pred" | ADE (m), FDE (m), MAE per geo-coordinate (§6 *Evaluation Metrics*) | **no** — the approach phase gets its own model (st-graph over aircraft), but the horizon is a fixed 5 min |

### 1.2 En route / whole-flight

| # | citation | data | history length | prediction horizon | metrics | closed loop / to the runway? |
|---|---|---|---|---|---|---|
| E1 | Dongyue Guo, Zheng Zhang, Zhen Yan, Jianwei Zhang, Yi Lin, *FlightBERT++: A Non-autoregressive Multi-Horizon Flight Trajectory Prediction Framework*, **AAAI** (arXiv:2305.01658 is "An extend version based on the AAAI version"; the AAAI year is unverified, §4) | "a real-world flight trajectory dataset was collected from an ATC system in China … 9 days … with **20 seconds intervals**" (§IV-A), ROI lon [94.616, 113.689], lat [19.305, 37.275]. "the test set comprises a large proportion of en-route samples" (§V) | "we use the latest **3-minute** observations … predicting 15 trajectory points based on **9 observed trajectory points**" (§IV-C) | same sentence: "to predict the flight status of the future **5 minutes**". **One-shot**: "generate multihorizon predictions directly (non-autoregressive)" (§III). Horizons reported at "1, 3, 9, 15 … corresponding to 20 seconds, 1, 3, and 5 minutes" (§V-A) | MAE, MAPE, RMSE per attribute; MDE (3-D distance); MDE curve over horizons 1–15 (Fig. 5) | **no** |
| E2 | Zheng Zhang, Dongyue Guo, Shizhong Zhou, Jianwei Zhang, Yi Lin, *Flight trajectory prediction enabled by time-frequency wavelet transform* (WTFTP), **Nature Communications 14:5258 (2023)**, DOI `10.1038/s41467-023-40903-9` | Fused **SSR + ADS-B** from a Chinese ATC system, ~45 days. "The update interval of the trajectory in this dataset is **20 s**" (*Dataset and preprocessing*) | **Not stated in the main text**: the Methods call it "M … the number of historical trajectory points"; the value is in the Supplementary (not fetched) | **One step = 20 s**: the framework "reconstruct[s] the historical trajectory sequence and predict[s] the trajectory point for the next instant" (Results). Multi-horizon only by iteration — WTFTP+ describes it as "performs the multi-horizon [prediction] … [with] pseudo observations" (E3, §Introduction) | RMSE, MAE, MRE per attribute (Lon/Lat in degrees, Alt in 10 m) plus MDE in km (Table 1 footnote) | **no** — a case study shows an approach-phase holding circle (Fig. 6), but the score is still the next 20 s |
| E3 | Dongyue Guo, Zheng Zhang, Jiayi Liu, Jianwei Zhang, Yi Lin, *Multi-horizon flight trajectory prediction enabled by time-frequency wavelet transform* (WTFTP+), **Nature Communications 17:633 (2026)**, DOI `10.1038/s41467-025-67399-9` | Same Chinese ATC dataset: "covers the period from February 19 to February 27, 2021, with **20-second intervals**", ROI lon [94.616°, 113.689°], lat [19.305°, 37.275°], alt [0, 12500] m (*Dataset and evaluation metrics*) | **Not stated**: Eq. (5) writes the input as `P_{N−M:N−1}`; M is not given in the main text | "predictions for 1-, 3-, 9-, and 15-step horizons (corresponding to **20 s, and 1, 3, 5 min**)" (*Overall results*). **One-shot**: "outputs the trajectory states over the entire H-step prediction horizon in a single inference process" (Eq. 5 text) | MAE, MAPE, RMSE, MDE **in nautical miles**, plus mean time cost (MTC) (*Dataset and evaluation metrics*) | **no** |
| E4 | Lan Wu, Xuebin Wang, Ruijuan Chu, Guangyi Liu, Jing Zhang, Linyu Wang, *FlightPatchNet: Multi-Scale Patch Network with Differential Coding for Flight Trajectory Prediction*, **UAI 2025**; arXiv:2405.16200 | **OpenSky** ADS-B state vectors 2020–2022, 274,605 trajectories of 100 consecutive points. "The interval between two adjacent flight trajectory points is **10 seconds**" (Appendix A.1) | "look-back window **L = 60**" → "the observation time is **10 minutes**" (Appendix A.3) | "prediction horizon T ∈ {1, 3, 9, 15} … the forecasting time is **10 seconds, 30 seconds, 1.5 minutes, 2.5 minutes**" (Appendix A.3). **One-shot** ("DMS-based", direct multi-step, vs the "IMS" recursive baselines, §4.1) | MAE and RMSE per variable (§4.1) | **no** |
| E5 | Lan Ma, Shan Tian, *A Hybrid CNN-LSTM Model for Aircraft 4D Trajectory Prediction*, **IEEE Access 8 (2020) 134668–134680**, DOI `10.1109/ACCESS.2020.3010963` | En route, **Qingdao → Beijing** route, **ADS-B** Feb–May 2017, gaps filled by cubic spline. **Sampling interval not stated** | "selecting the time, longitude, latitude, altitude, velocity and heading of the first **6 trajectory points** to predict … the next trajectory point" (§V-A-1); "the separation interval S is selected as 1" | 1 step (single-step), or "**3 and 5** are selected as the step length in the multi-step prediction" → datasets D1, D2, D3 (§V). **Direct or recursive is not stated** | RMSE, MAE, MAPE (§V-B) | **no** |
| E6 | Yulin Liu, Mark Hansen, *Predicting Aircraft Trajectories: A Deep Generative Convolutional Recurrent Neural Networks Approach* (DeepTP), **arXiv:1812.11670** (2018-12-31); no venue on the PDF | En route, **IAH → BOS**, 2013, FAA **TFMS** tracks: resolutions "about 1 minute, 1 minute, 100 feet, and 1 minute"; "down sampled … by eliminating one out of every two track points" → **≈ 2 min (reading)**. 1,679 flights, "average sequence length of 94". Flights starting or ending outside the terminal-area boxes were excluded | **No observed-history window**: the encoder takes "the last filed flight plan" (≤ 16 characteristic points) and, at inference, "first T′ states (1 ≤ T′ ≤ T)" (§5.2) | **The rest of the flight**: "By recursively repeating the process until the last timestamp T, we can generate a whole flight trajectory after time T′" (§5.2), with beam search over samples. (reading) 94 points × ≈ 2 min ≈ 3 h | Point-wise horizontal error (NM) and vertical error (ft); trajectory-wise averages of each (§6) | **no** — the 0.5° terminal boxes are the endpoints, not a threshold |
| E7 | Kaiwei Luo, Jiliu Zhou, *Large Language Models for Single-Step and Multi-Step Flight Trajectory Prediction* (FTP-LLM), **arXiv:2501.17459** (2025-01-29); no venue on the PDF | **ADS-B** for flights at **CAN, PEK, PVG** (in- and outbound). "The raw ADS-B data, recorded at irregular intervals in seconds, are aggregated at the **minute** level"; "the time interval within each window is strictly constrained to **1 minute**" (§III-B) | "the window size is set to 17 for single-step prediction, consisting of **16 consecutive time steps as input**" (§III-B-2) → **16 min (reading)** | 1 step (1 min); "For multi-step prediction, the window size is set to 20 or 24, corresponding to **4 or 8 prediction steps**" (§III-B-2) → 4 / 8 min (reading). **One-shot** per prompt | MAE and RMSE (§IV-B) | **no** |
| E8 | Yutian Pang, Nan Xu, Yongming Liu, *Aircraft Trajectory Prediction using LSTM Neural Network with Embedded Convolutional Layer*, **PHM Society Conference 11(1), 2019**, DOI `10.36001/phmconf.2019.v11i1.849` (PDF already in `../multimodal_intent/papers/`) | En route, **JFK → LAX**, "three months history data over the period from Nov 1, 2018 through Feb 5, 2019", plus CIWS weather cubes ("dataset is updated every 150 seconds"; forecast every 300 s) | **No observed history**: the input is the filed flight plan plus weather. Preprocessing: "Interpolate FP and Tr with 1 second interval" then "Equally sample n points from FP and Tr" (Algorithm 1) | **The whole flight, prior to takeoff**: "addressing the issue of convective weather-related aircraft trajectory prediction prior to takeoff" (Conclusion) | Training loss MSE; result stated as "47.0% of the predicted flight tracks can reduce the deviation compared to the last on-file flight plan. The overall variance is reduced by 12.3%" (§Abstract) | **no** |

Where the eight non-duplicated PDFs live:

| row | path |
|---|---|
| T3 | `../hierarchical_prediction/papers/TrajAirNet_Patrikar2021_predicting_like_a_pilot_dataset_and_method_to_predict_socially-aware_aircraft_trajectories.pdf` |
| T6 | `../procedure_hard_constraints/papers/phased_flight_tp_2203.09033.pdf` |
| E1 | `../hierarchical_prediction/papers/FlightBERTpp_Guo2023_non-autoregressive_multi-horizon_flight_trajectory_prediction_framework.pdf` |
| E2 | `../hierarchical_prediction/papers/WTFTP_Guo2023_flight_trajectory_prediction_enabled_by_time-frequency_wavelet_transform.pdf` |
| E3 | `../hierarchical_prediction/papers/WTFTPplus_Guo2025_multi-horizon_flight_trajectory_prediction_enabled_by_time-frequency_wavelet_transform.pdf` |
| E6 | `../multimodal_intent/papers/DeepTP_LiuHansen2018_predicting_aircraft_trajectories_deep_generative_convolutional_recurrent_neural_networks.pdf` |
| E7 | `../manoeuvre_tokens/papers/LLM-FTP_Luo2025_large_language_models_for_single-step_and_multi-step_flight_trajectory_prediction.pdf` |
| E8 | `../multimodal_intent/papers/PangXuLiu2019_aircraft_trajectory_prediction_using_lstm_neural_network_with_embedded_convolutional_layer.pdf` |

T1, T2, T4, T5, E4, E5 are in this folder's `papers/` (see `download.sh`).

## 2. The quotes, per paper, with their location

### 2.1 T1 — Zeng et al. 2020 (IEEE Access), Guangzhou TMA

- §III-A *ADS-B dataset*: "It contains all monitoring data of landing and takeoff aircraft at
  Baiyun International Airport (GBIA) of China from August 1 to November 30, 2018." and "We keep
  only the position measurements that are less than 20 km in the ''east'' and ''north'' dimensions
  and less than 2000 meters in the ''up'' dimension."
- §III-A: "A trajectory is deleted if there are two adjacent track points between which the time
  interval exceeds 20 seconds." — **(reading)** this bounds the raw gap; it is not the resampling
  interval, which the paper never gives.
- §V-A (parameters): "hence, we set the lengths of the decoding sequences for takeoff and landing
  aircraft to 90 seconds and 150 seconds, respectively."
- §V-A: "(b) For landing aircraft, the length of encoding sequence is 48, the size of each batch of
  samples is 600, the number of hidden layers is 6"; for takeoff, "the length of encoding sequence
  is 35".
- §V *metrics*: "we use the Euclidean error (EE), the along-track error (ATE), the cross-track
  error (CTE) and the altitude error (AE) as metrics".
- Fig. 9 caption: "Boxplots of four metrics for landing aircraft in different time intervals" — the
  text reads them as "time interval of 0 to 30 seconds" and "the time interval of 30 to 60 seconds",
  i.e. per-horizon error.

### 2.2 T2 — Yoon & Lee 2026 (IEEE T-ITS), Incheon TMA — the only stated reason for a horizon

- §III-A: "The dataset spatially covers the terminal airspace within a radius of 70 nautical miles
  (NM) of Incheon International Airport"; "restricting the dataset to arrivals does not compromise
  the validity of our analysis."
- §III-A: "we resampled all trajectories at a uniform 6-second interval using the piecewise cubic
  Hermite interpolating polynomial (PCHIP)"; "The resampling interval of 6 seconds was selected as
  a proxy for the average update rate of approximately 5 seconds in the original ADS-B data".
- §III-B: "we constructed a past air traffic scene X and a future air traffic scene Y over 2-minute
  intervals (i.e., 20 time steps at 6-second intervals). This was done by sliding a window with one
  time step (6 seconds)".
- §III-B: "The prediction horizon of 2 minutes was motivated by operational considerations in air
  traffic control as it corresponds to the typical look-ahead time for short-term conflict alert
  (STCA) system for ATCs".
- §III-C: "we used the mean absolute error (MAE), root mean squared error (RMSE), and mean absolute
  percentage error (MAPE). MAE, RMSE, and MAPE were applied separately to the latitude, longitude,
  and altitude dimensions."

### 2.3 T3 / T4 — TrajAir (KBTP): 11 s in, 120 s out, twice

- Patrikar et al., §III-D *Data Processing*: "Interpolating trajectory data every second for all
  agents." and "Segmenting the data into scenes with at least one active" agent.
- Patrikar et al., §V: "In order to focus on long-horizon predictions, we use tobs = 11 sec and
  tpred = 120 sec."
- Patrikar et al., §V: "Results for the best of N trajectories are used where the network is
  queried N times, and the best ADE/FDE scores are recorded. We nominally use N = 5." Table I gives
  "ADE/FDE (in Km)".
- Xiang & Chen, §V (Results), setup paragraph: "we use the length of past trajectory n = 11 sec, guide time interval Δt = 10
  sec, and predicting length k = 120 sec. Therefore, the prediction model observes 11 seconds of the
  past trajectory and predicts 120 seconds of the future trajectory with 12 guided points."
- Xiang & Chen, abstract: "The trajectories generated by the proposed method have a higher temporal
  resolution(1 timestep per second vs 0.1 timestep per second) and are closer to the ground truth."
  — **(reading)** the 0.1/s baseline is TrajAirNet's own 12-point output of the same 120 s.

### 2.4 T5 — Huang et al. 2023 (SIDS), Changi: the only threshold-referenced target

- §I: "the TRC has a radius of 50 nautical miles (NM), which is sufficient to cover the TMA of the
  airport".
- §IV: "time calculated as tj = T_THR − T_TRC, where T_THR and T_TRC are respectively the timestamps
  when the aircraft j arrives at the runway threshold and the TRC (50NM to the airport). Both
  timestamps are extracted from aircraft trajectory data."
- §IV: "The track position points for all aircraft inside TRC and within the time window
  [T_TRC − τ, T_TRC] are collected to plot the image. Specifically, the trajectory of the target
  aircraft j are marked in red, and all other planes' trajectories are marked in blue."
- Fig. 7 caption: "τ = 60, δ = 10." §V: "of the window size parameters τ = 60, 90 (in seconds)"; the
  τ = 30 results "are quite worse".
- Abstract: "mean absolute error (MAE) from 82.23 seconds to 43.96 seconds" and "predictions errors
  being less than 60 seconds" (for the reported share).
- §III-C: "the data sampling granularity is typically one positioning point per second".

### 2.5 T6 — Zhang & Chen 2022 (arXiv), Florida, phase-split

- §6.2 *Implementation Details*: "The ADS-B system records data per 0.5 second so that the distance
  between two adjecent way-points is too close. Hence, we scale the sampling rate to let one
  time-step corresponds to 10 seconds."
- §6.2: "During test time, we observe a trajectory for 3 minutes (T_obs = 18) and predict is future
  trajectory for the next 5 minutes (T_pred = 48)."
- §3.1: "we focus on short to short-term trajectory prediction" and "the forecast period is
  typically less than 10 minutes."
- §6 *Evaluation Metrics*: "Average displacement error (ADE): the mean euclidean distance (in meter)
  over all predicted way-points and actual trajectory." plus FDE "(in meter)" and MAE per
  geo-coordinate.
- **(reading)** the approach phase gets the spatio-temporal graph model and en route the
  dual-attention model, but both are scored on the same 3 min → 5 min window.

### 2.6 E1 / E2 / E3 — the Chinese ATC line (FlightBERT++, WTFTP, WTFTP+): 20 s data, 5 min horizon

- FlightBERT++, §IV-A: "The dataset contains a total of 9 days of trajectory data with 20 seconds
  intervals from February 19 to 27, 2021".
- FlightBERT++, §IV-C: "In the training phase, we use the latest 3-minute observations to predict
  the flight status of the future 5 minutes, i.e., predicting 15 trajectory points based on 9
  observed trajectory points."
- FlightBERT++, §V-A: "the experimental results are divided into four (1, 3, 9, 15) different
  horizons, corresponding to 20 seconds, 1, 3, and 5 minutes trajectories in the future."
- FlightBERT++, §III: "the proposed framework can generate multihorizon predictions directly
  (non-autoregressive)". §V: "the test set comprises a large proportion of en-route samples".
- WTFTP, *Dataset and preprocessing*: "The update interval of the trajectory in this dataset is
  20 s." and "the raw flight trajectories are collected by multi-source Secondary Surveillance Radar
  (SSR) and Automatic Dependent Surveillance-Broadcast (ADS-B) from a real-world ATC system in
  China."
- WTFTP, Results: the framework performs "the IDWT procedure to reconstruct the historical
  trajectory sequence and predict the trajectory point for the next instant" — **single step**.
  Its history length M is defined in Methods ("M is the number of historical trajectory points")
  but never given a value in the main text.
- WTFTP+, *Dataset*: "The dataset covers the period from February 19 to February 27, 2021, with
  20-second intervals".
- WTFTP+, Results: "predictions for 1-, 3-, 9-, and 15-step horizons (corresponding to 20 s, and 1,
  3, 5 min)"; and the direct paradigm "outputs the trajectory states over the entire H-step
  prediction horizon in a single inference process".
- WTFTP+, Introduction, on why iteration was replaced: "The pseudo observations inevitably
  introduce substantial cumulative errors with horizon-wise propagations".

### 2.7 E4 — Wu et al., UAI 2025 (FlightPatchNet): the longest stated history

- Appendix A.1: "The interval between two adjacent flight trajectory points is 10 seconds." and "we
  select 100 consecutive points without missing values as a complete flight trajectory."
- Appendix A.3: "all the models follow the same experimental setup with look-back window L = 60 and
  prediction horizon T ∈ {1, 3, 9, 15}, which means the observation time is 10 minutes and the
  forecasting time is 10 seconds, 30 seconds, 1.5 minutes, 2.5 minutes."
- §4.1: the baselines are split into "five IMS-based models" (iterated) and "five DMS-based model"
  (direct), FlightPatchNet being direct; "We adopt the Mean Absolute Error (MAE) and Root Mean
  Squared Error (RMSE) as evaluation metrics."
- §4.1 baseline list is itself the cleanest confirmation that Shi et al. 2018 and Ma & Tian 2020 are
  the standard LSTM / CNN-LSTM baselines of this line: "LSTM [Shi et al., 2018], CNN-LSTM [Ma and
  Tian, 2020], Bi-LSTM [Sahadevan et al., 2022], FlightBERT [Guo et al., 2023], WTFTP [Zhang et al.,
  2023]".

### 2.8 E5 — Ma & Tian 2020 (IEEE Access)

- §V *Experiment*: "we use real ADS-B historical trajectory data from Qingdao to Beijing route for
  experiments" and "We use the trajectory data collected and decoded by ADS-B from February to May
  2017."
- §V-A-1: "selecting the time, longitude, latitude, altitude, velocity and heading of the first 6
  trajectory points to predict the time, longitude, latitude and altitude of the next trajectory
  point (y1 in the figure)"; "the separation interval S is selected as 1".
- §V: "we performed single-step and multi-step prediction on trajectory respectively, in which 3 and
  5 are selected as the step length in the multi-step prediction."
- §V-B: "Root Mean Square Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error
  (MAPE)".
- §III-B: the only statement about the time base is that gaps are filled "by interpolation methods"
  / "cubic spline interpolation" — **no sampling interval is given**.

### 2.9 E6 / E8 — the two flight-plan models: no observed history at all

- Liu & Hansen §3.3: "contains 4D positions of each aircraft throughout its flight … whose
  resolutions are respectively about 1 minute, 1 minute, 100 feet, and 1 minute"; "We then down
  sampled flight by eliminating one out of every two track points"; "The final dataset includes
  1,679 flights with an average sequence length of 94."
- Liu & Hansen §5.2: "we are trying to answer to what the rest of trajectory will be, if we know an
  aircraft's last filed flight plan, first T′ states (1 ≤ T′ ≤ T), and the corresponding T′
  weather-related feature cubes"; "By recursively repeating the process until the last timestamp T,
  we can generate a whole flight trajectory after time T′."
- Liu & Hansen §6: "Point-wise horizontal error (PHE). The distance (in nautical mile) between every
  predicted point's 2D coordinate … and the ground truth." plus PVE (ft), THE, TVE.
- Pang, Xu & Liu, Algorithm 1: "Interpolate FP and Tr with 1 second interval" / "Equally sample n
  points from FP and Tr as Tr_new and FP_new" — **n is not given**.
- Pang, Xu & Liu, abstract: "the out-of-sample test shows that 47.0% of the predicted flight tracks
  can reduce the deviation compared to the last on-file flight plan. The overall variance is reduced
  by 12.3%."

### 2.10 E7 — Luo & Zhou 2025 (FTP-LLM)

- §III-B-1: "The raw ADS-B data, recorded at irregular intervals in seconds, are aggregated at the
  minute level to ensure temporal consistency."
- §III-B-2: "the window size is set to 17 for single-step prediction, consisting of 16 consecutive
  time steps as input and 1 time step for prediction. For multi-step prediction, the window size is
  set to 20 or 24, corresponding to 4 or 8 prediction steps, respectively."
- §III-B-2: "The stride is set to be larger than the window size to prevent overlap and enhance data
  diversity."
- §IV-B: "Two commonly used metrics, Mean Absolute Error (MAE) and Root Mean Square Error (RMSE),
  are employed".

## 3. What this says for the thesis paragraph

Stated only as what §1 supports:

1. **Our 60–120 s of history is inside the published range, and at the fine-sampling end of it.**
   The terminal-area rows run from 11 s (T3, T4) through 60–90 s (T5) and 2 min (T2) to 3 min (T6);
   the en-route rows from 3 min (E1) to 16 min (E7). No row combines a ≤ 2 s sampling interval with
   a > 2 min history, so the reading-(c) result ("no gain beyond 60 s") is not contradicted by any
   measurement in this table — and no paper here measures the history length as a variable at all.
   **(reading)** every row states one history length and never sweeps it.
2. **Our 20 s re-plan interval is finer than any published horizon except WTFTP's single 20 s step**
   (E2) and FlightPatchNet's T = 1 and T = 3 (10 s, 30 s) (E4). Papers that re-predict at all
   re-predict by sliding the window one sample (T2: "sliding a window with one time step (6
   seconds)").
3. **One-shot direct multi-step is the current default** and is argued for explicitly against
   iteration (E3's "pseudo observations inevitably introduce substantial cumulative errors"; E1's
   non-autoregressive claim; E4's IMS/DMS split). The recursive rows are the older ones (T1, E6) and
   the phase-split model (T6).
4. **Nobody in this table flies an arrival to the threshold and scores it there.** The closest are
   T5 (the target *is* `T_THR − T_TRC`, but the output is one number) and T1 (a 20 km box that
   contains the runway, but a 150 s horizon). **(reading)** that makes the closed-loop-to-threshold
   evaluation a gap rather than a comparison — any claim that our numbers beat published ones would
   be comparing different tasks.
5. **Metrics split by community**: the robotics-derived rows report ADE/FDE (T3, T4, T6), the
   ATC/ATM rows report per-axis MAE/RMSE plus a 3-D distance (MDE) at named horizons (E1–E4), and
   the scheduling-flavoured one reports MAE in seconds (T5). A per-horizon breakdown is standard in
   the ATM line (E1 Fig. 5; T1 Fig. 9) and absent from the ADE/FDE line.

## 4. Not verified

Everything below is either unreachable primary text or a number the fetched text does not state.
Nothing from these items is used in §1–§3.

1. **Shi Zhiyuan, Xu Min, Pan Quan, Yan Bing, Zhang Haimin, *LSTM-based Flight Trajectory
   Prediction*, IJCNN 2018, pp. 1–8, DOI `10.1109/IJCNN.2018.8489734`.** The IEEE stampPDF endpoint
   returns 0 bytes (conference paywall). Its existence, author list and venue come from the Semantic
   Scholar record found by search **and** from FlightPatchNet's baseline citation "LSTM [Shi et al.,
   2018]" (E4 §4.1) — but **no history length, horizon, interval or metric was read**.
2. **Pang & Liu, the Bayesian / weather-uncertainty TRACON papers.** Recorded as paywalled with no
   preprint in `../multimodal_intent/README.md` §4 and not re-attempted here: *Probabilistic
   Aircraft Trajectory Prediction Considering Weather Uncertainties Using Dropout as Bayesian
   Approximate Variational Inference*, AIAA SciTech 2020, AIAA 2020-1413; and Pang, Zhao, Yan & Liu,
   *Data-driven trajectory prediction with weather uncertainties: A Bayesian deep learning
   approach*, **Transportation Research Part C 130 (2021) 103326**. A **CGAN** terminal-area paper by
   this group was **not located at all** — treat the brief's mention of one as unconfirmed.
3. **Zhang X. & Mahadevan S., "Bayesian neural networks for flight trajectory prediction and safety
   assessment", Decision Support Systems, 2020.** Not fetched and **not even checked** in this
   session (Elsevier; no open copy sought inside the time box). No DOI confirmed.
4. **FlightBERT (the original).** Not fetched. The brief's "IJCAI 2022" is **unverified**: the only
   evidence read here is FlightBERT++'s arXiv comment ("An extend version based on the AAAI
   version", which is about FlightBERT++, not FlightBERT) and FlightPatchNet's citation "FlightBERT
   [Guo et al., 2023]". The **AAAI year of FlightBERT++ itself is likewise unverified** — the arXiv
   record gives no `journal_ref`, and FlightPatchNet cites it as "[Guo et al., 2024]".
5. **Hong Sungkwon & Lee Keumjin, *Trajectory Prediction for Vectored Area Navigation Arrivals*,
   Journal of Aerospace Information Systems 12(7) (2015) 490–502, DOI `10.2514/1.I010245`.** AIAA ARC
   paywall; no text read. The brief's question — whether it predicts closed to the runway — is
   therefore **unanswered**.
6. **The conflict-detection look-ahead paper**: *Trajectory Predictor and Conflict Detection Figures
   of Merit for a Performance-Based Adaptive Air Traffic Monitoring System*, Aerospace 11(2):155,
   DOI `10.3390/aerospace11020155` (open access). `mdpi.com` answers `curl` with "Access Denied" for
   both the PDF and the HTML, so it was **not saved and not read**. The only en-route/conflict
   statement used in §1–§3 is Yoon & Lee's STCA sentence (T2), which is in a fetched PDF.
7. **WTFTP and WTFTP+ history length (M).** Neither main text gives a value; the Supplementary was
   not fetched. Their rows say "not stated" rather than guessing from FlightBERT++'s 9 points.
8. **Zeng et al. 2020 resampling interval.** The paper says the reconstruction yields "equal time
   intervals" and never gives the value, so its 48 / 35 encoding points **cannot** be converted to
   seconds. The 20 s figure in that paper is a maximum raw gap, not the interval.
9. **Ma & Tian 2020 sampling interval.** Not stated, so its 6-point history and 1/3/5-step horizons
   have no seconds. Whether its multi-step output is direct or recursive is also not stated.
10. **Liu & Hansen's effective interval.** "About 1 minute" TFMS resolution with every second point
    dropped is read here as ≈ 2 min; the paper never prints the resulting interval, and the 94-point
    average sequence length is a dataset statistic, not a model horizon.
11. **Xiang & Chen venue.** The arXiv comment says "Submitted to AIAA-JAIS"; a JAIS DOI turned up in
    search but was not opened, so the row cites the arXiv record only.
12. **No paper in §1 was read in full.** Only the data / experimental-setup / metric-definition /
    results-table sections were extracted, per the brief. Claims about what a paper does *not* do
    (e.g. "no closed-loop evaluation") are therefore about those sections.

## 5. Layout

| file | what |
|---|---|
| `README.md` | this index |
| `download.sh` | re-fetches the 6 PDFs new to this repository; lists the 8 cited by path in sibling folders |
| `papers/Zeng2020_…terminal_airspace.pdf` | T1, IEEE Access open access |
| `papers/MAIFormer_YoonLee2026_….pdf` | T2, arXiv:2509.21004 |
| `papers/XiangChen2024_…terminal_airspace.pdf` | T4, arXiv:2409.17359 |
| `papers/Huang2023_aircraft_landing_time_…images.pdf` | T5, arXiv:2401.01083 |
| `papers/MaTian2020_hybrid_cnn_lstm_….pdf` | E5, IEEE Access open access |
| `papers/FlightPatchNet_Wu2025_….pdf` | E4, arXiv:2405.16200 |
