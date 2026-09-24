"""Cut verbatim from `autopilot/measure.py` (2026-09-24): the data measurement of r_turn and φ_cap — its docstring
lines, constants, the per-flight loop and the pooled values. Not importable; see README.md."""

# --- docstring lines
# - ``turn_rate_deg_s`` (r_turn): the turns the heading words describe — before the clearance, each run of rows turning
#   one way faster than the turn onset rate (`labeller.lateral.turn_runs`) — of at least `TURN_MIN_DEG`, their
#   middle half (the first and last quarter dropped: the 15 s velocity fit smears the roll-in and roll-out); each
#   turn's steady rate is the median of its middle rows, and r_turn the median over turns — a typical turn's steady
#   rate, every turn counted once whatever its length;
# - ``bank_cap_deg`` (φ_cap): the same middle rows at ground speeds in `FAST_BAND_MPS`, each turn's median
#   bank there, the median over the turns that have such rows, up to a whole degree, never past the
#   vocabulary's bank ceiling;

# --- constants
TURN_MIN_DEG = 90.0
FAST_BAND_MPS = (115.0, 140.0)


# --- in flight_measurements
#     turning = np.diff(track[: reading.join_row + 1]) / spec.step_s
#     for start, stop in turn_runs(turning, spec.turn_onset_rate_deg_s):
#         if abs(track[stop] - track[start]) < TURN_MIN_DEG:
#             continue
#         quarter = (stop - start) // 4
#         middle = slice(start + quarter, stop - quarter)
#         rate = np.abs(turning[middle])
#         middle_speed = speed[1:][middle]
#         out["turn_steady_rate_deg_s"].append(float(np.median(rate)))
#         fast = (middle_speed >= FAST_BAND_MPS[0]) & (middle_speed <= FAST_BAND_MPS[1])
#         if fast.any():
#             out["turn_fast_bank_deg"].append(float(np.median(
#                 envelope.bank_deg_from_turn_rate(rate[fast], middle_speed[fast]))))

# --- in measured_values
#         "turn_rate_deg_s": rounded(float(np.median(pooled["turn_steady_rate_deg_s"])), 0.05, round),
#         "bank_cap_deg": min(spec.turn_bank_max_deg, rounded(float(np.median(pooled["turn_fast_bank_deg"])), 1.0,
#                                                            math.ceil)),
