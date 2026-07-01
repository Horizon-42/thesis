"""Multiphase direct-collocation optimiser for a constrained approach.

A parallel transcription to the single-phase ``CasadiDirectCollocationOptimizer``, used **only**
when a procedure's path constraints are enforced. It replaces the single-phase + along-track
``partition`` (which assigned time-spaced nodes to legs by an index-fraction approximation) with a
proper **multiphase** formulation:

    phase 0:        start_state → IAF            (free transition; only the endpoint is pinned)
    phase 1..P:     fix_{j-1} → fix_j            (one phase per procedure leg, with its corridor /
                                                  glidepath / step-down constraints)

Every phase has its **own free duration** ``T_p``; phases are linked by full state continuity, and
each phase boundary pins the trajectory **through the fix** (position) — so every phase's nodes lie
*exactly* on their leg and get exactly that leg's constraints (no partition, no ``spans``). The
last phase pins the full target state. It is ONE NLP (block-structured by phase), solved jointly,
minimising ``Σ T_p`` (+ a small control regulariser/smoothness).

Reuses the single-phase module's building blocks (dynamics scheme, normalization, bounds, the
terminal-bank constraint, control costs, the solver factory) and the ``approach_constraints``
package's backend-agnostic ``segment_violations_from_components``.
"""

from __future__ import annotations

import math
import time
from collections import namedtuple

import casadi as ca
import numpy as np

from aircraft.aircraft_sets import Aircraft
from aircraft.aero_params import aero_params_for_aircraft
from aerodynamic_model.common import GeodeticState

import casadi_direct_collocation_optimizer as _dc

STATE_DIM = _dc.STATE_DIM
CONTROL_DIM = _dc.CONTROL_DIM
_DEFAULT_SCHEME = "hermiteSimpsonNormalizedFullTransport"
_INF = 1e9

# The FAF-intercept spec for the one leg that leads INTO a protected final approach:
# which phase it is, and the final-approach course to align with there.
_FafIntercept = namedtuple("_FafIntercept", ("phase", "course_rad"))


class MultiphaseCollocationOptimizer:
    """Constrained-approach optimiser: one phase per leg, free per-phase time, fixes pinned."""

    def __init__(
        self,
        aircraft: Aircraft,
        segments: list,
        *,
        scheme: str = _DEFAULT_SCHEME,
        n_seg_per_phase: int = 4,
        state_substeps: int = 3,
        max_terminal_bank_deg: float = _dc._DEFAULT_MAX_TERMINAL_BANK_DEG,
        smoothness_weights: tuple = _dc._DEFAULT_SMOOTHNESS_WEIGHTS,
        min_speed_ms: float | None = None,
        max_intercept_deg: float = 30.0,
        solver_backend: str = "ipopt",
        verbose: bool = False,
    ):
        if scheme not in _dc._NORMALIZED_FULL_TRANSPORT_SCHEMES:
            raise ValueError(
                "multiphase constraints require a normalized full-transport scheme "
                f"{sorted(_dc._NORMALIZED_FULL_TRANSPORT_SCHEMES)}; got {scheme!r}"
            )
        if not segments:
            raise ValueError("MultiphaseCollocationOptimizer needs at least one segment (leg)")
        if solver_backend != "ipopt":
            raise ValueError("multiphase constraints require the 'ipopt' solver_backend")
        if n_seg_per_phase < 1 or state_substeps < 1:
            raise ValueError("n_seg_per_phase and state_substeps must be >= 1")

        self.aircraft = aircraft
        self.segments = segments
        self.scheme = scheme
        self.n_seg_per_phase = n_seg_per_phase
        self.state_substeps = state_substeps
        self.max_terminal_bank_deg = max_terminal_bank_deg
        self.smoothness_weights = smoothness_weights
        # Velocity floor over the whole trajectory; defaults to the aircraft's Vref (see
        # _aircraft_meta). Pass a lower stall-margin floor for observed touchdown-speed targets.
        self.min_speed_ms = min_speed_ms
        # Max heading-vs-final-approach-course angle allowed at the FAF (the published intercept).
        self.max_intercept_deg = max_intercept_deg
        self.solver_backend = solver_backend
        self.verbose = verbose
        self.aero_params = aero_params_for_aircraft(aircraft)
        # Number of control segments overall (response/playback are aligned to these).
        self.n_segments = len(segments) * n_seg_per_phase
        # Per-control-segment durations of the most recent solve (for non-uniform playback/rollout).
        self.segment_durations_s: list[float] | None = None
        # The DENSE collocation nodes of the most recent solve (every node, geodetic degrees) —
        # a smooth planned trajectory for export; None until the first solve.
        self.last_dense_states_geo: np.ndarray | None = None
        self.last_solve_timings: dict[str, float] | None = None

    # ------------------------------------------------------------------
    def optimize_free_time(
        self,
        initial_state: GeodeticState,
        target_state: GeodeticState,
        max_duration: float,
        initial_guess=None,
    ):
        """Build + solve the multiphase NLP; returns ``(final_time, controls (Nc,3), states (Nc,6))``.

        ``states`` are geodetic degrees at the control-segment endpoints (aligned with ``controls``).
        The matching per-segment durations are stored on ``self.segment_durations_s``.
        """
        started = time.perf_counter()
        nlp, lbw, ubw, lbg, ubg, x0, layout = self._build(initial_state, target_state, max_duration)
        solver = _dc._make_nlp_solver(nlp, self.solver_backend, self.verbose)
        sol = solver(
            x0=(x0 if initial_guess is None else list(initial_guess)),
            lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg,
        )
        if not solver.stats()["success"]:
            raise ValueError(
                "multiphase optimization failed: "
                f"{solver.stats().get('return_status', 'unknown')}"
            )
        elapsed = time.perf_counter() - started
        self.last_solve_timings = {"solveTotalS": elapsed, "coldStartS": 0.0, "freeTimeSolveS": elapsed}
        return self._extract(np.array(sol["x"]).reshape(-1), layout)

    # ------------------------------------------------------------------
    def _build(self, initial_state, target_state, max_duration):
        make_dynamics, make_defect = _dc._DEFECT_SCHEMES[self.scheme]
        dynamics = make_dynamics()
        ip = _dc._geodetic_state_to_decision(initial_state)
        tp = _dc._unwrap_target_heading(ip, _dc._geodetic_state_to_decision(target_state))
        target_dm = ca.DM(tp)
        c, b = _dc._normalization_cb(target_dm)
        defect = make_defect(dynamics, target_dm)
        c_np = np.array(c).reshape(-1)[:STATE_DIM]
        b_np = np.array(b).reshape(-1)[:STATE_DIM]
        to_z = lambda phys: (np.asarray(phys, float) - b_np) / c_np
        init_z = to_z(ip[:STATE_DIM])
        tgt_z = to_z(tp[:STATE_DIM])
        mass = ca.DM(initial_state.m)
        aero = ca.vertcat(
            self.aero_params.S, self.aero_params.Cl_max, self.aero_params.Cd0,
            self.aero_params.k, self.aero_params.stall_threshold, self.aero_params.k_stall,
        )
        meta = self._aircraft_meta()
        state_lb, state_ub = _dc._normalized_position_bounds(
            *_dc.make_state_bounds(meta["min_altitude"], meta["min_terminal_speed"]), self.scheme
        )
        control_lb, control_ub = _dc.make_control_bounds(
            meta["max_thrust"], meta["min_load_factor"], meta["max_load_factor"]
        )

        # Phase plan — one phase per leg (phase p == segment p); the first phase starts at the
        # initial state. Constraint model (only the LPV final is laterally protected):
        #   * pre-final legs    — NO horizontal pin, NO lateral corridor; only altitude (floor +
        #     descent cap). The horizontal path to the FAF is free, so an irregular coded transition
        #     (e.g. a 90° turn at the FAF) no longer forces an unflyable pinned dog-leg.
        #   * the leg INTO the FAF — pin the FAF position + require the heading there to be within
        #     ``max_intercept_deg`` of the final approach course (a standard final intercept).
        #   * the final LPV leg(s) — full corridor + glidepath; the last leg pins the threshold.
        faf = self._faf_intercept()     # None | _FafIntercept(phase, course_rad); see the method
        intercept_floor = math.cos(math.radians(self.max_intercept_deg))

        phase_plan = [(np.asarray(s.end_ne, float), s) for s in self.segments]
        n_phases = len(phase_plan)
        n_seg, m_sub = self.n_seg_per_phase, self.state_substeps
        total_dist = self._cumulative_dist(init_z[:2], [pf for pf, _ in phase_plan])

        w, lbw, ubw, x0 = [], [], [], []
        g, lbg, ubg = [], [], []
        durations, phase_nodes, all_controls = [], [], []
        start_z = ca.DM(init_z)
        start_np = init_z.copy()
        cum = 0.0
        for p in range(n_phases):
            end_fix, seg = phase_plan[p]
            last = p == n_phases - 1
            Tp = ca.SX.sym(f"T_{p}")
            durations.append(Tp)
            leg = float(np.hypot(end_fix[0] - start_np[0], end_fix[1] - start_np[1]))
            state_h = (Tp / n_seg) / m_sub
            end_alt = tgt_z[2] if last else (
                init_z[2] + (tgt_z[2] - init_z[2]) * ((cum + leg) / total_dist if total_dist > 0 else 1.0)
            )
            end_guess = np.array([end_fix[0], end_fix[1], end_alt, tp[3], tp[4], tp[5]])

            nodes, controls, x_prev = [], [], start_z
            for k in range(n_seg):
                uk = ca.SX.sym(f"u_{p}_{k}", CONTROL_DIM)
                controls.append(uk)
                for j in range(m_sub):
                    xn = ca.SX.sym(f"x_{p}_{k}_{j}", STATE_DIM)
                    nodes.append(xn)
                    g.append(defect(ca.vertcat(x_prev, mass), ca.vertcat(xn, mass), uk, aero, state_h)[:STATE_DIM])
                    lbg += [0.0] * STATE_DIM
                    ubg += [0.0] * STATE_DIM
                    x_prev = xn

            # boundary pin: full target (last leg) | FAF position + intercept angle (leg into the
            # FAF) | nothing (pre-final legs — horizontal free, only altitude is constrained below)
            if last:
                g.append(nodes[-1] - ca.DM(tgt_z))
                lbg += [0.0] * STATE_DIM
                ubg += [0.0] * STATE_DIM
            elif faf is not None and p == faf.phase:
                g.append(nodes[-1][0:2] - ca.DM(end_fix))
                lbg += [0.0, 0.0]
                ubg += [0.0, 0.0]
                # heading at the FAF within max_intercept of the FAC (smooth: cos form, no kink):
                #   cos(intercept) − cos(psi_FAF − FAC) ≤ 0  ⇔  |psi_FAF − FAC| ≤ intercept
                g.append(intercept_floor - ca.cos(nodes[-1][_dc._PSI] - faf.course_rad))
                lbg.append(-_INF)
                ubg.append(0.0)

            # leg path constraints: only the protected LPV final keeps the lateral corridor; the
            # pre-final legs contribute altitude (floor + descent cap) only.
            if seg is not None:
                n_vec = ca.vertcat(*[nd[0] for nd in nodes])
                e_vec = ca.vertcat(*[nd[1] for nd in nodes])
                h_vec = ca.vertcat(*[nd[2] for nd in nodes])
                gamma_vec = ca.vertcat(*[nd[5] for nd in nodes])
                import approach_constraints as ac  # lazy
                viol = ac.segment_violations_from_components(
                    seg, n_vec, e_vec, h_vec, gamma_vec, include_lateral=(seg.lpv is not None)
                )
                for expr in viol.values():
                    g.append(expr)
                    rows = int(expr.shape[0])
                    lbg += [-_INF] * rows
                    ubg += [0.0] * rows

            # register decision variables, bounds, and the linear guess
            w += controls
            lbw += control_lb * n_seg
            ubw += control_ub * n_seg
            w += nodes
            lbw += state_lb * len(nodes)
            ubw += state_ub * len(nodes)
            for _ in range(n_seg):
                x0 += [self.aircraft.approach.thrust_guess_n, 0.0, 1.0]
            for i in range(len(nodes)):
                ratio = (i + 1) / len(nodes)
                x0 += [float(start_np[d] + (end_guess[d] - start_np[d]) * ratio) for d in range(STATE_DIM)]

            phase_nodes.append(nodes)
            all_controls += controls
            start_z, start_np, cum = nodes[-1], end_guess, cum + leg

        # durations appended last, with their initial guesses
        w += durations
        lbw += [meta["min_duration"]] * n_phases
        ubw += [max_duration] * n_phases
        x0 += self._duration_guesses(init_z[:2], [pf for pf, _ in phase_plan])

        # terminal realised-bank inequality on the last phase's last node
        bank_expr, bank_lb, bank_ub = _dc.terminal_bank_constraint_expr(
            phase_nodes[-1], ca.DM(np.zeros(7)), (durations[-1] / n_seg) / m_sub,
            math.radians(self.max_terminal_bank_deg),
        )
        g.append(bank_expr)
        lbg.append(bank_lb)
        ubg.append(bank_ub)

        cost = (
            ca.sum1(ca.vertcat(*durations)) / max_duration
            + _dc._DEFAULT_TIME_REGULARIZATION * _dc._scaled_control_cost(all_controls, meta)
            + _dc._control_smoothness_cost(all_controls, meta, self.smoothness_weights)
        )
        nlp = {"f": cost, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}
        layout = {"n_phases": n_phases, "c": c_np, "b": b_np}
        return nlp, lbw, ubw, lbg, ubg, x0, layout

    # ------------------------------------------------------------------
    def _faf_intercept(self):
        """The FAF-intercept for the one leg that leads INTO the protected final approach, or
        ``None`` when there is no such leg.

        The protected final approach is the first LPV segment (``lpv is not None``); its start IS
        the FAF, so the leg into it is the phase one earlier (``first_final - 1``). ``course_rad``
        is the inbound final-approach course FAF -> LTP in the model's math-ENU convention
        (0 = East, CCW toward North) — the SAME convention as the state ``psi`` the intercept is
        compared against. It is NOT the compass bearing ``atan2(de, dn)``: ne = (north, east), so
        the ENU heading of a direction (dn, de) is ``atan2(dn, de)``. The two agree ONLY at the
        45deg/225deg fixed points (why KRDU's 05/23 runways masked a 180deg error) and are ~180deg
        apart for e.g. RW32 (~315deg), which had made every constrained solve there infeasible.

        ``None`` when there is no LPV segment, or it is the very first segment (no pre-final leg).
        """
        first_final = next((i for i, s in enumerate(self.segments) if s.lpv is not None), None)
        if first_final is None or first_final == 0:
            return None
        faf_ne = np.asarray(self.segments[first_final].start_ne, float)
        ltp = np.asarray(self.segments[first_final].lpv.ltp_ne, float)
        course_rad = math.atan2(ltp[0] - faf_ne[0], ltp[1] - faf_ne[1])
        return _FafIntercept(first_final - 1, course_rad)

    # ------------------------------------------------------------------
    def _extract(self, x, layout):
        n_phases, c_np, b_np = layout["n_phases"], layout["c"], layout["b"]
        n_seg, m_sub = self.n_seg_per_phase, self.state_substeps
        controls_per_phase = n_seg * CONTROL_DIM
        states_per_phase = n_seg * m_sub * STATE_DIM
        block = controls_per_phase + states_per_phase

        durations = x[-n_phases:]
        controls, states, dense, seg_durations = [], [], [], []
        for p in range(n_phases):
            base = p * block
            ctrl = x[base: base + controls_per_phase].reshape((n_seg, CONTROL_DIM))
            nodes = x[base + controls_per_phase: base + block].reshape((n_seg * m_sub, STATE_DIM))
            for row in (nodes * c_np + b_np):                   # every collocation node (dense plan)
                dense.append(self._row_to_geo(row))
            endpoints = nodes[m_sub - 1:: m_sub]                # one per control segment
            for row in (endpoints * c_np + b_np):               # z -> physical
                states.append(self._row_to_geo(row))
            for row in ctrl:
                controls.append(list(row))
            seg_durations += [float(durations[p]) / n_seg] * n_seg

        self.segment_durations_s = seg_durations
        self.last_dense_states_geo = np.array(dense)
        return float(np.sum(durations)), np.array(controls), np.array(states)

    @staticmethod
    def _row_to_geo(phys_row):
        # phys_row = [lat_rad, lon_rad, alt, V, psi, gamma]
        return [
            math.degrees(phys_row[0]), math.degrees(phys_row[1]), phys_row[2],
            phys_row[3], phys_row[4], phys_row[5],
        ]

    def _aircraft_meta(self):
        a = self.aircraft
        return {
            "max_thrust": a.engine.max_thrust_total_n,
            "min_load_factor": 0.5,
            "max_load_factor": 2.0,
            "min_terminal_speed": (
                self.min_speed_ms if self.min_speed_ms is not None
                else a.approach.reference_speed_ms
            ),
            "min_altitude": a.approach.threshold_crossing_height_m + 10.0,
            "min_duration": 1.0,
        }

    @staticmethod
    def _cumulative_dist(start_ne, fixes):
        pts = [np.asarray(start_ne, float)] + [np.asarray(f, float) for f in fixes]
        return float(sum(np.hypot(*(pts[i] - pts[i - 1])) for i in range(1, len(pts))))

    def _duration_guesses(self, start_ne, fixes):
        guesses, prev = [], np.asarray(start_ne, float)
        for f in fixes:
            f = np.asarray(f, float)
            guesses.append(max(float(np.hypot(*(f - prev))) / 90.0, 5.0))
            prev = f
        return guesses
