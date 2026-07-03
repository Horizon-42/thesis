"""The unified direct-collocation optimiser.

ONE optimiser, ``CollocationOptimizer``, models the trajectory as a sequence of PHASES — each with
its own control segments, dense state nodes, free duration, and (optionally) path constraints.
Constraints are OPTIONAL:

  * ``segments=None``  -> one free phase ``initial -> target`` (no path constraints). The
                         unconstrained optimiser used by the ADS-B / runway-target modes.
  * ``segments=[...]`` -> one phase per procedure leg, each with its corridor / glidepath /
                         step-down floor + a FAF-intercept, the target pinned at the end. When the
                         start is away from the first leg's start fix, an UNCONSTRAINED
                         start -> first-fix transition phase is prepended, so leg constraints only
                         apply on the procedure itself. The procedure-constrained optimiser.

Both build the NLP fresh per solve (initial/target are concrete, not symbolic parameters) — the
procedure geometry differs per scenario, so there is nothing to amortise across solves. Free time
minimises ``Σ Tₚ``; fixed time adds ``Σ Tₚ = duration`` and drops the time term. The dense-state
transcription (control over N segments, state on N·M sub-intervals, M auto-selected per phase
from the phase's duration guess) makes the raw solution playback-consistent without a post-solve
polish. Mass is frozen at the initial mass (not a decision state; the RHS's fuel-burn rate is
discarded) — a deliberate approximation, small over an approach.

Reuses the whole foundation: :mod:`.schemes` (dynamics × fitting + metric-position normalization),
:mod:`.components` (bounds, altitude floor, control costs, terminal bank, solver factory,
state ↔ decision conversions), and :mod:`approach_constraints` (the ONE source of the
corridor / glidepath / floor / course math — nothing is re-derived here).
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

import approach_constraints as ac
from approach_constraints import geometry as _acgeo

from . import schemes as _schemes
from . import components as _components

STATE_DIM = _schemes.STATE_DIM
CONTROL_DIM = _schemes.CONTROL_DIM
_PSI = _schemes._PSI
_INF = 1e9

# Constrained solves: the segments' (n, e) coordinates MUST be in the frame the optimizer solves
# in — metric offsets anchored at the TARGET = the LTP. Validated at construction (a silent
# mis-anchor would shift every corridor); the tolerance absorbs threshold-vs-target data noise.
_FRAME_ANCHOR_TOLERANCE_M = 150.0
# A start farther than this from the first leg's start fix gets a dedicated unconstrained
# transition phase (closer counts as "joining the procedure at the first fix").
_TRANSITION_JOIN_TOLERANCE_M = 2000.0

# The FAF-intercept spec for the one leg that leads INTO a protected final approach:
# which phase it is, and the final-approach course to align with there.
_FafIntercept = namedtuple("_FafIntercept", ("phase", "course_rad"))


class CollocationOptimizer:
    """Direct-collocation optimiser; procedure constraints are optional (``segments``)."""

    # Mode default scheme: the constrained solve needs the well-conditioned metric-position state
    # (its path constraints live there); the unconstrained solve has no such requirement.
    _DEFAULT_CONSTRAINED_SCHEME = "hermiteSimpsonNormalizedFullTransport"

    def __init__(
        self,
        aircraft: Aircraft,
        *,
        scheme: str | None = None,
        segments: list | None = None,
        n_segments: int = 8,
        n_seg_per_phase: int = 4,
        state_substeps: int | None = None,
        max_duration: float = 600.0,
        max_terminal_bank_deg: float = _components._DEFAULT_MAX_TERMINAL_BANK_DEG,
        smoothness_weights: tuple = _components._DEFAULT_SMOOTHNESS_WEIGHTS,
        min_speed_ms: float | None = None,
        max_intercept_deg: float = ac.STANDARD_INTERCEPT_MAX_DEG,
        solver_backend: str = _components._DEFAULT_SOLVER_BACKEND,
        verbose: bool = False,
    ):
        if scheme is None:
            scheme = self._DEFAULT_CONSTRAINED_SCHEME if segments else _schemes._DEFAULT_SCHEME
        if scheme not in _schemes._DEFECT_SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}; choose from {sorted(_schemes._DEFECT_SCHEMES)}")
        if solver_backend not in _components._SOLVER_BACKENDS:
            raise ValueError(f"unknown solver_backend {solver_backend!r}; choose from {list(_components._SOLVER_BACKENDS)}")
        if min_speed_ms is not None and min_speed_ms <= 0:
            raise ValueError("min_speed_ms must be positive when given")
        if n_segments < 2 or n_seg_per_phase < 1 or (state_substeps is not None and state_substeps < 1):
            raise ValueError("n_segments must be >= 2, n_seg_per_phase and state_substeps >= 1")
        if segments is not None and not segments:
            raise ValueError("segments must be None (unconstrained) or a non-empty list of legs")
        if segments:
            # Path constraints live on the metric-position (n, e) decision state, so they REQUIRE a
            # normalized full-transport scheme; -inf-lower-bound rows also need the ipopt backend.
            if scheme not in _schemes._NORMALIZED_FULL_TRANSPORT_SCHEMES:
                raise ValueError(
                    "procedure constraints require a normalized full-transport scheme "
                    f"{sorted(_schemes._NORMALIZED_FULL_TRANSPORT_SCHEMES)}; got {scheme!r}"
                )
            if solver_backend != "ipopt":
                raise ValueError("procedure constraints require the 'ipopt' solver_backend")
            # Frame-coincidence contract: the terminal node is pinned at the (n, e) origin, so the
            # procedure must END there and every LPV spec must anchor its LTP there.
            end_gap = float(np.hypot(*np.asarray(segments[-1].end_ne, float)))
            if end_gap > _FRAME_ANCHOR_TOLERANCE_M:
                raise ValueError(
                    f"the final segment must end at the LTP/target — the (n, e) origin — but ends "
                    f"{end_gap:.0f} m away; the segments are not in the target-anchored frame"
                )
            for s in segments:
                if s.lpv is not None and float(np.hypot(*np.asarray(s.lpv.ltp_ne, float))) > _FRAME_ANCHOR_TOLERANCE_M:
                    raise ValueError(
                        f"LPV segment [{s.start_ident}->{s.end_ident}] has ltp_ne away from the "
                        "(n, e) origin; the LTP must be the frame anchor (the optimizer target)"
                    )

        self.aircraft = aircraft
        self.scheme = scheme
        self.segments = segments
        # The single MODE flag every branch keys on: procedure-constrained (one phase per leg) vs
        # free (one phase, initial→target). Derived once so the mode is defined in exactly one place.
        self.constrained = segments is not None
        self.n_seg_per_phase = n_seg_per_phase
        # Constrained: the per-leg count, EXCLUDING the optional start->first-fix transition phase
        # (whether one exists depends on the initial state, known only at solve time).
        self.n_segments = len(segments) * n_seg_per_phase if self.constrained else n_segments
        # State substeps M: None -> auto-selected PER PHASE from that phase's duration guess
        # (~3 s state step, components.select_state_substeps); an explicit value applies to all.
        self.state_substeps = state_substeps
        self.max_duration = max_duration
        self.max_terminal_bank_deg = max_terminal_bank_deg
        self.smoothness_weights = smoothness_weights
        self.min_speed_ms = min_speed_ms
        self.max_intercept_deg = max_intercept_deg
        self.solver_backend = solver_backend
        self.verbose = verbose
        self.aero_params = aero_params_for_aircraft(aircraft)
        # Outputs of the most recent solve.
        self.segment_durations_s: list[float] | None = None
        self.last_dense_states_geo: np.ndarray | None = None
        self.last_solve_timings: dict[str, float] | None = None

    # ------------------------------------------------------------------ public
    def optimize_free_time(self, initial_state, target_state, max_duration, initial_guess=None,
                           cold_start: bool | None = None):
        """Minimum-time solve; returns ``(final_time, controls (Nc,3), states (Nc,6))`` (geodetic
        degrees at the control-segment endpoints). ``self.segment_durations_s`` holds the matching
        per-segment durations.

        Cold start: an ill-conditioned (non-normalized) scheme seeds the free-time NLP with a
        fixed-time solution at ``max_duration`` (a dynamically-feasible trajectory to shrink along),
        far more robust than a cold linear guess. A normalized scheme (every constrained solve, and
        the normalized unconstrained schemes) is well-conditioned and solves directly — seeding it
        would just double the codegen for no gain. So the default keys on the SCHEME's conditioning,
        not on the mode."""
        started = time.perf_counter()
        if cold_start is None:
            cold_start = self.scheme not in _schemes._NORMALIZED_SCHEMES
        cold_s = 0.0
        seed_error = None
        nlp, lbw, ubw, lbg, ubg, x0, layout = self._build(initial_state, target_state, max_duration)
        if initial_guess is not None:
            x0 = list(initial_guess)
        elif cold_start:
            cs_started = time.perf_counter()
            try:
                x0 = list(self._solve_fixed_raw(initial_state, target_state, max_duration))
            except ValueError as exc:
                # Keep the build's linear guess, but remember why the seed failed so the root cause
                # surfaces if the free-time solve below also fails (rather than vanishing).
                seed_error = str(exc)
            cold_s = time.perf_counter() - cs_started
        solver = _components._make_nlp_solver(nlp, self.solver_backend, self.verbose)
        ft_started = time.perf_counter()
        sol = solver(x0=x0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        if not solver.stats()["success"]:
            status = solver.stats().get("return_status", "unknown")
            detail = f" (cold-start seed also failed: {seed_error})" if seed_error else ""
            raise ValueError(f"collocation free-time optimization failed: {status}{detail}")
        ft_s = time.perf_counter() - ft_started
        self.last_solve_timings = {
            "coldStartS": cold_s, "freeTimeSolveS": ft_s, "solveTotalS": time.perf_counter() - started,
        }
        return self._extract(np.array(sol["x"]).reshape(-1), layout)

    def optimize_trajectory(self, initial_state, target_state, duration=None, initial_guess=None):
        """Fixed-time solve at ``duration`` (``Σ Tₚ = duration``); same return shape as
        :meth:`optimize_free_time`. Defaults to the construction ``max_duration``."""
        fixed = self.max_duration if duration is None else duration
        started = time.perf_counter()
        nlp, lbw, ubw, lbg, ubg, x0, layout = self._build(
            initial_state, target_state, fixed, fixed_duration=fixed)
        if initial_guess is not None:
            x0 = list(initial_guess)
        solver = _components._make_nlp_solver(nlp, self.solver_backend, self.verbose)
        sol = solver(x0=x0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        if not solver.stats()["success"]:
            raise ValueError(f"collocation optimization failed: {solver.stats().get('return_status', 'unknown')}")
        self.last_solve_timings = {"solveTotalS": time.perf_counter() - started}
        return self._extract(np.array(sol["x"]).reshape(-1), layout)

    # ------------------------------------------------------------------ build
    def _solve_fixed_raw(self, initial_state, target_state, duration):
        """Solve the fixed-time NLP and return the raw decision vector (a free-time seed — the two
        share a decision layout, the per-phase durations already sum to ``duration``)."""
        nlp, lbw, ubw, lbg, ubg, x0, _ = self._build(
            initial_state, target_state, duration, fixed_duration=duration)
        solver = _components._make_nlp_solver(nlp, self.solver_backend, self.verbose)
        sol = solver(x0=x0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        if not solver.stats()["success"]:
            raise ValueError(f"fixed-time seed failed: {solver.stats().get('return_status', 'unknown')}")
        return np.array(sol["x"]).reshape(-1)

    def _build(self, initial_state, target_state, max_duration, fixed_duration=None):
        make_dynamics, make_defect = _schemes._DEFECT_SCHEMES[self.scheme]
        dynamics = make_dynamics()
        ip = _components._geodetic_state_to_decision(initial_state)
        tp = _components._unwrap_target_heading(ip, _components._geodetic_state_to_decision(target_state))
        target_dm = ca.DM(tp)
        c, b = _schemes._scheme_normalization(self.scheme, target_dm)
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
        meta = self._aircraft_meta(target_state.altitude)
        state_lb, state_ub = _schemes._normalized_position_bounds(
            *_components.make_state_bounds(meta["min_altitude"], meta["min_terminal_speed"]), self.scheme
        )
        control_lb, control_ub = _components.make_control_bounds(
            meta["max_thrust"], meta["min_load_factor"], meta["max_load_factor"]
        )

        phase_plan = self._phase_plan(init_z[:2], tgt_z)   # [(end_fix, seg_or_None, n_seg)]
        faf = self._faf_intercept(phase_plan)
        intercept_floor = math.cos(math.radians(self.max_intercept_deg))
        n_phases = len(phase_plan)
        fixes = [pf for pf, _s, _n in phase_plan]
        total_dist = self._cumulative_dist(init_z[:2], fixes)
        # Per-phase duration guesses. ``leg_guesses`` (fixed_duration-independent) drive the
        # state-substep selection so the fixed- and free-time NLPs share ONE decision layout (the
        # fixed solve seeds the free one); the x0 guess is rescaled to a fixed total when given.
        leg_guesses = self._leg_duration_guesses(init_z[:2], fixes, max_duration)
        if fixed_duration is not None:
            scale = fixed_duration / sum(leg_guesses)
            dur_guesses = [lg * scale for lg in leg_guesses]
        else:
            dur_guesses = leg_guesses

        w, lbw, ubw, x0 = [], [], [], []
        g, lbg, ubg = [], [], []
        durations, phase_nodes, phase_starts, all_controls = [], [], [], []
        phase_nseg, phase_msub = [], []
        start_z, start_np, cum = ca.DM(init_z), init_z.copy(), 0.0
        for p, (end_fix, seg, n_seg) in enumerate(phase_plan):
            last = p == n_phases - 1
            phase_starts.append(start_z)   # the node one step before this phase's first node
            Tp = ca.SX.sym(f"T_{p}")
            durations.append(Tp)
            # State substeps per phase, from the phase's duration guess (~3 s state step), so a
            # long leg keeps its playback-consistent state density instead of a one-size M.
            m_sub = self.state_substeps or _components.select_state_substeps(leg_guesses[p], n_seg)
            phase_msub.append(m_sub)
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
                for _j in range(m_sub):
                    xn = ca.SX.sym(f"x_{p}_{k}_{_j}", STATE_DIM)
                    nodes.append(xn)
                    g.append(defect(ca.vertcat(x_prev, mass), ca.vertcat(xn, mass), uk, aero, state_h)[:STATE_DIM])
                    lbg += [0.0] * STATE_DIM
                    ubg += [0.0] * STATE_DIM
                    x_prev = xn

            # Boundary pin: full target (last phase) | FAF position + intercept angle (the leg into
            # the FAF) | nothing (pre-final legs — horizontal free, only altitude is constrained).
            if last:
                g.append(nodes[-1] - ca.DM(tgt_z))
                lbg += [0.0] * STATE_DIM
                ubg += [0.0] * STATE_DIM
            elif faf is not None and p == faf.phase:
                g.append(nodes[-1][0:2] - ca.DM(end_fix))
                lbg += [0.0, 0.0]
                ubg += [0.0, 0.0]
                g.append(intercept_floor - ca.cos(nodes[-1][_PSI] - faf.course_rad))
                lbg.append(-_INF)
                ubg.append(0.0)

            # Leg path constraints (protected LPV final keeps the lateral corridor; pre-final legs
            # contribute altitude floor + descent cap only). Unconstrained phases (seg None — the
            # single free phase, or the start->first-fix transition) skip it.
            if seg is not None:
                n_vec = ca.vertcat(*[nd[0] for nd in nodes])
                e_vec = ca.vertcat(*[nd[1] for nd in nodes])
                h_vec = ca.vertcat(*[nd[2] for nd in nodes])
                gamma_vec = ca.vertcat(*[nd[5] for nd in nodes])
                viol = ac.segment_violations_from_components(
                    seg, n_vec, e_vec, h_vec, gamma_vec, include_lateral=(seg.lpv is not None)
                )
                for expr in viol.values():
                    g.append(expr)
                    rows = int(expr.shape[0])
                    lbg += [-_INF] * rows
                    ubg += [0.0] * rows

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
            phase_nseg.append(n_seg)
            start_z, start_np, cum = nodes[-1], end_guess, cum + leg

        # Per-phase durations (decision vars), their bounds and guess. The lower bound is clamped
        # below the upper so a very short horizon (max_duration < min_duration) still gives a
        # consistent [lb, ub] box instead of an inverted, infeasible one.
        upper = fixed_duration if fixed_duration is not None else max_duration
        w += durations
        lbw += [min(meta["min_duration"], upper * 0.5)] * n_phases
        ubw += [upper] * n_phases
        x0 += dur_guesses

        # Fixed time: constrain the total; else the objective minimises it.
        if fixed_duration is not None:
            g.append(ca.sum1(ca.vertcat(*durations)) - fixed_duration)
            lbg.append(0.0)
            ubg.append(0.0)

        # Terminal realised-bank inequality on the last phase's last node. The last phase's START
        # node is the ``prev`` sample when that phase has a single state node — it must be the
        # real preceding state, never a placeholder (a zeros vector would constrain V·cosγ·ψ,
        # i.e. the absolute heading).
        bank_expr, bank_lb, bank_ub = _components.terminal_bank_constraint_expr(
            phase_nodes[-1], phase_starts[-1], (durations[-1] / phase_nseg[-1]) / phase_msub[-1],
            math.radians(self.max_terminal_bank_deg),
        )
        g.append(bank_expr)
        lbg.append(bank_lb)
        ubg.append(bank_ub)

        # Objective. FREE time: minimise Σ Tp, with control effort a light 1e-3 tie-breaker (the
        # time term dominates). FIXED time: no time term, so control effort is the PRIMARY objective
        # at weight 1.0 (as the old fixed-time NLP did) — the 1e-3 free-time discount would leave the
        # controls under-determined, governed only by segment-to-segment smoothness.
        effort = _components._scaled_control_cost(all_controls, meta)
        smoothness = _components._control_smoothness_cost(all_controls, meta, self.smoothness_weights)
        if fixed_duration is not None:
            cost = effort + smoothness
        else:
            cost = ca.sum1(ca.vertcat(*durations)) / max_duration \
                + _components._DEFAULT_TIME_REGULARIZATION * effort + smoothness
        nlp = {"f": cost, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}
        layout = {"phase_nseg": phase_nseg, "phase_msub": phase_msub, "c": c_np, "b": b_np}
        return nlp, lbw, ubw, lbg, ubg, x0, layout

    # ------------------------------------------------------------------ helpers
    def _phase_plan(self, init_ne, tgt_z):
        """``[(end_fix, seg_or_None, n_seg)]``: one phase per constrained leg — prefixed by an
        UNCONSTRAINED start -> first-fix transition phase when the start is farther than
        ``_TRANSITION_JOIN_TOLERANCE_M`` from the first leg's start fix — or a single free phase
        to the target when unconstrained. The transition end fix is guess geometry only (not
        pinned), consistent with the pre-final path being horizontally free."""
        if not self.constrained:
            return [(np.asarray(tgt_z[:2], float), None, self.n_segments)]
        plan = [(np.asarray(s.end_ne, float), s, self.n_seg_per_phase) for s in self.segments]
        first_fix = np.asarray(self.segments[0].start_ne, float)
        gap = float(np.hypot(*(np.asarray(init_ne, float) - first_fix)))
        if gap > _TRANSITION_JOIN_TOLERANCE_M:
            plan.insert(0, (first_fix, None, self.n_seg_per_phase))
        return plan

    def _faf_intercept(self, phase_plan):
        """The FAF-intercept for the phase leading INTO the protected final approach, or ``None``.

        The final approach is the first LPV phase; its segment's start IS the FAF, so the
        intercept applies to the phase one earlier (which may be the start -> first-fix
        transition phase). The inbound course FAF -> LTP comes from
        ``approach_constraints.geometry.course_bearing`` — the single source of course math, in
        the model's heading convention (0 = East, CCW toward North), the SAME convention as the
        state ``psi`` the intercept is compared against."""
        final_idx = next(
            (i for i, (_f, seg, _n) in enumerate(phase_plan)
             if seg is not None and seg.lpv is not None),
            None,
        )
        if final_idx is None or final_idx == 0:
            return None
        final_seg = phase_plan[final_idx][1]
        course_rad = _acgeo.course_bearing(
            np.asarray(final_seg.start_ne, float), np.asarray(final_seg.lpv.ltp_ne, float)
        )
        return _FafIntercept(final_idx - 1, float(course_rad))

    def _extract(self, x, layout):
        phase_nseg, phase_msub = layout["phase_nseg"], layout["phase_msub"]
        c_np, b_np = layout["c"], layout["b"]
        n_phases = len(phase_nseg)
        durations = x[-n_phases:]
        controls, states, dense, seg_durations = [], [], [], []
        base = 0
        for p, (n_seg, m_sub) in enumerate(zip(phase_nseg, phase_msub)):
            cpp, spp = n_seg * CONTROL_DIM, n_seg * m_sub * STATE_DIM
            ctrl = x[base: base + cpp].reshape((n_seg, CONTROL_DIM))
            nodes = x[base + cpp: base + cpp + spp].reshape((n_seg * m_sub, STATE_DIM))
            for row in (nodes * c_np + b_np):                 # every collocation node (dense plan)
                dense.append(self._row_to_geo(row))
            for row in (nodes[m_sub - 1:: m_sub] * c_np + b_np):   # one endpoint per control segment
                states.append(self._row_to_geo(row))
            for row in ctrl:
                controls.append(list(row))
            seg_durations += [float(durations[p]) / n_seg] * n_seg
            base += cpp + spp
        self.segment_durations_s = seg_durations
        self.last_dense_states_geo = np.array(dense)
        return float(np.sum(durations)), np.array(controls), np.array(states)

    @staticmethod
    def _row_to_geo(phys_row):
        return [math.degrees(phys_row[0]), math.degrees(phys_row[1]),
                phys_row[2], phys_row[3], phys_row[4], phys_row[5]]

    def _aircraft_meta(self, target_altitude_m):
        a = self.aircraft
        return {
            "max_thrust": a.engine.max_thrust_total_n,
            "min_load_factor": 0.5,
            "max_load_factor": 2.0,
            "min_terminal_speed": (
                self.min_speed_ms if self.min_speed_ms is not None else a.approach.reference_speed_ms
            ),
            # Altitude floor: a generous margin below the destination threshold (the lowest point),
            # anchored to the target so it is correct at every field elevation.
            "min_altitude": _components.altitude_floor_m(target_altitude_m),
            "min_duration": _components._DEFAULT_MIN_DURATION_S,
        }

    @staticmethod
    def _cumulative_dist(start_ne, fixes):
        pts = [np.asarray(start_ne, float)] + [np.asarray(f, float) for f in fixes]
        return float(sum(np.hypot(*(pts[i] - pts[i - 1])) for i in range(1, len(pts))))

    def _leg_duration_guesses(self, start_ne, fixes, max_duration):
        """Per-phase duration guess: metric legs (constrained, transition phase included) at
        distance / 90 m·s⁻¹ (min 5 s); the unconstrained single phase seeds at the horizon (a
        fixed-time seed / free-time shrink refines it). Also the basis of the per-phase
        state-substep selection, so it deliberately does NOT depend on ``fixed_duration``: the
        fixed- and free-time NLPs must share one decision layout."""
        if not self.constrained:
            return [max_duration] * len(fixes)     # a single free phase
        guesses, prev = [], np.asarray(start_ne, float)
        for f in fixes:
            f = np.asarray(f, float)
            guesses.append(max(float(np.hypot(*(f - prev))) / 90.0, 5.0))
            prev = f
        return guesses
