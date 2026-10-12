"""A new executor for the lag plant — prototype, outside the repository's packages.

The one limit: it hears the words in force and no others (no future word, nothing else about the flight's intent). The
airframe (aero row, installed thrust, mass) and the actuator time constants are the executor's own knowledge, as the
airframe already was.

THE IDEA. A first-order actuator,  a' = (u − a)/τ,  is an integrator that the command u steers at any instant: the
executor can say "change the bank by ρ per second" by sending  u = â + τρ.  So the plant's bank is a rate-controlled
quantity, exactly as the point-mass executor's rate-limited bank was — if the executor knows the actual bank â. It
keeps an OBSERVER of the three actuators (thrust fraction, bank, load factor): the lag equation integrated on the
commands it sent. Nothing is read from the plant.

WHAT CHANGES against `autopilot.executor.Executor.cycle` (everything else — the words, the vertical law, the speed law,
the stall floor, the judge's records — is the old code):

1. BANK. The roll-rate limit p applies to the ACTUAL bank rate, not to the command. The wanted bank is the one the
   old inversion gives (turn rate → bank, within the spec's cap). It is cut to the largest bank from which the bank can
   still be taken out before the heading word's error is used up. With the bank rolled to zero at rate p the aircraft
   turns  Δψ = (g/V) ∫ tan φ dt = −g ln(cos φ) / (V p)  (exact for constant V). Solving Δψ = e gives
   φ_stop = arccos(exp(−e V p / g)); e is the error left at the end of the cycle. This replaces `stopping_rate`
   and uses the actual bank â, so a bank still growing is not cut as if it had arrived.
   The roll rate p is the peak the actuator is asked for in a cycle; the average over the cycle is
   p τ (1 − e^(−Δt/τ)) / Δt, and φ_stop uses that average.
2. LOAD FACTOR. n = B / cos φ needs the bank that is actually flown. The inversion uses the bank at mid-cycle
   predicted from â, so a rising bank raises n together with it (the old law used the commanded bank, which the lagged
   plant has not reached: the aircraft sank at roll-in). The load actuator is then steered to the value wanted at the
   end of the cycle.
3. THRUST. Steered to the wanted fraction at the end of the cycle through its own observer.

"Steered to" = the one-cycle dead-beat command  u = (a* − â e^(−Δt/τ)) / (1 − e^(−Δt/τ)),  inside the control box.

The observer starts at bank 0 (as the old executor's bank did), load factor 1 and the first wanted thrust; the plant is
started from the same belief (`make_lag_plant`).
"""
from __future__ import annotations

import math
from dataclasses import replace

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from ts_transformer.autopilot import ends, inverse
from ts_transformer.autopilot import executor as executor_module
from ts_transformer.autopilot import lateral, start as start_module
from ts_transformer.autopilot.executor import Executor
from ts_transformer.autopilot.inverse import Attitude
from ts_transformer.autopilot.lateral import relative
from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS, Plant
from ts_transformer.config import CONTROL_DYNAMICS_FIRST_ORDER_LAG
from ts_transformer.instructions.words import ALTITUDE, ANGLE, HEADING, RUNWAY
from ts_transformer.outputs.dynamics.rollout import rollout_control_endpoints
from ts_transformer.outputs.envelope import MAX_THRUST_FRACTION, MIN_THRUST_FRACTION

_ORIGINAL_WORD_RATE = lateral.word_rate
_LAST_ERROR: list[torch.Tensor] = []


def _word_rate_without_stop(error_deg, to_go_s, speed_mps, params, spec):
    """The old `word_rate` without its `stopping_rate` cap (the new bank law owns stopping); keeps the signed heading
    error for it."""
    _LAST_ERROR.clear()
    _LAST_ERROR.append(error_deg)
    rate = (error_deg / to_go_s.clamp(min=2.0 * params.cycle_s)).abs()
    return torch.sign(error_deg) * rate.clamp(max=spec.turn_rate_max_deg_s)


def make_lag_plant(true_taus):
    """The lag plant, started from the executor's belief: bank 0, load factor 1, thrust = the first command."""
    dynamics = replace(EXECUTOR_DYNAMICS, control_dynamics_model=CONTROL_DYNAMICS_FIRST_ORDER_LAG,
                       control_thrust_time_constant_s=true_taus[0], control_bank_time_constant_s=true_taus[1],
                       control_load_time_constant_s=true_taus[2])

    class LagPlant(Plant):
        def __init__(self, inputs) -> None:
            super().__init__(inputs)
            self.actual = None

        def step(self, state, command, duration_s):
            if self.actual is None:
                self.actual = command.clone()
                self.actual[:, 1] = 0.0
                self.actual[:, 2] = 1.0
            rollout = rollout_control_endpoints(
                command[:, None, :], torch.full((len(state), 1), duration_s, dtype=state.dtype, device=state.device),
                {"initial_state": state, "initial_controls": self.actual, "aero_params": self.inputs.aero_params,
                 "frame_params": self.inputs.frame_params, "max_thrust_n": self.inputs.max_thrust_n}, dynamics)
            self.actual = rollout.actual_controls[:, -1]
            return rollout.geodetic_states[:, -1]
    return LagPlant


def make_executor(assumed_taus):
    """`Executor` with the lag-aware bank / load / thrust of the module docstring; ``assumed_taus`` the actuator time
    constants it believes (thrust, bank, load)."""

    class LagExecutor(Executor):
        TAUS = tuple(float(x) for x in assumed_taus)

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            batch, dtype, device = len(self.time_limit_s), self.state.dtype, self.state.device
            #: the observer: thrust fraction, bank (the dynamics' sign), load factor
            self.est = torch.zeros((batch, 3), dtype=dtype, device=device)
            self.est[:, 2] = 1.0

        def _held(self):
            return super()._held() + [(self, "est", self.est)]

        def _attitude(self, now, track_rate_deg_s, gamma_rate, error_deg, dt):
            params, spec = self.params, self.words.spec
            tau = torch.tensor(self.TAUS, dtype=now.speed_mps.dtype, device=now.speed_mps.device)
            k = torch.exp(-dt / tau)                                            # [thrust, bank, load]
            g, speed, gamma = GRAVITY_MPS2, now.speed_mps, now.gamma_rad
            p = math.radians(params.bank_rate_deg_s)
            cap = math.radians(spec.turn_bank_max_deg)
            a = -torch.deg2rad(track_rate_deg_s) * speed * torch.cos(gamma) / g
            b = gamma_rate * speed / g + torch.cos(gamma)
            wanted = torch.atan2(a, b.clamp(min=0.0))
            capped = wanted.clamp(-cap, cap)
            phi = self.est[:, 1]                                                # the actual bank, estimated
            # the average roll rate one cycle's worth of peak-rate-p commands gives through the lag
            p_mean = p * tau[1] * (1.0 - k[1]) / dt
            e = torch.deg2rad(error_deg.abs())
            turning_toward = torch.sign(-error_deg) == torch.sign(phi)          # positive bank turns left
            psi_dot = g * torch.tan(phi.abs()) / speed
            e_left = torch.where(turning_toward, (e - psi_dot * dt).clamp(min=0.0), e + psi_dot * dt)
            phi_stop = torch.acos(torch.exp(-e_left * speed * p_mean / g).clamp(max=1.0))
            target = torch.sign(capped) * torch.minimum(capped.abs(), phi_stop)
            rho = ((target - phi) / dt).clamp(-p, p)                            # the bank rate asked for
            bank_max = inverse.BANK_MAX_RAD
            u_bank = (phi + tau[1] * rho).clamp(-bank_max, bank_max)
            phi_end = u_bank + (phi - u_bank) * k[1]
            phi_mid = 0.5 * (phi + phi_end)
            load_wanted = b / torch.cos(phi_mid)
            load = load_wanted.clamp(inverse.LOAD_FACTOR_MIN, inverse.LOAD_FACTOR_MAX)   # what the laws read
            load_at_end = (b / torch.cos(phi_end)).clamp(inverse.LOAD_FACTOR_MIN, inverse.LOAD_FACTOR_MAX)
            u_load = ((load_at_end - self.est[:, 2] * k[2]) / (1.0 - k[2])).clamp(
                inverse.LOAD_FACTOR_MIN, inverse.LOAD_FACTOR_MAX)
            binds = {"bank_cap": capped != wanted, "bank_rate": (target - phi).abs() > p * dt,
                     "load_factor": load != load_wanted}
            return Attitude(bank_rad=u_bank, load_factor=load, binds=binds), u_load, k

        def cycle(self, force, sentence_s) -> None:
            params, cycle = self.params, self.count
            dt = params.cycle_s
            now = self.now()
            self.sentence_times.append(sentence_s)
            frozen = None
            if self.staggered:
                own = self.own_cycle()
                waiting, fresh = own < 0, own == 0
                frozen = waiting | self.halted
                time_s = own.to(self.state.dtype) * dt
            else:
                waiting = fresh = None
                if bool(self.halted.any()):
                    frozen = self.halted
                time_s = cycle * dt
            first = fresh if fresh is not None else torch.full_like(self.done, cycle == 0)
            held = self._held() if frozen is not None else []
            issued = force.issued_step[:, RUNWAY]
            heard_go_around = force.go_around & (issued != self.runway_issued) & ~self.done
            if frozen is not None:
                heard_go_around = heard_go_around & ~frozen
            self.runway_issued = issued.clone()
            if bool(heard_go_around.any()):
                self.extend_time_limit(torch.where(heard_go_around, executor_module.GO_AROUND_EXTRA_S, 0.0)
                                       .to(self.time_limit_s.dtype))
            self.speed.hear_go_around(heard_go_around, now)
            track_rate = self.lateral.rate(now, force.heading_rel_deg, force.issued_step[:, HEADING], force.runway,
                                           self.runways, time_s, fresh=fresh)
            error_deg = _LAST_ERROR[0]
            e0, n0, course, elevation = self.runways.pointed(force.runway)
            before, right, _off = relative(now, e0, n0, course)
            gamma_rate, gamma_wanted, vertical_modes = self.vertical.rate(
                now, force.level_m, self.charts.elevation_m, force.no_level_off, force.angle_class, force.angle_deg,
                force.issued_step[:, [ALTITUDE, ANGLE]], force.go_around, self.inputs.aero_params,
                self.inputs.max_thrust_n)
            attitude, u_load, k = self._attitude(now, track_rate, gamma_rate, error_deg, dt)
            accel, accel_wanted, speed_modes = self.speed.rate(
                now, force.speed_mps, force.unspecified, force.go_around, attitude.load_factor,
                self.inputs.aero_params, torch.hypot(before, right))
            thrust = inverse.thrust(now, accel, attitude.load_factor, self.inputs.aero_params, self.inputs.max_thrust_n)
            est = self.est.clone()
            est[:, 0] = torch.where(first, thrust.fraction, est[:, 0])          # the first thrust: assumed in effect
            u_thrust = ((thrust.fraction - est[:, 0] * k[0]) / (1.0 - k[0])).clamp(MIN_THRUST_FRACTION,
                                                                                    MAX_THRUST_FRACTION)
            command = torch.stack((u_thrust, attitude.bank_rad, u_load), dim=1)
            self.est = est + (command - est) * (1.0 - k)[None, :]               # a ← u + (a − u) e^(−Δt/τ)
            self.bank = self.est[:, 1]
            self.state = self.plant.step(self.state, command, dt)
            limits = {**attitude.binds, **thrust.binds, **speed_modes,
                      "path_rate_limited": vertical_modes["path_rate_limited"]}
            modes = {"go_around": force.go_around.clone(), "level_captured": vertical_modes["level_captured"]}
            if frozen is not None and bool(frozen.any()):
                for owner, name, value in held:
                    after = getattr(owner, name)
                    if value is not None:
                        keep = frozen.view(-1, *([1] * (after.dim() - 1)))
                        setattr(owner, name, torch.where(keep, value, after))
                command = torch.where(self.halted[:, None], self.last_command, command)
                limits = {name: value & ~frozen for name, value in limits.items()}
                modes = {name: value & ~frozen for name, value in modes.items()}
            self.last_command = command
            self.states.append(self.state)
            self.commands.append(command)
            self.wanted.append(torch.stack((track_rate, gamma_wanted, accel_wanted), dim=1))
            self.runway_rows.append(force.runway.clone())
            for name, value in limits.items():
                self.limits[name].append(value)
            for name, value in modes.items():
                self.modes[name].append(value)
            # the end of the flight, as the old cycle reads it
            after = executor_module.read_state(self.state, self.charts)
            runways = self.runways
            all_before, all_right, _ = runways.relative(now)
            all_past, all_right_after, all_off_after = runways.relative(after)
            fraction = (all_before / (all_before - all_past)).clamp(0.0, 1.0)
            in_force = torch.arange(all_before.shape[1], device=all_before.device)[None, :] == force.runway[:, None]
            approach, other = ends.crossing_ends(all_right + fraction * (all_right_after - all_right), all_off_after,
                                                 force.go_around[:, None], in_force, runways.landing_limit_m,
                                                 runways.on_runway_m, self.words.spec)
            crossed = (ends.plane_crossed(all_before, all_past) & (approach | other)).any(dim=1)
            past = relative(after, e0, n0, course)[0]
            ended = (crossed | ends.ground_contact(past, after.height_m - elevation)
                     | ends.dynamics_failure(torch.isfinite(self.state).all(dim=1), after.speed_mps, limits["stall"]))
            if self.staggered:
                finished = (ended | ((own + 1).to(self.state.dtype) * dt >= self.time_limit_s)) & ~frozen
                self.done_cycle = torch.where(finished & ~self.done, own, self.done_cycle)
            else:
                finished = ended | ((cycle + 1) * dt >= self.time_limit_s)
                if frozen is not None:
                    finished = finished & ~frozen
                self.done_cycle = torch.where(finished & ~self.done, cycle, self.done_cycle)
            self.done = self.done | finished
            self.count += 1

    return LagExecutor


def install(assumed_taus, true_taus):
    """Patch this process: the new executor, the plant it flies, and the lateral law without its stopping cap."""
    executor_module.Executor = make_executor(assumed_taus)
    start_module.Executor = executor_module.Executor
    executor_module.Plant = make_lag_plant(true_taus)
    lateral.word_rate = _word_rate_without_stop


def uninstall(original_executor, original_plant) -> None:
    executor_module.Executor = original_executor
    start_module.Executor = original_executor
    executor_module.Plant = original_plant
    lateral.word_rate = _ORIGINAL_WORD_RATE
