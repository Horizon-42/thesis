"""
Simple 3D visualizer for the point-mass flight simulator.

Run from the repository root:

    python aerodynamic_model/visualize_simulator.py

The aircraft is drawn as one moving mass point. The right side of the
window has an input zone for simulator parameters and a state output zone
that updates while the trajectory is animated.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

try:
    # Works when this file is executed as a package module.
    from .simulator import Atmosphere, Control, Simulator, State
except ImportError:
    # Works when this file is executed directly:
    # python aerodynamic_model/visualize_simulator.py
    from simulator import Atmosphere, Control, Simulator, State


class FlightVisualizer:
    def __init__(self):
        # Reuse the existing simulator model. This script only handles plotting
        # and simple user input.
        self.simulator = Simulator()
        self.atmosphere = Atmosphere(rho0=1.225, H=8500.0)

        # Initial point-mass aircraft state:
        # x, y, h are position; V is speed; psi is heading; gamma is climb angle.
        self.initial_state = State(
            x=-2000.0,
            y=0.0,
            h=1000.0,
            V=120.0,
            psi=0.0,
            gamma=0.0,
            m=10000.0,
        )

        self.solution = None
        self.animation = None

        # The figure is split manually: 3D plot on the left, controls and state
        # output on the right.
        self.fig = plt.figure(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title("3DOF Point-Mass Flight Simulator")
        self.ax = self.fig.add_axes([0.05, 0.08, 0.64, 0.84], projection="3d")

        # Blue line: flown trajectory. Red dot: current aircraft mass point.
        self.path_line, = self.ax.plot([], [], [], color="tab:blue", lw=2)
        self.aircraft_point, = self.ax.plot(
            [], [], [], marker="o", markersize=9, color="tab:red", linestyle="None"
        )

        self._build_input_zone()
        self._build_state_output_zone()
        self._setup_plot()
        self._show_initial_state()

    def _build_input_zone(self):
        # Sliders are the input zone. Their values are read when Run is clicked.
        self.fig.text(0.74, 0.90, "Input zone", fontsize=13, weight="bold")

        self.thrust_slider = self._add_slider(
            y=0.82,
            label="Thrust N",
            min_value=0.0,
            max_value=40000.0,
            initial=3000.0,
        )
        self.bank_slider = self._add_slider(
            y=0.74,
            label="Bank deg",
            min_value=-60.0,
            max_value=60.0,
            initial=0.0,
        )
        self.load_slider = self._add_slider(
            y=0.66,
            label="Load n",
            min_value=0.3,
            max_value=3.0,
            initial=1.1,
        )
        self.duration_slider = self._add_slider(
            y=0.58,
            label="Time s",
            min_value=10.0,
            max_value=300.0,
            initial=120.0,
        )

        run_ax = self.fig.add_axes([0.74, 0.48, 0.10, 0.05])
        reset_ax = self.fig.add_axes([0.86, 0.48, 0.10, 0.05])
        self.run_button = Button(run_ax, "Run")
        self.reset_button = Button(reset_ax, "Reset")
        self.run_button.on_clicked(self.run)
        self.reset_button.on_clicked(self.reset)

    def _add_slider(self, y, label, min_value, max_value, initial):
        # add_axes uses normalized figure coordinates: [left, bottom, width, height].
        slider_ax = self.fig.add_axes([0.74, y, 0.22, 0.035])
        return Slider(slider_ax, label, min_value, max_value, valinit=initial)

    def _build_state_output_zone(self):
        # The state output zone is a text box updated once per animation frame.
        self.fig.text(0.74, 0.40, "State output zone", fontsize=13, weight="bold")
        self.state_text = self.fig.text(
            0.74,
            0.14,
            "",
            family="monospace",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#f5f5f5",
                "edgecolor": "#cccccc",
            },
        )

    def _setup_plot(self):
        self.ax.set_title("3D trajectory")
        self.ax.set_xlabel("x position (m)")
        self.ax.set_ylabel("y position (m)")
        self.ax.set_zlabel("altitude h (m)")
        self.ax.grid(True)

    def _read_control(self):
        # The simulator expects bank angle in radians, while the UI shows degrees.
        return Control(
            thrust=float(self.thrust_slider.val),
            bank=math.radians(float(self.bank_slider.val)),
            load_factor=float(self.load_slider.val),
        )

    def run(self, _event=None):
        duration = float(self.duration_slider.val)

        # Five output samples per second gives a smooth animation while keeping
        # the ODE result small and easy to inspect.
        t_eval = np.linspace(0.0, duration, int(duration * 5) + 1)
        control = self._read_control()

        self.solution = self.simulator.simulate(
            initial_state=self.initial_state,
            control=control,
            atmosphere=self.atmosphere,
            t_span=(0.0, duration),
            t_eval=t_eval,
        )

        if not self.solution.success:
            self.state_text.set_text(f"Simulation failed:\n{self.solution.message}")
            self.fig.canvas.draw_idle()
            return

        self._trim_solution_at_ground()
        self._show_initial_state()

        # Stop a previous animation before starting a new one with new inputs.
        self._stop_animation()

        self.animation = FuncAnimation(
            self.fig,
            self._draw_frame,
            frames=len(self.solution.t),
            interval=30,
            repeat=False,
        )
        self.fig.canvas.draw_idle()

    def _trim_solution_at_ground(self):
        # Stop the displayed simulation when altitude first reaches the ground.
        # The simulator returns samples at t_eval, so interpolate one final
        # point at h = 0 instead of displaying a below-ground sample.
        ground_hits = np.where(self.solution.y[2] <= 0.0)[0]
        if len(ground_hits) == 0:
            return

        hit_index = int(ground_hits[0])
        if hit_index == 0:
            self.solution.t = self.solution.t[:1]
            self.solution.y = self.solution.y[:, :1].copy()
            self.solution.y[2, 0] = 0.0
            return

        previous_index = hit_index - 1
        previous_h = self.solution.y[2, previous_index]
        hit_h = self.solution.y[2, hit_index]
        ground_fraction = previous_h / (previous_h - hit_h)

        ground_t = self.solution.t[previous_index] + ground_fraction * (
            self.solution.t[hit_index] - self.solution.t[previous_index]
        )
        ground_state = self.solution.y[:, previous_index] + ground_fraction * (
            self.solution.y[:, hit_index] - self.solution.y[:, previous_index]
        )
        ground_state[2] = 0.0

        self.solution.t = np.append(self.solution.t[: hit_index], ground_t)
        self.solution.y = np.column_stack(
            [self.solution.y[:, : hit_index], ground_state]
        )

    def reset(self, _event=None):
        # Return the view to the initial state but keep slider values unchanged.
        self._stop_animation()
        self.solution = None
        self._show_initial_state()
        self.fig.canvas.draw_idle()

    def _stop_animation(self):
        # Matplotlib can clear event_source after an animation finishes. Check it
        # before calling stop() so Reset is safe after completion.
        if self.animation is None:
            return
        if self.animation.event_source is not None:
            self.animation.event_source.stop()
        self.animation = None

    def _draw_frame(self, frame_index):
        # solution.y rows are ordered like the simulator state vector:
        # x, y, h, V, psi, gamma, m.
        x, y, h, V, psi, gamma, m = self.solution.y[:, frame_index]

        # Draw the trajectory up to the current frame.
        self.path_line.set_data(
            self.solution.y[0, : frame_index + 1],
            self.solution.y[1, : frame_index + 1],
        )
        self.path_line.set_3d_properties(self.solution.y[2, : frame_index + 1])

        # Move the aircraft point to the current state.
        self.aircraft_point.set_data([x], [y])
        self.aircraft_point.set_3d_properties([h])

        self._expand_axes_if_needed(x, y, h)
        self._update_state_text(self.solution.t[frame_index], x, y, h, V, psi, gamma, m)
        return self.path_line, self.aircraft_point

    def _show_initial_state(self):
        # Display one point before any simulation is run.
        s = self.initial_state
        self.path_line.set_data([s.x], [s.y])
        self.path_line.set_3d_properties([s.h])
        self.aircraft_point.set_data([s.x], [s.y])
        self.aircraft_point.set_3d_properties([s.h])
        self.ax.set_xlim(-5000.0, 5000.0)
        self.ax.set_ylim(-5000.0, 5000.0)
        self.ax.set_zlim(0.0, 4000.0)
        self._update_state_text(0.0, s.x, s.y, s.h, s.V, s.psi, s.gamma, s.m)

    def _expand_axes_if_needed(self, x, y, h):
        # Grow the coordinate system while the aircraft flies. This keeps the
        # aircraft visible without knowing the full path scale in advance.
        self._expand_axis_if_needed(self.ax.get_xlim, self.ax.set_xlim, x)
        self._expand_axis_if_needed(self.ax.get_ylim, self.ax.set_ylim, y)
        self._expand_axis_if_needed(self.ax.get_zlim, self.ax.set_zlim, h, floor=0.0)

    @staticmethod
    def _expand_axis_if_needed(getter, setter, value, floor=None):
        low, high = getter()
        axis_size = high - low
        edge_zone = axis_size * 0.15

        if low + edge_zone < value < high - edge_zone:
            return

        new_low = low
        new_high = high

        if value < low + edge_zone:
            # If a floor exists and the lower bound is already at that floor,
            # do not expand the opposite side. This avoids runaway z-axis
            # growth when altitude is close to ground level.
            if floor is None or low > floor:
                new_low = min(low - axis_size * 0.5, value - axis_size * 0.2)
        if value > high - edge_zone:
            new_high = max(high + axis_size * 0.5, value + axis_size * 0.2)
        if floor is not None:
            new_low = max(floor, new_low)

        if new_low != low or new_high != high:
            setter(new_low, new_high)

    def _update_state_text(self, t, x, y, h, V, psi, gamma, m):
        # Convert angles back to degrees for easier reading in the UI.
        self.state_text.set_text(
            "\n".join(
                [
                    f"t     : {t:8.1f} s",
                    f"x     : {x:8.1f} m",
                    f"y     : {y:8.1f} m",
                    f"h     : {h:8.1f} m",
                    f"V     : {V:8.1f} m/s",
                    f"psi   : {math.degrees(psi):8.2f} deg",
                    f"gamma : {math.degrees(gamma):8.2f} deg",
                    f"mass  : {m:8.1f} kg",
                ]
            )
        )


def main():
    visualizer = FlightVisualizer()
    visualizer.run()
    plt.show()


if __name__ == "__main__":
    main()
