###########################################################################
# Mode-2 Perturbation Experiment — Phase 1, Day 5
#
# Starts a spinning chain with a small elliptical distortion (mode-2)
# and watches how the shape evolves over time.
#
# PHYSICS BACKGROUND:
# A mode-n perturbation of a circle has n wavelengths around the
# circumference. Mode-2 (n=2) is an ellipse: squeezed in one direction,
# stretched in the other. If the spinning ring is stable, this ellipse
# should oscillate — the long axis bounces back and forth as the
# centripetal tension provides a restoring force.
#
# The oscillation frequency depends on:
#   - The rotation rate omega
#   - The mode number n
#   - The tension T = mu * omega^2 * r^2
#   - The bending stiffness (for higher modes)
#
# This experiment runs both a headless data-collection pass (for plots)
# and gives you a command to run the interactive viewer version.
#
# Usage:
#   Headless (data + plots):   /opt/homebrew/anaconda3/bin/python3 perturbation.py
#   Interactive (viewer):      /opt/homebrew/anaconda3/bin/python3 perturbation.py --viewer gl
###########################################################################

import math
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warp as wp
import newton
import newton.examples


class PerturbedChain:
    """Spinning chain with an initial mode-2 (elliptical) perturbation.

    Identical to SpinningChain except the initial positions are perturbed:
        r(theta) = R * (1 + epsilon * cos(n * theta))
    where n=2 gives an ellipse and epsilon controls the amplitude.
    """

    def __init__(self, viewer, args=None,
                 num_links=32, perturbation_mode=2, perturbation_amplitude=0.05):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        self.num_links = num_links
        self.chain_radius = 1.0
        self.link_radius = 0.02
        self.omega = 2.0 * math.pi
        self.perturbation_mode = perturbation_mode
        self.perturbation_amplitude = perturbation_amplitude

        # Build perturbed circle geometry
        # r(theta) = R * (1 + epsilon * cos(n * theta))
        # This creates a shape with n lobes. For n=2:
        #   theta=0, pi:   r = R*(1+epsilon)  (stretched)
        #   theta=pi/2, 3pi/2: r = R*(1-epsilon)  (squeezed)
        # The result is an ellipse with semi-axes R*(1+eps) and R*(1-eps).
        points = []
        for i in range(self.num_links + 1):
            theta = 2.0 * math.pi * i / self.num_links
            r = self.chain_radius * (1.0 + perturbation_amplitude * math.cos(perturbation_mode * theta))
            points.append(wp.vec3(r * math.cos(theta), r * math.sin(theta), 0.0))

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1.0e2
        builder.default_shape_cfg.kd = 1.0e1
        builder.default_shape_cfg.mu = 1.0

        self.chain_bodies, self.chain_joints = builder.add_rod(
            positions=points,
            quaternions=None,
            radius=self.link_radius,
            bend_stiffness=1.0,
            bend_damping=0.0,
            stretch_stiffness=1.0e9,
            stretch_damping=0.0,
            label="chain",
            closed=True,
        )

        builder.color()
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=self.sim_iterations, friction_epsilon=0.1)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Set initial velocities consistent with rotation at omega.
        # Use the PERTURBED positions for the velocity calculation so
        # each body's velocity is tangent to its actual orbit.
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        for body_idx in self.chain_bodies:
            x, y = body_q[body_idx, 0], body_q[body_idx, 1]
            vx, vy = -self.omega * y, self.omega * x
            body_qd[body_idx] = [vx, vy, 0.0, 0.0, 0.0, self.omega]
        self.state_0.body_qd.assign(body_qd)

        # Instrumentation
        self.history_time = []
        self.history_radius_std = []
        self.history_mean_radius = []
        self.history_omega_measured = []
        # Track the "ellipticity" — the mode-2 amplitude in the radial profile
        # This is computed by Fourier-decomposing the radial distances.
        self.history_mode_amplitude = []

        self.print_every = 60

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control,
                           self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def measure(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        chain_pos = body_q[self.chain_bodies, :3]
        chain_vel = body_qd[self.chain_bodies, :3]

        centroid = np.mean(chain_pos, axis=0)
        offsets = chain_pos - centroid
        radii = np.linalg.norm(offsets, axis=1)
        mean_radius = np.mean(radii)
        radius_std = np.std(radii)

        speeds = np.linalg.norm(chain_vel, axis=1)
        safe_radii = np.where(radii > 1e-8, radii, 1e-8)
        omega_measured = np.mean(speeds / safe_radii)

        # -----------------------------------------------------------------
        # MODE AMPLITUDE via discrete Fourier transform
        # -----------------------------------------------------------------
        # The radial profile r(theta) can be decomposed into Fourier modes:
        #   r(theta) = R_mean + sum_n [ a_n cos(n*theta) + b_n sin(n*theta) ]
        #
        # The amplitude of mode n is sqrt(a_n^2 + b_n^2).
        # We compute this by taking the DFT of the radial distances.
        #
        # For equally-spaced samples, np.fft.rfft gives the complex Fourier
        # coefficients. The magnitude of coefficient [n] is the amplitude
        # of mode n (up to normalization by N).
        #
        # We track the mode we perturbed (self.perturbation_mode) to see
        # if it oscillates, decays, or grows.
        fft_coeffs = np.fft.rfft(radii)
        # Normalize by N to get physical amplitude
        mode_amp = 2.0 * np.abs(fft_coeffs[self.perturbation_mode]) / len(radii)

        self.history_time.append(self.sim_time)
        self.history_radius_std.append(radius_std)
        self.history_mean_radius.append(mean_radius)
        self.history_omega_measured.append(omega_measured)
        self.history_mode_amplitude.append(mode_amp)

        frame_num = len(self.history_time)
        if frame_num % self.print_every == 0:
            print(
                f"t={self.sim_time:6.2f}s  "
                f"omega={omega_measured:.3f}  "
                f"r_mean={mean_radius:.4f}  "
                f"r_std={radius_std:.4f}  "
                f"mode{self.perturbation_mode}_amp={mode_amp:.5f}"
            )

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.measure()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if self.state_0.body_q is not None:
            body_positions = self.state_0.body_q.numpy()
            assert np.isfinite(body_positions).all(), "Non-finite values"


def make_perturbation_figures(example, output_dir="figures"):
    """Generate perturbation experiment figures."""
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "figure.dpi": 150,
    })

    t = np.array(example.history_time)
    mode_amp = np.array(example.history_mode_amplitude)
    r_std = np.array(example.history_radius_std)
    omega = np.array(example.history_omega_measured)
    n = example.perturbation_mode
    eps = example.perturbation_amplitude

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: mode amplitude over time
    ax = axes[0]
    ax.plot(t, mode_amp, color='#d95f02', linewidth=1.5)
    ax.axhline(y=eps * example.chain_radius, color='gray', linestyle='--',
               alpha=0.5, label=f'Initial amplitude ({eps*example.chain_radius:.3f} m)')
    ax.set_ylabel(f"Mode-{n} Amplitude [m]")
    ax.set_title(f"Mode-{n} Perturbation: N={example.num_links}, "
                 f"ε={eps}, ω₀=2π rad/s")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: r_std and omega
    ax = axes[1]
    ax.plot(t, r_std, color='#1b9e77', linewidth=1.0, label='r_std')
    ax2 = ax.twinx()
    ax2.plot(t, omega, color='#7570b3', linewidth=1.0, alpha=0.7, label='ω')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Radius Std Dev [m]", color='#1b9e77')
    ax2.set_ylabel("ω [rad/s]", color='#7570b3')
    ax.set_title("Shape Distortion and Angular Velocity")
    ax.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "perturbation_mode2.png"))
    plt.close(fig)
    print(f"  Saved {output_dir}/perturbation_mode2.png")


if __name__ == "__main__":
    # Check if user wants interactive viewer or headless
    use_viewer = "--viewer" in sys.argv

    if use_viewer:
        # Interactive mode: use Newton's standard viewer setup
        viewer, args = newton.examples.init()
        example = PerturbedChain(viewer, args, num_links=32,
                                  perturbation_mode=2,
                                  perturbation_amplitude=0.05)
        newton.examples.run(example, args)
    else:
        # Headless mode: run simulation and generate figures
        print("=" * 60)
        print("PERTURBATION EXPERIMENT: Mode-2 Elliptical Distortion")
        print("=" * 60)

        class DummyViewer:
            def set_model(self, m): pass
            def apply_forces(self, s): pass
            def begin_frame(self, t): pass
            def log_state(self, s): pass
            def log_contacts(self, c, s): pass
            def end_frame(self): pass

        sim_seconds = 20.0
        num_frames = int(sim_seconds * 60)

        example = PerturbedChain(
            DummyViewer(), None,
            num_links=32,
            perturbation_mode=2,
            perturbation_amplitude=0.05,
        )

        print(f"Running {sim_seconds}s of simulation (N=32, mode-2, ε=0.05)...\n")
        for frame in range(num_frames):
            example.step()

        print(f"\nGenerating figures...")
        make_perturbation_figures(example)
        print("Done!")
