###########################################################################
# N Sweep — Phase 1 Headline Experiment
#
# Runs the spinning chain simulation at multiple values of N (number of
# links), collects data headlessly, and generates matplotlib figures.
#
# This produces the key Phase 1 figures:
#   1. Omega decay over time for each N (shows energy dissipation)
#   2. Initial omega retention vs N (discrete-to-continuous transition)
#   3. Radius stability (r_std) vs N
#   4. Tension validation (T_kin / T_ana) vs time for each N
#
# Usage:
#   /opt/homebrew/anaconda3/bin/python3 sweep.py
#
# Runs headlessly (no viewer window) — takes a few minutes depending on
# how many N values and how many seconds of sim time per run.
###########################################################################

import math
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# We'll import and reuse the SpinningChain class, but drive it without
# the interactive viewer. Newton's examples.run() loop calls step() and
# render() — we just need step().


def run_single(num_links, sim_seconds=20.0):
    """Run a spinning chain with the given N for sim_seconds and return data.

    This is the core of the sweep. We construct a SpinningChain, step it
    forward frame by frame (no rendering), and collect the history.

    Returns a dict with all the time-series data for this run.
    """
    # We need to import newton inside the function because Warp initializes
    # on import, and we want each run to start clean.
    import warp as wp
    import newton
    import newton.examples

    # Create a headless viewer — no window, just data collection.
    # Newton's ViewerGL needs to be replaced with a dummy that does nothing.
    # We can use newton's --headless flag by faking args, or build a minimal
    # dummy viewer.

    class DummyViewer:
        """Minimal stand-in for the Newton viewer when running headlessly.

        The simulation calls viewer.apply_forces(), viewer.set_model(), etc.
        We just need these methods to exist and do nothing.
        """
        def set_model(self, model):
            pass
        def apply_forces(self, state):
            pass
        def begin_frame(self, t):
            pass
        def log_state(self, state):
            pass
        def log_contacts(self, contacts, state):
            pass
        def end_frame(self):
            pass
        def set_camera(self, **kwargs):
            pass

    # Build the chain by directly constructing the scene (same logic as
    # SpinningChain.__init__ but without importing the class, to keep
    # this script self-contained and easy to understand).
    fps = 60
    frame_dt = 1.0 / fps
    sim_substeps = 10
    sim_iterations = 5
    sim_dt = frame_dt / sim_substeps

    chain_radius = 1.0
    link_radius = 0.02
    omega = 2.0 * math.pi

    # Build circle geometry
    points = []
    for i in range(num_links + 1):
        theta = 2.0 * math.pi * i / num_links
        points.append(wp.vec3(chain_radius * math.cos(theta),
                              chain_radius * math.sin(theta), 0.0))

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 1.0

    chain_bodies, chain_joints = builder.add_rod(
        positions=points,
        quaternions=None,
        radius=link_radius,
        bend_stiffness=1.0,
        bend_damping=0.0,
        stretch_stiffness=1.0e9,
        stretch_damping=0.0,
        label="chain",
        closed=True,
    )

    builder.color()
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))

    solver = newton.solvers.SolverVBD(model, iterations=sim_iterations, friction_epsilon=0.1)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    # Set initial angular velocity
    body_q = state_0.body_q.numpy()
    body_qd = state_0.body_qd.numpy()
    for body_idx in chain_bodies:
        x, y = body_q[body_idx, 0], body_q[body_idx, 1]
        vx, vy = -omega * y, omega * x
        body_qd[body_idx] = [vx, vy, 0.0, 0.0, 0.0, omega]
    state_0.body_qd.assign(body_qd)

    # Read chain mass for tension calculation
    body_mass_all = model.body_mass.numpy()
    chain_mass = body_mass_all[chain_bodies]
    total_mass = np.sum(chain_mass)
    dtheta = 2.0 * math.pi / num_links

    # Data collection
    history = {
        "num_links": num_links,
        "total_mass": float(total_mass),
        "link_mass": float(chain_mass[0]),
        "time": [],
        "omega_measured": [],
        "mean_radius": [],
        "radius_std": [],
        "centroid_drift": [],
        "tension_kinematic": [],
        "tension_analytic": [],
    }

    # Run simulation
    num_frames = int(sim_seconds * fps)
    sim_time = 0.0

    for frame in range(num_frames):
        # Physics substeps
        for _ in range(sim_substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        sim_time += frame_dt

        # Measure
        bq = state_0.body_q.numpy()
        bqd = state_0.body_qd.numpy()
        chain_pos = bq[chain_bodies, :3]
        chain_vel = bqd[chain_bodies, :3]

        centroid = np.mean(chain_pos, axis=0)
        offsets = chain_pos - centroid
        radii = np.linalg.norm(offsets, axis=1)
        mean_radius = np.mean(radii)
        radius_std = np.std(radii)

        speeds = np.linalg.norm(chain_vel, axis=1)
        safe_radii = np.where(radii > 1e-8, radii, 1e-8)
        omega_per_body = speeds / safe_radii
        omega_measured = np.mean(omega_per_body)

        centroid_drift = np.linalg.norm(centroid)

        # Tension (kinematic method only — stretch doesn't work with VBD)
        T_per_link = chain_mass * speeds**2 / (safe_radii * dtheta)
        T_kinematic = np.mean(T_per_link)

        mu = total_mass / (2.0 * math.pi * mean_radius)
        T_analytic = mu * omega_measured**2 * mean_radius**2

        # Store
        history["time"].append(sim_time)
        history["omega_measured"].append(float(omega_measured))
        history["mean_radius"].append(float(mean_radius))
        history["radius_std"].append(float(radius_std))
        history["centroid_drift"].append(float(centroid_drift))
        history["tension_kinematic"].append(float(T_kinematic))
        history["tension_analytic"].append(float(T_analytic))

    return history


def make_figures(all_data, output_dir="figures"):
    """Generate publication-quality matplotlib figures from sweep data.

    Creates the headline figures for Phase 1 of the writeup.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use a clean style
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
    })

    omega_target = 2.0 * math.pi
    # Color map: darker for larger N
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(all_data)))

    # =====================================================================
    # FIGURE 1: Omega decay over time for each N
    # =====================================================================
    # This is the main "energy dissipation" figure. Shows how the VBD solver
    # bleeds angular momentum, and whether the rate depends on N.
    fig, ax = plt.subplots()
    for i, d in enumerate(all_data):
        ax.plot(d["time"], d["omega_measured"],
                color=colors[i], label=f'N={d["num_links"]}', linewidth=1.5)
    ax.axhline(y=omega_target, color='gray', linestyle='--', alpha=0.5, label=f'target ω={omega_target:.2f}')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Measured ω [rad/s]")
    ax.set_title("Angular Velocity Decay vs. Number of Links")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "omega_decay.png"))
    plt.close(fig)
    print(f"  Saved {output_dir}/omega_decay.png")

    # =====================================================================
    # FIGURE 2: Initial omega retention vs N
    # =====================================================================
    # Shows the discrete-to-continuous transition: at what N does the initial
    # velocity setup stop losing energy to polygon→circle mismatch?
    # We use omega at t=1s as "initial" to let the solver settle.
    fig, ax = plt.subplots()
    N_values = [d["num_links"] for d in all_data]
    # Find omega at t≈1.0s (frame 60)
    omega_at_1s = []
    for d in all_data:
        idx_1s = min(59, len(d["omega_measured"]) - 1)  # frame 60 = t=1.0s
        omega_at_1s.append(d["omega_measured"][idx_1s])

    retention = [o / omega_target * 100 for o in omega_at_1s]
    ax.semilogx(N_values, retention, 'o-', color='#2c7fb8', markersize=8, linewidth=2)
    ax.set_xlabel("Number of Links (N)")
    ax.set_ylabel("ω Retention at t=1s [%]")
    ax.set_title("Discrete-to-Continuous Transition: Initial Energy Retention")
    ax.set_ylim(50, 100)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(N_values)
    ax.set_xticklabels([str(n) for n in N_values])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "omega_retention_vs_N.png"))
    plt.close(fig)
    print(f"  Saved {output_dir}/omega_retention_vs_N.png")

    # =====================================================================
    # FIGURE 3: Radius stability (r_std) over time for each N
    # =====================================================================
    # Shows which N values produce stable circular shapes and which wobble.
    fig, ax = plt.subplots()
    for i, d in enumerate(all_data):
        ax.plot(d["time"], d["radius_std"],
                color=colors[i], label=f'N={d["num_links"]}', linewidth=1.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Radius Std Dev [m]")
    ax.set_title("Shape Stability: Deviation from Circularity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "radius_stability.png"))
    plt.close(fig)
    print(f"  Saved {output_dir}/radius_stability.png")

    # =====================================================================
    # FIGURE 4: Tension validation — T_kin / T_ana over time
    # =====================================================================
    # Should be flat at 1.0 for all N. Confirms kinematic self-consistency.
    fig, ax = plt.subplots()
    for i, d in enumerate(all_data):
        T_kin = np.array(d["tension_kinematic"])
        T_ana = np.array(d["tension_analytic"])
        # Avoid division by zero at very late times when omega is tiny
        valid = T_ana > 0.1
        ratio = np.where(valid, T_kin / T_ana, np.nan)
        ax.plot(np.array(d["time"])[valid], ratio[valid],
                color=colors[i], label=f'N={d["num_links"]}', linewidth=1.0)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("T_kinematic / T_analytic")
    ax.set_title("Tension Validation: Kinematic vs. Analytic Prediction")
    ax.set_ylim(0.95, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "tension_validation.png"))
    plt.close(fig)
    print(f"  Saved {output_dir}/tension_validation.png")

    # =====================================================================
    # FIGURE 5: Summary panel (2x2) — the headline figure
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: omega decay
    ax = axes[0, 0]
    for i, d in enumerate(all_data):
        ax.plot(d["time"], d["omega_measured"],
                color=colors[i], label=f'N={d["num_links"]}', linewidth=1.5)
    ax.axhline(y=omega_target, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ω [rad/s]")
    ax.set_title("(a) Angular Velocity Decay")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: omega retention vs N
    ax = axes[0, 1]
    ax.semilogx(N_values, retention, 'o-', color='#2c7fb8', markersize=8, linewidth=2)
    ax.set_xlabel("N (links)")
    ax.set_ylabel("ω retention at t=1s [%]")
    ax.set_title("(b) Discrete→Continuous Transition")
    ax.set_ylim(50, 100)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(N_values)
    ax.set_xticklabels([str(n) for n in N_values])

    # Bottom-left: radius stability
    ax = axes[1, 0]
    for i, d in enumerate(all_data):
        ax.plot(d["time"], d["radius_std"],
                color=colors[i], label=f'N={d["num_links"]}', linewidth=1.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("r std dev [m]")
    ax.set_title("(c) Shape Stability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: tension validation
    ax = axes[1, 1]
    for i, d in enumerate(all_data):
        T_kin = np.array(d["tension_kinematic"])
        T_ana = np.array(d["tension_analytic"])
        valid = T_ana > 0.1
        ratio = np.where(valid, T_kin / T_ana, np.nan)
        ax.plot(np.array(d["time"])[valid], ratio[valid],
                color=colors[i], label=f'N={d["num_links"]}', linewidth=1.0)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("T_kin / T_analytic")
    ax.set_title("(d) Tension Validation")
    ax.set_ylim(0.95, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Spinning Closed Chain: N-Sweep Summary", fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary_panel.png"))
    plt.close(fig)
    print(f"  Saved {output_dir}/summary_panel.png")


if __name__ == "__main__":
    # =====================================================================
    # SWEEP CONFIGURATION
    # =====================================================================
    N_values = [4, 8, 16, 32, 64, 128]
    sim_seconds = 20.0  # how long to run each simulation

    print("=" * 60)
    print("N-SWEEP: Spinning Closed Chain")
    print(f"N values: {N_values}")
    print(f"Sim time: {sim_seconds}s per run")
    print("=" * 60)

    all_data = []
    for N in N_values:
        print(f"\nRunning N={N}...")
        data = run_single(N, sim_seconds=sim_seconds)

        # Print summary for this run
        omega_1s = data["omega_measured"][min(59, len(data["omega_measured"])-1)]
        omega_final = data["omega_measured"][-1]
        print(f"  omega at t=1s: {omega_1s:.3f} (retention: {omega_1s/6.283*100:.1f}%)")
        print(f"  omega at t={sim_seconds}s: {omega_final:.3f}")
        print(f"  r_std range: {min(data['radius_std']):.4f} - {max(data['radius_std']):.4f}")

        all_data.append(data)

    # Save raw data as JSON for later analysis
    print("\nSaving raw data...")
    os.makedirs("data", exist_ok=True)
    for d in all_data:
        fname = f"data/sweep_N{d['num_links']}.json"
        with open(fname, "w") as f:
            json.dump(d, f)
        print(f"  Saved {fname}")

    # Generate figures
    print("\nGenerating figures...")
    make_figures(all_data)

    print("\nDone!")
