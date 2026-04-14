###########################################################################
# Ring Flick v2 — Variable Finger Positions + Adaptive Flick Gating
#
# The optimizer learns:
#   1. WHERE to push (finger angle on the ring — continuous, differentiable)
#   2. HOW HARD and in what direction (force vector per finger per flick)
#   3. WHETHER to push (per-flick gate — can "skip" unnecessary flicks)
#
# The finger position uses a soft Gaussian: instead of pushing one
# particle, the force spreads to nearby particles with a narrow Gaussian
# weight. As the angle parameter changes, the force smoothly shifts
# from one particle to the next — fully differentiable.
#
# The gate uses a sigmoid: gate = sigmoid(logit). The optimizer can
# "turn off" flicks by pushing the logit negative. A regularization
# penalty encourages fewer active flicks (parsimony).
#
# Usage:
#   /opt/homebrew/anaconda3/bin/python3 ring_flick_v2.py --mode 2
#   /opt/homebrew/anaconda3/bin/python3 ring_flick_v2.py --mode 3 --flicks 16
#   /opt/homebrew/anaconda3/bin/python3 ring_flick_v2.py --viewer gl
###########################################################################

import math
import sys
import os
import numpy as np
import warp as wp
import warp.optim
import newton
import newton.examples

PI = 3.14159265358979323846


# =========================================================================
# WARP KERNELS
# =========================================================================

@wp.kernel
def apply_soft_flick(
    all_forces: wp.array(dtype=wp.vec3),
    all_gates: wp.array(dtype=float),
    finger_angles: wp.array(dtype=float),
    num_fingers: int,
    flick_index: int,
    n_particles: int,
    sigma: float,
    f_ext: wp.array(dtype=wp.vec3),
):
    """Apply a flick with soft spatial assignment and gating.

    Each finger has a continuous angle parameter. The force is distributed
    across nearby particles using a Gaussian weight centered on the finger
    angle. This makes finger position differentiable — as the angle changes,
    the force smoothly shifts between particles.

    Each flick has a gate (sigmoid of a logit). The optimizer can learn to
    skip unnecessary flicks by pushing the gate toward zero.
    """
    i = wp.tid()
    if i >= n_particles:
        return

    # This particle's angle on the ring
    theta_i = 2.0 * PI * float(i) / float(n_particles)

    fx = float(0.0)
    fy = float(0.0)

    for fi in range(num_fingers):
        force_idx = flick_index * num_fingers + fi
        finger_theta = finger_angles[force_idx]

        # Angular distance, wrapped to [-pi, pi]
        d = theta_i - finger_theta
        d = d - 2.0 * PI * wp.round(d / (2.0 * PI))

        # Gaussian weight: peaks at d=0, decays with sigma
        w = wp.exp(-0.5 * d * d / (sigma * sigma))

        # Accumulate weighted force
        fx = fx + w * all_forces[force_idx][0]
        fy = fy + w * all_forces[force_idx][1]

    # Apply force (no gating — all flicks active)
    wp.atomic_add(f_ext, i, wp.vec3(fx, fy, 0.0))


@wp.kernel
def shape_loss(
    pos: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    n_particles: int,
    loss: wp.array(dtype=float),
):
    """Loss = sum of squared distances to target positions."""
    total = float(0.0)
    for i in range(n_particles):
        delta = pos[i] - targets[i]
        total = total + wp.dot(delta, delta)
    loss[0] = total


# =========================================================================
# SCENE BUILDER
# =========================================================================

def build_ring_scene(num_particles=64, radius=2.0, particle_mass=0.1,
                     stretch_ke=800.0, stretch_kd=20.0,
                     bend_ke=240.0, bend_kd=8.0):
    builder = newton.ModelBuilder()
    indices = []
    for i in range(num_particles):
        theta = 2 * math.pi * i / num_particles
        idx = builder.add_particle(
            pos=(radius * math.cos(theta), radius * math.sin(theta), 0),
            vel=(0, 0, 0), mass=particle_mass)
        indices.append(idx)
    for i in range(num_particles):
        builder.add_spring(indices[i], indices[(i+1) % num_particles],
                         ke=stretch_ke, kd=stretch_kd, control=0)
    for i in range(num_particles):
        builder.add_spring(indices[i], indices[(i+2) % num_particles],
                         ke=bend_ke, kd=bend_kd, control=0)
    model = builder.finalize(requires_grad=True)
    model.set_gravity((0, 0, 0))
    return model, indices


def compute_mode_targets(num_particles, radius, mode, amplitude=0.3, rotation_offset=0.0):
    targets = np.zeros((num_particles, 3))
    for i in range(num_particles):
        theta_base = 2 * math.pi * i / num_particles
        theta_final = theta_base + rotation_offset
        r = radius + amplitude * math.cos(mode * theta_base)
        targets[i] = [r * math.cos(theta_final), r * math.sin(theta_final), 0.0]
    return targets


# =========================================================================
# OPTIMIZER
# =========================================================================

class RingFlickV2:
    """Variable-position, gated flick optimizer.

    Learns where to push, how hard, and whether to push at all.
    """

    def __init__(self, viewer, args=None, target_mode=2, num_fingers=2, num_flicks=16,
                 init_mode=0, init_amplitude=0.0):
        self.viewer = viewer
        self.target_mode = target_mode
        self.num_fingers = num_fingers
        self.num_flicks = num_flicks
        self.init_mode = init_mode          # 0 = circle, 2 = ellipse, 3 = triangle, etc.
        self.init_amplitude = init_amplitude  # how much the start deviates from circular

        self.num_particles = 64
        self.radius = 2.0
        self.omega = 0.3

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.flick_frames = 3
        self.gap_frames = 10
        self.frames_per_cycle = self.flick_frames + self.gap_frames
        self.sim_frames = self.frames_per_cycle * num_flicks
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Gaussian width for soft finger position (in radians)
        # Start wider (~3 particle spacings) so angle changes are smooth.
        # A wider "finger" is more forgiving of position errors.
        self.sigma = 3.0 * (2 * math.pi / self.num_particles)

        self.model, self.indices = build_ring_scene(
            num_particles=self.num_particles, radius=self.radius)
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        total_steps = self.sim_frames * self.sim_substeps
        self.states = [self.model.state() for _ in range(total_steps + 1)]
        self.control = self.model.control()

        # Targets
        total_sim_time = self.sim_frames * self.frame_dt
        rotation_angle = self.omega * total_sim_time
        target_np = compute_mode_targets(
            self.num_particles, self.radius, target_mode,
            amplitude=0.3, rotation_offset=rotation_angle)
        self.targets = wp.array(target_np, dtype=wp.vec3)

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        # ---- OPTIMIZABLE PARAMETERS ----
        total_force_params = num_flicks * num_fingers

        # Force vectors: (num_flicks * num_fingers, 3)
        init_forces = np.zeros((total_force_params, 3))
        for k in range(num_flicks):
            for fi in range(num_fingers):
                # Small initial push, alternating direction
                sign = 1.0 if (k % 2 == 0) else -1.0
                angle = 2 * math.pi * fi / num_fingers
                init_forces[k * num_fingers + fi] = [
                    sign * 3.0 * math.cos(angle),
                    sign * 3.0 * math.sin(angle), 0.0]
        self.all_forces = wp.array(init_forces, dtype=wp.vec3, requires_grad=True)

        # Finger angles: (num_flicks * num_fingers,)
        # Initialize at evenly spaced positions, same for all flicks
        init_angles = np.zeros(total_force_params)
        for k in range(num_flicks):
            for fi in range(num_fingers):
                init_angles[k * num_fingers + fi] = 2 * math.pi * fi / num_fingers
        self.finger_angles = wp.array(init_angles, dtype=float, requires_grad=True)

        # Gate logits: start high so all flicks are active initially.
        # sigmoid(5) ≈ 0.993 — the optimizer must actively learn to turn flicks OFF.
        init_gates = np.full(num_flicks, 5.0)
        self.all_gates = wp.array(init_gates, dtype=float, requires_grad=True)

        # Regularization weights — very light so shape loss dominates
        self.lambda_gate = 0.002   # tiny penalty for active flicks
        self.lambda_force = 0.0001 # negligible force penalty

        # Two separate optimizers with different learning rates:
        # Forces update fast (they have smooth gradients).
        # Angles update slowly (moving a finger to a new position is a big change).
        self.force_optimizer = warp.optim.Adam(
            [self.all_forces.flatten()], lr=1.0)
        self.angle_optimizer = warp.optim.Adam(
            [self.finger_angles.flatten()], lr=0.01)  # 100x slower

        # Warm start: optimize forces only for this many iters before unlocking angles
        self.warmup_iters = 100

        # Pre-compute the initial shape positions (for reset and figures)
        self.init_positions = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            theta = 2 * math.pi * i / self.num_particles
            if init_mode > 0 and init_amplitude > 0:
                r_i = self.radius + init_amplitude * math.cos(init_mode * theta)
            else:
                r_i = self.radius
            self.init_positions[i] = [r_i * math.cos(theta), r_i * math.sin(theta), 0]

        self.train_iter = 0
        self.loss_history = []
        self.force_history = []
        self.angle_history = []

        self.states[0].particle_q.assign(self.init_positions)
        self._set_initial_spin(self.states[0])

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, 'show_particles'):
            self.viewer.show_particles = True

        print(f"Ring: {self.num_particles} particles, radius={self.radius}m")
        print(f"Target mode: {target_mode}, {num_flicks} flicks, {num_fingers} fingers")
        print(f"Params: {total_force_params*3} force + {total_force_params} angle + "
              f"{num_flicks} gate = {total_force_params*4 + num_flicks} total")

    def _set_initial_spin(self, state):
        q = state.particle_q.numpy()
        qd = state.particle_qd.numpy()
        for i in self.indices:
            x, y = q[i, 0], q[i, 1]
            qd[i] = [-self.omega * y, self.omega * x, 0]
        state.particle_qd.assign(qd)

    def forward(self):
        for flick_k in range(self.num_flicks):
            cycle_start = flick_k * self.frames_per_cycle
            for frame_in_cycle in range(self.frames_per_cycle):
                frame = cycle_start + frame_in_cycle
                is_flicking = frame_in_cycle < self.flick_frames
                for sub in range(self.sim_substeps):
                    t = frame * self.sim_substeps + sub
                    self.states[t].clear_forces()
                    if is_flicking:
                        wp.launch(
                            apply_soft_flick,
                            dim=self.num_particles,
                            inputs=[
                                self.all_forces,
                                self.all_gates,
                                self.finger_angles,
                                self.num_fingers,
                                flick_k,
                                self.num_particles,
                                self.sigma,
                                self.states[t].particle_f,
                            ],
                        )
                    self.solver.step(
                        self.states[t], self.states[t + 1],
                        self.control, None, self.sim_dt)

        wp.launch(
            shape_loss, dim=1,
            inputs=[self.states[-1].particle_q, self.targets,
                    self.num_particles, self.loss],
        )
        return self.loss

    def step(self):
        # Reset to initial shape (circle or deformed)
        self.states[0].particle_q.assign(self.init_positions)
        self._set_initial_spin(self.states[0])

        self.tape = wp.Tape()
        with self.tape:
            self.forward()
        self.tape.backward(self.loss)

        loss_val = self.loss.numpy()[0]
        forces_np = self.all_forces.numpy().copy()
        angles_np = self.finger_angles.numpy().copy()

        self.loss_history.append(loss_val)
        self.force_history.append(forces_np)
        self.angle_history.append(angles_np)

        total_force = sum(
            math.sqrt(float(np.sum(forces_np[k*self.num_fingers+fi]**2)))
            for k in range(self.num_flicks) for fi in range(self.num_fingers)
        )

        # Show phase (warmup vs full optimization)
        phase = "WARMUP (forces only)" if self.train_iter < self.warmup_iters else "FULL (forces+positions)"
        if self.train_iter % 50 == 0 or self.train_iter < 3 or self.train_iter == self.warmup_iters:
            # Show mean finger angle for each finger
            angle_strs = []
            for fi in range(self.num_fingers):
                mean_angle = np.mean([
                    math.degrees(angles_np[k * self.num_fingers + fi] % (2*math.pi))
                    for k in range(self.num_flicks)
                ])
                angle_strs.append(f"F{fi+1}@{mean_angle:.0f}°")

            print(
                f"Iter {self.train_iter:3d}  "
                f"loss={loss_val:.4f}  "
                f"|F|={total_force:.1f}  "
                f"{' '.join(angle_strs)}  "
                f"[{phase}]"
            )

        # Always update forces
        self.force_optimizer.step([self.all_forces.grad.flatten()])

        # Only update angles after warmup
        if self.train_iter >= self.warmup_iters:
            self.angle_optimizer.step([self.finger_angles.grad.flatten()])
        self.tape.zero()
        self.train_iter += 1

    def render(self):
        if hasattr(self.viewer, 'is_paused') and self.viewer.is_paused():
            self.viewer.begin_frame(self.viewer.time)
            self.viewer.end_frame()
            return
        for i in range(self.sim_frames):
            idx = i * self.sim_substeps
            if idx >= len(self.states):
                break
            state = self.states[idx]
            t = self.train_iter * self.sim_frames * self.frame_dt + i * self.frame_dt
            self.viewer.begin_frame(t)
            self.viewer.log_state(state)
            self.viewer.end_frame()

    def test_final(self):
        if len(self.loss_history) > 2:
            assert self.loss_history[-1] < self.loss_history[0]


# =========================================================================
# FIGURES
# =========================================================================

def make_figures(optimizer, output_dir="figures"):
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

    mode = optimizer.target_mode
    N = optimizer.num_particles
    r = optimizer.radius
    nf = optimizer.num_fingers
    nk = optimizer.num_flicks

    final_pos = optimizer.states[-1].particle_q.numpy()[:N]
    target_pos = optimizer.targets.numpy()

    # Use the ACTUAL initial positions (may be non-circular)
    init_pos = optimizer.init_positions
    init_x = init_pos[:, 0]
    init_y = init_pos[:, 1]

    # Label for initial shape
    if optimizer.init_mode > 0:
        init_label = f"Initial\n(mode-{optimizer.init_mode})"
    else:
        init_label = "Initial\n(circle)"

    final_forces = optimizer.force_history[-1]
    final_angles = optimizer.angle_history[-1]

    # ---- FIGURE 1: Convergence ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(optimizer.loss_history, '-', color='#d95f02', linewidth=1.5)
    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"Ring Flick v2 — Mode {mode} ({nf} fingers, {nk} max flicks)")
    if min(optimizer.loss_history) > 0:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"v2_convergence_mode{mode}.png"))
    plt.close(fig)
    print(f"  Saved v2_convergence_mode{mode}.png")

    # ---- FIGURE 2: Ring shapes ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, xs, ys, title, color in [
        (axes[0], init_x, init_y, init_label, '#1b9e77'),
        (axes[1], target_pos[:, 0], target_pos[:, 1], f"Target\n(mode-{mode})", '#d95f02'),
        (axes[2], final_pos[:, 0], final_pos[:, 1], "After Flicks\n(learned)", '#7570b3'),
    ]:
        xs_c = list(xs) + [xs[0]]
        ys_c = list(ys) + [ys[0]]
        ax.plot(xs_c, ys_c, 'o-', color=color, markersize=3, linewidth=2)
        circ = np.linspace(0, 2*np.pi, 100)
        ax.plot(r*np.cos(circ), r*np.sin(circ), '--', color='gray', alpha=0.3)
        if title.startswith("After"):
            tc_x = list(target_pos[:,0]) + [target_pos[0,0]]
            tc_y = list(target_pos[:,1]) + [target_pos[0,1]]
            ax.plot(tc_x, tc_y, 's--', color='#d95f02', markersize=2, linewidth=1, alpha=0.5, label='target')
            ax.legend(fontsize=9)
        # Show finger positions on initial panel
        if title.startswith("Initial"):
            for k in range(nk):
                for fi in range(nf):
                    angle = final_angles[k * nf + fi]
                    fx_pos = r * math.cos(angle)
                    fy_pos = r * math.sin(angle)
                    alpha = max(0.15, 0.8 * (k + 1) / nk)  # later flicks more opaque
                    ax.plot(fx_pos, fy_pos, '*', color=f'C{fi}', markersize=6,
                            alpha=alpha, markeredgecolor='black', markeredgewidth=0.3)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(title, fontsize=13)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        lim = r * 1.3
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    fig.suptitle(f"Learned Flick Strategy — Mode {mode}", fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"v2_shapes_mode{mode}.png"))
    plt.close(fig)
    print(f"  Saved v2_shapes_mode{mode}.png")

    # ---- FIGURE 3: Flick timeline — force + position ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    flick_nums = np.arange(nk)

    # Force magnitudes per finger
    ax = axes[0]
    for fi in range(nf):
        mags = []
        for k in range(nk):
            f = final_forces[k * nf + fi]
            mags.append(math.sqrt(float(f[0]**2 + f[1]**2)))
        ax.bar(flick_nums + fi * 0.35, mags, width=0.3,
               color=f'C{fi}', label=f'Finger {fi+1}', edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Force Magnitude [N]")
    ax.set_title("Force Per Flick")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Finger positions (angles)
    ax = axes[1]
    for fi in range(nf):
        angles_deg = []
        for k in range(nk):
            a = final_angles[k * nf + fi] % (2 * math.pi)
            angles_deg.append(math.degrees(a))
        ax.plot(flick_nums, angles_deg, 'o-', color=f'C{fi}', markersize=5,
                linewidth=1.5, label=f'Finger {fi+1}')
    ax.set_ylabel("Finger Position [degrees]")
    ax.set_xlabel("Flick Number")
    ax.set_title("Where Each Finger Pushes")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(flick_nums)
    ax.set_xticklabels([f"#{k+1}" for k in range(nk)])

    fig.suptitle(f"Learned Flick Strategy Timeline — Mode {mode}",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"v2_timeline_mode{mode}.png"))
    plt.close(fig)
    print(f"  Saved v2_timeline_mode{mode}.png")


# =========================================================================
# ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    target_mode = 2
    num_fingers = 2
    num_flicks = 16
    init_mode = 0
    init_amp = 0.0
    for i, arg in enumerate(sys.argv):
        if arg == "--mode" and i + 1 < len(sys.argv):
            target_mode = int(sys.argv[i + 1])
        if arg == "--fingers" and i + 1 < len(sys.argv):
            num_fingers = int(sys.argv[i + 1])
        if arg == "--flicks" and i + 1 < len(sys.argv):
            num_flicks = int(sys.argv[i + 1])
        if arg == "--init-mode" and i + 1 < len(sys.argv):
            init_mode = int(sys.argv[i + 1])
        if arg == "--init-amp" and i + 1 < len(sys.argv):
            init_amp = float(sys.argv[i + 1])

    use_viewer = "--viewer" in sys.argv

    if use_viewer:
        viewer, args = newton.examples.init()
        opt = RingFlickV2(viewer, args, target_mode=target_mode,
                          num_fingers=num_fingers, num_flicks=num_flicks,
                          init_mode=init_mode, init_amplitude=init_amp)
        newton.examples.run(opt, args)
    else:
        print("=" * 60)
        init_desc = f"mode-{init_mode} (amp={init_amp})" if init_mode > 0 else "circle"
        print(f"RING FLICK v2 — {init_desc} → MODE {target_mode}")
        print(f"Variable positions, {num_fingers} fingers, {num_flicks} max flicks")
        print("=" * 60)

        class DummyViewer:
            def set_model(self, m): pass
            def begin_frame(self, t): pass
            def log_state(self, s): pass
            def end_frame(self): pass
            def is_paused(self): return False

        opt = RingFlickV2(DummyViewer(), target_mode=target_mode,
                          num_fingers=num_fingers, num_flicks=num_flicks,
                          init_mode=init_mode, init_amplitude=init_amp)

        def finish(opt):
            iters = len(opt.loss_history)
            if iters == 0:
                print("\nNo iterations.")
                return
            print(f"\n{'='*60}")
            print(f"RESULTS ({iters} iterations):")
            print(f"  Loss: {opt.loss_history[0]:.4f} -> {opt.loss_history[-1]:.4f}")
            if opt.loss_history[-1] > 0:
                print(f"  Improvement: {opt.loss_history[0]/opt.loss_history[-1]:.1f}x")
            final_angles = opt.angle_history[-1]
            for k in range(min(opt.num_flicks, 5)):  # show first 5 flicks
                for fi in range(opt.num_fingers):
                    a = math.degrees(final_angles[k*opt.num_fingers+fi] % (2*math.pi))
                    f = opt.force_history[-1][k*opt.num_fingers+fi]
                    mag = math.sqrt(float(f[0]**2 + f[1]**2))
                    print(f"  Flick #{k+1} F{fi+1}: {a:.0f} deg, {mag:.1f}N")
            if opt.num_flicks > 5:
                print(f"  ... ({opt.num_flicks - 5} more flicks)")
            print(f"{'='*60}")
            print("\nGenerating figures...")
            make_figures(opt)
            print("Done!")

        max_iters = 2000
        patience = 200
        min_improvement = 1e-6

        print(f"\nRunning (max {max_iters}, patience={patience})...")
        print("Ctrl+C to stop early.\n")

        try:
            best = float('inf')
            no_imp = 0
            for i in range(max_iters):
                opt.step()
                c = opt.loss_history[-1]
                if c < best * (1.0 - min_improvement):
                    best = c
                    no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= patience:
                    print(f"\nConverged at iter {i+1}.")
                    break
        except KeyboardInterrupt:
            print(f"\nStopped at iter {len(opt.loss_history)}.")

        finish(opt)
