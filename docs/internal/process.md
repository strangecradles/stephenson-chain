# Process Log — The Great Chain / Newton Project

A running record of what I built, what I learned, and what went wrong along the way. This is the raw material for the writeup — every finding here is something I can reference or expand on later.

---

## Day 1 — April 12, 2026: Install, Explore, First Simulation

### Setup

- Installed Newton v1.0.0 via pip into conda base environment (Python 3.13, Apple Silicon ARM)
- No CUDA — running on CPU via Warp's ARM fallback. Slower but fully functional.
- Newton examples live at `~/newton-examples/`, organized by topic: basic, cable, cloth, contacts, diffsim, ik, mpm, multiphysics, robot, selection, sensors, softbody

### Understanding Newton's Architecture

Newton has four layers that I need to understand as a stack:

1. **ModelBuilder** — declarative scene description. You add bodies, joints, shapes, contact materials. Nothing is simulated yet. Think of it as writing the recipe.
2. **Model** — the finalized, immutable description. Created by `builder.finalize()`. Lives on CPU or GPU.
3. **State** — mutable physics state at one instant. Holds `body_q` (positions + quaternion orientations, shape [N,7]) and `body_qd` (velocities, shape [N,6]). You need two state objects for double-buffering.
4. **Solver** — advances State forward by one timestep. Two solvers matter for this project:
   - `SolverVBD` (Vertex Block Descent) — position-based, iterative, good for cables/contacts. All cable examples use this.
   - `SolverSemiImplicit` — classical semi-implicit Euler. Supports automatic differentiation via `wp.Tape()`. The diffsim examples use this.

Cables in Newton are **not** continuous objects — they're chains of rigid capsule bodies connected by "cable joints" that enforce stretch stiffness and bend stiffness. A cable with 50 elements is literally 50 separate capsules linked end-to-end. This is the discrete analog of a continuous elastic rod.

### Answering the Go/No-Go Questions

These were the critical feasibility questions from the roadmap:

| Question | Answer | Source |
|----------|--------|--------|
| Does the cable solver support bending stiffness? | **Yes.** `add_rod()` takes `bend_stiffness` parameter. | `example_cable_bend.py` sweeps 0.1 to 1000 |
| Can it form a closed loop? | **Yes.** `add_rod(closed=True)` is a first-class feature. | Found in `builder.py` source, lines 6087-6095 |
| Can deformable + rigid bodies coexist? | **Yes.** VBD solver handles soft bodies + cloth + rigid in one scene. | `example_softbody_dropping_to_cloth.py` |
| Can gravity be set to zero? | **Yes.** `model.set_gravity((0.0, 0.0, 0.0))` after finalize. | Tested directly |
| Joint types for finger (Phase 2)? | **Revolute (hinge) joints available** via `add_joint_revolute()`. | `builder.py` source |

Bottom line: Phase 1 is fully feasible, and Phase 2 (tendon-driven finger) should be buildable on Newton's existing primitives.

### Building the Annotated Cable Bend Example

Copied `example_cable_bend.py` into the project and added extensive comments explaining:
- The timing model (fps → substeps → sim_dt)
- What `builder.color()` actually does (graph coloring for parallel constraint solving, not visual)
- Penalty-based contacts vs. hard constraints (LCP)
- Making bodies kinematic by zeroing mass/inertia
- The double-buffer state swap pattern

### Building spinning_chain.py — The First Real Simulation

Built a closed chain of N rigid links in zero gravity with initial angular velocity. Key decisions and gotchas:

**Geometry: points on a circle.**
```python
theta = 2.0 * math.pi * i / self.num_links
points.append(wp.vec3(r * math.cos(theta), r * math.sin(theta), 0.0))
```
N+1 points where the last coincides with the first, then `add_rod(closed=True)` handles the loop-closing joint.

**Bug: body_qd velocity convention.**
Newton's `body_qd` array is `[vx, vy, vz, wx, wy, wz]` — linear first, then angular. This is the OPPOSITE of the Featherstone spatial algebra convention `[angular, linear]` that I initially assumed. Discovered when the chain floated upward in z — I had accidentally put omega (angular velocity) into the linear z slot.

**Bug: builder.gravity = 0.0 didn't work.**
Setting `builder.gravity = 0.0` before finalize didn't zero the gravity vector. Had to use `model.set_gravity((0.0, 0.0, 0.0))` after finalize instead.

**Initial velocity setup.** For rigid rotation at angular velocity omega about the z-axis:
- Linear velocity of body at position (x, y): v = omega_hat x r = (-omega*y, omega*x, 0)
- Angular velocity of each body: (0, 0, omega)
- Must call `state.body_qd.assign(numpy_array)` to write back — modifying the numpy copy alone has no effect.

---

## Day 1 continued — First Instrumentation and Measurements

### What we measure each frame

Added a `measure()` method called every display frame that computes:

1. **Centroid** — average position of all chain bodies. Should stay near origin (zero net momentum).
2. **Mean radius** — average distance from centroid to each body. Should equal `chain_radius` for a perfect circle.
3. **Radius std dev** — standard deviation of radii. Our "circularity" metric. 0 = perfect circle.
4. **Measured omega** — average of (speed / radius) across all bodies. Compares to target omega.

### First results: N=4, omega=2*pi, bend_stiffness=1.0

```
t=  1.00s  centroid=(-0.0000, +0.0000, -0.0000)  r_mean=1.0004  r_std=0.0006  omega=4.203 (target=6.283)
t=  5.00s  centroid=(-0.0008, +0.0003, -0.0000)  r_mean=1.0003  r_std=0.0003  omega=3.604 (target=6.283)
t= 10.00s  centroid=(-0.0032, +0.0033, +0.0000)  r_mean=1.0003  r_std=0.0004  omega=3.125 (target=6.283)
t= 20.00s  centroid=(-0.0034, +0.0079, +0.0000)  r_mean=1.0002  r_std=0.0001  omega=2.554 (target=6.283)
```

**Key findings:**

1. **Centroid is stable** — sub-millimeter drift over 22 seconds. No net momentum leak. Zero gravity works.

2. **Shape is stable** — r_mean holds at 1.0003 (essentially perfect), r_std < 0.001. Even a 4-link square-ish chain holds its circular shape.

3. **Omega decays steadily** — this is the most interesting finding. Two effects:
   - **Immediate drop from 6.283 to ~4.2** at t=0. About 1/3 of kinetic energy lost instantly. At N=4 the chain is a square, not a circle — initial velocities are tangent to a circle but the geometry is polygonal. The solver reconciles this mismatch by dissipating energy on the first step.
   - **Steady ~3% per second decay** after that. This is the VBD solver numerically dissipating energy. Position-based solvers are known for this — they don't conserve energy the way symplectic integrators do. The explicit `bend_damping=1.0e-2` parameter also contributes.

4. **This energy dissipation is itself a reportable finding.** The chain holds its shape perfectly but bleeds angular momentum. For the writeup, this characterizes a real limitation of the VBD solver for energy-conserving scenarios.

### Experiment: bend_damping=0.0 vs 1.0e-2

Set `bend_damping=0.0` (everything else identical: N=4, omega=2*pi, bend_stiffness=1.0).

**Result: identical omega decay.** Every value matches to three decimal places. The `bend_damping` parameter has zero measurable effect on energy dissipation for this scenario.

**Conclusion: the energy decay is entirely intrinsic to the VBD solver.** This is a known property of position-based methods — they project constraints by moving positions directly rather than applying forces, which doesn't conserve energy. This is worth reporting in the writeup: for scenarios where energy conservation matters (like a freely spinning chain), VBD introduces artificial dissipation that isn't controllable via the damping parameters.

This also means we can leave `bend_damping=0.0` going forward without losing anything.

### Experiment: N sweep — N=4, N=16, N=64

Ran the same simulation (omega=2*pi, bend_stiffness=1.0, bend_damping=0.0) at three link counts.

**Summary table:**

| | N=4 | N=16 | N=64 |
|---|---|---|---|
| Initial omega (t=1s) | 4.20 | 5.76 | 5.78 |
| Initial omega retention | 67% | 92% | 92% |
| omega at t=20s | 2.55 | 3.25 | 3.32 |
| r_std range | 0.000-0.001 | 0.000-0.001 | **0.005-0.018** |
| r_mean stability | rock solid (1.0003) | rock solid (1.002) | **oscillating (0.996-1.015)** |
| Out-of-plane motion | none | none | **visible z oscillations** |

**Finding 1: Discrete-to-continuous transition in initial energy retention.**
The N=4 chain (a square) loses 33% of kinetic energy instantly as the solver reconciles circular velocities with polygonal geometry. By N=16 (a 16-gon, close to a circle), only 8% is lost. N=16→64 gains almost nothing (92% → 92%). The transition is essentially complete by N~16 for this quantity.

**Finding 2: Long-term decay rate is independent of N.**
All three chains end up around omega ≈ 3.2-3.3 at t=20s regardless of N. The VBD solver's numerical dissipation is a property of the solver algorithm, not the mesh resolution. This is an important negative result — making the discretization finer doesn't fix the energy conservation problem.

**Finding 3 (surprise): N=64 is LESS shape-stable than N=4 or N=16.**
Counterintuitively, the finest discretization has the worst circularity. r_std is 10-100x larger than at N=4 or N=16 and oscillates. r_mean also oscillates between 0.996 and 1.015. The chain is "breathing" (expanding and contracting) and developing traveling waves.

**Explanation:** More links = more degrees of freedom = more modes that can be excited. At N=4, the system is so constrained there are barely any shape modes — it's essentially a rigid square. At N=64, tiny numerical perturbations (floating-point rounding in the solver) are enough to excite in-plane breathing modes and out-of-plane bending modes. The chain at high N is more "physically realistic" (a real chain would wobble), while the low-N chain is artificially stable because it's over-constrained.

**Finding 4: Out-of-plane oscillations at N=64.**
Visible z-direction oscillations appear in the viewer at N=64 but not at lower N. The chain is initialized perfectly in the XY plane with XY-only velocities, but with 64 degrees of freedom, numerical noise from the solver is sufficient to excite out-of-plane bending modes. With bend_stiffness=1.0 (nonzero), these modes oscillate rather than growing — they have a restoring force from bending stiffness. This is actually the stability observation the roadmap planned for Day 5, emerging naturally from the N sweep.

### Questions remaining

- ~~Does omega decay slower with bend_damping=0.0?~~ **No — no effect.**
- ~~Does omega decay slower at larger N?~~ **No — decay rate is solver-intrinsic, independent of N.**
- ~~Does the initial omega drop decrease at larger N?~~ **Yes — drops from 33% loss at N=4 to 8% at N=16, then plateaus.**
- What happens with a deliberate mode-2 perturbation? (Programmatic elliptical distortion)
- ~~Can we measure tension in the chain links and compare to T = μω²r²?~~ **Partially. See tension validation below.**
- Would increasing sim_substeps or sim_iterations reduce the decay rate?

---

## Day 2 — April 13, 2026: Tension Validation

### The two methods for measuring tension

Implemented two independent ways to measure chain tension and compare to the analytic prediction T = μω²r²:

**Method 1 — Kinematic (centripetal balance):**
For each link: T = m * v² / (r * dθ), where dθ = 2π/N. Uses measured positions and velocities.

**Method 2 — Stretch (Hooke's law from constraint gap):**
Measure the gap at each cable joint's attachment points, compute T = stretch_stiffness * gap / rest_length.

**Analytic prediction:**
T = μω²r² where μ = M_total / (2πr). Uses MEASURED omega and radius, not targets.

### Result: Method 1 works perfectly, Method 2 fails completely

**T_kin / T_ana = 1.000 at every timestep.** Perfect agreement.

However, this is actually **tautological** — both formulas are algebraically identical:
- T_kin = m * v² / (r * dθ) = m * (ωr)² / (r * 2π/N) = M * ω² * r / (2π)
- T_ana = μ * ω² * r² = (M/(2πr)) * ω² * r² = M * ω² * r / (2π)

Same formula, same inputs, same answer. This confirms self-consistency of the kinematics, but doesn't independently validate Newton's constraint solver.

**T_str was off by a factor of ~50,000.** At N=16, t=1s: T_str ≈ 2.4 million N vs T_ana ≈ 44 N.

### Why Method 2 fails: VBD is not a force-based solver

This is one of the most important findings of the project so far.

**VBD (Vertex Block Descent) is a position-based solver.** It looks at constraint violations (gaps between joint attachment points) and directly nudges positions to reduce them. It never computes explicit forces. After a finite number of solver iterations (we use 5), there's always a residual gap.

**The residual gap reflects solver convergence, not physical stretch.** With stretch_stiffness = 1e9 and real tension ~44 N, the physical stretch should be 44/1e9 ≈ 44 nanometers per link. But the VBD residual gap is ~1 mm — limited by the number of solver iterations, not by physics. Multiplying this solver error by 1e9 gives ~2 million N of nonsense.

**Attempted fixes:**
1. First tried measuring body-origin-to-origin distances (wrong — measures geometric expansion, not constraint stretch)
2. Then computed proper joint attachment points using quaternion-rotated frame offsets (still wrong by same factor — the gap is still solver residual, not physical stretch)

**Conclusion:** You cannot extract physically meaningful internal forces from VBD constraint residuals. This is a fundamental property of position-based solvers, not a bug. For applications where internal forces matter (like Phase 2's tendon-driven finger), this is a real limitation of VBD worth reporting.

**What we CAN validate:** The kinematic method confirms that Newton's VBD solver produces kinematically self-consistent results — the positions, velocities, and centripetal accelerations of the spinning chain are exactly what the analytic prediction says they should be (at the measured rotation rate). The solver gets the motion right even though it doesn't expose the forces.

### What this means for the writeup

This is actually a rich finding with three layers:
1. **Positive:** Newton's cable solver produces correct kinematics for a spinning closed chain (T_kin/T_ana = 1.000)
2. **Limitation:** VBD doesn't conserve energy (omega decays ~3%/s, independent of N or damping parameters)
3. **Limitation:** VBD doesn't expose physically meaningful internal forces (constraint residuals ≠ physical tension)

For Phase 2 (tendon-driven finger), limitation #3 means we may need to use SolverSemiImplicit instead of VBD if we want to measure tendon tension accurately. Or we use the kinematic approach.

### N-Sweep Completed — Headline Figures Generated

Ran `sweep.py` headlessly for N = 4, 8, 16, 32, 64, 128 at 20s each. Produced 5 figures in `figures/` and raw JSON data in `data/`.

**Summary of sweep results:**

| N | ω retention at t=1s | ω at t=20s | r_std range | Shape stability |
|---|---|---|---|---|
| 4 | 67% | 2.55 | < 0.001 | Perfect circle |
| 8 | 87% | 2.94 | < 0.001 | Perfect circle |
| 16 | 92% | 3.25 | < 0.001 | Perfect circle |
| 32 | 93% | 3.30 | < 0.002 | Nearly perfect |
| 64 | 92% | 3.32 | 0.005-0.050 | Oscillating (modes excited) |
| 128 | 91% | 3.25 | 0.005-0.060 | Oscillating (more modes) |

**Key figure descriptions (for writeup):**

- **Panel (a) — Omega decay:** All N values follow nearly parallel decay curves after the initial drop. Confirms VBD dissipation is solver-intrinsic.
- **Panel (b) — Discrete→continuous transition:** Clean sigmoid from 67% (N=4) to 93% (N=32), then slight dip at N=64,128 due to shape instability consuming energy. Transition complete by N~16.
- **Panel (c) — Shape stability:** Low N (4-32) are perfectly circular. High N (64, 128) develop growing oscillations in r_std — more degrees of freedom means more modes for numerical noise to excite.
- **Panel (d) — Tension validation:** T_kin/T_ana = 1.000 flat across all N and time. Perfect kinematic self-consistency, with noise at high N from shape oscillations.

### Solver comparison: Can we fix the angular momentum decay?

Tested all Newton solvers that could plausibly handle cable joints:

| Solver | omega at t=1s | omega at t=5s | r_mean | Cable joints? |
|---|---|---|---|---|
| **VBD** | 5.76 (92%) | 4.78 (76%) | 1.002 stable | Full support |
| **XPBD** | 5.79 (92%) | 4.80 (76%) | 1.000 stable | "joint type not handled" warnings — likely falls back to VBD |
| **SemiImplicit** | 0.99 (16%) | 0.20 (3%) | 6.35→31.8 (explodes) | Treats stretch as springs — can't hold stiff chains |
| **Featherstone** | — | — | — | Hangs on closed loops (tree-only algorithm) |

**Key insight:** The angular momentum decay is NOT just an energy issue. Energy dissipation is expected from position-based solvers. But angular momentum should be conserved even with dissipation (there are no external torques). The VBD solver violates this because constraint projections are not derived from a Hamiltonian — they move positions in directions that have small tangential components opposing the rotation. Over many timesteps, this accumulates into a systematic torque.

**None of Newton's solvers can simultaneously handle stiff cable constraints AND conserve angular momentum.** SemiImplicit is force-based (would conserve L) but can't handle high stretch stiffness without prohibitively small timesteps. VBD/XPBD handle stiffness but don't conserve L. Featherstone requires tree topology (no closed loops).

**For the writeup:** This is a genuine limitation worth reporting. A symplectic integrator with Lagrange-multiplier constraints (like SHAKE/RATTLE algorithms used in molecular dynamics) would conserve both energy and angular momentum while handling inextensibility constraints. Newton doesn't offer this, and naming it as a gap is exactly the kind of constructive feedback Frey would value.

### What remains for Phase 1

1. ~~One clean perturbation experiment~~ **Done** — perturbation.py ran, mode-2 oscillation visible
2. Minor figure polish if needed

---

## Day 2 continued — Phase 2: Differentiable Cable Shape Control

### What we built

`cable_shape_opt.py` — a hanging cable with per-particle force actuators, optimized via backpropagation through Newton's differentiable physics simulation.

**Architecture:**
- 12 particles, 11 free (one kinematic anchor at top)
- Springs: ke=500 stretch, ke=150 bending — stable with SemiImplicit
- Gravity enabled — cable hangs naturally, gravity provides restoring force
- Policy: 11 optimizable scalar forces (one lateral force per free particle)
- Loss: sum of squared distances from each particle to target positions
- Optimizer: Adam (lr=0.5) via warp.optim.Adam
- Training: wp.Tape() wraps 480 physics steps (60 frames × 8 substeps), backpropagates through the entire simulation to get exact ∂loss/∂forces

**This is differentiable RL:** instead of running thousands of noisy episodes to estimate policy gradients (PPO, SAC), we compute exact gradients by backpropagating through the physics. This is only possible with a differentiable simulator.

### Results

| Target | Initial Loss | Final Loss | Improvement | Iterations |
|--------|-------------|------------|-------------|------------|
| mode1 (bow) | 0.230 | 0.056 | 4.1x | 300 |
| mode2 (S-curve) | 0.230 | 0.031 | 7.4x | 300 |

**Key observations:**
- Gradients flow cleanly through 480 physics timesteps — no vanishing/exploding gradients
- Mode-2 converges better than mode-1 (7.4x vs 4.1x) — the S-curve shape requires less force because it's more compatible with the cable's natural bending modes
- The optimizer discovers the physically correct force distribution: positive forces on one side, negative on the other for mode-2
- Convergence is smooth on a log scale — Adam handles the loss landscape well
- Total runtime: ~60 seconds for 300 iterations on CPU (Apple Silicon)

### Connection to industry

Samsung is using Newton's VBD cable solver for cable manipulation in refrigerator assembly lines (announced GTC 2026). Our demo is the same problem class — learning to shape a deformable 1D object — solved via the differentiable physics path that Newton uniquely provides.

### Figures generated

- `figures/cable_opt_convergence_mode1.png` — loss vs iteration (log scale)
- `figures/cable_opt_shapes_mode1.png` — 3-panel: natural hang / target / optimized
- `figures/cable_opt_forces_mode1.png` — bar chart of learned force distribution
- `figures/cable_opt_modes_mode1.png` — DFT mode decomposition
- Same set for mode2

### Technical findings about Newton's differentiable simulation

1. **SemiImplicit solver is the only differentiable path.** VBD handles cables better (supports rigid rods, near-inextensible constraints) but doesn't support wp.Tape(). SemiImplicit is force-based and fully differentiable but requires softer spring constants.

2. **Spring stiffness must be moderate (~500) for SemiImplicit stability.** High stiffness (1e9) that works with VBD causes SemiImplicit to explode. This means the differentiable cable is stretchier than the VBD cable — a real trade-off.

3. **Gravity provides essential stability for open chains.** Zero-G cables with continuous applied forces accumulate unbounded displacement. Gravity acts as a natural restoring force, making the optimization landscape well-conditioned.

4. **Per-particle forces give better gradient flow than single-point actuation.** Each particle has a direct differentiable path from its force to the loss — no need for gradients to propagate through many spring connections.

---

## Roadmap Progress

| Roadmap Step | Status | Notes |
|---|---|---|
| Day 1: Install, explore, scope | **Done** | Go/no-go questions all answered positively |
| Day 2: Modify closest example | **Done** | Annotated cable_bend.py, understood API |
| Day 3: Closed loop, parametrized | **Done** | spinning_chain.py works, N is parametric, instrumentation added |
| Day 3: Test N=4,8,16,32 | **Done** | Tested N=4, 16, 64. Key findings on energy decay and stability. |
| Day 4: Validate T=mu*omega^2*r^2 | **Done** | Kinematic method: perfect (1.000). Stretch method: fails (VBD limitation). |
| Day 4: N sweep (2→128) | **Done** | sweep.py: N=4,8,16,32,64,128. Headline figures in figures/ |
| Day 5: Stability observation + figures | **Done** | Perturbation experiment + mode-2 oscillation observed |
| Phase 2: Differentiable cable control | **Done** | cable_shape_opt.py: mode1 (4.1x) and mode2 (7.4x) optimization |
| Solver comparison | **Done** | VBD vs XPBD vs SemiImplicit vs Featherstone tested |
| Writeup | **Not started** | All data and figures collected |

---

## Roadmap Progress

| Roadmap Step | Status | Notes |
|---|---|---|
| Day 1: Install, explore, scope | **Done** | Go/no-go questions all answered positively |
| Day 2: Modify closest example | **Done** | Annotated cable_bend.py, understood API |
| Day 3: Closed loop, parametrized | **Done** | spinning_chain.py works, N is parametric, instrumentation added |
| Day 3: Test N=4,8,16,32 | **Done** | Tested N=4, 16, 64. Key findings on energy decay and stability. |
| Day 4: Validate T=mu*omega^2*r^2 | **Done** | Kinematic method: perfect (1.000). Stretch method: fails (VBD limitation). |
| Day 4: N sweep (2→128) | **Done** | sweep.py: N=4,8,16,32,64,128. Headline figures in figures/ |
| Day 5: Stability observation + figures | Partially done | Out-of-plane modes observed. Mode-2 perturbation experiment remaining. |
