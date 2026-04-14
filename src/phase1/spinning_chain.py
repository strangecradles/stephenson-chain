###########################################################################
# Spinning Chain - Phase 1, Step 1
#
# A closed chain of N rigid links in zero gravity, given an initial
# angular velocity. This is the simples possible version: get a ring
# of capsule bodies spinning, and watch what happens! 
#
# What I aim to learn:
#   - how to construct a closed-loop cable using builder.add_rod(closed=True)
#   - how to set zero gravity
#   - how to give bodies initial velocities
#   - how to extract positions from the simulation state
###########################################################################

import math
import numpy as np
import warp as wp
import newton
import newton.examples

class SpinningChain:

    def __init__(self, viewer, args=None):
        # =====================================================================
        # SECTION 1: Timing parameters
        # =====================================================================
        # These control the relationship between display frames and physics steps
        # 
        # Physics simulators don't usually run at the same rate as your screen. Your screen
        # refreshes at 60fps (frame_dt = 1/60 ~= 16.7ms) , but the physics needs
        # smnaller timesteps to stay stable. So each display frame, we run multiple 
        # "substeps" of the physics simulation. 
        #
        # Think of it like numerical integration: smaller dt = more accurate but 
        # more expensive. the sim_dt here is ~1.67ms, which is typical for 
        # constraint-based rigid body simulation.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps          # seconds per display frame: ~16.7ms (from screen frame rate)
        self.sim_time = 0.0                  # running clock for the simulation
        self.sim_substeps = 10               # physics steps per display frame
        self.sim_iterations = 5              # solver iterations per substep (see below)
        self.sim_dt = self.frame_dt / self.sim_substeps     # physics timestep: ~1.67ms

        self.viewer = viewer
        self.args = args

        # =====================================================================
        # CHAIN PHYSICAL PARAMETERS
        # =====================================================================
        # Each chain is made of 16 connected links, connected into a loop 
        #
        #         

        self.num_links = 128             # number of rigid capsule segments
        self.chain_radius = 1.0         # radius of the ring [meters]
        self.link_radius = 0.02         # radius of each capsule cross-section [meters]
        self.omega = 2.0 * math.pi      # angular velocity [rad/s] - one full revolution per sec

        # =====================================================================
        # SECTION 3: Build the chain geometry
        # =====================================================================
        # need to place N+1 points equally spaced around a circle of radius 
        # chain_radius, lying in the XY plane (z=0). The Nth+1 point should be
        # the same as the 0th point (thus closing the loop), but add_roc(closed=True)
        # handles that - so you actually need exactly N+1 points where the last one 
        # coincides with the first. 
        points = []
        for i in range (self.num_links+ 1):
            theta = 2.0 * math.pi * i / self.num_links
            points.append(wp.vec3(self.chain_radius * math.cos(theta), self.chain_radius * math.sin(theta), 0.0))

        # =====================================================================
        # SECTION 4: ModelBuilder — create the scene
        # =====================================================================
        # ModelBuilder is Newton's scene-reconstruction API. You add bodies, joints,
        # shapes, and materials to it, then call finalize() to produce an immutable              
        # Model that the solver can simulate.
        #
        # I sort of think of it as writing a recipe (ModelBuilder) vs. cooking the dish (Solver)
        # The builder is purely declarative, so nothing is simulated yet. 

        builder = newton.ModelBuilder()

        # Default contact material properties. 
        # These control what happens when two bodies touch:
        #   ke: contact stiffness [N/m] - how hard the "spring" pushes back on penetration
        #   kd: contact damping [N*s/m] - how much energy is lost on contact (prevents bouncing)
        #   mu: friction coefficient [dimensionless] - Coulomb friction (0 = ice, 1 = rubber approx.)

        # Newton uses penalty-based contacts: when two shapes overlap, it applies a 
        # spring-like repulsion force proportional to penetration depth. This is simpler
        # than the "hard contact" approach (LCP) used by MuJoCu but works well for 
        # soft/cable scenarios

        builder.default_shape_cfg.ke = 1.0e2
        builder.default_shape_cfg.kd = 1.0e1
        builder.default_shape_cfg.mu = 1.0

        self.chain_bodies, self.chain_joints = builder.add_rod(
                positions=points,
                quaternions=None,
                radius=self.link_radius,
                bend_stiffness=1.0,
                bend_damping=0.0,
                stretch_stiffness=1.0e9,  # very high = nearly inextensible
                stretch_damping=0.0,
                label="chain",
                closed=True
            )

        # =====================================================================
        # SECTION 5: Zero gravity
        # =====================================================================
        # The cable bend example uses default gravity (-9.81 in z).
        # We want zero gravity for the spinning chain in space
         
        # builder.gravity = 0.0
        builder.add_ground_plane(-2)
        builder.color()
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations, friction_epsilon=0.1)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.control = self.model.control()
        self.contacts = self.model.contacts()

        body_q = self.state_0.body_q.numpy()       # shape [N_bodies, 7]
        body_qd = self.state_0.body_qd.numpy()      # shape [N_bodies, 6]
        
        for body_idx in self.chain_bodies:
            x, y = body_q[body_idx, 0], body_q[body_idx, 1]
            vx, vy = -self.omega * y, self.omega * x
            body_qd[body_idx] = [vx, vy, 0.0, 0.0, 0.0, self.omega]
        
        self.state_0.body_qd.assign(body_qd)
        print(f"Body 0 position: {body_q[self.chain_bodies[0], :3]}")
        print(f"Body 0 velocity: {body_qd[self.chain_bodies[0]]}")  

        # =====================================================================
        # SECTION 8: Instrumentation — data collection buffers
        # =====================================================================
        # To turn this from a visual demo into an experiment, we need to measure
        # things every frame and store the results. We'll track:
        #
        # 1. Centroid: the average position of all chain bodies.
        #    If the chain is stable and there's no gravity, the centroid should
        #    stay near the origin. If it drifts, something is wrong (numerical
        #    drift, or we accidentally gave the chain net linear momentum).
        #
        # 2. Mean radius: average distance from centroid to each body.
        #    For a perfect circle this equals chain_radius. If the chain is
        #    deforming, this will change.
        #
        # 3. Radius std dev: standard deviation of distances from centroid.
        #    This is our "circularity" metric. For a perfect circle, std = 0.
        #    Larger values mean the chain is distorted (elliptical, wavy, etc.)
        #    This is the number that will tell us whether the ring is stable.
        #
        # 4. Angular velocity: estimated from the average tangential velocity.
        #    We can compare this to self.omega to see if the chain is speeding
        #    up, slowing down, or staying constant.
        #
        # Each list stores one value per frame. After the simulation we can plot
        # these with matplotlib to see the full time history.
        self.history_time = []
        self.history_centroid_x = []
        self.history_centroid_y = []
        self.history_centroid_z = []
        self.history_mean_radius = []
        self.history_radius_std = []
        self.history_omega_measured = []

        # =====================================================================
        # SECTION 9: Tension validation setup
        # =====================================================================
        # We want to compare the tension in the chain to the analytic prediction
        # for a continuous spinning ring: T = μ * ω² * r²
        #
        # We'll compute tension two independent ways and compare both to theory:
        #
        # METHOD 1 — KINEMATIC (from centripetal balance):
        #   Each link moves in a circle. The centripetal force it needs is
        #   F = m * v² / r. That force is provided by the tensions at its two
        #   ends. For a link subtending angle dθ = 2π/N at the center, the
        #   net inward pull from two tensions T is: 2T sin(dθ/2) ≈ T * dθ.
        #   So: T = F / dθ = m * v² / (r * dθ) = m * v² * N / (2π * r)
        #
        # METHOD 2 — STRETCH (from link elongation):
        #   Newton's cable joints enforce stretch stiffness like a spring:
        #   if a link stretches from rest length L₀ to current length L,
        #   the tension is T = stretch_stiffness * (L - L₀) / L₀.
        #   This is Hooke's law in 1D: T = EA * strain, where EA = stretch_stiffness.
        #   We measure current inter-body distances and compare to rest length.
        #
        # ANALYTIC PREDICTION:
        #   T = μ * ω² * r²  where μ = M_total / (2πr) is mass per unit length.
        #   Equivalently: T = M_total * ω² * r / (2π)
        #   We use the MEASURED omega and radius (not targets) so we're comparing
        #   apples to apples — "what should the tension be at the current rotation
        #   rate" vs "what does the simulation produce."

        # Read body masses from the finalized model
        body_mass_all = self.model.body_mass.numpy()
        self.chain_mass = body_mass_all[self.chain_bodies]   # mass of each link
        self.total_mass = np.sum(self.chain_mass)
        self.link_mass = self.chain_mass[0]  # all links are identical

        # Rest length of each link: the chord length of the initial regular N-gon.
        # For N points equally spaced on a circle of radius r, consecutive points
        # are separated by: L = 2r sin(π/N). This is a chord, not an arc.
        # (Arc length would be 2πr/N. They converge as N → ∞.)
        self.rest_length = 2.0 * self.chain_radius * math.sin(math.pi / self.num_links)

        # Store stretch stiffness for Method 2
        self.stretch_stiffness = 1.0e9

        # Angle subtended by each link, used in Method 1
        self.dtheta = 2.0 * math.pi / self.num_links

        print(f"\n--- Tension validation setup ---")
        print(f"  N = {self.num_links} links")
        print(f"  link mass = {self.link_mass:.6f} kg")
        print(f"  total mass = {self.total_mass:.4f} kg")
        print(f"  rest length = {self.rest_length:.6f} m")
        print(f"  mu (mass/length) = {self.total_mass / (2 * math.pi * self.chain_radius):.4f} kg/m")
        print(f"  T_analytic at target omega: {self.total_mass * self.omega**2 * self.chain_radius / (2 * math.pi):.4f} N")
        print()

        self.history_tension_kinematic = []
        self.history_tension_stretch = []
        self.history_tension_analytic = []

        # How often to print a status line (every N frames).
        # At 60fps, every 60 frames = once per second of sim time.
        self.print_every = 60

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        """CUDA graph capture optimization (no-op on CPU).

        On GPU, this records all the kernel launches in simulate() into a replayable
        graph, avoiding the overhead of relaunching kernels each frame. On my mac
        where i only use the CPU, this sets self.graph = None and we fall through to direct
        execution. 
        """
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Run one display frame's worth of physics substeps.

        This is the core simulation loop. Each call advances the simulation by
        frame_dt (~16.7ms) using sim_substeps (10) smaller steps of sim_dt (~1.67ms).

        Within each substep:
        1. clear_forces() — reset accumulated forces to zero
        2. apply_forces() — add any external forces (e.g., user clicking in the viewer)
        3. collide() — detect all shape-shape contacts and fill the contacts buffer
        4. solver.step() — advance physics: resolve constraints, integrate velocities,
           update positions. Reads from state_0, writes to state_1.
        5. Swap state_0 and state_1 — the "new" state becomes the "current" state
           for the next substep.

        After all substeps, state_0 holds the result for this display frame.
        """
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # The viewer can apply forces when you click-drag on bodies.
            # This is purely interactive — no effect during headless runs.
            self.viewer.apply_forces(self.state_0)

            # Contact detection: finds all pairs of overlapping shapes and computes
            # penetration depth, normal direction, and contact points. These get
            # stored in self.contacts for the solver to use.
            self.model.collide(self.state_0, self.contacts)

            # THE ACTUAL PHYSICS STEP.
            # The solver reads positions/velocities from state_0, applies gravity,
            # resolves cable joints (stretch + bend constraints), resolves contacts,
            # and writes the updated positions/velocities into state_1.
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            # Double-buffer swap: state_1 (which the solver just wrote) becomes
            # state_0 (the current state) for the next substep.
            # This is a Python reference swap — no data is copied.
            self.state_0, self.state_1 = self.state_1, self.state_0

    def measure(self):
        """Extract physical measurements from the current simulation state.

        This is where simulation becomes experiment. We read the raw body
        transforms and velocities (numpy arrays on CPU), then compute
        physically meaningful quantities.

        Called once per display frame from step(), after the physics has advanced.
        """
        # Pull body state from Warp into numpy.
        # body_q shape: [N_bodies, 7] — columns 0:3 are position (x,y,z),
        #                                columns 3:7 are quaternion (qx,qy,qz,qw)
        # body_qd shape: [N_bodies, 6] — columns 0:3 are linear velocity (vx,vy,vz),
        #                                 columns 3:6 are angular velocity (wx,wy,wz)
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        # Extract just the positions of chain bodies (not the ground plane body).
        # self.chain_bodies is a list of integer indices into the body arrays.
        # "Fancy indexing" in numpy: body_q[list_of_indices] picks out those rows.
        chain_pos = body_q[self.chain_bodies, :3]   # shape [N_links, 3]
        chain_vel = body_qd[self.chain_bodies, :3]  # shape [N_links, 3] (linear vel)

        # -----------------------------------------------------------------
        # 1. CENTROID — average position of all chain bodies
        # -----------------------------------------------------------------
        # np.mean(array, axis=0) averages across rows, giving one [x, y, z].
        # If the chain has zero net linear momentum and no gravity, the centroid
        # should stay at the origin. Any drift means we accidentally gave the
        # chain a net push, or numerical errors are accumulating.
        centroid = np.mean(chain_pos, axis=0)  # shape [3]

        # -----------------------------------------------------------------
        # 2. RADII — distance from centroid to each body
        # -----------------------------------------------------------------
        # Subtract centroid from every body position, then compute the Euclidean
        # length of each difference vector. np.linalg.norm with axis=1 computes
        # one norm per row.
        #
        # For a perfect circle, every radius = chain_radius.
        # Deviations tell us the chain is distorted.
        offsets = chain_pos - centroid            # shape [N_links, 3]
        radii = np.linalg.norm(offsets, axis=1)   # shape [N_links]

        mean_radius = np.mean(radii)
        radius_std = np.std(radii)

        # -----------------------------------------------------------------
        # 3. ANGULAR VELOCITY — estimated from tangential speed
        # -----------------------------------------------------------------
        # For rigid rotation at angular velocity omega about the centroid,
        # every body has speed v = omega * r, so omega = v / r.
        #
        # We compute the speed of each body, divide by its radius, and average.
        # This gives us the "effective" angular velocity of the chain.
        #
        # If the chain is deforming rather than rigidly rotating, different
        # bodies will give different omega estimates — but the average is
        # still a useful single number.
        #
        # Guard against division by zero (if a body happens to be at the centroid).
        speeds = np.linalg.norm(chain_vel, axis=1)  # shape [N_links]
        safe_radii = np.where(radii > 1e-8, radii, 1e-8)  # avoid /0
        omega_per_body = speeds / safe_radii
        omega_measured = np.mean(omega_per_body)

        # -----------------------------------------------------------------
        # 4. TENSION — METHOD 1: Kinematic (centripetal balance)
        # -----------------------------------------------------------------
        # For each link i:
        #   - It has mass m_i, speed v_i, and sits at radius r_i from centroid
        #   - Centripetal force needed: F_i = m_i * v_i² / r_i
        #   - This force comes from the net inward pull of the two tensions
        #     at its ends: 2T sin(dθ/2) ≈ T * dθ where dθ = 2π/N
        #   - So: T_i = F_i / dθ = m_i * v_i² / (r_i * dθ)
        #
        # We compute T_i for each link and average. If the ring is uniformly
        # rotating, all T_i should be equal.
        T_per_link_kin = np.zeros(len(self.chain_bodies))
        for idx, body_idx in enumerate(self.chain_bodies):
            m = self.chain_mass[idx]
            v = speeds[idx]
            r = safe_radii[idx]
            T_per_link_kin[idx] = m * v * v / (r * self.dtheta)

        T_kinematic = np.mean(T_per_link_kin)

        # -----------------------------------------------------------------
        # 5. TENSION — METHOD 2: Stretch (DOES NOT WORK WITH VBD)
        # -----------------------------------------------------------------
        # IMPORTANT LESSON LEARNED:
        #
        # We tried to infer tension from the stretch of cable joints:
        #   T = stretch_stiffness * constraint_gap / rest_length
        #
        # This FAILS because VBD is a position-based solver, not a force-based
        # solver. A force-based solver (like MuJoCo) would compute "this joint
        # needs 44 N of tension" and apply that force explicitly. You could
        # read it directly.
        #
        # VBD does the opposite: it looks at constraint violations (gaps) and
        # directly nudges positions to reduce them, weighted by stiffness.
        # It never computes explicit forces. After a finite number of solver
        # iterations, there's always a residual gap — and that gap reflects
        # SOLVER ERROR (how well the iterative solver converged), not physical
        # stretch.
        #
        # With stretch_stiffness=1e9 and real tension ~44 N, the physical
        # stretch would be 44/1e9 ≈ 44 nanometers per link. But the VBD
        # residual gap is ~1 mm (solver convergence limit), giving a bogus
        # "tension" of ~2 million N — off by a factor of 50,000.
        #
        # CONCLUSION: You cannot extract physically meaningful internal forces
        # from VBD constraint residuals. The kinematic method (Method 1) works
        # because it infers tension from positions and velocities, which VBD
        # does get right. This is a real limitation of position-based solvers
        # worth reporting in the writeup.
        T_stretch = float('nan')  # intentionally invalid — method doesn't work with VBD

        # -----------------------------------------------------------------
        # 6. TENSION — ANALYTIC PREDICTION
        # -----------------------------------------------------------------
        # T = μ * ω² * r²  where μ = total_mass / circumference
        #
        # We use the MEASURED omega and radius, not the targets. This way we're
        # asking: "given the chain's actual rotation rate right now, what should
        # the tension be?" If the kinematic and stretch methods agree with this,
        # Newton's cable solver is force-accurate.
        mu = self.total_mass / (2.0 * math.pi * mean_radius)
        T_analytic = mu * omega_measured**2 * mean_radius**2

        # -----------------------------------------------------------------
        # 7. STORE everything in history lists
        # -----------------------------------------------------------------
        self.history_time.append(self.sim_time)
        self.history_centroid_x.append(centroid[0])
        self.history_centroid_y.append(centroid[1])
        self.history_centroid_z.append(centroid[2])
        self.history_mean_radius.append(mean_radius)
        self.history_radius_std.append(radius_std)
        self.history_omega_measured.append(omega_measured)
        self.history_tension_kinematic.append(T_kinematic)
        self.history_tension_stretch.append(T_stretch)
        self.history_tension_analytic.append(T_analytic)

        # -----------------------------------------------------------------
        # 8. PRINT a status line periodically
        # -----------------------------------------------------------------
        frame_num = len(self.history_time)
        if frame_num % self.print_every == 0:
            print(
                f"t={self.sim_time:6.2f}s  "
                f"omega={omega_measured:.3f}/{self.omega:.3f}  "
                f"r={mean_radius:.4f}  "
                f"T_kin={T_kinematic:.3f}  "
                f"T_ana={T_analytic:.3f}  "
                f"kin/ana={T_kinematic/T_analytic:.3f}"
                if T_analytic > 1e-6 else
                f"t={self.sim_time:6.2f}s  omega={omega_measured:.3f}  T_analytic≈0"
            )

    def step(self):
        """Called once per display frame by the run loop.

        Either replays the captured CUDA graph (GPU) or calls simulate() directly (CPU).
        Then advances the simulation clock and records measurements.
        """
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        # Record measurements AFTER physics has advanced for this frame
        self.measure()

    def render(self):
        """Called once per display frame to draw the current state.

        The viewer handles all the OpenGL rendering internally. We just need to
        tell it: "here's the current time, here's the state, here are the contacts,
        now draw."
        """
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):                                                                                                                                                  
        """Basic sanity check: no NaNs, chain hasn't exploded."""   
        if self.state_0.body_q is not None:                                                                                                                                
            body_positions = self.state_0.body_q.numpy()                                                                                                                   
            assert np.isfinite(body_positions).all(), "Non-finite values in body positions"                                                                                
            assert (np.abs(body_positions) < 1e3).all(), "Body positions too large" 

if __name__ == "__main__":
    # =========================================================================
    # ENTRY POINT
    # =========================================================================
    # newton.examples.init() handles command-line argument parsing and creates
    # the viewer window. It returns:
    #   viewer: the rendering window (ViewerGL for interactive, ViewerUSD for recording)
    #   args: parsed command-line arguments
    #
    # newton.examples.run() is the main loop: it calls example.step() then
    # example.render() at the target fps until you close the window.
    #
    # You can pass --headless to run without a window (useful for batch experiments),
    # or --num_frames N to run for exactly N frames then exit (useful for testing).
    viewer, args = newton.examples.init()

    example = SpinningChain(viewer, args)

    newton.examples.run(example, args)    




