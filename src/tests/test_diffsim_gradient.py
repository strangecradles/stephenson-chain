"""Minimal test: can we get gradients flowing through SemiImplicit?

Optimize an upward force on a particle to reach a target height.
If gradients flow, force_z should converge to mg = 9.81 (balance gravity).
"""
import warp as wp
import numpy as np
import newton

@wp.kernel
def apply_force(force_z: wp.array(dtype=float), f_ext: wp.array(dtype=wp.vec3)):
    f_ext[0] = wp.vec3(0.0, 0.0, force_z[0])

@wp.kernel
def compute_loss(pos: wp.array(dtype=wp.vec3), target_z: float, loss: wp.array(dtype=float)):
    loss[0] = (pos[0][2] - target_z) * (pos[0][2] - target_z)

builder = newton.ModelBuilder()
p = builder.add_particle(pos=(0, 0, 0.5), vel=(0, 0, 0), mass=1.0)
builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=5000, kf=0, kd=100, mu=0.5))
model = builder.finalize(requires_grad=True)
model.soft_contact_ke = 5000; model.soft_contact_kd = 100; model.soft_contact_mu = 0.5

solver = newton.solvers.SolverSemiImplicit(model)
pipeline = newton.CollisionPipeline(model, broad_phase="explicit",
    soft_contact_margin=1.0, requires_grad=True)
contacts = pipeline.contacts()

force_z = wp.array([5.0], dtype=float, requires_grad=True)
target_z = 0.5
loss = wp.zeros(1, dtype=float, requires_grad=True)

steps = 30; substeps = 8; dt = (1.0/60.0)/substeps
states = [model.state() for _ in range(steps*substeps+1)]
ctrl = model.control()

for opt_iter in range(15):
    tape = wp.Tape()
    with tape:
        for step in range(steps):
            for sub in range(substeps):
                t = step*substeps+sub
                states[t].clear_forces()
                wp.launch(apply_force, dim=1, inputs=[force_z, states[t].particle_f])
                pipeline.collide(states[t], contacts)
                solver.step(states[t], states[t+1], ctrl, contacts, dt)
        wp.launch(compute_loss, dim=1, inputs=[states[-1].particle_q, target_z, loss])
    tape.backward(loss)

    l = loss.numpy()[0]
    f = force_z.numpy()[0]
    g = force_z.grad.numpy()[0]
    z = states[-1].particle_q.numpy()[0][2]
    print(f"iter={opt_iter:2d}  loss={l:.6f}  force_z={f:.3f}  grad={g:.6f}  final_z={z:.4f}")

    force_z_np = force_z.numpy()
    force_z_np[0] -= 0.5 * g
    force_z.assign(force_z_np)
    tape.zero()
