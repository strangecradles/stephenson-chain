###########################################################################
# Demo GIF renderer
#
# Runs the mode-2 perturbed spinning chain headless (same setup as
# perturbation.py) and draws each frame with matplotlib instead of the
# OpenGL viewer, so it works anywhere — no GPU, no window. Output goes
# to figures/demo.gif for the README.
#
# Usage: python3 src/phase1/render_demo.py
###########################################################################

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perturbation import PerturbedChain


class DummyViewer:
    def set_model(self, m): pass
    def apply_forces(self, s): pass
    def begin_frame(self, t): pass
    def log_state(self, s): pass
    def log_contacts(self, c, s): pass
    def end_frame(self): pass


def run_and_capture(sim_seconds=10.0, capture_every=5):
    """Run the sim headless, grab XY positions of every link every few frames."""
    example = PerturbedChain(
        DummyViewer(), None,
        num_links=32,
        perturbation_mode=2,
        perturbation_amplitude=0.12,  # bigger than the experiment's 0.05 so the wobble reads on video
    )
    frames = []
    times = []
    num_frames = int(sim_seconds * example.fps)
    for frame in range(num_frames):
        example.step()
        if frame % capture_every == 0:
            body_q = example.state_0.body_q.numpy()
            frames.append(body_q[example.chain_bodies, :2].copy())
            times.append(example.sim_time)
    return frames, times


def render_gif(frames, times, out_path, fps=12):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=80)
    fig.patch.set_facecolor("#101018")
    ax.set_facecolor("#101018")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")

    # closed loop: repeat the first point at the end
    def closed(pts):
        return np.vstack([pts, pts[:1]])

    pts0 = closed(frames[0])
    (line,) = ax.plot(pts0[:, 0], pts0[:, 1], color="#7fd4ff", linewidth=1.2, alpha=0.8)
    scat = ax.scatter(frames[0][:, 0], frames[0][:, 1], s=14, color="#7fd4ff", zorder=3)
    # one link painted differently so the rotation itself is visible
    marker = ax.scatter(frames[0][:1, 0], frames[0][:1, 1], s=42, color="#ff9d5c", zorder=4)
    label = ax.text(0.03, 0.95, "", transform=ax.transAxes, color="#d0d0e0",
                    fontsize=9, family="monospace")

    def update(i):
        pts = closed(frames[i])
        line.set_data(pts[:, 0], pts[:, 1])
        scat.set_offsets(frames[i])
        marker.set_offsets(frames[i][:1])
        label.set_text(f"t = {times[i]:4.1f} s   N = 32   zero-g")
        return line, scat, marker, label

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    print("Running headless sim for the demo gif...")
    frames, times = run_and_capture()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "figures", "demo.gif")
    out = os.path.normpath(out)
    render_gif(frames, times, out)
    size_mb = os.path.getsize(out) / 1e6
    print(f"Saved {out} ({size_mb:.2f} MB, {len(frames)} frames)")
