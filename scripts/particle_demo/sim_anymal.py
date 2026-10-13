"""Record Newton 1.5.2's example_mpm_anymal (ANYmal C + walking policy on implicit-MPM sand) for the particle
renderer demo: per simulated frame, the sand particle positions and the robot's body poses.

Runs the example's own model, solvers and policy (newton.examples.mpm.example_mpm_anymal) without a viewer, with two
scene edits made from outside the example: a larger sand bed (``--bed``) so the robot walks on sand for the whole clip,
and a constant forward command (``--speed``) after ``--settle`` seconds standing still.

    .venv-newton152/bin/python scripts/particle_demo/sim_anymal.py --out runs/demo/anymal_sim --seconds 9

Output: ``particles_%05d.npy`` (P, 3) float32 [m], ``bodies.npz`` (body_q (F, B, 7) [px py pz qx qy qz qw],
joint_q (F, J), labels), ``meta.json`` (frame dt, bed, command, particle radius, timings).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seconds", type=float, default=9.0)
ap.add_argument("--settle", type=float, default=0.6, help="seconds with a zero command before walking")
ap.add_argument("--speed", type=float, default=0.4, help="forward command [m/s] (the policy's lin_vel_x)")
ap.add_argument("--yaw-rate", type=float, default=0.0, help="yaw command [rad/s]")
ap.add_argument("--bed", type=float, nargs=6, default=[-0.6, -0.6, 0.0, 0.6, 4.0, 0.10],
                help="sand bed lower and upper corner [m] (the example's own: -0.5 -0.5 0 0.5 2.5 0.15)")
ap.add_argument("--voxel-size", type=float, default=0.03)
ap.add_argument("--particles-per-cell", type=float, default=3.0)
ap.add_argument("--device", default="metal:0")
a = ap.parse_args()

import warp as wp

wp.config.quiet = True
wp.init()
from metalsim.physics.mpm import newton_compat

newton_compat.install(log=False)
import newton.examples.mpm.example_mpm_anymal as ex_mod
from newton.viewer import ViewerNull

lo, hi = np.array(a.bed[:3]), np.array(a.bed[3:])
_spawn = ex_mod._spawn_particles


def spawn_bed(builder, res, bounds_lo, bounds_hi, density):
    res = np.array(np.ceil(a.particles_per_cell * (hi - lo) / a.voxel_size), dtype=int)
    return _spawn(builder, res, lo, hi, density)


ex_mod._spawn_particles = spawn_bed
ex_mod.Example.capture = lambda self: (setattr(self, "graph", None), setattr(self, "sand_graph", None))
parser = ex_mod.Example.create_parser()
args = parser.parse_args([])
args.voxel_size, args.particles_per_cell = a.voxel_size, a.particles_per_cell
with wp.ScopedDevice(a.device):
    ex = ex_mod.Example(ViewerNull(num_frames=1 << 30), args)
os.makedirs(a.out, exist_ok=True)
model = ex.model
frames = int(round(a.seconds / ex.frame_dt))
labels = list(model.body_label)
body_q, joint_q, t_step = [], [], []
print(f"[sim] particles {model.particle_count}, bodies {model.body_count}, frames {frames} at dt {ex.frame_dt}", flush=True)


def save(f):
    np.save(os.path.join(a.out, f"particles_{f:05d}.npy"), ex.state_0.particle_q.numpy().astype(np.float32))
    body_q.append(ex.state_0.body_q.numpy().astype(np.float32).copy())
    joint_q.append(ex.state_0.joint_q.numpy().astype(np.float32).copy())


save(0)
t0 = time.time()
for f in range(1, frames + 1):
    walking = f * ex.frame_dt > a.settle
    ex._command = wp.vec3(a.speed if walking else 0.0, 0.0, a.yaw_rate if walking else 0.0)
    ts = time.time()
    with wp.ScopedDevice(a.device):
        ex.apply_control()
        ex.simulate_robot()
        ex.simulate_sand()
    wp.synchronize_device(a.device)
    ex.sim_time += ex.frame_dt
    t_step.append(time.time() - ts)
    save(f)
    if f % 25 == 0:
        bq = body_q[-1][0]
        print(f"[sim] frame {f}/{frames} t={ex.sim_time:.2f}s base=({bq[0]:.2f},{bq[1]:.2f},{bq[2]:.2f}) "
              f"step {np.mean(t_step[-25:]) * 1e3:.0f} ms", flush=True)
    if not np.isfinite(body_q[-1]).all():
        sys.exit(f"non-finite body state at frame {f}")
np.savez(os.path.join(a.out, "bodies.npz"), body_q=np.stack(body_q), joint_q=np.stack(joint_q), labels=np.array(labels))
meta = dict(frame_dt=ex.frame_dt, frames=frames + 1, particle_count=int(model.particle_count), bed=a.bed, speed=a.speed,
            yaw_rate=a.yaw_rate, settle=a.settle, voxel_size=a.voxel_size, particles_per_cell=a.particles_per_cell,
            particle_radius=float(ex.mpm_solver._mpm_model.particle_radius.numpy().mean()),
            step_ms_mean=float(np.mean(t_step) * 1e3), step_ms_median=float(np.median(t_step) * 1e3),
            wall_s=time.time() - t0, device=a.device, labels=labels,
            source="newton 1.5.2 examples/mpm/example_mpm_anymal.py (policy anymal_walking_policy_physx.onnx)")
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print(f"[sim] done: {json.dumps({k: meta[k] for k in ('particle_count', 'step_ms_mean', 'wall_s')})}", flush=True)
