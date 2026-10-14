"""Smoke test of metalsim.physics.mpm.coupled: a free box dropped onto a sand bed, N worlds, BatchSim + MPM.

    PYTHONPATH=<worktree> .venv/bin/python smoke_box.py [--worlds 2] [--steps 60] [--capture 1] [--mass effective]
"""
import argparse
import time

import mujoco
import numpy as np
import warp as wp

wp.config.quiet = True

from metalsim.physics.batch import BatchSim, BatchSimOptions  # noqa: E402
from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling, SAND  # noqa: E402

XML = """<mujoco><option timestep="0.0025"/>
<worldbody><geom name="floor" type="plane" size="0 0 1" friction="0.5"/>
<body name="box" pos="0 0 {z}"><freejoint/><geom name="box" type="box" size="{h} {h} {h}" mass="{m}"/></body>
</worldbody></mujoco>"""

ap = argparse.ArgumentParser()
ap.add_argument("--worlds", type=int, default=2)
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--substeps", type=int, default=4)
ap.add_argument("--capture", type=int, default=1)
ap.add_argument("--mass", default="effective")
ap.add_argument("--mode", default="staggered")
ap.add_argument("--voxel", type=float, default=0.02)
a = ap.parse_args()

model = mujoco.MjModel.from_xml_string(XML.format(z=0.25, h=0.05, m=2.0))
sim = BatchSim(model, a.worlds, options=BatchSimOptions(substeps=a.substeps))
cfg = MPMConfig(voxel_size=a.voxel, proxy_mass=a.mass, mode=a.mode, capture=bool(a.capture), max_iterations=50)
mpm = MPMCoupling(sim, cfg)
mpm.add_emitter(GridEmitter(lower=(-0.15, -0.15, 0.0), upper=(0.15, 0.15, 0.08), material=SAND))
t0 = time.time()
mpm.finalize()
print(f"finalize {time.time() - t0:.1f} s: particles/world {mpm.particles_per_world}, proxies {mpm.nb}, "
      f"proxy mass {mpm.proxy_mass}, capturable {mpm.capturable}, dt {mpm.dt}")
for s in range(a.steps):
    t0 = time.time()
    mpm.step()
    sim.synchronize()
    dt = time.time() - t0
    if s % 10 == 0 or s == a.steps - 1:
        z = sim.d.xpos.numpy()[:, 1, 2]
        f = mpm.wrenches()[:, 0, :3]
        q = mpm.particles()
        print(f"step {s:3d} {dt*1e3:7.1f} ms  box z {np.round(z, 4)}  sand Fz {np.round(f[:, 2], 2)}  "
              f"particle z [{q[..., 2].min():.4f}, {q[..., 2].max():.4f}] finite {np.isfinite(q).all()}")
mpm.check_status()
print("weight of box", 2.0 * 9.81, "N")
