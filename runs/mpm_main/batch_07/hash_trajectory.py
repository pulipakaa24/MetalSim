"""Per-step hashes of the coupled box-on-sand scene (scale_coupled.py's) with the MetalSim tree given by --root, for a
bitwise comparison of two trees (main vs the coupling-opt branch). Resets a world every 10 steps through the public
``mpm.reset(mask)`` (host mask), which both trees have.

    python -P hash_trajectory.py --root TREE --worlds 8 --steps 60 --json out.json
"""
import argparse
import hashlib
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--worlds", type=int, default=8)
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--json")
a = ap.parse_args()
sys.path.insert(0, a.root)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import metalsim  # noqa: E402
from metalsim.physics.batch import BatchSim, BatchSimOptions  # noqa: E402
from metalsim.physics.mpm.coupled import SAND, GridEmitter, MPMConfig, MPMCoupling  # noqa: E402
from metalsim.physics.mpm.probes.scale_coupled import XML  # noqa: E402

assert metalsim.__file__.startswith(a.root), metalsim.__file__
wp.config.quiet = True
with wp.ScopedDevice("metal:0"):
    model = mujoco.MjModel.from_xml_string(XML.format(dt=1.0 / 120.0 / 3))
    sim = BatchSim(model, a.worlds, options=BatchSimOptions(substeps=3))
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=0.02, max_iterations=50, max_active_cell_count=512 * a.worlds))
    mpm.add_emitter(GridEmitter(lower=(-0.06, -0.06, 0.0), upper=(0.06, 0.06, 0.05), material=SAND))
    mpm.finalize()
    hashes = []
    for k in range(a.steps):
        if k and k % 10 == 0:
            m = np.zeros(a.worlds, bool)
            m[(k // 10) % a.worlds] = True
            mpm.reset(m)
        mpm.step()
        h = hashlib.sha256()
        for arr in (sim.d.qpos, sim.d.qvel, mpm.coupling_f, mpm.particle_q, mpm.particle_qd):
            h.update(arr.numpy().tobytes())
        hashes.append(h.hexdigest()[:16])
res = {"root": a.root, "metalsim": metalsim.__file__, "hashes": hashes}
print(json.dumps({"root": a.root, "last": hashes[-1]}))
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
