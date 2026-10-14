"""Per-step state hashes of the MPM-main scene ``metalsim/physics/mpm/probes/multibody_sand.py`` (MPMCoupling: BatchSim
boxes on MPM sand, lagged proxy-wrench exchange), for repeatability and A/B checks of the coupling harvest.

    PYTHONPATH=<warp>:<metalsim tree with that probe> python mainsim_hash.py --bodies 16 --steps 30 --json out.json
"""
import argparse
import hashlib
import json

import numpy as np
import warp as wp

ap = argparse.ArgumentParser()
ap.add_argument("--bodies", type=int, default=16)
ap.add_argument("--worlds", type=int, default=1)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--json")
args = ap.parse_args()
wp.config.quiet = True
from metalsim.physics.mpm.probes.multibody_sand import build  # noqa: E402

with wp.ScopedDevice("metal:0"):
    sim, mpm = build(args.bodies, args.worlds, capture=True)
    hashes = []
    for _ in range(args.steps):
        mpm.step()
        sim.synchronize()
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(mpm.particle_q.numpy()).tobytes())
        h.update(np.ascontiguousarray(sim.d.xpos.numpy()).tobytes())
        hashes.append(h.hexdigest())
    mpm.check_status()
res = {"bodies": args.bodies, "worlds": args.worlds, "steps": args.steps, "hashes": hashes,
       "box_z_mean": float(sim.d.xpos.numpy()[:, 1:, 2].mean())}
print(json.dumps({k: v for k, v in res.items() if k != "hashes"}))
if args.json:
    json.dump(res, open(args.json, "w"), indent=1)
