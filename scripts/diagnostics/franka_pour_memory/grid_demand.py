"""Peak NanoVDB node demand of Franka-Pour's rebuildable MPM grid over a rollout with random actions (resets
included), against the capacities the solver reserved.   grid_demand.py NUM_ENVS POLICY_STEPS [EVERY] [JSON]"""
import json
import sys
import time

import numpy as np
import torch
import warp as wp

N = int(sys.argv[1]); STEPS = int(sys.argv[2]); EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 10
OUT = sys.argv[4] if len(sys.argv) > 4 else None
wp.config.quiet = True
from metalsim.learn import franka_pour as P  # noqa: E402

pcfg = P.S.PourPhysicsConfig(num_envs=N)
env = P.FrankaPourEnv(P.PourTaskConfig(num_envs=N, seed=0, physics=pcfg))
env.reset_all()
mpm = env.physics.mpm
g = torch.Generator(device=env.device).manual_seed(0)


def counts(vol):
    ijk = vol.get_voxels().numpy()
    out = {"slots": int(ijk.shape[0])}
    for name, d in (("voxels", 1), ("leaf", 8), ("lower", 128), ("upper", 4096)):
        out[name] = int(np.unique(np.floor_divide(ijk, d), axis=0).shape[0])
    return out


peak = {}
t0 = time.perf_counter()
samples = []
pipe = env.physics.pipeline
np_ = pipe.narrow_phase
coll_counters = {"rigid_contacts": env.physics.contacts.rigid_contact_count, "broad_phase_pairs": pipe.broad_phase_pair_count,
                 "triangle_pairs": getattr(np_, "triangle_pairs_count", None),
                 "gjk_candidate_pairs": getattr(np_, "gjk_candidate_pairs_count", None),
                 "mesh_shape_pairs": getattr(np_, "shape_pairs_mesh_count", None)}
coll_caps = {"rigid_contact_max": pipe.rigid_contact_max, "shape_pairs_max": getattr(pipe, "shape_pairs_max", None),
             "max_triangle_pairs": int(np_.triangle_pairs.shape[0]) if getattr(np_, "triangle_pairs", None) is not None else None}
coll_peak = {}
for k in range(STEPS):
    a = torch.rand(env.num_envs, 8, device=env.device, generator=g) * 2 - 1
    env.step(a)
    for kk, arr in coll_counters.items():
        if arr is not None:
            coll_peak[kk] = max(coll_peak.get(kk, 0), int(arr.numpy()[0]))
    if k % EVERY == 0 or k == STEPS - 1:
        wp.synchronize_device()
        grid = mpm._scratchpad.grid
        rec = {"step": k}
        for nm in ("cell_grid", "vertex_grid"):
            v = getattr(grid, nm, None)
            if v is not None:
                c = counts(v)
                rec[nm] = c
                p = peak.setdefault(nm, {})
                for kk, vv in c.items():
                    p[kk] = max(p.get(kk, 0), vv)
        samples.append(rec)
        print(k, rec, flush=True)
mpm.check_sparse_grid_rebuild_status()
res = {"num_envs": N, "policy_steps": STEPS, "peak": peak, "collision_peak": coll_peak, "collision_caps": coll_caps, "samples": samples, "seconds": time.perf_counter() - t0}
print("PEAK", json.dumps(peak))
print("COLLISION peak", json.dumps(coll_peak), "caps", json.dumps(coll_caps))
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
