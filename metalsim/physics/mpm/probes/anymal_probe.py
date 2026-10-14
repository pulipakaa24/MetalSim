"""ANYmal on sand (example_mpm_anymal, the example's own capture: robot graph + eager sand step) split into its parts,
under solver / launch variants, with per-frame state hashes so variants can be checked bitwise against each other.

Per frame (after ``--warmup``): robot part (``apply_control`` + the captured robot graph) and sand part
(``simulate_sand``: the MPM step, chunk graph + GPU exit as in the timing runs), each between two device
synchronizations, plus a SHA-1 of particle_q / body_q.

Variants:
  --solver gs|gs-soa|...          SolverImplicitMPM.Config.solver (the example uses "auto" = "gs")
  --color-blocks-per-core K       Newton's colored Gauss-Seidel launches K threadgroups per GPU core (fork: 2);
                                  colour blocks are independent, so K changes scheduling, not results

    PYTHONPATH=<warp>:<newton>:<metalsim> python anymal_probe.py --frames 60 --json out.json
"""
import argparse
import hashlib
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="metal:0")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--solver", default=None)
ap.add_argument("--color-blocks-per-core", type=int, default=None)
ap.add_argument("--json")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
from metalsim.physics.mpm import newton_compat, scenes

newton_compat.install(accumulate="float32", loop="cuda", log=False, chunk_graph=True, gpu_exit=True)
from newton._src.solvers.implicit_mpm import solve_rheology as sr  # noqa: E402
from newton.solvers import SolverImplicitMPM  # noqa: E402

if args.color_blocks_per_core:
    k = args.color_blocks_per_core
    sr._color_block_count = lambda device: device.sm_count * k if (device.is_cuda or getattr(device, "is_metal", False)) else 1
if args.solver:
    _init = SolverImplicitMPM.__init__

    def init(self, model, *a, config=None, **kw):
        if config is not None:
            config.solver = args.solver
        _init(self, model, *a, config=config, **kw)

    SolverImplicitMPM.__init__ = init

with wp.ScopedDevice(args.device):
    sc = scenes.make_scene("example:mpm_anymal", capture=True)
    ex = sc.example
    print("solver variants:", getattr(ex.mpm_solver, "solver", None) if hasattr(ex.mpm_solver, "solver") else "?",
          "| robot graph:", ex.graph is not None, "| sand graph:", ex.sand_graph is not None, flush=True)
    for _ in range(args.warmup):
        sc.step()
    rob, sand, hashes = [], [], []
    for f in range(args.frames):
        wp.synchronize_device()
        t0 = time.perf_counter()
        ex.apply_control()
        if ex.graph:
            wp.capture_launch(ex.graph)
        else:
            ex.simulate_robot()
        wp.synchronize_device()
        t1 = time.perf_counter()
        ex.simulate_sand()
        wp.synchronize_device()
        t2 = time.perf_counter()
        ex.sim_time += ex.frame_dt
        rob.append((t1 - t0) * 1e3)
        sand.append((t2 - t1) * 1e3)
        h = hashlib.sha1()
        for arr in (ex.state_0.particle_q, ex.state_0.body_q):
            if arr is not None:
                h.update(np.ascontiguousarray(arr.numpy()).tobytes())
        hashes.append(h.hexdigest())
res = {"solver": args.solver, "color_blocks_per_core": args.color_blocks_per_core, "frames": args.frames,
       "robot_ms_median": float(np.median(rob)), "sand_ms_median": float(np.median(sand)),
       "total_ms_median": float(np.median(np.array(rob) + np.array(sand))), "robot_ms": rob, "sand_ms": sand,
       "hashes": hashes}
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}))
if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    json.dump(res, open(args.json, "w"), indent=1)
