"""Gauss-Seidel colour passes of Newton's strain solve on Metal: structure and in-replay cost (research_notes/gpu_latency).

Hooks one strain solve after warm-up (like gs_pass.py) and records:
  * colours; per colour: colour blocks, strain nodes, nodes per block (min / median / max), the launch's thread count
    (threads with a block to do vs threads launched);
  * microseconds per colour pass in a 40-sweep graph replay (sustained clocks);
  * the replay cost of an empty one-thread dispatch and of an empty dispatch of the pass's width (the dispatch floor),
    so chain latency per pass = pass - floor.
Mutates the solve it measures (timing only).

    PYTHONPATH=<warp>:<newton>:<metalsim> python gs_structure_probe.py example:mpm_anymal --json out.json
"""
import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--seconds", type=float, default=1.5)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--json")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
from metalsim.physics.mpm import newton_compat, scenes
from newton._src.solvers.implicit_mpm import solve_rheology as sr

result = {}


@wp.kernel
def _empty(a: wp.array[float]):
    i = wp.tid()
    if i < 0:
        a[0] = 1.0


def replay_us(fn, per_unit, reset=None):
    """Median us per unit over --seconds of replays of a 40-call graph of fn."""
    if reset:
        reset()
    fn()
    wp.synchronize_device()
    k = 40
    with wp.ScopedCapture(force_module_load=False) as cap:
        for _ in range(k):
            fn()
    per = []
    t_end = time.perf_counter() + args.seconds
    while time.perf_counter() < t_end:
        if reset:
            reset()
        wp.synchronize_device()
        t0 = time.perf_counter()
        wp.capture_launch(cap.graph)
        wp.synchronize_device()
        per.append((time.perf_counter() - t0) * 1e6 / (k * per_unit))
    return float(np.median(per[len(per) // 2:]))


def probe(rs, cs):
    rh = rs.rheology
    offs = rh.color_offsets.numpy()
    ncol = len(offs) - 1
    nblk = int(offs[-1])
    blocks = rh.color_blocks.numpy()[:, :nblk]
    lens = blocks[1] - blocks[0]
    launch_threads = sr._color_block_count(wp.get_device()) * 64
    per_color = []
    for c in range(ncol):
        L = lens[offs[c]:offs[c + 1]]
        per_color.append({"blocks": int(L.size), "nodes": int(L.sum()),
                          "nodes_per_block_min": int(L.min()) if L.size else 0,
                          "nodes_per_block_median": float(np.median(L)) if L.size else 0.0,
                          "nodes_per_block_max": int(L.max()) if L.size else 0,
                          "threads_busy": int(min(L.size, launch_threads))})
    result.update(colors=ncol, launch_threads=launch_threads, strain_nodes=int(lens.sum()), per_color=per_color)
    vel0 = wp.clone(rs.momentum.velocity)
    stress0 = wp.clone(rh.stress)

    def reset():
        wp.copy(rs.momentum.velocity, vel0)
        wp.copy(rh.stress, stress0)

    replay_us(rs.solve, ncol, reset)  # warm
    result["us_per_color_pass"] = replay_us(rs.solve, ncol, reset)
    result["us_contact_solve"] = replay_us(cs.solve, 1, reset)
    a = wp.zeros(1, dtype=float)
    result["us_empty_dispatch_1_thread"] = replay_us(lambda: wp.launch(_empty, dim=1, inputs=[a]), 1)
    result["us_empty_dispatch_pass_width"] = replay_us(lambda: wp.launch(_empty, dim=launch_threads, inputs=[a]), 1)
    result["us_chain_per_pass"] = result["us_per_color_pass"] - result["us_empty_dispatch_pass_width"]
    reset()


def hook(rheology_solver, contact_solver, *rest):
    if result.get("armed") and "us_per_color_pass" not in result and rheology_solver.name == "Gauss-Seidel":
        probe(rheology_solver, contact_solver)
    return cuda_loop(rheology_solver, contact_solver, *rest)


newton_compat.install(accumulate="float32", loop="cuda", log=False)
cuda_loop = sr._run_solver_loop
sr._run_solver_loop = hook
kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    for _ in range(args.warmup):
        sc.step()
    result["armed"] = True
    sc.step()
result.pop("armed", None)
result["scene"] = args.scene
print(json.dumps({k: v for k, v in result.items() if k != "per_color"}))
for c, pc in enumerate(result.get("per_color", [])):
    print(f"colour {c}: {pc}")
if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    json.dump(result, open(args.json, "w"), indent=1)
