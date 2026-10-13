"""us per Gauss-Seidel color pass of Newton's own strain solve on Metal, at sustained clocks (graph replay).

The cut-down form of gs_phases.py for A/B runs between Warp / Newton trees: hooks one strain solve after warm-up and
replays a recorded graph of 40 whole sweeps (``rheology_solver.solve()``, all colors) back to back for --seconds;
also the contact solve and one 5-iteration chunk as the solver loop records it (5 x (contact + sweep), residual,
update_condition). Mutates the solve it measures (timing only).

    PYTHONPATH=<warp>:<newton>:<main> .venv-newton152/bin/python metalsim/physics/mpm/probes/gs_pass.py \
        example:mpm_granular
"""
import argparse, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--seconds", type=float, default=1.0)
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


def probe(rs, cs):
    rh = rs.rheology
    ncol = len(rh.color_offsets.numpy()) - 1
    vel0 = wp.clone(rs.momentum.velocity)
    stress0 = wp.clone(rh.stress)

    def timed(fn, per_unit):
        wp.copy(rs.momentum.velocity, vel0)
        wp.copy(rh.stress, stress0)
        fn()
        wp.synchronize_device()
        k = 40
        with wp.ScopedCapture(force_module_load=False) as cap:
            for _ in range(k):
                fn()
        per = []
        t_end = time.perf_counter() + args.seconds
        while time.perf_counter() < t_end:
            wp.copy(rs.momentum.velocity, vel0)
            wp.copy(rh.stress, stress0)
            wp.synchronize_device()
            t0 = time.perf_counter()
            wp.capture_launch(cap.graph)
            wp.synchronize_device()
            per.append((time.perf_counter() - t0) * 1e6 / (k * per_unit))
        return float(np.median(per[len(per) // 2:]))

    result["colors"] = ncol
    timed(rs.solve, ncol)  # warm
    result["us_per_color_pass"] = timed(rs.solve, ncol)
    result["us_contact_solve"] = timed(cs.solve, 1)

    def chunk():
        for _ in range(5):
            cs.solve()
            rs.solve()
        rs.eval_residual()

    result["us_per_iteration_in_chunk"] = timed(chunk, 5)
    wp.copy(rs.momentum.velocity, vel0)
    wp.copy(rh.stress, stress0)


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
result.pop("armed")
result["scene"] = args.scene
result["warp"] = wp.__file__
print(json.dumps(result))
if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    json.dump(result, open(args.json, "w"), indent=1)
