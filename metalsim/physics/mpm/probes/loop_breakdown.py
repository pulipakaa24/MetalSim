"""Cost of the pieces of one rheology iteration on Metal (the strain solve is ~90 % of a frame).

Hooks one solve after warm-up and times each component N times in isolation, synchronized: the contact solve, one
Gauss-Seidel sweep over all colors, one launch of the largest and the smallest color, the residual evaluation, and a
bare host sync. Mutates the solve it measures, so the trajectory after it is meaningless (timing only).

    PYTHONPATH=. .venv-newton152/bin/python metalsim/physics/mpm/probes/loop_breakdown.py \
        example:mpm_granular
"""
import argparse, json, time

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--json")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
from metalsim.physics.mpm import newton_compat, scenes
from newton._src.solvers.implicit_mpm import solve_rheology as sr

newton_compat.install(accumulate="float32", loop="cuda")
cuda_loop = sr._run_solver_loop
result = {}


def timed(fn, n):
    """Per-call time of ``fn`` replayed from a recorded graph of ``n`` calls (GPU-side cost, no Python launch cost);
    host-only callables (with a .numpy() read) are timed eagerly."""
    if getattr(fn, "eager", False):
        wp.synchronize_device()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        wp.synchronize_device()
        return (time.perf_counter() - t0) * 1e3 / n
    with wp.ScopedCapture(force_module_load=False) as cap:
        for _ in range(n):
            fn()
    wp.capture_launch(cap.graph)
    wp.synchronize_device()
    t0 = time.perf_counter()
    for _ in range(3):
        wp.capture_launch(cap.graph)
    wp.synchronize_device()
    return (time.perf_counter() - t0) * 1e3 / (3 * n)


def eager(fn):
    fn.eager = True
    return fn


def probe_loop(rheology_solver, contact_solver, *rest):
    if result.get("armed") and "contact_ms" not in result:
        n = args.n
        offs = rheology_solver.rheology.color_offsets.numpy()
        sizes = np.diff(offs)
        result.update(
            colors=int(len(sizes)), color_blocks_min=int(sizes.min()), color_blocks_max=int(sizes.max()),
            strain_nodes=int(rheology_solver.rheology.stress.shape[0]),
            velocity_nodes=int(rheology_solver.momentum.velocity.shape[0]),
            solver=rheology_solver.name,
        )
        result["contact_ms"] = timed(contact_solver.solve, n)
        result["rheology_sweep_ms"] = timed(rheology_solver.solve, n)
        if hasattr(rheology_solver, "solve_local_launch"):
            launch = rheology_solver.solve_local_launch
            big, small = int(np.argmax(sizes)), int(np.argmin(sizes))

            def one(c):
                launch.set_param_at_index(0, c)
                launch.launch()

            result["largest_color_ms"] = timed(lambda: one(big), n)
            result["smallest_color_ms"] = timed(lambda: one(small), n)
        result["residual_ms"] = timed(rheology_solver.eval_residual, n)
        result["residual_readback_ms"] = timed(eager(lambda: rheology_solver.eval_residual().numpy()), n)
        result["empty_sync_ms"] = timed(eager(lambda: wp.synchronize_device()), n)
    return cuda_loop(rheology_solver, contact_solver, *rest)


sr._run_solver_loop = probe_loop
kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    for _ in range(args.warmup):
        sc.step()
    result["armed"] = True
    sc.step()
result.pop("armed")
it = 5 * result["contact_ms"] + 5 * result["rheology_sweep_ms"] + result["residual_readback_ms"]
result["chunk_est_ms"] = it
print(json.dumps(result, indent=1))
if args.json:
    with open(args.json, "w") as fh:
        json.dump(result, fh, indent=1)
