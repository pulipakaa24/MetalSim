"""Interleaved A/B timing of bit-identical solver variants on one trajectory.

Every variant listed computes the same numbers (verified separately with record_mpm.py + compare.py), so frames can
alternate between variants along a single trajectory; background GPU load from other processes then hits all
variants alike and the per-variant medians and their ratios are comparable.

    PYTHONPATH=. .venv-newton152/bin/python metalsim/physics/mpm/probes/ab_timing.py \
        example:mpm_granular --variants oracle,cores,cores+graph --rounds 6
Variants: ``oracle`` (Newton 1.5.2's one 64-thread block per color off CUDA), ``cores`` (two blocks per GPU core),
``+graph`` (replay the 5-iteration chunk as a recorded graph), ``+fused`` (one-dispatch solve, fused_gs.py), ``+soa`` (Newton's gs-soa layout; bit-identical to gs),
``+gpuexit`` (chunk graph with the stop decided on the GPU, newton_compat gpu_exit).
"""
import argparse, json, time

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--device", default="metal:0")
ap.add_argument("--variants", default="oracle,cores,cores+graph")
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--warmup", type=int, default=2)
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
branch_fn = getattr(sr, "_color_block_count", None)
variants = args.variants.split(",")
if branch_fn is None and any(v.startswith("cores") for v in variants):
    raise SystemExit("'cores' needs the Newton fork (metalsim-1.5.2, scripts/setup_newton152.sh)")


def apply(v):
    base, _, extra = v.partition("+")
    sc.solver.solver = "gs-soa" if extra == "soa" else "gs"
    if branch_fn is not None:
        sr._color_block_count = branch_fn if base == "cores" else (lambda device: 1)
    newton_compat._STATE["fused"] = extra == "fused"
    if extra.startswith("simd"):
        raise SystemExit("the +simd / +simdgraph variants were removed with the SIMD sweep (2026-10-14)")
    newton_compat._STATE["chunk_graph"] = extra in ("graph", "gpuexit")
    newton_compat._STATE["gpu_exit"] = extra == "gpuexit"
    newton_compat._STATE["fused_max_color_blocks"] = 1 << 30


kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
times = {v: [] for v in variants}
iters = []
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    for v in variants:  # compile every variant's kernels and graphs
        apply(v)
        for _ in range(max(1, args.warmup // len(variants))):
            sc.step()
    newton_compat.drain_log()
    for r in range(args.rounds):
        order = variants if r % 2 == 0 else variants[::-1]
        for v in order:
            apply(v)
            wp.synchronize_device()
            t0 = time.perf_counter()
            sc.step()
            wp.synchronize_device()
            times[v].append((time.perf_counter() - t0) * 1e3)
    iters = newton_compat.drain_log()["iterations"]
res = {v: {"median_ms": float(np.median(t)), "min_ms": float(np.min(t)), "all_ms": t} for v, t in times.items()}
base = res[variants[0]]["median_ms"]
print(f"{args.scene}: {sc.model.particle_count} particles, iterations/solve mean {iters.mean():.1f}")
for v in variants:
    print(f"  {v:16s} median {res[v]['median_ms']:8.1f} ms  min {res[v]['min_ms']:8.1f} ms  "
          f"speedup {base / res[v]['median_ms']:.2f}x")
if args.json:
    with open(args.json, "w") as fh:
        json.dump({"scene": args.scene, "particles": int(sc.model.particle_count), "variants": res,
                   "iterations_mean": float(iters.mean())}, fh, indent=1)
