"""Per-step particle state of an MPM scene (positions, velocities and Newton's MPM state: Jp, elastic strain, stress,
velocity gradient) for the first N steps, on CUDA (the L4) or Metal, for step-by-step device comparison.

    python dump_steps.py --device cuda:0|metal:0 --scene example:mpm_multi_material --steps 12 --out DIR
        [--precise-cuda] [--cuda-nofma] [--particles A:B] [--metal-fast-math off]

Runs next to newton_compat.py / scenes.py on the L4 machine (same copies as the floor runs) or from metalsim on the Mac, with
record_mpm.py's floor settings: float32 accumulation, CUDA stopping rule (host-run), logged solves.
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
ap.add_argument("--scene", default="example:mpm_multi_material")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--seconds", type=float, default=None)
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--out", required=True)
ap.add_argument("--particles", default=None, help="A:B slice of particles to store (default all)")
ap.add_argument("--precise-cuda", action="store_true")
ap.add_argument("--cuda-nofma", action="store_true")
ap.add_argument("--metal-fast-math", default=None)
ap.add_argument("--chunk-graph", action="store_true")
ap.add_argument("--gpu-exit", action="store_true")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
_variant = "_".join(n for n, on in (("precise", args.precise_cuda), ("nofma", args.cuda_nofma)) if on)
if _variant:
    wp.config.kernel_cache_dir = os.path.expanduser(f"~/.cache/warp_{_variant}")
wp.init()
if args.precise_cuda or args.cuda_nofma:
    from warp._src.context import runtime

    _compile = runtime.core.wp_cuda_compile_program

    def _compile_variant(*a):
        a = list(a)
        if args.precise_cuda:
            a[11] = False
        if args.cuda_nofma:
            a[12] = False
        return _compile(*a)

    runtime.core.wp_cuda_compile_program = _compile_variant

here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(here, "newton_compat.py")):
    sys.path.insert(0, here)
    import newton_compat, scenes  # noqa: E401
else:
    from metalsim.physics.mpm import newton_compat, scenes

kw_install = {}
if args.metal_fast_math:
    kw_install["metal_fast_math"] = args.metal_fast_math
if args.gpu_exit:
    kw_install["gpu_exit"] = True
newton_compat.install(accumulate="float32", loop="cuda", log=True, chunk_graph=args.chunk_graph, **kw_install)

kw = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
if args.seconds is not None:
    kw["seconds"] = args.seconds
sl = slice(*[int(x) for x in args.particles.split(":")]) if args.particles else slice(None)
os.makedirs(args.out, exist_ok=True)
FIELDS = ["particle_q", "particle_qd"]
MPM = ["particle_Jp", "particle_elastic_strain", "particle_stress", "particle_qd_grad"]
with wp.ScopedDevice(args.device):
    sc = scenes.make_scene(args.scene, **kw)
    acc = {}
    for step in range(args.steps):
        sc.step()
        st = sc.state
        for f in FIELDS:
            acc.setdefault(f, []).append(getattr(st, f).numpy()[sl].copy())
        m = getattr(st, "mpm", None)
        for f in MPM:
            a = getattr(m, f, None) if m is not None else None
            if a is not None:
                acc.setdefault(f, []).append(a.numpy()[sl].copy())
    np.savez_compressed(os.path.join(args.out, "steps.npz"), **{k: np.stack(v) for k, v in acc.items()})
    log = newton_compat.drain_log()
    np.savez_compressed(os.path.join(args.out, "solves.npz"), **log)
print("done", args.out, {k: len(v) for k, v in acc.items()}, "iterations", log["iterations"].tolist())
