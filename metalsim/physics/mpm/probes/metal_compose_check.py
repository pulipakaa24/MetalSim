"""Warp's fast_math compositions on Metal against CUDA's, per ``warp.config.metal_fast_math`` mode: fraction of
outputs bit-identical to CUDA, per expression.

The CUDA reference is the per-operation MUFU model (``mufu_model.py``: MUFU.RCP + FMUL.FTZ per ``/``, MUFU.SQRT per
sqrt, Warp's FMA contraction of the dot products), which equals the L4's output of ``cuda_approx_compose.py`` (same
inputs) on 100 % of the 2^20 x 16 values (measured 2026-10-11). ``--cuda DIR/compose.npz`` scores an L4 dump instead.

    python metal_compose_check.py [--cuda DIR/compose.npz] [--json out.json]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cuda")
ap.add_argument("--json")
ap.add_argument("--mode", help="one warp.config.metal_fast_math mode (default: each, one process per mode)")
args = ap.parse_args()
if args.mode is None:
    res = {}
    for mode in ("cuda", "relaxed", "off"):
        out = subprocess.run([sys.executable, __file__, *(["--cuda", args.cuda] if args.cuda else []), "--mode", mode],
                             check=True, capture_output=True, text=True).stdout
        print(out.strip().splitlines()[-2], flush=True)
        res[mode] = json.loads(out.strip().splitlines()[-1])
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
    sys.exit(0)

import warp as wp

wp.config.quiet = True
wp.config.metal_fast_math = args.mode  # read when the fast_math module is hashed (at the kernel definition below)
wp.init()

NOUT = 16
NAMES = ["x/3", "x/0.1", "x/z+y/z", "x/sqrt|y|", "length", "normalize[0]", "normalize[2]", "cw_div[0]", "cw_div[1]",
         "(v/z)[1]", "1/x", "(x*y)/z", "x/(y+z)", "sqrt(xx+yy)", "2x/z", "x/z-y/z"]


@wp.kernel(module="unique", module_options={"fast_math": True}, enable_backward=False)
def compose(a: wp.array[float], b: wp.array[float], c: wp.array[float], o: wp.array2d[float]):
    i = wp.tid()
    x = a[i]
    y = b[i]
    z = c[i]
    o[0, i] = x / 3.0
    o[1, i] = x / 0.1
    o[2, i] = x / z + y / z
    o[3, i] = x / wp.sqrt(wp.abs(y))
    v = wp.vec3(x, y, z)
    o[4, i] = wp.length(v)
    n = wp.normalize(v)
    o[5, i] = n[0]
    o[6, i] = n[2]
    w = wp.cw_div(v, wp.vec3(z, x, y))
    o[7, i] = w[0]
    o[8, i] = w[1]
    u = v / z
    o[9, i] = u[1]
    o[10, i] = 1.0 / x
    o[11, i] = (x * y) / z
    o[12, i] = x / (y + z)
    o[13, i] = wp.sqrt(x * x + y * y)
    o[14, i] = 2.0 * x / z
    o[15, i] = x / z - y / z


if args.cuda:
    z = dict(np.load(args.cuda))
    ref = z["o"]
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mufu_model import cuda_div, cuda_rcp, cuda_sqrt, ftz, load_tables

    T = load_tables()
    rng = np.random.default_rng(3)  # cuda_approx_compose.py's inputs
    N = 1 << 20
    a, b, c = ((rng.uniform(-2, 2, N) * 2.0 ** rng.integers(-8, 8, N)).astype(np.float32) for _ in range(3))
    z = {"a": a, "b": b, "c": c}
    f32 = np.float32

    def mul(p, q):
        return ftz(np.float32(p) * np.float32(q))

    def fma(p, q, s):
        return ftz((np.float64(p) * np.float64(q) + np.float64(s)).astype(np.float32))

    def add(p, q):
        return ftz(np.float32(p) + np.float32(q))

    def div(p, q):
        return cuda_div(p, q, T)

    r = cuda_rcp(c, T)
    ln = cuda_sqrt(fma(c, c, fma(a, a, mul(b, b))), T)
    ref = np.stack([div(a, f32(3)), div(a, f32(0.1)), add(mul(a, r), mul(b, r)), div(a, cuda_sqrt(np.abs(b), T)), ln,
                    mul(a, cuda_rcp(ln, T)), mul(c, cuda_rcp(ln, T)), div(a, c), div(b, a), div(b, c), cuda_rcp(a, T),
                    div(mul(a, b), c), div(a, add(b, c)), cuda_sqrt(fma(a, a, mul(b, b)), T), div(mul(2, a), c),
                    add(mul(a, r), -mul(b, r))])

with wp.ScopedDevice("metal:0"):
    A, B, C = (wp.array(z[k]) for k in "abc")
    o = wp.zeros((NOUT, len(z["a"])), dtype=float)
    wp.launch(compose, len(z["a"]), inputs=[A, B, C, o])
    r = o.numpy()
    eq = (r.view(np.uint32) == ref.view(np.uint32)) | (np.isnan(r) & np.isnan(ref))
    res = {n: float(eq[k].mean()) for k, n in enumerate(NAMES)}
print(args.mode, " ".join(f"{n}={v:.4f}" for n, v in res.items()))
print(json.dumps(res))
