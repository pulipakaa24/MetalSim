"""Metal float32 reciprocal / division / sqrt candidates against the L4's ``--use_fast_math`` results, bit for bit.

The reference is ``mufu_model.py``, which equals the L4 bit for bit on every input of these sweeps
(``cuda_approx_dump.py``: rcp.approx.ftz.f32 = MUFU.RCP over [1, 2), sqrt.approx.ftz.f32 = MUFU.SQRT over [1, 4);
both exponent-invariant on the L4, so the sweeps cover every normal input).

    python metal_approx_probe.py [--json out.json]

For every candidate (an MSL expression in ``x``, or ``a`` and ``b`` for division) it reports the fraction of inputs
equal to CUDA bit for bit, the correctly rounded fraction and the ulp-difference histogram. Division is scored on
random pairs over a wide range against CUDA's ``a / b`` in a fast_math kernel = FMUL.FTZ(a, MUFU.RCP(b)) (measured
bit-identical on 2^24 pairs), which numpy reproduces exactly from the reciprocal table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from approx_emulation import CANDIDATES_DIV, CANDIDATES_UNARY, MSL_PRELUDE  # noqa: E402
from mufu_model import cuda_div, load_tables, mufu_bits  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--json")
ap.add_argument("--only", default="")
args = ap.parse_args()

import warp as wp

wp.config.quiet = True
wp.init()

TABLES = load_tables()
N24 = 1 << 24
X = (np.arange(N24, dtype=np.uint32) + np.uint32(0x3F800000)).view(np.float32)


def cuda_unary(name):
    x = X[: 1 << 23] if name == "rcp" else X
    x64 = x.astype(np.float64)
    ref = {"rcp": 1 / x64, "sqrt": np.sqrt(x64)}[name]
    cr = ref.astype(np.float32)
    xb = x.view(np.uint32)
    cu = np.concatenate([mufu_bits(name, xb[i:i + (1 << 21)], TABLES) for i in range(0, len(xb), 1 << 21)])
    return x, cr, cu.view(np.float32)


def make_unary_kernel(expr):
    snippet = MSL_PRELUDE + (expr if "return" in expr else f"return {expr};")

    @wp.func_native(snippet)
    def f(x: float) -> float: ...

    @wp.kernel(module="unique", enable_backward=False)
    def k(xs: wp.array[float], out: wp.array[float]):
        i = wp.tid()
        out[i] = f(xs[i])

    return k


def make_div_kernel(expr):
    snippet = MSL_PRELUDE + (expr if "return" in expr else f"return {expr};")

    @wp.func_native(snippet)
    def f(a: float, b: float) -> float: ...

    @wp.kernel(module="unique", enable_backward=False)
    def k(a: wp.array[float], b: wp.array[float], out: wp.array[float]):
        i = wp.tid()
        out[i] = f(a[i], b[i])

    return k


def ulp_hist(r, ref):
    dd = r.view(np.int32).astype(np.int64) - ref.view(np.int32).astype(np.int64)
    u, c = np.unique(np.clip(dd, -3, 3), return_counts=True)
    return {int(k): round(float(v) / len(dd), 6) for k, v in zip(u, c)}


def ftz(v):
    v = v.copy()
    v[np.abs(v) < np.float32(2.0 ** -126)] = 0.0
    return v


out = {}
with wp.ScopedDevice("metal:0"):
    for name, cands in CANDIDATES_UNARY.items():
        x, cr, cu = cuda_unary(name)
        xs = wp.array(x)
        o = wp.zeros(len(x), dtype=float)
        out[name] = {"cuda": {"eq_cr": float((cu == cr).mean()), "ulp_vs_cr": ulp_hist(cu, cr)}}
        for cname, expr in cands.items():
            if args.only and args.only not in cname:
                continue
            wp.launch(make_unary_kernel(expr), len(x), inputs=[xs, o])
            r = o.numpy()
            out[name][cname] = {"eq_cuda": float((r.view(np.uint32) == cu.view(np.uint32)).mean()),
                                "eq_cr": float((r == cr).mean()), "ulp_vs_cuda": ulp_hist(r, cu)}
            print(f"{name:6s} {cname:28s} eq_cuda {out[name][cname]['eq_cuda']:.6f} eq_cr "
                  f"{out[name][cname]['eq_cr']:.4f} {out[name][cname]['ulp_vs_cuda']}", flush=True)

    rng = np.random.default_rng(7)
    m = 1 << 22
    a = (rng.uniform(1, 2, m) * 2.0 ** rng.integers(-30, 30, m) * rng.choice([-1, 1], m)).astype(np.float32)
    b = (rng.uniform(1, 2, m) * 2.0 ** rng.integers(-30, 30, m) * rng.choice([-1, 1], m)).astype(np.float32)
    ref = cuda_div(a, b, TABLES)
    cr = (a.astype(np.float64) / b).astype(np.float32)
    A, B = wp.array(a), wp.array(b)
    o = wp.zeros(m, dtype=float)
    out["div"] = {"cuda": {"eq_cr": float((ref == cr).mean())}}
    for cname, expr in CANDIDATES_DIV.items():
        if args.only and args.only not in cname:
            continue
        wp.launch(make_div_kernel(expr), m, inputs=[A, B, o])
        r = o.numpy()
        out["div"][cname] = {"eq_cuda": float((r.view(np.uint32) == ref.view(np.uint32)).mean()),
                             "eq_cr": float((r == cr).mean()), "ulp_vs_cuda": ulp_hist(r, ref)}
        print(f"div    {cname:28s} eq_cuda {out['div'][cname]['eq_cuda']:.6f} eq_cr {out['div'][cname]['eq_cr']:.4f} "
              f"{out['div'][cname]['ulp_vs_cuda']}", flush=True)
for k, v in out.items():
    print(k, "cuda", v["cuda"])
if args.json:
    json.dump(out, open(args.json, "w"), indent=1)
