"""Floating-point semantics of Warp kernels on a device: FMA contraction of ``a*b+c``, and division / sqrt / rsqrt
rounding with and without ``fast_math``. Prints a JSON fingerprint per test so devices can be compared exactly.

    python fp_semantics.py --device metal:0|cuda:0|cpu
"""
import argparse, hashlib, json

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()

N = 1 << 16
rng = np.random.default_rng(0)
a = rng.uniform(-2, 2, N).astype(np.float32)
b = rng.uniform(-2, 2, N).astype(np.float32)
c = (-(a.astype(np.float64) * b) + rng.uniform(-1e-6, 1e-6, N)).astype(np.float32)  # cancellation exposes fma
x = rng.uniform(1e-3, 10, N).astype(np.float32)


def kernels(fast):
    opts = {"fast_math": fast}

    @wp.kernel(module="unique", module_options=opts)
    def madd(a: wp.array[float], b: wp.array[float], c: wp.array[float], o: wp.array[float]):
        i = wp.tid()
        o[i] = a[i] * b[i] + c[i]

    @wp.kernel(module="unique", module_options=opts)
    def dot3(a: wp.array[float], b: wp.array[float], c: wp.array[float], o: wp.array[float]):
        i = wp.tid()
        o[i] = wp.dot(wp.vec3(a[i], b[i], c[i]), wp.vec3(b[i], c[i], a[i]))

    @wp.kernel(module="unique", module_options=opts)
    def div(a: wp.array[float], x: wp.array[float], o: wp.array[float]):
        i = wp.tid()
        o[i] = a[i] / x[i]

    @wp.kernel(module="unique", module_options=opts)
    def sqrt_(x: wp.array[float], o: wp.array[float]):
        i = wp.tid()
        o[i] = wp.sqrt(x[i])

    @wp.kernel(module="unique", module_options=opts)
    def norm(a: wp.array[float], b: wp.array[float], x: wp.array[float], o: wp.array[float]):
        i = wp.tid()
        o[i] = wp.length(wp.vec3(a[i], b[i], x[i]))

    return madd, dot3, div, sqrt_, norm


out = {}
with wp.ScopedDevice(args.device):
    A, B, C, X = (wp.array(v) for v in (a, b, c, x))
    O = wp.zeros(N, dtype=float)
    ref = {
        "madd_sep": (a * b + c).astype(np.float32),  # numpy: separate roundings
        "madd_fma": (a.astype(np.float64) * b + c).astype(np.float32),  # single rounding (exact product)
        "div": (a / x).astype(np.float32),
        "sqrt": np.sqrt(x).astype(np.float32),
    }
    for fast in (False, True):
        madd, dot3, div, sqrt_, norm = kernels(fast)
        tag = "fast" if fast else "precise"
        for name, k, inp in (("madd", madd, [A, B, C]), ("dot3", dot3, [A, B, C]), ("div", div, [A, X]),
                             ("sqrt", sqrt_, [X]), ("norm", norm, [A, B, X])):
            wp.launch(k, N, inputs=inp + [O])
            r = O.numpy()
            e = {"md5": hashlib.md5(r.tobytes()).hexdigest()[:12]}
            if name == "madd":
                e["eq_separate"] = float((r == ref["madd_sep"]).mean())
                e["eq_fma"] = float((r == ref["madd_fma"]).mean())
            if name in ("div", "sqrt"):
                e["eq_correctly_rounded"] = float((r == ref[name]).mean())
            out[f"{tag}.{name}"] = e
print(json.dumps({"device": args.device, "warp": wp.__version__, **out}, indent=1))
