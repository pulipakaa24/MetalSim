"""How NVRTC --use_fast_math composes MUFU.RCP / MUFU.SQRT in Warp expressions (run on the L4, Warp 1.16).

Constant divisors, shared divisors, a / sqrt(b), normalize, length, cw_div, vec / scalar, in a fast_math kernel; the
outputs are scored on the Mac against compositions of the bit-exact MUFU model (approx_agreement.py --compose).

    python cuda_approx_compose.py --out DIR      ->  DIR/compose.npz
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import warp as wp

wp.init()

N = 1 << 20
NOUT = 16


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


rng = np.random.default_rng(3)
a = (rng.uniform(-2, 2, N) * 2.0 ** rng.integers(-8, 8, N)).astype(np.float32)
b = (rng.uniform(-2, 2, N) * 2.0 ** rng.integers(-8, 8, N)).astype(np.float32)
c = (rng.uniform(-2, 2, N) * 2.0 ** rng.integers(-8, 8, N)).astype(np.float32)
with wp.ScopedDevice("cuda:0"):
    o = wp.zeros((NOUT, N), dtype=float)
    wp.launch(compose, N, inputs=[wp.array(a), wp.array(b), wp.array(c), o])
    np.savez_compressed(os.path.join(args.out, "compose.npz"), a=a, b=b, c=c, o=o.numpy())
print("ok")
