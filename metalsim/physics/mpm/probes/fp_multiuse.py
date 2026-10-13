"""Does the compiler fuse a product that has several uses? m = a*b; o1 = m + c; o2 = m - d; o3 = m (stored).
CUDA (ptxas) contracts only single-use products; prints the fraction of results equal to the fused form.

    python fp_multiuse.py --device D
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
args = ap.parse_args()
import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
N = 1 << 15
rng = np.random.default_rng(2)
a, b = (rng.uniform(-2, 2, N).astype(np.float32) for _ in range(2))
c = (-(a.astype(np.float64) * b) + rng.uniform(-1e-6, 1e-6, N)).astype(np.float32)
d = (a.astype(np.float64) * b + rng.uniform(-1e-6, 1e-6, N)).astype(np.float32)


@wp.kernel
def k2(a: wp.array[float], b: wp.array[float], c: wp.array[float], d: wp.array[float], o: wp.array2d[float]):
    i = wp.tid()
    m = a[i] * b[i]
    o[0, i] = m + c[i]
    o[1, i] = m - d[i]
    o[2, i] = m


@wp.kernel
def k1(a: wp.array[float], b: wp.array[float], c: wp.array[float], d: wp.array[float], o: wp.array2d[float]):
    i = wp.tid()
    m = a[i] * b[i]
    o[0, i] = m + c[i]


fused_add = (a.astype(np.float64) * b + c).astype(np.float32)
sep_add = (a * b).astype(np.float32) + c
res = {}
with wp.ScopedDevice(args.device):
    A, B, C, D = (wp.array(x) for x in (a, b, c, d))
    for name, k in (("single_use", k1), ("multi_use", k2)):
        O = wp.zeros((3, N), dtype=float)
        wp.launch(k, N, inputs=[A, B, C, D, O])
        r = O.numpy()[0]
        res[name] = {"eq_fused": float((r == fused_add).mean()), "eq_separate": float((r == sep_add).mean())}
print(json.dumps({"device": args.device, **res}))
