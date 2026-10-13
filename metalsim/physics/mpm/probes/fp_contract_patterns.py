"""Which multiply-adds a device's compiler fuses in multi-term Warp expressions. For each pattern, saves the kernel
result; ``--analyze`` labels each device's result against candidate fusings emulated in NumPy.

    python fp_contract_patterns.py --device cuda:0|metal:0 --out FILE.npz
    python fp_contract_patterns.py --analyze A.npz B.npz
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--device")
ap.add_argument("--out")
ap.add_argument("--analyze", nargs="*")
args = ap.parse_args()

import numpy as np

N = 1 << 15
rng = np.random.default_rng(1)
V = {k: rng.uniform(-2, 2, N).astype(np.float32) for k in "abcdef"}


def f32(x):
    return np.asarray(x, np.float64).astype(np.float32)


def fma(x, y, z):  # single rounding of x*y+z (product exact in float64)
    return f32(np.float64(1) * x.astype(np.float64) * y + z.astype(np.float64))


a, b, c, d, e, f = (V[k] for k in "abcdef")
CANDIDATES = {
    "ab_plus_cd": {"none": f32(a * b) + f32(c * d), "fma(a,b,cd)": fma(a, b, f32(c * d)), "fma(c,d,ab)": fma(c, d, f32(a * b))},
    "ab_minus_cd": {"none": f32(a * b) - f32(c * d), "fma(a,b,-cd)": fma(a, b, -f32(c * d)),
                    "fma(-c,d,ab)": fma(-c, d, f32(a * b))},
    "sum3": {"none": (f32(a * b) + f32(c * d)) + f32(e * f),
             "fma(e,f,fma(a,b,cd))": fma(e, f, fma(a, b, f32(c * d))),
             "fma(e,f,fma(c,d,ab))": fma(e, f, fma(c, d, f32(a * b))),
             "fma(e,f,ab+cd)": fma(e, f, f32(a * b) + f32(c * d))},
    "sq3": {"none": (f32(a * a) + f32(b * b)) + f32(c * c), "fma(c,c,fma(b,b,aa))": fma(c, c, fma(b, b, f32(a * a))),
            "fma(c,c,fma(a,a,bb))": fma(c, c, fma(a, a, f32(b * b)))},
    "sub_mul_add": {"none": f32((a - b) * (c - d)) + e, "fma": fma(f32(a - b), f32(c - d), e)},
    "mul_sub": {"none": f32(a * b) - c, "fma": fma(a, b, -c)},
    "c_minus_ab": {"none": c - f32(a * b), "fma": fma(-a, b, c)},
}

if args.analyze:
    for path in args.analyze:
        R = np.load(path)
        print(path)
        for name, cands in CANDIDATES.items():
            r = R[name]
            match = {k: float((r == v).mean()) for k, v in cands.items()}
            print(f"  {name:12s} " + "  ".join(f"{k}={m:.3f}" for k, m in match.items()))
    if len(args.analyze) == 2:
        A, B = (np.load(p) for p in args.analyze)
        print("A vs B identical:", {n: bool(np.array_equal(A[n], B[n])) for n in CANDIDATES})
    raise SystemExit

import warp as wp

wp.config.quiet = True
wp.init()


@wp.kernel
def patterns(a: wp.array[float], b: wp.array[float], c: wp.array[float], d: wp.array[float], e: wp.array[float],
             f: wp.array[float], o: wp.array2d[float]):
    i = wp.tid()
    A, B, C, D, E, F = a[i], b[i], c[i], d[i], e[i], f[i]
    o[0, i] = A * B + C * D
    o[1, i] = A * B - C * D
    o[2, i] = A * B + C * D + E * F
    o[3, i] = A * A + B * B + C * C
    o[4, i] = (A - B) * (C - D) + E
    o[5, i] = A * B - C
    o[6, i] = C - A * B


with wp.ScopedDevice(args.device):
    arrs = [wp.array(V[k]) for k in "abcdef"]
    O = wp.zeros((7, N), dtype=float)
    wp.launch(patterns, N, inputs=arrs + [O])
    r = O.numpy()
np.savez(args.out, **{n: r[i] for i, n in enumerate(CANDIDATES)})
print("saved", args.out)
