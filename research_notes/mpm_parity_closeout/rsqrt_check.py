"""Metal ``wp_cuda_rsqrt`` (MUFU.RSQ emulation) against the sass-math model, which equals the L4's rsqrt.approx on
every input in [1, 4) (MPM.md section 8 1b, cuda_approx_math.md), plus special and subnormal inputs; and how often
CUDA's rsqrt differs from the correctly rounded 1 / sqrt that Warp's svd3 / eig3 used on Metal before.

    python rsqrt_check.py MUFU_MODEL_DIR
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import mufu_model  # noqa: E402

import warp as wp  # noqa: E402

wp.config.quiet = True
wp.init()

hdr = open(os.path.join(os.path.dirname(wp.__file__), "native", "metal_cuda_fast_math.h")).read()
body = hdr.split("wp_mufu_rsqrt_packed[128] = {", 1)[1].split("};", 1)[0]
v = np.array([int(h, 16) for h in re.findall(r"0x[0-9a-fA-F]+", body)], dtype=np.uint64).reshape(128, 2)
T = {"rsqrt": np.stack([v[:, 0], v[:, 1] & np.uint64((1 << 17) - 1), v[:, 1] >> np.uint64(17)], axis=1)}


@wp.func_native("return wp_cuda_rsqrt(x);")
def cuda_rsqrt(x: float) -> float: ...


@wp.kernel(enable_backward=False)
def k(xs: wp.array[float], out: wp.array[float], ieee: wp.array[float]):
    i = wp.tid()
    out[i] = cuda_rsqrt(xs[i])
    ieee[i] = 1.0 / wp.sqrt(xs[i])


def run(x):
    xs = wp.array(x, dtype=float, device="metal:0")
    o = wp.zeros_like(xs)
    ie = wp.zeros_like(xs)
    wp.launch(k, dim=len(x), inputs=[xs, o, ie], device="metal:0")
    return o.numpy(), ie.numpy()


X = (np.arange(1 << 24, dtype=np.uint32) + np.uint32(0x3F800000)).view(np.float32)
got, ieee = run(X)
ref = np.concatenate([mufu_model.mufu_bits("rsqrt", X.view(np.uint32)[i:i + (1 << 21)], T)
                      for i in range(0, 1 << 24, 1 << 21)])
eq = got.view(np.uint32) == ref
print(f"[1, 4): wp_cuda_rsqrt == MUFU.RSQ model on {eq.mean() * 100:.4f} % ({(~eq).sum()} mismatches)")
cr = (1.0 / np.sqrt(X.astype(np.float64))).astype(np.float32)
print(f"  MUFU.RSQ correctly rounded {np.mean(ref == cr.view(np.uint32)) * 100:.2f} %; "
      f"Metal 1/sqrt (old svd3 path) == CUDA rsqrt on {np.mean(ieee.view(np.uint32) == ref) * 100:.2f} %")
# exponent invariance of the emulation over the normal range (spot: 4096 significands x every even exponent shift)
rng = np.random.default_rng(0)
m = X[rng.integers(0, 1 << 24, 4096)]
bad = 0
for k2 in range(-63, 63):
    xx = (m.astype(np.float64) * 4.0 ** k2).astype(np.float32)
    ok = np.isfinite(xx) & (xx >= np.float32(2.0 ** -126))
    g, _ = run(xx[ok])
    r0 = mufu_model.mufu_bits("rsqrt", m[ok].view(np.uint32), T).view(np.float32).astype(np.float64) * 2.0 ** -k2
    bad += int((g != r0.astype(np.float32)).sum())
print("exponent shifts 4^-63..4^62:", "all equal" if bad == 0 else f"{bad} mismatches")
sp = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, -1.0, 2.0 ** -149, 2.0 ** -140, 3 * 2.0 ** -140, 2.0 ** -127,
               1.0, 4.0, 0.25, 2.0, 2.0 ** 127, 3.4e38], dtype=np.float32)
g, _ = run(sp)
# subnormal reference: CUDA pre-scales by 2^24 and post-scales by 2^12 (model of the normalised input)
exp_sub = []
for x in sp:
    if 0 < x < 2.0 ** -126:
        xs_ = np.float32(np.float64(x) * 2.0 ** 24)
        exp_sub.append(float(mufu_model.mufu_bits("rsqrt", np.array([xs_]).view(np.uint32), T).view(np.float32)[0] * 2.0 ** 12))
    else:
        exp_sub.append(None)
for x, y, e in zip(sp, g, exp_sub):
    print(f"  rsqrt({x!r}) = {y!r}" + (f"  (CUDA pre-scaled model {e!r})" if e is not None else ""))
