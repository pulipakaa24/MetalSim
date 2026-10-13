"""Exhaustive float32 fingerprints of Warp's wp.log / wp.exp / wp.sinh (non-fast_math kernels) on one device, so CUDA
(the L4: libdevice __nv_logf / __nv_expf / __nv_sinhf) and Metal can be compared on all 2^32 inputs without moving
16 GB per function: every thread hashes 4,096 consecutive input bit patterns; the per-thread hashes (2^20 x uint32)
are saved. Equal hashes = equal results on that chunk (NaN payloads canonicalised). Also saves the results on
a 2^22-input random sample for agreement statistics.

    python libm_hash.py --device cuda:0|metal:0 --out DIR [--fns log,exp,sinh]
"""
import argparse
import os

import numpy as np
import warp as wp

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fns", default="log,exp,sinh")
args = ap.parse_args()
wp.config.quiet = True
wp.init()

BITCAST = """
#if defined(__METAL_VERSION__)
return as_type<float>(u);
#else
return __uint_as_float(u);
#endif
"""
BITS = """
#if defined(__METAL_VERSION__)
return as_type<uint>(x);
#else
return __float_as_uint(x);
#endif
"""


@wp.func_native(BITCAST)
def u2f(u: wp.uint32) -> float: ...


@wp.func_native(BITS)
def f2u(x: float) -> wp.uint32: ...


@wp.func
def canon(r: float) -> wp.uint32:
    b = f2u(r)
    if r != r:
        b = wp.uint32(0x7FC00000)
    return b


@wp.func
def mix(h: wp.uint32, v: wp.uint32) -> wp.uint32:
    h = h ^ v
    h = h * wp.uint32(0x01000193)
    h = h ^ (h >> wp.uint32(15))
    return h


CHUNK = 4096


# One kernel pair per function, written out (a closure over the function would hash to one "unique" module).
@wp.kernel(enable_backward=False)
def hash_log(out: wp.array[wp.uint32]):
    t = wp.tid()
    h = wp.uint32(0x811C9DC5)
    base = wp.uint32(t) * wp.uint32(CHUNK)
    for i in range(CHUNK):
        h = mix(h, canon(wp.log(u2f(base + wp.uint32(i)))))
    out[t] = h


@wp.kernel(enable_backward=False)
def hash_exp(out: wp.array[wp.uint32]):
    t = wp.tid()
    h = wp.uint32(0x811C9DC5)
    base = wp.uint32(t) * wp.uint32(CHUNK)
    for i in range(CHUNK):
        h = mix(h, canon(wp.exp(u2f(base + wp.uint32(i)))))
    out[t] = h


@wp.kernel(enable_backward=False)
def hash_sinh(out: wp.array[wp.uint32]):
    t = wp.tid()
    h = wp.uint32(0x811C9DC5)
    base = wp.uint32(t) * wp.uint32(CHUNK)
    for i in range(CHUNK):
        h = mix(h, canon(wp.sinh(u2f(base + wp.uint32(i)))))
    out[t] = h


@wp.kernel(enable_backward=False)
def eval_all(xs: wp.array[wp.uint32], lo: wp.array[wp.uint32], ex: wp.array[wp.uint32], sh: wp.array[wp.uint32]):
    t = wp.tid()
    x = u2f(xs[t])
    lo[t] = canon(wp.log(x))
    ex[t] = canon(wp.exp(x))
    sh[t] = canon(wp.sinh(x))


HASH = {"log": hash_log, "exp": hash_exp, "sinh": hash_sinh}


os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(1234)
sample = rng.integers(0, 1 << 32, 1 << 22, dtype=np.uint64).astype(np.uint32)
with wp.ScopedDevice(args.device):
    xs = wp.array(sample, dtype=wp.uint32)
    ys = {k: wp.zeros(sample.size, dtype=wp.uint32) for k in ("log", "exp", "sinh")}
    wp.launch(eval_all, dim=sample.size, inputs=[xs, ys["log"], ys["exp"], ys["sinh"]])
    for fn in args.fns.split(","):
        h = wp.zeros((1 << 32) // CHUNK, dtype=wp.uint32)
        wp.launch(HASH[fn], dim=h.shape[0], inputs=[h])
        np.savez_compressed(os.path.join(args.out, f"{fn}.npz"), hash=h.numpy(), sample_x=sample,
                            sample_y=ys[fn].numpy())
        print(fn, "done", flush=True)
