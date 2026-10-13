"""What makes up the fixed per-dispatch cost of a large Warp kernel on Metal (graph replay, sustained clock).

Zero-work dispatches (every thread exits after reading two color offsets, as Newton's colored Gauss-Seidel kernel
does for an empty color) of kernels that differ in one thing at a time: number of array arguments, whether the
kernel body behind the early exit is large, and Newton's own kernel. 5120 threads in 64-thread groups (Newton's
Metal launch of the colored kernels), K dispatches per graph replayed back to back for --seconds.

    PYTHONPATH=<warp>:<newton>:<main> .venv-newton152/bin/python metalsim/physics/mpm/probes/fixed_cost.py
"""
import argparse, json, time

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="metal:0")
ap.add_argument("--seconds", type=float, default=1.0)
ap.add_argument("--json")
args = ap.parse_args()

import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()

vec6 = wp.types.vector(6, float)


@wp.kernel
def empty3(color: int, offsets: wp.array[int], out: wp.array[float]):
    i = wp.tid()
    if i < 0:
        out[0] = float(color)


@wp.kernel
def walk4(color: int, launch_dim: int, offsets: wp.array[int], out: wp.array[float]):
    i = wp.tid()
    for k in range(offsets[color] + i, offsets[color + 1], launch_dim):
        out[k] = 1.0


@wp.kernel
def walk19(color: int, launch_dim: int, offsets: wp.array[int], a0: wp.array[float], a1: wp.array[float],
           a2: wp.array[float], a3: wp.array[float], a4: wp.array[float], a5: wp.array[float], a6: wp.array[float],
           a7: wp.array[float], a8: wp.array[float], a9: wp.array[float], a10: wp.array[float],
           a11: wp.array[float], a12: wp.array[float], a13: wp.array[float], a14: wp.array[float],
           a15: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    for k in range(offsets[color] + i, offsets[color + 1], launch_dim):
        out[k] = a0[k] + a1[k] + a2[k] + a3[k] + a4[k] + a5[k] + a6[k] + a7[k] + a8[k] + a9[k] + a10[k] + a11[k] \
            + a12[k] + a13[k] + a14[k] + a15[k]


@wp.kernel
def walk4_big(color: int, launch_dim: int, offsets: wp.array[int], out: wp.array[vec6]):
    """walk4 with a large body (a long dependent chain of 6-vector math) behind the empty range."""
    i = wp.tid()
    for k in range(offsets[color] + i, offsets[color + 1], launch_dim):
        v = out[k]
        for _j in range(wp.static(64)):
            v = vec6(wp.sqrt(wp.abs(v[1])) + v[0] / (1.0 + v[5]), v[2] * v[3] - v[4], wp.length(v), v[0] + v[1],
                     v[3] / (v[2] + 2.0), wp.max(v[4], v[5]))
        out[k] = v


mat66 = wp.types.matrix((6, 6), float)


@wp.kernel
def walk4_regs(color: int, launch_dim: int, offsets: wp.array[int], m: wp.array[mat66], out: wp.array[mat66]):
    """walk4 with a register-heavy body (products of 6x6 matrices kept live) behind the empty range."""
    i = wp.tid()
    for k in range(offsets[color] + i, offsets[color + 1], launch_dim):
        a = m[k]
        b = m[k + 1]
        c = m[k + 2]
        d = a * b
        e = b * c
        f = c * a
        out[k] = d * e + e * f + f * d + a * e * c


def _replay_us(fn, k):
    with wp.ScopedCapture(force_module_load=False) as cap:
        for _ in range(k):
            fn()
    per = []
    t_end = time.perf_counter() + args.seconds
    while time.perf_counter() < t_end:
        wp.synchronize_device()
        t0 = time.perf_counter()
        wp.capture_launch(cap.graph)
        wp.synchronize_device()
        per.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(per[len(per) // 2:]))


def timed(fn, ncol=8, k=40):
    """GPU us per dispatch: slope of the replay time between graphs of k and 4k sweeps of ncol dispatches (removes
    the per-replay commit / wait cost, which the k-sweep figure alone spreads over its dispatches)."""
    fn()
    wp.synchronize_device()
    t1, t4 = _replay_us(fn, k), _replay_us(fn, 4 * k)
    return {"per_dispatch_us": (t4 - t1) / (3 * k * ncol), "per_dispatch_us_incl_replay": t1 / (k * ncol)}


res = {}
with wp.ScopedDevice(args.device):
    dev = wp.get_device()
    blocks = 2 * dev.sm_count
    ld = blocks * 64
    zeros = wp.zeros(9, dtype=int)
    out = wp.zeros(1 << 16, dtype=float)
    out6 = wp.zeros(1 << 16, dtype=vec6)
    arrs = [wp.zeros(1 << 16, dtype=float) for _ in range(16)]

    def L(kernel, inputs_fn):
        def fn():
            for c in range(8):
                wp.launch(kernel, dim=ld, inputs=inputs_fn(c), block_dim=64, max_blocks=blocks)
        return fn

    res["empty3"] = timed(L(empty3, lambda c: [c, zeros, out]))
    res["walk4"] = timed(L(walk4, lambda c: [c, ld, zeros, out]))
    res["walk19"] = timed(L(walk19, lambda c: [c, ld, zeros, *arrs, out]))
    res["walk4_big"] = timed(L(walk4_big, lambda c: [c, ld, zeros, out6]))
    m66 = wp.zeros(1 << 12, dtype=mat66)
    res["walk4_regs"] = timed(L(walk4_regs, lambda c: [c, ld, zeros, m66, m66]))
print(json.dumps(res, indent=1))
if args.json:
    json.dump(res, open(args.json, "w"), indent=1)
