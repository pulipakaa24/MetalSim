"""Dump CUDA's approximate float32 reciprocal / division / sqrt / rsqrt bit for bit (run on the L4, Warp 1.16).

What Warp's ``fast_math`` modules compute (NVRTC ``--use_fast_math``) and the PTX instructions behind them, over
exhaustive mantissa sweeps, so Metal emulations can be scored against the real hardware results
(``metal_approx_probe.swift`` / ``approx_agreement.py``).

    python cuda_approx_dump.py --out DIR

Writes DIR/{rcp_1_2,sqrt_1_4,rsqrt_1_4}.npy (uint32 bit patterns: rcp.approx.ftz.f32 for every float in [1, 2),
sqrt.approx.ftz.f32 / rsqrt.approx.ftz.f32 for every float in [1, 4)), DIR/div_pairs.npz (random pairs a, b and
the results of a / b in a fast_math kernel, div.full.ftz.f32, div.approx.ftz.f32, mul.rn.ftz(a, rcp.approx.ftz(b))),
DIR/summary.json (identity checks: exponent invariance, fast_math `/` and `wp.sqrt` vs the PTX instructions) and
DIR/kernels/ (the generated PTX).
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import warp as wp

wp.config.kernel_cache_dir = os.path.join(args.out, "kcache")
shutil.rmtree(wp.config.kernel_cache_dir, ignore_errors=True)
wp.config.quiet = True
wp.init()


def native(op):
    return f"float r; asm(\"{op}\" : \"=f\"(r) : \"f\"(x)); return r;"


def native2(op):
    return f"float r; asm(\"{op}\" : \"=f\"(r) : \"f\"(a), \"f\"(b)); return r;"


@wp.func_native("return __uint_as_float(u);")
def u2f(u: wp.uint32) -> float: ...


@wp.func_native("return __float_as_uint(x);")
def f2u(x: float) -> wp.uint32: ...


@wp.func_native(native("rcp.approx.ftz.f32 %0, %1;"))
def rcp_approx_ftz(x: float) -> float: ...


@wp.func_native(native("rcp.approx.f32 %0, %1;"))
def rcp_approx(x: float) -> float: ...


@wp.func_native(native("sqrt.approx.ftz.f32 %0, %1;"))
def sqrt_approx_ftz(x: float) -> float: ...


@wp.func_native(native("rsqrt.approx.ftz.f32 %0, %1;"))
def rsqrt_approx_ftz(x: float) -> float: ...


@wp.func_native(native2("div.full.ftz.f32 %0, %1, %2;"))
def div_full_ftz(a: float, b: float) -> float: ...


@wp.func_native(native2("div.approx.ftz.f32 %0, %1, %2;"))
def div_approx_ftz(a: float, b: float) -> float: ...


@wp.func_native(native2("mul.rn.ftz.f32 %0, %1, %2;"))
def mul_rn_ftz(a: float, b: float) -> float: ...


@wp.kernel(module="unique", module_options={"fast_math": True}, enable_backward=False)
def k_unary(base: wp.uint32, rcp_o: wp.array[wp.uint32], rcpn_o: wp.array[wp.uint32], sqrt_o: wp.array[wp.uint32],
            rsqrt_o: wp.array[wp.uint32], wsqrt_o: wp.array[wp.uint32], wrsq_o: wp.array[wp.uint32],
            wrcp_o: wp.array[wp.uint32]):
    i = wp.tid()
    x = u2f(base + wp.uint32(i))
    rcp_o[i] = f2u(rcp_approx_ftz(x))
    rcpn_o[i] = f2u(rcp_approx(x))
    sqrt_o[i] = f2u(sqrt_approx_ftz(x))
    rsqrt_o[i] = f2u(rsqrt_approx_ftz(x))
    wsqrt_o[i] = f2u(wp.sqrt(x))  # what a fast_math kernel's wp.sqrt is
    wrsq_o[i] = f2u(1.0 / wp.sqrt(x))  # and 1 / sqrt
    wrcp_o[i] = f2u(1.0 / x)  # and 1 / x


@wp.kernel(module="unique", module_options={"fast_math": True}, enable_backward=False)
def k_div(a: wp.array[float], b: wp.array[float], fm: wp.array[wp.uint32], full: wp.array[wp.uint32],
          appr: wp.array[wp.uint32], mrcp: wp.array[wp.uint32], ln: wp.array[wp.uint32], nrm: wp.array[wp.uint32]):
    i = wp.tid()
    fm[i] = f2u(a[i] / b[i])
    full[i] = f2u(div_full_ftz(a[i], b[i]))
    appr[i] = f2u(div_approx_ftz(a[i], b[i]))
    mrcp[i] = f2u(mul_rn_ftz(a[i], rcp_approx_ftz(b[i])))
    v = wp.vec3(a[i], b[i], a[i] - b[i])
    ln[i] = f2u(wp.length(v))
    n = wp.normalize(v)
    nrm[i] = f2u(n[0])


# exponent invariance: f(m * 2^e) == f(m) * 2^-e (rcp) for every mantissa m in [1, 2), e in [-126, 125]
@wp.kernel(module="unique", module_options={"fast_math": True}, enable_backward=False)
def k_rcp_scale(e0: int, ne: int, bad: wp.array2d[int]):
    i = wp.tid()
    m = u2f(wp.uint32(0x3F800000) + wp.uint32(i))
    r1 = f2u(rcp_approx_ftz(m))
    for k in range(ne):
        e = e0 + k
        xm = u2f(wp.uint32(0x3F800000 + (e << 23)) + wp.uint32(i))
        r = f2u(rcp_approx_ftz(xm))
        # expected: r1 with exponent field lowered by e (only when it stays a normal number)
        ex = int((r1 >> wp.uint32(23)) & wp.uint32(0xFF)) - e
        if ex >= 1 and ex <= 254:
            expect = (r1 & wp.uint32(0x807FFFFF)) | (wp.uint32(ex) << wp.uint32(23))
            if r != expect:
                wp.atomic_add(bad, k, 0, 1)
        else:
            wp.atomic_add(bad, k, 1, 1)  # out of range: counted separately


@wp.kernel(module="unique", module_options={"fast_math": True}, enable_backward=False)
def k_sqrt_scale(e0: int, ne: int, bad: wp.array2d[int]):
    i = wp.tid()  # x in [1, 4)
    m = u2f(wp.uint32(0x3F800000) + wp.uint32(i))
    s1 = f2u(sqrt_approx_ftz(m))
    q1 = f2u(rsqrt_approx_ftz(m))
    for k in range(ne):
        e = e0 + k  # scale x by 4^e
        xm = u2f(wp.uint32(0x3F800000 + (e << 24)) + wp.uint32(i))
        s = f2u(sqrt_approx_ftz(xm))
        q = f2u(rsqrt_approx_ftz(xm))
        es = int((s1 >> wp.uint32(23)) & wp.uint32(0xFF)) + e
        eq = int((q1 >> wp.uint32(23)) & wp.uint32(0xFF)) - e
        if es >= 1 and es <= 254:
            if s != ((s1 & wp.uint32(0x807FFFFF)) | (wp.uint32(es) << wp.uint32(23))):
                wp.atomic_add(bad, k, 0, 1)
        if eq >= 1 and eq <= 254:
            if q != ((q1 & wp.uint32(0x807FFFFF)) | (wp.uint32(eq) << wp.uint32(23))):
                wp.atomic_add(bad, k, 1, 1)


dev = "cuda:0"
summary = {"warp": wp.__version__, "device": wp.get_device(dev).name}
with wp.ScopedDevice(dev):
    # unary sweeps: [1, 4) is 2^24 floats starting at 0x3F800000
    n = 1 << 24
    outs = [wp.zeros(n, dtype=wp.uint32) for _ in range(7)]
    wp.launch(k_unary, n, inputs=[wp.uint32(0x3F800000), *outs])
    rcp, rcpn, sq, rsq, wsq, wrsq, wrcp = (o.numpy() for o in outs)
    np.save(os.path.join(args.out, "rcp_1_2.npy"), rcp[: 1 << 23])
    np.save(os.path.join(args.out, "sqrt_1_4.npy"), sq)
    np.save(os.path.join(args.out, "rsqrt_1_4.npy"), rsq)
    np.save(os.path.join(args.out, "wrsq_1_4.npy"), wrsq)
    summary["rcp_ftz_eq_rcp_noftz"] = float((rcp == rcpn).mean())
    summary["rcp_1_4_scale_2"] = float((rcp[: 1 << 23] == rcp[1 << 23:] + (1 << 23)).mean())  # rcp(2m) = rcp(m)/2
    summary["fastmath_sqrt_eq_sqrt_approx_ftz"] = float((wsq == sq).mean())
    summary["fastmath_1_over_x_eq_rcp_approx_ftz"] = float((wrcp == rcp).mean())
    summary["fastmath_1_over_sqrt_eq_rsqrt_approx_ftz"] = float((wrsq == rsq).mean())

    # exponent invariance over the whole normal range
    ne = 252
    bad = wp.zeros((ne, 2), dtype=int)
    wp.launch(k_rcp_scale, 1 << 23, inputs=[-126, ne, bad])
    b = bad.numpy()
    summary["rcp_scale_mismatch_by_e"] = {int(-126 + k): int(b[k, 0]) for k in range(ne) if b[k, 0]}
    summary["rcp_scale_out_of_range_by_e"] = {int(-126 + k): int(b[k, 1]) for k in range(ne) if b[k, 1]}
    ne = 126
    bad = wp.zeros((ne, 2), dtype=int)
    wp.launch(k_sqrt_scale, 1 << 24, inputs=[-63, ne, bad])
    b = bad.numpy()
    summary["sqrt_scale_mismatch_by_e"] = {int(-63 + k): int(b[k, 0]) for k in range(ne) if b[k, 0]}
    summary["rsqrt_scale_mismatch_by_e"] = {int(-63 + k): int(b[k, 1]) for k in range(ne) if b[k, 1]}

    # division: random pairs over a wide range, plus special values
    rng = np.random.default_rng(1)
    m = 1 << 24
    a = (rng.uniform(1, 2, m) * 2.0 ** rng.integers(-40, 40, m) * rng.choice([-1, 1], m)).astype(np.float32)
    bb = (rng.uniform(1, 2, m) * 2.0 ** rng.integers(-40, 40, m) * rng.choice([-1, 1], m)).astype(np.float32)
    # edge denominators: near 2^126 .. 2^128, denormal, zero, inf, nan; edge numerators
    spec = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-40, -1e-40, 2.0 ** -126, 2.0 ** 126, 1.5 * 2.0 ** 126,
                     2.0 ** 127, 3.0e38, 1.0, -1.0, 3.0], dtype=np.float32)
    sa, sb = np.meshgrid(spec, spec)
    a[: sa.size] = sa.ravel()
    bb[: sb.size] = sb.ravel()
    k = 4096
    a[sa.size: sa.size + k] = rng.uniform(-4, 4, k).astype(np.float32)
    bb[sa.size: sa.size + k] = (rng.uniform(1, 4, k) * 2.0 ** rng.integers(124, 128, k)).astype(np.float32)
    A, B = wp.array(a), wp.array(bb)
    outs = [wp.zeros(m, dtype=wp.uint32) for _ in range(6)]
    wp.launch(k_div, m, inputs=[A, B, *outs])
    fm, full, appr, mrcp, ln, nrm = (o.numpy() for o in outs)
    np.savez_compressed(os.path.join(args.out, "div_pairs.npz"), a=a, b=bb, fastmath=fm, full=full, approx=appr,
                        mul_rcp=mrcp, length=ln, normalize0=nrm)

    def eq(x, y):  # bitwise, NaN payloads equal if both NaN
        fx, fy = x.view(np.float32), y.view(np.float32)
        return (x == y) | (np.isnan(fx) & np.isnan(fy))

    summary["div_fastmath_eq_full"] = float(eq(fm, full).mean())
    summary["div_fastmath_eq_approx"] = float(eq(fm, appr).mean())
    summary["div_fastmath_eq_mul_rcp"] = float(eq(fm, mrcp).mean())
    summary["div_full_eq_mul_rcp"] = float(eq(full, mrcp).mean())
    nm = ~eq(fm, mrcp)
    summary["div_fastmath_ne_mul_rcp_examples"] = [
        [float(a[i]), float(bb[i]), hex(fm[i]), hex(mrcp[i]), hex(full[i]), hex(appr[i])] for i in np.nonzero(nm)[0][:40]]
    summary["div_specials"] = [[float(a[i]), float(bb[i]), float(fm.view(np.float32)[i]), float(appr.view(np.float32)[i])]
                               for i in range(sa.size)]

# generated sources for PTX / SASS inspection
kd = os.path.join(args.out, "kernels")
os.makedirs(kd, exist_ok=True)
for f in glob.glob(os.path.join(wp.config.kernel_cache_dir, "**", "*"), recursive=True):
    if f.endswith((".ptx", ".cu", ".cubin")):
        shutil.copy(f, kd)
json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if "examples" not in k and "specials" not in k}, indent=1))
