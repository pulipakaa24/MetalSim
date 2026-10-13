"""Bit-level numpy model of NVIDIA's MUFU.RCP / MUFU.SQRT / MUFU.RSQ (float32, flush-to-zero forms).

The algorithm is Oberman & Siu's multifunction interpolator (ARITH-17, 2005): a 2^m-entry table indexed by the top
m significand bits holds (C0, C1, C2); the result is C0 -/+ C1 * xl +/- C2 * sq(xl) in fixed point, where sq is a
truncated squarer (the 19 low columns of its partial-product array are dropped), plus a bias, truncated to 23
fraction bits. Term alignments, biases and tables are those of Benjamin Carleton's ptx-math / pyxis-roc sass-math
(MIT License, Copyright (c) 2020 Benjamin Carleton; https://github.com/pyxis-roc/sass-math, src/sassm/{rcp,sqrt,
rsqrt}.c and tables/), reimplemented here; ``load_tables`` reads their table files. Validated against the L4 (sm_89)
exhaustively by ``approx_agreement.py``.
"""
from __future__ import annotations

import re

import numpy as np

U64 = np.uint64

# (m, c0 alignment, c1 alignment, c2 alignment, sum weight, bias)
PARAMS = {
    "rcp": (7, 30, 17, 0, 56, 0x67E7000000000000),
    "sqrt": (6, 33, 18, 0, 58, 0x7FFF800000000000),
    "rsqrt": (6, 31, 17, 0, 57, 0x7FFF800000000000),
}


HEADER = None  # warp/native/metal_cuda_fast_math.h of the Warp tree in use (found from the warp package)


def load_tables(paths: dict[str, str] | None = None) -> dict[str, np.ndarray]:
    """{name: (128, 3) uint64 table}: from sass-math's C table files (``paths``), or by default from the Warp fork's
    ``metal_cuda_fast_math.h`` (``wp_mufu_rcp_packed`` / ``wp_mufu_sqrt_packed``, the same numbers)."""
    out = {}
    if paths is None:
        import os

        import warp

        header = HEADER or os.path.join(os.path.dirname(warp.__file__), "native", "metal_cuda_fast_math.h")
        src = open(header).read()
        for name, shift in (("rcp", 16), ("sqrt", 17)):
            body = src.split(f"wp_mufu_{name}_packed[128] = {{", 1)[1].split("};", 1)[0]
            v = np.array([int(h, 16) for h in re.findall(r"0x[0-9a-fA-F]+", body)], dtype=U64).reshape(128, 2)
            out[name] = np.stack([v[:, 0], v[:, 1] & U64((1 << shift) - 1), v[:, 1] >> U64(shift)], axis=1)
        return out
    for name, p in paths.items():
        src = open(p).read().split("=", 1)[1]
        out[name] = np.array([int(h, 16) for h in re.findall(r"0x[0-9a-fA-F]+", src)], dtype=U64).reshape(128, 3)
    return out


def cuda_rcp(x: np.ndarray, tables) -> np.ndarray:
    """rcp.approx.ftz.f32 of any float32 array (MUFU.RCP model, FTZ, specials as measured on the L4)."""
    x = np.atleast_1d(np.asarray(x, np.float32))
    u = x.view(np.uint32)
    ax = u & np.uint32(0x7FFFFFFF)
    ex = (ax >> np.uint32(23)).astype(np.int64)
    mb = mufu_bits("rcp", ((ax & np.uint32(0x7FFFFF)) | np.uint32(0x3F800000)), tables).astype(np.int64)
    re_ = ((mb >> 23) & 0xFF) - (ex - 127)
    rb = np.where(re_ >= 1, (mb & 0x7FFFFF) | (re_ << 23), 0).astype(np.uint32)
    r = rb.view(np.float32).copy()
    r[ex == 0] = np.inf
    r[np.isinf(x)] = 0
    r = np.copysign(r, x).astype(np.float32)
    r[np.isnan(x)] = np.nan
    return r


def cuda_sqrt(x: np.ndarray, tables) -> np.ndarray:
    """sqrt.approx.ftz.f32 for non-negative float32 (MUFU.SQRT model, FTZ)."""
    x = np.atleast_1d(np.asarray(x, np.float32))
    u = x.view(np.uint32)
    small = u < np.uint32(0x00800000)
    r = mufu_bits("sqrt", np.where(small, np.uint32(0x3F800000), u).astype(np.uint32), tables).view(np.float32).copy()
    r[small] = 0
    return r


def ftz(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32).copy()
    t = np.abs(v) < np.float32(2.0 ** -126)
    v[t] = np.copysign(np.float32(0), v[t])
    return v


def cuda_div(a: np.ndarray, b: np.ndarray, tables) -> np.ndarray:
    """div.approx.ftz.f32: FMUL.FTZ(a, MUFU.RCP(b)) (a * r is exact in float64, then rounded once)."""
    a = ftz(np.atleast_1d(np.asarray(a, np.float32)))
    return ftz((a.astype(np.float64) * cuda_rcp(b, tables).astype(np.float64)).astype(np.float32))


def square_approx(x: np.ndarray) -> np.ndarray:
    """x*x minus the low 19 columns of the antidiagonal and the first 9 partial-product rows (sass-math squarer.c)."""
    x = x.astype(U64)
    mask19 = U64((1 << 19) - 1)
    anti = np.zeros_like(x)
    for i in range(16):
        anti |= ((x >> U64(i)) & U64(1)) << U64(2 * i)
    err = anti & mask19
    for i in range(9):
        bit = (x >> U64(i)) & U64(1)
        row = ((x << U64(i + 1)) & U64(0xFFFFFFFF)) & ~U64((1 << (2 * i + 2)) - 1)
        err = err + np.where(bit == 1, row & mask19, U64(0))
    return x * x - err


def mufu_bits(name: str, xbits: np.ndarray, tables: dict[str, np.ndarray]) -> np.ndarray:
    """Model result bits for positive normal float32 inputs (bit patterns ``xbits``)."""
    m, a0, a1, a2, w, bias = PARAMS[name]
    tbl = tables[name]
    xbits = xbits.astype(np.uint32)
    e = (xbits >> np.uint32(23)).astype(np.int64) - 127
    frac = (xbits & np.uint32(0x7FFFFF)).astype(U64)
    xh = frac >> U64(23 - m)
    xl = frac & U64((1 << (23 - m)) - 1)
    if name == "rcp":
        c = tbl[xh.astype(np.int64)]
        s = (c[:, 0] << U64(a0)) - ((c[:, 1] * xl) << U64(a1)) + ((c[:, 2] * (square_approx(xl << U64(1)) >> U64(2)))
                                                                  << U64(a2))
        s = s + (U64(bias) >> U64((64 - w) + 23))
        top = (s >> U64(w)) != 0
        rexp = np.where(top, 127, 126).astype(np.int64)
        s = np.where(top, s, s << U64(1))
        rfrac = ((s >> U64(w - 23)) & U64(0x7FFFFF)).astype(np.int64)
        rexp = rexp - e
    else:
        odd = (e & 1) == 1
        idx = (xh + np.where(odd, U64(1 << m), U64(0))).astype(np.int64)
        half = (e - odd.astype(np.int64)) // 2
        c = tbl[idx]
        sq = square_approx(xl) << U64(a2)
        if name == "sqrt":
            s = (c[:, 0] << U64(a0)) + ((c[:, 1] * xl) << U64(a1)) - c[:, 2] * sq
            s = s + (U64(bias) >> U64((64 - w) + 23))
            rexp = 127 + half
        else:
            s = (c[:, 0] << U64(a0)) - ((c[:, 1] * xl) << U64(a1)) + c[:, 2] * sq
            s = s + (U64(bias) >> U64((64 - w) + 23))
            s = s << U64(1)
            rexp = 126 - half
        rfrac = ((s >> U64(w - 23)) & U64(0x7FFFFF)).astype(np.int64)
        if name == "rsqrt":  # even exponent, zero fraction: exact power of two
            exact = (~odd) & (frac == 0)
            rfrac = np.where(exact, 0, rfrac)
            rexp = np.where(exact, 127 - half, rexp)
    return ((rexp << 23) | rfrac).astype(np.uint32)
