"""Custom Metal kernels for the two slow NatureCNN gradients on MPS (torch.mps.compile_shader, simdgroup matrices).

Measured on the M4 Max (docs/research/metal_cnn_update_2026-09-24.md): MPSGraph computes the camera-cartpole
encoder's conv1 weight gradient (3->32, k8 s4, 100->24) at 1.6 TFLOP/s and conv2's input gradient (32->64, k4 s2,
24->11) at 2.5 TFLOP/s, while everything else runs at 8-11 TFLOP/s. These kernels replace exactly those two ops:

  conv1 weight gradient: dW[o, (c,kh,kw)] = sum_{n,p} gy[n,o,p] * im2col(x)[n,p,(c,kh,kw)], a 32 x 192 output with a
      reduction over N*576 positions -> split-K GEMM whose im2col tiles are strided simdgroup loads from x, then a
      reduction of the per-threadgroup partials.
  conv2 input gradient: for stride 2 / kernel 4 every input pixel parity (py,px) is a stride-1 2x2 correlation of the
      output gradient with a sub-kernel: dx[n,c,2a+py,2b+px] = sum_{o,ty,tx} gy[n,o,a-ty,b-tx] * W[o,c,py+2ty,px+2tx]
      -> four GEMMs (N*144 x 256) @ (256 x 32) whose A tiles are transposed simdgroup loads from a zero-padded gy.

Each is used through a torch.autograd.Function whose forward is F.conv2d (MPSGraph's forward is fast) and whose
backward swaps in the kernel for that one gradient; the other gradients stay on aten. Shapes other than the ones the
kernels were written for fall back to aten. The kernel entry points are torch.library custom ops so the Functions
can sit inside a torch.compile region (opaque to Inductor).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_HDR = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"""

# ---------------------------------------------------------------- conv1 weight gradient (3->32, k8, s4, 100x100 -> 24x24)
# dW[o, c, kh, kw] = sum_{n, oh, ow} gy[n, o, oh, ow] * x[n, c, 4 oh + kh, 4 ow + kw]. For fixed (n, c, kh, oh) the
# im2col block over 8 consecutive ow and the 8 kw is an 8x8 matrix with row stride S=4 and unit column stride, so it is
# read straight from the input with simdgroup_load (no im2col buffer, no threadgroup memory, no barriers). Each
# threadgroup (8 simdgroups) reduces SPT samples into a (32 x 192) partial; a second kernel sums the partials.
_C1 = dict(O=32, C=3, K=8, S=4, HI=100, WI=100, HO=24, WO=24, SPT=4, JT=2, VARIANT="direct")   # JT/SPT: measured sweep
_C1_SRC = _HDR + """
constant constexpr uint O = {O}, C = {C}, K = {K}, S = {S}, HI = {HI}, WI = {WI}, HO = {HO}, WO = {WO};
constant constexpr uint P = HO * WO, KK = C * K * K, SPT = {SPT}, JT = {JT};

kernel void conv1_wgrad_partial(device const float* x [[buffer(0)]],      // (N, C, HI, WI)
                                device const float* gy [[buffer(1)]],     // (N, O, HO, WO)
                                device float* part [[buffer(2)]],         // (G, O, KK)
                                constant long& N [[buffer(3)]],
                                uint tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {{
    // simdgroup sg owns column tiles j = sg*JT .. sg*JT+JT-1, j = c*K + kh (8 kw columns each), and all 4 row tiles (o)
    simdgroup_float8x8 acc[4][JT];
    for (uint i = 0; i < 4; ++i) for (uint j = 0; j < JT; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    uint xoff[JT];
    for (uint j = 0; j < JT; ++j) {{ const uint jj = sg * JT + j; xoff[j] = ((jj / K) * HI + (jj % K)) * WI; }}
    for (uint s = 0; s < SPT; ++s) {{
        const long n = (long)tg * SPT + s;
        if (n >= N) break;
        device const float* xn = x + n * (C * HI * WI);
        device const float* gn = gy + n * (O * P);
        for (uint oh = 0; oh < HO; ++oh) {{
            for (uint ow0 = 0; ow0 < WO; ow0 += 8) {{
                simdgroup_float8x8 a[4], b[JT];
                for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], gn + (i * 8) * P + oh * WO + ow0, P);
                for (uint j = 0; j < JT; ++j) simdgroup_load(b[j], xn + xoff[j] + (oh * S) * WI + ow0 * S, S);
                for (uint i = 0; i < 4; ++i)
                    for (uint j = 0; j < JT; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }}
        }}
    }}
    device float* out = part + (long)tg * (O * KK);
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < JT; ++j) simdgroup_store(acc[i][j], out + (i * 8) * KK + (sg * JT + j) * 8, KK);
}}

// Variant: software-pipelined direct loads (next step's tiles are loaded before this step's MMAs).
kernel void conv1_wgrad_partial_pf(device const float* x [[buffer(0)]],
                                   device const float* gy [[buffer(1)]],
                                   device float* part [[buffer(2)]],
                                   constant long& N [[buffer(3)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {{
    simdgroup_float8x8 acc[4][JT];
    for (uint i = 0; i < 4; ++i) for (uint j = 0; j < JT; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    uint xoff[JT];
    for (uint j = 0; j < JT; ++j) {{ const uint jj = sg * JT + j; xoff[j] = ((jj / K) * HI + (jj % K)) * WI; }}
    const long n0 = (long)tg * SPT;
    const uint ns = (uint)min((long)SPT, N - n0);
    const uint steps = ns * HO * (WO / 8);
    simdgroup_float8x8 a[4], b[JT], an[4], bn[JT];
    // step t -> (s, oh, ow0)
    {{
        device const float* xn = x + n0 * (C * HI * WI);
        device const float* gn = gy + n0 * (O * P);
        for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], gn + (i * 8) * P, P);
        for (uint j = 0; j < JT; ++j) simdgroup_load(b[j], xn + xoff[j], S);
    }}
    for (uint t = 0; t < steps; ++t) {{
        const uint t1 = min(t + 1, steps - 1);
        const uint s1 = t1 / (HO * (WO / 8)), r1 = t1 % (HO * (WO / 8)), oh1 = r1 / (WO / 8), ow1 = (r1 % (WO / 8)) * 8;
        device const float* xn = x + (n0 + s1) * (C * HI * WI);
        device const float* gn = gy + (n0 + s1) * (O * P);
        for (uint i = 0; i < 4; ++i) simdgroup_load(an[i], gn + (i * 8) * P + oh1 * WO + ow1, P);
        for (uint j = 0; j < JT; ++j) simdgroup_load(bn[j], xn + xoff[j] + (oh1 * S) * WI + ow1 * S, S);
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < JT; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        for (uint i = 0; i < 4; ++i) a[i] = an[i];
        for (uint j = 0; j < JT; ++j) b[j] = bn[j];
    }}
    device float* out = part + (long)tg * (O * KK);
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < JT; ++j) simdgroup_store(acc[i][j], out + (i * 8) * KK + (sg * JT + j) * 8, KK);
}}

// Variant: per (sample, output row) stage the 24 input rows (c, kh) and the gy row block in threadgroup memory with
// coalesced loads, then run the same simdgroup GEMM from threadgroup memory.
kernel void conv1_wgrad_partial_tg(device const float* x [[buffer(0)]],
                                   device const float* gy [[buffer(1)]],
                                   device float* part [[buffer(2)]],
                                   constant long& N [[buffer(3)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {{
    constexpr uint NT = 32 * (C * K / JT);
    threadgroup float Xs[C * K * WI + 8];     // rows (c, kh) of the input for this output row, + slack
    threadgroup float Gs[O * WO];             // gy[n, :, oh, :]
    simdgroup_float8x8 acc[4][JT];
    for (uint i = 0; i < 4; ++i) for (uint j = 0; j < JT; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    for (uint s = 0; s < SPT; ++s) {{
        const long n = (long)tg * SPT + s;
        if (n >= N) break;
        device const float* xn = x + n * (C * HI * WI);
        device const float* gn = gy + n * (O * P);
        for (uint oh = 0; oh < HO; ++oh) {{
            for (uint i = tid; i < C * K * WI; i += NT) {{
                const uint r = i / WI, col = i % WI, c = r / K, kh = r % K;
                Xs[i] = xn[(c * HI + oh * S + kh) * WI + col];
            }}
            for (uint i = tid; i < O * WO; i += NT) {{
                const uint o = i / WO, ow = i % WO;
                Gs[i] = gn[o * P + oh * WO + ow];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ow0 = 0; ow0 < WO; ow0 += 8) {{
                simdgroup_float8x8 a[4], b[JT];
                for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], Gs + (i * 8) * WO + ow0, WO);
                for (uint j = 0; j < JT; ++j) simdgroup_load(b[j], Xs + (sg * JT + j) * WI + ow0 * S, S);
                for (uint i = 0; i < 4; ++i)
                    for (uint j = 0; j < JT; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}
    device float* out = part + (long)tg * (O * KK);
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < JT; ++j) simdgroup_store(acc[i][j], out + (i * 8) * KK + (sg * JT + j) * 8, KK);
}}

kernel void conv1_wgrad_reduce(device const float* part [[buffer(0)]],
                               device float* dw [[buffer(1)]],
                               constant long& G [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {{
    if (i >= O * KK) return;
    float s = 0.0f;
    for (long g = 0; g < G; ++g) s += part[g * (O * KK) + i];
    dw[i] = s;
}}
"""

# ---------------------------------------------------------------- conv2 input gradient (32->64, k4, s2, 24x24 -> 11x11)
# For input parity (py, px) and input pixel (2a+py, 2b+px), a, b in 0..11:
#   dx = sum_{ty,tx in 0..1, o} gy[n, o, a-ty, b-tx] * W[o, c, py+2ty, px+2tx].
# With gy zero-padded to 13x13 (gyp[.., r+1, s+1] = gy[.., r, s]) and q = 13 a + b, the gy index is q + off(ty,tx),
# off = 13 (1-ty) + (1-tx): an (8 q x 8 o) A tile is a transposed simdgroup_load straight from gyp (q contiguous, o
# stride 169). Each simdgroup computes 16 q (2 tiles, of 156 = 12 x 13 per sample; column b = 12 and rows a = 12 are
# discarded) x 32 c for one sample and parity, K = 4 taps x 64 o, B tiles from wp[(py,px), (ty,tx,o), c].
_C2 = dict(O=64, C=32, HI=24, WI=24, HO=11, WO=11)
_C2_SRC = _HDR + """
constant constexpr uint O = {O}, C = {C}, HI = {HI}, WI = {WI}, HO = {HO}, WO = {WO};
constant constexpr uint PW = HO + 2, PP = PW * PW, AH = HI / 2, AW = WI / 2, QM = AH * PW, QPAIRS = (QM + 15) / 16;

kernel void conv2_dgrad(device const float* gyp [[buffer(0)]],  // (N, O, PW, PW) zero-padded, plus slack
                        device const float* wp [[buffer(1)]],   // (4 parities, 4 taps * O, C)
                        device float* dx [[buffer(2)]],         // (N, C, HI, WI)
                        constant long& N [[buffer(3)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {{
    threadgroup float Os[4][16 * C];
    const uint par = tgp.y, py = par >> 1, px = par & 1;
    const long gid = (long)tgp.x * 4 + sg;
    const long n = gid / QPAIRS;
    const uint q0 = (uint)(gid % QPAIRS) * 16;
    if (n >= N) return;                                   // whole simdgroup; no threadgroup barriers below
    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    device const float* gn = gyp + n * (O * PP);
    device const float* wpp = wp + par * (4 * O * C);
    for (uint tap = 0; tap < 4; ++tap) {{
        const uint off = PW * (1 - (tap >> 1)) + (1 - (tap & 1));
        for (uint o0 = 0; o0 < O; o0 += 8) {{
            simdgroup_float8x8 a[2], b[4];
            for (uint i = 0; i < 2; ++i) simdgroup_load(a[i], gn + o0 * PP + q0 + i * 8 + off, PP, ulong2(0, 0), true);
            for (uint j = 0; j < 4; ++j) simdgroup_load(b[j], wpp + (tap * O + o0) * C + j * 8, C);
            for (uint i = 0; i < 2; ++i)
                for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }}
    }}
    threadgroup float* os = Os[sg];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j) simdgroup_store(acc[i][j], os + (i * 8) * C + j * 8, C);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 16 * C; e += 32) {{
        const uint qq = e % 16, c = e / 16, q = q0 + qq, a = q / PW, b = q % PW;
        if (a < AH && b < AW) dx[((n * C + c) * HI + 2 * a + py) * WI + 2 * b + px] = os[qq * C + c];
    }}
}}
"""

# ---------------------------------------------------------------- stride-1 weight gradient (conv3 directly, conv2 after
# space-to-depth). dW[o, tap, c'] = sum_{n,q} gyp[n, o, q] * x[n, c', q + ty*XW + tx]: with the output gradient
# zero-padded to the input's row width XW (q = oh*XW + ow, the extra columns are 0) the shifted input window is affine
# in q, so an (8 q x 8 c') B tile is a transposed simdgroup_load from x (c' stride XP, q contiguous). Split-K over
# samples (SPT per threadgroup) and split-N over column-tile pairs (grid y); each simdgroup: 4 o-tiles x 2 col tiles.
_WS1_SRC = _HDR + """
constant constexpr uint O = {O}, CP = {CP}, TH = {TH}, TW = {TW}, XW = {XW}, XP = {XP}, GQ = {GQ}, SPT = {SPT}, NSG = {NSG};
constant constexpr uint T = TH * TW, QT = GQ / 8, CPT = CP / 8, NCT = CPT * T, NPAIRS = NCT / 2, OQ = O / 32;

kernel void wgrad_s1(device const float* x [[buffer(0)]],      // (N, CP, XP) + zeroed slack
                     device const float* gyp [[buffer(1)]],    // (N, O, GQ) zero-padded output gradient
                     device float* part [[buffer(2)]],         // (G, O, T * CP)
                     constant long& N [[buffer(3)]],
                     uint2 tgp [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {{
    const uint oq = sg % OQ, pair = tgp.y * (NSG / OQ) + sg / OQ;
    if (pair >= NPAIRS) return;
    uint boff[2];
    for (uint j = 0; j < 2; ++j) {{
        const uint ct = pair * 2 + j, tap = ct / CPT, cpt = ct % CPT;
        boff[j] = cpt * 8 * XP + (tap / TW) * XW + (tap % TW);
    }}
    simdgroup_float8x8 acc[4][2];
    for (uint i = 0; i < 4; ++i) for (uint j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    for (uint s = 0; s < SPT; ++s) {{
        const long n = (long)tgp.x * SPT + s;
        if (n >= N) break;
        device const float* gn = gyp + (n * O + oq * 32) * GQ;
        device const float* xn = x + n * (CP * XP);
        for (uint q0 = 0; q0 < GQ; q0 += 8) {{
            simdgroup_float8x8 a[4], b[2];
            for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], gn + (i * 8) * GQ + q0, GQ);
            for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], xn + boff[j] + q0, XP, ulong2(0, 0), true);
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }}
    }}
    device float* out = part + (long)tgp.x * (O * T * CP);
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], out + (oq * 32 + i * 8) * (T * CP) + (pair * 2 + j) * 8, T * CP);
}}

kernel void reduce_part(device const float* part [[buffer(0)]],
                        device float* dw [[buffer(1)]],
                        constant long& G [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {{
    if (i >= O * T * CP) return;
    float s = 0.0f;
    for (long g = 0; g < G; ++g) s += part[g * (O * T * CP) + i];
    dw[i] = s;
}}
"""
_WS1 = {"c2": dict(O=64, CP=128, TH=2, TW=2, XW=12, XP=144, GQ=136, SPT=4, NSG=8),     # conv2 after space-to-depth
        "c3": dict(O=64, CP=64, TH=3, TW=3, XW=11, XP=121, GQ=104, SPT=4, NSG=8)}      # conv3 directly


def _wgrad_s1(which, xflat, gyp, N, spt=None):
    p = dict(_WS1[which]); p["SPT"] = spt or p["SPT"]
    key = ("ws1", tuple(sorted(p.items())))
    if key not in _LIB:
        _LIB[key] = torch.mps.compile_shader(_WS1_SRC.format(**p))
    lib = _LIB[key]
    T = p["TH"] * p["TW"]
    G = (N + p["SPT"] - 1) // p["SPT"]
    part = torch.empty(G, p["O"], T * p["CP"], device=gyp.device, dtype=torch.float32)
    out = torch.empty(p["O"], T, p["CP"], device=gyp.device, dtype=torch.float32)
    ngroups = (p["CP"] // 8 * T // 2 + p["NSG"] // (p["O"] // 32) - 1) // (p["NSG"] // (p["O"] // 32))
    lib.wgrad_s1(xflat, gyp, part, N, threads=(G * p["NSG"] * 32, ngroups), group_size=(p["NSG"] * 32, 1))
    lib.reduce_part(part, out, G, threads=out.numel(), group_size=256)
    return out


def _pad_gy(gy, width, gq):
    """(N, O, H, W) -> (N, O, gq) with rows of `width` (zero columns W..width-1) and zero tail."""
    N, O, Hh, Ww = gy.shape
    gyp = torch.zeros(N, O, gq, device=gy.device, dtype=gy.dtype)
    gyp[:, :, :Hh * width].view(N, O, Hh, width)[:, :, :, :Ww] = gy
    return gyp


def _with_slack(n_elems, like):
    buf = torch.empty(n_elems + 64, device=like.device, dtype=torch.float32)
    buf[n_elems:].zero_()
    return buf


@torch.library.custom_op("metalsim::conv2_weight_grad", mutates_args=())
def conv2_weight_grad(x: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """dW (64,32,4,4) of conv2d(x, W, stride=2), x (N,32,24,24), gy (N,64,11,11): space-to-depth -> stride-1 k2."""
    N = x.shape[0]
    buf = _with_slack(N * 128 * 144, x)
    buf[:N * 128 * 144].view(N, 32, 2, 2, 12, 12).copy_(x.view(N, 32, 12, 2, 12, 2).permute(0, 1, 3, 5, 2, 4))
    out = _wgrad_s1("c2", buf, _pad_gy(gy.contiguous(), 12, 136), N)          # (o, (ty,tx), (c,ry,rx))
    return out.view(64, 2, 2, 32, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(64, 32, 4, 4).contiguous()


@conv2_weight_grad.register_fake
def _(x, gy):
    return x.new_empty(64, 32, 4, 4)


@torch.library.custom_op("metalsim::conv3_weight_grad", mutates_args=())
def conv3_weight_grad(x: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """dW (64,64,3,3) of conv2d(x, W), x (N,64,11,11), gy (N,64,9,9)."""
    N = x.shape[0]
    buf = _with_slack(N * 64 * 121, x)
    buf[:N * 64 * 121].view_as(x).copy_(x)
    out = _wgrad_s1("c3", buf, _pad_gy(gy.contiguous(), 11, 104), N)          # (o, (ty,tx), c)
    return out.view(64, 3, 3, 64).permute(0, 3, 1, 2).contiguous()


@conv3_weight_grad.register_fake
def _(x, gy):
    return x.new_empty(64, 64, 3, 3)


# ---------------------------------------------------------------- conv1 forward (3->32, k8, s4, 100x100 -> 24x24)
# y[n, o, oh, ow] = b[o] + sum_{c,kh,kw} x[n, c, 4oh+kh, 4ow+kw] W[o, c, kh, kw]: per (n, oh) a (24 ow x 192) @ (192 x 32)
# GEMM; the A tile (8 ow x 8 kw) for fixed (c, kh) is a row-stride-4 simdgroup_load from x, the B tile a transposed
# load of W (o stride 192). Each simdgroup: one (n, oh), 3 ow tiles x 2 o tiles (o half from the simdgroup index);
# accumulators start from the bias; the output tile is stored transposed into NCHW.
_C1F = dict(O=32, C=3, K=8, S=4, HI=100, WI=100, HO=24, WO=24)
_C1F_SRC = _HDR + """
constant constexpr uint O = {O}, C = {C}, K = {K}, S = {S}, HI = {HI}, WI = {WI}, HO = {HO}, WO = {WO}, KK = C * K * K;

kernel void conv1_fwd(device const float* x [[buffer(0)]],      // (N, C, HI, WI)
                      device const float* w [[buffer(1)]],      // (O, C, K, K)
                      device const float* bm [[buffer(2)]],     // (8, O): bias repeated over 8 rows
                      device float* y [[buffer(3)]],            // (N, O, HO, WO)
                      constant long& N [[buffer(4)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {{
    const long gid = (long)tg * 8 + sg;
    const long n = gid / (HO * 2);
    const uint rem = (uint)(gid % (HO * 2)), oh = rem / 2, oh2 = rem % 2;
    if (n >= N) return;
    simdgroup_float8x8 acc[3][2];
    for (uint j = 0; j < 2; ++j) {{
        simdgroup_float8x8 bt;
        simdgroup_load(bt, bm + (oh2 * 2 + j) * 8, O);
        for (uint i = 0; i < 3; ++i) acc[i][j] = bt;
    }}
    device const float* xn = x + n * (C * HI * WI);
    for (uint ck = 0; ck < C * K; ++ck) {{                 // (c, kh)
        const uint c = ck / K, kh = ck % K;
        simdgroup_float8x8 a[3], b[2];
        for (uint i = 0; i < 3; ++i) simdgroup_load(a[i], xn + (c * HI + oh * S + kh) * WI + (i * 8) * S, S);
        for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], w + ((oh2 * 2 + j) * 8) * KK + ck * K, KK, ulong2(0, 0), true);
        for (uint i = 0; i < 3; ++i)
            for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }}
    for (uint i = 0; i < 3; ++i)
        for (uint j = 0; j < 2; ++j)
            simdgroup_store(acc[i][j], y + ((n * O + (oh2 * 2 + j) * 8) * HO + oh) * WO + i * 8, HO * WO, ulong2(0, 0), true);
}}
"""


@torch.library.custom_op("metalsim::conv1_forward", mutates_args=())
def conv1_forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """conv2d(x, w, b, stride=4) for x (N,3,100,100), w (32,3,8,8)."""
    N = x.shape[0]
    key = ("c1f",)
    if key not in _LIB:
        _LIB[key] = torch.mps.compile_shader(_C1F_SRC.format(**_C1F))
    y = torch.empty(N, 32, 24, 24, device=x.device, dtype=torch.float32)
    bm = b.reshape(1, 32).expand(8, 32).contiguous()
    tgs = (N * 48 + 7) // 8
    _LIB[key].conv1_fwd(x.contiguous(), w.contiguous(), bm, y, N, threads=tgs * 256, group_size=256)
    return y


@conv1_forward.register_fake
def _(x, w, b):
    return x.new_empty(x.shape[0], 32, 24, 24)


# Which custom kernels are active. Measured whole-update effect on the M4 Max (runs/camera_update/profile_4c.log,
# separate processes, 6 reps): c1_wgrad + c2_dgrad 3.68 s; + c1_fwd 3.58 s (-3 %); gather_in_graph then 3.20 s;
# + c3_wgrad 3.15 s (-1.6 %). c2_wgrad is slower than MPSGraph once its space-to-depth copy (2.4 ms) and gy padding
# (0.9 ms) are counted (4.9 vs 3.2 ms; the kernel itself is 1.9 ms), so it is off; c3_wgrad (2.5 vs 2.9 ms incl. its
# copy) is below the 5 % bar and off. Both stay available and tested.
ENABLED = {"c1_wgrad": True, "c2_dgrad": True, "c2_wgrad": False, "c3_wgrad": False, "c1_fwd": True}

_LIB = {}


def _lib(name, **over):
    key = (name, tuple(sorted(over.items())))
    if key not in _LIB:
        src = {"c1": _C1_SRC.format(**{**_C1, **over}), "c2": _C2_SRC.format(**{**_C2, **over})}[name]
        _LIB[key] = torch.mps.compile_shader(src)
    return _LIB[key]


@torch.library.custom_op("metalsim::conv1_weight_grad", mutates_args=())
def conv1_weight_grad(x: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    return _conv1_weight_grad(x, gy)


def _conv1_weight_grad(x, gy, spt=None, jt=None, variant=None):
    """dW (32,3,8,8) of conv2d(x, W, stride=4) for x (N,3,100,100), gy (N,32,24,24), fp32 contiguous."""
    N = x.shape[0]
    x = x.contiguous(); gy = gy.contiguous()
    spt = spt or _C1["SPT"]; jt = jt or _C1["JT"]
    G = (N + spt - 1) // spt
    O, KK = _C1["O"], _C1["C"] * _C1["K"] ** 2
    part = torch.empty(G, O, KK, device=x.device, dtype=torch.float32)
    dw = torch.empty(O, _C1["C"], _C1["K"], _C1["K"], device=x.device, dtype=torch.float32)
    lib = _lib("c1", SPT=spt, JT=jt)
    tgsize = 32 * (_C1["C"] * _C1["K"] // jt)
    kern = {"tg": lib.conv1_wgrad_partial_tg, "pf": lib.conv1_wgrad_partial_pf,
            "direct": lib.conv1_wgrad_partial}[variant or _C1["VARIANT"]]
    kern(x, gy, part, N, threads=G * tgsize, group_size=tgsize)
    lib.conv1_wgrad_reduce(part, dw, G, threads=O * KK, group_size=256)
    return dw


@conv1_weight_grad.register_fake
def _(x, gy):
    return x.new_empty(_C1["O"], _C1["C"], _C1["K"], _C1["K"])


@torch.library.custom_op("metalsim::conv2_input_grad", mutates_args=())
def conv2_input_grad(gy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """dx (N,32,24,24) of conv2d(x, w, stride=2) for w (64,32,4,4), gy (N,64,11,11), fp32 contiguous."""
    N = gy.shape[0]
    O, C, HO = _C2["O"], _C2["C"], _C2["HO"]
    PW = HO + 2
    # zero-padded gy with slack: the last tiles read up to 2*PW + 16 elements past a channel's padded 13x13 slab
    flat = torch.zeros(N * O * PW * PW + 64, device=gy.device, dtype=torch.float32)
    flat[:N * O * PW * PW].view(N, O, PW, PW)[:, :, 1:HO + 1, 1:HO + 1] = gy
    # wp[(py,px), (ty,tx,o), c] = w[o, c, py + 2ty, px + 2tx]
    wp = w.reshape(O, C, 2, 2, 2, 2).permute(3, 5, 2, 4, 0, 1).reshape(4, 4 * O, C).contiguous()
    dx = torch.empty(N, C, _C2["HI"], _C2["WI"], device=gy.device, dtype=torch.float32)
    qpairs = ((_C2["HI"] // 2) * PW + 15) // 16
    tgs = (N * qpairs + 3) // 4
    _lib("c2").conv2_dgrad(flat, wp, dx, N, threads=(tgs * 128, 4), group_size=(128, 1))
    return dx


@conv2_input_grad.register_fake
def _(gy, w):
    return gy.new_empty(gy.shape[0], _C2["C"], _C2["HI"], _C2["WI"])


def _c1_ok(x, w, stride):
    return (x.device.type == "mps" and x.dtype == torch.float32 and w.dtype == torch.float32 and tuple(w.shape) == (32, 3, 8, 8)
            and tuple(x.shape[1:]) == (3, 100, 100) and stride == 4)


def _c2_ok(x, w, stride):
    return (x.device.type == "mps" and x.dtype == torch.float32 and w.dtype == torch.float32 and tuple(w.shape) == (64, 32, 4, 4)
            and tuple(x.shape[1:]) == (32, 24, 24) and stride == 2)


def _aten_bwd(gy, x, w, stride, mask):
    return torch.ops.aten.convolution_backward(gy, x, w, [w.shape[0]], [stride, stride], [0, 0], [1, 1], False, [0, 0], 1, mask)


class Conv1WGrad(torch.autograd.Function):
    """conv2d(x, w, b, stride) whose weight gradient comes from the custom kernel (input grad via aten if needed)."""

    @staticmethod
    def forward(ctx, x, w, b, stride):
        ctx.save_for_backward(x, w); ctx.stride = stride
        if ENABLED["c1_fwd"] and b is not None:
            return conv1_forward(x, w, b)
        return F.conv2d(x, w, b, stride=stride)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        gy = gy.contiguous()
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        dx = _aten_bwd(gy, x, w, ctx.stride, [True, False, False])[0] if need_x else None
        if need_w:
            dw = conv1_weight_grad(x, gy) if ENABLED["c1_wgrad"] else _aten_bwd(gy, x, w, ctx.stride, [False, True, False])[1]
        else:
            dw = None
        db = gy.sum(dim=(0, 2, 3)) if need_b else None
        return dx, dw, db, None


class Conv2DGrad(torch.autograd.Function):
    """conv2d(x, w, b, stride) whose input gradient comes from the custom kernel (weight/bias grads via aten)."""

    @staticmethod
    def forward(ctx, x, w, b, stride):
        ctx.save_for_backward(x, w); ctx.stride = stride
        return F.conv2d(x, w, b, stride=stride)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        gy = gy.contiguous()
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        if need_x:
            dx = conv2_input_grad(gy, w) if ENABLED["c2_dgrad"] else _aten_bwd(gy, x, w, 2, [True, False, False])[0]
        else:
            dx = None
        dw = db = None
        if need_w:
            dw = conv2_weight_grad(x, gy) if ENABLED["c2_wgrad"] else _aten_bwd(gy, x, w, 2, [False, True, False])[1]
        if need_b:
            db = gy.sum(dim=(0, 2, 3))
        return dx, dw, db, None


class Conv3WGrad(torch.autograd.Function):
    """conv2d(x, w, b) (stride 1) whose weight gradient comes from the custom kernel (input grad via aten)."""

    @staticmethod
    def forward(ctx, x, w, b, stride):
        ctx.save_for_backward(x, w)
        return F.conv2d(x, w, b, stride=stride)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        gy = gy.contiguous()
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        dx = _aten_bwd(gy, x, w, 1, [True, False, False])[0] if need_x else None
        if need_w:
            dw = conv3_weight_grad(x, gy) if ENABLED["c3_wgrad"] else _aten_bwd(gy, x, w, 1, [False, True, False])[1]
        else:
            dw = None
        db = gy.sum(dim=(0, 2, 3)) if need_b else None
        return dx, dw, db, None


def _c3_ok(x, w, stride):
    return (x.device.type == "mps" and x.dtype == torch.float32 and w.dtype == torch.float32 and tuple(w.shape) == (64, 64, 3, 3)
            and tuple(x.shape[1:]) == (64, 11, 11) and stride == 1)


def conv3(x, w, b, stride):
    return Conv3WGrad.apply(x, w, b, stride) if _c3_ok(x, w, stride) else F.conv2d(x, w, b, stride=stride)


def conv1(x, w, b, stride):
    return Conv1WGrad.apply(x, w, b, stride) if _c1_ok(x, w, stride) else F.conv2d(x, w, b, stride=stride)


def conv2(x, w, b, stride):
    return Conv2DGrad.apply(x, w, b, stride) if _c2_ok(x, w, stride) else F.conv2d(x, w, b, stride=stride)
