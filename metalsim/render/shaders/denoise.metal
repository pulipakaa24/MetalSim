// metalsim tier 2 post-processing: edge-avoiding a-trous denoiser and RTX-default tone mapping.
//
// atrous_step: one iteration of the edge-avoiding a-trous wavelet filter (Dammertz et al., HPG 2010) as used
// for the spatial pass of SVGF (Schied et al., HPG 2017), without SVGF's temporal accumulation and variance
// estimate (tier 2 accumulates progressively instead): 5x5 B3-spline taps at spacing `step`, weights from the
// first-hit normal (cos^128), z-depth (relative, scaled by the tap distance) and radiance luminance (relative).
// Iteration 0 divides radiance by the sample-averaged first-hit albedo (demodulation: texture/material detail is
// not blurred); the last iteration multiplies it back. All envs of the batch in one dispatch (x, y, env).
#include <metal_stdlib>
using namespace metal;

struct AtrousConsts { uint n_envs, width, height, step; uint first, last, _p0, _p1; };
struct TMConsts { uint n_envs, width, height, flags; float exposure, _f0, _f1, _f2; };

constant float KERNEL_B3[3] = {3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0};

inline float luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

kernel void atrous_step(
    constant AtrousConsts& k      [[buffer(0)]],
    device const float4*   src    [[buffer(1)]],   // (n, h, w) radiance (first) or demodulated irradiance
    device float4*         dst    [[buffer(2)]],
    device const float*    albedo [[buffer(3)]],   // (n, h, w, 3)
    device const float*    nrm    [[buffer(4)]],   // (n, h, w, 3)
    device const float*    depth  [[buffer(5)]],   // (n, h, w) z-depth, 0 = miss
    uint3 tid [[thread_position_in_grid]])
{
    uint x = tid.x, y = tid.y, e = tid.z;
    if (x >= k.width || y >= k.height || e >= k.n_envs) return;
    uint img = e * k.height * k.width;
    uint p = img + y * k.width + x;
    float3 ap = max(float3(albedo[p * 3], albedo[p * 3 + 1], albedo[p * 3 + 2]), float3(1e-3));
    float3 cp = src[p].xyz; if (k.first) cp /= ap;
    float3 np_ = float3(nrm[p * 3], nrm[p * 3 + 1], nrm[p * 3 + 2]); float nl = length(np_); np_ = nl > 0 ? np_ / nl : np_;
    float zp = depth[p];
    float lp = luma(cp);
    float3 sum = float3(0.0); float wsum = 0.0;
    int s = int(k.step);
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            int qx = int(x) + i * s, qy = int(y) + j * s;
            if (qx < 0 || qy < 0 || qx >= int(k.width) || qy >= int(k.height)) continue;
            uint q = img + uint(qy) * k.width + uint(qx);
            float3 cq = src[q].xyz;
            if (k.first) cq /= max(float3(albedo[q * 3], albedo[q * 3 + 1], albedo[q * 3 + 2]), float3(1e-3));
            float zq = depth[q];
            float w = KERNEL_B3[abs(i)] * KERNEL_B3[abs(j)];
            if ((zp > 0.0) != (zq > 0.0)) continue;                           // sky vs surface
            if (zp > 0.0) {
                float3 nq = float3(nrm[q * 3], nrm[q * 3 + 1], nrm[q * 3 + 2]); float ql = length(nq); nq = ql > 0 ? nq / ql : nq;
                w *= pow(max(dot(np_, nq), 0.0), 128.0);
                float dist = length(float2(i, j)) * float(s);
                w *= exp(-fabs(zp - zq) / (0.002 * zp * dist + 1e-4));
            }
            float lq = luma(cq);
            w *= exp(-fabs(lp - lq) / (0.25 * max(lp, lq) + 1e-4));
            sum += cq * w; wsum += w;
        }
    }
    float3 outc = wsum > 0.0 ? sum / wsum : cp;
    if (k.last) outc *= ap;
    dst[p] = float4(outc, 1.0);
}

inline float3 aces_fit(float3 x) { return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
inline float3 srgb_oetf(float3 c) { return select(1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308); }

kernel void tonemap_hdr(
    constant TMConsts&   k   [[buffer(0)]],
    device const float4* hdr [[buffer(1)]],
    device uchar*        rgb [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]])
{
    uint x = tid.x, y = tid.y, e = tid.z;
    if (x >= k.width || y >= k.height || e >= k.n_envs) return;
    uint p = (e * k.height + y) * k.width + x;
    float3 c = hdr[p].xyz * k.exposure;
    c = (k.flags & 4u) ? srgb_oetf(aces_fit(c)) : c;
    c = clamp(c, 0.0, 1.0);
    rgb[p * 3] = uchar(c.x * 255.0 + 0.5); rgb[p * 3 + 1] = uchar(c.y * 255.0 + 0.5); rgb[p * 3 + 2] = uchar(c.z * 255.0 + 0.5);
}
