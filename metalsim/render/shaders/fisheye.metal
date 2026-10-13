// Tier-0 fisheye (metalsim.render.fisheye.FisheyeTier0): resample the cube-face renders of each env into the lens
// image. Per output pixel a host-built table gives the face, the continuous face pixel coordinates and the z-depth
// factor (lens z-depth = face z-depth x factor); colour is bilinear inside the face, depth / segmentation / normal
// nearest. Pixels outside the lens field of view: black, depth 0, segmentation 0.
#include <metal_stdlib>
using namespace metal;

struct FishConsts { uint n_envs, width, height, face_res; uint face, out_mask, first, _p1; };   // one dispatch per face
struct FishLut { int face; float u, v, kz; };

kernel void fisheye_resample(
    device const FishLut* lut       [[buffer(0)]],    // (height, width)
    device const uchar*   face_rgb  [[buffer(1)]],    // this face's renderer outputs: (n_envs, fr, fr, 3)
    device const float*   face_dep  [[buffer(2)]],    // (n_envs, fr, fr)
    device const int*     face_seg  [[buffer(3)]],
    device const float*   face_nrm  [[buffer(4)]],    // (n_envs, fr, fr, 3)
    device uchar*         out_rgb   [[buffer(5)]],
    device float*         out_dep   [[buffer(6)]],
    device int*           out_seg   [[buffer(7)]],
    device float*         out_nrm   [[buffer(8)]],
    constant FishConsts&  c         [[buffer(9)]],
    uint3 tid [[thread_position_in_grid]])
{
    uint x = tid.x, y = tid.y, e = tid.z;
    if (x >= c.width || y >= c.height || e >= c.n_envs) return;
    uint pix = (e * c.height + y) * c.width + x;
    FishLut L = lut[y * c.width + x];
    uint fr = c.face_res;
    if (L.face < 0 && c.first) {
        if (c.out_mask & 1u) { out_rgb[pix * 3] = 0; out_rgb[pix * 3 + 1] = 0; out_rgb[pix * 3 + 2] = 0; }
        if (c.out_mask & 2u) out_dep[pix] = 0.0;
        if (c.out_mask & 4u) out_seg[pix] = 0;
        if (c.out_mask & 8u) { out_nrm[pix * 3] = 0.0; out_nrm[pix * 3 + 1] = 0.0; out_nrm[pix * 3 + 2] = 0.0; }
        return;
    }
    if (L.face != int(c.face)) return;
    uint base = e * fr * fr;
    uint nx = min(uint(max(L.u, 0.0f)), fr - 1), ny = min(uint(max(L.v, 0.0f)), fr - 1);
    uint np_ = base + ny * fr + nx;
    if (c.out_mask & 1u) {
        float fx = clamp(L.u - 0.5f, 0.0f, float(fr - 1)), fy = clamp(L.v - 0.5f, 0.0f, float(fr - 1));
        uint x0 = uint(fx), y0 = uint(fy), x1 = min(x0 + 1, fr - 1), y1 = min(y0 + 1, fr - 1);
        float ax = fx - float(x0), ay = fy - float(y0);
        for (uint k = 0; k < 3; ++k) {
            float v00 = face_rgb[(base + y0 * fr + x0) * 3 + k], v01 = face_rgb[(base + y0 * fr + x1) * 3 + k];
            float v10 = face_rgb[(base + y1 * fr + x0) * 3 + k], v11 = face_rgb[(base + y1 * fr + x1) * 3 + k];
            float v = mix(mix(v00, v01, ax), mix(v10, v11, ax), ay);
            out_rgb[pix * 3 + k] = uchar(clamp(v + 0.5f, 0.0f, 255.0f));
        }
    }
    if (c.out_mask & 2u) out_dep[pix] = face_dep[np_] * L.kz;
    if (c.out_mask & 4u) out_seg[pix] = face_seg[np_];
    if (c.out_mask & 8u) for (uint k = 0; k < 3; ++k) out_nrm[pix * 3 + k] = face_nrm[np_ * 3 + k];
}
