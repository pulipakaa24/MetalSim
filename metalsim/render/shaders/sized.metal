// Per-world geom sizes for the renderers. Sized geoms are drawn from one mesh (a unit primitive, or for capsules the
// compiled mesh) scaled per world by MuJoCo Warp's geom_size: a copy of geom_xmat with the scale folded into the
// columns (R * diag(s)) replaces the physics xmat for these geoms; the others are copied unchanged. Raster (tier 0) and
// the ray-tracing instance refit (tier 1 / 2, lidar) read the copy. Normals: every consumer transforms a mesh normal by
// R' * (n / s^2), s = the column norms of R' = R diag(s) (the inverse transpose of R diag(s), up to scale), so
// non-uniform scales (ellipsoids, boxes, cylinders) shade correctly.
// Modes (scene_tables.SIZE_*): 1 diag(size) (box, ellipsoid, terrain slots), 2 sphere (r, r, r), 3 cylinder (r, r, h),
// 4 capsule (r / r0, r / r0, (h + r) / (h0 + r0)) of the compiled mesh, 5 finite plane (sx, sy, 1).
#include <metal_stdlib>
using namespace metal;

struct SizedConsts { uint n_envs, n_geoms, size_world_stride, _pad; };

inline float3 size_scale(int mode, float3 s, float3 s0) {
    if (mode == 1) return s;
    if (mode == 2) return float3(s.x);
    if (mode == 3) return float3(s.x, s.x, s.y);
    if (mode == 4) return float3(s.x / s0.x, s.x / s0.x, (s.y + s.x) / (s0.y + s0.x));
    if (mode == 5) return float3(s.x, s.y, 1.0);
    return float3(1.0);
}

kernel void scale_geom_xmat(
    device const float*   xmat_in   [[buffer(0)]],   // (n_envs, n_geoms, 9) MuJoCo row-major
    device const float*   geom_size [[buffer(1)]],   // (n_size_worlds, n_geoms, 3)
    device const int*     sized     [[buffer(2)]],   // (n_geoms,) size mode, 0 = copy
    device float*         xmat_out  [[buffer(3)]],
    constant SizedConsts& c         [[buffer(4)]],
    device const float*   size0     [[buffer(5)]],   // (n_geoms, 3) compiled sizes (capsule reference)
    uint i [[thread_position_in_grid]])
{
    if (i >= c.n_envs * c.n_geoms) return;
    uint env = i / c.n_geoms, g = i - env * c.n_geoms;
    device const float* a = xmat_in + i * 9;
    device float* o = xmat_out + i * 9;
    int mode = sized[g];
    if (mode == 0) {
        for (uint k = 0; k < 9; ++k) o[k] = a[k];
        return;
    }
    device const float* sp = geom_size + (env * c.size_world_stride * c.n_geoms + g) * 3;
    float3 s = size_scale(mode, float3(sp[0], sp[1], sp[2]), float3(size0[g * 3], size0[g * 3 + 1], size0[g * 3 + 2]));
    for (uint r = 0; r < 3; ++r)
        for (uint k = 0; k < 3; ++k) o[r * 3 + k] = a[r * 3 + k] * s[k];      // row-major: column k scaled by s[k]
}
