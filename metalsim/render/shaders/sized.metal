// Per-world geom sizes for the renderers: the rough terrain's box slots (metalsim.learn.terrain.BoxWindow) are one
// unit box mesh drawn at each world's geom_size. A copy of geom_xmat with the size folded into the columns
// (R * diag(size)) replaces the physics xmat for these geoms; the others are copied unchanged. Raster (tier 0) and
// the ray-tracing instance refit (tier 1 / 2, lidar) read the copy, so no draw or refit shader changes: a box's face
// normals are axis-aligned, so R * diag(s) * n keeps R * n's direction (every shader normalizes).
#include <metal_stdlib>
using namespace metal;

struct SizedConsts { uint n_envs, n_geoms, size_world_stride, _pad; };

kernel void scale_geom_xmat(
    device const float*   xmat_in   [[buffer(0)]],   // (n_envs, n_geoms, 9) MuJoCo row-major
    device const float*   geom_size [[buffer(1)]],   // (n_size_worlds, n_geoms, 3)
    device const int*     sized     [[buffer(2)]],   // (n_geoms,) 1 = scale by geom_size
    device float*         xmat_out  [[buffer(3)]],
    constant SizedConsts& c         [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= c.n_envs * c.n_geoms) return;
    uint env = i / c.n_geoms, g = i - env * c.n_geoms;
    device const float* a = xmat_in + i * 9;
    device float* o = xmat_out + i * 9;
    if (sized[g] == 0) {
        for (uint k = 0; k < 9; ++k) o[k] = a[k];
        return;
    }
    device const float* s = geom_size + (env * c.size_world_stride * c.n_geoms + g) * 3;
    for (uint r = 0; r < 3; ++r)
        for (uint k = 0; k < 3; ++k) o[r * 3 + k] = a[r * 3 + k] * s[k];      // row-major: column k scaled by s[k]
}
