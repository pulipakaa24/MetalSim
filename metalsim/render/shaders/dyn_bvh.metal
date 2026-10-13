// Per-env BVH of the dynamic objects ("sbvh" layout, metalsim/sensors/dyn_bvh.py): node bounds refit level by level
// (deepest first) for every env from the regenerated vertex records, then one bounding box per env (its root bounds
// placed at the env's grid offset) for the hardware structure. Topology (shared by every env) is built once on the host.
#include <metal_stdlib>
using namespace metal;
#include "env_grid.h"

struct RefitArgs { uint n_envs, n_nodes, D, mesh_base; uint level_off, level_count, n_tris, _p0; EnvGrid grid; };

// one thread per (node of this level, env)
kernel void dyn_bvh_refit(
    device const float*  verts       [[buffer(0)]],
    device const uint*   indices     [[buffer(1)]],
    device const uint4*  meshes      [[buffer(2)]],
    device const uint4*  topo        [[buffer(3)]],
    device const uint*   order       [[buffer(4)]],
    device float4*       bounds      [[buffer(5)]],
    device const uint*   level_nodes [[buffer(6)]],
    constant RefitArgs&  a           [[buffer(7)]],
    device float*        tris        [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint x = tid.x, e = tid.y;
    if (x >= a.level_count || e >= a.n_envs) return;
    uint node = level_nodes[a.level_off + x];
    uint4 n = topo[node];
    float3 lo = float3(3.0e38), hi = float3(-3.0e38);       // no infinities (fast math)
    device float4* B = bounds + (ulong)e * a.n_nodes * 2;
    if (n.w > 0) {
        for (uint j = n.z; j < n.z + n.w; ++j) {
            uint ent = order[j];
            uint k = ent >> 24, p = ent & 0xFFFFFFu;
            uint base = meshes[a.mesh_base + e * a.D + k].x + p * 3;
            device float* tw = tris + ((ulong)e * a.n_tris + j) * 9;
            for (uint q = 0; q < 3; ++q) {
                uint vi = indices[base + q];
                float3 v = float3(verts[vi * 8], verts[vi * 8 + 1], verts[vi * 8 + 2]);
                tw[q * 3] = v.x; tw[q * 3 + 1] = v.y; tw[q * 3 + 2] = v.z;
                lo = min(lo, v); hi = max(hi, v);
            }
        }
        float pad = 1e-6 * max(max(fabs(lo.x), fabs(hi.x)), max(max(fabs(lo.y), fabs(hi.y)), max(fabs(lo.z), fabs(hi.z)))) + 1e-7;
        lo -= pad; hi += pad;
    } else {
        lo = min(B[n.x * 2].xyz, B[n.y * 2].xyz);
        hi = max(B[n.x * 2 + 1].xyz, B[n.y * 2 + 1].xyz);
    }
    B[node * 2] = float4(lo, 0.0);
    B[node * 2 + 1] = float4(hi, 0.0);
}

// one thread per env: MTLAxisAlignedBoundingBox (min, max as packed floats) of its root at its grid offset
kernel void dyn_bvh_boxes(
    device const float4* bounds [[buffer(0)]],
    device float*        boxes  [[buffer(1)]],
    constant RefitArgs&  a      [[buffer(2)]],
    device atomic_uint*  grid_flag [[buffer(3)]],     // set when an env's dynamic objects near its cell boundary
    uint e [[thread_position_in_grid]])
{
    if (e >= a.n_envs) return;
    if (env_grid_left(bounds[(ulong)e * a.n_nodes * 2].xyz, a.grid) || env_grid_left(bounds[(ulong)e * a.n_nodes * 2 + 1].xyz, a.grid))
        atomic_store_explicit(grid_flag, 1u, memory_order_relaxed);
    float3 off = grid_offset(e, a.grid);
    float3 lo = bounds[(ulong)e * a.n_nodes * 2].xyz + off, hi = bounds[(ulong)e * a.n_nodes * 2 + 1].xyz + off;
    float pad = 2.5e-7 * max(max(fabs(off.x), fabs(off.y)), fabs(off.z)) + 1e-4;   // rounding of the offset sum
    lo -= pad; hi += pad;
    boxes[e * 6] = lo.x; boxes[e * 6 + 1] = lo.y; boxes[e * 6 + 2] = lo.z;
    boxes[e * 6 + 3] = hi.x; boxes[e * 6 + 4] = hi.y; boxes[e * 6 + 5] = hi.z;
}
