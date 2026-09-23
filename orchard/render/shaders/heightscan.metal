#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

inline float3x3 load_mat33(device const float* p) {
    return float3x3(float3(p[0], p[3], p[6]), float3(p[1], p[4], p[7]), float3(p[2], p[5], p[8]));
}

// Isaac RayCaster height scan: grid points in the body's yaw frame, rays straight down from 20 m,
// output = body_z - hit_z - offset (0.5), or offset-based sentinel on miss.
kernel void height_scan(
    primitive_acceleration_structure accel [[buffer(0)]],
    device const float* xpos   [[buffer(1)]],   // (n_envs, nbody, 3)
    device const float* xmat   [[buffer(2)]],   // (n_envs, nbody, 9)
    device const float2* grid  [[buffer(3)]],
    constant uint4& c          [[buffer(4)]],   // n_envs, n_rays, nbody, body
    constant float4& p         [[buffer(5)]],   // offset_z, ray height
    device float* out          [[buffer(6)]],   // (n_envs, n_rays)
    uint2 tid [[thread_position_in_grid]])
{
    uint k = tid.x, e = tid.y;
    if (k >= c.y || e >= c.x) return;
    uint bi = e * c.z + c.w;
    float3 bp = float3(xpos[bi * 3], xpos[bi * 3 + 1], xpos[bi * 3 + 2]);
    float3x3 R = load_mat33(xmat + bi * 9);
    float yaw = atan2(R[0][1], R[0][0]);   // column-major: R[0] is the first column
    float2 g = grid[k];
    float2 w = float2(cos(yaw) * g.x - sin(yaw) * g.y, sin(yaw) * g.x + cos(yaw) * g.y);
    float3 origin = float3(bp.x + w.x, bp.y + w.y, bp.z + p.y);
    ray r(origin, float3(0, 0, -1), 0.0, p.y + 50.0);
    intersector<triangle_data> isect;
    intersection_result<triangle_data> res = isect.intersect(r, accel);
    float hit_z = (res.type == intersection_type::none) ? (bp.z - 10.0) : (origin.z - res.distance);
    out[e * c.y + k] = bp.z - hit_z - p.x;
}
