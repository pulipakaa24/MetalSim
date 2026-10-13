// World X axes of the area lights' emitters per env (metalsim/render/area_lights.py): xmat of the light's body times
// the prim's X axis in that body's frame. Written after the light records in the renderer's light buffer.
#include <metal_stdlib>
using namespace metal;

kernel void area_axes(
    device const float*  xmat   [[buffer(0)]],   // (n_envs, nbody, 9) row-major
    device const int*    body   [[buffer(1)]],   // (n_light) body of each light
    device const float4* x_body [[buffer(2)]],   // (n_light) the emitter's X axis in its body frame
    device float4*       out    [[buffer(3)]],   // (n_envs, n_light)
    constant uint4&      k      [[buffer(4)]],   // n_envs, n_light, nbody, xmat has one world per env (1) or one (0)
    uint i [[thread_position_in_grid]])
{
    if (i >= k.x * k.y) return;
    uint e = i / k.y, l = i - e * k.y;
    device const float* m = xmat + ((ulong)(e * k.w) * k.z + uint(body[l])) * 9;
    float3 x = x_body[l].xyz;
    out[i] = float4(m[0] * x.x + m[1] * x.y + m[2] * x.z, m[3] * x.x + m[4] * x.y + m[5] * x.z,
                    m[6] * x.x + m[7] * x.y + m[8] * x.z, 0.0);
}
