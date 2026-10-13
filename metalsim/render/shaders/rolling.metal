// Rolling shutter (metalsim.render.rolling): the pose inputs of a row band at time tau between the previous frame's
// state (tau = 0) and the current one (tau = 1), as Omniverse RTX's per-pixel shutter time ("0 = the start of the previous
// frame, 1 = the start of the current frame"). Positions are interpolated linearly, orientations by quaternion slerp
// (exact for a constant angular velocity), light directions linearly and renormalized.
#include <metal_stdlib>
using namespace metal;

struct InterpConsts { uint n_pos, n_mat, n_dir, _p; float tau, _a, _b, _c; };

inline float4 mat_to_quat(device const float* r) {   // MuJoCo row-major 3x3 -> (w, x, y, z), Shepperd's method
    float m00 = r[0], m01 = r[1], m02 = r[2], m10 = r[3], m11 = r[4], m12 = r[5], m20 = r[6], m21 = r[7], m22 = r[8];
    float tr = m00 + m11 + m22;
    float4 q;
    if (tr > 0.0f) {
        float s = sqrt(tr + 1.0f) * 2.0f;
        q = float4(0.25f * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s);
    } else if (m00 > m11 && m00 > m22) {
        float s = sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q = float4((m21 - m12) / s, 0.25f * s, (m01 + m10) / s, (m02 + m20) / s);
    } else if (m11 > m22) {
        float s = sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q = float4((m02 - m20) / s, (m01 + m10) / s, 0.25f * s, (m12 + m21) / s);
    } else {
        float s = sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q = float4((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25f * s);
    }
    return normalize(q);
}

inline float4 slerp(float4 a, float4 b, float t) {
    float d = dot(a, b);
    if (d < 0.0f) { b = -b; d = -d; }
    if (d > 0.9995f) return normalize(a + t * (b - a));
    float th = acos(clamp(d, -1.0f, 1.0f));
    float s = sin(th);
    return (sin((1.0f - t) * th) / s) * a + (sin(t * th) / s) * b;
}

inline void quat_to_rowmat(float4 q, device float* o) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    o[0] = 1 - 2 * (y * y + z * z); o[1] = 2 * (x * y - w * z);     o[2] = 2 * (x * z + w * y);
    o[3] = 2 * (x * y + w * z);     o[4] = 1 - 2 * (x * x + z * z); o[5] = 2 * (y * z - w * x);
    o[6] = 2 * (x * z - w * y);     o[7] = 2 * (y * z + w * x);     o[8] = 1 - 2 * (x * x + y * y);
}

// one thread per item: n_pos positions (3 floats), then n_mat rotation matrices (9), then n_dir directions (3)
kernel void interp_poses(device const float* p_prev [[buffer(0)]], device const float* p_cur [[buffer(1)]], device float* p_out [[buffer(2)]],
                         device const float* m_prev [[buffer(3)]], device const float* m_cur [[buffer(4)]], device float* m_out [[buffer(5)]],
                         device const float* d_prev [[buffer(6)]], device const float* d_cur [[buffer(7)]], device float* d_out [[buffer(8)]],
                         constant InterpConsts& c [[buffer(9)]], uint i [[thread_position_in_grid]])
{
    float t = c.tau;
    if (i < c.n_pos) {
        for (uint k = 0; k < 3; ++k) p_out[i * 3 + k] = mix(p_prev[i * 3 + k], p_cur[i * 3 + k], t);
        return;
    }
    i -= c.n_pos;
    if (i < c.n_mat) {
        float4 q = slerp(mat_to_quat(m_prev + i * 9), mat_to_quat(m_cur + i * 9), t);
        quat_to_rowmat(q, m_out + i * 9);
        return;
    }
    i -= c.n_mat;
    if (i < c.n_dir) {
        float3 a = float3(d_prev[i * 3], d_prev[i * 3 + 1], d_prev[i * 3 + 2]);
        float3 b = float3(d_cur[i * 3], d_cur[i * 3 + 1], d_cur[i * 3 + 2]);
        float3 v = mix(a, b, t);
        float l = length(v);
        v = l > 1e-12f ? v / l : b;
        d_out[i * 3] = v.x; d_out[i * 3 + 1] = v.y; d_out[i * 3 + 2] = v.z;
    }
}
