// metalsim particle rendering: one instance per render grain in the renderer's instance acceleration structure.
//
// Each simulated particle carries `grains_per_particle` render grains (metalsim/render/PARTICLES.md). A grain is an
// instance of one of a few small irregular grain meshes (own primitive acceleration structures), placed at the particle
// position plus a fixed random offset inside the particle's cube (half side `offset`, the sampling of Newton's
// sample_render_grains), with a random rotation, a random mesh and a random scale around `radius`. Every random number
// is a hash of (particle index or its stable id, grain index, seed): a grain keeps its shape, offset, orientation and colour from frame
// to frame (no flicker), and two runs build the same instances bit for bit. Non-finite particle positions and particles
// below `cull_z` get mask 0 (never intersected).
#include <metal_stdlib>
using namespace metal;

struct InstanceDesc {          // MTLAccelerationStructureInstanceDescriptor (64 bytes), as in rt.metal
    packed_float3 col0, col1, col2, col3;
    uint options;
    uint mask;
    uint if_table_offset;
    uint as_index;
};

struct GrainConsts {
    uint n_envs, n_particles, k_per, n_meshes;   // particles per env, grains per particle, grain meshes
    uint as_base, tiles_per_row, seed, inst_base; // first grain mesh's index in the instanced-structure list; env grid; first grain instance
    float stride, radius, radius_jitter, offset;  // env grid spacing; grain radius (volume-equivalent) and relative jitter; offset half side
    float cull_z;                                 // particles with z < cull_z get no grains (mask 0); -inf = none
    uint has_ids, _u0, _u1;                       // has_ids: grain identity from ids[] (stable ids of a compacted particle set)
};

inline uint hash_u(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
inline float u01(thread uint& s) { s = hash_u(s + 0x9e3779b9u); return float(s & 0xFFFFFFu) / 16777216.0; }

// uniform random rotation (Shoemake 1992) as a matrix
inline float3x3 random_rotation(thread uint& s) {
    float u1 = u01(s), u2 = u01(s) * 6.2831853f, u3 = u01(s) * 6.2831853f;
    float a = sqrt(1.0 - u1), b = sqrt(u1);
    float w = a * sin(u2), x = a * cos(u2), y = b * sin(u3), z = b * cos(u3);
    return float3x3(float3(1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)),
                    float3(2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)),
                    float3(2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)));
}

kernel void grain_instances(
    device InstanceDesc*    inst [[buffer(0)]],   // the renderer's instance descriptors; grains from c.inst_base
    device const float*     pos  [[buffer(1)]],   // (n_envs, n_particles, 3) particle positions [m]
    constant GrainConsts&   c    [[buffer(2)]],
    device const uint*      ids  [[buffer(3)]],   // (n_envs, n_particles) stable particle ids (c.has_ids)
    device uint*            keys [[buffer(4)]],   // per grain: its identity key (the renderers' per-grain colour hash)
    uint i [[thread_position_in_grid]])
{
    uint per_env = c.n_particles * c.k_per;
    if (i >= c.n_envs * per_env) return;
    uint e = i / per_env, pk = i - e * per_env, p = pk / c.k_per;
    uint pi = (e * c.n_particles + p) * 3;
    float3 x = float3(pos[pi], pos[pi + 1], pos[pi + 2]);
    uint key = c.has_ids ? ids[e * c.n_particles + p] * c.k_per + (pk - p * c.k_per) : pk;
    uint s = hash_u(key * 0x9E3779B1u ^ hash_u(c.seed + 0x632BE5ABu));
    float3 off = float3(u01(s), u01(s), u01(s)) * 2.0 - 1.0;
    float scale = c.radius * (1.0 + c.radius_jitter * (2.0 * u01(s) - 1.0));
    uint mesh = min(uint(u01(s) * float(c.n_meshes)), c.n_meshes - 1);
    float3x3 R = random_rotation(s) * scale;
    float3 env_off = float3(float(e % c.tiles_per_row) * c.stride, float(e / c.tiles_per_row) * c.stride, 0.0);
    uint3 bits = as_type<uint3>(x);
    bool ok = !any((bits & 0x7f800000u) == 0x7f800000u) && x.z >= c.cull_z;
    InstanceDesc d;
    d.col0 = R[0]; d.col1 = R[1]; d.col2 = R[2];
    d.col3 = ok ? x + off * c.offset + env_off : env_off;
    d.options = 4;                     // opaque
    d.mask = ok ? 0xFFFFFFFFu : 0u;    // mask 0: never intersected
    d.if_table_offset = 0;
    d.as_index = c.as_base + mesh;
    inst[c.inst_base + i] = d;
    keys[i] = key;
}
