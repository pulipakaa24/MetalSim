// Per-world MuJoCo Warp model fields into the renderers' per-env tables (metalsim.render.model_sync), one pass per
// frame in the render command buffer after the physics wait:
//   sync_colors: per (env, slot) colour = t_colors (the renderer's DR modulation) x the model colour of that world:
//                geom_rgba, or mat_rgba[geom_matid] when the geom has a material and its rgba is MuJoCo's default
//                (0.5, 0.5, 0.5, 1) (MuJoCo's rule; MuJoCo Warp render.py), or the scene tables' static colour for
//                slots whose colour is the renderer's own (terrain slots). The material table the shaders read has
//                rgba = 1, so base = material.rgba x colour is unchanged in form.
//   sync_lights: per (env, light) light record (tier0.metal Light) from light_type / castshadow / active / cutoff /
//                exponent / ambient / diffuse / specular / attenuation of that world; inactive lights emit nothing.
// A field with one world (shared) has world stride 0.
#include <metal_stdlib>
using namespace metal;

struct SyncConsts {
    uint n_envs, n_slots, n_geoms, n_mat;
    uint n_light, s_rgba, s_matid, s_matrgba;
    uint s_type, s_cast, s_active, s_cutoff;
    uint s_exp, s_amb, s_diff, s_spec;
    uint s_att, use_lights, opaque, _p1;
};
struct Light { float4 kind; float4 diffuse; float4 specular; float4 ambient; float4 atten; };

inline float3 v3(device const float* p, uint i) { return float3(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]); }

kernel void sync_colors(
    device const int*    slot_geom  [[buffer(0)]],
    device const float4* t_colors   [[buffer(1)]],   // (n_envs, n_slots)
    device const float4* geom_rgba  [[buffer(2)]],   // (W, n_geoms)
    device const int*    geom_matid [[buffer(3)]],   // (W, n_geoms)
    device const float4* mat_rgba   [[buffer(4)]],   // (W, n_mat)
    device const float4* static_rgba [[buffer(5)]],  // (n_slots) colour for slots with src == 0
    device const int*    src        [[buffer(6)]],   // (n_slots) 1 = model colour
    device float4*       out        [[buffer(7)]],   // (n_envs, n_slots)
    constant SyncConsts& c          [[buffer(8)]],
    device atomic_uint*  env_flag   [[buffer(9)]],   // (n_envs) 1: a partially transparent slot (alpha < 0.995)
    uint i [[thread_position_in_grid]])
{
    if (i >= c.n_envs * c.n_slots) return;
    uint e = i / c.n_slots, s = i - e * c.n_slots;
    float4 col = static_rgba[s];
    if (src[s] != 0) {
        uint g = uint(slot_geom[s]);
        float4 rg = geom_rgba[e * c.s_rgba * c.n_geoms + g];
        int mid = geom_matid[e * c.s_matid * c.n_geoms + g];
        bool dflt = rg.x == 0.5f && rg.y == 0.5f && rg.z == 0.5f && rg.w == 1.0f;
        col = (mid >= 0 && dflt && c.n_mat > 0) ? mat_rgba[e * c.s_matrgba * c.n_mat + uint(mid)] : rg;
    }
    if (c.opaque != 0) col.w = 1.0;          // the renderer's transparency=False
    float4 o = t_colors[i] * col;
    out[i] = o;
    if (o.w < 0.995f) atomic_store_explicit(env_flag + e, 1u, memory_order_relaxed);
}

kernel void sync_clear_env_flags(device uint* env_flag [[buffer(0)]], constant uint& n [[buffer(1)]],
                                 uint i [[thread_position_in_grid]])
{
    if (i < n) env_flag[i] = 0u;
}

kernel void sync_lights(
    device const Light*  tmpl      [[buffer(0)]],    // (n_light) static records (scene_tables.build_lights + overrides)
    device const uint*   lock      [[buffer(1)]],    // (n_light) bit0: keep the template's diffuse
    device const int*    ltype     [[buffer(2)]],
    device const uchar*  lcast     [[buffer(3)]],
    device const uchar*  lactive   [[buffer(4)]],
    device const float*  lcutoff   [[buffer(5)]],
    device const float*  lexp      [[buffer(6)]],
    device const float*  lamb      [[buffer(7)]],
    device const float*  ldiff     [[buffer(8)]],
    device const float*  lspec     [[buffer(9)]],
    device const float*  latt      [[buffer(10)]],
    device Light*        out       [[buffer(11)]],   // (n_envs, n_light)
    constant SyncConsts& c         [[buffer(12)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= c.n_envs * c.n_light) return;
    uint e = i / c.n_light, l = i - e * c.n_light;
    Light L = tmpl[l];
    if (c.use_lights != 0) {
        int t = ltype[e * c.s_type * c.n_light + l];            // mjLIGHT_SPOT 0, DIRECTIONAL 1, POINT 2, IMAGE 3
        L.kind.x = (t == 2) ? 1.0 : (t == 0) ? 2.0 : 0.0;
        L.kind.y = lcast[e * c.s_cast * c.n_light + l] ? 1.0 : 0.0;
        L.kind.z = lcutoff[e * c.s_cutoff * c.n_light + l];
        L.kind.w = lexp[e * c.s_exp * c.n_light + l];
        if ((lock[l] & 1u) == 0u) L.diffuse.xyz = v3(ldiff, e * c.s_diff * c.n_light + l);
        L.specular.xyz = v3(lspec, e * c.s_spec * c.n_light + l);
        L.ambient.xyz = v3(lamb, e * c.s_amb * c.n_light + l);
        L.atten.xyz = v3(latt, e * c.s_att * c.n_light + l);
        if (lactive[e * c.s_active * c.n_light + l] == 0) {
            L.diffuse.xyz = float3(0.0); L.specular.xyz = float3(0.0); L.ambient.xyz = float3(0.0); L.kind.y = 0.0;
        }
    }
    out[i] = L;
}

// Per-env material entries (metalsim.render.env_materials, appearance.h MatEntry): per (env, slot) the entry the
// shaders use, -1 = the slot's own material. The renderer's choice (user_ids, e.g. a texture-pool randomization) wins;
// else, with a per-world geom_matid, a geom whose material differs from its compiled one gets the entry of that
// (geom, material) pair: slot_base[slot] + 1 + matid (slot_base + 0 = no material).
struct MatIdConsts { uint n_envs, n_slots, n_geoms, s_matid, use_model, use_user, _a, _b; };

kernel void resolve_mat_ids(
    device const int*    user_ids   [[buffer(0)]],   // (n_envs, n_slots), -1 = none
    device const int*    geom_matid [[buffer(1)]],   // (W, n_geoms)
    device const int*    slot_geom  [[buffer(2)]],
    device const int*    compiled   [[buffer(3)]],   // (n_slots) compiled matid of the slot's geom
    device const int*    slot_base  [[buffer(4)]],   // (n_slots) first model entry of the slot, -1 = none
    device int*          out        [[buffer(5)]],   // (n_envs, n_slots)
    constant MatIdConsts& c         [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= c.n_envs * c.n_slots) return;
    uint e = i / c.n_slots, s = i - e * c.n_slots;
    int id = c.use_user != 0 ? user_ids[i] : -1;
    if (id < 0 && c.use_model != 0 && slot_base[s] >= 0) {
        int mid = geom_matid[e * c.s_matid * c.n_geoms + uint(slot_geom[s])];
        if (mid != compiled[s]) id = slot_base[s] + 1 + mid;
    }
    out[i] = id;
}
