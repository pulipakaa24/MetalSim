// metalsim tier-0 renderer: batched tile atlas, instanced unique meshes, PBR (GGX) shading lit by the
// model's light list (directional / point / spot, moving with bodies), MuJoCo's headlight and ambient
// model, a tiled shadow map for one caster per env, optional per-env background compositing, and
// RGB / depth / segmentation / normal targets written into learner-layout tensors by `untile`.
//
// Instance transforms are read straight from the physics state (MuJoCo Warp geom_xpos / geom_xmat
// in unified memory): there is no host pack step. Cameras and lights come from cam_xpos / cam_xmat /
// light_xpos / light_xdir with optional per-env deltas (domain randomization).
#include <metal_stdlib>
using namespace metal;

// Lights lit per fragment: every model light by default (Isaac / RTX has no light limit); Tier0Renderer(max_lights=8)
// compiles the variant with MuJoCo's effective cap (its fixed-function OpenGL path lights GL_LIGHT0..7). The value is
// also the DR light's caster sentinel.
#ifndef MAX_LIGHTS
#define MAX_LIGHTS 1048576
#endif

struct EnvParams {            // built on the GPU by build_env_params
    float4x4 view_proj;
    float4x4 light_vp;        // shadow caster view-projection (orthographic)
    float4   tile;            // ndc offset x, y, scale x, y (render atlas)
    float4   shadow_tile;     // same for the shadow atlas
    float4   dr_light;        // xyz direction of the DR light (normalized), w intensity (0: none)
    float4   ambient;         // rgb global ambient (headlight + model), w = background mode
    float4   clear_color;
    float4   cam_pos;         // world position, w = near
    float4   cam_fwd;         // world forward (-Z of the camera), w = far
    float4   misc;            // x: shadow caster light index (-1 none, MAX_LIGHTS = DR light), y: n lights, z: headlight active
    float4   caster_dir;      // xyz direction the shadow caster's light travels (normalized), w = 1 if there is a caster
};

struct CameraSpec {           // per env, host- or torch-written (28 floats)
    float4 intrinsics;        // fx, fy, cx, cy (pixels)
    float4 pos_delta;         // xyz camera offset in world frame, w unused
    float4 rot_delta;         // quaternion (w,x,y,z) applied after the model camera rotation
    float4 light;             // xyz DR directional light direction, w intensity (0 disables)
    float4 ambient;           // rgb ambient scale (1 = model), w background mode
    float4 clear_color;
    int    cam_id;            // model camera index
    int    _pad0, _pad1, _pad2;
};

struct Light {                // per model light (static part), 20 floats
    float4 kind;              // type (0 dir, 1 point, 2 spot), castshadow, cutoff (deg), exponent
    float4 diffuse;
    float4 specular;
    float4 ambient;
    float4 atten;             // attenuation constant, linear, quadratic; w = body id
};

struct SceneParams {          // 5 x float4
    float4 bounds;            // scene center xyz, radius (from mjModel.stat)
    float4 hl_ambient;
    float4 hl_diffuse;
    float4 hl_specular;       // w = headlight active
    float4 sky;               // mean skybox colour, w = skybox present
    float4 sky_rect;          // skybox cube texture in the atlas (u0, v0, du, dv)
    float4 sky_info;          // x faces (1 or 6), y = 1 when the skybox texture is drawn
};

struct Material {
    float4 rgba;
    float4 mrse;              // metallic, roughness, specular, emission
    float4 rect;              // atlas u0, v0, du, dv
    float4 par;               // texrepeat.xy, textured flag, reflectance
};

struct Semantic { int geom, body, root, group; };

#include "appearance.h"

// corner-attribute layout (see corner_fetch below)
constant bool corner_fc [[function_constant(0)]];
constant bool CORNER_ATTRS = is_function_constant_defined(corner_fc) ? corner_fc : false;
struct CornerAttr { float4 a; float4 b; };   // a: normal xyz, u | b: v

struct Consts {
    uint n_envs, n_geoms, n_slots, tiles_per_row;
    uint tile_w, tile_h, atlas_w, atlas_h;
    uint n_cam, bg_layers, flags, n_light;      // flags bit0: backgrounds, bit1: shadows, bit2: skybox, bit3: extra shadow views
    uint shadow_w, shadow_h, shadow_atlas_w, shadow_atlas_h;
    uint n_shadow_views, _c0, _c1, _c2;             // extra shadow views (flags bit3), see build_shadow_views
};

// -------------------------------------------------------------------------------------------------
// math helpers

inline float3x3 quat_to_mat(float4 q) {    // q = (w, x, y, z), columns
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return float3x3(
        float3(1 - 2*(y*y + z*z), 2*(x*y + w*z),     2*(x*z - w*y)),
        float3(2*(x*y - w*z),     1 - 2*(x*x + z*z), 2*(y*z + w*x)),
        float3(2*(x*z + w*y),     2*(y*z - w*x),     1 - 2*(x*x + y*y)));
}

inline float3x3 load_mat33(device const float* p) {   // MuJoCo row-major 3x3 -> metal column matrix
    return float3x3(float3(p[0], p[3], p[6]), float3(p[1], p[4], p[7]), float3(p[2], p[5], p[8]));
}

inline float4 tile_clip(float4 clip, float4 tile) {
    return float4(clip.x * tile.z + tile.x * clip.w, clip.y * tile.w + tile.y * clip.w, clip.z, clip.w);
}

// -------------------------------------------------------------------------------------------------
// per-env parameter build (one thread per env)

kernel void build_env_params(
    device EnvParams*        envs       [[buffer(0)]],
    device const CameraSpec* cams       [[buffer(1)]],
    device const float*      cam_xpos   [[buffer(2)]],   // (n_envs, n_cam, 3)
    device const float*      cam_xmat   [[buffer(3)]],   // (n_envs, n_cam, 9)
    constant Consts&         c          [[buffer(4)]],
    device const Light*      lights     [[buffer(5)]],
    device const float*      light_xdir [[buffer(6)]],   // (n_envs, n_light, 3)
    constant SceneParams&    sp         [[buffer(7)]],
    uint e [[thread_position_in_grid]])
{
    if (e >= c.n_envs) return;
    CameraSpec cs = cams[e];
    uint ci = uint(max(cs.cam_id, 0));
    device const float* xp = cam_xpos + (e * c.n_cam + ci) * 3;
    float3 pos = float3(xp[0], xp[1], xp[2]) + cs.pos_delta.xyz;
    float3x3 R = load_mat33(cam_xmat + (e * c.n_cam + ci) * 9);   // camera axes in world (columns)
    R = R * quat_to_mat(cs.rot_delta);
    float3x3 Rt = transpose(R);
    float4x4 V = float4x4(float4(Rt[0], 0), float4(Rt[1], 0), float4(Rt[2], 0), float4(-(Rt * pos), 1));
    float fx = cs.intrinsics.x, fy = cs.intrinsics.y, cx = cs.intrinsics.z, cy = cs.intrinsics.w;
    float w = float(c.tile_w), h = float(c.tile_h);
    float n = 0.01, f = 10.0;
    float4x4 P = float4x4(
        float4(2*fx/w, 0, 0, 0), float4(0, 2*fy/h, 0, 0),
        float4(1 - 2*cx/w, 2*cy/h - 1, f/(n - f), -1), float4(0, 0, n*f/(n - f), 0));
    EnvParams p;
    p.view_proj = P * V;
    float sx = float(c.tile_w) / float(c.atlas_w), sy = float(c.tile_h) / float(c.atlas_h);
    uint col = e % c.tiles_per_row, row = e / c.tiles_per_row;
    p.tile = float4(-1 + (2*col + 1) * sx, 1 - (2*row + 1) * sy, sx, sy);
    float ssx = float(c.shadow_w) / float(c.shadow_atlas_w), ssy = float(c.shadow_h) / float(c.shadow_atlas_h);
    p.shadow_tile = float4(-1 + (2*col + 1) * ssx, 1 - (2*row + 1) * ssy, ssx, ssy);
    float3 L = cs.light.xyz;
    float ll = length(L);
    p.dr_light = float4(ll > 0 ? L / ll : float3(0, 0, -1), cs.light.w);
    // shadow caster: the DR light if enabled, else the first shadow-casting directional model light
    int caster = -1;
    float3 cdir = float3(0, 0, -1);
    if (cs.light.w > 0.0) { caster = MAX_LIGHTS; cdir = p.dr_light.xyz; }
    else {
        for (uint i = 0; i < c.n_light && i < MAX_LIGHTS; ++i) {
            if (lights[i].kind.x == 0.0 && lights[i].kind.y > 0.5) {
                device const float* ld = light_xdir + (e * c.n_light + i) * 3;
                caster = int(i); cdir = normalize(float3(ld[0], ld[1], ld[2])); break;
            }
        }
    }
    {   // orthographic light view-projection around the scene bounds (depth in [0,1])
        float3 fwd = normalize(cdir);
        float3 up = fabs(fwd.z) < 0.99 ? float3(0, 0, 1) : float3(1, 0, 0);
        float3 right = normalize(cross(fwd, up));
        float3 lup = cross(right, fwd);
        float r = sp.bounds.w;
        float3 eye = sp.bounds.xyz - fwd * (r * 2.0);
        float3x3 B = float3x3(right, lup, -fwd);           // columns: light camera axes (looks along -Z)
        float3x3 Bt = transpose(B);
        float4x4 LV = float4x4(float4(Bt[0], 0), float4(Bt[1], 0), float4(Bt[2], 0), float4(-(Bt * eye), 1));
        float zn = 0.01, zf = r * 4.0;
        float4x4 LP = float4x4(float4(1/r, 0, 0, 0), float4(0, 1/r, 0, 0), float4(0, 0, -1/(zf - zn), 0), float4(0, 0, -zn/(zf - zn), 1));
        p.light_vp = LP * LV;
    }
    float3 amb = sp.hl_ambient.xyz;
    for (uint i = 0; i < c.n_light && i < MAX_LIGHTS; ++i) amb += lights[i].ambient.xyz;
    p.ambient = float4(amb * cs.ambient.xyz, cs.ambient.w);
    p.clear_color = cs.clear_color;
    p.cam_pos = float4(pos, n);
    p.cam_fwd = float4(-R[2], f);
    p.misc = float4(float(caster), float(min(c.n_light, uint(MAX_LIGHTS))), sp.hl_specular.w, 0);
    p.caster_dir = float4(normalize(cdir), caster >= 0 ? 1.0 : 0.0);
    envs[e] = p;
}

// -------------------------------------------------------------------------------------------------
// geometry pass

struct VSIn {
    float3 pos    [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv     [[attribute(2)]];
};

struct VSOut {
    float4 clip [[position]];
    float  clip_distance [[clip_distance]] [4];   // tile edges: geometry is clipped, not discarded
    float3 world_pos;
    float3 normal_w;
    float2 uv;
    uint   env  [[flat]];
    uint   slot [[flat]];
    float3 local_pos;         // geom-frame position (cube-map texture coordinates)
};

struct FSIn {
    float4 clip [[position]];
    float3 world_pos;
    float3 normal_w;
    float2 uv;
    uint   env  [[flat]];
    uint   slot [[flat]];
    float3 local_pos;         // geom-frame position (cube-map texture coordinates)
};

inline void slot_world(uint env, uint slot, device const int* slot_geom,
                       device const float* geom_xpos, device const float* geom_xmat, constant Consts& c,
                       float3 pos_l, float3 nrm_l, thread float3& p, thread float3& n)
{
    uint g = uint(slot_geom[slot]);
    uint gi = env * c.n_geoms + g;
    device const float* xp = geom_xpos + gi * 3;
    float3x3 R = load_mat33(geom_xmat + gi * 9);
    p = R * pos_l + float3(xp[0], xp[1], xp[2]);
    n = R * nrm_l;
}

inline void instance_world(uint inst, device const uint2* inst_tab, device const int* slot_geom,
                           device const float* geom_xpos, device const float* geom_xmat, constant Consts& c,
                           float3 pos_l, float3 nrm_l, thread float3& p, thread float3& n, thread uint& env, thread uint& slot)
{
    uint2 pair = inst_tab[inst];
    env = pair.x; slot = pair.y;
    slot_world(env, slot, slot_geom, geom_xpos, geom_xmat, c, pos_l, nrm_l, p, n);
}

inline void clip_to_tile(thread float* cd, float4 clip, float4 tile) {
    cd[0] = clip.x - (tile.x - tile.z) * clip.w;
    cd[1] = (tile.x + tile.z) * clip.w - clip.x;
    cd[2] = clip.y - (tile.y - tile.w) * clip.w;
    cd[3] = (tile.y + tile.w) * clip.w - clip.y;
}

vertex VSOut geom_vs(
    VSIn in [[stage_in]],
    uint inst [[instance_id]],
    device const EnvParams* envs      [[buffer(1)]],
    device const uint2*     inst_tab  [[buffer(2)]],
    device const int*       slot_geom [[buffer(3)]],
    device const float*     geom_xpos [[buffer(4)]],
    device const float*     geom_xmat [[buffer(5)]],
    device const float4*    per_inst  [[buffer(6)]],
    constant Consts&        c         [[buffer(7)]])
{
    float3 p, n; uint env, slot;
    instance_world(inst, inst_tab, slot_geom, geom_xpos, geom_xmat, c, in.pos, in.normal, p, n, env, slot);
    EnvParams e = envs[env];
    float4 clip = tile_clip(e.view_proj * float4(p, 1.0), e.tile);
    VSOut o;
    o.clip = clip;
    clip_to_tile(o.clip_distance, clip, e.tile);
    o.world_pos = p; o.normal_w = n; o.uv = in.uv; o.env = env; o.slot = slot; o.local_pos = in.pos;
    return o;
}

// -------------------------------------------------------------------------------------------------
// Transparent geoms (material-resolved rgba alpha < 0.995, MuJoCo's test): drawn after every opaque geom, as MuJoCo's
// mjr_render does: no depth write (depth = what is behind), blending src-alpha / one-minus-src-alpha, back faces
// culled, objects sorted per env by camera distance minus bounding radius (planes last), drawn front to back and then
// back to front. One draw per pass covers every env: instance r * n_envs + e is env e's rank-r object (rank-major, so
// the hardware's in-order blending follows the sort), and the vertex shader pulls that object's mesh by vertex id
// (the vertex count is the largest transparent mesh's index count; ids past a smaller mesh's count are degenerate).

struct TransConsts { uint n_trans, max_index, _p0, _p1; };
struct MeshRange { uint i_off, i_count, _p0, _p1; };

kernel void sort_transparent(
    device const EnvParams*  envs        [[buffer(0)]],
    device const int*        trans_slot  [[buffer(1)]],   // (T,) slot of each transparent geom
    device const float*      trans_rb    [[buffer(2)]],   // (T,) bounding radius, < 0 for planes (always last)
    device const int*        slot_geom   [[buffer(3)]],
    device const float*      geom_xpos   [[buffer(4)]],
    constant Consts&         c           [[buffer(5)]],
    constant TransConsts&    tc          [[buffer(6)]],
    device uint*             order       [[buffer(7)]],   // (n_envs, T) slots by increasing distance
    uint e [[thread_position_in_grid]])
{
    if (e >= c.n_envs) return;
    float3 cam = envs[e].cam_pos.xyz;
    uint T = tc.n_trans;
    device uint* o = order + e * T;
    float dist[64];
    uint n = min(T, 64u);
    for (uint k = 0; k < n; ++k) {
        uint slot = uint(trans_slot[k]);
        device const float* xp = geom_xpos + (e * c.n_geoms + uint(slot_geom[slot])) * 3;
        float d = trans_rb[k] < 0.0 ? 1e10 : length(float3(xp[0], xp[1], xp[2]) - cam) - trans_rb[k];
        uint j = k;                              // insertion sort (T is small)
        while (j > 0 && dist[j - 1] > d) { dist[j] = dist[j - 1]; o[j] = o[j - 1]; --j; }
        dist[j] = d; o[j] = slot;
    }
    for (uint k = n; k < T; ++k) o[k] = uint(trans_slot[k]);   // beyond 64: unsorted, drawn last
}

vertex VSOut trans_vs(
    uint vid [[vertex_id]],
    uint inst [[instance_id]],
    device const float*      verts     [[buffer(0)]],
    device const EnvParams*  envs      [[buffer(1)]],
    device const uint*       order     [[buffer(2)]],
    device const int*        slot_geom [[buffer(3)]],
    device const float*      geom_xpos [[buffer(4)]],
    device const float*      geom_xmat [[buffer(5)]],
    device const int*        slot_mesh [[buffer(6)]],
    constant Consts&         c         [[buffer(7)]],
    constant TransConsts&    tc        [[buffer(8)]],
    device const MeshRange*  meshes    [[buffer(9)]],
    device const uint*       indices   [[buffer(10)]],
    constant uint&           reverse   [[buffer(11)]],
    device const CornerAttr* corners   [[buffer(12)]])
{
    uint r = inst / c.n_envs, env = inst - r * c.n_envs;
    if (reverse) r = tc.n_trans - 1 - r;
    uint slot = order[env * tc.n_trans + r];
    MeshRange mr = meshes[slot_mesh[slot]];
    VSOut o;
    if (vid >= mr.i_count) {                     // past this object's mesh: a degenerate (culled) triangle
        o.clip = float4(0.0, 0.0, 2.0, 1.0);
        for (uint k = 0; k < 4; ++k) o.clip_distance[k] = -1.0;
        o.world_pos = float3(0.0); o.normal_w = float3(0, 0, 1); o.uv = float2(0.0); o.env = env; o.slot = slot; o.local_pos = float3(0.0);
        return o;
    }
    uint vi = indices[mr.i_off + vid];
    device const float* v = verts + vi * 8;
    float3 pos = float3(v[0], v[1], v[2]), nrm = float3(v[3], v[4], v[5]);
    float2 tuv = float2(v[6], v[7]);
    if (CORNER_ATTRS) { CornerAttr ca = corners[mr.i_off + vid]; nrm = ca.a.xyz; tuv = float2(ca.a.w, ca.b.x); }
    float3 p, n;
    slot_world(env, slot, slot_geom, geom_xpos, geom_xmat, c, pos, nrm, p, n);
    EnvParams e = envs[env];
    float4 clip = tile_clip(e.view_proj * float4(p, 1.0), e.tile);
    o.clip = clip;
    clip_to_tile(o.clip_distance, clip, e.tile);
    o.world_pos = p; o.normal_w = n; o.uv = tuv; o.env = env; o.slot = slot; o.local_pos = pos;
    return o;
}


// -------------------------------------------------------------------------------------------------
// Corner attributes (Tier0Renderer(normals="corner_fs" / "corner_vs")): vertices are shared by position and each
// triangle corner's normal / uv lives in a per-corner table in index order (scene_tables corner_attributes), so a hard
// edge does not duplicate vertices. corner_fs: the fragment stage reads the triangle's three corners by primitive_id
// and blends them with the hardware's perspective-correct barycentrics (what the rasterizer's own interpolation of
// per-vertex attributes computes). corner_vs: a non-indexed draw pulls position and corner attributes per corner.

inline FSIn corner_fetch(FSIn in, uint prim, float3 bary, device const CornerAttr* corners, device const uint* slot_ibase,
                         device const Semantic* sem, device const float* geom_xmat, constant Consts& c)
{
    uint ib = slot_ibase[in.slot];
    if (ib == 0xFFFFFFFFu) return in;              // a slot outside the corner table (sites, extras)
    uint b = ib + prim * 3u;
    CornerAttr c0 = corners[b], c1 = corners[b + 1], c2 = corners[b + 2];
    float3 n = c0.a.xyz * bary.x + c1.a.xyz * bary.y + c2.a.xyz * bary.z;
    in.uv = float2(c0.a.w, c0.b.x) * bary.x + float2(c1.a.w, c1.b.x) * bary.y + float2(c2.a.w, c2.b.x) * bary.z;
    float3x3 R = load_mat33(geom_xmat + (in.env * c.n_geoms + uint(sem[in.slot].geom)) * 9);
    in.normal_w = R * n;
    return in;
}

vertex VSOut geom_vs_pull(
    uint vid [[vertex_id]],                       // global corner index (vertexStart = the mesh's first index)
    uint inst [[instance_id]],
    device const float*      verts     [[buffer(0)]],
    device const EnvParams*  envs      [[buffer(1)]],
    device const uint2*      inst_tab  [[buffer(2)]],
    device const int*        slot_geom [[buffer(3)]],
    device const float*      geom_xpos [[buffer(4)]],
    device const float*      geom_xmat [[buffer(5)]],
    constant Consts&         c         [[buffer(7)]],
    device const uint*       indices   [[buffer(8)]],
    device const CornerAttr* corners   [[buffer(9)]])
{
    device const float* v = verts + indices[vid] * 8;
    CornerAttr ca = corners[vid];
    float3 p, n; uint env, slot;
    float3 pos = float3(v[0], v[1], v[2]);
    instance_world(inst, inst_tab, slot_geom, geom_xpos, geom_xmat, c, pos, ca.a.xyz, p, n, env, slot);
    EnvParams e = envs[env];
    float4 clip = tile_clip(e.view_proj * float4(p, 1.0), e.tile);
    VSOut o;
    o.clip = clip;
    clip_to_tile(o.clip_distance, clip, e.tile);
    o.world_pos = p; o.normal_w = n; o.uv = float2(ca.a.w, ca.b.x); o.env = env; o.slot = slot; o.local_pos = pos;
    return o;
}

struct ShadowOut {
    float4 clip [[position]];
    float  clip_distance [[clip_distance]] [4];
};

vertex ShadowOut shadow_vs(
    VSIn in [[stage_in]],
    uint inst [[instance_id]],
    device const EnvParams* envs      [[buffer(1)]],
    device const uint2*     inst_tab  [[buffer(2)]],
    device const int*       slot_geom [[buffer(3)]],
    device const float*     geom_xpos [[buffer(4)]],
    device const float*     geom_xmat [[buffer(5)]],
    device const float4*    per_inst  [[buffer(6)]],
    constant Consts&        c         [[buffer(7)]])
{
    float3 p, n; uint env, slot;
    instance_world(inst, inst_tab, slot_geom, geom_xpos, geom_xmat, c, in.pos, in.normal, p, n, env, slot);
    EnvParams e = envs[env];
    float4 clip = tile_clip(e.light_vp * float4(p, 1.0), e.shadow_tile);
    ShadowOut o;
    o.clip = clip;
    clip_to_tile(o.clip_distance, clip, e.shadow_tile);
    return o;
}

struct FSOut {
    float4 color  [[color(0)]];
    uint   seg    [[color(1)]];
    float4 normal [[color(2)]];   // world normal xyz, w = 1 where geometry
};

constant float PI = 3.14159265358979f;

inline float D_ggx(float NdotH, float a2) {
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-7);
}
inline float G_smith(float NdotV, float NdotL, float a2) {
    float gv = 2.0 * NdotV / (NdotV + sqrt(a2 + (1.0 - a2) * NdotV * NdotV) + 1e-7);
    float gl = 2.0 * NdotL / (NdotL + sqrt(a2 + (1.0 - a2) * NdotL * NdotL) + 1e-7);
    return gv * gl;
}
inline float3 F_schlick(float3 f0, float VdotH) {
    return f0 + (1.0 - f0) * pow(1.0 - VdotH, 5.0);
}

inline float3 shade_light(float3 N, float3 V, float3 L, float3 rad, float3 base, float3 f0, float metallic, float a2, float NdotV) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return float3(0.0);
    float3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0), VdotH = max(dot(V, H), 0.0);
    float3 F = F_schlick(f0, VdotH);
    float3 spec = D_ggx(NdotH, a2) * G_smith(NdotV, NdotL, a2) * F / (4.0 * NdotV * NdotL + 1e-4);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    return (kd * base / PI + spec) * NdotL * rad;
}

// -------------------------------------------------------------------------------------------------
// Extra shadow views (Tier0Renderer(shadow_lights="all")): every shadow-casting model light other than the legacy
// caster above gets its own shadow map, a layer of `smaps` with the same per-env tile layout as the caster's atlas:
// directional = orthographic around the scene bounds (as the caster), spot = perspective over the cone (full cone
// angle, capped at 170 deg; MuJoCo covers cutoff x shadowscale), point = six 90-degree cube faces (Isaac / RTX shadow
// point lights; MuJoCo's classic renderer does not light point lights at all). Spot / point layers store the linear
// distance to the light over the view's far distance, so one bias works at every depth.

struct ShadowView { float4x4 vp; float4 info; };   // info: x light, y face (-1 not a cube face), z perspective (1/0), w far
// per model light (light_sv): x first view (-1 none), y views (1 or 6), z kind (0 dir, 1 point, 2 spot)

inline float4x4 look_vp(float3 eye, float3 fwd, float3 up_hint, float4x4 P) {
    float3 f = normalize(fwd);
    float3 up = fabs(dot(f, up_hint)) < 0.99 ? up_hint : (fabs(f.z) < 0.99 ? float3(0, 0, 1) : float3(1, 0, 0));
    float3 r = normalize(cross(f, up));
    float3 u = cross(r, f);
    float3x3 B = float3x3(r, u, -f);
    float3x3 Bt = transpose(B);
    float4x4 V = float4x4(float4(Bt[0], 0), float4(Bt[1], 0), float4(Bt[2], 0), float4(-(Bt * eye), 1));
    return P * V;
}

inline float4x4 persp(float fov, float n, float f) {
    float k = 1.0 / tan(0.5 * fov);
    return float4x4(float4(k, 0, 0, 0), float4(0, k, 0, 0), float4(0, 0, f / (n - f), -1), float4(0, 0, n * f / (n - f), 0));
}

constant float3 CUBE_FWD[6] = { float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0), float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1) };
constant float3 CUBE_UP[6]  = { float3(0, 0, 1), float3(0, 0, 1), float3(0, 0, 1), float3(0, 0, 1), float3(0, 1, 0), float3(0, 1, 0) };

kernel void build_shadow_views(
    device ShadowView*       views      [[buffer(0)]],   // (n_envs, n_views)
    device const int4*       view_light [[buffer(1)]],   // per view: light, face, kind (0 dir, 1 point, 2 spot)
    device const Light*      lights     [[buffer(2)]],
    device const float*      light_xpos [[buffer(3)]],
    device const float*      light_xdir [[buffer(4)]],
    constant Consts&         c          [[buffer(5)]],
    constant SceneParams&    sp         [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint n_views = c.n_shadow_views;
    if (tid >= c.n_envs * n_views) return;
    uint e = tid / n_views, v = tid - e * n_views;
    int4 vl = view_light[v];
    uint i = uint(vl.x);
    device const float* lp = light_xpos + (e * c.n_light + i) * 3;
    device const float* ld = light_xdir + (e * c.n_light + i) * 3;
    float3 Lpos = float3(lp[0], lp[1], lp[2]), Ldir = normalize(float3(ld[0], ld[1], ld[2]));
    float r = sp.bounds.w;
    ShadowView sv;
    if (vl.z == 0) {          // directional: orthographic around the scene bounds (the caster's construction)
        float3 eye = sp.bounds.xyz - Ldir * (r * 2.0);
        float zn = 0.01, zf = r * 4.0;
        float4x4 LP = float4x4(float4(1/r, 0, 0, 0), float4(0, 1/r, 0, 0), float4(0, 0, -1/(zf - zn), 0), float4(0, 0, -zn/(zf - zn), 1));
        sv.vp = look_vp(eye, Ldir, float3(0, 0, 1), LP);
        sv.info = float4(float(i), -1.0, 0.0, zf);
    } else {
        float far = max(length(Lpos - sp.bounds.xyz) + 2.0 * r, 0.1);
        float near = max(far * 1e-4, 1e-3);
        if (vl.z == 2) {      // spot: the cone
            float fov = min(2.0 * lights[i].kind.z, 170.0) * PI / 180.0;
            sv.vp = look_vp(Lpos, Ldir, float3(0, 0, 1), persp(fov, near, far));
            sv.info = float4(float(i), -1.0, 1.0, far);
        } else {              // point: cube face vl.y
            uint f = uint(vl.y);
            sv.vp = look_vp(Lpos, CUBE_FWD[f], CUBE_UP[f], persp(0.5 * PI, near, far));
            sv.info = float4(float(i), float(f), 1.0, far);
        }
    }
    views[tid] = sv;
}

struct ShadowViewOut {
    float4 clip [[position]];
    float  clip_distance [[clip_distance]] [4];
    float3 world_pos;
    uint   env [[flat]];
};

vertex ShadowViewOut shadow_view_vs(
    VSIn in [[stage_in]],
    uint inst [[instance_id]],
    device const EnvParams*  envs      [[buffer(1)]],
    device const uint2*      inst_tab  [[buffer(2)]],
    device const int*        slot_geom [[buffer(3)]],
    device const float*      geom_xpos [[buffer(4)]],
    device const float*      geom_xmat [[buffer(5)]],
    device const ShadowView* views     [[buffer(6)]],
    constant Consts&         c         [[buffer(7)]],
    constant uint&           view      [[buffer(8)]])
{
    float3 p, n; uint env, slot;
    instance_world(inst, inst_tab, slot_geom, geom_xpos, geom_xmat, c, in.pos, in.normal, p, n, env, slot);
    EnvParams e = envs[env];
    ShadowView sv = views[env * c.n_shadow_views + view];
    float4 clip = tile_clip(sv.vp * float4(p, 1.0), e.shadow_tile);
    ShadowViewOut o;
    o.clip = clip;
    clip_to_tile(o.clip_distance, clip, e.shadow_tile);
    o.world_pos = p; o.env = env;
    return o;
}

struct ShadowDepthOut { float depth [[depth(any)]]; };
struct ShadowViewIn {
    float4 clip [[position]];
    float3 world_pos;
    uint   env [[flat]];
};

fragment ShadowDepthOut shadow_view_fs(
    ShadowViewIn in [[stage_in]],
    device const ShadowView* views      [[buffer(6)]],
    constant Consts&         c          [[buffer(7)]],
    constant uint&           view       [[buffer(8)]],
    device const float*      light_xpos [[buffer(9)]])
{
    ShadowView sv = views[in.env * c.n_shadow_views + view];
    ShadowDepthOut o;
    if (sv.info.z > 0.5) {   // perspective views store the linear distance to the light
        device const float* lp = light_xpos + (in.env * c.n_light + uint(sv.info.x)) * 3;
        o.depth = saturate(length(in.world_pos - float3(lp[0], lp[1], lp[2])) / sv.info.w);
    } else {
        o.depth = in.clip.z;
    }
    return o;
}

// visibility of world point `wp` from light lsv (fragment of env `env`): 1 lit, 0 shadowed (2x2 hardware-filtered)
inline float extra_shadow(depth2d_array<float> smaps, sampler ss, device const ShadowView* sviews, int4 lsv, uint env,
                          constant Consts& c, float4 tile, float3 wp, float3 Lpos, float3 N, float3 L)
{
    uint v = uint(lsv.x);
    if (lsv.y == 6) {        // point light: the cube face the point lies in
        float3 d = wp - Lpos, a = fabs(d);
        uint f = (a.x >= a.y && a.x >= a.z) ? (d.x >= 0 ? 0u : 1u) : (a.y >= a.z ? (d.y >= 0 ? 2u : 3u) : (d.z >= 0 ? 4u : 5u));
        v += f;
    }
    ShadowView sv = sviews[env * c.n_shadow_views + v];
    float4 lc = sv.vp * float4(wp, 1.0);
    if (lc.w <= 0.0) return 1.0;
    float2 ndc = lc.xy / lc.w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < 0.0) || any(uv > 1.0)) return 1.0;
    float2 st = float2(tile.x * 0.5 + 0.5, 0.5 - tile.y * 0.5);
    float2 auv = st + (uv - 0.5) * tile.zw;
    float cosl = saturate(dot(N, L));
    float ref;
    if (sv.info.z > 0.5) {   // texel footprint at this distance x slope, plus 4 mm
        float dist = length(wp - Lpos);
        float texel = 2.0 * dist / float(c.shadow_w);   // a 90-degree view's texel at this distance
        float bias = 0.004 + 1.5 * texel * sqrt(max(1.0 - cosl * cosl, 0.0)) / max(cosl, 0.2);
        ref = (dist - bias) / sv.info.w;
    } else {
        ref = lc.z - 0.0015;
    }
    return smaps.sample_compare(ss, auv, v, ref);
}

// Per-env culling of the extra shadow views (Tier0Renderer(shadow_cull=True)): an instance is drawn into view v only
// when its bounding sphere (geom_xpos, rbound) meets v's frustum. One thread per (view, instance) appends the
// instance to its mesh's region of view v's list and bumps that draw's indirect instance count; shadow_view_vs reads
// the compacted list as its instance table. Conservative (sphere vs the six planes), so the maps are unchanged.
struct SvDraw { uint i_off, i_count, first, _p; };            // one per shadow draw (mesh)

kernel void shadow_cull_reset(
    device uint*           args   [[buffer(0)]],   // (n_views, n_draws, 5) MTLDrawIndexedPrimitivesIndirectArguments
    device const SvDraw*   draws  [[buffer(1)]],
    constant uint4&        k      [[buffer(2)]],   // n_views, n_draws, n_inst (list stride per view)
    uint i [[thread_position_in_grid]])
{
    if (i >= k.x * k.y) return;
    uint v = i / k.y, d = i - v * k.y;
    device uint* a = args + i * 5;
    a[0] = draws[d].i_count; a[1] = 0u; a[2] = draws[d].i_off; a[3] = 0u; a[4] = v * k.z + draws[d].first;
}

kernel void shadow_cull(
    device const ShadowView* views     [[buffer(0)]],   // (n_envs, n_views)
    device const uint2*      inst_tab  [[buffer(1)]],
    device const int*        inst_draw [[buffer(2)]],   // per instance: shadow draw index, -1 not drawn
    device const int*        slot_geom [[buffer(3)]],
    device const float*      geom_xpos [[buffer(4)]],
    device const float*      slot_rb   [[buffer(5)]],   // bounding radius per slot
    constant Consts&         c         [[buffer(6)]],
    constant uint4&          k         [[buffer(7)]],
    device atomic_uint*      args      [[buffer(8)]],
    device uint2*            list      [[buffer(9)]],   // (n_views, n_inst)
    device const SvDraw*     draws     [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    uint n_inst = k.z;
    if (tid >= k.x * n_inst) return;
    uint v = tid / n_inst, i = tid - v * n_inst;
    int d = inst_draw[i];
    if (d < 0) return;
    uint2 pr = inst_tab[i];
    device const float* xp = geom_xpos + (pr.x * c.n_geoms + uint(slot_geom[pr.y])) * 3;
    float4 p = float4(xp[0], xp[1], xp[2], 1.0);
    float r = slot_rb[pr.y];
    float4x4 M = views[pr.x * c.n_shadow_views + v].vp;
    float4 r0 = float4(M[0][0], M[1][0], M[2][0], M[3][0]), r1 = float4(M[0][1], M[1][1], M[2][1], M[3][1]);
    float4 r2 = float4(M[0][2], M[1][2], M[2][2], M[3][2]), r3 = float4(M[0][3], M[1][3], M[2][3], M[3][3]);
    float4 pl[6] = { r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2 };
    for (uint q = 0; q < 6; ++q) {
        float len = length(pl[q].xyz);
        if (len > 0.0 && dot(pl[q], p) / len < -r) return;     // fully outside one plane
    }
    uint slot_in_draw = atomic_fetch_add_explicit(args + (v * k.y + uint(d)) * 5 + 1, 1u, memory_order_relaxed);
    list[v * n_inst + draws[d].first + slot_in_draw] = pr;
}

fragment FSOut geom_fs(
    FSIn in0 [[stage_in]],
    uint prim [[primitive_id]],
    float3 bary [[barycentric_coord]],
    device const CornerAttr* corners   [[buffer(21)]],
    device const uint*       slot_ibase [[buffer(22)]],
    device const float*      geom_xmat [[buffer(23)]],
    device const EnvParams* envs       [[buffer(1)]],
    device const Material*  mats       [[buffer(2)]],
    device const Semantic*  sem        [[buffer(3)]],
    device const float4*    per_inst   [[buffer(4)]],
    constant Consts&        c          [[buffer(5)]],
    device const Light*     lights     [[buffer(6)]],
    device const float*     light_xpos [[buffer(7)]],
    device const float*     light_xdir [[buffer(8)]],
    constant SceneParams&   sp         [[buffer(9)]],
    texture2d<float>        atlas      [[texture(0)]],
    depth2d<float>          shadow     [[texture(2)]],
    sampler                 samp       [[sampler(0)]],
    sampler                 shadow_samp [[sampler(1)]],
    device const MaterialExt* mext     [[buffer(18)]],
    device const ShadowView* sviews    [[buffer(19)]],
    device const int4*       light_sv  [[buffer(20)]],
    depth2d_array<float>     smaps     [[texture(3)]])
{
    FSIn in = CORNER_ATTRS ? corner_fetch(in0, prim, bary, corners, slot_ibase, sem, geom_xmat, c) : in0;
    EnvParams e = envs[in.env];
    Material m = mats[in.slot];
    float4 mod = per_inst[in.env * c.n_slots + in.slot];
    float3 base = m.rgba.rgb * mod.rgb;
    float4 texc = material_tex(m.rect, m.par, mext[in.slot], in.uv, in.local_pos, atlas, samp);
    base *= texc.rgb;
    float metallic = m.mrse.x, rough = max(m.mrse.y, 0.03), specw = m.mrse.z, emission = m.mrse.w;
    float3 N = normalize(in.normal_w);
    float3 V = normalize(e.cam_pos.xyz - in.world_pos);
    if (dot(N, V) < 0.0) N = -N;
    float NdotV = max(dot(N, V), 1e-4);
    float a2 = rough * rough * rough * rough;
    float3 f0 = mix(float3(0.08 * specw), base, metallic);

    float vis = 1.0;
    int caster = int(e.misc.x);
    if ((c.flags & 2u) && caster >= 0) {
        float4 lc = e.light_vp * float4(in.world_pos, 1.0);
        float2 uv = float2(lc.x * 0.5 + 0.5, 0.5 - lc.y * 0.5);
        float2 st = float2(e.shadow_tile.x * 0.5 + 0.5, 0.5 - e.shadow_tile.y * 0.5);
        float2 sz = float2(e.shadow_tile.z, e.shadow_tile.w);
        float2 auv = st + (uv - 0.5) * 2.0 * sz * 0.5;
        if (all(uv >= 0.0) && all(uv <= 1.0))
            vis = shadow.sample_compare(shadow_samp, auv, lc.z - 0.0015);
    }

    float3 direct = float3(0.0);
    uint nl = uint(e.misc.y);
    for (uint i = 0; i < nl; ++i) {
        Light l = lights[i];
        device const float* lp = light_xpos + (in.env * c.n_light + i) * 3;
        device const float* ld = light_xdir + (in.env * c.n_light + i) * 3;
        float3 Lpos = float3(lp[0], lp[1], lp[2]);
        float3 Ldir = normalize(float3(ld[0], ld[1], ld[2]));
        float3 L; float att = 1.0;
        if (l.kind.x == 0.0) { L = -Ldir; }
        else {
            float3 dv = Lpos - in.world_pos; float dist = length(dv); L = dv / max(dist, 1e-6);
            att = 1.0 / max(l.atten.x + l.atten.y * dist + l.atten.z * dist * dist, 1e-4);
            if (l.kind.x == 2.0) {
                float cosang = dot(-L, Ldir);
                float cutoff = cos(l.kind.z * PI / 180.0);
                att *= cosang > cutoff ? pow(max(cosang, 0.0), l.kind.w) : 0.0;
            }
        }
        // MuJoCo/OpenGL light "diffuse" values are unnormalized: a head-on Lambertian surface returns
        // albedo * diffuse. With the energy-conserving BRDF (albedo / pi) that means radiance = pi * diffuse.
        float s = (int(i) == caster) ? vis : 1.0;
        int4 lsv = light_sv[i];
        if ((c.flags & 8u) && lsv.x >= 0 && int(i) != caster && att > 0.0)   // own shadow view(s): spot, point, other directional
            s = extra_shadow(smaps, shadow_samp, sviews, lsv, in.env, c, e.shadow_tile, in.world_pos, Lpos, N, L);
        direct += shade_light(N, V, L, l.diffuse.xyz * (PI * att), base, f0, metallic, a2, NdotV) * s;
    }
    if (e.dr_light.w > 0.0) {
        float s = (caster == MAX_LIGHTS) ? vis : 1.0;
        direct += shade_light(N, V, -e.dr_light.xyz, float3(PI * e.dr_light.w), base, f0, metallic, a2, NdotV) * s;
    }
    if (e.misc.z > 0.5)   // MuJoCo headlight: directional light along the view direction, no shadow
        direct += shade_light(N, V, V, sp.hl_diffuse.xyz * PI, base, f0, metallic, a2, NdotV);
    float3 color = direct + e.ambient.rgb * base + emission * base;   // flat ambient, as MuJoCo
    FSOut o;
    o.color = float4(color, m.rgba.a * mod.a * texc.a);   // alpha: used by the transparent pass only
    o.seg = in.slot + 1u;
    o.normal = float4(N, 1.0);
    return o;
}

// -------------------------------------------------------------------------------------------------
// background pass

struct BGOut {
    float4 clip [[position]];
    float2 uv;
    uint env [[flat]];
};

// -------------------------------------------------------------------------------------------------
// heightfield tile culling (Tier0Renderer(hfield_cull=True)): a heightfield is drawn as T x T-cell tiles; per frame
// hf_cull appends the (env, tile) pairs whose box meets the env's view frustum (render pass) and whose box swept
// along the shadow caster's light down to the field's lowest point, grown by a shadow-map margin, meets it (shadow
// pass: every occluder of a visible point, including the texels its filtered lookup reads). hf_tile_vs fetches the
// same vertex records and transforms them with the same code as geom_vs, so the visible pixels are unchanged.

struct HfTile  { int slot, vbase, nrow, ncol, r0, c0, field, _p; };
struct HfBox   { float4 lo, hi; };                                     // geom-frame AABB (w unused)
struct CullConsts { uint n_envs, n_tiles, T, flags; float margin, _a, _b, _c; };   // flags bit0: shadow list

inline void box_world(HfBox b, float3x3 R, float3 t, thread float3& lo, thread float3& hi) {
    lo = float3(INFINITY); hi = float3(-INFINITY);
    for (uint k = 0; k < 8; ++k) {
        float3 q = float3((k & 1) ? b.hi.x : b.lo.x, (k & 2) ? b.hi.y : b.lo.y, (k & 4) ? b.hi.z : b.lo.z);
        float3 w = R * q + t; lo = min(lo, w); hi = max(hi, w);
    }
}

inline bool box_in_frustum(float4x4 M, float3 lo, float3 hi) {   // conservative: false only if all corners are outside one plane
    uint o0 = 0, o1 = 0, o2 = 0, o3 = 0, o4 = 0, o5 = 0;
    for (uint k = 0; k < 8; ++k) {
        float4 q = M * float4((k & 1) ? hi.x : lo.x, (k & 2) ? hi.y : lo.y, (k & 4) ? hi.z : lo.z, 1.0);
        o0 += q.x < -q.w; o1 += q.x > q.w; o2 += q.y < -q.w; o3 += q.y > q.w; o4 += q.z < 0.0; o5 += q.z > q.w;
    }
    return !(o0 == 8 || o1 == 8 || o2 == 8 || o3 == 8 || o4 == 8 || o5 == 8);
}

kernel void hf_cull_reset(device atomic_uint* args [[buffer(0)]], constant CullConsts& cc [[buffer(1)]],
                          uint i [[thread_position_in_grid]])
{   // two MTLDrawIndexedPrimitivesIndirectArguments, 8 uints apart: index count, instance count, 0, 0, 0
    if (i >= 2) return;
    device atomic_uint* a = args + i * 8;
    atomic_store_explicit(a + 0, cc.T * cc.T * 6, memory_order_relaxed);
    for (uint k = 1; k < 5; ++k) atomic_store_explicit(a + k, 0u, memory_order_relaxed);
}

kernel void hf_cull(
    device const EnvParams* envs      [[buffer(0)]],
    device const HfTile*    tiles     [[buffer(1)]],
    device const HfBox*     boxes     [[buffer(2)]],
    device const HfBox*     fields    [[buffer(3)]],
    device const int*       slot_geom [[buffer(4)]],
    device const float*     geom_xpos [[buffer(5)]],
    device const float*     geom_xmat [[buffer(6)]],
    constant Consts&        c         [[buffer(7)]],
    constant CullConsts&    cc        [[buffer(8)]],
    device uint2*           vis       [[buffer(9)]],
    device uint2*           svis      [[buffer(10)]],
    device atomic_uint*     args      [[buffer(11)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= cc.n_envs * cc.n_tiles) return;
    uint env = i / cc.n_tiles, k = i - env * cc.n_tiles;
    HfTile td = tiles[k];
    uint gi = env * c.n_geoms + uint(slot_geom[td.slot]);
    device const float* xp = geom_xpos + gi * 3;
    float3x3 R = load_mat33(geom_xmat + gi * 9);
    float3 t = float3(xp[0], xp[1], xp[2]);
    float3 lo, hi; box_world(boxes[k], R, t, lo, hi);
    EnvParams e = envs[env];
    if (box_in_frustum(e.view_proj, lo, hi)) {
        uint idx = atomic_fetch_add_explicit(args + 1, 1u, memory_order_relaxed);
        vis[idx] = uint2(env, k);
    }
    if ((cc.flags & 1u) && e.caster_dir.w > 0.5) {
        float3 L = e.caster_dir.xyz;
        bool s = true;
        if (L.z < -1e-3) {
            float3 flo, fhi; box_world(fields[td.field], R, t, flo, fhi);
            float3 d = L * ((hi.z - flo.z) / -L.z);            // down to the field's lowest point
            float3 slo = min(lo, lo + d) - cc.margin, shi = max(hi, hi + d) + cc.margin;
            s = box_in_frustum(e.view_proj, slo, shi);
        }
        if (s) {
            uint idx = atomic_fetch_add_explicit(args + 9, 1u, memory_order_relaxed);
            svis[idx] = uint2(env, k);
        }
    }
}

inline void hf_vertex(uint vid, uint2 ek, device const HfTile* tiles, device const float* verts, constant CullConsts& cc,
                      thread float3& pos, thread float3& nrm, thread float2& uv, thread uint& slot)
{
    HfTile td = tiles[ek.y];
    uint T1 = cc.T + 1, lr = vid / T1, lc = vid - lr * T1;
    int r = min(td.r0 + int(lr), td.nrow - 1), col = min(td.c0 + int(lc), td.ncol - 1);   // past the edge: degenerate
    device const float* v = verts + (uint(td.vbase) + uint(r * td.ncol + col)) * 8;
    pos = float3(v[0], v[1], v[2]); nrm = float3(v[3], v[4], v[5]); uv = float2(v[6], v[7]);
    slot = uint(td.slot);
}

vertex VSOut hf_tile_vs(
    uint vid [[vertex_id]],
    uint inst [[instance_id]],
    device const float*     verts     [[buffer(0)]],
    device const EnvParams* envs      [[buffer(1)]],
    device const uint2*     vis       [[buffer(2)]],
    device const int*       slot_geom [[buffer(3)]],
    device const float*     geom_xpos [[buffer(4)]],
    device const float*     geom_xmat [[buffer(5)]],
    device const HfTile*    tiles     [[buffer(6)]],
    constant Consts&        c         [[buffer(7)]],
    constant CullConsts&    cc        [[buffer(8)]])
{
    uint2 ek = vis[inst];
    float3 pos, nrm; float2 uv; uint slot;
    hf_vertex(vid, ek, tiles, verts, cc, pos, nrm, uv, slot);
    uint env = ek.x;
    float3 p, n;
    slot_world(env, slot, slot_geom, geom_xpos, geom_xmat, c, pos, nrm, p, n);
    EnvParams e = envs[env];
    float4 clip = tile_clip(e.view_proj * float4(p, 1.0), e.tile);
    VSOut o;
    o.clip = clip;
    clip_to_tile(o.clip_distance, clip, e.tile);
    o.world_pos = p; o.normal_w = n; o.uv = uv; o.env = env; o.slot = slot; o.local_pos = pos;
    return o;
}

vertex ShadowOut hf_tile_shadow_vs(
    uint vid [[vertex_id]],
    uint inst [[instance_id]],
    device const float*     verts     [[buffer(0)]],
    device const EnvParams* envs      [[buffer(1)]],
    device const uint2*     vis       [[buffer(2)]],
    device const int*       slot_geom [[buffer(3)]],
    device const float*     geom_xpos [[buffer(4)]],
    device const float*     geom_xmat [[buffer(5)]],
    device const HfTile*    tiles     [[buffer(6)]],
    constant Consts&        c         [[buffer(7)]],
    constant CullConsts&    cc        [[buffer(8)]])
{
    uint2 ek = vis[inst];
    float3 pos, nrm; float2 uv; uint slot;
    hf_vertex(vid, ek, tiles, verts, cc, pos, nrm, uv, slot);
    float3 p, n;
    slot_world(ek.x, slot, slot_geom, geom_xpos, geom_xmat, c, pos, nrm, p, n);
    EnvParams e = envs[ek.x];
    float4 clip = tile_clip(e.light_vp * float4(p, 1.0), e.shadow_tile);
    ShadowOut o;
    o.clip = clip;
    clip_to_tile(o.clip_distance, clip, e.shadow_tile);
    return o;
}

vertex BGOut bg_vs(uint vid [[vertex_id]], uint env [[instance_id]], device const EnvParams* envs [[buffer(1)]])
{
    float2 quad[6] = { float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1) };
    float2 q = quad[vid];
    EnvParams e = envs[env];
    BGOut o;
    o.clip = float4(q.x * e.tile.z + e.tile.x, q.y * e.tile.w + e.tile.y, 0.99999, 1.0);
    o.uv = float2(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5);
    o.env = env;
    return o;
}

fragment FSOut bg_fs(BGOut in [[stage_in]], device const EnvParams* envs [[buffer(1)]], constant Consts& c [[buffer(5)]],
                     texture2d_array<float> bg [[texture(1)]], sampler samp [[sampler(0)]],
                     constant SceneParams& sp [[buffer(9)]], texture2d<float> atlas [[texture(0)]])
{
    EnvParams e = envs[in.env];
    FSOut o;
    if (e.ambient.w > 0.5 && (c.flags & 1u))
        o.color = float4(bg.sample(samp, in.uv, in.env % c.bg_layers).rgb, 1.0);
    else if ((c.flags & 4u) && sp.sky_info.y > 0.5) {   // MuJoCo skybox: cube map of the view direction
        float4 ndc = float4(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 1.0, 1.0);
        float4 wp = inverse4(e.view_proj) * ndc;
        float3 d = normalize(wp.xyz / wp.w - e.cam_pos.xyz);
        o.color = float4(sky_color(atlas, samp, sp.sky_rect, sp.sky_info, d), 1.0);
    }
    else
        o.color = e.clear_color;
    o.seg = 0u;
    o.normal = float4(0, 0, 0, 0);
    return o;
}

// -------------------------------------------------------------------------------------------------
// untile: atlas -> per-env learner tensors

kernel void untile(
    texture2d<float, access::read>  color   [[texture(0)]],
    texture2d<float, access::read>  depth   [[texture(1)]],
    texture2d<uint,  access::read>  seg     [[texture(2)]],
    texture2d<float, access::read>  normal  [[texture(3)]],
    device uchar*   out_rgb    [[buffer(0)]],
    device float*   out_depth  [[buffer(1)]],
    device int*     out_seg    [[buffer(2)]],
    device float*   out_normal [[buffer(3)]],
    device const Semantic* sem [[buffer(4)]],
    device const EnvParams* envs [[buffer(5)]],
    constant Consts& c         [[buffer(6)]],
    constant uint4& outputs    [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]])
{
    uint x = tid.x, y = tid.y, e = tid.z;
    if (x >= c.tile_w || y >= c.tile_h || e >= c.n_envs) return;
    uint ax = (e % c.tiles_per_row) * c.tile_w + x;
    uint ay = (e / c.tiles_per_row) * c.tile_h + y;
    uint pix = (e * c.tile_h + y) * c.tile_w + x;
    if (outputs.x) {
        float4 cval = color.read(uint2(ax, ay));
        out_rgb[pix * 3 + 0] = uchar(clamp(cval.r, 0.0f, 1.0f) * 255.0f + 0.5f);
        out_rgb[pix * 3 + 1] = uchar(clamp(cval.g, 0.0f, 1.0f) * 255.0f + 0.5f);
        out_rgb[pix * 3 + 2] = uchar(clamp(cval.b, 0.0f, 1.0f) * 255.0f + 0.5f);
    }
    if (outputs.y) {
        float z = depth.read(uint2(ax, ay)).r;
        float n = envs[e].cam_pos.w, f = envs[e].cam_fwd.w;
        out_depth[pix] = (z >= 1.0f) ? 0.0f : n * f / (f - z * (f - n));
    }
    if (outputs.z) {
        uint s = seg.read(uint2(ax, ay)).r;
        int v = 0;
        if (s > 0) {
            Semantic sm = sem[s - 1];
            uint mode = outputs.z - 1;
            v = (mode == 0) ? int(s) : (mode == 1) ? sm.geom + 1 : sm.body + 1;
        }
        out_seg[pix] = v;
    }
    if (outputs.w) {
        float4 nv = normal.read(uint2(ax, ay));
        out_normal[pix * 3 + 0] = nv.x; out_normal[pix * 3 + 1] = nv.y; out_normal[pix * 3 + 2] = nv.z;
    }
}

// =================================================================================================
// Tier 1: hybrid ray tracing from the fragment stage (shadows, ambient occlusion, mirror reflections)
// against the sensor layer's instance acceleration structure (envs spatially separated by RTOffset).
#include <metal_raytracing>
using namespace metal::raytracing;

struct RTOffset { uint tiles_per_row; float stride; uint n_samples; uint _pad; };
struct MeshInfoT { uint i_off, i_count, _p0, _p1; };
struct InstanceDescT { packed_float3 col0, col1, col2, col3; uint options, mask, if_table_offset, as_index; };

inline float3 rt_env_offset(uint e, constant RTOffset& ro) {
    return float3(float(e % ro.tiles_per_row) * ro.stride, float(e / ro.tiles_per_row) * ro.stride, 0.0);
}

inline uint hash_u(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
inline float hash_f(uint x) { return float(hash_u(x) & 0xFFFFFFu) / 16777216.0; }

inline float3 ortho_basis_x(float3 n) {
    float3 a = fabs(n.z) < 0.99 ? float3(0, 0, 1) : float3(1, 0, 0);
    return normalize(cross(a, n));
}

inline bool occluded(instance_acceleration_structure accel, float3 origin, float3 dir, float tmax) {
    ray r(origin, dir, 0.0005, tmax);
    intersector<triangle_data, instancing> i;
    i.accept_any_intersection(true);
    return i.intersect(r, accel).type != intersection_type::none;
}

fragment FSOut geom_fs_rt(
    FSIn in0 [[stage_in]],
    uint prim [[primitive_id]],
    float3 bary [[barycentric_coord]],
    device const CornerAttr* corners   [[buffer(21)]],
    device const uint*       slot_ibase [[buffer(22)]],
    device const float*      geom_xmat [[buffer(23)]],
    device const EnvParams* envs       [[buffer(1)]],
    device const Material*  mats       [[buffer(2)]],
    device const Semantic*  sem        [[buffer(3)]],
    device const float4*    per_inst   [[buffer(4)]],
    constant Consts&        c          [[buffer(5)]],
    device const Light*     lights     [[buffer(6)]],
    device const float*     light_xpos [[buffer(7)]],
    device const float*     light_xdir [[buffer(8)]],
    constant SceneParams&   sp         [[buffer(9)]],
    instance_acceleration_structure accel [[buffer(10)]],
    constant RTOffset&      ro         [[buffer(11)]],
    device const uint2*     inst_tab   [[buffer(12)]],
    device const int*       slot_mesh  [[buffer(13)]],
    device const MeshInfoT* meshes     [[buffer(14)]],
    device const float*     verts      [[buffer(15)]],
    device const uint*      indices    [[buffer(16)]],
    device const InstanceDescT* inst   [[buffer(17)]],
    device const MaterialExt* mext     [[buffer(18)]],
    texture2d<float>        atlas      [[texture(0)]],
    sampler                 samp       [[sampler(0)]])
{
    FSIn in = CORNER_ATTRS ? corner_fetch(in0, prim, bary, corners, slot_ibase, sem, geom_xmat, c) : in0;
    EnvParams e = envs[in.env];
    Material m = mats[in.slot];
    float4 mod = per_inst[in.env * c.n_slots + in.slot];
    float3 base = m.rgba.rgb * mod.rgb;
    float4 texc = material_tex(m.rect, m.par, mext[in.slot], in.uv, in.local_pos, atlas, samp);
    base *= texc.rgb;
    float metallic = m.mrse.x, rough = max(m.mrse.y, 0.03), specw = m.mrse.z, emission = m.mrse.w;
    float3 N = normalize(in.normal_w);
    float3 V = normalize(e.cam_pos.xyz - in.world_pos);
    if (dot(N, V) < 0.0) N = -N;
    float NdotV = max(dot(N, V), 1e-4);
    float a2 = rough * rough * rough * rough;
    float3 f0 = mix(float3(0.08 * specw), base, metallic);
    float3 P = in.world_pos + rt_env_offset(in.env, ro) + N * 0.002;
    uint seed = hash_u(uint(in.clip.x) * 1973u + uint(in.clip.y) * 9277u + in.env * 26699u);
    uint ns = max(ro.n_samples, 1u);
    float3 tx = ortho_basis_x(N), ty = cross(N, tx);

    // direct lighting with ray-traced shadows (soft: cone of half-angle ~0.7 deg for the caster)
    float3 direct = float3(0.0);
    uint nl = uint(e.misc.y);
    int caster = int(e.misc.x);
    auto shadow_term = [&](float3 L, bool is_caster) -> float {
        if (!is_caster) return 1.0;
        float3 lx = ortho_basis_x(L), ly = cross(L, lx);
        float occ = 0.0;
        for (uint s = 0; s < ns; ++s) {
            float u = hash_f(seed + s * 7919u), v = hash_f(seed + s * 104729u + 17u);
            float rr = 0.012 * sqrt(u), ph = 6.2831853 * v;   // ~0.7 deg angular radius
            float3 d = normalize(L + lx * (rr * cos(ph)) + ly * (rr * sin(ph)));
            occ += occluded(accel, P, d, 100.0) ? 1.0 : 0.0;
        }
        return 1.0 - occ / float(ns);
    };
    for (uint i = 0; i < nl; ++i) {
        Light l = lights[i];
        device const float* lp = light_xpos + (in.env * c.n_light + i) * 3;
        device const float* ld = light_xdir + (in.env * c.n_light + i) * 3;
        float3 Lpos = float3(lp[0], lp[1], lp[2]);
        float3 Ldir = normalize(float3(ld[0], ld[1], ld[2]));
        float3 L; float att = 1.0;
        if (l.kind.x == 0.0) { L = -Ldir; }
        else {
            float3 dv = Lpos - in.world_pos; float dist = length(dv); L = dv / max(dist, 1e-6);
            att = 1.0 / max(l.atten.x + l.atten.y * dist + l.atten.z * dist * dist, 1e-4);
            if (l.kind.x == 2.0) {
                float cosang = dot(-L, Ldir);
                att *= cosang > cos(l.kind.z * PI / 180.0) ? pow(max(cosang, 0.0), l.kind.w) : 0.0;
            }
        }
        float s = shadow_term(L, int(i) == caster);
        if (int(i) != caster && l.kind.y > 0.5 && att > 0.0 && dot(N, L) > 0.0) {   // every other shadow-casting light
            if (l.kind.x == 0.0) s = shadow_term(L, true);
            else {          // point / spot: one ray to the light position (hard shadow)
                float3 to = Lpos + rt_env_offset(in.env, ro) - P;
                float dl = length(to);
                s = (dl > 0.003 && occluded(accel, P, to / dl, dl - 0.002)) ? 0.0 : 1.0;   // tmax > tmin always
            }
        }
        direct += shade_light(N, V, L, l.diffuse.xyz * (PI * att), base, f0, metallic, a2, NdotV) * s;
    }
    if (e.dr_light.w > 0.0) {
        float s = shadow_term(-e.dr_light.xyz, caster == MAX_LIGHTS);
        direct += shade_light(N, V, -e.dr_light.xyz, float3(PI * e.dr_light.w), base, f0, metallic, a2, NdotV) * s;
    }
    if (e.misc.z > 0.5)
        direct += shade_light(N, V, V, sp.hl_diffuse.xyz * PI, base, f0, metallic, a2, NdotV);

    // ambient occlusion: cosine-weighted hemisphere rays, 0.5 m horizon
    float ao_occ = 0.0;
    for (uint s = 0; s < ns; ++s) {
        float u = hash_f(seed + s * 6151u + 3u), v = hash_f(seed + s * 12289u + 5u);
        float rr = sqrt(u), ph = 6.2831853 * v;
        float3 d = tx * (rr * cos(ph)) + ty * (rr * sin(ph)) + N * sqrt(max(1.0 - u, 0.0));
        ao_occ += occluded(accel, P, d, 0.5) ? 1.0 : 0.0;
    }
    float ao = 1.0 - 0.5 * ao_occ / float(ns);
    float3 color = direct + e.ambient.rgb * base * ao + emission * base;

    // mirror reflection for reflective materials (MuJoCo `reflectance`) and metals: one ray
    float refl = max(m.par.w, metallic * (1.0 - rough));
    if (refl > 0.01) {
        float3 Rd = reflect(-V, N);
        ray r(P, Rd, 0.001, 50.0);
        intersector<triangle_data, instancing> isect;
        intersection_result<triangle_data, instancing> res = isect.intersect(r, accel);
        float3 rc = sp.sky_info.y > 0.5 ? sky_color(atlas, samp, sp.sky_rect, sp.sky_info, Rd)
                  : sp.sky.w > 0.5 ? sp.sky.xyz : e.clear_color.rgb;
        if (res.type != intersection_type::none) {
            uint slot2 = inst_tab[res.instance_id].y;
            uint mesh2 = uint(slot_mesh[slot2]);
            uint b = meshes[mesh2].i_off + res.primitive_id * 3;
            uint i0 = indices[b], i1 = indices[b + 1], i2 = indices[b + 2];
            float2 bc = res.triangle_barycentric_coord;
            float w0 = 1.0 - bc.x - bc.y;
            float2 uv2 = float2(verts[i0*8+6], verts[i0*8+7]) * w0 + float2(verts[i1*8+6], verts[i1*8+7]) * bc.x + float2(verts[i2*8+6], verts[i2*8+7]) * bc.y;
            float3 n2 = normalize(float3(verts[i0*8+3], verts[i0*8+4], verts[i0*8+5]) * w0 + float3(verts[i1*8+3], verts[i1*8+4], verts[i1*8+5]) * bc.x + float3(verts[i2*8+3], verts[i2*8+4], verts[i2*8+5]) * bc.y);
            InstanceDescT id2 = inst[res.instance_id];       // rigid instance: rotation columns
            float3x3 R2 = float3x3(float3(id2.col0), float3(id2.col1), float3(id2.col2));
            n2 = normalize(R2 * n2);
            if (dot(n2, Rd) > 0.0) n2 = -n2;
            Material m2 = mats[slot2];
            float4 mod2 = per_inst[in.env * c.n_slots + slot2];
            float3 base2 = m2.rgba.rgb * mod2.rgb;
            float3 lp2 = transpose(R2) * (P + Rd * res.distance - float3(id2.col3));   // hit in the geom frame
            base2 *= material_tex(m2.rect, m2.par, mext[slot2], uv2, lp2, atlas, samp).rgb;
            float3 V2 = -Rd;
            float3 d2 = float3(0.0);
            for (uint i = 0; i < nl; ++i) {
                Light l = lights[i];
                device const float* ld = light_xdir + (in.env * c.n_light + i) * 3;
                if (l.kind.x != 0.0) continue;
                float3 L = -normalize(float3(ld[0], ld[1], ld[2]));
                d2 += base2 / PI * max(dot(n2, L), 0.0) * l.diffuse.xyz * PI;
            }
            if (e.dr_light.w > 0.0) d2 += base2 / PI * max(dot(n2, -e.dr_light.xyz), 0.0) * PI * e.dr_light.w;
            if (e.misc.z > 0.5) d2 += base2 / PI * max(dot(n2, V2), 0.0) * sp.hl_diffuse.xyz * PI;
            rc = d2 + e.ambient.rgb * base2 + m2.mrse.w * base2;
        }
        color = mix(color, rc, refl);   // MuJoCo blends a constant reflectance
    }
    FSOut o;
    o.color = float4(color, m.rgba.a * mod.a * texc.a);   // alpha: used by the transparent pass only
    o.seg = in.slot + 1u;
    o.normal = float4(N, 1.0);
    return o;
}

// =================================================================================================
// Dynamic slots (metalsim.render.dynamic_mesh): flexes, skins, spatial tendons and user deformable meshes. Their
// vertex records (position, normal, uv; world frame plus the ray tracer's env grid offset when there is one) are
// generated per env each frame; one draw per object, instance = env, local indices. Fragments go through geom_fs /
// geom_fs_rt like every geom.
struct DynDraw { uint rbase, vstride, slot, tpr; float stride, _a, _b, _c; };

inline float3 dyn_record(device const float* rec, uint vid, uint env, constant DynDraw& dd, thread float3& n, thread float2& uv) {
    device const float* r = rec + (dd.rbase + env * dd.vstride + vid) * 8;
    n = float3(r[3], r[4], r[5]); uv = float2(r[6], r[7]);
    float3 off = dd.stride == 0.0f ? float3(0.0) : float3(float(env % dd.tpr) * dd.stride, float(env / dd.tpr) * dd.stride, 0.0);
    return float3(r[0], r[1], r[2]) - off;
}

vertex VSOut dyn_vs(uint vid [[vertex_id]], uint env [[instance_id]],
                    device const float* rec [[buffer(0)]], device const EnvParams* envs [[buffer(1)]],
                    constant DynDraw& dd [[buffer(2)]])
{
    float3 n; float2 uv;
    float3 p = dyn_record(rec, vid, env, dd, n, uv);
    EnvParams e = envs[env];
    float4 clip = tile_clip(e.view_proj * float4(p, 1.0), e.tile);
    VSOut o;
    o.clip = clip;
    clip_to_tile(o.clip_distance, clip, e.tile);
    o.world_pos = p; o.normal_w = n; o.uv = uv; o.env = env; o.slot = dd.slot;
    return o;
}

vertex ShadowOut dyn_shadow_vs(uint vid [[vertex_id]], uint env [[instance_id]],
                               device const float* rec [[buffer(0)]], device const EnvParams* envs [[buffer(1)]],
                               constant DynDraw& dd [[buffer(2)]])
{
    float3 n; float2 uv;
    float3 p = dyn_record(rec, vid, env, dd, n, uv);
    EnvParams e = envs[env];
    float4 clip = tile_clip(e.light_vp * float4(p, 1.0), e.shadow_tile);
    ShadowOut o;
    o.clip = clip;
    clip_to_tile(o.clip_distance, clip, e.shadow_tile);
    return o;
}

vertex ShadowViewOut dyn_shadow_view_vs(uint vid [[vertex_id]], uint env [[instance_id]],
                                        device const float* rec [[buffer(0)]], device const EnvParams* envs [[buffer(1)]],
                                        constant DynDraw& dd [[buffer(2)]], device const ShadowView* views [[buffer(6)]],
                                        constant Consts& c [[buffer(7)]], constant uint& view [[buffer(8)]])
{   // the dynamic slots in the extra lights' shadow maps (shadow_view_vs for generated records)
    float3 n; float2 uv;
    float3 p = dyn_record(rec, vid, env, dd, n, uv);
    EnvParams e = envs[env];
    ShadowView sv = views[env * c.n_shadow_views + view];
    float4 clip = tile_clip(sv.vp * float4(p, 1.0), e.shadow_tile);
    ShadowViewOut o;
    o.clip = clip;
    clip_to_tile(o.clip_distance, clip, e.shadow_tile);
    o.world_pos = p; o.env = env;
    return o;
}
