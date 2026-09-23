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

#define MAX_LIGHTS 8

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
};

struct Material {
    float4 rgba;
    float4 mrse;              // metallic, roughness, specular, emission
    float4 rect;              // atlas u0, v0, du, dv
    float4 par;               // texrepeat.xy, textured flag, reflectance
};

struct Semantic { int geom, body, root, group; };

struct Consts {
    uint n_envs, n_geoms, n_slots, tiles_per_row;
    uint tile_w, tile_h, atlas_w, atlas_h;
    uint n_cam, bg_layers, flags, n_light;      // flags bit0: draw backgrounds, bit1: shadows
    uint shadow_w, shadow_h, shadow_atlas_w, shadow_atlas_h;
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
};

struct FSIn {
    float4 clip [[position]];
    float3 world_pos;
    float3 normal_w;
    float2 uv;
    uint   env  [[flat]];
    uint   slot [[flat]];
};

inline void instance_world(uint inst, device const uint2* inst_tab, device const int* slot_geom,
                           device const float* geom_xpos, device const float* geom_xmat, constant Consts& c,
                           float3 pos_l, float3 nrm_l, thread float3& p, thread float3& n, thread uint& env, thread uint& slot)
{
    uint2 pair = inst_tab[inst];
    env = pair.x; slot = pair.y;
    uint g = uint(slot_geom[slot]);
    uint gi = env * c.n_geoms + g;
    device const float* xp = geom_xpos + gi * 3;
    float3x3 R = load_mat33(geom_xmat + gi * 9);
    p = R * pos_l + float3(xp[0], xp[1], xp[2]);
    n = R * nrm_l;
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
    o.world_pos = p; o.normal_w = n; o.uv = in.uv; o.env = env; o.slot = slot;
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

fragment FSOut geom_fs(
    FSIn in [[stage_in]],
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
    sampler                 shadow_samp [[sampler(1)]])
{
    EnvParams e = envs[in.env];
    Material m = mats[in.slot];
    float4 mod = per_inst[in.env * c.n_slots + in.slot];
    float3 base = m.rgba.rgb * mod.rgb;
    if (m.par.z > 0.5) {
        float2 tuv = fract(in.uv * m.par.xy);
        float2 auv = m.rect.xy + tuv * m.rect.zw;
        base *= atlas.sample(samp, auv).rgb;
    }
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
    o.color = float4(color, 1.0);
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
                     texture2d_array<float> bg [[texture(1)]], sampler samp [[sampler(0)]])
{
    EnvParams e = envs[in.env];
    FSOut o;
    if (e.ambient.w > 0.5 && (c.flags & 1u))
        o.color = float4(bg.sample(samp, in.uv, in.env % c.bg_layers).rgb, 1.0);
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
    FSIn in [[stage_in]],
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
    texture2d<float>        atlas      [[texture(0)]],
    sampler                 samp       [[sampler(0)]])
{
    EnvParams e = envs[in.env];
    Material m = mats[in.slot];
    float4 mod = per_inst[in.env * c.n_slots + in.slot];
    float3 base = m.rgba.rgb * mod.rgb;
    if (m.par.z > 0.5) {
        float2 tuv = fract(in.uv * m.par.xy);
        base *= atlas.sample(samp, m.rect.xy + tuv * m.rect.zw).rgb;
    }
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
        float3 rc = sp.sky.w > 0.5 ? sp.sky.xyz : e.clear_color.rgb;
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
            if (m2.par.z > 0.5) base2 *= atlas.sample(samp, m2.rect.xy + fract(uv2 * m2.par.xy) * m2.rect.zw).rgb;
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
    o.color = float4(color, 1.0);
    o.seg = in.slot + 1u;
    o.normal = float4(N, 1.0);
    return o;
}
