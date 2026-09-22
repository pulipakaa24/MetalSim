// orchard tier-0 renderer: batched tile atlas, instanced unique meshes, PBR (GGX) shading with one
// directional light + hemisphere ambient, optional per-env background compositing, RGB / depth /
// segmentation / normal targets, and an untile kernel that writes learner-layout tensors.
//
// Instance transforms are read straight from the physics state (MuJoCo Warp geom_xpos / geom_xmat
// in unified memory): there is no host pack step. Cameras come from cam_xpos / cam_xmat (so
// body-mounted cameras move with the robot) with an optional per-env pose delta and intrinsics.
#include <metal_stdlib>
using namespace metal;

struct EnvParams {            // built on the GPU by build_env_params
    float4x4 view_proj;
    float4   tile;            // ndc offset x, y, scale x, y
    float4   light_dir;       // world-space direction the light travels (normalized), w = intensity
    float4   ambient;         // rgb, w = background mode (0 clear color, 1 background texture)
    float4   clear_color;
    float4   cam_pos;         // world position, w = near
    float4   misc;            // far, seg mode, unused
};

struct CameraSpec {           // per env, host-written (static unless randomized)
    float4 intrinsics;        // fx, fy, cx, cy (pixels)
    float4 pos_delta;         // xyz offset in world frame (domain randomization), w unused
    float4 rot_delta;         // quaternion (w,x,y,z) applied after the model camera rotation
    float4 light;             // xyz light direction, w intensity
    float4 ambient;           // rgb, w background mode
    float4 clear_color;
    int    cam_id;            // model camera index (0 == first camera)
    int    _pad0, _pad1, _pad2;
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
    uint n_cam;               // model cameras (stride of cam_xpos per env)
    uint bg_layers;
    uint flags;               // bit0: draw backgrounds
    uint _pad;
};

// -------------------------------------------------------------------------------------------------
// math helpers

inline float3x3 quat_to_mat(float4 q) {    // q = (w, x, y, z)
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return float3x3(
        float3(1 - 2*(y*y + z*z), 2*(x*y + w*z),     2*(x*z - w*y)),
        float3(2*(x*y - w*z),     1 - 2*(x*x + z*z), 2*(y*z + w*x)),
        float3(2*(x*z + w*y),     2*(y*z - w*x),     1 - 2*(x*x + y*y)));  // columns
}

inline float3x3 load_mat33(device const float* p) {   // MuJoCo row-major 3x3 -> metal column matrix
    return float3x3(float3(p[0], p[3], p[6]), float3(p[1], p[4], p[7]), float3(p[2], p[5], p[8]));
}

// -------------------------------------------------------------------------------------------------
// per-env parameter build (one thread per env)

kernel void build_env_params(
    device EnvParams*        envs      [[buffer(0)]],
    device const CameraSpec* cams      [[buffer(1)]],
    device const float*      cam_xpos  [[buffer(2)]],   // (n_envs, n_cam, 3)
    device const float*      cam_xmat  [[buffer(3)]],   // (n_envs, n_cam, 9)
    constant Consts&         c         [[buffer(4)]],
    uint e [[thread_position_in_grid]])
{
    if (e >= c.n_envs) return;
    CameraSpec cs = cams[e];
    uint ci = uint(max(cs.cam_id, 0));
    device const float* xp = cam_xpos + (e * c.n_cam + ci) * 3;
    float3 pos = float3(xp[0], xp[1], xp[2]) + cs.pos_delta.xyz;
    float3x3 R = load_mat33(cam_xmat + (e * c.n_cam + ci) * 9);   // camera axes in world (columns)
    R = R * quat_to_mat(cs.rot_delta);
    // MuJoCo camera looks along -Z with +Y up; view = R^T (p - pos)
    float3x3 Rt = transpose(R);
    float4x4 V = float4x4(
        float4(Rt[0], 0), float4(Rt[1], 0), float4(Rt[2], 0),
        float4(-(Rt * pos), 1));
    float fx = cs.intrinsics.x, fy = cs.intrinsics.y, cx = cs.intrinsics.z, cy = cs.intrinsics.w;
    float w = float(c.tile_w), h = float(c.tile_h);
    float n = 0.01, f = 10.0;
    float4x4 P = float4x4(
        float4(2*fx/w, 0, 0, 0),
        float4(0, 2*fy/h, 0, 0),
        float4(1 - 2*cx/w, 2*cy/h - 1, f/(n - f), -1),
        float4(0, 0, n*f/(n - f), 0));
    EnvParams p;
    p.view_proj = P * V;
    float sx = float(c.tile_w) / float(c.atlas_w), sy = float(c.tile_h) / float(c.atlas_h);
    uint col = e % c.tiles_per_row, row = e / c.tiles_per_row;
    p.tile = float4(-1 + (2*col + 1) * sx, 1 - (2*row + 1) * sy, sx, sy);
    float3 L = cs.light.xyz;
    float ll = length(L);
    p.light_dir = float4(ll > 0 ? L / ll : float3(0, 0, -1), cs.light.w);
    p.ambient = cs.ambient;
    p.clear_color = cs.clear_color;
    p.cam_pos = float4(pos, n);
    p.misc = float4(f, 0, 0, 0);
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

vertex VSOut geom_vs(
    VSIn in [[stage_in]],
    uint inst [[instance_id]],
    device const EnvParams* envs      [[buffer(1)]],
    device const uint2*     inst_tab  [[buffer(2)]],   // (env, slot)
    device const int*       slot_geom [[buffer(3)]],   // slot -> model geom id
    device const float*     geom_xpos [[buffer(4)]],   // (n_envs, n_geoms, 3)
    device const float*     geom_xmat [[buffer(5)]],   // (n_envs, n_geoms, 9)
    device const float4*    per_inst  [[buffer(6)]],   // (n_envs, n_slots) color modulation rgba
    constant Consts&        c         [[buffer(7)]])
{
    uint2 pair = inst_tab[inst];
    uint env = pair.x, slot = pair.y;
    uint g = uint(slot_geom[slot]);
    uint gi = env * c.n_geoms + g;
    device const float* xp = geom_xpos + gi * 3;
    float3x3 R = load_mat33(geom_xmat + gi * 9);
    float3 p = R * in.pos + float3(xp[0], xp[1], xp[2]);
    float3 n = R * in.normal;
    EnvParams e = envs[env];
    float4 clip = e.view_proj * float4(p, 1.0);
    clip = float4(clip.x * e.tile.z + e.tile.x * clip.w,
                  clip.y * e.tile.w + e.tile.y * clip.w, clip.z, clip.w);
    VSOut o;
    o.clip = clip;
    // clip-space tile bounds: x in [(tx - sx) w, (tx + sx) w], y likewise
    o.clip_distance[0] = clip.x - (e.tile.x - e.tile.z) * clip.w;
    o.clip_distance[1] = (e.tile.x + e.tile.z) * clip.w - clip.x;
    o.clip_distance[2] = clip.y - (e.tile.y - e.tile.w) * clip.w;
    o.clip_distance[3] = (e.tile.y + e.tile.w) * clip.w - clip.y;
    o.world_pos = p;
    o.normal_w = n;
    o.uv = in.uv;
    o.env = env;
    o.slot = slot;
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

fragment FSOut geom_fs(
    VSOut in [[stage_in]],
    device const EnvParams* envs     [[buffer(1)]],
    device const Material*  mats     [[buffer(2)]],
    device const Semantic*  sem      [[buffer(3)]],
    device const float4*    per_inst [[buffer(4)]],
    constant Consts&        c        [[buffer(5)]],
    texture2d<float>        atlas    [[texture(0)]],
    sampler                 samp     [[sampler(0)]])
{
    // clip to the env's tile (instances of one env must not bleed into neighbours)
    float col = float(in.env % c.tiles_per_row), row = float(in.env / c.tiles_per_row);
    float tw = float(c.tile_w), th = float(c.tile_h);
    if (in.clip.x < col * tw || in.clip.x >= (col + 1) * tw || in.clip.y < row * th || in.clip.y >= (row + 1) * th)
        discard_fragment();
    EnvParams e = envs[in.env];
    Material m = mats[in.slot];
    float4 mod = per_inst[in.env * c.n_slots + in.slot];
    float3 base = m.rgba.rgb * mod.rgb;
    if (m.par.z > 0.5) {
        float2 tuv = fract(in.uv * m.par.xy);
        float2 auv = m.rect.xy + tuv * m.rect.zw;
        base *= atlas.sample(samp, auv).rgb;
    }
    float metallic = m.mrse.x, rough = max(m.mrse.y, 0.03), spec = m.mrse.z, emission = m.mrse.w;
    float3 N = normalize(in.normal_w);
    float3 V = normalize(e.cam_pos.xyz - in.world_pos);
    if (dot(N, V) < 0.0) N = -N;                       // two-sided
    float3 L = -e.light_dir.xyz;
    float3 H = normalize(L + V);
    float NdotL = max(dot(N, L), 0.0), NdotV = max(dot(N, V), 1e-4);
    float NdotH = max(dot(N, H), 0.0), VdotH = max(dot(V, H), 0.0);
    float a2 = rough * rough * rough * rough;
    float3 f0 = mix(float3(0.08 * spec), base, metallic);
    float3 F = F_schlick(f0, VdotH);
    float3 spec_brdf = D_ggx(NdotH, a2) * G_smith(NdotV, NdotL, a2) * F / (4.0 * NdotV * NdotL + 1e-4);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    float3 direct = (kd * base / PI + spec_brdf) * NdotL * e.light_dir.w;
    float hemi = 0.5 + 0.5 * N.z;                        // sky/ground hemisphere ambient
    float3 ambient = e.ambient.rgb * base * hemi;
    float3 color = direct + ambient + emission * base;
    FSOut o;
    o.color = float4(color, 1.0);
    o.seg = in.slot + 1u;
    o.normal = float4(N, 1.0);
    return o;
}

// -------------------------------------------------------------------------------------------------
// background pass (one quad per env, behind everything)

struct BGOut {
    float4 clip [[position]];
    float2 uv;
    uint env [[flat]];
};

vertex BGOut bg_vs(uint vid [[vertex_id]], uint env [[instance_id]],
                   device const EnvParams* envs [[buffer(1)]])
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

fragment FSOut bg_fs(BGOut in [[stage_in]],
                     device const EnvParams* envs [[buffer(1)]],
                     constant Consts& c [[buffer(5)]],
                     texture2d_array<float> bg [[texture(1)]],
                     sampler samp [[sampler(0)]])
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
// untile: atlas -> per-env learner tensors. One thread per output pixel of the whole batch.

kernel void untile(
    texture2d<float, access::read>  color   [[texture(0)]],
    texture2d<float, access::read>  depth   [[texture(1)]],
    texture2d<uint,  access::read>  seg     [[texture(2)]],
    texture2d<float, access::read>  normal  [[texture(3)]],
    device uchar*   out_rgb    [[buffer(0)]],   // (n_envs, h, w, 3) uint8, or null
    device float*   out_depth  [[buffer(1)]],   // (n_envs, h, w) metric depth, or null
    device int*     out_seg    [[buffer(2)]],   // (n_envs, h, w) semantic id, or null
    device float*   out_normal [[buffer(3)]],   // (n_envs, h, w, 3), or null
    device const Semantic* sem [[buffer(4)]],
    device const EnvParams* envs [[buffer(5)]],
    constant Consts& c         [[buffer(6)]],
    constant uint4& outputs    [[buffer(7)]],   // which outputs are bound: rgb, depth, seg(mode+1), normal
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
        float z = depth.read(uint2(ax, ay)).r;               // [0,1] reversed-style GL depth
        float n = envs[e].cam_pos.w, f = envs[e].misc.x;
        out_depth[pix] = (z >= 1.0f) ? 0.0f : n * f / (f - z * (f - n));
    }
    if (outputs.z) {
        uint s = seg.read(uint2(ax, ay)).r;
        int v = 0;
        if (s > 0) {
            Semantic sm = sem[s - 1];
            uint mode = outputs.z - 1;                  // 0 slot+1, 1 model geom id+1, 2 body id+1
            v = (mode == 0) ? int(s) : (mode == 1) ? sm.geom + 1 : sm.body + 1;
        }
        out_seg[pix] = v;
    }
    if (outputs.w) {
        float4 nv = normal.read(uint2(ax, ay));
        out_normal[pix * 3 + 0] = nv.x;
        out_normal[pix * 3 + 1] = nv.y;
        out_normal[pix * 3 + 2] = nv.z;
    }
}
