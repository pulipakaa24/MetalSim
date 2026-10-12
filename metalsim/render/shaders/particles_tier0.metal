// metalsim particle rendering, raster (tier 0): render grains drawn as instanced grain meshes in tier 0's geometry and
// shadow passes. Compiled appended to tier0.metal (metalsim/render/particles.py GrainRaster), so it uses tier 0's
// EnvParams / VSOut / FSIn / FSOut / Light / SceneParams / Consts and its shading helpers, and writes the same
// colour / segmentation / normal attachments. Grain transforms are the instance descriptors written by
// particles.metal's grain_instances (no env offset in tier 0: every env renders in its own tile). All grain meshes
// share one topology (same icosphere level), so one instanced draw covers every mesh: the vertex shader picks the
// grain's own mesh by its instance descriptor.

struct GrainInstT { packed_float3 col0, col1, col2, col3; uint options, mask, if_table_offset, as_index; };
struct GrainRasterConsts { uint per_env, verts_per_mesh, as_base, seg_id; };
struct GrainMatT { float4 pal[8]; float4 p0; float4 mrse; float4 p1; };

inline bool grain_vertex(uint vid, uint inst, device const float* gverts, device const GrainInstT* gi,
                         constant GrainRasterConsts& gc, thread float3& p, thread float3& n)
{
    GrainInstT d = gi[inst];
    uint v = (d.as_index - gc.as_base) * gc.verts_per_mesh + vid;
    float3 pl = float3(gverts[v * 8], gverts[v * 8 + 1], gverts[v * 8 + 2]);
    float3 nl = float3(gverts[v * 8 + 3], gverts[v * 8 + 4], gverts[v * 8 + 5]);
    float3x3 R = float3x3(float3(d.col0), float3(d.col1), float3(d.col2));
    p = R * pl + float3(d.col3);
    n = R * nl;
    return d.mask != 0u;
}

vertex VSOut grain_vs(
    uint vid [[vertex_id]], uint inst [[instance_id]],
    device const float*             gverts [[buffer(0)]],
    device const EnvParams*         envs   [[buffer(1)]],
    device const GrainInstT*        gi     [[buffer(2)]],
    constant GrainRasterConsts&     gc     [[buffer(3)]])
{
    float3 p, n;
    bool ok = grain_vertex(vid, inst, gverts, gi, gc, p, n);
    uint env = inst / gc.per_env;
    EnvParams e = envs[env];
    float4 clip = tile_clip(e.view_proj * float4(p, 1.0), e.tile);
    VSOut o;
    o.clip = clip;
    clip_to_tile(o.clip_distance, clip, e.tile);
    if (!ok) { for (int k = 0; k < 4; ++k) o.clip_distance[k] = -1.0; }   // culled grain: clipped away
    o.world_pos = p; o.normal_w = n; o.uv = float2(0.0); o.env = env; o.slot = inst;   // slot carries the grain index
    return o;
}

vertex ShadowOut grain_shadow_vs(
    uint vid [[vertex_id]], uint inst [[instance_id]],
    device const float*             gverts [[buffer(0)]],
    device const EnvParams*         envs   [[buffer(1)]],
    device const GrainInstT*        gi     [[buffer(2)]],
    constant GrainRasterConsts&     gc     [[buffer(3)]])
{
    float3 p, n;
    bool ok = grain_vertex(vid, inst, gverts, gi, gc, p, n);
    EnvParams e = envs[inst / gc.per_env];
    float4 clip = tile_clip(e.light_vp * float4(p, 1.0), e.shadow_tile);
    ShadowOut o;
    o.clip = clip;
    clip_to_tile(o.clip_distance, clip, e.shadow_tile);
    if (!ok) { for (int k = 0; k < 4; ++k) o.clip_distance[k] = -1.0; }
    return o;
}

// the same palette pick as the path tracer's grain_material (shaders/pathtrace.metal), per grain identity key
inline float3 grain_base(constant GrainMatT& g, uint key) {
    uint hh = hash_u(key * 0x9E3779B1u + 0x85EBCA6Bu);
    float u = float(hh & 0xFFFFu) / 65536.0, v = float(hh >> 16) / 65536.0;
    uint n = clamp(uint(g.p0.x), 1u, 8u), k = 0;
    while (k + 1 < n && u > g.pal[k].w) ++k;
    return g.pal[k].rgb * max(1.0 + g.p0.y * (2.0 * v - 1.0), 0.0);
}

// geom_fs's lighting (model lights with the shadow-map caster, DR light, headlight, flat ambient) on a grain
fragment FSOut grain_fs(
    FSIn in [[stage_in]],
    device const EnvParams* envs       [[buffer(1)]],
    constant GrainMatT&     gm         [[buffer(2)]],
    constant GrainRasterConsts& gc     [[buffer(3)]],
    constant Consts&        c          [[buffer(5)]],
    device const Light*     lights     [[buffer(6)]],
    device const float*     light_xpos [[buffer(7)]],
    device const float*     light_xdir [[buffer(8)]],
    constant SceneParams&   sp         [[buffer(9)]],
    device const uint*      keys       [[buffer(10)]],   // per-grain identity key (particles.metal)
    depth2d<float>          shadow     [[texture(2)]],
    sampler                 shadow_samp [[sampler(1)]])
{
    EnvParams e = envs[in.env];
    float3 base = grain_base(gm, keys[in.slot]);
    float metallic = gm.mrse.x, rough = max(gm.mrse.y, 0.03), specw = gm.mrse.z;
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
        float3 Ldir = normalize(float3(ld[0], ld[1], ld[2]));
        float3 L; float att = 1.0;
        if (l.kind.x == 0.0) { L = -Ldir; }
        else {
            float3 dv = float3(lp[0], lp[1], lp[2]) - in.world_pos; float dist = length(dv); L = dv / max(dist, 1e-6);
            att = 1.0 / max(l.atten.x + l.atten.y * dist + l.atten.z * dist * dist, 1e-4);
            if (l.kind.x == 2.0) {
                float cosang = dot(-L, Ldir);
                att *= cosang > cos(l.kind.z * PI / 180.0) ? pow(max(cosang, 0.0), l.kind.w) : 0.0;
            }
        }
        float s = (int(i) == caster) ? vis : 1.0;
        direct += shade_light(N, V, L, l.diffuse.xyz * (PI * att), base, f0, metallic, a2, NdotV) * s;
    }
    if (e.dr_light.w > 0.0) {
        float s = (caster == MAX_LIGHTS) ? vis : 1.0;
        direct += shade_light(N, V, -e.dr_light.xyz, float3(PI * e.dr_light.w), base, f0, metallic, a2, NdotV) * s;
    }
    if (e.misc.z > 0.5)
        direct += shade_light(N, V, V, sp.hl_diffuse.xyz * PI, base, f0, metallic, a2, NdotV);
    FSOut o;
    o.color = float4(direct + e.ambient.rgb * base, 1.0);
    o.seg = gc.seg_id;
    o.normal = float4(N, 1.0);
    return o;
}
