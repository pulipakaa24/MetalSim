// metalsim tier 2: unidirectional path tracer on Metal hardware ray tracing.
//
// Per (pixel, env): `spp` paths per call accumulated into an HDR buffer (progressive), up to
// `max_bounces` bounces, next-event estimation for directional lights (shadow rays), BRDF sampling
// (Lambert diffuse + GGX specular with Fresnel, same material model and light radiances as tiers
// 0/1: MuJoCo diffuse x pi), Russian roulette after 3 bounces, environment radiance = model sky
// colour (uniform), emissive materials as area sources through BRDF sampling. Primary-hit depth,
// normal and segmentation are written for annotators. Envs share one instance acceleration
// structure and are spatially separated by RTOffset (see rt.metal).
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

struct Material { float4 rgba; float4 mrse; float4 rect; float4 par; };
struct Light { float4 kind; float4 diffuse; float4 specular; float4 ambient; float4 atten; };
struct SceneParams { float4 bounds; float4 hl_ambient; float4 hl_diffuse; float4 hl_specular; float4 sky; };
struct MeshInfo { uint i_off, i_count, _p0, _p1; };
struct InstanceDesc { packed_float3 col0, col1, col2, col3; uint options, mask, if_table_offset, as_index; };
struct PTConsts {
    uint n_envs, width, height, n_cam;
    uint n_light, n_slots, spp, max_bounces;
    uint frame, tiles_per_row, flags, seed;     // flags bit0: reset accumulation
    float stride, exposure, _f0, _f1;
};
struct CamSpec { float4 intrinsics; float4 pos_delta; float4 rot_delta; float4 light; float4 ambient; float4 clear_color; int cam_id, p0, p1, p2; };

constant float PI = 3.14159265358979f;

inline float3x3 load_mat33(device const float* p) {
    return float3x3(float3(p[0], p[3], p[6]), float3(p[1], p[4], p[7]), float3(p[2], p[5], p[8]));
}
inline float3x3 quat_to_mat(float4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return float3x3(float3(1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)),
                    float3(2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)),
                    float3(2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)));
}
inline uint hash_u(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
struct Rng { uint s; };
inline float rnd(thread Rng& r) { r.s = hash_u(r.s + 0x9e3779b9u); return float(r.s & 0xFFFFFFu) / 16777216.0; }
inline void basis(float3 n, thread float3& t, thread float3& b) {
    float3 a = fabs(n.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    t = normalize(cross(a, n)); b = cross(n, t);
}
inline float3 cosine_sample(float3 n, float u, float v) {
    float3 t, b; basis(n, t, b);
    float r = sqrt(u), ph = 2.0 * PI * v;
    return normalize(t * (r * cos(ph)) + b * (r * sin(ph)) + n * sqrt(max(1.0 - u, 0.0)));
}
inline float D_ggx(float NdotH, float a2) { float d = NdotH * NdotH * (a2 - 1.0) + 1.0; return a2 / (PI * d * d + 1e-7); }
inline float G_smith(float NdotV, float NdotL, float a2) {
    float gv = 2.0 * NdotV / (NdotV + sqrt(a2 + (1.0 - a2) * NdotV * NdotV) + 1e-7);
    float gl = 2.0 * NdotL / (NdotL + sqrt(a2 + (1.0 - a2) * NdotL * NdotL) + 1e-7);
    return gv * gl;
}
inline float3 F_schlick(float3 f0, float c) { return f0 + (1.0 - f0) * pow(1.0 - c, 5.0); }

// full BRDF value (diffuse + specular), for NEE and for evaluating the sampled direction
inline float3 brdf(float3 N, float3 V, float3 L, float3 base, float3 f0, float metallic, float a2) {
    float NdotL = max(dot(N, L), 0.0), NdotV = max(dot(N, V), 1e-4);
    if (NdotL <= 0.0) return float3(0.0);
    float3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0), VdotH = max(dot(V, H), 0.0);
    float3 F = F_schlick(f0, VdotH);
    float3 spec = D_ggx(NdotH, a2) * G_smith(NdotV, NdotL, a2) * F / (4.0 * NdotV * NdotL + 1e-4);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    return kd * base / PI + spec;
}

struct Hit { bool ok; float t; float3 p; float3 n; float2 uv; uint slot; };

inline Hit trace(instance_acceleration_structure accel, float3 o, float3 d, float tmax,
                 device const uint2* inst_tab, device const int* slot_mesh, device const MeshInfo* meshes,
                 device const float* verts, device const uint* indices, device const InstanceDesc* inst)
{
    Hit h; h.ok = false;
    ray r(o, d, 0.0005, tmax);
    intersector<triangle_data, instancing> isect;
    intersection_result<triangle_data, instancing> res = isect.intersect(r, accel);
    if (res.type == intersection_type::none) return h;
    uint slot = inst_tab[res.instance_id].y;
    uint mesh = uint(slot_mesh[slot]);
    uint b = meshes[mesh].i_off + res.primitive_id * 3;
    uint i0 = indices[b], i1 = indices[b + 1], i2 = indices[b + 2];
    float2 bc = res.triangle_barycentric_coord; float w0 = 1.0 - bc.x - bc.y;
    float3 n0 = float3(verts[i0*8+3], verts[i0*8+4], verts[i0*8+5]);
    float3 n1 = float3(verts[i1*8+3], verts[i1*8+4], verts[i1*8+5]);
    float3 n2 = float3(verts[i2*8+3], verts[i2*8+4], verts[i2*8+5]);
    float2 uv = float2(verts[i0*8+6], verts[i0*8+7]) * w0 + float2(verts[i1*8+6], verts[i1*8+7]) * bc.x + float2(verts[i2*8+6], verts[i2*8+7]) * bc.y;
    InstanceDesc id = inst[res.instance_id];
    float3x3 R = float3x3(float3(id.col0), float3(id.col1), float3(id.col2));
    float3 n = normalize(R * (n0 * w0 + n1 * bc.x + n2 * bc.y));
    if (dot(n, d) > 0.0) n = -n;
    h.ok = true; h.t = res.distance; h.p = o + d * res.distance; h.n = n; h.uv = uv; h.slot = slot;
    return h;
}

inline float3 material_base(Material m, float4 mod, float2 uv, texture2d<float> atlas, sampler samp) {
    float3 base = m.rgba.rgb * mod.rgb;
    if (m.par.z > 0.5) base *= atlas.sample(samp, m.rect.xy + fract(uv * m.par.xy) * m.rect.zw, level(0.0)).rgb;
    return base;
}

kernel void path_trace(
    instance_acceleration_structure accel [[buffer(0)]],
    device const uint2*     inst_tab   [[buffer(1)]],
    device const int*       slot_mesh  [[buffer(2)]],
    device const MeshInfo*  meshes     [[buffer(3)]],
    device const float*     verts      [[buffer(4)]],
    device const uint*      indices    [[buffer(5)]],
    device const InstanceDesc* inst    [[buffer(6)]],
    device const Material*  mats       [[buffer(7)]],
    device const float4*    per_inst   [[buffer(8)]],
    device const Light*     lights     [[buffer(9)]],
    device const float*     light_xpos [[buffer(10)]],
    device const float*     light_xdir [[buffer(11)]],
    constant SceneParams&   sp         [[buffer(12)]],
    device const CamSpec*   cams       [[buffer(13)]],
    device const float*     cam_xpos   [[buffer(14)]],
    device const float*     cam_xmat   [[buffer(15)]],
    constant PTConsts&      c          [[buffer(16)]],
    device float*           accum      [[buffer(17)]],   // (n_envs, h, w, 4): rgb sum, sample count
    device uchar*           out_rgb    [[buffer(18)]],   // (n_envs, h, w, 3) tone-mapped
    device float*           out_depth  [[buffer(19)]],   // (n_envs, h, w)
    device int*             out_seg    [[buffer(20)]],   // (n_envs, h, w) slot+1
    device float*           out_normal [[buffer(21)]],   // (n_envs, h, w, 3)
    texture2d<float>        atlas      [[texture(0)]],
    sampler                 samp       [[sampler(0)]],
    uint3 tid [[thread_position_in_grid]])
{
    uint x = tid.x, y = tid.y, e = tid.z;
    if (x >= c.width || y >= c.height || e >= c.n_envs) return;
    uint pix = (e * c.height + y) * c.width + x;
    CamSpec cs = cams[e];
    uint ci = e * c.n_cam + uint(max(cs.cam_id, 0));
    float3 cpos = float3(cam_xpos[ci * 3], cam_xpos[ci * 3 + 1], cam_xpos[ci * 3 + 2]) + cs.pos_delta.xyz;
    float3x3 R = load_mat33(cam_xmat + ci * 9) * quat_to_mat(cs.rot_delta);
    float3 env_off = float3(float(e % c.tiles_per_row) * c.stride, float(e / c.tiles_per_row) * c.stride, 0.0);
    Rng rng; rng.s = hash_u(pix * 7919u + c.frame * 104729u + c.seed);
    float3 sum = float3(0.0);
    float depth0 = 0.0; float3 normal0 = float3(0.0); int seg0 = 0;
    for (uint s = 0; s < c.spp; ++s) {
        float jx = rnd(rng), jy = rnd(rng);
        float3 dl = float3((float(x) + jx - cs.intrinsics.z) / cs.intrinsics.x, -(float(y) + jy - cs.intrinsics.w) / cs.intrinsics.y, -1.0);
        float3 d = normalize(R * dl);
        float3 o = cpos + env_off;
        float3 throughput = float3(1.0);
        float3 radiance = float3(0.0);
        for (uint bounce = 0; bounce <= c.max_bounces; ++bounce) {
            Hit h = trace(accel, o, d, 1000.0, inst_tab, slot_mesh, meshes, verts, indices, inst);
            if (!h.ok) {
                radiance += throughput * (sp.sky.w > 0.5 ? sp.sky.xyz : cs.clear_color.rgb);
                break;
            }
            Material m = mats[h.slot];
            float4 mod = per_inst[e * c.n_slots + h.slot];
            float3 base = material_base(m, mod, h.uv, atlas, samp);
            float metallic = m.mrse.x, rough = max(m.mrse.y, 0.03), specw = m.mrse.z, emission = m.mrse.w;
            float a2 = rough * rough * rough * rough;
            float3 f0 = mix(float3(0.08 * specw), base, metallic);
            float3 N = h.n; float3 V = -d;
            if (bounce == 0 && s == 0) { depth0 = -dot(h.p - o, R[2]); normal0 = N; seg0 = int(h.slot) + 1; }
            radiance += throughput * emission * base;
            // next-event estimation: model lights (directional and point/spot) with shadow rays
            for (uint i = 0; i < c.n_light; ++i) {
                Light l = lights[i];
                device const float* lp = light_xpos + (e * c.n_light + i) * 3;
                device const float* ld = light_xdir + (e * c.n_light + i) * 3;
                float3 L; float att = 1.0; float tmax = 1000.0;
                if (l.kind.x == 0.0) { L = -normalize(float3(ld[0], ld[1], ld[2])); }
                else {
                    float3 dv = float3(lp[0], lp[1], lp[2]) + env_off - h.p; float dist = length(dv); L = dv / max(dist, 1e-6); tmax = dist - 0.001;
                    att = 1.0 / max(l.atten.x + l.atten.y * dist + l.atten.z * dist * dist, 1e-4);
                    if (l.kind.x == 2.0) { float ca = dot(-L, normalize(float3(ld[0], ld[1], ld[2]))); att *= ca > cos(l.kind.z * PI / 180.0) ? pow(max(ca, 0.0), l.kind.w) : 0.0; }
                }
                float NdotL = dot(N, L);
                if (NdotL <= 0.0 || att <= 0.0) continue;
                Hit sh = trace(accel, h.p + N * 0.001, L, tmax, inst_tab, slot_mesh, meshes, verts, indices, inst);
                if (sh.ok) continue;
                radiance += throughput * brdf(N, V, L, base, f0, metallic, a2) * NdotL * l.diffuse.xyz * (PI * att);
            }
            if (cs.light.w > 0.0) {   // per-env DR directional light
                float3 L = -normalize(cs.light.xyz);
                float NdotL = dot(N, L);
                if (NdotL > 0.0) {
                    Hit sh = trace(accel, h.p + N * 0.001, L, 1000.0, inst_tab, slot_mesh, meshes, verts, indices, inst);
                    if (!sh.ok) radiance += throughput * brdf(N, V, L, base, f0, metallic, a2) * NdotL * float3(PI * cs.light.w);
                }
            }
            if (bounce == 0 && sp.hl_specular.w > 0.5)   // MuJoCo headlight on the primary hit (view-aligned, unshadowed)
                radiance += throughput * brdf(N, V, V, base, f0, metallic, a2) * max(dot(N, V), 0.0) * sp.hl_diffuse.xyz * PI;
            // sample the next direction: diffuse (cosine) or specular (GGX) lobe
            float pd = (1.0 - metallic) * 0.5 + 0.5 * (1.0 - specw);
            pd = clamp(pd, 0.1, 0.95);
            float3 Lnew; float pdf;
            if (rnd(rng) < pd) {
                Lnew = cosine_sample(N, rnd(rng), rnd(rng));
                pdf = max(dot(N, Lnew), 0.0) / PI;
            } else {
                float u = rnd(rng), v = rnd(rng);
                float a = rough * rough;
                float ct = sqrt((1.0 - u) / (1.0 + (a * a - 1.0) * u));
                float st = sqrt(max(1.0 - ct * ct, 0.0)); float ph = 2.0 * PI * v;
                float3 t, b; basis(N, t, b);
                float3 H = normalize(t * (st * cos(ph)) + b * (st * sin(ph)) + N * ct);
                Lnew = reflect(-V, H);
                float NdotH = max(dot(N, H), 0.0), VdotH = max(dot(V, H), 1e-4);
                pdf = D_ggx(NdotH, a2) * NdotH / (4.0 * VdotH);
            }
            float NdotL = dot(N, Lnew);
            if (NdotL <= 0.0 || pdf <= 1e-6) break;
            // mixture pdf of both lobes for unbiased weighting
            float3 Hm = normalize(Lnew + V);
            float pdf_d = NdotL / PI;
            float pdf_s = D_ggx(max(dot(N, Hm), 0.0), a2) * max(dot(N, Hm), 0.0) / (4.0 * max(dot(V, Hm), 1e-4));
            float pdf_mix = pd * pdf_d + (1.0 - pd) * pdf_s;
            throughput *= brdf(N, V, Lnew, base, f0, metallic, a2) * NdotL / max(pdf_mix, 1e-6);
            if (bounce >= 3) {   // Russian roulette
                float q = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
                if (rnd(rng) > q) break;
                throughput /= q;
            }
            o = h.p + N * 0.001; d = Lnew;
        }
        sum += radiance;
    }
    // progressive accumulation
    uint a = pix * 4;
    if (c.flags & 1u) { accum[a] = 0; accum[a + 1] = 0; accum[a + 2] = 0; accum[a + 3] = 0; }
    accum[a] += sum.x; accum[a + 1] += sum.y; accum[a + 2] += sum.z; accum[a + 3] += float(c.spp);
    float inv = 1.0 / max(accum[a + 3], 1.0);
    float3 col = float3(accum[a], accum[a + 1], accum[a + 2]) * inv * c.exposure;
    out_rgb[pix * 3 + 0] = uchar(clamp(col.x, 0.0f, 1.0f) * 255.0f + 0.5f);
    out_rgb[pix * 3 + 1] = uchar(clamp(col.y, 0.0f, 1.0f) * 255.0f + 0.5f);
    out_rgb[pix * 3 + 2] = uchar(clamp(col.z, 0.0f, 1.0f) * 255.0f + 0.5f);
    if (c.flags & 1u) {
        out_depth[pix] = depth0; out_seg[pix] = seg0;
        out_normal[pix * 3] = normal0.x; out_normal[pix * 3 + 1] = normal0.y; out_normal[pix * 3 + 2] = normal0.z;
    }
}
