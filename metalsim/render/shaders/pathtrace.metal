// metalsim tier 2: unidirectional path tracer on Metal hardware ray tracing.
//
// Per (pixel, env): `spp` paths per call accumulated into an HDR buffer (progressive), up to
// `max_bounces` bounces, next-event estimation for directional lights (shadow rays), BRDF sampling
// (Lambert diffuse + GGX specular with Fresnel, same material model and light radiances as tiers
// 0/1: MuJoCo diffuse x pi), Russian roulette after 3 bounces, environment radiance = model sky
// colour (uniform) or, with FLAG_ENVMAP, an equirectangular HDR environment map (a textured USD
// DomeLight: pole +z as Kit/RTX orients it, rotated about z by env_yaw) that is importance-sampled
// for next-event estimation and combined with BRDF sampling by the power heuristic (Veach 1997;
// PBRT 3rd ed. 14.2.4), emissive materials as area sources through BRDF sampling. Primary-hit depth,
// normal and segmentation are written for annotators. Envs share one instance acceleration
// structure and are spatially separated by RTOffset (see rt.metal).
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

struct Material { float4 rgba; float4 mrse; float4 rect; float4 par; };
struct Light { float4 kind; float4 diffuse; float4 specular; float4 ambient; float4 atten; };
struct SceneParams { float4 bounds; float4 hl_ambient; float4 hl_diffuse; float4 hl_specular; float4 sky;
                     float4 sky_rect; float4 sky_info; };   // skybox cube in the atlas; sky_info: faces, drawn
#include "appearance.h"
#include "dyn_bvh.h"
#include "env_grid.h"
struct MeshInfo { uint i_off, i_count, _p0, _p1; };
struct InstanceDesc { packed_float3 col0, col1, col2, col3; uint options, mask, if_table_offset, as_index; };
struct PTConsts {
    uint n_envs, width, height, n_cam;
    uint n_light, n_slots, spp, max_bounces;
    uint frame, tiles_per_row, flags, seed;     // flags: see FLAG_* below
    float stride, exposure, env_scale, clip_far;   // clip_far: camera far clipping plane (z-depth), 0 = none
    uint env_w, env_h, grain_inst0, grain_as0;  // FLAG_ENVMAP: importance-sampling grid of the environment map; FLAG_GRAINS: first grain
                                                // instance and first grain mesh in the instanced-structure list (particles.py)
    float env_yaw, clamp_lum, grain_eps; uint trans_ids;   // env_yaw: dome rotation about +z (rad); clamp_lum: per-path luminance cap, 0 = off (biased);
                                                // grain_eps: ray offset off a grain surface [m] (grains are mm-sized; scene hits use 1 mm);
                                                // trans_ids: annotations under transparent geoms (0 surface, 1 behind)
    float4 env_mul;                             // FLAG_ENVMAP: rgb multiplier on the map (dome intensity * 2^exposure * color)
    EnvGrid grid;                               // where the envs sit (env_grid.h); rays end at their env's cell
#ifdef DYN_SBVH
    ulong dyn_args;                             // GPU address of the DynBVHArgs (sbvh layout; all 31 buffer slots are taken)
    ulong _pad_dyn;
#endif
};
struct CamSpec { float4 intrinsics; float4 pos_delta; float4 rot_delta; float4 light; float4 ambient; float4 clear_color; int cam_id, p0, p1, p2; };

constant float PI = 3.14159265358979f;
// PTConsts.flags
constant uint FLAG_RESET = 1u;        // restart accumulation (also resets the auxiliary buffers)
constant uint FLAG_MATMODEL = 2u;     // per-material BRDF model from Material.par.w (0 legacy, 1 UsdPreviewSurface/Storm, 2 OmniPBR curve,
                                      // 3 UsdPreviewSurface/MDL layering, 4 OmniPBR/MDL layering, 5/6 = 3/4 with the base
                                      // weighted by the Fresnel curve at N.V)
constant uint FLAG_TONEMAP_RTX = 4u;  // RTX default tone mapping (exposure, ACES, sRGB) instead of a linear clamp
constant uint FLAG_CENTER0 = 16u;     // first sample of the reset pass through the pixel centre (annotators match raster / Isaac)
constant uint FLAG_AUX = 8u;          // write mean HDR radiance and sample-averaged first-hit albedo / normal (denoiser inputs)
constant uint FLAG_ENVMAP = 32u;      // equirectangular HDR environment map (texture(1), CDF table buffer(26)) replaces the sky colour
constant uint FLAG_GRAINS = 64u;      // particle render grains: instances from grain_inst0 on (metalsim/render/particles.py, PARTICLES.md)

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

// UsdPreviewSurface as Hydra Storm evaluates it (previewSurface.glslfx): GGX D with alpha = roughness^2,
// Schlick-GGX geometry with k = alpha / 2, F = mix(F0, F90, (1 - E.H)^5), Lambert diffuse weighted by (1 - metallic);
// metallic workflow (spec): F0 = mix(((1 - ior) / (1 + ior))^2, base, metallic), F90 = mix(1, base, metallic).
// Here F0_dielectric = 0.08 * specular (0.04 for ior 1.5, written by the host).
inline float3 brdf_preview(float3 N, float3 V, float3 L, float3 base, float specw, float metallic, float rough) {
    float NdotL = max(dot(N, L), 0.0), NdotV = max(dot(N, V), 1e-4);
    if (NdotL <= 0.0) return float3(0.0);
    float3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0), VdotH = max(dot(V, H), 0.0);
    float alpha = rough * rough, a2 = alpha * alpha;
    float dd = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = (a2 + 1e-3) / (PI * dd * dd);
    float k = alpha * 0.5;
    float G = (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));
    float3 F0 = mix(float3(0.08 * specw), base, metallic), F90 = mix(float3(1.0), base, metallic);
    float fr = pow(max(1.0 - VdotH, 0.0), 5.0);
    float3 F = mix(F0, F90, fr);
    float3 spec = F * G * D / (4.0 * NdotL * NdotV + 1e-3);
    float3 diff = base * (1.0 - metallic) * (1.0 - F) / PI;
    return diff + spec;
}

// OmniPBR (OmniPBR.mdl): df::custom_curve_layer(normal_reflectivity 0.08, grazing_reflectivity 1, exponent 5,
// weight specular_level, layer GGX-Smith with alpha = roughness^2, base Lambert(diffuse)); metals:
// df::weighted_layer(metallic, GGX tinted by the base colour). The layer's curve is evaluated at V.H for the
// specular lobe; the base is attenuated by the layer's directional reflectance at the viewing direction.
inline float3 brdf_omnipbr(float3 N, float3 V, float3 L, float3 base, float specw, float metallic, float rough) {
    float NdotL = max(dot(N, L), 0.0), NdotV = max(dot(N, V), 1e-4);
    if (NdotL <= 0.0) return float3(0.0);
    float3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0), VdotH = max(dot(V, H), 0.0);
    float alpha = rough * rough, a2 = alpha * alpha;
    float D = D_ggx(NdotH, a2), G = G_smith(NdotV, NdotL, a2);
    float ggx = D * G / (4.0 * NdotL * NdotV + 1e-4);
    float cH = 0.08 + 0.92 * pow(max(1.0 - VdotH, 0.0), 5.0);
    float cV = 0.08 + 0.92 * pow(max(1.0 - NdotV, 0.0), 5.0);
    float cL = 0.08 + 0.92 * pow(max(1.0 - NdotL, 0.0), 5.0);
    float3 dielectric = specw * cH * ggx + (1.0 - specw * max(cV, cL)) * base / PI;
    float3 metal = base * ggx;
    return mix(dielectric, metal, metallic);
}

// Directional albedo of a GGX lobe with Schlick Fresnel (F0, F90 = 1), Karis' analytic fit of the split-sum
// table ("Physically Based Shading on Mobile", 2014): E = F0 * A + B.
inline float ggx_albedo(float F0, float rough, float NdotV) {
    float4 r = rough * float4(-1.0, -0.0275, -0.572, 0.022) + float4(1.0, 0.0425, 1.04, -0.04);
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    float2 AB = float2(-1.04, 1.04) * a004 + r.zw;
    return clamp(F0 * AB.x + AB.y, 0.0, 1.0);
}

// MDL-style layering (how RTX evaluates OmniPBR's custom_curve_layer and a UsdPreviewSurface's Fresnel layer):
// a GGX layer with a Schlick-shaped curve (F0 -> 1, exponent 5) scaled by `w` over a Lambert base; the base is
// attenuated by the layer's directional reflectance at the viewing direction, w * E_ggx(F0, roughness, N.V), so a
// rough dielectric darkens toward grazing view and reflects the environment instead. Metals: GGX tinted by base.
inline float3 brdf_layered(float3 N, float3 V, float3 L, float3 base, float F0, float w, float metallic, float rough, bool curve_v = false) {
    float NdotL = max(dot(N, L), 0.0), NdotV = max(dot(N, V), 1e-4);
    if (NdotL <= 0.0) return float3(0.0);
    float3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0), VdotH = max(dot(V, H), 0.0);
    float alpha = rough * rough, a2 = alpha * alpha;
    float ggx = D_ggx(NdotH, a2) * G_smith(NdotV, NdotL, a2) / (4.0 * NdotL * NdotV + 1e-4);
    float cH = F0 + (1.0 - F0) * pow(max(1.0 - VdotH, 0.0), 5.0);
    // base weight: the layer's directional albedo (rough-averaged), or (curve_v) the Fresnel curve itself at N.V,
    // which removes the base entirely at grazing view whatever the roughness
    float cV = F0 + (1.0 - F0) * pow(max(1.0 - NdotV, 0.0), 5.0);
    float3 dielectric = w * cH * ggx + (1.0 - w * (curve_v ? cV : ggx_albedo(F0, rough, NdotV))) * base / PI;
    return mix(dielectric, base * ggx, metallic);
}

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

inline float3 brdf_any(uint model, float3 N, float3 V, float3 L, float3 base, float3 f0, float specw, float metallic, float rough, float a2) {
    if (model == 1u) return brdf_preview(N, V, L, base, specw, metallic, rough);
    if (model == 2u) return brdf_omnipbr(N, V, L, base, specw, metallic, rough);
    if (model == 3u) return brdf_layered(N, V, L, base, 0.08 * specw, 1.0, metallic, rough);   // UsdPreviewSurface, MDL layering (F0 from ior)
    if (model == 4u) return brdf_layered(N, V, L, base, 0.08, specw, metallic, rough);         // OmniPBR custom_curve_layer, MDL layering
    if (model == 5u) return brdf_layered(N, V, L, base, 0.08 * specw, 1.0, metallic, rough, true);   // as 3, base weight = Fresnel at N.V
    if (model == 6u) return brdf_layered(N, V, L, base, 0.08, specw, metallic, rough, true);         // as 4, base weight = curve at N.V
    return brdf(N, V, L, base, f0, metallic, a2);
}

// RTX's default display transform: exposure, ACES filmic curve (Narkowicz fit of the RRT+ODT), sRGB OETF
inline float3 aces_fit(float3 x) { return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
inline float3 srgb_oetf(float3 c) {
    return select(1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
}
inline float3 display(float3 hdr, float exposure, bool rtx) {
    if (!rtx) return clamp(hdr * exposure, 0.0, 1.0);
    return clamp(srgb_oetf(aces_fit(hdr * exposure)), 0.0, 1.0);
}
// uniform direction in the cone of half-angle `ang` about `axis` (a sun disk of angular radius `ang`)
inline float3 sample_cone(float3 axis, float ang, float u, float v) {
    float ct = 1.0 - u * (1.0 - cos(ang)), st = sqrt(max(1.0 - ct * ct, 0.0)), ph = 2.0 * PI * v;
    float3 t, b; basis(axis, t, b);
    return normalize(t * (st * cos(ph)) + b * (st * sin(ph)) + axis * ct);
}

struct Hit { bool ok; float t; float3 p; float3 n; float2 uv; uint slot; bool grain; uint gid; float3 lp; };   // lp: geom-frame position
// render-grain geometry (FLAG_GRAINS): instances >= inst0 are grains, instancing grain mesh (as_index - as0)
struct GrainGeo { device const float* verts; device const uint* indices; device const MeshInfo* meshes; uint inst0, as0; };

inline Hit trace(instance_acceleration_structure accel, float3 o, float3 d, float tmax,
                 device const uint2* inst_tab, device const int* slot_mesh, device const MeshInfo* meshes,
                 device const float* verts, device const uint* indices, device const InstanceDesc* inst,
                 GrainGeo gg, float tmin DYN_PARAM)
{
    Hit h; h.ok = false;
    ray r(o, d, tmin, tmax);
#ifdef DYN_SBVH
    RTRes res = RT_CLOSEST(r, accel);
#else
    intersector<triangle_data, instancing> isect;
    intersection_result<triangle_data, instancing> res = isect.intersect(r, accel);
#endif
    if (res.type == intersection_type::none) return h;
    bool grain = res.instance_id >= gg.inst0;
    uint slot = 0, b;
    if (grain) {
        b = gg.meshes[inst[res.instance_id].as_index - gg.as0].i_off + res.primitive_id * 3;
        verts = gg.verts; indices = gg.indices;
    } else {
        slot = inst_tab[res.instance_id].y + res.geometry_id;   // geometry: 0 for geoms; the object within a dynamic env structure
        b = meshes[RT_MESH(res, inst[res.instance_id].as_index)].i_off + res.primitive_id * 3;   // mesh: = slot_mesh[slot] for geoms
    }
    uint i0 = indices[b], i1 = indices[b + 1], i2 = indices[b + 2];
    float2 bc = res.triangle_barycentric_coord; float w0 = 1.0 - bc.x - bc.y;
    float3 n0 = float3(verts[i0*8+3], verts[i0*8+4], verts[i0*8+5]);
    float3 n1 = float3(verts[i1*8+3], verts[i1*8+4], verts[i1*8+5]);
    float3 n2 = float3(verts[i2*8+3], verts[i2*8+4], verts[i2*8+5]);
    float2 uv = float2(verts[i0*8+6], verts[i0*8+7]) * w0 + float2(verts[i1*8+6], verts[i1*8+7]) * bc.x + float2(verts[i2*8+6], verts[i2*8+7]) * bc.y;
    h.lp = float3(verts[i0*8], verts[i0*8+1], verts[i0*8+2]) * w0 + float3(verts[i1*8], verts[i1*8+1], verts[i1*8+2]) * bc.x
         + float3(verts[i2*8], verts[i2*8+1], verts[i2*8+2]) * bc.y;
    InstanceDesc id = inst[res.instance_id];
    float3x3 R = float3x3(float3(id.col0), float3(id.col1), float3(id.col2));
    float3 n = normalize(R * (n0 * w0 + n1 * bc.x + n2 * bc.y));
    if (dot(n, d) > 0.0) n = -n;
    h.ok = true; h.t = res.distance; h.p = o + d * res.distance; h.n = n; h.uv = uv; h.slot = slot;
    h.grain = grain; h.gid = grain ? res.instance_id - gg.inst0 : 0u;
    return h;
}

// render-grain material (particles.GrainMaterial.pack): a palette entry picked by the grain's hash with the
// cumulative weights, a brightness jitter, shared roughness / specular / metallic and BRDF model
struct GrainMat { float4 pal[8]; float4 p0; float4 mrse; float4 p1; MeshInfo mesh[8]; };   // particles.GrainField.mat_buf
inline Material grain_material(device const GrainMat& g, uint key) {
    uint hh = hash_u(key * 0x9E3779B1u + 0x85EBCA6Bu);
    float u = float(hh & 0xFFFFu) / 65536.0, v = float(hh >> 16) / 65536.0;
    uint n = clamp(uint(g.p0.x), 1u, 8u), k = 0;
    while (k + 1 < n && u > g.pal[k].w) ++k;
    Material m;
    m.rgba = float4(g.pal[k].rgb * max(1.0 + g.p0.y * (2.0 * v - 1.0), 0.0), 1.0);
    m.mrse = g.mrse; m.rect = float4(0.0); m.par = float4(1.0, 1.0, 0.0, g.p1.x);
    return m;
}

// Partial transparency (material-resolved rgba alpha < 1) is surface presence, as a USD opacity in Isaac's path
// tracer (and RTX Real-Time 2.0's fractional cutout opacity): a ray meets the surface with probability alpha and
// otherwise continues unchanged; shadow rays carry the product of (1 - alpha) over the partially transparent surfaces
// they cross (an opaque one stops them). Up to TRANSMIT_MAX surfaces per ray.
constant uint TRANSMIT_MAX = 8u;
inline float surface_alpha(device const Material* mats, device const float4* per_inst, uint e, uint n_slots, uint slot) {
    return mats[slot].rgba.a * per_inst[e * n_slots + slot].a;
}
inline float transmit(instance_acceleration_structure accel, float3 o, float3 d, float tmax,
                      device const uint2* inst_tab, device const int* slot_mesh, device const MeshInfo* meshes,
                      device const float* verts, device const uint* indices, device const InstanceDesc* inst,
                      GrainGeo gg, float tmin, device const Material* mats, device const float4* per_inst, uint e, uint n_slots DYN_PARAM)
{
    float T = 1.0;
    for (uint k = 0; k < TRANSMIT_MAX; ++k) {
        Hit h = trace(accel, o, d, tmax, inst_tab, slot_mesh, meshes, verts, indices, inst, gg, k == 0 ? tmin : 0.0005 DYN_PASS);
        if (!h.ok) return T;
        if (h.grain) return 0.0;                 // grains are opaque
        float a = surface_alpha(mats, per_inst, e, n_slots, h.slot);
        if (a >= 0.995) return 0.0;
        T *= 1.0 - a;
        o = h.p; tmax -= h.t;
        if (tmax <= 0.0005) return T;
    }
    return T;
}

inline float3 material_base(Material m, float4 mod, float2 uv, texture2d<float> atlas, sampler samp) {
    float3 base = m.rgba.rgb * mod.rgb;
    if (m.par.z > 0.5) base *= atlas.sample(samp, m.rect.xy + fract(uv * m.par.xy) * m.rect.zw, level(0.0)).rgb;
    return base;
}
// with the material extension (MuJoCo's texture mapping: offsets, cube maps, sRGB; shaders/appearance.h)
inline float4 material_base_ext(Material m, MaterialExt x, float4 mod, float2 uv, float3 lp, texture2d<float> atlas, sampler samp) {
    float4 t = material_tex(m.rect, m.par, x, uv, lp, atlas, samp);
    return float4(m.rgba.rgb * mod.rgb * t.rgb, m.rgba.a * mod.a * t.a);
}

// ---- environment map: equirectangular ("latlong") as Kit/RTX orients a USD DomeLight on a z-up stage.
// USD/OpenEXR define the map with the pole at +y, u = longitude (u = 0.5 faces +z, u = 0.25 faces +x),
// v = latitude from the top; Kit/RTX treats the dome as z-up (NVIDIA usd-exchange-samples createLights
// README), i.e. that map rotated +90 deg about x: pole +z, u = 0.5 faces -y, u = 0.25 faces +x (the azimuth
// reference is the rotation's consequence, not documented by NVIDIA: docs/PARITY.md 1.7). env_yaw rotates
// the dome about +z (a rotateZ on the DomeLight prim). lon = atan2(x, -y) in [-pi, pi], u = 0.5 - lon / 2pi.
inline float2 env_uv(float3 d, float yaw) {
    float u = 0.5 - (atan2(d.x, -d.y) - yaw) / (2.0 * PI);
    return float2(u - floor(u), acos(clamp(d.z, -1.0f, 1.0f)) / PI);
}
inline float3 env_dir(float2 uv, float yaw) {
    float lon = (0.5 - uv.x) * 2.0 * PI + yaw, th = uv.y * PI;   // th: polar angle from +z
    float st = sin(th);
    return float3(st * sin(lon), -st * cos(lon), cos(th));
}
inline float3 env_radiance(texture2d<float> env, sampler s, float3 d, constant PTConsts& c) {
    return env.sample(s, env_uv(d, c.env_yaw), level(0.0)).rgb * c.env_mul.rgb;
}
// solid-angle pdf of the importance sampler: p(omega) = p(u, v) / (2 pi^2 sin theta) (PBRT 3rd ed. 14.2.4);
// `cdf` = [marginal CDF over rows (H) | per-row conditional CDFs (H*W) | density over (u, v) (H*W)]
inline float env_pdf(float3 d, device const float* cdf, constant PTConsts& c) {
    float2 uv = env_uv(d, c.env_yaw);
    uint x = min(uint(uv.x * c.env_w), c.env_w - 1), y = min(uint(uv.y * c.env_h), c.env_h - 1);
    float st = sin(uv.y * PI);
    if (st <= 1e-6) return 0.0;
    return cdf[c.env_h + c.env_h * c.env_w + y * c.env_w + x] / (2.0 * PI * PI * st);
}
inline uint upper_bound(device const float* a, uint n, float v) {   // first index with a[i] > v
    uint lo = 0, hi = n;
    while (lo < hi) { uint mid = (lo + hi) >> 1; if (a[mid] <= v) lo = mid + 1; else hi = mid; }
    return min(lo, n - 1);
}
// sample a direction proportionally to the table (row by the marginal CDF, column by that row's CDF,
// uniform inside the cell), returning its solid-angle pdf
inline float3 env_sample(device const float* cdf, constant PTConsts& c, float u1, float u2, thread float& pdf) {
    uint y = upper_bound(cdf, c.env_h, u1);
    device const float* row = cdf + c.env_h + y * c.env_w;
    uint x = upper_bound(row, c.env_w, u2);
    float r0 = x > 0 ? row[x - 1] : 0.0, w = max(row[x] - r0, 1e-9);
    float m0 = y > 0 ? cdf[y - 1] : 0.0, mh = max(cdf[y] - m0, 1e-9);
    float2 uv = float2((float(x) + clamp((u2 - r0) / w, 0.0, 1.0)) / float(c.env_w), (float(y) + clamp((u1 - m0) / mh, 0.0, 1.0)) / float(c.env_h));
    float3 d = env_dir(uv, c.env_yaw);
    pdf = env_pdf(d, cdf, c);
    return d;
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
    device float*           out_hdr    [[buffer(22)]],   // (n_envs, h, w, 4) mean radiance (FLAG_AUX)
    device float*           aux_accum  [[buffer(23)]],   // (n_envs, h, w, 8) albedo sum, normal sum, count (FLAG_AUX)
    device float*           out_albedo [[buffer(24)]],   // (n_envs, h, w, 3) mean first-hit albedo (FLAG_AUX)
    device float*           out_nrm    [[buffer(25)]],   // (n_envs, h, w, 3) mean first-hit normal (FLAG_AUX)
    device const float*     env_cdf    [[buffer(26)]],   // environment-map sampling table (FLAG_ENVMAP)
    device const float*     g_verts    [[buffer(27)]],   // FLAG_GRAINS: grain mesh vertices (8 floats each, as verts)
    device const uint*      g_indices  [[buffer(28)]],   // FLAG_GRAINS: grain mesh indices
    device const uint*      g_keys     [[buffer(29)]],   // FLAG_GRAINS: per-grain identity key (colour hash)
    device const GrainMat&  g_mat      [[buffer(30)]],   // FLAG_GRAINS: grain material + grain mesh index ranges
    texture2d<float>        atlas      [[texture(0)]],
    texture2d<float>        envmap     [[texture(1)]],   // equirectangular RGBA32F environment (FLAG_ENVMAP)
    sampler                 samp       [[sampler(0)]],
    sampler                 env_samp   [[sampler(1)]],   // linear, repeat in u, clamp in v
    uint3 tid [[thread_position_in_grid]])
{
#ifdef DYN_SBVH
    device const DynBVHArgs& dbv = *reinterpret_cast<device const DynBVHArgs*>(c.dyn_args);
#endif
    uint x = tid.x, y = tid.y, e = tid.z;
    if (x >= c.width || y >= c.height || e >= c.n_envs) return;
    uint pix = (e * c.height + y) * c.width + x;
    // per-slot texture mapping (scene_tables.MATERIAL_EXT_LAYOUT): appended to the material table (G x 16 floats, then G x 8)
    device const MaterialExt* mext = (device const MaterialExt*)(mats + c.n_slots);
    CamSpec cs = cams[e];
    uint ci = e * c.n_cam + uint(max(cs.cam_id, 0));
    float3 cpos = float3(cam_xpos[ci * 3], cam_xpos[ci * 3 + 1], cam_xpos[ci * 3 + 2]) + cs.pos_delta.xyz;
    float3x3 R = load_mat33(cam_xmat + ci * 9) * quat_to_mat(cs.rot_delta);
    float3 env_off = grid_offset(e, c.grid);
#define PT_CLIP(o_, d_, t_) grid_tmax((o_), (d_), (t_), e, c.grid)
    Rng rng; rng.s = hash_u(pix * 7919u + c.frame * 104729u + c.seed);
    float3 sum = float3(0.0);
    float depth0 = 0.0; float3 normal0 = float3(0.0); int seg0 = 0;
    float3 alb_sum = float3(0.0), nrm_sum = float3(0.0);
    bool matmodel = (c.flags & FLAG_MATMODEL) != 0u;
    bool use_env = (c.flags & FLAG_ENVMAP) != 0u;
    float3 env_rad = (sp.sky.w > 0.5 ? sp.sky.xyz : cs.clear_color.rgb) * c.env_scale;
    GrainGeo gg; gg.verts = g_verts; gg.indices = g_indices; gg.meshes = g_mat.mesh;
    gg.inst0 = (c.flags & FLAG_GRAINS) ? c.grain_inst0 : 0xFFFFFFFFu; gg.as0 = c.grain_as0;
    bool sky_tex = !use_env && sp.sky_info.y > 0.5;     // MuJoCo skybox as a directional environment (cube map)
    for (uint s = 0; s < c.spp; ++s) {
        float jx = rnd(rng), jy = rnd(rng);
        if (s == 0u && (c.flags & FLAG_CENTER0) && (c.flags & FLAG_RESET)) { jx = 0.5; jy = 0.5; }   // annotator sample at the pixel centre (1 of spp*passes)
        float3 dl = float3((float(x) + jx - cs.intrinsics.z) / cs.intrinsics.x, -(float(y) + jy - cs.intrinsics.w) / cs.intrinsics.y, -1.0);
        float3 d = normalize(R * dl);
        float3 o = cpos + env_off;
        float3 throughput = float3(1.0);
        float3 radiance = float3(0.0);
        float prev_pdf = 0.0;   // BSDF-sampling pdf of the ray being traced (0: camera ray, no MIS on a miss)
        float tmin = 0.0005;     // ray start distance: 0.5 mm, or grain_eps leaving a grain
        uint passes = 0;        // transparent surfaces passed through (presence), not counted as bounces
        for (uint bounce = 0; bounce <= c.max_bounces; ++bounce) {
            Hit h = trace(accel, o, d, PT_CLIP(o, d, 1000.0), inst_tab, slot_mesh, meshes, verts, indices, inst, gg, tmin DYN_PASS);
            if (bounce == 0 && c.clip_far > 0.0 && h.ok && -dot(h.p - o, R[2]) > c.clip_far) h.ok = false;   // beyond the far plane: background
            if (!h.ok) {
                float3 er = sky_tex ? sky_color(atlas, samp, sp.sky_rect, sp.sky_info, d) * c.env_scale : env_rad;
                if (use_env) {   // map radiance, weighted by the power heuristic against the env NEE that could have sampled it
                    er = env_radiance(envmap, env_samp, d, c);
                    if (prev_pdf > 0.0) { float pe = env_pdf(d, env_cdf, c); er *= prev_pdf * prev_pdf / (prev_pdf * prev_pdf + pe * pe); }
                }
                radiance += throughput * er;
                if (bounce == 0) {   // denoiser guide: the background's normalised colour
                    float3 eg = use_env ? env_radiance(envmap, env_samp, d, c) : er;
                    alb_sum += clamp(eg / max(max(eg.x, max(eg.y, eg.z)), 1e-6), 0.0, 1.0); nrm_sum += -d;
                }
                break;
            }
            Material m; float4 mod; float eps = 0.001;   // eps: offset of rays leaving this surface
            float4 base4;
            if (h.grain) {   // mm-sized grains: a small offset, but at least ~8 ulp of the position (far envs sit km away)
                m = grain_material(g_mat, g_keys[h.gid]); mod = float4(1.0);
                eps = max(c.grain_eps, 1e-6 * max(fabs(h.p.x), max(fabs(h.p.y), fabs(h.p.z)))); tmin = eps;
                base4 = float4(material_base(m, mod, h.uv, atlas, samp), 1.0);
            }
            else {
                m = mats[h.slot]; mod = per_inst[e * c.n_slots + h.slot]; tmin = 0.0005;
                base4 = material_base_ext(m, mext[h.slot], mod, h.uv, h.lp, atlas, samp);
            }
            float3 base = base4.rgb;
            if (bounce == 0 && s == 0 && passes == 0) {   // annotations of the primary ray (see PTConsts.trans_ids)
                depth0 = -dot(h.p - o, R[2]); normal0 = h.n; seg0 = h.grain ? int(c.n_slots) + 1 : int(h.slot) + 1;
                if (!h.grain && m.rgba.a * mod.a < 0.995 && c.trans_ids == 1u) {    // "behind": the first opaque surface
                    float3 ob = h.p; float zb = 0.0; float3 nb = float3(0.0); int sb = 0;
                    for (uint k = 0; k < TRANSMIT_MAX; ++k) {
                        Hit hb = trace(accel, ob, d, PT_CLIP(ob, d, 1000.0), inst_tab, slot_mesh, meshes, verts, indices, inst, gg, 0.0005 DYN_PASS);
                        if (!hb.ok) break;
                        if (hb.grain || surface_alpha(mats, per_inst, e, c.n_slots, hb.slot) >= 0.995) {
                            zb = -dot(hb.p - o, R[2]); nb = hb.n; sb = hb.grain ? int(c.n_slots) + 1 : int(hb.slot) + 1; break;
                        }
                        ob = hb.p;
                    }
                    depth0 = (c.clip_far > 0.0 && zb > c.clip_far) ? 0.0 : zb;
                    normal0 = nb; seg0 = (depth0 > 0.0) ? sb : 0;
                }
            }
            if (base4.a < 0.995 && passes < 16u && rnd(rng) >= base4.a) {   // presence: the ray passes this surface
                ++passes; o = h.p; --bounce;            // (unsigned wrap at bounce 0 is undone by the loop's ++)
                continue;
            }
            float metallic = m.mrse.x, rough = max(m.mrse.y, 0.03), specw = m.mrse.z, emission = m.mrse.w;
            float a2 = rough * rough * rough * rough;
            float3 f0 = mix(float3(0.08 * specw), base, metallic);
            float3 N = h.n; float3 V = -d;
            uint model = matmodel ? uint(m.par.w + 0.5) : 0u;
            if (bounce == 0) { alb_sum += base; nrm_sum += N; }
            radiance += throughput * emission * base;
            // next-event estimation: model lights (directional and point/spot) with shadow rays
            for (uint i = 0; i < c.n_light; ++i) {
                Light l = lights[i];
                device const float* lp = light_xpos + (e * c.n_light + i) * 3;
                device const float* ld = light_xdir + (e * c.n_light + i) * 3;
                float3 L; float att = 1.0; float tmax = 1000.0;
                if (l.kind.x == 0.0) {
                    L = -normalize(float3(ld[0], ld[1], ld[2]));
                    if (l.diffuse.w > 0.0) L = sample_cone(L, l.diffuse.w, rnd(rng), rnd(rng));   // sun disk (angular radius), irradiance preserved
                }
                else {
                    float3 dv = float3(lp[0], lp[1], lp[2]) + env_off - h.p; float dist = length(dv); L = dv / max(dist, 1e-6); tmax = dist - 0.001;
                    att = 1.0 / max(l.atten.x + l.atten.y * dist + l.atten.z * dist * dist, 1e-4);
                    if (l.kind.x == 2.0) { float ca = dot(-L, normalize(float3(ld[0], ld[1], ld[2]))); att *= ca > cos(l.kind.z * PI / 180.0) ? pow(max(ca, 0.0), l.kind.w) : 0.0; }
                }
                float NdotL = dot(N, L);
                if (NdotL <= 0.0 || att <= 0.0) continue;
                float tr = transmit(accel, h.p + N * eps, L, PT_CLIP(h.p + N * eps, L, tmax), inst_tab, slot_mesh, meshes, verts, indices, inst, gg, tmin, mats, per_inst, e, c.n_slots DYN_PASS);
                if (tr <= 0.0) continue;
                radiance += throughput * brdf_any(model, N, V, L, base, f0, specw, metallic, rough, a2) * NdotL * l.diffuse.xyz * (PI * att * tr);
            }
            if (cs.light.w > 0.0) {   // per-env DR directional light
                float3 L = -normalize(cs.light.xyz);
                float NdotL = dot(N, L);
                if (NdotL > 0.0) {
                    float tr = transmit(accel, h.p + N * eps, L, PT_CLIP(h.p + N * eps, L, 1000.0), inst_tab, slot_mesh, meshes, verts, indices, inst, gg, tmin, mats, per_inst, e, c.n_slots DYN_PASS);
                    if (tr > 0.0) radiance += throughput * brdf_any(model, N, V, L, base, f0, specw, metallic, rough, a2) * NdotL * float3(PI * cs.light.w * tr);
                }
            }
            if (bounce == 0 && sp.hl_specular.w > 0.5)   // MuJoCo headlight on the primary hit (view-aligned, unshadowed)
                radiance += throughput * brdf_any(model, N, V, V, base, f0, specw, metallic, rough, a2) * max(dot(N, V), 0.0) * sp.hl_diffuse.xyz * PI;
            // lobe-mixture weight of the BSDF sampler below (also the pdf the environment MIS weights against)
            float pd = (1.0 - metallic) * 0.5 + 0.5 * (1.0 - specw);
            pd = clamp(pd, 0.1, 0.95);
            if (use_env) {   // next-event estimation on the environment map, MIS (power heuristic) against BSDF sampling
                float pe; float3 L = env_sample(env_cdf, c, rnd(rng), rnd(rng), pe);
                float NdotL = dot(N, L);
                if (NdotL > 0.0 && pe > 1e-8) {
                    float tr = transmit(accel, h.p + N * eps, L, PT_CLIP(h.p + N * eps, L, 1000.0), inst_tab, slot_mesh, meshes, verts, indices, inst, gg, tmin, mats, per_inst, e, c.n_slots DYN_PASS);
                    if (tr > 0.0) {
                        float3 Hm = normalize(L + V);
                        float pb = pd * NdotL / PI + (1.0 - pd) * D_ggx(max(dot(N, Hm), 0.0), a2) * max(dot(N, Hm), 0.0) / (4.0 * max(dot(V, Hm), 1e-4));
                        float w = pe * pe / (pe * pe + pb * pb);
                        radiance += throughput * brdf_any(model, N, V, L, base, f0, specw, metallic, rough, a2) * NdotL * env_radiance(envmap, env_samp, L, c) * (w / pe * tr);
                    }
                }
            }
            // sample the next direction: diffuse (cosine) or specular (GGX) lobe
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
            throughput *= brdf_any(model, N, V, Lnew, base, f0, specw, metallic, rough, a2) * NdotL / max(pdf_mix, 1e-6);
            prev_pdf = pdf_mix;
            if (bounce >= 3) {   // Russian roulette
                float q = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
                if (rnd(rng) > q) break;
                throughput /= q;
            }
            o = h.p + N * eps; d = Lnew;
        }
        if (c.clamp_lum > 0.0) {   // opt-in firefly clamp (biased: caps the energy of rare high-contribution paths)
            float l = dot(radiance, float3(0.2126, 0.7152, 0.0722));
            if (l > c.clamp_lum) radiance *= c.clamp_lum / l;
        }
        // never let a non-finite path estimate reach the accumulator (fast-math safe: test the exponent bits)
        uint3 bits = as_type<uint3>(radiance);
        if (any((bits & 0x7f800000u) == 0x7f800000u)) radiance = float3(0.0);
        sum += radiance;
    }
    // progressive accumulation
    uint a = pix * 4;
    if (c.flags & 1u) { accum[a] = 0; accum[a + 1] = 0; accum[a + 2] = 0; accum[a + 3] = 0; }
    accum[a] += sum.x; accum[a + 1] += sum.y; accum[a + 2] += sum.z; accum[a + 3] += float(c.spp);
    float inv = 1.0 / max(accum[a + 3], 1.0);
    float3 mean = float3(accum[a], accum[a + 1], accum[a + 2]) * inv;
    float3 col = display(mean, c.exposure, (c.flags & FLAG_TONEMAP_RTX) != 0u);
    out_rgb[pix * 3 + 0] = uchar(clamp(col.x, 0.0f, 1.0f) * 255.0f + 0.5f);
    out_rgb[pix * 3 + 1] = uchar(clamp(col.y, 0.0f, 1.0f) * 255.0f + 0.5f);
    out_rgb[pix * 3 + 2] = uchar(clamp(col.z, 0.0f, 1.0f) * 255.0f + 0.5f);
    if (c.flags & FLAG_AUX) {
        uint b8 = pix * 8;
        if (c.flags & 1u) { for (uint k = 0; k < 8; ++k) aux_accum[b8 + k] = 0.0; }
        aux_accum[b8] += alb_sum.x; aux_accum[b8 + 1] += alb_sum.y; aux_accum[b8 + 2] += alb_sum.z;
        aux_accum[b8 + 3] += nrm_sum.x; aux_accum[b8 + 4] += nrm_sum.y; aux_accum[b8 + 5] += nrm_sum.z; aux_accum[b8 + 6] += float(c.spp);
        float ia = 1.0 / max(aux_accum[b8 + 6], 1.0);
        out_hdr[pix * 4] = mean.x; out_hdr[pix * 4 + 1] = mean.y; out_hdr[pix * 4 + 2] = mean.z; out_hdr[pix * 4 + 3] = 1.0;
        for (uint k = 0; k < 3; ++k) { out_albedo[pix * 3 + k] = aux_accum[b8 + k] * ia; out_nrm[pix * 3 + k] = aux_accum[b8 + 3 + k] * ia; }
    }
    if (c.flags & 1u) {
        out_depth[pix] = depth0; out_seg[pix] = seg0;
        out_normal[pix * 3] = normal0.x; out_normal[pix * 3 + 1] = normal0.y; out_normal[pix * 3 + 2] = normal0.z;
    }
}
