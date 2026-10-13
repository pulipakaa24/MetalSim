// Ray-traced sensors on Metal hardware ray tracing: instance descriptor refit from physics state,
// lidar beams and depth cameras as ray queries against a shared instance acceleration structure.
//
// All envs live in one instance acceleration structure, spatially separated by a per-env offset
// (grid stride larger than scene extent + max range) so a ray from env e only ever meets env e's
// geometry. Instance transforms are read from MuJoCo Warp geom_xpos / geom_xmat each frame.
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;
#include "env_grid.h"
#include "lens.h"

struct RTConsts {
    uint n_envs, n_geoms, n_slots, tiles_per_row;
    float stride;              // env grid spacing (m)
    float max_range;
    uint n_beams;              // beams per lidar
    uint n_site;               // model sites (stride of site_xpos per env)
    uint sensor_site;          // site index the lidar is attached to
    uint _pad0, _pad1, _pad2;
    EnvGrid grid;              // where the envs sit (env_grid.h)
};

struct InstanceDesc {          // MTLAccelerationStructureInstanceDescriptor (64 bytes)
    packed_float3 col0, col1, col2, col3;   // MTLPackedFloat4x3: rotation columns + translation
    uint options;
    uint mask;
    uint if_table_offset;
    uint as_index;
};

struct MeshInfo { uint i_off, i_count, _p0, _p1; };
struct Semantic { int geom, body, root, group; };
struct Material { float4 rgba; float4 mrse; float4 rect; float4 par; };

inline float3x3 load_mat33(device const float* p) {
    return float3x3(float3(p[0], p[3], p[6]), float3(p[1], p[4], p[7]), float3(p[2], p[5], p[8]));
}

#include "dyn_bvh.h"

inline float3 env_offset(uint e, constant RTConsts& c) {
    return grid_offset(e, c.grid);
}

// one thread per (env, slot) instance
kernel void refit_instances(
    device InstanceDesc*    inst      [[buffer(0)]],
    device const uint2*     inst_tab  [[buffer(1)]],
    device const int*       slot_geom [[buffer(2)]],
    device const int*       slot_mesh [[buffer(3)]],
    device const float*     geom_xpos [[buffer(4)]],
    device const float*     geom_xmat [[buffer(5)]],
    constant RTConsts&      c         [[buffer(6)]],
    device atomic_uint*     grid_flag [[buffer(7)]],   // set when a geom nears its env's cell boundary (env_grid.h)
    uint i [[thread_position_in_grid]])
{
    if (i >= c.n_envs * c.n_slots) return;
    uint2 pair = inst_tab[i];
    uint env = pair.x, slot = pair.y;
    uint g = uint(slot_geom[slot]);
    uint gi = env * c.n_geoms + g;
    device const float* xp = geom_xpos + gi * 3;
    float3x3 R = load_mat33(geom_xmat + gi * 9);
    InstanceDesc d;
    d.col0 = R[0]; d.col1 = R[1]; d.col2 = R[2];
    d.col3 = float3(xp[0], xp[1], xp[2]) + env_offset(env, c);
    if (env_grid_left(float3(xp[0], xp[1], xp[2]), c.grid)) atomic_store_explicit(grid_flag, 1u, memory_order_relaxed);
    d.options = 4;   // MTLAccelerationStructureInstanceOptionOpaque
    d.mask = 0xFFFFFFFFu;
    d.if_table_offset = 0;
    d.as_index = uint(slot_mesh[slot]);
    inst[i] = d;
}

struct Hit { float dist; float3 point; float3 normal; int slot; };

// world-space hit attributes for an instance/primitive pair
inline float3 tri_normal(uint mesh, uint prim, device const MeshInfo* meshes, device const float* verts,
                         device const uint* indices, float3x3 R)
{
    uint base = meshes[mesh].i_off + prim * 3;
    float3 a = float3(verts[indices[base] * 8], verts[indices[base] * 8 + 1], verts[indices[base] * 8 + 2]);
    float3 b = float3(verts[indices[base + 1] * 8], verts[indices[base + 1] * 8 + 1], verts[indices[base + 1] * 8 + 2]);
    float3 cc = float3(verts[indices[base + 2] * 8], verts[indices[base + 2] * 8 + 1], verts[indices[base + 2] * 8 + 2]);
    // R may carry a per-world size scale (R diag(s), metalsim.render.sized): the inverse transpose keeps the normal
    float3 s2 = float3(dot(R[0], R[0]), dot(R[1], R[1]), dot(R[2], R[2]));
    return normalize(R * (cross(b - a, cc - a) / max(s2, float3(1e-20))));
}

#ifdef SENSOR_TR
// Transparent surfaces in the sensors (compiled only for sensors that need them: RayTracer._tr_library; the plain
// kernels' source is unchanged without SENSOR_TR, so scenes without transparency run the same binaries).
// Per-slot lidar / depth material, two float4 per slot (raytrace.py LidarMaterial):
//   m0 = (unused, coverage a, specular reflectance at normal incidence R0, GGX/Beckmann roughness)   a < 0: see-through
//   m1 = (transmission of the uncovered part, ghost (follow the mirror reflection), 0, 0)
constant uint SKIP_MAX = 16u;          // see-through surfaces skipped per ray before the next surface counts anyway
constant uint LIDAR_TRANS_MAX = 8u;    // partially transparent / reflecting objects a lidar sub-ray passes

struct SensorHit { bool hit; bool plane; float dist; uint slot; uint inst; uint prim; uint mesh; float3 pn; };

// the first surface of env e along (org, dir) from tmin whose slot is not see-through (lmat[2 slot].y < 0), the env's
// cell first, then the infinite planes past it (env_grid.h); after SKIP_MAX see-through surfaces the next one counts
inline SensorHit first_surface(instance_acceleration_structure accel, float3 org, float3 dir, float tmin, float max_range,
                               uint e, constant RTConsts& c, device const uint2* inst_tab, device const InstanceDesc* inst,
                               device const float4* lmat DYN_PARAM)
{
    SensorHit h; h.hit = false; h.plane = false; h.dist = 0.0; h.slot = 0u; h.inst = 0u; h.prim = 0u; h.mesh = 0u; h.pn = 0.0;
    float tcell = grid_tmax(org, dir, max_range, e, c.grid);
    float t0 = tmin;
    for (uint k = 0; k <= SKIP_MAX; ++k) {
        ray r(org, dir, t0, tcell);
#ifdef DYN_SBVH
        RTRes res = RT_CLOSEST(r, accel);
#else
        intersector<triangle_data, instancing> isect;
        isect.accept_any_intersection(false);
        intersection_result<triangle_data, instancing> res = isect.intersect(r, accel);
#endif
        if (res.type == intersection_type::none) break;
        uint ii = res.instance_id;
        uint slot = inst_tab[ii].y + res.geometry_id;
        if (k < SKIP_MAX && lmat[slot * 2].y < 0.0) { t0 = res.distance + 1e-4; continue; }
        h.hit = true; h.dist = res.distance; h.slot = slot; h.inst = ii; h.prim = res.primitive_id;
        h.mesh = RT_MESH(res, inst[ii].as_index);
        return h;
    }
    float tf = tcell;                    // past the cell: the nearest infinite plane that is not see-through
    for (uint k = 0; k < GRID_PLANES; ++k) {
        GridPlaneHit ph = grid_plane(org, dir, tf, max_range, e, c.grid);
        if (!ph.hit) break;
        if (lmat[ph.slot * 2].y < 0.0) { tf = ph.t + 1e-4; continue; }
        h.hit = true; h.plane = true; h.dist = ph.t; h.slot = ph.slot; h.pn = ph.n;
        return h;
    }
    return h;
}

// Specular backscatter of a smooth surface toward the sensor (the retro direction, H = N): pi f cos(theta) with the
// microfacet BRDF f = F D G / (4 cos^2), Beckmann D (its tail falls as exp(-tan^2 / alpha^2): glass returns only near
// normal incidence) and Smith G for a coincident source and receiver; same units as the Lambertian rho cos(theta).
inline float specular_backscatter(float F, float rough, float cosi) {
    if (cosi <= 1e-4) return 0.0;
    float a = max(rough * rough, 1e-4), a2 = a * a;
    float c2 = cosi * cosi, t2 = (1.0 - c2) / c2;
    float D = exp(-t2 / a2) / (3.14159265 * a2 * c2 * c2);
    float g1 = 2.0 * cosi / (cosi + sqrt(a2 + (1.0 - a2) * c2));
    return 3.14159265 * F * D * g1 * g1 / (4.0 * cosi);
}
#endif

// Lidar: one thread per (env, beam). Beams: (azimuth, elevation) in the sensor site frame, MuJoCo
// site convention (x forward here: beam direction = R_site * (cos el cos az, cos el sin az, sin el)).
kernel void lidar(
    instance_acceleration_structure accel [[buffer(0)]],
    device const float2*    beams      [[buffer(1)]],   // (n_beams) azimuth, elevation (rad)
    device const float*     site_xpos  [[buffer(2)]],   // (n_envs, n_site, 3)
    device const float*     site_xmat  [[buffer(3)]],   // (n_envs, n_site, 9)
    device const uint2*     inst_tab   [[buffer(4)]],
    device const int*       slot_mesh  [[buffer(5)]],
    device const MeshInfo*  meshes     [[buffer(6)]],
    device const float*     verts      [[buffer(7)]],
    device const uint*      indices    [[buffer(8)]],
    device const Material*  mats       [[buffer(9)]],
    device const InstanceDesc* inst    [[buffer(10)]],
    constant RTConsts&      c          [[buffer(11)]],
    device float*   out_dist   [[buffer(12)]],   // (n_envs, n_beams): range (0 = no return)
    device float*   out_points [[buffer(13)]],   // (n_envs, n_beams, 3) world frame
    device float*   out_normal [[buffer(14)]],   // (n_envs, n_beams, 3)
    device int*     out_slot   [[buffer(15)]],   // (n_envs, n_beams) hit slot (-1 none)
    device float*   out_intensity [[buffer(16)]],// (n_envs, n_beams)
#ifdef SENSOR_TR
    device const float4*    lmat       [[buffer(17)]],  // (n_slots, 2) see-through slots (m0.y < 0; first_surface)
#endif
    DYN_KERNEL_ARG
    uint2 tid [[thread_position_in_grid]])
{
    uint b = tid.x, e = tid.y;
    if (b >= c.n_beams || e >= c.n_envs) return;
    uint si = e * c.n_site + c.sensor_site;
    float3 origin = float3(site_xpos[si * 3], site_xpos[si * 3 + 1], site_xpos[si * 3 + 2]);
    float3x3 R = load_mat33(site_xmat + si * 9);
    float2 ae = beams[b];
    float3 dl = float3(cos(ae.y) * cos(ae.x), cos(ae.y) * sin(ae.x), sin(ae.y));
    float3 dir = normalize(R * dl);
    float3 org0 = origin + env_offset(e, c);
#ifdef SENSOR_TR
    SensorHit sh = first_surface(accel, org0, dir, 0.001, c.max_range, e, c, inst_tab, inst, lmat DYN_PASS);
    uint o = e * c.n_beams + b;
    float dist; uint slot; float3 n;
    if (!sh.hit) {
        out_dist[o] = 0.0; out_slot[o] = -1; out_intensity[o] = 0.0;
        out_points[o * 3] = out_points[o * 3 + 1] = out_points[o * 3 + 2] = 0.0;
        out_normal[o * 3] = out_normal[o * 3 + 1] = out_normal[o * 3 + 2] = 0.0;
        return;
    }
    dist = sh.dist; slot = sh.slot;
    if (sh.plane) n = sh.pn;
    else {
        InstanceDesc d = inst[sh.inst];
        n = tri_normal(sh.mesh, sh.prim, meshes, verts, indices, float3x3(float3(d.col0), float3(d.col1), float3(d.col2)));
    }
#else
    ray r(org0, dir, 0.001, grid_tmax(org0, dir, c.max_range, e, c.grid));
#ifdef DYN_SBVH
    RTRes res = RT_CLOSEST(r, accel);
#else
    intersector<triangle_data, instancing> isect;
    isect.accept_any_intersection(false);
    intersection_result<triangle_data, instancing> res = isect.intersect(r, accel);
#endif
    uint o = e * c.n_beams + b;
    float dist; uint slot; float3 n;
    if (res.type == intersection_type::none) {
        // compact env grid: an infinite plane past the env's cell (env_grid.h)
        GridPlaneHit ph = grid_plane(org0, dir, r.max_distance, c.max_range, e, c.grid);
        if (!ph.hit) {
            out_dist[o] = 0.0; out_slot[o] = -1; out_intensity[o] = 0.0;
            out_points[o * 3] = out_points[o * 3 + 1] = out_points[o * 3 + 2] = 0.0;
            out_normal[o * 3] = out_normal[o * 3 + 1] = out_normal[o * 3 + 2] = 0.0;
            return;
        }
        dist = ph.t; slot = ph.slot; n = ph.n;
    } else {
        uint ii = res.instance_id;
        slot = inst_tab[ii].y + res.geometry_id;          // geometry: 0 for geoms; the object within a dynamic env structure
        uint mesh = RT_MESH(res, inst[ii].as_index);      // = slot_mesh[slot] for geoms
        InstanceDesc d = inst[ii];
        float3x3 Ri = float3x3(float3(d.col0), float3(d.col1), float3(d.col2));
        n = tri_normal(mesh, res.primitive_id, meshes, verts, indices, Ri);
        dist = res.distance;
    }
#endif
    if (dot(n, dir) > 0.0) n = -n;
    float3 p = origin + dir * dist;
    float refl = mats[slot].par.w > 0.0 ? mats[slot].par.w : 0.5;   // material reflectance (MuJoCo) as albedo proxy
    float inten = refl * max(dot(n, -dir), 0.0) / max(dist * dist, 1e-4);
    out_dist[o] = dist;
    out_slot[o] = int(slot);
    out_intensity[o] = inten;
    out_points[o * 3] = p.x; out_points[o * 3 + 1] = p.y; out_points[o * 3 + 2] = p.z;
    out_normal[o * 3] = n.x; out_normal[o * 3 + 1] = n.y; out_normal[o * 3 + 2] = n.z;
}

// Ray-cast depth camera: one thread per (x, y, env); pinhole from the camera pose + intrinsics.
kernel void raycast_depth(
    instance_acceleration_structure accel [[buffer(0)]],
    device const float*     cam_xpos   [[buffer(1)]],   // (n_envs, n_cam, 3)
    device const float*     cam_xmat   [[buffer(2)]],   // (n_envs, n_cam, 9)
    constant float4&        intrinsics [[buffer(3)]],   // fx, fy, cx, cy
    constant uint4&         dims       [[buffer(4)]],   // width, height, n_cam, cam_id
    constant RTConsts&      c          [[buffer(5)]],
    device float*           out_depth  [[buffer(6)]],   // (n_envs, h, w) distance to image plane
    constant LensParams&    lens       [[buffer(7)]],   // lens model (lens.h; 0 = pinhole from `intrinsics`)
#ifdef SENSOR_TR
    device const uint2*     inst_tab   [[buffer(8)]],
    device const InstanceDesc* inst    [[buffer(9)]],
    device const float4*    lmat       [[buffer(10)]],  // (n_slots, 2) see-through slots (m0.y < 0; first_surface)
#endif
    DYN_KERNEL_ARG
    uint3 tid [[thread_position_in_grid]])
{
    uint x = tid.x, y = tid.y, e = tid.z;
    uint w = dims.x, h = dims.y;
    if (x >= w || y >= h || e >= c.n_envs) return;
    uint ci = e * dims.z + dims.w;
    float3 origin = float3(cam_xpos[ci * 3], cam_xpos[ci * 3 + 1], cam_xpos[ci * 3 + 2]);
    float3x3 R = load_mat33(cam_xmat + ci * 9);
    // camera looks along -Z, +Y up; pixel (x+0.5, y+0.5) with y down
    float3 dl;
    uint o0 = (e * h + y) * w + x;
    if (!lens_ray(lens, float(x) + 0.5, float(y) + 0.5, float(w), float(h), intrinsics, dl)) { out_depth[o0] = 0.0; return; }
    float3 dir = normalize(R * dl);
    float3 org0 = origin + env_offset(e, c);
#ifdef SENSOR_TR
    uint o = (e * h + y) * w + x;
    SensorHit sh = first_surface(accel, org0, dir, 0.001, c.max_range, e, c, inst_tab, inst, lmat DYN_PASS);
    if (!sh.hit) { out_depth[o] = 0.0; return; }
    float dist = sh.dist;
#else
    ray r(org0, dir, 0.001, grid_tmax(org0, dir, c.max_range, e, c.grid));
#ifdef DYN_SBVH
    RTRes res = RT_CLOSEST(r, accel);
#else
    intersector<triangle_data, instancing> isect;
    intersection_result<triangle_data, instancing> res = isect.intersect(r, accel);
#endif
    uint o = (e * h + y) * w + x;
    float dist = res.distance;
    if (res.type == intersection_type::none) {
        GridPlaneHit ph = grid_plane(org0, dir, r.max_distance, c.max_range, e, c.grid);   // a plane past the cell
        if (!ph.hit) { out_depth[o] = 0.0; return; }
        dist = ph.t;
    }
#endif
    float3 hit = dir * dist;
    out_depth[o] = -dot(hit, R[2]);   // depth along the camera's -Z axis
}

// Lidar extras (beam divergence, multi-return, per-slot reflectance). The default lidar() kernel
// above is untouched; this kernel runs only when an extra is requested.
//
// Per beam: n_sub sub-rays spread over the beam's divergence cone (offsets given as tangent-plane
// slopes along the beam's azimuth / elevation directions, with a per-sub-ray power weight). Each
// sub-ray is cast up to hits_per_ray times: after a hit it continues past the hit object (further
// hits on the same instance, i.e. its exit faces, are skipped) with its power scaled by
// `transmission`. Every hit is a candidate with power p = P * reflectance * cos(incidence) / r^2.
// Candidates are sorted by range and grouped into echoes: a new echo starts when a candidate lies
// more than min_sep beyond the first candidate of the current echo (NVIDIA's minDistBetweenEchosM).
// An echo's range and point are power-weighted means (a mixed pixel when a spot straddles an edge
// closer than min_sep), its intensity is the summed power, normal and slot come from its strongest
// candidate. The max_returns strongest echoes are reported in range order.
//
// With SENSOR_TR (transparent surfaces, lidar materials, return modes, detection threshold; raytrace.py Lidar): each
// hit's slot has a lidar material (lmat, see first_surface). A see-through slot (a < 0) is skipped. Otherwise the
// candidate's power is P (a rho cos + specular_backscatter(F)) / r^2 with F = R0 + (1 - R0)(1 - cos)^5 (Schlick), and
// the sub-ray goes on with the transmitted fraction (1 - a) tau (1 - F), or, for a ghost material whose reflected
// fraction (1 - a) F is the larger, along the mirror direction (the echo is then reported along the beam at the
// path length, where a real lidar puts it). Such passes do not count against hits_per_ray (at most
// LIDAR_TRANS_MAX per sub-ray); a candidate of a partially transparent surface weaker than 1e-12 of the sub-ray's
// power is dropped (a glass pane seen obliquely returns nothing). Echoes weaker than min_power are dropped, and
// return_mode (flags bits 8-15) picks the reported echoes: 0 the strongest, 1 the first, 2 the last, 3 the last plus
// the strongest of the others, 4 the first and the last, 5 the first plus the strongest of the others.
struct LidarExt { uint n_sub; uint max_returns; uint hits_per_ray; uint _p0; float min_sep; float transmission; float _p1, _p2; };
#define LIDAR_MAX_CAND 32

#ifdef SENSOR_TR
// one lidar sub-ray through transparent / reflecting surfaces (lidar_ext's model above), one surface per call
struct TrRay { float3 o, d; float t_off, tmin, P; int last, last_g; uint hits, passes; bool done; };
struct TrCand { bool ok; float rng, p; float3 n; uint slot; };
inline TrRay tr_ray(float3 org, float3 dir, float P) {
    TrRay s; s.o = org; s.d = dir; s.t_off = 0.0; s.tmin = 0.001; s.P = P; s.last = -1; s.last_g = -1;
    s.hits = 0; s.passes = 0; s.done = false;
    return s;
}
// the next surface of the sub-ray: a candidate echo (ok) of range rng (path length) and power p, or none (a skipped
// see-through / exit face, or the end: s.done); advances the ray past it (transmitted, reflected or, for an opaque
// surface, on with x.transmission counting a hit)
inline TrCand tr_step(instance_acceleration_structure accel, thread TrRay& s, float w0, uint e, constant RTConsts& c,
                      constant LidarExt& x, device const uint2* inst_tab, device const MeshInfo* meshes,
                      device const float* verts, device const uint* indices, device const InstanceDesc* inst,
                      device const float* slot_refl, device const float4* lmat DYN_PARAM)
{
    TrCand cd; cd.ok = false; cd.rng = 0.0; cd.p = 0.0; cd.n = 0.0; cd.slot = 0u;
    float rem = c.max_range - s.t_off;
    if (rem <= s.tmin) { s.done = true; return cd; }
    float tcell = grid_tmax(s.o, s.d, rem, e, c.grid);
    float dist = 0.0; uint slot = 0u; float3 n = 0.0; int ii = -1, g = -1;
    // the next surface that is neither see-through (transparency "surface" / "behind") nor an exit / inner face of
    // the object just hit; after SKIP_MAX skipped surfaces the next one counts
    for (uint k = 0; k <= SKIP_MAX; ++k) {
        ray r(s.o, s.d, s.tmin, tcell);
#ifdef DYN_SBVH
        RTRes res = RT_CLOSEST(r, accel);
#else
        intersector<triangle_data, instancing> isect;
        isect.accept_any_intersection(false);
        intersection_result<triangle_data, instancing> res = isect.intersect(r, accel);
#endif
        if (res.type == intersection_type::none) {
            GridPlaneHit ph = grid_plane(s.o, s.d, max(tcell, s.tmin), rem, e, c.grid);
            if (!ph.hit) { s.done = true; return cd; }
            dist = ph.t; slot = ph.slot; n = ph.n; ii = -2; g = int(ph.slot);
            s.tmin = rem + 1.0;                                  // a plane past the cell ends the sub-ray
            if (lmat[slot * 2].y < 0.0) { s.done = true; return cd; }   // ... also a see-through one
            break;
        }
        dist = res.distance;
        s.tmin = res.distance + 1e-4;
        ii = int(res.instance_id);
        g = int(res.geometry_id);
        slot = inst_tab[ii].y + res.geometry_id;
        if (k < SKIP_MAX && (lmat[slot * 2].y < 0.0 || (ii == s.last && g == s.last_g))) continue;
        InstanceDesc di = inst[ii];
        float3x3 Ri = float3x3(float3(di.col0), float3(di.col1), float3(di.col2));
        n = tri_normal(RT_MESH(res, di.as_index), res.primitive_id, meshes, verts, indices, Ri);
        break;
    }
    if (ii == s.last && g == s.last_g) return cd;           // a plane already counted
    s.last = ii; s.last_g = g;
    if (dot(n, s.d) > 0.0) n = -n;
    float4 m0 = lmat[slot * 2], m1 = lmat[slot * 2 + 1];
    float a = m0.y, cosi = max(dot(n, -s.d), 0.0);
    float F = m0.z > 0.0 ? m0.z + (1.0 - m0.z) * pow(1.0 - cosi, 5.0) : 0.0;
    float rng = s.t_off + dist;                               // range along the beam: the path length
    float P = s.P;
    // (the plain kernel's P rho cos / r^2 bit for bit when a = 1 and F = 0)
    float p = (P * (a * slot_refl[slot]) * cosi + P * (F > 0.0 ? specular_backscatter(F, m0.w, cosi) : 0.0)) / max(rng * rng, 1e-4);
    float T = (1.0 - a) * m1.x * (1.0 - F), Rf = (1.0 - a) * F;
    bool ghost = m1.y > 0.5 && Rf > 0.0 && Rf >= T;
    bool partial = a < 0.995 || F > 0.0;
    cd.ok = !partial || p > 1e-12f * w0;
    cd.rng = rng; cd.p = max(p, 1e-12f * P); cd.n = n; cd.slot = slot;
    if ((T > 0.0 || ghost) && s.passes < LIDAR_TRANS_MAX && ii != -2) {
        ++s.passes;
        if (ghost) {                                          // on along the mirror direction from the hit point
            s.o = s.o + s.d * dist; s.d = reflect(s.d, n); s.t_off += dist; s.tmin = 1e-4;
            s.last = -1; s.last_g = -1;
            s.P = P * Rf;
        } else {
            s.P = P * T;
        }
        return cd;
    }
    ++s.hits;
    s.P = P * x.transmission;
    return cd;
}
#endif

kernel void lidar_ext(
    instance_acceleration_structure accel [[buffer(0)]],
    device const float2*    beams      [[buffer(1)]],
    device const float*     site_xpos  [[buffer(2)]],
    device const float*     site_xmat  [[buffer(3)]],
    device const uint2*     inst_tab   [[buffer(4)]],
    device const int*       slot_mesh  [[buffer(5)]],
    device const MeshInfo*  meshes     [[buffer(6)]],
    device const float*     verts      [[buffer(7)]],
    device const uint*      indices    [[buffer(8)]],
    device const float*     slot_refl  [[buffer(9)]],    // (n_slots) reflectance per slot
    device const InstanceDesc* inst    [[buffer(10)]],
    constant RTConsts&      c          [[buffer(11)]],
    device const float4*    sub        [[buffer(12)]],   // (n_sub) slope u, slope v, weight, 0
    constant LidarExt&      x          [[buffer(13)]],
    device float*   out_dist   [[buffer(14)]],   // (n_envs, n_beams, R)
    device float*   out_points [[buffer(15)]],   // (n_envs, n_beams, R, 3)
    device float*   out_normal [[buffer(16)]],   // (n_envs, n_beams, R, 3)
    device int*     out_slot   [[buffer(17)]],   // (n_envs, n_beams, R)
    device float*   out_intensity [[buffer(18)]],// (n_envs, n_beams, R)
    device int*     out_nret   [[buffer(19)]],   // (n_envs, n_beams)
#ifdef SENSOR_TR
    device const float4*    lmat       [[buffer(20)]],   // (n_slots, 2) lidar materials (see first_surface)
#endif
    DYN_KERNEL_ARG
    uint2 tid [[thread_position_in_grid]])
{
    uint b = tid.x, e = tid.y;
    if (b >= c.n_beams || e >= c.n_envs) return;
    uint si = e * c.n_site + c.sensor_site;
    float3 origin = float3(site_xpos[si * 3], site_xpos[si * 3 + 1], site_xpos[si * 3 + 2]);
    float3 org = origin + env_offset(e, c);
    float3x3 R = load_mat33(site_xmat + si * 9);
    float2 ae = beams[b];
    float3 d = R * float3(cos(ae.y) * cos(ae.x), cos(ae.y) * sin(ae.x), sin(ae.y));
    float3 u = R * float3(-sin(ae.x), cos(ae.x), 0.0);
    float3 v = R * float3(-sin(ae.y) * cos(ae.x), -sin(ae.y) * sin(ae.x), cos(ae.y));

    float cr[LIDAR_MAX_CAND]; float cp[LIDAR_MAX_CAND]; float3 cpt[LIDAR_MAX_CAND]; float3 cn[LIDAR_MAX_CAND]; int cs[LIDAR_MAX_CAND];
    uint nc = 0;
    intersector<triangle_data, instancing> isect;
    isect.accept_any_intersection(false);
#ifdef SENSOR_TR
    uint rmode = (x._p0 >> 8) & 0xFFu;
    for (uint k = 0; k < x.n_sub; ++k) {
        float4 s = sub[k];
        float3 dir = normalize(d + s.x * u + s.y * v);
        TrRay st = tr_ray(org, dir, s.z);
        for (uint it = 0; it < x.hits_per_ray + 8 + 3 * LIDAR_TRANS_MAX && st.hits < x.hits_per_ray && !st.done; ++it) {
            TrCand cd = tr_step(accel, st, s.z, e, c, x, inst_tab, meshes, verts, indices, inst, slot_refl, lmat DYN_PASS);
            if (cd.ok && nc < LIDAR_MAX_CAND) {
                // insertion by range
                uint j = nc;
                while (j > 0 && cr[j - 1] > cd.rng) {
                    cr[j] = cr[j - 1]; cp[j] = cp[j - 1]; cpt[j] = cpt[j - 1]; cn[j] = cn[j - 1]; cs[j] = cs[j - 1]; --j;
                }
                cr[j] = cd.rng; cp[j] = cd.p; cpt[j] = origin + dir * cd.rng; cn[j] = cd.n; cs[j] = int(cd.slot);
                ++nc;
            }
        }
    }
#else
    for (uint k = 0; k < x.n_sub; ++k) {
        float4 s = sub[k];
        float3 dir = normalize(d + s.x * u + s.y * v);
        float P = s.z;
        float tmin = 0.001;
        int last = -1, last_g = -1;
        uint hits = 0;
        for (uint it = 0; it < x.hits_per_ray + 8 && hits < x.hits_per_ray; ++it) {
            ray r(org, dir, tmin, grid_tmax(org, dir, c.max_range, e, c.grid));
#ifdef DYN_SBVH
            RTRes res = RT_CLOSEST(r, accel);
#else
            intersection_result<triangle_data, instancing> res = isect.intersect(r, accel);
#endif
            float dist; uint slot; float3 n; int ii, g;
            if (res.type == intersection_type::none) {
                // compact env grid: an infinite plane past the env's cell ends the sub-ray (env_grid.h)
                GridPlaneHit ph = grid_plane(org, dir, max(r.max_distance, tmin), c.max_range, e, c.grid);
                if (!ph.hit) break;
                dist = ph.t; slot = ph.slot; n = ph.n; ii = -2; g = int(ph.slot);
                tmin = c.max_range + 1.0;
            } else {
                dist = res.distance;
                tmin = res.distance + 1e-4;
                ii = int(res.instance_id);
                g = int(res.geometry_id);             // the object: instance, and geometry within a dynamic env structure
                slot = inst_tab[ii].y + res.geometry_id;
                InstanceDesc di = inst[ii];
                float3x3 Ri = float3x3(float3(di.col0), float3(di.col1), float3(di.col2));
                n = tri_normal(RT_MESH(res, di.as_index), res.primitive_id, meshes, verts, indices, Ri);
            }
            if (ii == last && g == last_g) continue;  // exit / inner faces of the object just hit
            last = ii; last_g = g;
            if (dot(n, dir) > 0.0) n = -n;
            float p = P * slot_refl[slot] * max(dot(n, -dir), 0.0) / max(dist * dist, 1e-4);
            if (nc < LIDAR_MAX_CAND) {
                // insertion by range
                uint j = nc;
                while (j > 0 && cr[j - 1] > dist) {
                    cr[j] = cr[j - 1]; cp[j] = cp[j - 1]; cpt[j] = cpt[j - 1]; cn[j] = cn[j - 1]; cs[j] = cs[j - 1]; --j;
                }
                cr[j] = dist; cp[j] = max(p, 1e-12f * P); cpt[j] = origin + dir * dist; cn[j] = n; cs[j] = int(slot);
                ++nc;
            }
            ++hits;
            P *= x.transmission;
        }
    }
#endif
    // group into echoes (in place: echo q summarised into index q)
    uint ne = 0;
    uint i = 0;
    float er[LIDAR_MAX_CAND]; float ep[LIDAR_MAX_CAND];
    while (i < nc) {
        float r0 = cr[i];
        float sp = 0.0, sr = 0.0; float3 spt = 0.0; float best = -1.0; float3 bn = 0.0; int bs = -1;
        while (i < nc && cr[i] - r0 <= x.min_sep) {
            sp += cp[i]; sr += cp[i] * cr[i]; spt += cp[i] * cpt[i];
            if (cp[i] > best) { best = cp[i]; bn = cn[i]; bs = cs[i]; }
            ++i;
        }
#ifdef SENSOR_TR
        if (sp < x._p1) continue;        // below the detection threshold (min_power)
#endif
        er[ne] = sr / sp; ep[ne] = sp; cpt[ne] = spt / sp; cn[ne] = bn; cs[ne] = bs;
        ++ne;
    }
    // keep the max_returns strongest echoes, in range order
    uint R_ = x.max_returns;
    uint keep_mask = 0;
    uint nk = min(ne, R_);
#ifdef SENSOR_TR
    uint q0 = 0;
    if (nk > 0) {
        if (rmode == 1u) { for (uint q = 0; q < nk; ++q) keep_mask |= 1u << q; q0 = nk; }                  // first
        else if (rmode == 2u) { for (uint q = 0; q < nk; ++q) keep_mask |= 1u << (ne - 1 - q); q0 = nk; }  // last
        else if (rmode == 3u) { keep_mask = 1u << (ne - 1); q0 = 1; }                                     // last + strongest
        else if (rmode == 4u) { keep_mask = 1u | (1u << (ne - 1)); q0 = ne > 1 ? 2u : 1u; }                // first + last
        else if (rmode == 5u) { keep_mask = 1u; q0 = 1; }                                                 // first + strongest
    }
    for (uint q = q0; q < nk; ++q) {
#else
    for (uint q = 0; q < nk; ++q) {
#endif
        int arg = -1; float mx = -1.0;
        for (uint t = 0; t < ne; ++t)
            if (!(keep_mask & (1u << t)) && ep[t] > mx) { mx = ep[t]; arg = int(t); }
        keep_mask |= (1u << uint(arg));
    }
    uint o = (e * c.n_beams + b) * R_;
    uint w = 0;
    for (uint t = 0; t < ne; ++t) {
        if (!(keep_mask & (1u << t))) continue;
        uint oo = o + w;
        out_dist[oo] = er[t]; out_intensity[oo] = ep[t]; out_slot[oo] = cs[t];
        out_points[oo * 3] = cpt[t].x; out_points[oo * 3 + 1] = cpt[t].y; out_points[oo * 3 + 2] = cpt[t].z;
        out_normal[oo * 3] = cn[t].x; out_normal[oo * 3 + 1] = cn[t].y; out_normal[oo * 3 + 2] = cn[t].z;
        ++w;
    }
    out_nret[e * c.n_beams + b] = int(w);
    for (; w < R_; ++w) {
        uint oo = o + w;
        out_dist[oo] = 0.0; out_intensity[oo] = 0.0; out_slot[oo] = -1;
        out_points[oo * 3] = out_points[oo * 3 + 1] = out_points[oo * 3 + 2] = 0.0;
        out_normal[oo * 3] = out_normal[oo * 3 + 1] = out_normal[oo * 3 + 2] = 0.0;
    }
}

#ifdef SENSOR_TR
// lidar_ext's model for one sub-ray (no divergence) and one return, echoes formed and chosen as they come (a sub-ray's
// candidates arrive in range order): no candidate arrays, the common transparent-scene configuration
kernel void lidar_tr1(
    instance_acceleration_structure accel [[buffer(0)]],
    device const float2*    beams      [[buffer(1)]],
    device const float*     site_xpos  [[buffer(2)]],
    device const float*     site_xmat  [[buffer(3)]],
    device const uint2*     inst_tab   [[buffer(4)]],
    device const int*       slot_mesh  [[buffer(5)]],
    device const MeshInfo*  meshes     [[buffer(6)]],
    device const float*     verts      [[buffer(7)]],
    device const uint*      indices    [[buffer(8)]],
    device const float*     slot_refl  [[buffer(9)]],
    device const InstanceDesc* inst    [[buffer(10)]],
    constant RTConsts&      c          [[buffer(11)]],
    device const float4*    sub        [[buffer(12)]],
    constant LidarExt&      x          [[buffer(13)]],
    device float*   out_dist   [[buffer(14)]],
    device float*   out_points [[buffer(15)]],
    device float*   out_normal [[buffer(16)]],
    device int*     out_slot   [[buffer(17)]],
    device float*   out_intensity [[buffer(18)]],
    device int*     out_nret   [[buffer(19)]],
    device const float4*    lmat       [[buffer(20)]],
    DYN_KERNEL_ARG
    uint2 tid [[thread_position_in_grid]])
{
    uint b = tid.x, e = tid.y;
    if (b >= c.n_beams || e >= c.n_envs) return;
    uint si = e * c.n_site + c.sensor_site;
    float3 origin = float3(site_xpos[si * 3], site_xpos[si * 3 + 1], site_xpos[si * 3 + 2]);
    float3 org = origin + env_offset(e, c);
    float3x3 R = load_mat33(site_xmat + si * 9);
    float2 ae = beams[b];
    float3 d = R * float3(cos(ae.y) * cos(ae.x), cos(ae.y) * sin(ae.x), sin(ae.y));
    float3 u = R * float3(-sin(ae.x), cos(ae.x), 0.0);
    float3 v = R * float3(-sin(ae.y) * cos(ae.x), -sin(ae.y) * sin(ae.x), cos(ae.y));
    uint rmode = (x._p0 >> 8) & 0xFFu;
    float4 s0 = sub[0];
    float3 dir = normalize(d + s0.x * u + s0.y * v);
    TrRay st = tr_ray(org, dir, s0.z);
    // the open echo (candidates within min_sep of its first) and the chosen one
    // (every candidate lies on the beam: an echo's point is origin + dir x its range, so only ranges are summed)
    bool g_open = false; float g_r0 = 0.0, g_sp = 0.0, g_sr = 0.0, g_best = -1.0; float3 g_bn = 0.0; int g_bs = -1;
    bool have = false; float s_r = 0.0, s_p = -1.0; float3 s_n = 0.0; int s_slot = -1;
    uint guard = x.hits_per_ray + 8 + 3 * LIDAR_TRANS_MAX;
    for (uint it = 0; it <= guard; ++it) {
        bool more = it < guard && st.hits < x.hits_per_ray && !st.done;
        TrCand cd; cd.ok = false;
        if (more) cd = tr_step(accel, st, s0.z, e, c, x, inst_tab, meshes, verts, indices, inst, slot_refl, lmat DYN_PASS);
        if (more && !cd.ok) continue;
        if (g_open && (!more || cd.rng - g_r0 > x.min_sep)) {     // the open echo is complete
            g_open = false;
            bool take = g_sp >= x._p1 && (rmode == 1u ? !have : (rmode == 2u ? true : g_sp > s_p));   // first / last / strongest
            if (take) { s_r = g_sr / g_sp; s_p = g_sp; s_n = g_bn; s_slot = g_bs; have = true; }
            if (rmode == 1u && have) break;                        // first: found
        }
        if (!more) break;
        if (!g_open) { g_open = true; g_r0 = cd.rng; g_sp = 0.0; g_sr = 0.0; g_best = -1.0; }
        g_sp += cd.p; g_sr += cd.p * cd.rng;
        if (cd.p > g_best) { g_best = cd.p; g_bn = cd.n; g_bs = int(cd.slot); }
    }
    uint o = e * c.n_beams + b;
    float3 s_pt = origin + dir * s_r;
    if (!have) { s_r = 0.0; s_p = 0.0; s_pt = 0.0; s_n = 0.0; s_slot = -1; }
    out_dist[o] = s_r; out_intensity[o] = s_p; out_slot[o] = s_slot;
    out_points[o * 3] = s_pt.x; out_points[o * 3 + 1] = s_pt.y; out_points[o * 3 + 2] = s_pt.z;
    out_normal[o * 3] = s_n.x; out_normal[o * 3 + 1] = s_n.y; out_normal[o * 3 + 2] = s_n.z;
    out_nret[o] = have ? 1 : 0;
}
#endif
