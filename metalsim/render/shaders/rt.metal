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

struct RTConsts {
    uint n_envs, n_geoms, n_slots, tiles_per_row;
    float stride;              // env grid spacing (m)
    float max_range;
    uint n_beams;              // beams per lidar
    uint n_site;               // model sites (stride of site_xpos per env)
    uint sensor_site;          // site index the lidar is attached to
    uint _pad0, _pad1, _pad2;
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

inline float3 env_offset(uint e, constant RTConsts& c) {
    return float3(float(e % c.tiles_per_row) * c.stride, float(e / c.tiles_per_row) * c.stride, 0.0);
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
    return normalize(R * cross(b - a, cc - a));
}

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
    ray r(origin + env_offset(e, c), dir, 0.001, c.max_range);
    intersector<triangle_data, instancing> isect;
    isect.accept_any_intersection(false);
    intersection_result<triangle_data, instancing> res = isect.intersect(r, accel);
    uint o = e * c.n_beams + b;
    if (res.type == intersection_type::none) {
        out_dist[o] = 0.0; out_slot[o] = -1; out_intensity[o] = 0.0;
        out_points[o * 3] = out_points[o * 3 + 1] = out_points[o * 3 + 2] = 0.0;
        out_normal[o * 3] = out_normal[o * 3 + 1] = out_normal[o * 3 + 2] = 0.0;
        return;
    }
    uint ii = res.instance_id;
    uint slot = inst_tab[ii].y;
    uint mesh = uint(slot_mesh[slot]);
    InstanceDesc d = inst[ii];
    float3x3 Ri = float3x3(float3(d.col0), float3(d.col1), float3(d.col2));
    float3 n = tri_normal(mesh, res.primitive_id, meshes, verts, indices, Ri);
    if (dot(n, dir) > 0.0) n = -n;
    float3 p = origin + dir * res.distance;
    float refl = mats[slot].par.w > 0.0 ? mats[slot].par.w : 0.5;   // material reflectance (MuJoCo) as albedo proxy
    float inten = refl * max(dot(n, -dir), 0.0) / max(res.distance * res.distance, 1e-4);
    out_dist[o] = res.distance;
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
    uint3 tid [[thread_position_in_grid]])
{
    uint x = tid.x, y = tid.y, e = tid.z;
    uint w = dims.x, h = dims.y;
    if (x >= w || y >= h || e >= c.n_envs) return;
    uint ci = e * dims.z + dims.w;
    float3 origin = float3(cam_xpos[ci * 3], cam_xpos[ci * 3 + 1], cam_xpos[ci * 3 + 2]);
    float3x3 R = load_mat33(cam_xmat + ci * 9);
    // camera looks along -Z, +Y up; pixel (x+0.5, y+0.5) with y down
    float3 dl = float3((float(x) + 0.5 - intrinsics.z) / intrinsics.x, -(float(y) + 0.5 - intrinsics.w) / intrinsics.y, -1.0);
    float3 dir = normalize(R * dl);
    ray r(origin + env_offset(e, c), dir, 0.001, c.max_range);
    intersector<triangle_data, instancing> isect;
    intersection_result<triangle_data, instancing> res = isect.intersect(r, accel);
    uint o = (e * h + y) * w + x;
    if (res.type == intersection_type::none) { out_depth[o] = 0.0; return; }
    float3 hit = dir * res.distance;
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
struct LidarExt { uint n_sub; uint max_returns; uint hits_per_ray; uint _p0; float min_sep; float transmission; float _p1, _p2; };
#define LIDAR_MAX_CAND 32

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
    for (uint k = 0; k < x.n_sub; ++k) {
        float4 s = sub[k];
        float3 dir = normalize(d + s.x * u + s.y * v);
        float P = s.z;
        float tmin = 0.001;
        int last = -1;
        uint hits = 0;
        for (uint it = 0; it < x.hits_per_ray + 8 && hits < x.hits_per_ray; ++it) {
            ray r(org, dir, tmin, c.max_range);
            intersection_result<triangle_data, instancing> res = isect.intersect(r, accel);
            if (res.type == intersection_type::none) break;
            tmin = res.distance + 1e-4;
            int ii = int(res.instance_id);
            if (ii == last) continue;                 // exit / inner faces of the object just hit
            last = ii;
            uint slot = inst_tab[ii].y;
            InstanceDesc di = inst[ii];
            float3x3 Ri = float3x3(float3(di.col0), float3(di.col1), float3(di.col2));
            float3 n = tri_normal(uint(slot_mesh[slot]), res.primitive_id, meshes, verts, indices, Ri);
            if (dot(n, dir) > 0.0) n = -n;
            float p = P * slot_refl[slot] * max(dot(n, -dir), 0.0) / max(res.distance * res.distance, 1e-4);
            if (nc < LIDAR_MAX_CAND) {
                // insertion by range
                uint j = nc;
                while (j > 0 && cr[j - 1] > res.distance) {
                    cr[j] = cr[j - 1]; cp[j] = cp[j - 1]; cpt[j] = cpt[j - 1]; cn[j] = cn[j - 1]; cs[j] = cs[j - 1]; --j;
                }
                cr[j] = res.distance; cp[j] = max(p, 1e-12f * P); cpt[j] = origin + dir * res.distance; cn[j] = n; cs[j] = int(slot);
                ++nc;
            }
            ++hits;
            P *= x.transmission;
        }
    }
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
        er[ne] = sr / sp; ep[ne] = sp; cpt[ne] = spt / sp; cn[ne] = bn; cs[ne] = bs;
        ++ne;
    }
    // keep the max_returns strongest echoes, in range order
    uint R_ = x.max_returns;
    uint keep_mask = 0;
    uint nk = min(ne, R_);
    for (uint q = 0; q < nk; ++q) {
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
