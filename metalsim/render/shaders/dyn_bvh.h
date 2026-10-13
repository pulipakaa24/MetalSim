// Dynamic objects as one bounding box per env inside the hardware instance structure ("sbvh" layout,
// metalsim/sensors/dyn_bvh.py). A ray that enters env e's box traverses e's own BVH over its triangles (topology shared
// by every env, node bounds refit on the GPU each frame) in the env's frame, as the per-env structures did. Closest-hit
// and any-hit queries go through intersection_query: opaque triangles (the rigid geoms) are committed by the hardware,
// boxes come back here. The result mimics intersection_result (type, distance, instance / geometry / primitive ids,
// barycentrics): a dynamic hit reports the box instance (identity transform; inst_tab x = number of objects D, y = the
// first dynamic slot), geometry = the object, primitive = the triangle within the object's env mesh, plus the env.
//
// Without DYN_SBVH the macros below expand to the plain intersector and nothing here is compiled.
#ifndef METALSIM_DYN_BVH_H   // include guard: shaders are inlined from strings (metal_context._inline_includes)
#define METALSIM_DYN_BVH_H

#ifdef DYN_SBVH
#include "env_grid.h"
struct DynBVHArgs {
    device const float*  verts;     // records (8 floats: position, normal, uv), the ray tracer's vertex buffer
    device const uint*   indices;   // the ray tracer's index buffer
    device const uint4*  meshes;    // MeshInfo (i_off, i_count, -, -)
    device const uint4*  topo;      // per node: left, right, first leaf entry, leaf count (0: inner node)
    device const uint*   order;     // leaf entries: object << 24 | triangle
    device const float4* bounds;    // (n_envs, n_nodes, 2): lo, hi in the env's frame
    device const float*  tris;      // (n_envs, n_tris, 9): each env's triangles in leaf order (written by the refit)
    uint n_envs, n_nodes, D, mesh_base;   // mesh entry of (env, object) = mesh_base + env * D + object
    uint box_inst, n_tris, _p0, _p1;
    EnvGrid grid;                   // where the envs sit (env_grid.h); byte 96 (dyn_bvh.py _write_args)
};

struct DynHit { bool hit; float t; uint k, prim, env; float2 bc; };

inline float3 dyn_env_offset(device const DynBVHArgs& a, uint e) {
    return grid_offset(e, a.grid);
}

inline float3 dyn_vert(device const DynBVHArgs& a, uint i) {
    return float3(a.verts[i * 8], a.verts[i * 8 + 1], a.verts[i * 8 + 2]);
}

// slab test: entry distance, or -1 when the box is missed within [tmin, tmax] (no infinities: fast math)
inline float dyn_box_entry(float4 lo, float4 hi, float3 o, float3 inv, float tmin, float tmax) {
    float3 t0 = (lo.xyz - o) * inv, t1 = (hi.xyz - o) * inv;
    float3 tn = min(t0, t1), tf = max(t0, t1);
    float a = max(max(tn.x, tn.y), max(tn.z, tmin));
    float b = min(min(tf.x, tf.y), min(tf.z, tmax));
    return a <= b ? a : -1.0;
}

// env e's BVH, ray (o, d) in world coordinates: the env's frame is a translation by its grid offset
inline DynHit dyn_traverse(device const DynBVHArgs& a, uint e, float3 ow, float3 d, float tmin, float tmax, bool any) {
    DynHit h; h.hit = false; h.t = tmax; h.env = e;
    float3 o = ow - dyn_env_offset(a, e);
    // finite reciprocals (fast math assumes no infinities): a zero component becomes +-1e30
    float3 inv = 1.0 / select(d, copysign(float3(1e-30), d), fabs(d) < 1e-30);
    device const float4* B = a.bounds + (ulong)e * a.n_nodes * 2;
    // watertight ray-triangle test (Woop, Benthin, Wald 2013), as hardware ray tracers do: shear to the ray's
    // dominant axis once per ray
    float3 ad = fabs(d);
    int kz = ad.x > ad.y ? (ad.x > ad.z ? 0 : 2) : (ad.y > ad.z ? 1 : 2);
    int kx = kz == 2 ? 0 : kz + 1, ky = kx == 2 ? 0 : kx + 1;
    if (d[kz] < 0.0) { int tmp = kx; kx = ky; ky = tmp; }
    float Sz = 1.0 / d[kz], Sx = d[kx] * Sz, Sy = d[ky] * Sz;
    uint stack[32];
    int sp = 0;
    uint node = 0;
    if (dyn_box_entry(B[0], B[1], o, inv, tmin, h.t) < 0.0) return h;
    for (uint guard = 0; guard < 4u * a.n_nodes + 8u; ++guard) {      // bounded: every node is visited at most once
        uint4 n = a.topo[node];
        if (n.w > 0) {        // leaf: its triangles
            for (uint j = n.z; j < n.z + n.w; ++j) {
                device const float* tv = a.tris + ((ulong)e * a.n_tris + j) * 9;
                float3 v0 = float3(tv[0], tv[1], tv[2]), v1 = float3(tv[3], tv[4], tv[5]), v2 = float3(tv[6], tv[7], tv[8]);
                float3 A = v0 - o, Bv = v1 - o, C = v2 - o;
                float Ax = A[kx] - Sx * A[kz], Ay = A[ky] - Sy * A[kz];
                float Bx = Bv[kx] - Sx * Bv[kz], By = Bv[ky] - Sy * Bv[kz];
                float Cx = C[kx] - Sx * C[kz], Cy = C[ky] - Sy * C[kz];
                float U = Cx * By - Cy * Bx, V = Ax * Cy - Ay * Cx, W = Bx * Ay - By * Ax;
                if ((U < 0.0 || V < 0.0 || W < 0.0) && (U > 0.0 || V > 0.0 || W > 0.0)) continue;   // two-sided
                float det = U + V + W;
                if (det == 0.0) continue;
                float T = U * (Sz * A[kz]) + V * (Sz * Bv[kz]) + W * (Sz * C[kz]);
                float t = T / det;
                if (t > tmin && t < h.t) {
                    uint ent = a.order[j];
                    h.hit = true; h.t = t; h.k = ent >> 24; h.prim = ent & 0xFFFFFFu; h.bc = float2(V / det, W / det);
                    if (any) return h;
                }
            }
        } else {              // inner: nearer child first
            float tl = dyn_box_entry(B[n.x * 2], B[n.x * 2 + 1], o, inv, tmin, h.t);
            float tr = dyn_box_entry(B[n.y * 2], B[n.y * 2 + 1], o, inv, tmin, h.t);
            if (tl >= 0.0 && tr >= 0.0) {
                uint nearc = tl <= tr ? n.x : n.y, farc = tl <= tr ? n.y : n.x;
                if (sp < 32) stack[sp++] = farc;
                node = nearc;
                continue;
            }
            if (tl >= 0.0) { node = n.x; continue; }
            if (tr >= 0.0) { node = n.y; continue; }
        }
        // pop the next subtree still in front of the closest hit
        bool found = false;
        while (sp > 0) {
            uint c = stack[--sp];
            if (dyn_box_entry(B[c * 2], B[c * 2 + 1], o, inv, tmin, h.t) >= 0.0) { node = c; found = true; break; }
        }
        if (!found) break;
    }
    return h;
}

struct RTRes {
    intersection_type type; float distance;
    uint instance_id, geometry_id, primitive_id;
    float2 triangle_barycentric_coord;
    bool dyn; uint dyn_env;
};

inline RTRes rt_intersect(ray r, instance_acceleration_structure accel, device const DynBVHArgs& a, bool any) {
    intersection_params prm;
    prm.accept_any_intersection(any);
    intersection_query<triangle_data, instancing> q(r, accel, prm);
    DynHit best; best.hit = false;
    while (q.next()) {
        if (q.get_candidate_intersection_type() == intersection_type::bounding_box) {
            float cur = q.get_committed_intersection_type() == intersection_type::none ? r.max_distance : q.get_committed_distance();
            // the box instance has the identity transform: the candidate ray is the world ray
            DynHit h = dyn_traverse(a, q.get_candidate_primitive_id(), r.origin, r.direction, r.min_distance, cur, any);
            if (h.hit) {
                q.commit_bounding_box_intersection(h.t);
                best = h;
                if (any) q.abort();
            }
        } else {
            q.commit_triangle_intersection();     // non-opaque triangles (none in these scenes): as the intersector
        }
    }
    RTRes o;
    o.type = q.get_committed_intersection_type();
    if (any && best.hit) o.type = intersection_type::bounding_box;    // after abort(): the box hit found
    o.dyn = false; o.dyn_env = 0; o.distance = 0.0; o.instance_id = 0; o.geometry_id = 0; o.primitive_id = 0;
    o.triangle_barycentric_coord = float2(0.0);
    if (o.type == intersection_type::triangle) {
        o.distance = q.get_committed_distance();
        o.instance_id = q.get_committed_instance_id();
        o.geometry_id = q.get_committed_geometry_id();
        o.primitive_id = q.get_committed_primitive_id();
        o.triangle_barycentric_coord = q.get_committed_triangle_barycentric_coord();
    } else if (o.type == intersection_type::bounding_box) {
        o.type = intersection_type::triangle;
        o.distance = best.t;
        o.instance_id = q.get_committed_instance_id();
        o.geometry_id = best.k; o.primitive_id = best.prim; o.triangle_barycentric_coord = best.bc;
        o.dyn = true; o.dyn_env = best.env;
    }
    return o;
}

#define DYN_PARAM , device const DynBVHArgs& dbv
#define DYN_PASS , dbv
#define DYN_KERNEL_ARG device const DynBVHArgs& dbv [[buffer(30)]],
#define RT_RESULT RTRes
#define RT_CLOSEST(r, accel) rt_intersect((r), (accel), dbv, false)
#define RT_ANY(r, accel) rt_intersect((r), (accel), dbv, true)
// mesh entry of a hit: the instance's first listing + geometry (rigid, per-env layouts), or (env, object) for sbvh
#define RT_MESH(res, as_index) ((res).dyn ? dbv.mesh_base + (res).dyn_env * dbv.D + (res).geometry_id : (as_index) + (res).geometry_id)
// translation of the hit's frame (the instance's; the env's grid offset for sbvh hits)
#define RT_TRANSLATION(res, col3) ((res).dyn ? dyn_env_offset(dbv, (res).dyn_env) : float3(col3))
#else
#define DYN_PARAM
#define DYN_PASS
#define DYN_KERNEL_ARG
#define RT_RESULT intersection_result<triangle_data, instancing>
#define RT_CLOSEST(r, accel) intersector<triangle_data, instancing>().intersect((r), (accel))
#define RT_ANY(r, accel) rt_any_plain((r), (accel))
#define RT_MESH(res, as_index) ((as_index) + (res).geometry_id)
#define RT_TRANSLATION(res, col3) float3(col3)
inline intersection_result<triangle_data, instancing> rt_any_plain(ray r, instance_acceleration_structure accel) {
    intersector<triangle_data, instancing> i;
    i.accept_any_intersection(true);
    return i.intersect(r, accel);
}
#endif
#endif  // METALSIM_DYN_BVH_H
