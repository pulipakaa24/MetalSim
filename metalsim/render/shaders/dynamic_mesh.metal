// metalsim dynamic meshes: per-env vertices of the objects MuJoCo's renderer draws besides geoms (flexes, skins,
// spatial tendons) and of user deformable meshes (XPBD / PhysX-port cloth), generated on the GPU each frame from the
// simulation state, plus the per-env pose gather that appends the drawn sites' poses to the geom poses.
//
// Stages (each a dispatch over (record, env); all loops are bounded by host-built tables):
//   copy_block    simulation vertex array (N, V, 3) -> the position scratch P (N, Ptot, 3)
//   skin_lbs      linear blend skinning (mjv_updateActiveSkin) -> P
//   vnormal       smoothed vertex normals from adjacent faces (flex: sum of unit face normals, MuJoCo's addNormal;
//                 skin: area-weighted) -> Nn (N, Ptot, 3)
//   emit_surface  surface vertex records (MuJoCo's makeSmooth / makeSide / makeFace offsets by the flex radius)
//   emit_tube     open cylinders (flex edges), capsules (tendon segments) and spheres (flex vertices) from a template
//                 (MuJoCo's builtin tessellation), oriented by mju_quatZ2Vec as mjv_connector / render_gl3 do
// Output records are 8 floats (position + env offset, normal, uv), the layout of the renderer's static vertices.
#include <metal_stdlib>
using namespace metal;

struct DynConsts {
    uint n_envs, ptot, tpr, rbase;      // rbase: first output record of the dynamic region in the vertex buffer
    float stride, _f0, _f1, _f2;        // env grid spacing of the ray tracer (0: raster only)
    uint n_src, src_stride, src_start, dst;     // copy_block: vertices, per-env stride and first vertex of the source, P offset
    uint nbody, ntendon, nwrap2, n_rec;         // nwrap2: wrap points per world (2 * nwrap); n_rec: records of this stage
};

struct SRec { int mode, a, b, c; int d, dst, vstride, _p; float pr, ns, u, v; };
struct TRec { int mode, a, b, tv; int dst, vstride, _p0, _p1; float r, _f0, _f1, _f2; };
struct NEnt { int p, start, count, mode; };          // vertex, face-list range, weighting (0 unit face normals, 1 area)
struct SkinV { int p, src, start, count; };          // P index, skin_vert index, bone-weight range
struct SkinB { int bone; float w; };
struct Bone  { float4 bindpos_body; float4 bindquat; };   // bindpos xyz, body id in w (float); bindquat wxyz

inline float3 env_offset(uint e, constant DynConsts& c) {
    return c.stride == 0.0f ? float3(0.0) : float3(float(e % c.tpr) * c.stride, float(e / c.tpr) * c.stride, 0.0);
}
inline float3 ld3(device const float* p, uint i) { return float3(p[3 * i], p[3 * i + 1], p[3 * i + 2]); }
inline void st3(device float* p, uint i, float3 v) { p[3 * i] = v.x; p[3 * i + 1] = v.y; p[3 * i + 2] = v.z; }
inline void put(device float* out, uint r, float3 pos, float3 n, float2 uv) {
    device float* o = out + r * 8;
    o[0] = pos.x; o[1] = pos.y; o[2] = pos.z; o[3] = n.x; o[4] = n.y; o[5] = n.z; o[6] = uv.x; o[7] = uv.y;
}
inline float3 unit_or_x(float3 v) {             // mju_normalize3: below mjMINVAL the result is (1, 0, 0)
    float l = length(v);
    return l < 1e-15f ? float3(1, 0, 0) : v / l;
}
inline float3x3 quat_mat(float4 q) {           // (w, x, y, z), columns
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return float3x3(float3(1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)),
                    float3(2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)),
                    float3(2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)));
}
inline float4 quat_mul(float4 a, float4 b) {
    return float4(a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w, a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
                  a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y, a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x);
}

// ---- positions ---------------------------------------------------------------------------------------------------

kernel void copy_block(device const float* src [[buffer(0)]], device float* P [[buffer(1)]], constant DynConsts& c [[buffer(2)]],
                       uint2 tid [[thread_position_in_grid]])
{
    uint v = tid.x, e = tid.y;
    if (v >= c.n_src || e >= c.n_envs) return;
    st3(P, e * c.ptot + c.dst + v, ld3(src, e * c.src_stride + c.src_start + v));
}

kernel void skin_lbs(device const SkinV* sv [[buffer(0)]], device const SkinB* sb [[buffer(1)]], device const Bone* bones [[buffer(2)]],
                     device const float* skin_vert [[buffer(3)]], device const float* xpos [[buffer(4)]],
                     device const float* xquat [[buffer(5)]], device float* P [[buffer(6)]], constant DynConsts& c [[buffer(7)]],
                     uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.x, e = tid.y;
    if (i >= c.n_rec || e >= c.n_envs) return;
    SkinV s = sv[i];
    float3 v0 = ld3(skin_vert, uint(s.src));
    float3 acc = float3(0.0);
    for (int k = 0; k < s.count; ++k) {
        SkinB b = sb[s.start + k];
        Bone bn = bones[b.bone];
        uint body = uint(bn.bindpos_body.w);
        device const float* q = xquat + (e * c.nbody + body) * 4;
        float4 bq = bn.bindquat;
        float4 quat = quat_mul(float4(q[0], q[1], q[2], q[3]), float4(bq.x, -bq.y, -bq.z, -bq.w));   // xquat * neg(bindquat)
        float3x3 R = quat_mat(quat);
        float3 t = ld3(xpos, e * c.nbody + body) - R * bn.bindpos_body.xyz;
        acc += b.w * (R * v0 + t);
    }
    st3(P, e * c.ptot + uint(s.p), acc);
}

// ---- normals ------------------------------------------------------------------------------------------------------

kernel void vnormal(device const NEnt* ent [[buffer(0)]], device const int* flist [[buffer(1)]], device const int4* faces [[buffer(2)]],
                    device const float* P [[buffer(3)]], device float* Nn [[buffer(4)]], constant DynConsts& c [[buffer(5)]],
                    uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.x, e = tid.y;
    if (i >= c.n_rec || e >= c.n_envs) return;
    NEnt n = ent[i];
    uint base = e * c.ptot;
    float3 acc = float3(0.0);
    for (int k = 0; k < n.count; ++k) {
        int4 f = faces[flist[n.start + k]];
        float3 a = ld3(P, base + uint(f.x));
        float3 fn = cross(ld3(P, base + uint(f.y)) - a, ld3(P, base + uint(f.z)) - a);
        acc += n.mode == 0 ? unit_or_x(fn) : fn;
    }
    float3 r;
    if (n.mode == 0) r = unit_or_x(acc);
    else { float l = length(acc); r = acc / max(l, 1e-15f); }
    st3(Nn, base + uint(n.p), r);
}

// ---- surface records ------------------------------------------------------------------------------------------------

kernel void emit_surface(device const SRec* rec [[buffer(0)]], device const float* P [[buffer(1)]], device const float* Nn [[buffer(2)]],
                         device float* out [[buffer(3)]], constant DynConsts& c [[buffer(4)]],
                         uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.x, e = tid.y;
    if (i >= c.n_rec || e >= c.n_envs) return;
    SRec r = rec[i];
    uint base = e * c.ptot;
    float3 pa = ld3(P, base + uint(r.a));
    float3 pos, nrm;
    if (r.mode == 3) {                         // flat face offset along its own normal (makeFace)
        float3 b = ld3(P, base + uint(r.b));
        float3 fn = unit_or_x(cross(ld3(P, base + uint(r.c)) - b, ld3(P, base + uint(r.d)) - b));
        pos = pa + r.pr * fn; nrm = r.ns * fn;
    } else {
        float3 na = ld3(Nn, base + uint(r.a));
        pos = pa + r.pr * na;
        if (r.mode == 0) nrm = r.ns * na;                                   // smooth (makeSmooth)
        else if (r.mode == 1) {                                             // flat skin face (makeSmooth, flg_flat)
            float3 b = ld3(P, base + uint(r.b));
            nrm = r.ns * unit_or_x(cross(ld3(P, base + uint(r.c)) - b, ld3(P, base + uint(r.d)) - b));
        } else {                                                            // side of a 2D skin (makeSide)
            float3 pb = ld3(P, base + uint(r.b)), pc = ld3(P, base + uint(r.c));
            nrm = r.ns * unit_or_x(cross(pc - pb, ld3(Nn, base + uint(r.c))));
        }
    }
    put(out, c.rbase + uint(r.dst) + e * uint(r.vstride), pos + env_offset(e, c), nrm, float2(r.u, r.v));
}

// ---- tubes: flex edges (open cylinders), tendon segments (capsules), flex vertices (spheres) ------------------------

inline float3x3 rot_z_to(float3 vec) {        // mju_quatZ2Vec -> mju_quat2Mat
    float n = length(vec);
    if (n < 1e-15f) return float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1));
    float3 vn = vec / n;
    float3 axis = float3(-vn.y, vn.x, 0.0);   // cross(z, vn)
    float a = length(axis);
    if (a < 1e-15f) return vn.z < 0 ? float3x3(float3(1, 0, 0), float3(0, -1, 0), float3(0, 0, -1)) : float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1));
    float ang = atan2(a, vn.z);
    axis /= a;
    float s = sin(0.5f * ang);
    return quat_mat(float4(cos(0.5f * ang), axis * s));
}

// ---- catenary of a slack two-site tendon (MuJoCo's mjv_isCatenary / mjv_catenary, engine_vis_visualize.c) -----------
// A spatial tendon between exactly two sites with a deadband spring (stiffness, lengthspring = [0, L]) or, without
// stiffness, a length limit [0, L] it is not at, under nonzero gravity, without damping, frictionloss or an actuator, is
// drawn as the catenary of length L through its two sites (dist > L: straight; L > 100 h: one hanging bead; else
// numslices + 1 points, at most 100). The static conditions are checked on the host (dynamic_mesh.catenary_mode); a
// limit-type tendon at or past L is drawn straight between its wrap points (as MuJoCo's non-catenary branch). One thread
// per (catenary tendon, env) writes its points (CP) and their count (CPn) for emit_tube's mode 3. float32 throughout
// (MuJoCo: float64; same Newton iteration on Pérez's change of variables, same 50-step bound).
struct CatRec { int tendon, mode; float length, _p; };     // mode 0: deadband spring, 1: length limit (slack only)
struct CatConsts { float4 up; uint npts, n_cat, _a, _b; }; // up: -gravity, normalized; npts: points of a full catenary

inline float cosh_sinh(float x, thread float& sh) {
    float ex = exp(clamp(x, -80.0f, 80.0f));
    sh = 0.5f * (ex - 1.0f / ex);
    return 0.5f * (ex + 1.0f / ex);
}
inline float catenary_residual(float b, float intercept, thread float& grad) {
    float a = 0.5f / b;
    float sh, ch = cosh_sinh(a, sh);
    float q = max(2.0f * b * sh - 1.0f, 1e-30f);
    grad = (a * ch - sh) * pow(q, -1.5f);
    return 1.0f / sqrt(q) - intercept;
}
inline float solve_catenary(float v, float h, float L) {
    float intercept = 1.0f / sqrt(sqrt(L * L - v * v) / h - 1.0f);
    float b = intercept / sqrt(24.0f);
    float tol = 1e-6f * max(intercept, 1.0f);       // MuJoCo: 1e-9 absolute in float64; float32's resolution here
    for (int i = 0; i < 50; ++i) {
        float grad;
        float res = catenary_residual(b, intercept, grad);
        if (fabs(res) < tol) break;
        float step = -res / grad;
        if (!isfinite(step)) break;
        for (int j = 0; j < 10; ++j) {
            float g2;
            if (fabs(catenary_residual(b + step, intercept, g2)) < fabs(res)) break;
            step *= 0.5f;
        }
        b += step;
    }
    return b;
}
// points of the catenary of length L from x0 to x1 into out; returns their number (2, 3 or npts)
inline int catenary_points(float3 x0, float3 x1, float3 up, float L, uint npts, device float4* out) {
    float3 x01 = x1 - x0;
    if (length(x01) > L) { out[0] = float4(x0, 1.0f); out[1] = float4(x1, 1.0f); return 2; }
    float3 across = x01 - up * dot(up, x01);
    float nrm = length(across);
    across = nrm < 1e-15f ? float3(0.0f) : across / nrm;
    float h = dot(x01, across), v = dot(x01, up);
    if (L > 100.0f * h) {                  // near vertical: hanging bead
        float d_up = -0.5f * (sqrt(L * L - h * h) - v);
        float d_across = h * d_up / (2.0f * d_up - v);
        out[0] = float4(x0, 1.0f);
        out[1] = float4(x0 + up * d_up + across * d_across, 1.0f);
        out[2] = float4(x1, 1.0f);
        return 3;
    }
    float bh = solve_catenary(v, h, L) * h;
    float h_offset = -0.5f * (log((L + v) / (L - v)) * bh - h);
    float sh;
    float v_offset = -cosh_sinh(h_offset / bh, sh) * bh;
    out[0] = float4(x0, 1.0f);
    for (uint i = 1; i + 1 < npts; ++i) {
        float horizontal = float(i) * h / float(npts);
        float vertical = bh * cosh_sinh((horizontal - h_offset) / bh, sh) + v_offset;
        out[i] = float4(x0 + across * horizontal + up * vertical, 1.0f);
    }
    out[npts - 1] = float4(x1, 1.0f);
    return int(npts);
}

kernel void catenary(device const CatRec* cat [[buffer(0)]], device const int* ten_wrapadr [[buffer(1)]],
                     device const float* wrap_xpos [[buffer(2)]], device const float* ten_length [[buffer(3)]],
                     device float4* CP [[buffer(4)]], device int* CPn [[buffer(5)]],
                     constant DynConsts& c [[buffer(6)]], constant CatConsts& cc [[buffer(7)]],
                     uint2 tid [[thread_position_in_grid]])
{
    uint k = tid.x, e = tid.y;
    if (k >= cc.n_cat || e >= c.n_envs) return;
    CatRec r = cat[k];
    device float4* out = CP + (e * cc.n_cat + k) * cc.npts;
    int adr = ten_wrapadr[e * c.ntendon + uint(r.tendon)];
    if (adr < 0 || adr + 1 >= int(c.nwrap2)) { out[0] = float4(0.0f); CPn[e * cc.n_cat + k] = 0; return; }
    float3 x0 = ld3(wrap_xpos, e * c.nwrap2 + uint(adr)), x1 = ld3(wrap_xpos, e * c.nwrap2 + uint(adr + 1));
    bool slack = r.mode == 0 || ten_length[e * c.ntendon + uint(r.tendon)] < r.length;
    int n;
    if (!slack) { out[0] = float4(x0, 1.0f); out[1] = float4(x1, 1.0f); n = 2; }   // at the limit: straight
    else n = catenary_points(x0, x1, cc.up.xyz, r.length, cc.npts, out);
    CPn[e * cc.n_cat + k] = n;
}

// ---- limit colour of a tendon (addSpatialTendonGeoms): a default-grey tendon without material is mixed toward
// vis.rgba.constraint by the impedance of its active limit constraint (getimpedance of tendon_solimp_lim at the limit
// distance; the upper limit's when both are active, as MuJoCo's efc loop leaves it). Written over the model sync's
// per-env colour of the tendon's slot (t_colors x colour). Catenary-drawn tendons keep their colour.
struct TintRec { int tendon, slot, cat_mode, limited; float lo, hi, margin, _p; float4 si0; float4 si1; float4 rgba; };
struct TintConsts { float4 constraint; uint n_tint, n_slots, ntendon, n_envs; };

inline float impedance(float4 si0, float si4, float pos, float margin) {   // getimpedance (solimp = si0.xyzw, si4)
    if (si0.x == si0.y || si0.z <= 1e-15f) return 0.5f * (si0.x + si0.y);
    float x = (pos - margin) / si0.z;
    if (x < 0.0f) x = -x;
    if (x >= 1.0f || x <= 0.0f) return x >= 1.0f ? si0.y : si0.x;
    float y;
    if (si4 == 1.0f) y = x;
    else if (x <= si0.w) { float a = 1.0f / pow(si0.w, si4 - 1.0f); y = a * pow(x, si4); }
    else { float b = 1.0f / pow(1.0f - si0.w, si4 - 1.0f); y = 1.0f - b * pow(1.0f - x, si4); }
    return si0.x + y * (si0.y - si0.x);
}

kernel void tendon_tint(device const TintRec* tr [[buffer(0)]], device const float* ten_length [[buffer(1)]],
                        device const float4* t_colors [[buffer(2)]], device float4* inst_col [[buffer(3)]],
                        constant TintConsts& tc [[buffer(4)]], uint2 tid [[thread_position_in_grid]])
{
    uint k = tid.x, e = tid.y;
    if (k >= tc.n_tint || e >= tc.n_envs) return;
    TintRec r = tr[k];
    float len = ten_length[e * tc.ntendon + uint(r.tendon)];
    if (r.cat_mode == 1 && len < r.hi) return;            // slack limit-type two-site tendon: drawn as a catenary
    float imp = 0.0f;
    if (r.limited != 0) {
        float dl = len - r.lo, du = r.hi - len;            // lower then upper (mj_instantiateLimit's order)
        if (dl < r.margin) imp = impedance(r.si0, r.si1.x, dl, r.margin);
        if (du < r.margin) imp = impedance(r.si0, r.si1.x, du, r.margin);
    }
    float3 rgb = (1.0f - imp) * r.rgba.rgb + imp * tc.constraint.rgb;
    uint i = e * tc.n_slots + uint(r.slot);
    inst_col[i] = t_colors[i] * float4(rgb, r.rgba.a);
}

kernel void emit_tube(device const TRec* rec [[buffer(0)]], device const float4* tmpl [[buffer(1)]], device const float* P [[buffer(2)]],
                      device const int* ten_wrapadr [[buffer(3)]], device const int* ten_wrapnum [[buffer(4)]],
                      device const int* wrap_obj [[buffer(5)]], device const float* wrap_xpos [[buffer(6)]],
                      device float* out [[buffer(7)]], constant DynConsts& c [[buffer(8)]],
                      device const float4* CP [[buffer(9)]], device const int* CPn [[buffer(10)]],
                      constant CatConsts& cc [[buffer(11)]],
                      uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.x, e = tid.y;
    if (i >= c.n_rec || e >= c.n_envs) return;
    TRec r = rec[i];
    float4 t = tmpl[r.tv];                     // unit direction xyz (the normal), end (-1, 0, +1) along the axis
    float3 p0, p1;
    float rad = r.r;
    bool ok = true;
    if (r.mode == 3) {                         // catenary tendon a (CatRec index), segment b
        uint q = e * cc.n_cat + uint(r.a);
        ok = r.b + 1 < CPn[q];
        if (ok) { p0 = CP[q * cc.npts + uint(r.b)].xyz; p1 = CP[q * cc.npts + uint(r.b) + 1].xyz; }
        else p0 = CP[q * cc.npts].xyz;
    } else if (r.mode == 2) {                        // tendon segment b of tendon a (addSpatialTendonGeoms)
        int adr = ten_wrapadr[e * c.ntendon + uint(r.a)], num = ten_wrapnum[e * c.ntendon + uint(r.a)];
        int j = adr + r.b;
        ok = r.b < num - 1 && adr >= 0 && j + 1 < int(c.nwrap2);
        if (ok) {
            int o0 = wrap_obj[e * c.nwrap2 + uint(j)], o1 = wrap_obj[e * c.nwrap2 + uint(j + 1)];
            ok = o0 != -2 && o1 != -2;
            if (o0 >= 0 && o1 >= 0) rad *= 0.5f;       // inside a wrapping geom: half width
            p0 = ld3(wrap_xpos, e * c.nwrap2 + uint(j)); p1 = ld3(wrap_xpos, e * c.nwrap2 + uint(j + 1));
        }
    } else {
        p0 = ld3(P, e * c.ptot + uint(r.a));
        p1 = r.mode == 0 ? ld3(P, e * c.ptot + uint(r.b)) : p0;
    }
    uint o = c.rbase + uint(r.dst) + e * uint(r.vstride);
    if (!ok) {                                 // unused segment slot: degenerate (draws nothing, never hit)
        float3 q = r.mode == 2 ? ld3(wrap_xpos, e * c.nwrap2 + uint(max(ten_wrapadr[e * c.ntendon + uint(r.a)], 0))) : p0;
        put(out, o, q + env_offset(e, c), float3(0, 0, 1), float2(0.0));
        return;
    }
    float3 d = p1 - p0;
    float3x3 R = rot_z_to(d);
    float h = 0.5f * length(d);
    float3 pos = 0.5f * (p0 + p1) + R * float3(rad * t.x, rad * t.y, rad * t.z + h * t.w);
    put(out, o, pos + env_offset(e, c), R * t.xyz, float2(0.0));
}

// ---- site poses after the geom poses ---------------------------------------------------------------------------------

struct GatherConsts { uint n_envs, ngeom, nsite_model, n_site; };

kernel void gather_poses(device const float* geom_xpos [[buffer(0)]], device const float* geom_xmat [[buffer(1)]],
                         device const float* site_xpos [[buffer(2)]], device const float* site_xmat [[buffer(3)]],
                         device const int* site_ids [[buffer(4)]], device float* out_xpos [[buffer(5)]],
                         device float* out_xmat [[buffer(6)]], constant GatherConsts& g [[buffer(7)]],
                         uint2 tid [[thread_position_in_grid]])
{
    uint k = tid.x, e = tid.y;
    uint rows = g.ngeom + g.n_site;
    if (k >= rows || e >= g.n_envs) return;
    device const float* sp; device const float* sm;
    if (k < g.ngeom) { sp = geom_xpos + (e * g.ngeom + k) * 3; sm = geom_xmat + (e * g.ngeom + k) * 9; }
    else { uint s = uint(site_ids[k - g.ngeom]); sp = site_xpos + (e * g.nsite_model + s) * 3; sm = site_xmat + (e * g.nsite_model + s) * 9; }
    uint o = e * rows + k;
    for (uint q = 0; q < 3; ++q) out_xpos[o * 3 + q] = sp[q];
    for (uint q = 0; q < 9; ++q) out_xmat[o * 9 + q] = sm[q];
}
