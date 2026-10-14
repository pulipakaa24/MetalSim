// metalsim 3D Gaussian splat rasterizer (compute, tile-based; see metalsim/render/SPLAT.md).
//
// Per sub-batch of up to 32 cameras, all in one serial compute encoder:
//   splat_cull     conservative sphere-vs-frustum cull, per-splat camera bitmask + per-(camera, block) counts
//   scan           -> deterministic record offsets (records are written in splat order per camera)
//   splat_project  EWA projection (3DGS/gsplat math), SH colour, opacity-aware exact box, exact ellipse/tile
//                  intersection count per record
//   scan           -> key offsets
//   splat_emit     32-bit keys ((camera * tiles_per_cam + tile) << 16 | log-depth[23:8]), value = record index
//                  (24 bits) with log-depth[7:0] in its top byte
//   radix sort     5 x 8-bit LSD passes (value top byte first; 4 with depth_bits=16), stable (hist / scan / scatter
//                  with simd-ballot multisplit ranking)
//   splat_blend    one threadgroup per 16x16 tile (its key range by binary search in the sorted keys): front-to-back alpha blending with the 3DGS rules (alpha <= 0.99,
//                  skip < 1/255, stop at T < 1e-4), median + expected depth, optional mesh composite (stop at
//                  the mesh depth, mesh colour behind the remaining transmittance), written straight into the
//                  learner-layout output buffers (N, H, W, 3) uint8 / (N, H, W) float32.
#include <metal_stdlib>
using namespace metal;

#define TILE 16
#define BLOCK 256
#define ALPHA_MIN (1.0f / 255.0f)
#define ALPHA_MAX 0.99f
#define T_MIN 1e-4f

struct SplatCam {                // built on the GPU (splat_cams) or written by the host
    float4 r0, r1, r2;           // world -> camera rows (OpenCV axes: x right, y down, z forward)
    float4 pos;                  // camera position (world), w = near
    float4 intr;                 // fx, fy, cx, cy (pixels; pixel i covers [i, i + 1))
    float4 frust[4];             // inward frustum plane normals in camera space (unit), for the sphere cull
};

struct SplatConsts {
    uint n_splats, n_envs, width, height;
    uint tiles_x, tiles_y, tiles_per_cam, cam0;        // cam0: first env of this sub-batch
    uint n_cams, sh_deg, flags, sh_stride;             // flags: 1 antialiased, 2 sh uint8, 4 composite, 8 expected depth
    uint cap_rec, cap_keys, n_blocks, depth_bits;      // depth_bits 24 (value top byte sorted first) or 16
    float near_, depth_lo, depth_scale, eps2d;         // eps2d: 2D dilation (0.3 px^2)
    float4 a0, a1, a2;           // file -> world linear map rows (covariance), w unused
    float4 d0, d1, d2;           // world -> file direction map rows (SH lookup)
    float4 clear_color;
};

struct Rec {                     // one projected splat for one camera (32 bytes)
    float4 a;                    // mean.x, mean.y, conic A, conic B   (q(d) = A dx^2 + 2 B dx dy + C dy^2)
    float4 b;                    // conic C, view depth z, half2(r, g), half2(b, opacity)
};

inline float2 unpack_h2(float v) { return float2(as_type<half2>(v)); }
inline float pack_h2(float x, float y) { return as_type<float>(half2(half(x), half(y))); }

// tile rectangle of the alpha >= 1/255 support (from the conic: Sigma = inverse); identical in project and emit
inline uint4 support_rect(float2 mean, float A, float B, float C, float t, constant SplatConsts& k, thread bool& ok) {
    float det = A * C - B * B;
    float sxx = C / det, syy = A / det;
    float rx = sqrt(sxx * t), ry = sqrt(syy * t);
    int px0 = max(int(ceil(mean.x - rx - 0.5f)), 0), px1 = min(int(floor(mean.x + rx - 0.5f)), int(k.width) - 1);
    int py0 = max(int(ceil(mean.y - ry - 0.5f)), 0), py1 = min(int(floor(mean.y + ry - 0.5f)), int(k.height) - 1);
    ok = (px0 <= px1) && (py0 <= py1);
    return uint4(uint(max(px0, 0)) / TILE, uint(max(py0, 0)) / TILE, uint(max(px1, 0)) / TILE + 1, uint(max(py1, 0)) / TILE + 1);
}

struct Counters { atomic_uint overflow_rec; atomic_uint overflow_keys; uint n_rec; uint n_keys; };

// ------------------------------------------------------------------------------------------------ cameras

inline float3x3 quat_to_mat_wxyz(float4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return float3x3(
        float3(1 - 2*(y*y + z*z), 2*(x*y + w*z),     2*(x*z - w*y)),
        float3(2*(x*y - w*z),     1 - 2*(x*x + z*z), 2*(y*z + w*x)),
        float3(2*(x*z + w*y),     2*(y*z - w*x),     1 - 2*(x*x + y*y)));
}

// tier0's CameraSpec (28 floats): intrinsics 0:4, pos_delta 4:8, rot_delta 8:12 (wxyz), ..., cam_id at 24 (int)
kernel void splat_cams(
    device SplatCam*      cams     [[buffer(0)]],
    device const float*   spec     [[buffer(1)]],
    device const float*   cam_xpos [[buffer(2)]],   // (n_envs, n_cam, 3)
    device const float*   cam_xmat [[buffer(3)]],   // (n_envs, n_cam, 9) MuJoCo row-major, camera looks along -z, y up
    constant uint4&       cc       [[buffer(4)]],   // n_envs, n_cam, width, height
    constant float4&      cf       [[buffer(5)]],   // near
    uint e [[thread_position_in_grid]])
{
    if (e >= cc.x) return;
    device const float* s = spec + e * 28;
    int ci = max(as_type<int>(s[24]), 0);
    device const float* xp = cam_xpos + (e * cc.y + ci) * 3;
    device const float* xm = cam_xmat + (e * cc.y + ci) * 9;
    float3 pos = float3(xp[0], xp[1], xp[2]) + float3(s[4], s[5], s[6]);
    float3x3 R = float3x3(float3(xm[0], xm[3], xm[6]), float3(xm[1], xm[4], xm[7]), float3(xm[2], xm[5], xm[8]));
    R = R * quat_to_mat_wxyz(float4(s[8], s[9], s[10], s[11]));
    // OpenCV axes in world: x = R[0], y = -R[1], z = -R[2]
    float3 cx = R[0], cy = -R[1], cz = -R[2];
    SplatCam c;
    c.r0 = float4(cx, -dot(cx, pos)); c.r1 = float4(cy, -dot(cy, pos)); c.r2 = float4(cz, -dot(cz, pos));
    c.pos = float4(pos, cf.x);
    float fx = s[0], fy = s[1], px = s[2], py = s[3], W = float(cc.z), H = float(cc.w);
    c.intr = float4(fx, fy, px, py);
    c.frust[0] = float4(normalize(float3(fx, 0, px)), 0);          // u >= 0
    c.frust[1] = float4(normalize(float3(-fx, 0, W - px)), 0);     // u <= W
    c.frust[2] = float4(normalize(float3(0, fy, py)), 0);          // v >= 0
    c.frust[3] = float4(normalize(float3(0, -fy, H - py)), 0);     // v <= H
    cams[e] = c;
}

// ------------------------------------------------------------------------------------------------ cull

kernel void splat_cull(
    device const float4*     g0     [[buffer(0)]],
    device const half4*      g1     [[buffer(1)]],      // 2 x half4 per splat: quat, (scale xyz, r_cull)
    device const SplatCam*   cams   [[buffer(2)]],
    constant SplatConsts&    k      [[buffer(3)]],
    device uint*             mask   [[buffer(4)]],
    device uint*             bcount [[buffer(5)]],      // (n_cams, n_blocks)
    device const float4*     chunks [[buffer(6)]],      // per 256-splat block: bounding sphere (centre, radius incl. extents)
    uint i   [[thread_position_in_grid]],
    uint tl  [[thread_index_in_threadgroup]],
    uint blk [[threadgroup_position_in_grid]],
    uint sg  [[simdgroup_index_in_threadgroup]],
    uint ln  [[thread_index_in_simdgroup]])
{
    threadgroup atomic_uint cnt[32];
    threadgroup atomic_uint cmask;
    if (tl < 32) atomic_store_explicit(&cnt[tl], 0u, memory_order_relaxed);
    if (tl == 0) atomic_store_explicit(&cmask, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tl < k.n_cams) {                               // chunk-level frustum test, one thread per camera
        SplatCam cam = cams[k.cam0 + tl];
        float4 sp = chunks[blk];
        float3 q = float3(dot(cam.r0.xyz, sp.xyz) + cam.r0.w, dot(cam.r1.xyz, sp.xyz) + cam.r1.w, dot(cam.r2.xyz, sp.xyz) + cam.r2.w);
        float rr = 1.5f * sp.w + 4.0f * max(q.z, 0.0f) / min(cam.intr.x, cam.intr.y);
        bool v = q.z + rr > cam.pos.w;
        for (uint f = 0; f < 4; ++f) v = v && (dot(cam.frust[f].xyz, q) > -rr);
        if (v) atomic_fetch_or_explicit(&cmask, 1u << tl, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint chunk_cams = atomic_load_explicit(&cmask, memory_order_relaxed);
    uint m = 0;
    if (i < k.n_splats && chunk_cams) {
        float4 p = g0[i];
        float r = float(g1[2 * i + 1].w);
        for (uint c = 0; c < k.n_cams; ++c) {
            if (!((chunk_cams >> c) & 1u)) continue;
            SplatCam cam = cams[k.cam0 + c];
            float3 q = float3(dot(cam.r0.xyz, p.xyz) + cam.r0.w, dot(cam.r1.xyz, p.xyz) + cam.r1.w, dot(cam.r2.xyz, p.xyz) + cam.r2.w);
            bool v = q.z > cam.pos.w;
            // conservative: sphere of the opacity-aware 3D extent, grown by 1.5x + 4 px for the EWA linearisation and dilation
            float rr = 1.5f * r + 4.0f * q.z / min(cam.intr.x, cam.intr.y);
            for (uint f = 0; f < 4; ++f) v = v && (dot(cam.frust[f].xyz, q) > -rr);
            if (v) m |= (1u << c);
        }
    }
    if (i < k.n_splats) mask[i] = m;
    for (uint c = 0; c < k.n_cams; ++c) {
        uint s = simd_sum((m >> c) & 1u);
        if (ln == 0 && s) atomic_fetch_add_explicit(&cnt[c], s, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tl < k.n_cams) bcount[tl * k.n_blocks + blk] = atomic_load_explicit(&cnt[tl], memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------ project

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f};
constant float SH_C3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                           -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

inline float3 sh_coef(device const uchar* row, uint j, bool u8) {      // j = 0 .. K-1 (higher bands)
    if (u8) {
        device const uchar* p = row + 8 + 3 * j;
        return (float3(p[0], p[1], p[2]) - 128.0f) * (1.0f / 128.0f);
    }
    device const half* p = (device const half*)(row + 8) + 3 * j;
    return float3(p[0], p[1], p[2]);
}

inline float3 eval_sh(device const uchar* row, uint deg, bool u8, float3 d) {
    half4 dc = *((device const half4*)row);
    float3 res = SH_C0 * float3(dc.xyz);
    if (deg > 0) {
        float x = d.x, y = d.y, z = d.z;
        res += -SH_C1 * y * sh_coef(row, 0, u8) + SH_C1 * z * sh_coef(row, 1, u8) - SH_C1 * x * sh_coef(row, 2, u8);
        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
            res += SH_C2[0] * xy * sh_coef(row, 3, u8) + SH_C2[1] * yz * sh_coef(row, 4, u8)
                 + SH_C2[2] * (2.0f * zz - xx - yy) * sh_coef(row, 5, u8) + SH_C2[3] * xz * sh_coef(row, 6, u8)
                 + SH_C2[4] * (xx - yy) * sh_coef(row, 7, u8);
            if (deg > 2) {
                res += SH_C3[0] * y * (3.0f * xx - yy) * sh_coef(row, 8, u8) + SH_C3[1] * xy * z * sh_coef(row, 9, u8)
                     + SH_C3[2] * y * (4.0f * zz - xx - yy) * sh_coef(row, 10, u8)
                     + SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh_coef(row, 11, u8)
                     + SH_C3[4] * x * (4.0f * zz - xx - yy) * sh_coef(row, 12, u8)
                     + SH_C3[5] * z * (xx - yy) * sh_coef(row, 13, u8) + SH_C3[6] * x * (xx - 3.0f * yy) * sh_coef(row, 14, u8);
            }
        }
    }
    return max(res + 0.5f, 0.0f);
}

// exact test: does the ellipse  q(d) = A dx^2 + 2 B dx dy + C dy^2 <= t  contain any point of the rectangle
// [x0, x1] x [y0, y1] (pixel centres of a tile, relative to the mean)? Minimum of the convex quadratic over the box.
inline float edge_min_x(float A, float B, float C, float y, float x0, float x1) {     // min over x in [x0,x1] at fixed y
    float x = clamp(-B * y / A, x0, x1);
    return A * x * x + 2.0f * B * x * y + C * y * y;
}
inline float edge_min_y(float A, float B, float C, float x, float y0, float y1) {
    float y = clamp(-B * x / C, y0, y1);
    return A * x * x + 2.0f * B * x * y + C * y * y;
}
inline bool tile_hit(float A, float B, float C, float t, float x0, float x1, float y0, float y1) {
    if (x0 <= 0.0f && x1 >= 0.0f && y0 <= 0.0f && y1 >= 0.0f) return true;
    float m = min(min(edge_min_x(A, B, C, y0, x0, x1), edge_min_x(A, B, C, y1, x0, x1)),
                  min(edge_min_y(A, B, C, x0, y0, y1), edge_min_y(A, B, C, x1, y0, y1)));
    return m <= t;
}

// thread-local projection result
struct Proj { bool ok; float2 mean; float3 conic; float op; float z; float t; uint4 rect; float3 col; };

inline Proj project_one(float4 p, half4 qh, half4 sh_, device const uchar* shrow, SplatCam cam, constant SplatConsts& k) {
    Proj o; o.ok = false;
    float3 pc = float3(dot(cam.r0.xyz, p.xyz) + cam.r0.w, dot(cam.r1.xyz, p.xyz) + cam.r1.w, dot(cam.r2.xyz, p.xyz) + cam.r2.w);
    if (pc.z <= cam.pos.w) return o;
    // 3D covariance in the world frame: M = A R S, Sigma = M M^T
    float4 q = float4(qh);
    q /= max(length(q), 1e-12f);
    float3x3 Rq = quat_to_mat_wxyz(q);                       // columns
    float3 s = float3(sh_.xyz);
    float3x3 A = transpose(float3x3(k.a0.xyz, k.a1.xyz, k.a2.xyz));   // rows -> columns
    float3x3 M = A * float3x3(Rq[0] * s.x, Rq[1] * s.y, Rq[2] * s.z);
    float3x3 Sig = M * transpose(M);
    // view rotation (rows r0, r1, r2) and EWA Jacobian with the 3DGS 1.3 x fov clamp (gsplat: asymmetric principal point)
    float3x3 Wv = transpose(float3x3(cam.r0.xyz, cam.r1.xyz, cam.r2.xyz));   // columns = rows^T  -> Wv = [r0; r1; r2]
    float fx = cam.intr.x, fy = cam.intr.y, cx = cam.intr.z, cy = cam.intr.w;
    float W = float(k.width), H = float(k.height);
    float limx_p = 1.3f * (W - cx) / fx, limx_n = 1.3f * cx / fx, limy_p = 1.3f * (H - cy) / fy, limy_n = 1.3f * cy / fy;
    float rz = 1.0f / pc.z;
    float tx = pc.z * clamp(pc.x * rz, -limx_n, limx_p), ty = pc.z * clamp(pc.y * rz, -limy_n, limy_p);
    float3x3 J = float3x3(float3(fx * rz, 0, 0), float3(0, fy * rz, 0), float3(-fx * tx * rz * rz, -fy * ty * rz * rz, 0));
    float3x3 T = J * Wv;
    float3x3 cov = T * Sig * transpose(T);
    float a = cov[0][0], b = cov[1][0], c = cov[1][1];
    float det0 = a * c - b * b;
    a += k.eps2d; c += k.eps2d;
    float det = a * c - b * b;
    if (det <= 0.0f) return o;
    float op = p.w;
    if (k.flags & 1u) op *= sqrt(max(det0 / det, 0.0f));
    op = float(half(op));                                    // stored as half: project and emit use the same value
    if (op < ALPHA_MIN) return o;
    float id = 1.0f / det;
    float3 conic = float3(c * id, -b * id, a * id);
    float t = 2.0f * log(255.0f * op);                       // alpha >= 1/255 inside q(d) <= t
    float2 mean = float2(cx + fx * pc.x * rz, cy + fy * pc.y * rz);
    bool inb;
    o.rect = support_rect(mean, conic.x, conic.y, conic.z, t, k, inb);
    if (!inb) return o;
    o.ok = true; o.mean = mean; o.conic = conic; o.op = op; o.z = pc.z; o.t = t;
    if (k.sh_deg > 0) {
        float3 dw = normalize(p.xyz - cam.pos.xyz);
        float3 df = normalize(float3(dot(k.d0.xyz, dw), dot(k.d1.xyz, dw), dot(k.d2.xyz, dw)));
        o.col = eval_sh(shrow, k.sh_deg, (k.flags & 2u) != 0, df);
    } else {
        o.col = eval_sh(shrow, 0, true, float3(0));
    }
    return o;
}

inline uint count_tiles(Proj o, float A, float B, float C) {
    uint n = 0;
    for (uint ty = o.rect.y; ty < o.rect.w; ++ty)
        for (uint tx = o.rect.x; tx < o.rect.z; ++tx) {
            float x0 = float(tx * TILE) + 0.5f - o.mean.x, y0 = float(ty * TILE) + 0.5f - o.mean.y;
            if (tile_hit(A, B, C, o.t, x0, x0 + float(TILE - 1), y0, y0 + float(TILE - 1))) ++n;
        }
    return n;
}

kernel void splat_project(
    device const float4*     g0     [[buffer(0)]],
    device const half4*      g1     [[buffer(1)]],
    device const uchar*      sh     [[buffer(2)]],
    device const SplatCam*   cams   [[buffer(3)]],
    constant SplatConsts&    k      [[buffer(4)]],
    device const uint*       mask   [[buffer(5)]],
    device const uint*       boff   [[buffer(6)]],      // exclusive scan of bcount, (n_cams, n_blocks)
    device Rec*              recs   [[buffer(7)]],
    device uint*             ntile  [[buffer(8)]],
    device Counters*         ctr    [[buffer(9)]],
    uint i   [[thread_position_in_grid]],
    uint tl  [[thread_index_in_threadgroup]],
    uint blk [[threadgroup_position_in_grid]],
    uint sg  [[simdgroup_index_in_threadgroup]],
    uint ln  [[thread_index_in_simdgroup]])
{
    threadgroup uint sgsum[BLOCK / 32];
    threadgroup atomic_uint tgmask;
    if (tl == 0) atomic_store_explicit(&tgmask, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint m = (i < k.n_splats) ? mask[i] : 0u;
    uint sm = simd_or(m);
    if (ln == 0 && sm) atomic_fetch_or_explicit(&tgmask, sm, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint any_cams = atomic_load_explicit(&tgmask, memory_order_relaxed);     // uniform: cameras that see any splat of the block
    float4 p = float4(0); half4 qh = half4(0), sc = half4(0);
    if (m) { p = g0[i]; qh = g1[2 * i]; sc = g1[2 * i + 1]; }
    for (uint c = 0; c < k.n_cams; ++c) {
        if (!((any_cams >> c) & 1u)) continue;
        uint f = (m >> c) & 1u;
        uint pre = simd_prefix_exclusive_sum(f);
        uint tot = simd_sum(f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ln == 0) sgsum[sg] = tot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint base = 0;
        for (uint s2 = 0; s2 < sg; ++s2) base += sgsum[s2];
        if (f) {
            uint r = boff[c * k.n_blocks + blk] + base + pre;
            if (r >= k.cap_rec) { atomic_fetch_add_explicit(&ctr->overflow_rec, 1u, memory_order_relaxed); continue; }
            Proj o = project_one(p, qh, sc, sh + size_t(i) * k.sh_stride, cams[k.cam0 + c], k);
            Rec rec;
            uint n = 0;
            if (o.ok) {
                n = count_tiles(o, o.conic.x, o.conic.y, o.conic.z);    // conic = inverse covariance: A, B, C
                rec.a = float4(o.mean, o.conic.x, o.conic.y);
                rec.b = float4(o.conic.z, o.z, pack_h2(o.col.r, o.col.g), pack_h2(o.col.b, o.op));
            } else {
                rec.a = float4(0); rec.b = float4(0);
            }
            recs[r] = rec;
            ntile[r] = n;
        }
    }
}

// ------------------------------------------------------------------------------------------------ emit keys

inline uint cam_of_record(uint r, constant SplatConsts& k, device const uint* recoff) {
    uint c = 0;
    while (c + 1 < k.n_cams && recoff[(c + 1) * k.n_blocks] <= r) ++c;     // boff: camera c starts at boff[c * n_blocks]
    return c;
}

kernel void splat_emit(
    device const Rec*        recs   [[buffer(0)]],
    device const uint*       koff   [[buffer(1)]],      // exclusive scan of ntile
    device const uint*       ntile  [[buffer(2)]],
    constant SplatConsts&    k      [[buffer(3)]],
    device uint*             keys   [[buffer(4)]],
    device uint*             vals   [[buffer(5)]],
    device Counters*         ctr    [[buffer(6)]],
    device const uint*       recoff [[buffer(7)]],      // boff (record offsets per (camera, block))
    uint r [[thread_position_in_grid]])
{
    if (r >= ctr->n_rec || r >= k.cap_rec) return;
    uint n = ntile[r];
    if (n == 0) return;
    Rec rec = recs[r];
    uint off = koff[r];
    if (off + n > k.cap_keys) { atomic_fetch_add_explicit(&ctr->overflow_keys, 1u, memory_order_relaxed); return; }
    float A = rec.a.z, B = rec.a.w, C = rec.b.x, z = rec.b.y;
    float t = 2.0f * log(255.0f * unpack_h2(rec.b.w).y);
    bool inb;
    uint4 rect = support_rect(rec.a.xy, A, B, C, t, k, inb);
    uint cam = cam_of_record(r, k, recoff);
    uint dq = uint(clamp((log(z) - k.depth_lo) * k.depth_scale, 0.0f, 16777215.0f));     // 24-bit log depth
    uint base = cam * k.tiles_per_cam;
    uint end = off + n;                                  // never write more than counted (guards FP contraction differences)
    for (uint ty = rect.y; ty < rect.w && off < end; ++ty)
        for (uint tx = rect.x; tx < rect.z && off < end; ++tx) {
            float x0 = float(tx * TILE) + 0.5f - rec.a.x, y0 = float(ty * TILE) + 0.5f - rec.a.y;
            if (tile_hit(A, B, C, t, x0, x0 + float(TILE - 1), y0, y0 + float(TILE - 1))) {
                keys[off] = ((base + ty * k.tiles_x + tx) << 16) | (dq >> 8);
                vals[off] = r | ((k.depth_bits == 24 ? (dq & 255u) : 0u) << 24);   // low depth byte sorted by the first LSD pass
                ++off;
            }
        }
    for (; off < end; ++off) { keys[off] = 0xFFFFFFFFu; vals[off] = 0u; }   // sorts last, tile 65535 is never blended
}

// ------------------------------------------------------------------------------------------------ scans (u32, exclusive)
// Reduce-then-scan over n = *n_ptr elements with a fixed grid of SCAN_PARTS threadgroups of 1024 threads.
#define SCAN_T 1024
#define SCAN_PARTS 256

inline uint tg_exclusive_scan(uint v, threadgroup uint* tmp, uint tl, uint sg, uint ln, thread uint& total) {
    uint pre = simd_prefix_exclusive_sum(v);
    uint s = simd_sum(v);
    if (ln == 0) tmp[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint x = (ln < SCAN_T / 32) ? tmp[ln] : 0u;
        uint xp = simd_prefix_exclusive_sum(x);
        uint xs = simd_sum(x);
        if (ln < SCAN_T / 32) tmp[ln] = xp;
        if (ln == 0) tmp[32] = xs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = tmp[sg] + pre;
    total = tmp[32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

kernel void scan_reduce(
    device const uint* in     [[buffer(0)]],
    device const uint* n_ptr  [[buffer(1)]],
    device uint*       parts  [[buffer(2)]],
    constant uint&     n_cap  [[buffer(3)]],
    uint tl [[thread_index_in_threadgroup]], uint p [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]])
{
    threadgroup uint tmp[33];
    uint n = min(*n_ptr, n_cap);
    uint chunk = ((n + SCAN_PARTS - 1) / SCAN_PARTS + SCAN_T - 1) / SCAN_T * SCAN_T;
    uint b = p * chunk, e = min(b + chunk, n);
    uint acc = 0;
    for (uint j = b + tl; j < e; j += SCAN_T) acc += in[j];
    uint s = simd_sum(acc);
    if (ln == 0) tmp[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tl == 0) { uint t = 0; for (uint q = 0; q < SCAN_T / 32; ++q) t += tmp[q]; parts[p] = t; }
}

kernel void scan_top(
    device uint*  parts [[buffer(2)]],
    device uint*  total [[buffer(4)]],
    uint tl [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]])
{
    threadgroup uint tmp[33];
    uint v = (tl < SCAN_PARTS) ? parts[tl] : 0u;
    uint tot;
    uint r = tg_exclusive_scan(v, tmp, tl, sg, ln, tot);
    if (tl < SCAN_PARTS) parts[tl] = r;
    if (tl == 0) *total = tot;
}

kernel void scan_down(
    device const uint* in     [[buffer(0)]],
    device const uint* n_ptr  [[buffer(1)]],
    device const uint* parts  [[buffer(2)]],
    constant uint&     n_cap  [[buffer(3)]],
    device uint*       out    [[buffer(5)]],
    uint tl [[thread_index_in_threadgroup]], uint p [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]])
{
    threadgroup uint tmp[33];
    uint n = min(*n_ptr, n_cap);
    uint chunk = ((n + SCAN_PARTS - 1) / SCAN_PARTS + SCAN_T - 1) / SCAN_T * SCAN_T;
    uint b = p * chunk, e = min(b + chunk, n);
    uint carry = parts[p];
    for (uint j0 = b; j0 < e; j0 += SCAN_T) {
        uint j = j0 + tl;
        uint v = (j < e) ? in[j] : 0u;
        uint tot;
        uint r = tg_exclusive_scan(v, tmp, tl, sg, ln, tot);
        if (j < e) out[j] = carry + r;
        carry += tot;
    }
}

// ------------------------------------------------------------------------------------------------ indirect args

struct DispatchArgs { uint x, y, z; };

kernel void splat_args(
    device Counters*      ctr    [[buffer(0)]],
    device const uint*    totals [[buffer(1)]],     // [0] records (scan total), [1] keys (scan total)
    constant SplatConsts& k      [[buffer(2)]],
    device DispatchArgs*  args   [[buffer(3)]],     // [0] per-record (256), [1] radix blocks (4096 keys), [2] per-key (256)
    device uint*          nblk   [[buffer(4)]],     // radix: n_blocks, 256 * n_blocks
    constant uint&        which  [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    if (which == 0) {
        uint n = min(totals[0], k.cap_rec);
        if (totals[0] > k.cap_rec) atomic_fetch_add_explicit(&ctr->overflow_rec, totals[0] - k.cap_rec, memory_order_relaxed);
        ctr->n_rec = n;
        args[0] = DispatchArgs{(n + 255) / 256, 1, 1};
    } else {
        uint n = min(totals[1], k.cap_keys);
        if (totals[1] > k.cap_keys) atomic_fetch_add_explicit(&ctr->overflow_keys, 1u, memory_order_relaxed);
        ctr->n_keys = n;
        uint nb = (n + 4095) / 4096;
        args[1] = DispatchArgs{nb, 1, 1};
        args[2] = DispatchArgs{(n + 255) / 256, 1, 1};
        nblk[0] = nb; nblk[1] = 256 * nb;
    }
}

// ------------------------------------------------------------------------------------------------ radix sort (LSD, 8-bit digits)
#define RS_ITEMS 16           // 256 threads x 16 = 4096 keys per block

kernel void rs_hist(
    device const uint*  keys  [[buffer(0)]],
    device Counters*    ctr   [[buffer(1)]],
    device const uint*  nblk  [[buffer(2)]],
    device uint*        hist  [[buffer(3)]],     // digit-major (256, n_blocks)
    constant uint2&     shift [[buffer(4)]],     // bit shift, digit source (0 key, 1 value)
    device const uint*  vals  [[buffer(5)]],
    uint tl [[thread_index_in_threadgroup]], uint b [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint h[256];
    atomic_store_explicit(&h[tl], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = ctr->n_keys;
    uint base = b * 4096;
    for (uint j = 0; j < RS_ITEMS; ++j) {
        uint idx = base + j * 256 + tl;
        if (idx < n) atomic_fetch_add_explicit(&h[(((shift.y ? vals[idx] : keys[idx])) >> shift.x) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hist[tl * nblk[0] + b] = atomic_load_explicit(&h[tl], memory_order_relaxed);
}

kernel void rs_scatter(
    device const uint*  keys_in  [[buffer(0)]],
    device const uint*  vals_in  [[buffer(1)]],
    device Counters*    ctr      [[buffer(2)]],
    device const uint*  nblk     [[buffer(3)]],
    device const uint*  hoff     [[buffer(4)]],     // exclusive scan of hist
    device uint*        keys_out [[buffer(5)]],
    device uint*        vals_out [[buffer(6)]],
    constant uint2&     shift    [[buffer(7)]],
    uint tl [[thread_index_in_threadgroup]], uint b [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]])
{
    threadgroup uint sgc[8][256];
    threadgroup uint dbase[256];
    uint n = ctr->n_keys;
    uint nb = nblk[0];
    dbase[tl] = hoff[tl * nb + b];
    ulong lt = (ln == 0) ? 0ul : ((~0ul) >> (64 - ln));
    uint base = b * 4096;
    for (uint j = 0; j < RS_ITEMS; ++j) {
        for (uint q = 0; q < 8; ++q) sgc[q][tl] = 0;
        uint idx = base + j * 256 + tl;
        bool valid = idx < n;
        uint key = valid ? keys_in[idx] : 0u;
        uint val = valid ? vals_in[idx] : 0u;
        uint d = ((shift.y ? val : key) >> shift.x) & 255u;
        ulong peers = ulong(simd_ballot(valid));
        for (uint bit = 0; bit < 8; ++bit) {
            bool on = (d >> bit) & 1u;
            ulong bv = ulong(simd_ballot(on));
            peers &= on ? bv : ~bv;
        }
        uint rank = popcount(peers & lt);
        uint cnt = popcount(peers);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid && rank == 0) sgc[sg][d] = cnt;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // thread tl owns digit tl: offsets of each simdgroup's run
            uint run = dbase[tl];
            for (uint q = 0; q < 8; ++q) { uint c = sgc[q][tl]; sgc[q][tl] = run; run += c; }
            dbase[tl] = run;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) {
            uint pos = sgc[sg][d] + rank;
            keys_out[pos] = key;
            vals_out[pos] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ------------------------------------------------------------------------------------------------ blend

kernel void splat_blend(
    device const Rec*        recs     [[buffer(0)]],
    device const uint*       vals     [[buffer(1)]],
    device const uint*       keys     [[buffer(2)]],     // sorted keys (tile range by binary search)
    constant SplatConsts&    k        [[buffer(3)]],
    device uchar*            out_rgb  [[buffer(4)]],
    device float*            out_dep  [[buffer(5)]],     // median depth
    device float*            out_dexp [[buffer(6)]],     // expected depth (flags & 8)
    device const uchar*      bg_rgb   [[buffer(7)]],     // composite (flags & 4): mesh colour / planar depth (0 = none)
    device const float*      bg_dep   [[buffer(8)]],
    device float*            out_alpha[[buffer(9)]],
    constant uint4&          outs     [[buffer(10)]],    // rgb, depth, dexp, alpha enabled
    device Counters*         ctr      [[buffer(11)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 lt  [[thread_position_in_threadgroup]],
    uint  tl  [[thread_index_in_threadgroup]])
{
    threadgroup float4 s_a[256];
    threadgroup float4 s_b[256];
    threadgroup uint s_m[256];                      // 4x2-pixel subtiles (4 per simdgroup row) touched by the splat's support
    threadgroup ushort s_list[8][256];              // per simdgroup: loaded entries touching its subtile row
    threadgroup uint s_rg[2];
    threadgroup atomic_uint n_done;
    uint tile = tg.x;
    uint c = tile / k.tiles_per_cam, lt_ = tile % k.tiles_per_cam;
    uint tx = lt_ % k.tiles_x, ty = lt_ / k.tiles_x;
    uint px = tx * TILE + lt.x, py = ty * TILE + lt.y;
    bool inside = px < k.width && py < k.height;
    uint env = k.cam0 + c;
    size_t pix = (size_t(env) * k.height + py) * k.width + px;
    float2 pf = float2(px, py) + 0.5f;
    float mesh_z = INFINITY;
    float3 bg = k.clear_color.rgb;
    if ((k.flags & 4u) && inside) {
        float d = bg_dep[pix];
        if (d > 0.0f) {
            mesh_z = d;
            bg = float3(bg_rgb[pix * 3], bg_rgb[pix * 3 + 1], bg_rgb[pix * 3 + 2]) * (1.0f / 255.0f);
        }
    }
    if (tl == 0) atomic_store_explicit(&n_done, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool done = !inside;
    if (done) atomic_fetch_add_explicit(&n_done, 1u, memory_order_relaxed);
    float T = 1.0f;
    float3 C = float3(0);
    float Dw = 0.0f, med = 0.0f;
    bool hit_mesh = false;
    uint n_eval = 0, n_contrib = 0;
    if (tl < 2) {                                   // lower_bound of (tile + tl) << 16 in the sorted keys
        uint target = (tile + tl) << 16, lo = 0, hi = ctr->n_keys;
        while (lo < hi) { uint mid = (lo + hi) >> 1; if (keys[mid] < target) lo = mid + 1; else hi = mid; }
        s_rg[tl] = lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint2 rg = uint2(s_rg[0], s_rg[1]);
    uint my_sub = (lt.y >> 1) * 4 + (lt.x >> 2);    // simdgroup sg = pixel rows 2 sg, 2 sg + 1 = subtile row sg
    uint sgi = tl >> 5, lane = tl & 31u;
    ulong lane_lt = (lane == 0) ? 0ul : ((~0ul) >> (64 - lane));
    float2 tile_org = float2(tx * TILE, ty * TILE) + 0.5f;
    for (uint s0 = rg.x; s0 < rg.y; s0 += 256) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&n_done, memory_order_relaxed) >= 256) break;
        uint j = s0 + tl;
        if (j < rg.y) {
            Rec r = recs[vals[j] & 0xFFFFFFu];
            s_a[tl] = r.a; s_b[tl] = r.b;
            float t = 2.0f * log(255.0f * unpack_h2(r.b.w).y);
            uint m = 0;
#ifdef EXP_NO_MASK
            m = 0xFFFFFFFFu;
#else
            for (uint sb = 0; sb < 32; ++sb) {
                float x0 = tile_org.x + float((sb & 3u) * 4u) - r.a.x, y0 = tile_org.y + float((sb >> 2) * 2u) - r.a.y;
                if (tile_hit(r.a.z, r.a.w, r.b.x, t, x0, x0 + 3.0f, y0, y0 + 1.0f)) m |= (1u << sb);
            }
#endif
            s_m[tl] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool was_done = done;
        if (!simd_all(done)) {
            uint cnt = min(256u, rg.y - s0);
            uint nl = 0;                                     // compact the entries touching this simdgroup's 4 subtiles
            for (uint c0 = 0; c0 < cnt; c0 += 32) {
                uint qi = c0 + lane;
                bool h = qi < cnt && ((s_m[qi] >> (4u * sgi)) & 15u);
                ulong bal = ulong(simd_ballot(h));
                if (h) s_list[sgi][nl + popcount(bal & lane_lt)] = ushort(qi);
                nl += popcount(bal);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
#ifdef EXP_NO_EVAL
            nl = 0;
#endif
            for (uint li = 0; li < nl && !done; ++li) {
                uint q = s_list[sgi][li];
                if (!((s_m[q] >> my_sub) & 1u)) continue;
                ++n_eval;
                float4 a = s_a[q], b = s_b[q];
                float2 d = a.xy - pf;
                float power = -0.5f * (a.z * d.x * d.x + b.x * d.y * d.y) - a.w * d.x * d.y;
                if (power > 0.0f) continue;
                float2 bo = unpack_h2(b.w);
                float alpha = min(ALPHA_MAX, bo.y * exp(power));
                if (alpha < ALPHA_MIN) continue;
                float z = b.y;
                if (z > mesh_z) { done = true; hit_mesh = true; break; }
                float tT = T * (1.0f - alpha);
                if (tT < T_MIN) { done = true; break; }
                float w = alpha * T;
                ++n_contrib;
                C += float3(unpack_h2(b.z), bo.x) * w;
                Dw += z * w;
                if (med == 0.0f && tT < 0.5f) med = z;
                T = tT;
            }
            if (done && !was_done) atomic_fetch_add_explicit(&n_done, 1u, memory_order_relaxed);
        }
    }
    if (!inside) return;
    bool mesh = isfinite(mesh_z);
    C += T * bg;
    if (med == 0.0f && mesh && T >= 0.5f) med = mesh_z;           // the opaque mesh is where transmittance crosses 0.5
    float acc = 1.0f - T;
    if (outs.x) {
        out_rgb[pix * 3 + 0] = uchar(clamp(C.r, 0.0f, 1.0f) * 255.0f + 0.5f);
        out_rgb[pix * 3 + 1] = uchar(clamp(C.g, 0.0f, 1.0f) * 255.0f + 0.5f);
        out_rgb[pix * 3 + 2] = uchar(clamp(C.b, 0.0f, 1.0f) * 255.0f + 0.5f);
    }
    if (outs.y) out_dep[pix] = med;
    if (outs.z) {
        float wsum = acc + (mesh ? T : 0.0f);
        float dsum = Dw + (mesh ? T * mesh_z : 0.0f);
        out_dexp[pix] = (wsum >= 0.5f) ? dsum / wsum : 0.0f;
    }
    if (outs.w == 1) out_alpha[pix] = mesh ? 1.0f : acc;
    if (outs.w == 2) { out_alpha[pix] = float(n_eval); out_dexp[pix] = float(n_contrib); }   // debug: work counters
    (void)hit_mesh;
}

// ------------------------------------------------------------------------------------------------ per-frame reset
kernel void splat_reset(device uint* ctr [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < n) ctr[i] = 0u;
}

// ------------------------------------------------------------------------------------------------ blend, 2 pixels per thread
// Same result as splat_blend (tested bit-identical); 128 threads per 16x16 tile, thread = pixels (x, 2 y) and (x, 2 y + 1),
// which lie in the same 4x2 subtile: one mask test, one threadgroup load and one loop iteration serve both pixels.

struct PixState { float T, Dw, med, mesh_z; float3 C, bg; bool done, inside; };

inline void px_setup(thread PixState& s, uint px, uint py, size_t pix, constant SplatConsts& k,
                     device const uchar* bg_rgb, device const float* bg_dep) {
    s.inside = px < k.width && py < k.height;
    s.T = 1.0f; s.Dw = 0.0f; s.med = 0.0f; s.C = float3(0); s.done = !s.inside;
    s.mesh_z = INFINITY; s.bg = k.clear_color.rgb;
    if ((k.flags & 4u) && s.inside) {
        float d = bg_dep[pix];
        if (d > 0.0f) { s.mesh_z = d; s.bg = float3(bg_rgb[pix * 3], bg_rgb[pix * 3 + 1], bg_rgb[pix * 3 + 2]) * (1.0f / 255.0f); }
    }
}

inline void px_step(thread PixState& s, float4 a, float4 b, float2 pf) {
    float2 d = a.xy - pf;
    float power = -0.5f * (a.z * d.x * d.x + b.x * d.y * d.y) - a.w * d.x * d.y;
    if (power > 0.0f) return;
    float2 bo = unpack_h2(b.w);
    float alpha = min(ALPHA_MAX, bo.y * exp(power));
    if (alpha < ALPHA_MIN) return;
    float z = b.y;
    if (z > s.mesh_z) { s.done = true; return; }
    float tT = s.T * (1.0f - alpha);
    if (tT < T_MIN) { s.done = true; return; }
    float w = alpha * s.T;
    s.C += float3(unpack_h2(b.z), bo.x) * w;
    s.Dw += z * w;
    if (s.med == 0.0f && tT < 0.5f) s.med = z;
    s.T = tT;
}

inline void px_write(thread PixState& s, size_t pix, device uchar* out_rgb, device float* out_dep, device float* out_dexp,
                     device float* out_alpha, constant uint4& outs) {
    if (!s.inside) return;
    bool mesh = isfinite(s.mesh_z);
    float3 C = s.C + s.T * s.bg;
    float med = s.med;
    if (med == 0.0f && mesh && s.T >= 0.5f) med = s.mesh_z;
    float acc = 1.0f - s.T;
    if (outs.x) {
        out_rgb[pix * 3 + 0] = uchar(clamp(C.r, 0.0f, 1.0f) * 255.0f + 0.5f);
        out_rgb[pix * 3 + 1] = uchar(clamp(C.g, 0.0f, 1.0f) * 255.0f + 0.5f);
        out_rgb[pix * 3 + 2] = uchar(clamp(C.b, 0.0f, 1.0f) * 255.0f + 0.5f);
    }
    if (outs.y) out_dep[pix] = med;
    if (outs.z) {
        float wsum = acc + (mesh ? s.T : 0.0f), dsum = s.Dw + (mesh ? s.T * s.mesh_z : 0.0f);
        out_dexp[pix] = (wsum >= 0.5f) ? dsum / wsum : 0.0f;
    }
    if (outs.w == 1) out_alpha[pix] = mesh ? 1.0f : acc;
}

kernel void splat_blend2(
    device const Rec*        recs     [[buffer(0)]],
    device const uint*       vals     [[buffer(1)]],
    device const uint*       keys     [[buffer(2)]],
    constant SplatConsts&    k        [[buffer(3)]],
    device uchar*            out_rgb  [[buffer(4)]],
    device float*            out_dep  [[buffer(5)]],
    device float*            out_dexp [[buffer(6)]],
    device const uchar*      bg_rgb   [[buffer(7)]],
    device const float*      bg_dep   [[buffer(8)]],
    device float*            out_alpha[[buffer(9)]],
    constant uint4&          outs     [[buffer(10)]],
    device Counters*         ctr      [[buffer(11)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 lt  [[thread_position_in_threadgroup]],     // (16, 8)
    uint  tl  [[thread_index_in_threadgroup]])
{
    threadgroup float4 s_a[256];
    threadgroup float4 s_b[256];
    threadgroup uint s_m[256];
    threadgroup ushort s_list[4][256];
    threadgroup uint s_rg[2];
    threadgroup atomic_uint n_done;
    uint tile = tg.x;
    uint c = tile / k.tiles_per_cam, lt_ = tile % k.tiles_per_cam;
    uint tx = lt_ % k.tiles_x, ty = lt_ / k.tiles_x;
    uint px = tx * TILE + lt.x, py0 = ty * TILE + 2 * lt.y;
    uint env = k.cam0 + c;
    size_t pix0 = (size_t(env) * k.height + py0) * k.width + px, pix1 = pix0 + k.width;
    float2 pf0 = float2(px, py0) + 0.5f, pf1 = pf0 + float2(0, 1);
    PixState s0, s1;
    px_setup(s0, px, py0, pix0, k, bg_rgb, bg_dep);
    px_setup(s1, px, py0 + 1, pix1, k, bg_rgb, bg_dep);
    if (tl == 0) atomic_store_explicit(&n_done, 0u, memory_order_relaxed);
    if (tl < 2) {
        uint target = (tile + tl) << 16, lo = 0, hi = ctr->n_keys;
        while (lo < hi) { uint mid = (lo + hi) >> 1; if (keys[mid] < target) lo = mid + 1; else hi = mid; }
        s_rg[tl] = lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nd0 = uint(s0.done) + uint(s1.done);
    if (nd0) atomic_fetch_add_explicit(&n_done, nd0, memory_order_relaxed);
    uint2 rg = uint2(s_rg[0], s_rg[1]);
    uint my_sub = lt.y * 4 + (lt.x >> 2);
    uint sgi = tl >> 5, lane = tl & 31u;
    ulong lane_lt = (lane == 0) ? 0ul : ((~0ul) >> (64 - lane));
    float2 tile_org = float2(tx * TILE, ty * TILE) + 0.5f;
    for (uint b0 = rg.x; b0 < rg.y; b0 += 256) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&n_done, memory_order_relaxed) >= 256) break;
        for (uint h = 0; h < 2; ++h) {
            uint li = tl + 128 * h, j = b0 + li;
            if (j < rg.y) {
                Rec r = recs[vals[j] & 0xFFFFFFu];
                s_a[li] = r.a; s_b[li] = r.b;
                float t = 2.0f * log(255.0f * unpack_h2(r.b.w).y);
                uint m = 0;
                for (uint sb = 0; sb < 32; ++sb) {
                    float x0 = tile_org.x + float((sb & 3u) * 4u) - r.a.x, y0 = tile_org.y + float((sb >> 2) * 2u) - r.a.y;
                    if (tile_hit(r.a.z, r.a.w, r.b.x, t, x0, x0 + 3.0f, y0, y0 + 1.0f)) m |= (1u << sb);
                }
                s_m[li] = m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint was = uint(s0.done) + uint(s1.done);
        if (!simd_all(s0.done && s1.done)) {
            uint cnt = min(256u, rg.y - b0);
            uint nl = 0;                                  // entries touching this simdgroup's 8 subtiles (rows 2 sgi, 2 sgi + 1)
            for (uint c0 = 0; c0 < cnt; c0 += 32) {
                uint qi = c0 + lane;
                bool hh = qi < cnt && ((s_m[qi] >> (8u * sgi)) & 255u);
                ulong bal = ulong(simd_ballot(hh));
                if (hh) s_list[sgi][nl + popcount(bal & lane_lt)] = ushort(qi);
                nl += popcount(bal);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint li = 0; li < nl && !(s0.done && s1.done); ++li) {
                uint q = s_list[sgi][li];
                if (!((s_m[q] >> my_sub) & 1u)) continue;
                float4 a = s_a[q], b = s_b[q];
                if (!s0.done) px_step(s0, a, b, pf0);
                if (!s1.done) px_step(s1, a, b, pf1);
            }
            uint now = uint(s0.done) + uint(s1.done);
            if (now > was) atomic_fetch_add_explicit(&n_done, now - was, memory_order_relaxed);
        }
    }
    px_write(s0, pix0, out_rgb, out_dep, out_dexp, out_alpha, outs);
    px_write(s1, pix1, out_rgb, out_dep, out_dexp, out_alpha, outs);
}
