// Placement of the envs in the ray tracers' shared world (metalsim/sensors/env_grid.py). Env e sits at a grid offset;
// its rays only meet its own geometry. "legacy": a 2D grid of stride 2 (extent + max range) + 1 from the origin, rays as
// long as asked (the offsets reach kilometres, where float32 spacing is millimetres). "compact": a 3D grid centred on the
// origin with cells the size of the env's content (geometry, cameras, sites, lights at the initial state, plus a
// margin); every ray ends where it leaves its env's cell, so cells can touch and the offsets stay ~100 m at 4096 envs.
#ifndef METALSIM_ENV_GRID_H   // include guard: shaders are inlined from strings (metal_context._inline_includes)
#define METALSIM_ENV_GRID_H

#define ENV_GRID_PLANES 4
struct EnvGrid {
    uint nx, ny, clip, n_planes;   // envs along x and y (z: the rest); clip: rays end at the env's cell; infinite planes
    float4 S;                   // cell size per axis
    float4 c;                   // grid centre (in cells) per axis: (n - 1) / 2 compact, 0 legacy
    float4 lo, hi;              // the cell relative to the env's origin (clip); lo.w: slack of the "left the content"
                                // check (env_grid_left)
    // compact grid: MuJoCo's infinite planes (size 0) are a disc covering the cell in the acceleration structure and
    // continue analytically past it (grid_plane): slot, then per plane (env frame, initial state) the normal and
    // n . origin, the origin and the finite half size x (0: infinite), the local x axis and half size y, the local y
    // axis, MuJoCo's plane texture mapping (u = x us + ua, v = y vs + va: scene_tables plane_uv_affine)
    uint4 pl_slot;
    float4 pl_n[ENV_GRID_PLANES], pl_o[ENV_GRID_PLANES], pl_x[ENV_GRID_PLANES], pl_y[ENV_GRID_PLANES];
    float4 pl_uv[ENV_GRID_PLANES];
};

inline float3 grid_offset(uint e, EnvGrid g) {
    uint ix = e % g.nx, iy = (e / g.nx) % g.ny, iz = e / (g.nx * g.ny);
    return (float3(float(ix), float(iy), float(iz)) - g.c.xyz) * g.S.xyz;
}

// tmax of a ray of env e from world point o along d: unchanged without clip, else at most the exit of e's cell
// a point (env frame) closer than lo.w to its cell's boundary: the cell is about to be too small (EnvGrid.grow)
inline bool env_grid_left(float3 p, EnvGrid g) {
    return g.clip != 0u && (any(p < g.lo.xyz + g.lo.w) || any(p > g.hi.xyz - g.lo.w));
}

inline float grid_tmax(float3 o, float3 d, float tmax, uint e, EnvGrid g) {
    if (g.clip == 0u) return tmax;
    float3 off = grid_offset(e, g);
    float3 b = select(off + g.lo.xyz, off + g.hi.xyz, d > 0.0);
    bool3 flat = fabs(d) < 1e-30;                     // an axis the ray does not move along never ends it
    float3 t = select((b - o) / select(d, float3(1.0), flat), float3(3.0e38), flat);
    return min(tmax, max(min(min(t.x, t.y), t.z), 0.0));
}

struct GridPlaneHit { bool hit; float t; uint slot; float3 n; float2 uv; float3 lp; };

// the nearest infinite plane of env e met by the ray (o, d) in (t_from, t_to]: past the env's cell (t_from = where it
// leaves the cell; nearer, the plane's disc in the acceleration structure is hit instead). Two-sided, as the disc.
inline GridPlaneHit grid_plane(float3 o, float3 d, float t_from, float t_to, uint e, EnvGrid g) {
    GridPlaneHit h; h.hit = false; h.t = t_to; h.slot = 0u; h.n = float3(0.0, 0.0, 1.0); h.uv = float2(0.0); h.lp = float3(0.0);
    if (g.clip == 0u || g.n_planes == 0u || t_to <= t_from) return h;
    float3 ol = o - grid_offset(e, g);
    for (uint k = 0; k < min(g.n_planes, uint(ENV_GRID_PLANES)); ++k) {
        float3 N = g.pl_n[k].xyz;
        float den = dot(d, N);
        if (fabs(den) < 1e-12) continue;
        float t = (g.pl_n[k].w - dot(ol, N)) / den;
        if (!(t > t_from && t <= h.t)) continue;
        float3 q = ol + d * t - g.pl_o[k].xyz;
        float lx = dot(q, g.pl_x[k].xyz), ly = dot(q, g.pl_y[k].xyz);
        if ((g.pl_o[k].w > 0.0 && fabs(lx) > g.pl_o[k].w) || (g.pl_x[k].w > 0.0 && fabs(ly) > g.pl_x[k].w)) continue;
        h.hit = true; h.t = t; h.slot = g.pl_slot[k]; h.n = den > 0.0 ? -N : N;
        h.uv = float2(lx * g.pl_uv[k].x + g.pl_uv[k].y, ly * g.pl_uv[k].z + g.pl_uv[k].w);
        h.lp = float3(lx, ly, 0.0);
    }
    return h;
}
#endif  // METALSIM_ENV_GRID_H
