// Placement of the envs in the ray tracers' shared world (metalsim/sensors/env_grid.py). Env e sits at a grid offset;
// its rays only meet its own geometry. "legacy": a 2D grid of stride 2 (extent + max range) + 1 from the origin, rays as
// long as asked (the offsets reach kilometres, where float32 spacing is millimetres). "compact": a 3D grid centred on the
// origin with cells the size of the env's content (geometry, cameras, sites, lights at the initial state, plus a
// margin); every ray ends where it leaves its env's cell, so cells can touch and the offsets stay ~100 m at 4096 envs.
#ifndef METALSIM_ENV_GRID_H   // include guard: shaders are inlined from strings (metal_context._inline_includes)
#define METALSIM_ENV_GRID_H

struct EnvGrid {
    uint nx, ny, clip, _p;      // envs along x and y (z: the rest); clip: rays end at the env's cell
    float4 S;                   // cell size per axis
    float4 c;                   // grid centre (in cells) per axis: (n - 1) / 2 compact, 0 legacy
    float4 lo, hi;              // the cell relative to the env's origin (clip); lo.w: slack of the "left the content"
                                // check (env_grid_left)
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
#endif  // METALSIM_ENV_GRID_H
