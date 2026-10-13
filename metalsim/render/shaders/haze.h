// MuJoCo's haze over infinite planes and its linear fog, as mujoco.Renderer draws them (MuJoCo 3.14.0
// src/render/classic/render_context.c haze() / setVertexHaze(), render_gl3.c mjr_render's skybox pass and initGL3's fog;
// metalsim/render/haze.py builds HazeParams and holds the NumPy reference).
//
// Haze: drawn with the skybox (only when a skybox texture is drawn and mjRND_HAZE is on), for the first infinite plane
// (size 0 0) of the scene, if the camera is above it. A truncated cone of numslices flat facets around the camera:
// radius S (1 - r (1 - h)) at height h H above the plane (h = 0 on the plane, 1 at the camera's height H), S = 0.7 x
// the camera's far plane (vis.map.zfar x stat.extent), r = vis.map.haze; colour vis.rgba.haze with alpha 0 at h = 0,
// 1 at the transition height hh and 0 at h = 1, blended over whatever lies behind it (depth-tested against the opaque
// geometry, before the transparent geoms). Per pixel that is: the camera ray's exit through the cone's facet below the
// horizon, if nearer than the opaque surface.
// Fog (mjRND_FOG, off by default): GL linear fog of every lit fragment, the skybox and the haze, on the eye-space depth:
// f = clamp((fogend - z) / (fogend - fogstart)), colour = f c + (1 - f) fog (fogstart / fogend = vis.map x extent).
#ifndef METALSIM_HAZE_H
#define METALSIM_HAZE_H

// HAZE_ON (compile time): 0 when the renderer has neither haze nor fog, so the composite is compiled out of the shaders
// (code in the ray-tracing shaders costs registers whether it runs or not)
#ifndef HAZE_ON
#define HAZE_ON 1
#endif

struct HazeParams {
    float4 org;      // plane origin xyz (world / env frame), S = 0.7 x MuJoCo's far plane
    float4 ax;       // plane x axis, r = vis.map.haze
    float4 ay;       // plane y axis, hh = transition height (fraction of the camera height)
    float4 az;       // plane normal, number of facets (vis.quality.numslices)
    float4 rgba;     // haze colour rgb, w: flags (bit0 haze, bit1 fog)
    float4 fog;      // fogstart, fogend (m)
    float4 fog_rgba; // fog colour
};

constant float HAZE_PI = 3.14159265358979f;
constant float HAZE_NONE = 3.0e38f;     // "no opaque surface" (finite: the shaders compile with fast math)

inline bool haze_on(HazeParams hp) { return (uint(hp.rgba.w) & 1u) != 0u; }
inline bool fog_on(HazeParams hp) { return (uint(hp.rgba.w) & 2u) != 0u; }

// alpha of the haze along the camera ray (C, unit d) whose opaque surface lies at distance t_surf (HAZE_NONE: none);
// t_cone: the distance of the haze surface along the ray (valid when the result is > 0)
inline float haze_alpha(HazeParams hp, float3 C, float3 d, float t_surf, thread float& t_cone) {
    float3 n = hp.az.xyz;
    float H = dot(C - hp.org.xyz, n);
    float dz = dot(d, n);
    if (!(H > 0.0f) || !(dz < 0.0f)) return 0.0f;            // camera below the plane (MuJoCo draws none) or ray not downward
    float dx = dot(d, hp.ax.xyz), dy = dot(d, hp.ay.xyz);
    float ns = max(hp.az.w, 3.0f);
    float az = atan2(dy, dx);
    if (az < 0.0f) az += 2.0f * HAZE_PI;
    float j = floor(az * ns / (2.0f * HAZE_PI));
    float mid = (j + 0.5f) * 2.0f * HAZE_PI / ns;
    float S = hp.org.w, r = hp.ax.w, hh = hp.ay.w;
    float cs = cos(HAZE_PI / ns);
    float den = dx * cos(mid) + dy * sin(mid) - S * cs * r * dz / H;    // exit through facet j's plane
    if (!(den > 0.0f)) return 0.0f;
    float t = S * cs / den;
    float h = 1.0f + t * dz / H;
    if (h < 0.0f || h > 1.0f || !(t < t_surf)) return 0.0f;
    t_cone = t;
    return h <= hh ? h / hh : (1.0f - h) / (1.0f - hh);
}

// GL linear fog at eye-space depth z
inline float3 fog_mix(HazeParams hp, float3 c, float z) {
    float f = clamp((hp.fog.y - z) / max(hp.fog.y - hp.fog.x, 1e-12f), 0.0f, 1.0f);
    return f * c + (1.0f - f) * hp.fog_rgba.rgb;
}

// eye-space depth of MuJoCo's skybox (a closed cylinder of radius S and half height S around the camera, world z axis)
inline float skybox_depth(HazeParams hp, float3 d, float3 fwd) {
    float rxy = length(d.xy);
    float t = hp.org.w / max(max(rxy, fabs(d.z)), 1e-12f);
    return t * dot(d, fwd);
}

// colour of a fragment at distance t_surf (HAZE_NONE: the background) along the camera ray (C, unit d) with eye forward
// fwd: fog, then the haze over it. `sky`: a skybox is drawn behind this pixel (MuJoCo draws the haze only in its skybox
// pass and fogs the skybox; a clear colour is neither fogged nor hazed). `haze_rgb`: the haze colour in output units.
inline float3 haze_composite(HazeParams hp, float3 col, float3 C, float3 d, float t_surf, float3 fwd, bool sky, float3 haze_rgb) {
    bool fg = fog_on(hp);
    bool bg = !(t_surf < HAZE_NONE);
    if (fg && (!bg || sky)) col = fog_mix(hp, col, bg ? skybox_depth(hp, d, fwd) : t_surf * dot(d, fwd));
    if (sky && haze_on(hp)) {
        float tc = 0.0f;
        float a = haze_alpha(hp, C, d, t_surf, tc);
        if (a > 0.0f) {
            float3 hz = fg ? fog_mix(hp, haze_rgb, tc * dot(d, fwd)) : haze_rgb;
            col = a * hz + (1.0f - a) * col;
        }
    }
    return col;
}

#endif  // METALSIM_HAZE_H
