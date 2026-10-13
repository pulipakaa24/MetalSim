// USD area lights (metalsim/render/area_lights.py): sampling for next-event estimation (tier 2) and soft shadows with
// an analytic irradiance (tier 1). The includer defines Light (20 floats, scene_tables.build_lights) and PI.
//
// Record: specular.w shape (1 rect, 2 disk, 3 cylinder, 4 sphere), ambient.w width / radius, diffuse.w height /
// length, diffuse.xyz radiance Le, kind.z the shaping cone (deg, rect / disk). C: the emitter's centre, ax: its
// emission axis (the prim's -Z, the light's direction), X: the prim's X axis (world).
#ifndef METALSIM_AREA_LIGHT_H   // include guard: shaders are inlined from strings (metal_context._inline_includes)
#define METALSIM_AREA_LIGHT_H

struct AreaSample { float3 L; float tmax; float g; };

inline void al_frame(float3 ax, float3 X, thread float3& ax_o, thread float3& X_o, thread float3& Y_o) {
    ax = normalize(ax);
    X = X - dot(X, ax) * ax;
    X = dot(X, X) > 1e-12 ? normalize(X) : normalize(cross(fabs(ax.z) < 0.9 ? float3(0, 0, 1) : float3(1, 0, 0), ax));
    ax_o = ax; X_o = X; Y_o = cross(ax, X);
}

// One sample toward the emitter from p (normal N). Rect / disk / cylinder: a uniform point on the emitting surface
// (rect / disk emit along ax; the cylinder from the half of its side facing p, no caps), area measure:
// g = cos(theta_r) cos(theta_e) A / d^2. Sphere: a uniform direction in the cone it subtends: g = cos(theta_r) Omega.
// The contribution is f Le g; g = 0: nothing (behind the emitter, below the horizon, outside the shaping cone).
inline AreaSample area_light_sample(Light l, float3 C, float3 ax, float3 X, float3 p, float3 N, float u1, float u2) {
    AreaSample s; s.L = float3(0.0, 0.0, 1.0); s.tmax = 0.0; s.g = 0.0;
    uint shape = uint(l.specular.w + 0.5);
    float a = l.ambient.w, b = l.diffuse.w;
    float3 Y; al_frame(ax, X, ax, X, Y);
    if (shape == 4u) {
        float3 dc = C - p; float dd = length(dc);
        if (dd <= a * 1.0001) return s;
        float st2 = a * a / (dd * dd), ct = sqrt(max(1.0 - st2, 0.0));
        float3 w = dc / dd;
        float c1 = 1.0 - u1 * (1.0 - ct), s1 = sqrt(max(1.0 - c1 * c1, 0.0)), ph = 2.0 * PI * u2;
        float3 aa = fabs(w.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
        float3 t = normalize(cross(aa, w)), bb = cross(w, t);
        float3 L = normalize(t * (s1 * cos(ph)) + bb * (s1 * sin(ph)) + w * c1);
        float NdotL = dot(N, L);
        if (NdotL <= 0.0) return s;
        float bq = dot(L, dc), disc = bq * bq - (dd * dd - a * a);
        s.L = L; s.tmax = max(bq - sqrt(max(disc, 0.0)) - 0.001, 0.0005);
        s.g = NdotL * 2.0 * PI * (1.0 - ct);
        return s;
    }
    float3 S, ne; float A;
    if (shape == 1u) { S = C + X * ((u1 - 0.5) * a) + Y * ((u2 - 0.5) * b); ne = ax; A = a * b; }
    else if (shape == 2u) { float r = a * sqrt(u1), ph = 2.0 * PI * u2; S = C + X * (r * cos(ph)) + Y * (r * sin(ph)); ne = ax; A = PI * a * a; }
    else {      // the half of the side facing p (every point p sees lies in it): pdf 1 / (pi a b)
        float3 q = p - C;
        float ph = atan2(dot(q, ax), dot(q, Y)) + (u2 - 0.5) * PI;
        ne = Y * cos(ph) + ax * sin(ph); S = C + X * ((u1 - 0.5) * b) + ne * a; A = PI * a * b;
    }
    float3 dv = S - p; float d2 = dot(dv, dv), d = sqrt(d2);
    if (d < 1e-6) return s;
    float3 L = dv / d;
    float ce = -dot(ne, L), NdotL = dot(N, L);
    if (ce <= 0.0 || NdotL <= 0.0) return s;
    if (shape <= 2u && l.kind.z < 89.99 && ce < cos(l.kind.z * PI / 180.0)) return s;   // UsdLux shaping cone
    s.L = L; s.tmax = d - 0.001; s.g = NdotL * ce * A / d2;
    return s;
}

// Unoccluded irradiance factor E / Le at p (normal N), exact when the whole emitter is above p's horizon and in front
// of its emitting side: Lambert's polygon formula (rect; disk as an area-matched 32-gon, error < 0.2 %), the sphere's
// pi (R / d)^2 cos. Returns -1 when no closed form applies (cylinder, emitter crossing the horizon, shaping cone).
inline float area_light_irradiance(Light l, float3 C, float3 ax, float3 X, float3 p, float3 N) {
    uint shape = uint(l.specular.w + 0.5);
    float a = l.ambient.w, b = l.diffuse.w;
    float3 Y; al_frame(ax, X, ax, X, Y);
    if (shape == 4u) {
        float3 dc = C - p; float dd = length(dc);
        if (dd <= a || dot(N, dc) < a) return -1.0;            // inside, or crossing the horizon
        return PI * (a / dd) * (a / dd) * dot(N, dc) / dd;
    }
    if (shape == 3u || (l.kind.z < 89.99)) return -1.0;
    if (dot(p - C, ax) <= 0.0) return 0.0;                    // behind the emitting side
    const uint K = shape == 1u ? 4u : 32u;
    float rr = shape == 1u ? 0.0 : a * sqrt(2.0 * PI / (32.0 * sin(2.0 * PI / 32.0)));
    float sum = 0.0;
    float3 v0 = 0.0, vp = 0.0;
    for (uint k = 0; k <= K; ++k) {
        uint kk = k % K;
        float3 S;
        if (shape == 1u) {
            float sx = (kk == 1u || kk == 2u) ? 0.5 : -0.5, sy = (kk >= 2u) ? 0.5 : -0.5;
            S = C + X * (sx * a) + Y * (sy * b);
        } else {
            float t = 2.0 * PI * float(kk) / float(K);
            S = C + X * (rr * cos(t)) + Y * (rr * sin(t));
        }
        float3 v = S - p;
        if (dot(v, N) <= 0.0) return -1.0;                    // a vertex below the horizon: no closed form here
        v = normalize(v);
        if (k > 0) {
            float3 cr = cross(vp, v); float nc = length(cr);
            if (nc > 1e-12) sum += atan2(nc, dot(vp, v)) * dot(cr / nc, N);
        } else v0 = v;
        vp = v;
    }
    return 0.5 * fabs(sum);
}
#endif  // METALSIM_AREA_LIGHT_H
