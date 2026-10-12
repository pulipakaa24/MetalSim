// Camera lens models (metalsim.render.lens): pixel -> camera-frame ray direction (MuJoCo / USD camera frame:
// x right, y up, looking along -z). Pixel coordinates are continuous (pixel centres at +0.5), y down.
//   0 pinhole            (x - cx) / fx, -(y - cy) / fy, -1 from the per-env intrinsics (CamSpec)
//   1 f-theta polynomial  Isaac / RTX "fisheyePolynomial": the pixel is mapped to the nominal sensor resolution
//                         (a.xy = nominal width, height), r = distance from the optical centre (a.zw, nominal pixels),
//                         theta = A + B r + C r^2 + D r^3 + E r^4 + F r^5 (b = A..D, c.xy = E, F), rays with
//                         theta > max_fov / 2 (c.z, full angle, rad) see nothing
//   2 Kannala-Brandt      OpenCV fisheye / RTX "fisheyeKannalaBrandtK3": (x, y) = ((u - cx) / fx, (v - cy) / fy),
//                         theta_d = |(x, y)| = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
//                         (a = fx, fy, cx, cy in render pixels; b = k1..k4), inverted by Newton; max_fov in c.z
#ifndef METALSIM_LENS_H
#define METALSIM_LENS_H
struct LensParams { uint model, _p0, _p1, _p2; float4 a; float4 b; float4 c; };

inline float3 lens_polar_dir(float theta, float dx, float dy, float r) {
    float s = sin(theta);
    return r > 1e-12 ? float3(s * dx / r, -s * dy / r, -cos(theta)) : float3(0.0, 0.0, -1.0);
}

// false: the pixel lies outside the lens's field of view (background, depth 0)
inline bool lens_ray(constant LensParams& L, float px, float py, float W, float H, float4 intr, thread float3& d) {
    if (L.model == 1u) {
        float dx = px * L.a.x / W - L.a.z, dy = py * L.a.y / H - L.a.w;
        float r = sqrt(dx * dx + dy * dy);
        float th = L.b.x + r * (L.b.y + r * (L.b.z + r * (L.b.w + r * (L.c.x + r * L.c.y))));
        if (th < 0.0 || th > 0.5 * L.c.z || th > 3.14159265f) return false;
        d = lens_polar_dir(th, dx, dy, r);
        return true;
    }
    if (L.model == 2u) {
        float x = (px - L.a.z) / L.a.x, y = (py - L.a.w) / L.a.y;
        float td = sqrt(x * x + y * y);
        float th = td;
        for (int i = 0; i < 12; ++i) {           // Newton on f(th) = th (1 + k1 th^2 + ...) - td
            float t2 = th * th;
            float f = th * (1.0 + t2 * (L.b.x + t2 * (L.b.y + t2 * (L.b.z + t2 * L.b.w)))) - td;
            float fp = 1.0 + t2 * (3.0 * L.b.x + t2 * (5.0 * L.b.y + t2 * (7.0 * L.b.z + t2 * 9.0 * L.b.w)));
            th -= f / max(fp, 1e-6);
        }
        if (!(th >= 0.0) || th > 0.5 * L.c.z || th > 3.14159265f) return false;
        d = lens_polar_dir(th, x, y, td);
        return true;
    }
    d = float3((px - intr.z) / intr.x, -(py - intr.w) / intr.y, -1.0);
    return true;
}
#endif
