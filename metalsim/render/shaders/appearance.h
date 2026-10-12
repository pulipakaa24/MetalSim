// Shared texture mapping for every renderer (tier 0/1 raster, tier 2 path tracer): MuJoCo's conventions
// (src/render/classic/render_gl3.c settexture, render_context.c mjr_uploadTexture, MuJoCo 3.14.0).
//
// * 2D: atlas rows are stored as in tex_data, t = 0 is the first row (MuJoCo / OpenGL upload order); coordinates are
//   scaled by Material.par.xy (texrepeat, texuniform) and offset by MaterialExt.e0.xy (MuJoCo's object-linear mapping).
// * cube: OpenGL cube-map lookup (OpenGL 4.6 table 8.19) of MuJoCo's object coordinates (geom-frame position times
//   MaterialExt.e1.xyz); the texture is one square face used for all six, or a vertical strip +X, -X, +Y, -Y, +Z, -Z.
// * skybox: the first skybox texture as a cube map of the view direction d at (d.x, d.z, -d.y) ("rotate 90 deg
//   around X"), unlit.
// * sRGB textures (tex_colorspace) are decoded to linear on sampling, as MuJoCo's GL_SRGB8 upload does (2D only;
//   MuJoCo uploads cube maps as plain RGB).
#ifndef METALSIM_APPEARANCE_H
#define METALSIM_APPEARANCE_H

struct MaterialExt { float4 e0; float4 e1; };   // e0: uv offset xy, kind (0 2D, 1 cube), srgb | e1: cube scale xyz, faces

inline float3 srgb_to_linear(float3 c) {
    return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

// OpenGL cube-map face selection: (s, t) in [0, 1] on face 0..5 (+X, -X, +Y, -Y, +Z, -Z)
inline float2 cube_face_st(float3 r, thread uint& face) {
    float3 a = fabs(r);
    float sc, tc, ma;
    if (a.x >= a.y && a.x >= a.z) {
        ma = a.x;
        if (r.x >= 0.0) { face = 0; sc = -r.z; tc = -r.y; } else { face = 1; sc = r.z; tc = -r.y; }
    } else if (a.y >= a.z) {
        ma = a.y;
        if (r.y >= 0.0) { face = 2; sc = r.x; tc = r.z; } else { face = 3; sc = r.x; tc = -r.z; }
    } else {
        ma = a.z;
        if (r.z >= 0.0) { face = 4; sc = r.x; tc = -r.y; } else { face = 5; sc = -r.x; tc = -r.y; }
    }
    ma = max(ma, 1e-20);
    return float2(0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0));
}

// cube texture `rect` in the atlas (one face, or a 6-face vertical strip) at direction `dir`
inline float4 atlas_cube(texture2d<float> atlas, sampler s, float4 rect, float faces, float3 dir) {
    uint f;
    float2 st = cube_face_st(dir, f);
    float fw = max(rect.z * float(atlas.get_width()), 1.0);   // face width in texels: clamp to the face (no bleed)
    st = clamp(st, 0.5 / fw, 1.0 - 0.5 / fw);
    float row = faces > 1.5 ? (float(f) + st.y) / 6.0 : st.y;
    return atlas.sample(s, rect.xy + float2(st.x, row) * rect.zw, level(0.0));
}

// texture colour (rgb, alpha) of a material at stored coordinates `uv` and geom-frame position `local`; (1, 1) when
// the material is untextured. rect / par: Material.rect / Material.par.
inline float4 material_tex(float4 rect, float4 par, MaterialExt x, float2 uv, float3 local,
                           texture2d<float> atlas, sampler s) {
    if (par.z <= 0.5) return float4(1.0);
    if (x.e0.z > 0.5) return atlas_cube(atlas, s, rect, x.e1.w, local * x.e1.xyz);
    float4 c = atlas.sample(s, rect.xy + fract(uv * par.xy + x.e0.xy) * rect.zw, level(0.0));
    if (x.e0.w > 0.5) c.rgb = srgb_to_linear(c.rgb);
    return c;
}

// MuJoCo skybox colour in world direction d; sky_rect / sky_info: SceneParams rows 5 / 6
inline float3 sky_color(texture2d<float> atlas, sampler s, float4 sky_rect, float4 sky_info, float3 d) {
    return atlas_cube(atlas, s, sky_rect, sky_info.x, float3(d.x, d.z, -d.y)).rgb;
}

// inverse of a general 4x4 matrix (cofactor expansion); used once per background fragment
inline float4x4 inverse4(float4x4 m) {
    float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3];
    float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3];
    float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3];
    float a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3];
    float b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
    float b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
    float b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
    float b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    float det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    float id = 1.0 / det;
    return float4x4(
        float4(a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09, a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03) * id,
        float4(a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07, a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01) * id,
        float4(a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06, a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00) * id,
        float4(a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06, a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00) * id);
}

#endif
