"""Slow, independent float64 reference for the splat rasterizer: the original 3DGS forward math (Kerbl et al. 2023,
diff-gaussian-rasterization forward.cu / auxiliary.h; gsplat's opacity-aware support), written per pixel with no
tiles, no depth quantisation and an exact float64 depth sort.

For every pixel: all splats with alpha = min(0.99, o * exp(-0.5 d^T Sigma2D^-1 d)) >= 1/255, front to back by view
depth; stop before the splat that would take T below 1e-4; C = sum c_i a_i T_i + T * background. Median depth =
depth of the splat at which T first drops below 0.5; expected depth = sum z_i a_i T_i / (1 - T) when 1 - T >= 0.5.
Pixel i covers [i, i + 1) (centre i + 0.5), as tier0. Pixels are processed in row bands (memory bounded).

``kernel="3dgut"``: the particle evaluation of 3DGRT / 3DGUT (Moenne-Loccoz et al. 2024, Wu et al. 2025; NVIDIA 3DGRUT, the
NuRec trainer) instead of the EWA splat: per pixel ray o + t d the response is the Gaussian's maximum along the ray,
exp(-0.5 |o_g x d_g|^2 / |d_g|^2) with o_g = M^-1 (o - mu), d_g = M^-1 d, M = A R S (no 2D linearisation, no dilation);
contributions need response >= ``min_response`` (NuRec: 0.0113, ~3 sigma) and alpha = min(0.99, o * response) >= 1/255;
same global per-splat view-depth order and T < 1e-4 stop. Used to measure what an EWA rasterizer changes on 3DGUT models.
"""
from __future__ import annotations

import numpy as np

from metalsim.render.splat.scene import PackedSplats

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435)


def quat_wxyz_to_R(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def sh_eval(dc, rest, deg, d):
    """dc (N, 3), rest (N, K, 3), unit directions d (N, 3) -> colour (N, 3) (3DGS computeColorFromSH)."""
    res = C0 * dc
    if deg > 0:
        x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
        res = res - C1 * y * rest[:, 0] + C1 * z * rest[:, 1] - C1 * x * rest[:, 2]
        if deg > 1:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            res = res + C2[0] * xy * rest[:, 3] + C2[1] * yz * rest[:, 4] + C2[2] * (2 * zz - xx - yy) * rest[:, 5] \
                + C2[3] * xz * rest[:, 6] + C2[4] * (xx - yy) * rest[:, 7]
            if deg > 2:
                res = res + C3[0] * y * (3 * xx - yy) * rest[:, 8] + C3[1] * xy * z * rest[:, 9] \
                    + C3[2] * y * (4 * zz - xx - yy) * rest[:, 10] + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * rest[:, 11] \
                    + C3[4] * x * (4 * zz - xx - yy) * rest[:, 12] + C3[5] * z * (xx - yy) * rest[:, 13] \
                    + C3[6] * x * (xx - 3 * yy) * rest[:, 14]
    return np.maximum(res + 0.5, 0.0)


def unpack(s: PackedSplats, sl=slice(None)):
    """PackedSplats rows ``sl`` -> float64 arrays (mean world, opacity, quat file, scale file, dc, rest)."""
    g1 = s.g1[sl].astype(np.float64)
    sh = s.sh[sl]
    n = len(g1)
    dc = sh[:, 0:8].copy().view(np.float16)[:, :3].astype(np.float64)
    K = (s.sh_degree + 1) ** 2 - 1 if s.sh_degree > 0 else 0
    if K == 0:
        rest = np.zeros((n, 0, 3))
    elif s.sh_u8:
        rest = (sh[:, 8:8 + 3 * K].astype(np.float64) - 128.0) / 128.0
    else:
        rest = sh[:, 8:8 + 6 * K].copy().view(np.float16).astype(np.float64)
    return s.g0[sl, :3].astype(np.float64), s.g0[sl, 3].astype(np.float64), g1[:, 0:4], g1[:, 4:7], dc, rest.reshape(n, K, 3)


def project(s: PackedSplats, cam_pos, cam_mat_mj, K, width, height, near=0.05, eps2d=0.3, antialiased=None, chunk=1 << 20,
            kernel="ewa", min_response=0.0):
    """One camera. cam_mat_mj: MuJoCo camera frame (columns x right, y up, z back). Returns per-splat arrays of the
    splats with a non-empty support inside the image. Processed in chunks of splats (memory bound)."""
    parts = [_project(s, slice(i, min(i + chunk, s.n)), cam_pos, cam_mat_mj, K, width, height, near, eps2d, antialiased, kernel, min_response)
             for i in range(0, s.n, chunk)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _project(s, sl, cam_pos, cam_mat_mj, K, width, height, near, eps2d, antialiased, kernel="ewa", min_response=0.0):
    mean, op, q, sc, dc, rest = unpack(s, sl)
    base = sl.start
    Rc = np.asarray(cam_mat_mj, np.float64).reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])     # OpenCV axes (columns)
    Wv = Rc.T
    pc = (mean - np.asarray(cam_pos, np.float64)) @ Wv.T
    fx, fy, cx, cy = K
    ok = pc[:, 2] > near
    idx = np.nonzero(ok)[0]
    pc = pc[idx]
    R = quat_wxyz_to_R(q[idx])
    if kernel == "3dgut":
        sc = np.maximum(sc, 1e-6)          # fp16 scales of very flat 3DGUT particles can flush to 0; 1 um changes no ray response
    M = np.asarray(s.A, np.float64)[None] @ (R * sc[idx][:, None, :])
    Sig = M @ M.transpose(0, 2, 1)
    z = pc[:, 2]
    tx = z * np.clip(pc[:, 0] / z, -1.3 * cx / fx, 1.3 * (width - cx) / fx)
    ty = z * np.clip(pc[:, 1] / z, -1.3 * cy / fy, 1.3 * (height - cy) / fy)
    J = np.zeros((len(idx), 2, 3))
    J[:, 0, 0] = fx / z; J[:, 0, 2] = -fx * tx / z ** 2
    J[:, 1, 1] = fy / z; J[:, 1, 2] = -fy * ty / z ** 2
    T = J @ Wv[None]
    cov = T @ Sig @ T.transpose(0, 2, 1)
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det0 = a * c - b * b
    a = a + eps2d; c = c + eps2d
    det = a * c - b * b
    o = op[idx].copy()
    aa = s.antialiased if antialiased is None else antialiased
    if aa:
        o = o * np.sqrt(np.clip(det0 / det, 0, None))
    keep = (det > 0) & (o >= 1 / 255)
    tthr = 2 * np.log(255 * np.maximum(o, 1e-30))
    if kernel == "3dgut":
        if min_response > 0:
            tthr = np.minimum(tthr, -2 * np.log(min_response))
        tthr = tthr * 1.69 + 4.0                   # the ray-max response is not the EWA ellipse: generous box (1.3x radius + 2 px)
    mx = cx + fx * pc[:, 0] / z; my = cy + fy * pc[:, 1] / z
    rx = np.sqrt(np.maximum(a * tthr, 0)); ry = np.sqrt(np.maximum(c * tthr, 0))
    x0 = np.maximum(np.ceil(mx - rx - 0.5), 0); x1 = np.minimum(np.floor(mx + rx - 0.5), width - 1)
    y0 = np.maximum(np.ceil(my - ry - 0.5), 0); y1 = np.minimum(np.floor(my + ry - 0.5), height - 1)
    keep &= (x0 <= x1) & (y0 <= y1)
    k = np.nonzero(keep)[0]
    li = idx[k]
    gi = li + base
    dw = mean[li] - np.asarray(cam_pos, np.float64)
    dw /= np.linalg.norm(dw, axis=1, keepdims=True)
    df = dw @ np.linalg.inv(np.asarray(s.A, np.float64)).T
    df /= np.linalg.norm(df, axis=1, keepdims=True)
    col = sh_eval(dc[li], rest[li], s.sh_degree, df)
    dd = det[k]
    out = dict(idx=gi, mx=mx[k], my=my[k], A=c[k] / dd, B=-b[k] / dd, C=a[k] / dd, o=o[k], z=z[k], col=col,
               x0=x0[k].astype(np.int64), x1=x1[k].astype(np.int64), y0=y0[k].astype(np.int64), y1=y1[k].astype(np.int64))
    if kernel == "3dgut":
        Mi = np.linalg.inv(M[k])                                   # world -> Gaussian unit frame
        out["Mi"] = Mi.reshape(-1, 9)
        out["og"] = np.einsum("nij,nj->ni", Mi, np.asarray(cam_pos, np.float64) - mean[li])
        out["o"] = op[li]                                          # no 2D-filter compensation in 3DGUT
    return out


def render(s: PackedSplats, cam_pos, cam_mat_mj, K, width, height, near=0.05, eps2d=0.3, antialiased=None, background=(0, 0, 0),
           band=8, mesh_rgb=None, mesh_depth=None, kernel="ewa", min_response=0.0):
    """-> dict(rgb float (H, W, 3) in [0, inf), depth (median), depth_expected, alpha). Optional mesh composite.
    kernel: "ewa" (3DGS) or "3dgut" (ray-max 3D evaluation, module docstring; eps2d is ignored)."""
    if kernel not in ("ewa", "3dgut"):
        raise ValueError(kernel)
    p = project(s, cam_pos, cam_mat_mj, K, width, height, near, 0.0 if kernel == "3dgut" else eps2d, antialiased, kernel=kernel,
                min_response=min_response)
    if kernel == "3dgut":
        Rc = np.asarray(cam_mat_mj, np.float64).reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        fx_, fy_, cx_, cy_ = K
    H, W = height, width
    rgb = np.zeros((H, W, 3)); med = np.zeros((H, W)); dex = np.zeros((H, W)); alpha = np.zeros((H, W))
    bg = np.broadcast_to(np.asarray(background, np.float64), (H, W, 3)).copy()
    mz = np.full((H, W), np.inf)
    if mesh_depth is not None:
        hit = mesh_depth > 0
        mz[hit] = mesh_depth[hit]
        bg[hit] = mesh_rgb[hit].astype(np.float64) / 255.0
    for r0 in range(0, H, band):
        r1 = min(r0 + band, H)
        sel = np.nonzero((p["y1"] >= r0) & (p["y0"] < r1))[0]
        if len(sel) == 0:
            rgb[r0:r1] = bg[r0:r1]
            continue
        ya = np.maximum(p["y0"][sel], r0); yb = np.minimum(p["y1"][sel], r1 - 1)
        nx = p["x1"][sel] - p["x0"][sel] + 1; ny = yb - ya + 1
        cnt = nx * ny
        tot = int(cnt.sum())
        rep = np.repeat(np.arange(len(sel)), cnt)
        local = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        sj = sel[rep]
        px = p["x0"][sj] + local % nx[rep]
        py = ya[rep] + local // nx[rep]
        if kernel == "3dgut":
            dcam = np.stack([(px + 0.5 - cx_) / fx_, (py + 0.5 - cy_) / fy_, np.ones(len(px))], 1)
            dw = dcam @ Rc.T
            Mi = p["Mi"][sj].reshape(-1, 3, 3)
            dg = np.einsum("nij,nj->ni", Mi, dw); og = p["og"][sj]
            cr = np.cross(og, dg)
            power = -0.5 * (cr * cr).sum(1) / np.maximum((dg * dg).sum(1), 1e-300)
            tmax = -(og * dg).sum(1)
            al = np.minimum(0.99, p["o"][sj] * np.exp(power))
            good = (tmax > 0) & (power >= np.log(max(min_response, 1e-300))) & (al >= 1 / 255)
        else:
            dx = p["mx"][sj] - (px + 0.5); dy = p["my"][sj] - (py + 0.5)
            power = -0.5 * (p["A"][sj] * dx * dx + p["C"][sj] * dy * dy) - p["B"][sj] * dx * dy
            al = np.minimum(0.99, p["o"][sj] * np.exp(np.minimum(power, 0.0)))
            good = (power <= 0) & (al >= 1 / 255)
        sj, px, py, al = sj[good], px[good], py[good], al[good]
        zz = p["z"][sj]
        pix = (py - r0) * W + px
        good = zz <= mz[py, px]          # splats behind the mesh never contribute
        sj, px, py, al, zz, pix = sj[good], px[good], py[good], al[good], zz[good], pix[good]
        o = np.lexsort((zz, pix))
        sj, px, py, al, zz, pix = sj[o], px[o], py[o], al[o], zz[o], pix[o]
        # per-pixel exclusive transmittance
        l1 = np.log1p(-al)
        cs = np.cumsum(l1)
        start = np.r_[True, pix[1:] != pix[:-1]]
        seg_first = np.maximum.accumulate(np.where(start, np.arange(len(pix)), 0))
        base = np.where(seg_first > 0, cs[seg_first - 1], 0.0)
        incl = cs - base
        Tb = np.exp(incl - l1); Ta = np.exp(incl)
        stop = Ta < 1e-4
        # everything from the first stop in a pixel on is excluded
        stopcum = np.cumsum(stop)
        stop_base = np.where(seg_first > 0, stopcum[seg_first - 1], 0)
        use = (stopcum - stop_base) == 0
        w = al * Tb * use
        nb = (r1 - r0) * W
        c_acc = np.stack([np.bincount(pix, w * p["col"][sj, ch], nb) for ch in range(3)], -1)
        wsum = np.bincount(pix, w, nb)
        zsum = np.bincount(pix, w * zz, nb)
        Tfin = 1.0 - wsum
        # median: first used splat with Ta < 0.5
        cross = use & (Ta < 0.5)
        mdep = np.zeros(nb)
        ci = np.nonzero(cross)[0]
        if len(ci):
            first = np.r_[True, pix[ci][1:] != pix[ci][:-1]]
            mdep[pix[ci][first]] = zz[ci][first]
        sl = (slice(r0, r1),)
        mzb = mz[r0:r1].ravel(); mesh = np.isfinite(mzb)
        Tf = Tfin.copy()
        rgb[sl] = (c_acc + Tf[:, None] * bg[r0:r1].reshape(-1, 3)).reshape(r1 - r0, W, 3)
        mdep = np.where((mdep == 0) & mesh & (Tf >= 0.5), mzb, mdep)
        med[sl] = mdep.reshape(r1 - r0, W)
        ws = wsum + np.where(mesh, Tf, 0); ds = zsum + np.where(mesh, Tf * np.where(mesh, mzb, 0), 0)
        dex[sl] = np.where(ws >= 0.5, ds / np.maximum(ws, 1e-12), 0).reshape(r1 - r0, W)
        alpha[sl] = np.where(mesh, 1.0, wsum).reshape(r1 - r0, W)
    return dict(rgb=rgb, depth=med, depth_expected=dex, alpha=alpha, n_visible=len(p["idx"]))
