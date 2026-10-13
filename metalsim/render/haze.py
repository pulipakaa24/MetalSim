"""MuJoCo's haze over infinite planes and its linear fog (``shaders/haze.h``), as ``mujoco.Renderer`` draws them.

MuJoCo 3.14.0 (``render_context.c`` ``haze`` / ``setVertexHaze``, ``render_gl3.c`` ``mjr_render`` skybox pass, ``initGL3``):

* **haze** (``mjRND_HAZE``, on by default): drawn in the skybox pass, so only when a skybox texture is drawn, for the
  first infinite plane (``size="0 0"``) of the scene and only when the camera is above it. It is a truncated cone of
  ``vis.quality.numslices`` flat facets centred on the camera: at height ``h H`` above the plane (``H`` the camera's
  height, ``h`` in [0, 1]) its radius is ``S (1 - r (1 - h))``, ``S = 0.7 x`` the camera's far plane
  (``vis.map.zfar x stat.extent``), ``r = vis.map.haze``. Its colour is ``vis.rgba.haze`` with alpha 0 on the plane, 1
  at the transition height ``hh = sqrt(1/2) r sin(a) / sin(3 pi / 4 - a)``, ``a = atan2(1, r)``, and 0 at the camera's
  height, blended over the opaque geometry it is in front of (depth-tested; transparent geoms are drawn after it).
* **fog** (``mjRND_FOG``, off by default): OpenGL linear fog on the eye-space depth of every lit fragment, the skybox
  and the haze: ``f = clamp((fogend - z) / (fogend - fogstart))``, ``fogstart / fogend = vis.map.fogstart / fogend x
  stat.extent``, colour ``vis.rgba.fog``. Measured on ``mujoco.Renderer`` (macOS): the fog coordinate is the eye-space
  z-depth, not the radial distance.

``HazeParams`` packs both for the shaders; ``haze_alpha`` is the NumPy reference of the shader's per-pixel haze (the
same faceted cone), checked against ``mujoco.Renderer`` in ``tests/test_render_haze.py``.

Where MetalSim differs: the plane frame is the plane's pose at the initial state (an infinite plane is almost always
static; MuJoCo uses the current pose); the depth image is unchanged (MuJoCo's depth buffer holds the haze cone and the
skybox); a transparent geom behind the cone gets the haze blended into its own colour (MuJoCo's depth test hides it).
"""
from __future__ import annotations

import mujoco
import numpy as np

WORDS = 28     # 7 float4 (shaders/haze.h HazeParams)


def transition_height(r: float) -> float:
    a = np.arctan2(np.float32(1.0), np.float32(r))
    b = np.float32(0.75 * np.pi) - a
    return float(np.float32(np.sqrt(0.5)) * np.float32(r) * np.sin(a) / np.sin(b))


def first_infinite_plane(m: mujoco.MjModel, geoms) -> int:
    """Geom id of the first infinite plane among the drawn geoms (MuJoCo's scene order: geom id), or -1."""
    for g in sorted(int(x) for x in geoms):
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE) and m.geom_size[g][0] == 0 and m.geom_size[g][1] == 0:
            return g
    return -1


def haze_params(m: mujoco.MjModel, geoms, *, haze: bool = True, fog: bool = False, sky_drawn: bool = True) -> np.ndarray:
    """(28,) float32 HazeParams. Haze needs a drawn skybox texture and an infinite plane among ``geoms``; returns
    zeros (both off) when neither haze nor fog applies."""
    p = np.zeros(WORDS, np.float32)
    g = first_infinite_plane(m, geoms)
    use_haze = bool(haze and sky_drawn and g >= 0)
    if not (use_haze or fog):
        return p
    S = np.float32(m.vis.map.zfar * m.stat.extent) * np.float32(0.7)
    r = float(m.vis.map.haze)
    if g >= 0:
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        o = np.asarray(d.geom_xpos[g], np.float64)
        R = np.asarray(d.geom_xmat[g], np.float64).reshape(3, 3)
    else:
        o, R = np.zeros(3), np.eye(3)
    p[0:3] = o; p[3] = S
    p[4:7] = R[:, 0]; p[7] = r
    p[8:11] = R[:, 1]; p[11] = transition_height(r)
    p[12:15] = R[:, 2]; p[15] = float(m.vis.quality.numslices)
    p[16:19] = m.vis.rgba.haze[:3]; p[19] = float((1 if use_haze else 0) | (2 if fog else 0))
    p[20] = m.vis.map.fogstart * m.stat.extent; p[21] = m.vis.map.fogend * m.stat.extent
    p[24:28] = m.vis.rgba.fog
    return p


def active(params: np.ndarray) -> bool:
    return int(params[19]) != 0


def haze_alpha(params: np.ndarray, cam_pos, dirs, t_surf=None) -> np.ndarray:
    """NumPy reference of ``haze.h`` ``haze_alpha``: per unit direction (..., 3) from ``cam_pos``, the haze alpha (0
    where the ray misses the cone or meets an opaque surface at ``t_surf`` first)."""
    p = np.asarray(params, np.float64)
    o, S = p[0:3], p[3]
    ax, r = p[4:7], p[7]
    ay, hh = p[8:11], p[11]
    n, ns = p[12:15], max(p[15], 3.0)
    dirs = np.asarray(dirs, np.float64)
    H = float(np.dot(np.asarray(cam_pos, np.float64) - o, n))
    out = np.zeros(dirs.shape[:-1])
    if not H > 0 or not int(p[19]) & 1:
        return out
    dx, dy, dz = dirs @ ax, dirs @ ay, dirs @ n
    az = np.arctan2(dy, dx) % (2 * np.pi)
    j = np.floor(az * ns / (2 * np.pi))
    mid = (j + 0.5) * 2 * np.pi / ns
    cs = np.cos(np.pi / ns)
    den = dx * np.cos(mid) + dy * np.sin(mid) - S * cs * r * dz / H
    with np.errstate(divide="ignore", invalid="ignore"):
        t = S * cs / np.where(den > 0, den, np.nan)
    h = 1 + t * dz / H
    ok = (dz < 0) & (den > 0) & (h >= 0) & (h <= 1)
    if t_surf is not None:
        ok &= t < np.asarray(t_surf, np.float64)
    a = np.where(h <= hh, h / hh, (1 - h) / (1 - hh))
    out[ok] = a[ok]
    return out


def camera_dirs(width: int, height: int, fx: float, fy: float, cx: float, cy: float, cam_mat) -> np.ndarray:
    """(H, W, 3) unit world directions through the pixel centres of a pinhole camera (MuJoCo camera frame: -z forward,
    +y up; ``cam_mat`` the camera's 3x3 orientation)."""
    ys, xs = np.mgrid[0:height, 0:width]
    dc = np.stack([(xs + 0.5 - cx) / fx, -(ys + 0.5 - cy) / fy, -np.ones(xs.shape)], -1)
    dw = dc @ np.asarray(cam_mat, np.float64).reshape(3, 3).T
    return dw / np.linalg.norm(dw, axis=-1, keepdims=True)
