"""Where the envs sit in the ray tracers' shared world (``shaders/env_grid.h``).

All envs share one instance acceleration structure; env e's instances are translated by its grid offset and its rays
start at its offset, so a ray must never reach another env's geometry.

* ``legacy``: a 2D grid from the origin with stride 2 (extent + max range) + 1 (rays as long as asked cannot reach a
  neighbour). The offsets grow to kilometres (tier 2's max range is 1000 m: stride ~2 km, 64 km at 1024 envs), where
  float32 spacing is millimetres: the same state renders differently in different envs (tier 2 depth up to 1.8 mm
  apart between env 0 and env 4).
* ``compact`` (default): a 3D grid centred on the origin whose cells are the env's content box (every drawn geom, camera,
  site, light and dynamic object at the initial state) plus a margin; each ray ends where it leaves its env's cell.
  Geometry or sensors that later move beyond the cell are cut off at its boundary: ``margin`` (default max(1 m, a
  tenth of the content's largest side)) bounds how far they may go. Vertical stacking is weighted 8x in the layout
  choice (a floor at grazing angles amplifies the origin's z rounding; horizontal offsets do not change floor depths).
"""
from __future__ import annotations

import numpy as np

WORDS = 104     # 416 bytes: uint nx, ny, clip, n_planes; float4 S, c, lo, hi; uint4 plane slots; 5 x 4 float4 planes
MAX_PLANES = 4


class EnvGrid:
    def __init__(self, n_envs: int, mode: str = "compact", *, stride: float = 0.0, tpr: int = 1, lo=None, hi=None,
                 margin: float | None = None):
        self.n, self.mode = int(n_envs), mode
        self.planes = []        # infinite planes continued past the cell (compact; set_planes)
        if mode == "legacy":
            self.nx, self.ny, self.nz = int(tpr), max(self.n, 1), 1
            self.S = np.array([stride, stride, stride], np.float64)
            self.c = np.zeros(3)
            self.lo = self.hi = np.zeros(3)
            self.clip = 0
            return
        if mode != "compact":
            raise ValueError(f"env layout must be compact or legacy, not {mode!r}")
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        ext = hi - lo
        m = max(1.0, 0.1 * float(ext.max())) if margin is None else float(margin)
        self.margin = m
        self.lo, self.hi = lo - m, hi + m
        self.S = (self.hi - self.lo) + 1e-3                      # touching cells, a millimetre apart
        self.nx, self.ny, self.nz = _best_dims(self.n, self.S)
        self.c = (np.array([self.nx, self.ny, self.nz], np.float64) - 1.0) / 2.0
        self.clip = 1

    def words(self) -> np.ndarray:
        a = np.zeros(WORDS, np.uint32)
        a[0:4] = (self.nx, self.ny, self.clip, len(self.planes) if self.clip else 0)
        f = a.view(np.float32)
        f[4:7] = self.S
        f[8:11] = self.c
        f[12:15] = self.lo
        f[16:19] = self.hi
        if self.clip:
            f[15] = 0.5 * self.margin       # lo.w: the "left the content" slack (env_grid.h env_grid_left)
            for k, p in enumerate(self.planes):
                a[20 + k] = p["slot"]
                f[24 + 4 * k: 28 + 4 * k] = list(p["n"]) + [float(np.dot(p["n"], p["o"]))]
                f[40 + 4 * k: 44 + 4 * k] = list(p["o"]) + [p["sx"]]
                f[56 + 4 * k: 60 + 4 * k] = list(p["x"]) + [p["sy"]]
                f[72 + 4 * k: 76 + 4 * k] = list(p["y"]) + [0.0]
                f[88 + 4 * k: 92 + 4 * k] = p["uv"]
        return a

    def set_planes(self, planes):
        """Infinite planes continued past the cell: dicts of slot, n, o, x, y (env frame), sx, sy (finite half sizes,
        0 = infinite), uv (us, ua, vs, va)."""
        if len(planes) > MAX_PLANES:
            raise ValueError(f"{len(planes)} infinite planes: the compact env grid continues at most {MAX_PLANES}")
        self.planes = list(planes)

    def grown(self) -> "EnvGrid":
        """The same layout with the content box doubled about its centre (and the margin with it)."""
        lo, hi = self.lo + self.margin, self.hi - self.margin
        c, h = (lo + hi) / 2, (hi - lo) / 2
        g = EnvGrid(self.n, self.mode, lo=c - 2 * h - 1.0, hi=c + 2 * h + 1.0, margin=2 * self.margin)
        g.planes = list(self.planes)
        return g

    def offsets(self, envs) -> np.ndarray:
        """(k, 3) float64 offsets as the shaders compute them in float32 (legacy: float64 products, as before)."""
        e = np.asarray(envs, np.int64)
        idx = np.stack([e % self.nx, (e // self.nx) % self.ny, e // (self.nx * self.ny)], -1).astype(np.float64)
        if self.mode == "legacy":
            return idx * self.S
        return ((idx.astype(np.float32) - self.c.astype(np.float32)) * self.S.astype(np.float32)).astype(np.float64)

    def max_offset(self) -> float:
        return float(np.max(np.abs(self.offsets([0, self.n - 1]))) if self.n else 0.0)


Z_WEIGHT = 8.0     # vertical offsets cost more: a floor seen at grazing angles amplifies the ray origin's z rounding


def _best_dims(n: int, S: np.ndarray):
    """(nx, ny, nz) with nx ny nz >= n minimising the largest centred offset max((nx-1) Sx, (ny-1) Sy, Z_WEIGHT (nz-1) Sz)
    (horizontal offsets leave depths to a horizontal floor unchanged; vertical ones do not)."""
    best, best_v = (n, 1, 1), np.inf
    for nx in range(1, n + 1):
        for ny in range(1, (n + nx - 1) // nx + 1):
            nz = (n + nx * ny - 1) // (nx * ny)
            v = max((nx - 1) * S[0], (ny - 1) * S[1], Z_WEIGHT * (nz - 1) * S[2])
            if v < best_v - 1e-9 or (abs(v - best_v) <= 1e-9 and nx * ny * nz < np.prod(best)):
                best, best_v = (nx, ny, nz), v
            if ny > 1 and (ny - 1) * S[1] > best_v:
                break
        if (nx - 1) * S[0] > best_v:
            break
    return best


def content_box(model, max_group: int = 3):
    """(lo, hi) of everything an env draws or looks from at the initial state (key frame 0, else qpos0): each geom of a
    drawn group (its bounding box, MuJoCo's geom_aabb, at its pose; of an infinite plane only its origin: the ray
    tracers continue planes past the cell), cameras, sites, lights, flex vertices and tendon wrap points."""
    import mujoco
    d = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    pts = [np.zeros((1, 3))]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64)
    for g in range(model.ngeom):
        if model.geom_group[g] > max_group:
            continue
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE) and (model.geom_size[g][0] <= 0 or model.geom_size[g][1] <= 0):
            pts.append(np.asarray(d.geom_xpos[g], np.float64).reshape(1, 3))     # the cell reaches down to the plane
            continue
        aabb = np.asarray(model.geom_aabb[g], np.float64)      # local centre (3), half sizes (3)
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):    # MuJoCo gives planes an unbounded aabb
            aabb = np.array([0.0, 0.0, 0.0, model.geom_size[g][0], model.geom_size[g][1], 0.0])
        corners = aabb[:3] + signs * aabb[3:]
        R = np.asarray(d.geom_xmat[g]).reshape(3, 3)
        pts.append(corners @ R.T + np.asarray(d.geom_xpos[g]))
    for arr in (d.cam_xpos, d.site_xpos, d.light_xpos, d.flexvert_xpos if model.nflexvert else None,
                d.wrap_xpos if model.ntendon else None):
        if arr is not None and len(arr):
            pts.append(np.asarray(arr, np.float64).reshape(-1, 3))
    P = np.concatenate(pts)
    P = P[np.isfinite(P).all(1)]
    return P.min(0), P.max(0)


def infinite_planes(model, tables):
    """The scene's infinite-plane slots (tables.plane flag) as EnvGrid.set_planes entries, posed at the initial state
    (planes on moving bodies keep that pose past the cell, with a warning)."""
    import mujoco
    if tables.plane is None:
        return []
    d = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    out = []
    for slot, row in enumerate(np.asarray(tables.plane)):
        if slot >= len(tables.geoms) or row[0] < 0.5:
            continue
        g = int(tables.geoms[slot])
        if model.geom_bodyid[g] != 0 or model.body_mocapid[model.geom_bodyid[g]] >= 0:
            import warnings
            warnings.warn(f"infinite plane geom {g} is not on the world body: continued past the env's cell at its "
                          "initial pose")
        R = np.asarray(d.geom_xmat[g], np.float64).reshape(3, 3)
        out.append({"slot": slot, "n": R[:, 2], "o": np.asarray(d.geom_xpos[g], np.float64), "x": R[:, 0], "y": R[:, 1],
                    "sx": float(row[1]), "sy": float(row[2]), "uv": [float(v) for v in row[4:8]]})
    return out
