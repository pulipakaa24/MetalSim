"""Rough terrain and height scanning for locomotion tasks, after Isaac Lab's terrain generator.

``isaac_rough_terrain`` builds a heightfield in the layout of Isaac Lab's ``ROUGH_TERRAINS_CFG``:
a grid of ``num_rows x num_cols`` sub-terrains of ``size`` (8 x 8 m), difficulty increasing along
rows (curriculum), with the same sub-terrain mix and proportions (pyramid stairs up/down 0.2 each,
boxes 0.2, random rough 0.2, pyramid slopes up/down 0.1 each) at Isaac's parameters (stair step
height 0.05-0.23 m, step width 0.3 m, box height 0.05-0.2 m, random rough noise 0.02-0.1 m at
0.1 m grid, slopes 0-0.4). Isaac's generator is procedural on a 0.1 m grid as well; shapes are
re-implemented here from the config, not copied, so terrain statistics match and exact heights do not.

``HeightScanner`` is Isaac's ``RayCasterCfg`` height scan: a 1.6 x 1.0 m grid at 0.1 m resolution
(17 x 11 = 187 rays) attached to a body, yaw-aligned, rays cast straight down from 20 m above,
returning ``body_z - hit_z - 0.5`` per ray, evaluated by Metal ray queries against the terrain mesh.
"""
from __future__ import annotations

import numpy as np
import warp as wp


def _pyramid_stairs(h, res, step_w, step_h, up=True, platform=3.0):
    n = h.shape[0]; c = n // 2
    r = np.maximum(np.abs(np.arange(n) - c)[:, None], np.abs(np.arange(n) - c)[None, :]) * res
    size_m = n * res / 2
    levels = np.floor(np.clip(size_m - platform / 2 - r, 0, None) / step_w)
    z = levels * step_h
    return z if up else -z


def _boxes(h, res, rng, box_h, n_boxes=40, box_size=(1.0, 3.0)):
    n = h.shape[0]
    z = np.zeros_like(h)
    for _ in range(n_boxes):
        w = int(rng.uniform(*box_size) / res); l = int(rng.uniform(*box_size) / res)
        x = rng.integers(0, max(n - w, 1)); y = rng.integers(0, max(n - l, 1))
        z[x:x + w, y:y + l] = rng.uniform(-box_h, box_h)
    c = n // 2; p = int(1.0 / res)
    z[c - p:c + p, c - p:c + p] = 0.0     # central platform
    return z


def _random_rough(h, res, rng, noise, step=0.02, downsample=0.1):
    n = h.shape[0]
    k = max(int(round(downsample / res)), 1)
    coarse = rng.uniform(-noise, noise, (n // k + 1, n // k + 1))
    coarse = np.round(coarse / step) * step
    z = np.kron(coarse, np.ones((k, k)))[:n, :n]
    return z


def _pyramid_slope(h, res, slope, up=True, platform=2.0):
    n = h.shape[0]; c = n // 2
    r = np.maximum(np.abs(np.arange(n) - c)[:, None], np.abs(np.arange(n) - c)[None, :]) * res
    size_m = n * res / 2
    z = np.clip(size_m - platform / 2 - r, 0, None) * slope
    return z if up else -z


def isaac_rough_terrain(size=8.0, res=0.1, num_rows=10, num_cols=20, seed=0, border=2.0):
    """Returns dict with MuJoCo hfield fields (nrow, ncol, size, data in [0,1]) and helpers:
    ``mesh()`` -> (vertices, faces) and ``origins(n, seed)`` -> (n,3) env origins on sub-terrain
    centres (rows sampled uniformly: Isaac's curriculum starts from the easiest rows; we start at
    the full distribution as a benchmark-time choice)."""
    rng = np.random.default_rng(seed)
    n_cell = int(round(size / res))
    H = np.zeros((num_rows * n_cell, num_cols * n_cell))
    kinds = (["stairs_up"] * 2 + ["stairs_down"] * 2 + ["boxes"] * 2 + ["rough"] * 2 + ["slope_up"] + ["slope_down"])
    for r in range(num_rows):
        difficulty = (r + 1) / num_rows
        for c in range(num_cols):
            kind = kinds[c % len(kinds)]
            cell = np.zeros((n_cell, n_cell))
            if kind == "stairs_up":
                z = _pyramid_stairs(cell, res, 0.3, 0.05 + difficulty * (0.23 - 0.05), True)
            elif kind == "stairs_down":
                z = _pyramid_stairs(cell, res, 0.3, 0.05 + difficulty * (0.23 - 0.05), False)
            elif kind == "boxes":
                z = _boxes(cell, res, rng, 0.05 + difficulty * (0.2 - 0.05))
            elif kind == "rough":
                z = _random_rough(cell, res, rng, 0.02 + difficulty * (0.1 - 0.02))
            elif kind == "slope_up":
                z = _pyramid_slope(cell, res, difficulty * 0.4, True)
            else:
                z = _pyramid_slope(cell, res, difficulty * 0.4, False)
            H[r * n_cell:(r + 1) * n_cell, c * n_cell:(c + 1) * n_cell] = z
    # border of flat ground
    nb = int(round(border / res))
    Hb = np.zeros((H.shape[0] + 2 * nb, H.shape[1] + 2 * nb))
    Hb[nb:-nb, nb:-nb] = H
    zmin, zmax = float(Hb.min()), float(Hb.max())
    data = (Hb - zmin) / max(zmax - zmin, 1e-6)     # MuJoCo hfield data in [0, 1]
    nrow, ncol = Hb.shape
    # MuJoCo hfield: size = (radius_x, radius_y, elevation_z, base_z); rows along y, cols along x;
    # the ncol samples span [-radius_x, radius_x] exactly, so radius = (n - 1) * res / 2 keeps the
    # sample spacing at ``res`` (the mesh() triangulation and the collision surface then coincide)
    size_x = (ncol - 1) * res / 2; size_y = (nrow - 1) * res / 2
    hf = {"nrow": nrow, "ncol": ncol, "size": [size_x, size_y, max(zmax - zmin, 1e-3), max(-zmin, 0.0) + 0.05],
          "data": data.astype(np.float32), "zmin": zmin, "zmax": zmax, "res": res, "H": Hb}
    n_cell_m = size

    def origins(n, seed=0):
        rr = np.random.default_rng(seed + 1)
        rows = rr.integers(0, num_rows, n); cols = rr.integers(0, num_cols, n)
        x = (cols + 0.5) * n_cell_m - size_x + border
        y = (rows + 0.5) * n_cell_m - size_y + border
        z = np.array([Hb[int(round((yy + size_y) / res)), int(round((xx + size_x) / res))] for xx, yy in zip(x, y)])
        return np.stack([x, y, z], 1).astype(np.float32)

    def mesh():
        ys, xs = np.mgrid[0:nrow, 0:ncol]
        X = xs * res - size_x; Y = ys * res - size_y; Z = Hb
        V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
        i = (ys[:-1, :-1] * ncol + xs[:-1, :-1]).ravel()
        F = np.concatenate([np.stack([i, i + 1, i + ncol + 1], 1), np.stack([i, i + ncol + 1, i + ncol], 1)]).astype(np.uint32)
        return V, F

    hf["origins"] = origins; hf["mesh"] = mesh
    return hf


@wp.kernel
def height_scan_grid(xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33), body: int,
                     grid: wp.array2d(dtype=float), H: wp.array2d(dtype=float), res: float, size_x: float, size_y: float,
                     offset_z: float, out: wp.array2d(dtype=float)):
    """Isaac's RayCaster height scan on a heightfield terrain: the ray hit on the triangulated grid is
    the triangle interpolation of the heights at the ray's (x, y), so no ray tracing is needed."""
    e, k = wp.tid()
    bp = xpos[e, body]
    R = xmat[e, body]
    yaw = wp.atan2(R[1, 0], R[0, 0])
    gx = grid[k, 0]; gy = grid[k, 1]
    x = bp[0] + wp.cos(yaw) * gx - wp.sin(yaw) * gy
    y = bp[1] + wp.sin(yaw) * gx + wp.cos(yaw) * gy
    nrow = H.shape[0]; ncol = H.shape[1]
    u = (x + size_x) / res; v = (y + size_y) / res
    u = wp.clamp(u, 0.0, float(ncol - 1) - 1e-4); v = wp.clamp(v, 0.0, float(nrow - 1) - 1e-4)
    j = int(wp.floor(u)); i = int(wp.floor(v))
    fx = u - float(j); fy = v - float(i)
    z00 = H[i, j]; z10 = H[i, j + 1]; z11 = H[i + 1, j + 1]; z01 = H[i + 1, j]
    if fx >= fy:      # triangle (i,j) (i,j+1) (i+1,j+1), the mesh() diagonal
        z = z00 + fx * (z10 - z00) + fy * (z11 - z10)
    else:             # triangle (i,j) (i+1,j+1) (i+1,j)
        z = z00 + fy * (z01 - z00) + fx * (z11 - z01)
    out[e, k] = bp[2] - z - offset_z


class HeightScanner:
    """Isaac's height scan (1.6 x 1.0 m grid, 0.1 m, 187 rays, yaw-aligned, ``body_z - hit_z - 0.5``)
    evaluated as a Warp kernel on the simulation queue (capturable into rollout graphs); the terrain is
    a heightfield, so the ray hit is the triangle interpolation of the grid. ``metalsim.sensors.raytrace``
    remains the path for scanning arbitrary meshes."""

    def __init__(self, task, hfield, body="torso_link", size=(1.6, 1.0), res=0.1, offset_z=0.5, device="metal:0"):
        import mujoco
        self.task, self.device = task, device
        self.body = mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_BODY, body)
        xs = np.arange(-size[0] / 2, size[0] / 2 + 1e-6, res); ys = np.arange(-size[1] / 2, size[1] / 2 + 1e-6, res)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self.grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)   # (187, 2)
        self.n_rays = len(self.grid)
        self.offset_z = offset_z
        self.grid_wp = wp.array(self.grid, dtype=float, device=device)
        self.H = wp.array(hfield["H"].astype(np.float32), dtype=float, device=device)
        self.res = float(hfield["res"]); self.size_x = float(hfield["size"][0]); self.size_y = float(hfield["size"][1])

    def launch(self, step_idx=None):
        sim = self.task.sim
        wp.launch(height_scan_grid, dim=(self.task.n, self.n_rays),
                  inputs=[sim.d.xpos, sim.d.xmat, self.body, self.grid_wp, self.H, self.res, self.size_x, self.size_y,
                          self.offset_z, self.task.height_scan], device=self.device)
