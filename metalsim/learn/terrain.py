"""Rough terrain and height scanning for locomotion tasks: Isaac Lab's terrain, exactly.

``isaac_rough_terrain`` is the terrain of Isaac-Velocity-Rough-G1-v0: Isaac Lab v2.3.2's
``TerrainGenerator`` on ``ROUGH_TERRAINS_CFG`` with the curriculum on, ported in
``metalsim.learn.isaac_terrain`` (same sub-terrain functions, proportions and column assignment,
difficulty ``(row + U) / num_rows`` from the generator's ``default_rng(seed)``, the global numpy and
torch random streams Isaac's sub-terrain functions draw from, 0.1 m horizontal / 5 mm vertical scale,
slope threshold 0.75, the 20 m border, Isaac's centring and Isaac's own sub-terrain origins). It is
returned as a MuJoCo heightfield: the top surface of Isaac's terrain mesh sampled on the 0.1 m grid, in
Isaac's world frame (x along the 10 difficulty rows, y along the 20 type columns, centred at 0).
``tests/test_terrain.py::test_isaac_rough_terrain_exact`` checks it against Isaac's generator code run
here. Between grid points the heightfield interpolates, so Isaac's vertical walls (stair risers, box
edges, slope-threshold walls) become 0.1 m ramps.

``HeightScanner`` is Isaac's ``RayCasterCfg`` height scan: a 1.6 x 1.0 m grid at 0.1 m resolution
(17 x 11 = 187 rays) attached to a body, yaw-aligned, rays cast straight down from 20 m above,
returning ``body_z - hit_z - 0.5`` per ray, evaluated by triangle interpolation of the heightfield
with Isaac's diagonal.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from metalsim.learn.isaac_terrain import isaac_rough_terrain_generator


def isaac_rough_terrain(size=8.0, res=0.1, num_rows=10, num_cols=20, seed=0, border=20.0, torch_device="cuda",
                        curriculum=True):
    """Isaac's rough terrain for env seed ``seed`` as a dict with the MuJoCo hfield fields (nrow, ncol,
    size, data in [0, 1], zmin, zmax, res, ``H`` = heights [y index, x index]) and helpers ``mesh()`` ->
    (vertices, faces), ``origin_table()`` -> (num_rows, num_cols, 3) Isaac's sub-terrain origins
    (rows = difficulty levels, columns = terrain types) and ``origins(n, seed)`` -> (n, 3) random
    sub-terrain origins. ``torch_device``: the device whose torch generator Isaac's random-grid boxes
    are drawn from ("cuda" = what Isaac does on an NVIDIA GPU; "cpu" = Isaac's code on a CPU-only
    machine). ``size``, ``res`` and ``border`` override ROUGH_TERRAINS_CFG's size, horizontal scale and
    border width (defaults are Isaac's)."""
    gen = isaac_rough_terrain_generator(seed=seed, num_rows=num_rows, num_cols=num_cols, torch_device=torch_device,
                                        curriculum=curriculum, size=(size, size), horizontal_scale=res,
                                        border_width=border)
    Hb = np.ascontiguousarray(gen.surface.T).astype(np.float64)    # MuJoCo layout: rows along y, columns along x
    zmin, zmax = float(Hb.min()), float(Hb.max())
    data = (Hb - zmin) / max(zmax - zmin, 1e-6)     # MuJoCo hfield data in [0, 1]
    nrow, ncol = Hb.shape
    # MuJoCo hfield: size = (radius_x, radius_y, elevation_z, base_z); the ncol samples span
    # [-radius_x, radius_x] exactly, so radius = (n - 1) * res / 2 keeps the sample spacing at ``res``;
    # Isaac centres its terrain at the origin, so the hfield centre is the world origin
    size_x = (ncol - 1) * res / 2; size_y = (nrow - 1) * res / 2
    assert abs(size_x + gen.x0) < 1e-9 and abs(size_y + gen.y0) < 1e-9
    hf = {"nrow": nrow, "ncol": ncol, "size": [size_x, size_y, max(zmax - zmin, 1e-3), max(-zmin, 0.0) + 0.05],
          "data": data.astype(np.float32), "zmin": zmin, "zmax": zmax, "res": res, "H": Hb, "generator": gen}
    table = gen.terrain_origins.astype(np.float32)     # Isaac: torch.float tensor of terrain_origins

    def origins(n, seed=0):
        rr = np.random.default_rng(seed + 1)
        rows = rr.integers(0, num_rows, n); cols = rr.integers(0, num_cols, n)
        return table[rows, cols].copy()

    def mesh():
        ys, xs = np.mgrid[0:nrow, 0:ncol]
        X = xs * res - size_x; Y = ys * res - size_y; Z = Hb
        V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
        i = (ys[:-1, :-1] * ncol + xs[:-1, :-1]).ravel()
        F = np.concatenate([np.stack([i, i + 1, i + ncol + 1], 1), np.stack([i, i + ncol + 1, i + ncol], 1)]).astype(np.uint32)
        return V, F

    def origin_table():
        """(num_rows, num_cols, 3) sub-terrain origins, Isaac's ``terrain_origins``: rows are difficulty
        levels (Isaac's terrain curriculum moves envs between rows), columns are terrain types."""
        return table.copy()

    hf["origins"] = origins; hf["mesh"] = mesh; hf["origin_table"] = origin_table
    hf["num_rows"] = num_rows; hf["num_cols"] = num_cols; hf["cell_size"] = size
    return hf


@wp.kernel
def height_scan_grid(xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33), body: int,
                     grid: wp.array2d(dtype=float), H: wp.array2d(dtype=float), res: float, size_x: float, size_y: float,
                     offset_z: float, out: wp.array2d(dtype=float)):
    """Isaac's RayCaster height scan on a heightfield terrain: the ray hit on the triangulated grid is
    the triangle interpolation of the heights at the ray's (x, y), so no ray tracing is needed. The
    diagonal is Isaac's height-field mesh diagonal ((i, j) -> (i+1, j+1))."""
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
