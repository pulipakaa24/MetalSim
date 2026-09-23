"""Collision terrain for TRON1 from a scanned space (colored triangle-mesh PLY).

    .venv/bin/python -m metalsim.tron1.scene ~/Downloads/spatial_station.ply assets/tron1/scenes/spatial_station.npz

The scan (`spatial_station.ply`: 18.9 M vertices, 37.4 M faces, metres, +y up) becomes a MuJoCo
height field in a z-up world, (x, y, z)_world = (x, -z, y)_scan:

* floor: the dominant surface per 10 cm cell below the scanner (mode of the vertex heights). The
  scan's floor is not level: in one corner it sits 0.3-0.5 m lower while the ceiling above it
  does not drop, which is either a real sunken area or glossy-floor reconstruction artefacts.
  `floor="flat"` (default) puts the floor at one plane (the RANSAC fit, 0.2 deg tilt, 1.1 cm
  residual) and keeps only obstacles; `floor="scan"` keeps the scanned floor heights.
* obstacles: the highest scanned surface between 4 cm and 1.3 m above the local floor, per 5 cm
  cell, where at least 3 vertices agree, after a morphological opening (removes speckle). Tables
  become solid to their top, which is what the robot's body would hit anyway.
* unscanned cells become 1.5 m walls so the robot stays inside the captured space.

The Gaussian splat (`Spatial-Station.spz`, 8.8 M splats, SH degree 3) is visual only and is not
used here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WALL = 1.5


def read_ply_vertices(path):
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.readline()
    lines = hdr.decode().splitlines()
    nv = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
    props = [l.split()[-1] for l in lines[lines.index(next(l for l in lines if l.startswith("element vertex"))) + 1:]
             if l.startswith("property")][:6]
    if props != ["x", "y", "z", "red", "green", "blue"]:
        raise ValueError(f"unexpected vertex layout {props}")
    v = np.memmap(path, dtype=np.dtype([("p", "<f4", 3), ("c", "u1", 3)]), mode="r", offset=len(hdr), shape=(nv,))
    return np.asarray(v["p"]), np.asarray(v["c"])


def _mode_per_cell(keys, h, n_cells, lo, hi, bin_=0.02, min_count=5):
    nb = int((hi - lo) / bin_)
    b = np.clip(((h - lo) / bin_).astype(int), 0, nb - 1)
    hist = np.zeros((n_cells, nb), np.int32)
    np.add.at(hist, (keys, b), 1)
    out = lo + (hist.argmax(1) + 0.5) * bin_
    out[hist.sum(1) < min_count] = np.nan
    return out


def _opening(mask, r=1):
    from scipy.ndimage import binary_dilation, binary_erosion
    st = np.ones((2 * r + 1, 2 * r + 1), bool)
    return binary_dilation(binary_erosion(mask, st), st)


def build(ply, res=0.05, floor="flat", band=(0.04, 1.3), log=print):
    import time
    t0 = time.time()
    P, C = read_ply_vertices(ply)
    W = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1).astype(np.float64)        # z-up world
    # Global floor plane: RANSAC on low points, refined by SVD.
    rng = np.random.default_rng(0)
    low = W[W[:, 2] < np.percentile(W[:, 2], 40)][::10]
    best = None
    for _ in range(300):
        s = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        c = (np.abs((low - s[0]) @ n) < 0.02).sum()
        if best is None or c > best[0]:
            best = (c, n, s[0])
    inl = low[np.abs((low - best[2]) @ best[1]) < 0.02]
    m = inl.mean(0)
    n = np.linalg.svd(inl - m, full_matrices=False)[2][2]
    n *= np.sign(n[2])
    floor_z = lambda xy: m[2] - (n[0] * (xy[..., 0] - m[0]) + n[1] * (xy[..., 1] - m[1])) / n[2]
    H = W[:, 2] - floor_z(W[:, :2])                                         # height above floor plane
    log(f"floor plane fitted ({time.time() - t0:.0f} s)")

    x0, y0 = W[:, 0].min(), W[:, 1].min()
    nx = int((W[:, 0].max() - x0) / res) + 1
    ny = int((W[:, 1].max() - y0) / res) + 1
    ix = ((W[:, 0] - x0) / res).astype(int)
    iy = ((W[:, 1] - y0) / res).astype(int)
    key = iy * nx + ix

    # Local floor from a coarse (2x) grid of dominant low surfaces, upsampled.
    f = 2
    kc = (iy // f) * ((nx + f - 1) // f) + ix // f
    lowm = H < 0.6
    local = np.full(((ny + f - 1) // f) * ((nx + f - 1) // f), np.nan)
    local = _mode_per_cell(kc[lowm], H[lowm], local.size, -1.2, 0.6)
    local = local.reshape((ny + f - 1) // f, (nx + f - 1) // f).repeat(f, 0).repeat(f, 1)[:ny, :nx]
    seen = np.isfinite(local)
    log(f"local floor ({time.time() - t0:.0f} s)")

    rel = H - local.ravel()[key]
    inband = np.isfinite(rel) & (rel > band[0]) & (rel < band[1])
    top = np.full(ny * nx, -np.inf)
    np.maximum.at(top, key[inband], rel[inband])
    cnt = np.bincount(key[inband], minlength=ny * nx)
    obst = ((cnt >= 3) & np.isfinite(top)).reshape(ny, nx)
    obst = _opening(obst, 1)
    log(f"obstacles ({time.time() - t0:.0f} s)")
    heights = np.where(obst, np.clip(top.reshape(ny, nx), 0, band[1]), 0.0)
    if floor == "scan":
        heights = heights + np.nan_to_num(local, nan=0.0) - np.nanmin(local)
    heights[~seen] = WALL

    # Spawn: free cell farthest from any obstacle or wall.
    from scipy.ndimage import distance_transform_edt
    free = (heights < 0.02) & seen
    dist = distance_transform_edt(free)
    j, i = np.unravel_index(dist.argmax(), dist.shape)
    spawn = (x0 + (i + 0.5) * res, y0 + (j + 0.5) * res)
    return dict(heights=heights.astype(np.float32), res=res, origin=(x0, y0), spawn_xy=spawn,
                floor_mode=floor, floor_normal=n, floor_resid=float(np.std((inl - m) @ n)))


@dataclass
class ScanTerrain:
    heights: np.ndarray          # (ny, nx) metres above the floor plane, row = y
    res: float
    origin: tuple
    spawn_xy: tuple

    @staticmethod
    def load(path):
        z = np.load(path)
        return ScanTerrain(z["heights"], float(z["res"]), tuple(z["origin"]), tuple(z["spawn_xy"]))

    def height_at(self, x, y):
        i = int(np.clip((x - self.origin[0]) / self.res, 0, self.heights.shape[1] - 1))
        j = int(np.clip((y - self.origin[1]) / self.res, 0, self.heights.shape[0] - 1))
        return float(self.heights[j, i])

    def attach(self, spec):
        """Add the height field to a MuJoCo spec; the flat floor plane stops colliding."""
        import mujoco
        ny, nx = self.heights.shape
        top = max(float(self.heights.max()), 1e-3)
        sx, sy = (nx - 1) * self.res / 2, (ny - 1) * self.res / 2
        hf = spec.add_hfield(name="scan", nrow=ny, ncol=nx, size=[sx, sy, top, 0.1])
        hf.userdata = (self.heights / top).ravel().tolist()
        g = spec.worldbody.add_geom(name="scan_terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="scan",
                                    pos=[self.origin[0] + sx, self.origin[1] + sy, 0.0], rgba=[0.62, 0.6, 0.55, 1])
        floor = spec.geom("floor")
        floor.contype = floor.conaffinity = 0
        return g

    def geom_ids(self, m):
        return [m.geom("scan_terrain").id]


def main():
    ply, out = sys.argv[1], sys.argv[2]
    floor = sys.argv[3] if len(sys.argv) > 3 else "flat"
    t = build(ply, floor=floor)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in t.items()})
    h = t["heights"]
    print(f"{out}: {h.shape[1]} x {h.shape[0]} cells at {t['res']} m, obstacles {(h > 0.02).mean() * 100:.1f} % "
          f"(walls incl.), floor fit residual {t['floor_resid'] * 100:.1f} cm, spawn {np.round(t['spawn_xy'], 2)}")


if __name__ == "__main__":
    main()
