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
                        curriculum=True, collision="hfield", fine_res=0.025):
    """Isaac's rough terrain for env seed ``seed`` as a dict with the MuJoCo hfield fields (nrow, ncol,
    size, data in [0, 1], zmin, zmax, res, ``H`` = heights [y index, x index]) and helpers ``mesh()`` ->
    (vertices, faces), ``origin_table()`` -> (num_rows, num_cols, 3) Isaac's sub-terrain origins
    (rows = difficulty levels, columns = terrain types) and ``origins(n, seed)`` -> (n, 3) random
    sub-terrain origins. ``torch_device``: the device whose torch generator Isaac's random-grid boxes
    are drawn from ("cuda" = what Isaac does on an NVIDIA GPU; "cpu" = Isaac's code on a CPU-only
    machine). ``size``, ``res`` and ``border`` override ROUGH_TERRAINS_CFG's size, horizontal scale and
    border width (defaults are Isaac's).

    ``collision`` selects the collision surface (the height scan always reads the exact 0.1 m grid ``H``;
    see ``TERRAIN_COLLISION``): "hfield" = the 0.1 m heightfield of ``H`` (walls become 0.1 m ramps);
    "boxes" = Isaac's trimesh sub-terrains (pyramid stairs, inverted stairs, random-grid boxes) as the
    box geoms Isaac builds them from, over a heightfield lowered under those cells; "meshes" = the same
    boxes as 8-vertex mesh geoms (Isaac's ``trimesh.creation.box`` triangles); "boxes_local" = the same boxes, but
    each world holds only the ``BOX_WINDOW_SLOTS`` boxes within ``BOX_WINDOW_HALF`` of its robot (``BoxWindow``; the
    same geometry wherever the robot can touch it, at a broadphase cost independent of the box count); "hfield_fine" = a single
    heightfield of Isaac's exact top surface at ``fine_res`` (0.05 or 0.025 m); "boxes_fine" = "boxes" over a
    ``fine_res`` heightfield of the exact surface (the height-field cells' slope-threshold walls too)."""
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
    if collision not in TERRAIN_COLLISION:
        raise ValueError(f"collision must be one of {TERRAIN_COLLISION}, not {collision!r}")
    hf["collision"] = collision
    if collision in ("boxes", "meshes", "boxes_local"):
        hf["boxes"] = terrain_boxes(gen)
        hf["collision_hfield"] = _hfield_fields(lowered_under_boxes(Hb, gen), res)
    elif collision == "boxes_fine":
        hf["boxes"] = terrain_boxes(gen)
        S = np.ascontiguousarray(exact_surface(gen, fine_res).T).astype(np.float64)
        hf["collision_hfield"] = _hfield_fields(lowered_under_boxes(S, gen, k=int(round(res / fine_res))), fine_res)
    elif collision == "hfield_fine":
        hf["collision_hfield"] = _hfield_fields(np.ascontiguousarray(exact_surface(gen, fine_res).T).astype(np.float64), fine_res)
    return hf


# --------------------------------------------------------------------------------------------------
# collision surfaces for Isaac's vertical walls (docs/research/terrain_walls_2026-09-25.md)

TERRAIN_COLLISION = ("hfield", "boxes", "meshes", "hfield_fine", "boxes_fine", "boxes_local")
BOX_WINDOW_SLOTS = 96        # "boxes_local": box geoms per world (the boxes near the robot, refreshed every substep)
BOX_WINDOW_HALF = 1.2        # half width of the square window around the root body (feet stay within ~0.9 m of it)
TERRAIN_FRICTION = (0.8, 0.005, 0.0001)
BOX_HFIELD_DROP = 0.05       # heightfield inside box-built cells: this far below the cell's lowest box top


def _hfield_fields(Hb, res):
    """MuJoCo hfield fields for heights ``Hb`` [y index, x index] on a ``res`` grid centred at the origin."""
    zmin, zmax = float(Hb.min()), float(Hb.max())
    nrow, ncol = Hb.shape
    return {"nrow": nrow, "ncol": ncol, "res": res, "zmin": zmin, "zmax": zmax,
            "size": [(ncol - 1) * res / 2, (nrow - 1) * res / 2, max(zmax - zmin, 1e-3), max(-zmin, 0.0) + 0.05],
            "data": ((Hb - zmin) / max(zmax - zmin, 1e-6)).astype(np.float32)}


def terrain_boxes(gen):
    """World-frame boxes of Isaac's trimesh sub-terrains, as ``trimesh.creation.box`` builds them in
    ``mesh_terrains.py`` (Isaac concatenates these boxes into its terrain mesh): (N, 8) float64 rows
    ``cx, cy, cz, hx, hy, hz, row, col`` (centre, half extents; the top of a random-grid cell is its
    float32 noise value exactly). World frame = Isaac's (x along rows, y along columns, centred)."""
    cfg = gen.cfg
    out = []
    for s in gen.sub_terrains:
        if s.boxes is None:
            continue
        ox = s.row * cfg.size[0] - cfg.size[0] * cfg.num_rows * 0.5
        oy = s.col * cfg.size[1] - cfg.size[1] * cfg.num_cols * 0.5
        for b in s.boxes:
            (cx, cy, cz), (ex, ey, ez) = b[0], b[1]
            top = float(b[2]) if len(b) > 2 else ez * 0.5 + cz
            bot = cz - ez * 0.5
            out.append((cx + ox, cy + oy, 0.5 * (top + bot), 0.5 * ex, 0.5 * ey, 0.5 * (top - bot), s.row, s.col))
    return np.array(out, np.float64).reshape(-1, 8)


def lowered_under_boxes(Hb, gen, drop=BOX_HFIELD_DROP, k=1):
    """Heightfield [y, x] with the interior grid points of every box-built cell set ``drop`` below the
    cell's lowest top, so the boxes alone form the surface there (every such cell is tiled by its
    boxes; the cell's boundary lines keep their height, 0, which the border boxes cover). ``k``: grid points per
    0.1 m (a finer heightfield)."""
    H = Hb.copy()
    n, nb = gen.n_cell * k, gen.nb * k
    for s in gen.sub_terrains:
        if s.boxes is None:
            continue
        i0 = nb + s.col * n; j0 = nb + s.row * n            # MuJoCo layout: rows along y (columns), cols along x (rows)
        blk = H[i0:i0 + n + 1, j0:j0 + n + 1]
        blk[1:-1, 1:-1] = float(blk.min()) - drop
    return H


def exact_surface(gen, res):
    """Top surface of Isaac's whole terrain mesh on a ``res`` grid (0.1 / k m) in Isaac's frame [x, y]:
    trimesh cells from their boxes, height-field cells from Isaac's triangles after the slope-threshold
    vertex moves, the 20 m border at 0. With res = 0.025 every wall of ROUGH_TERRAINS_CFG lies on a grid
    line (stairs every 0.3 m from 1.0 m, random-grid boxes every 0.45 m from 0.175 m)."""
    from metalsim.learn.isaac_terrain import (HfCfg, boxes_surface, height_field_triangles, height_field_vertex_moves,
                                              top_surface_of_triangles)
    cfg = gen.cfg; hs = cfg.horizontal_scale
    k = int(round(hs / res)); assert abs(k * res - hs) < 1e-9, "res must divide the horizontal scale"
    n = gen.n_cell * k; nb = gen.nb * k
    S = np.full((cfg.num_rows * n + 2 * nb + 1, cfg.num_cols * n + 2 * nb + 1), -np.inf, np.float32)
    tri_cache = {}
    for s in gen.sub_terrains:
        if s.boxes is not None:
            surf = boxes_surface(s.boxes, n + 1, res)
        else:
            sc = cfg.sub_terrains[s.name]; assert isinstance(sc, HfCfg)
            h = s.heights; n0, n1 = h.shape
            dx, dy = height_field_vertex_moves(h, sc.horizontal_scale, sc.vertical_scale, sc.slope_threshold)
            ii, jj = np.meshgrid(np.arange(n0), np.arange(n1), indexing="ij")
            P = np.stack([((ii + dx) * k).ravel(), ((jj + dy) * k).ravel()], 1).astype(np.float64)
            Z = (h.flatten() * sc.vertical_scale).astype(np.float32).astype(np.float64)
            if (n0, n1) not in tri_cache:
                tri_cache[(n0, n1)] = height_field_triangles(n0, n1)
            surf = top_surface_of_triangles(P, Z, tri_cache[(n0, n1)], ((n0 - 1) * k + 1, (n1 - 1) * k + 1)).astype(np.float32)
        i0 = nb + s.row * n; j0 = nb + s.col * n
        blk = S[i0:i0 + n + 1, j0:j0 + n + 1]
        np.maximum(blk, surf, out=blk)
    ring = np.ones(S.shape, bool); ring[nb + 1:S.shape[0] - nb - 1, nb + 1:S.shape[1] - nb - 1] = False
    S[ring] = np.maximum(S[ring], np.float32(0.0))
    assert np.isfinite(S).all()
    return S


def collision_top(hf, x, y):
    """Top of the MuJoCo collision geometry that ``hf["collision"]`` builds, straight above world points (x, y): the
    heightfield prisms with MuJoCo's triangulation (MuJoCo Warp's hfield kernel and mjc_ConvexHField split each cell
    along (c, r)-(c+1, r+1), Isaac's diagonal) from the float32 data MuJoCo stores, and the highest terrain box above
    the point. Equals ``mj_ray`` on the compiled model (scripts/diagnostics/terrain_wall_fidelity.py checks it)."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    ch = hf.get("collision_hfield") or hf
    D = ch["data"].astype(np.float64) * ch["size"][2] + ch["zmin"]
    res = ch["res"]; nrow, ncol = D.shape
    u = np.clip((x + ch["size"][0]) / res, 0, ncol - 1 - 1e-9); v = np.clip((y + ch["size"][1]) / res, 0, nrow - 1 - 1e-9)
    c = np.floor(u).astype(int); r = np.floor(v).astype(int); fx = u - c; fy = v - r
    z00 = D[r, c]; z10 = D[r, c + 1]; z01 = D[r + 1, c]; z11 = D[r + 1, c + 1]
    z = np.where(fx >= fy, z00 + fx * (z10 - z00) + fy * (z11 - z10), z00 + fy * (z01 - z00) + fx * (z11 - z01))
    B = hf.get("boxes") if hf.get("collision") in ("boxes", "meshes", "boxes_fine", "boxes_local") else None
    if B is not None and len(x):
        top = np.full(len(x), -np.inf)
        order = np.argsort(x); xs = x[order]
        for b in B:
            lo, hi = np.searchsorted(xs, b[0] - b[3]), np.searchsorted(xs, b[0] + b[3], side="right")
            if hi > lo:
                idx = order[lo:hi]; sel = idx[np.abs(y[idx] - b[1]) <= b[4]]
                np.maximum.at(top, sel, b[2] + b[5])
        z = np.maximum(z, top)
    return z


def box_mesh(half):
    """Vertices and faces of ``trimesh.creation.box(extents=2 * half)`` (Isaac's box primitive): 8 corners,
    12 outward triangles."""
    v = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
                 np.float64) * np.asarray(half)[None]
    f = np.array([[1, 3, 0], [4, 1, 0], [0, 3, 2], [2, 4, 0], [1, 7, 3], [5, 1, 4], [5, 7, 1], [3, 7, 2], [6, 4, 2],
                  [2, 7, 6], [6, 5, 4], [7, 5, 6]], np.int32)
    return v, f


def add_terrain_geoms(spec, hfield, ground_geom):
    """Complete the rough-terrain collision surface on ``spec`` for ``hfield["collision"]``: "hfield" leaves
    the 0.1 m heightfield ``ground_geom`` untouched; otherwise its hfield asset is replaced by the collision
    heightfield (lowered under box cells, or the fine exact surface), and "boxes" / "meshes" add one world
    geom per Isaac terrain box (named ``tbox<i>``)."""
    import mujoco
    mode = hfield.get("collision", "hfield")
    if mode == "hfield":
        return
    ch = hfield["collision_hfield"]
    hf = next(h for h in spec.hfields if h.name == ground_geom.hfieldname)
    hf.nrow, hf.ncol = ch["nrow"], ch["ncol"]; hf.size = ch["size"]; hf.userdata = ch["data"].reshape(-1)
    ground_geom.pos = [0.0, 0.0, ch["zmin"]]
    if mode == "boxes_local":           # per-world slots, placed by BoxWindow (parked far below until then)
        for i in range(BOX_WINDOW_SLOTS):
            g = spec.worldbody.add_geom(); g.name = f"tslot{i}"; g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.pos = [0.0, 0.0, -1000.0]; g.size = [1e-3, 1e-3, 1e-3]; g.friction = list(ground_geom.friction)
        return
    if mode not in ("boxes", "meshes", "boxes_fine"):
        return
    for i, b in enumerate(hfield["boxes"]):
        g = spec.worldbody.add_geom(); g.name = f"tbox{i}"; g.pos = b[:3].tolist()
        g.friction = list(ground_geom.friction)
        if mode in ("boxes", "boxes_fine"):
            g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = b[3:6].tolist()
        else:
            v, f = box_mesh(b[3:6])
            m = spec.add_mesh(); m.name = f"tbox{i}"; m.uservert = v.reshape(-1).tolist(); m.userface = f.reshape(-1).tolist()
            g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = m.name


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


@wp.kernel
def height_scan_exact(xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33), body: int,
                      grid: wp.array2d(dtype=float), H: wp.array2d(dtype=float), res: float, size_x: float, size_y: float,
                      offset_z: float, blk: wp.array2d(dtype=int), F: wp.array3d(dtype=float), cell: float, x0c: float,
                      y0c: float, res_f: float, out: wp.array2d(dtype=float)):
    """``height_scan_grid`` with Isaac's exact heights off the grid on the box-built sub-terrains: there Isaac's
    top surface is constant on every ``res_f`` (0.025 m) cell (all walls lie on those grid lines), so the ray hit
    is the cell's value ``F[block, i, j]``; elsewhere the triangle interpolation of the 0.1 m grid (exact for
    Isaac's height-field meshes except at their slope-threshold walls)."""
    e, k = wp.tid()
    bp = xpos[e, body]
    R = xmat[e, body]
    yaw = wp.atan2(R[1, 0], R[0, 0])
    gx = grid[k, 0]; gy = grid[k, 1]
    x = bp[0] + wp.cos(yaw) * gx - wp.sin(yaw) * gy
    y = bp[1] + wp.sin(yaw) * gx + wp.cos(yaw) * gy
    r = int(wp.floor((x - x0c) / cell)); c = int(wp.floor((y - y0c) / cell))
    b = int(-1)
    if r >= 0 and r < blk.shape[0] and c >= 0 and c < blk.shape[1]:
        b = blk[r, c]
    if b >= 0:
        nf = F.shape[1]
        i = wp.clamp(int(wp.floor((x - x0c - float(r) * cell) / res_f)), 0, nf - 1)
        j = wp.clamp(int(wp.floor((y - y0c - float(c) * cell) / res_f)), 0, nf - 1)
        out[e, k] = bp[2] - F[b, i, j] - offset_z
        return
    nrow = H.shape[0]; ncol = H.shape[1]
    u = (x + size_x) / res; v = (y + size_y) / res
    u = wp.clamp(u, 0.0, float(ncol - 1) - 1e-4); v = wp.clamp(v, 0.0, float(nrow - 1) - 1e-4)
    j2 = int(wp.floor(u)); i2 = int(wp.floor(v))
    fx = u - float(j2); fy = v - float(i2)
    z00 = H[i2, j2]; z10 = H[i2, j2 + 1]; z11 = H[i2 + 1, j2 + 1]; z01 = H[i2 + 1, j2]
    if fx >= fy:
        z = z00 + fx * (z10 - z00) + fy * (z11 - z10)
    else:
        z = z00 + fy * (z01 - z00) + fx * (z11 - z01)
    out[e, k] = bp[2] - z - offset_z


def box_cell_tops(gen, res_f=0.025):
    """Isaac's exact top surface on the box-built sub-terrains as cell values: ``blk`` (num_rows, num_cols) int32 block
    index (-1 for height-field sub-terrains) and ``F`` (nblocks, n, n) float32, F[b, i, j] = the top at the centre of
    local cell (i, j) of size ``res_f`` (Isaac frame: i along x). With res_f = 0.025 every wall of ROUGH_TERRAINS_CFG is
    a cell boundary, so the value holds on the whole cell."""
    from metalsim.learn.isaac_terrain import boxes_surface
    cfg = gen.cfg; n = int(round(cfg.size[0] / res_f))
    blk = -np.ones((cfg.num_rows, cfg.num_cols), np.int32); F = []
    for s in gen.sub_terrains:
        if s.boxes is None:
            continue
        sh = [(((b[0][0] - 0.5 * res_f), (b[0][1] - 0.5 * res_f), b[0][2]),) + tuple(b[1:]) for b in s.boxes]
        blk[s.row, s.col] = len(F)
        F.append(boxes_surface(sh, n + 1, res_f)[:n, :n])
    return blk, np.array(F, np.float32).reshape(-1, n, n)


def scan_grid(ordering="xy", size=(1.6, 1.0), res=0.1):
    """(187, 2) ray offsets of Isaac's GridPatternCfg. ``ordering="xy"`` is Isaac's (GridPatternCfg default:
    torch.meshgrid(x, y, indexing="xy") flattened, x varies fastest; verified against Isaac Sim's recorded
    height scan, runs/parity/isaac/rough/play_rough). ``"ij"`` (y fastest) is the order MetalSim used before
    2026-09-25; policies trained with it need ``scan_perm`` to run on Isaac-ordered scans."""
    xs = np.arange(-size[0] / 2, size[0] / 2 + 1e-6, res); ys = np.arange(-size[1] / 2, size[1] / 2 + 1e-6, res)
    gx, gy = np.meshgrid(xs, ys, indexing=ordering)
    return np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)


def scan_perm(src="xy", dst="ij", size=(1.6, 1.0), res=0.1):
    """Index array p with scan_dst = scan_src[..., p] (reorders a height scan between ray orderings)."""
    a = np.round(scan_grid(src, size, res) / res).astype(int); b = np.round(scan_grid(dst, size, res) / res).astype(int)
    idx = {tuple(v): i for i, v in enumerate(a)}
    return np.array([idx[tuple(v)] for v in b])


class HeightScanner:
    """Isaac's height scan (1.6 x 1.0 m grid, 0.1 m, 187 rays, yaw-aligned, ``body_z - hit_z - 0.5``)
    evaluated as a Warp kernel on the simulation queue (capturable into rollout graphs); the terrain is
    a heightfield, so the ray hit is the triangle interpolation of the grid. ``metalsim.sensors.raytrace``
    remains the path for scanning arbitrary meshes."""

    def __init__(self, task, hfield, body="torso_link", size=(1.6, 1.0), res=0.1, offset_z=0.5, device="metal:0", ordering="xy",
                 surface="grid"):
        """``surface``: "grid" = triangle interpolation of the exact 0.1 m grid (walls between grid points read as
        0.1 m ramps; the default); "exact" = Isaac's exact top on the box-built sub-terrains (``height_scan_exact``)."""
        import mujoco
        self.task, self.device = task, device
        self.body = mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_BODY, body)
        self.ordering = ordering          # "xy" = Isaac's ray order; "ij" = MetalSim's order before 2026-09-25
        self.grid = scan_grid(ordering, size, res)   # (187, 2)
        self.n_rays = len(self.grid)
        self.offset_z = offset_z
        self.grid_wp = wp.array(self.grid, dtype=float, device=device)
        self.H = wp.array(hfield["H"].astype(np.float32), dtype=float, device=device)
        self.res = float(hfield["res"]); self.size_x = float(hfield["size"][0]); self.size_y = float(hfield["size"][1])
        if surface not in ("grid", "exact"):
            raise ValueError(f"surface must be 'grid' or 'exact', not {surface!r}")
        self.surface = surface
        if surface == "exact":
            gen = hfield["generator"]; cfg = gen.cfg
            blk, F = box_cell_tops(gen)
            self.blk = wp.array(blk, dtype=int, device=device); self.F = wp.array(F, dtype=float, device=device)
            self.cell = float(cfg.size[0]); self.res_f = 0.025
            self.x0c = -cfg.size[0] * cfg.num_rows * 0.5; self.y0c = -cfg.size[1] * cfg.num_cols * 0.5

    def launch(self, step_idx=None):
        sim = self.task.sim
        if self.surface == "exact":
            wp.launch(height_scan_exact, dim=(self.task.n, self.n_rays),
                      inputs=[sim.d.xpos, sim.d.xmat, self.body, self.grid_wp, self.H, self.res, self.size_x, self.size_y,
                              self.offset_z, self.blk, self.F, self.cell, self.x0c, self.y0c, self.res_f, self.task.height_scan],
                      device=self.device)
            return
        wp.launch(height_scan_grid, dim=(self.task.n, self.n_rays),
                  inputs=[sim.d.xpos, sim.d.xmat, self.body, self.grid_wp, self.H, self.res, self.size_x, self.size_y,
                          self.offset_z, self.task.height_scan], device=self.device)


@wp.kernel
def _box_window(qpos: wp.array2d(dtype=float), centre: wp.array(dtype=wp.vec3), half: wp.array(dtype=wp.vec3),
                cell_start: wp.array(dtype=int), num_rows: int, num_cols: int, cell: float, x0c: float, y0c: float,
                W: float, slot0: int, K: int, geom_size: wp.array2d(dtype=wp.vec3), geom_aabb: wp.array3d(dtype=wp.vec3),
                geom_rbound: wp.array2d(dtype=float), geom_xpos: wp.array2d(dtype=wp.vec3), overflow: wp.array(dtype=int)):
    """Per world: the terrain boxes whose footprint meets the square [x +- W] x [y +- W] around the root body go into the
    world's K box slots (geom_size / geom_aabb / geom_rbound per world, geom_xpos of the static slot geoms); the rest
    are parked 1 km below the terrain."""
    w = wp.tid()
    x = qpos[w, 0]; y = qpos[w, 1]
    r0 = wp.max(int(wp.floor((x - W - x0c) / cell)), 0); r1 = wp.min(int(wp.floor((x + W - x0c) / cell)), num_rows - 1)
    c0 = wp.max(int(wp.floor((y - W - y0c) / cell)), 0); c1 = wp.min(int(wp.floor((y + W - y0c) / cell)), num_cols - 1)
    k = int(0)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            ci = r * num_cols + c
            for b in range(cell_start[ci], cell_start[ci + 1]):
                p = centre[b]; h = half[b]
                if wp.abs(p[0] - x) <= W + h[0] and wp.abs(p[1] - y) <= W + h[1]:
                    if k < K:
                        g = slot0 + k
                        geom_size[w, g] = h
                        geom_aabb[w, g, 0] = wp.vec3(0.0, 0.0, 0.0)
                        geom_aabb[w, g, 1] = h
                        geom_rbound[w, g] = wp.length(h)
                        geom_xpos[w, g] = p
                    else:
                        wp.atomic_add(overflow, 0, 1)
                    k += 1
    for j in range(wp.min(k, K), K):
        g = slot0 + j
        geom_size[w, g] = wp.vec3(1.0e-3, 1.0e-3, 1.0e-3)
        geom_aabb[w, g, 0] = wp.vec3(0.0, 0.0, 0.0)
        geom_aabb[w, g, 1] = wp.vec3(1.0e-3, 1.0e-3, 1.0e-3)
        geom_rbound[w, g] = 1.7e-3
        geom_xpos[w, g] = wp.vec3(0.0, 0.0, -1000.0)
    wp.atomic_max(overflow, 1, k)


class BoxWindow:
    """``terrain_collision="boxes_local"``: fills each world's box slots with the terrain boxes around its robot. Needs a
    BatchSim built with ``per_world_fields=BoxWindow.FIELDS``. ``launch()`` reads the root position from qpos, so it is
    valid right after a reset; the task runs it before every control step and after every physics substep.
    ``overflow`` = [boxes dropped for lack of slots (must stay 0), most boxes any world needed]."""
    FIELDS = ("geom_size", "geom_aabb", "geom_rbound")

    def __init__(self, sim, hfield, device="metal:0", half_width=BOX_WINDOW_HALF):
        import mujoco
        m = sim.model if hasattr(sim, "model") else None
        B = hfield["boxes"]; gen = hfield["generator"]; cfg = gen.cfg
        order = np.lexsort((B[:, 7], B[:, 6])); B = B[order]
        ci = (B[:, 6] * cfg.num_cols + B[:, 7]).astype(np.int64)
        start = np.searchsorted(ci, np.arange(cfg.num_rows * cfg.num_cols + 1)).astype(np.int32)
        self.sim, self.device = sim, device
        self.centre = wp.array(B[:, :3].astype(np.float32), dtype=wp.vec3, device=device)
        self.half = wp.array(B[:, 3:6].astype(np.float32), dtype=wp.vec3, device=device)
        self.cell_start = wp.array(start, dtype=int, device=device)
        self.num_rows, self.num_cols, self.cell = cfg.num_rows, cfg.num_cols, float(cfg.size[0])
        self.x0c = -cfg.size[0] * cfg.num_rows * 0.5; self.y0c = -cfg.size[1] * cfg.num_cols * 0.5
        self.W = float(half_width)
        model = sim._wmodel if hasattr(sim, "_wmodel") else m
        self.slot0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tslot0")
        self.K = BOX_WINDOW_SLOTS
        self.overflow = wp.zeros(2, dtype=int, device=device)
        for f in self.FIELDS:
            assert getattr(sim.m, f).shape[0] == sim.d.nworld, f"BatchSim needs per_world_fields {self.FIELDS}"

    def launch(self):
        m, d = self.sim.m, self.sim.d
        wp.launch(_box_window, dim=d.nworld,
                  inputs=[d.qpos, self.centre, self.half, self.cell_start, self.num_rows, self.num_cols, self.cell, self.x0c,
                          self.y0c, self.W, self.slot0, self.K, m.geom_size, m.geom_aabb, m.geom_rbound, d.geom_xpos, self.overflow],
                  device=self.device)
