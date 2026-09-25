"""Isaac Lab's terrain generator (v2.3.2), ported to plain numpy so MetalSim builds the same terrain.

Port of ``isaaclab.terrains.TerrainGenerator`` with ``ROUGH_TERRAINS_CFG`` (the terrain of
Isaac-Velocity-Rough-G1-v0) and the sub-terrain functions it uses, from the reference copies in
``assets/isaac/terrains/`` (fetched from github.com/isaac-sim/IsaacLab at tag v2.3.2):

* height-field terrains (``height_field/hf_terrains.py``): random_uniform, pyramid_sloped (+ inverted),
  pyramid_stairs (+ inverted), discrete_obstacles, with ``height_field_to_mesh`` (border pixels, origin
  height = max over the central 2 x 2 m) and ``convert_height_field_to_mesh`` (slope-threshold vertex
  moves that turn steep cells into vertical walls);
* trimesh terrains (``trimesh/mesh_terrains.py``): pyramid_stairs, inverted_pyramid_stairs,
  random_grid, as the axis-aligned boxes Isaac's trimesh code builds;
* the generator: curriculum column assignment from the proportions, difficulty
  ``(row + U(0,1)) / num_rows`` from ``np.random.default_rng(seed)`` drawn per (column, row) in Isaac's
  loop order, sub-terrain placement, the 20 m border, centring, and env origins.

Random streams, as Isaac consumes them (``ROUGH_TERRAINS_CFG.seed`` is None, so the generator falls back
to the global numpy seed set by the env's ``configure_seed(seed)``):

* generator ``np_rng = np.random.default_rng(np.random.get_state()[1][0])`` = ``default_rng(seed)``;
* ``random_uniform_terrain`` draws from the *global* ``np.random`` (legacy MT19937 seeded with ``seed``);
* ``random_grid_terrain`` draws its box heights with ``torch.Tensor.uniform_`` on ``cuda`` when available
  (Isaac always runs on an NVIDIA GPU): torch's Philox4x32-10 CUDA generator seeded with ``seed``,
  reproduced here in numpy (``torch_device="cuda"``, default); ``torch_device="cpu"`` uses torch's CPU
  generator, which is what Isaac's code produces on a machine without CUDA (and what the tests run).

The terrain is returned as the exact top surface of Isaac's terrain mesh sampled on the
``horizontal_scale`` grid (0.1 m) in Isaac's world frame (x along rows / difficulty, y along columns,
centred at the origin). At grid points the heights equal a straight-down ray on Isaac's mesh (at a
vertical wall that lies on a grid line, the top of the wall). Between grid points MuJoCo's heightfield
and the height scan interpolate the grid, so vertical walls (stair risers, box edges, slope-threshold
walls) become ramps one grid cell wide; that is the one part a heightfield cannot reproduce.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from fractions import Fraction

import numpy as np

# --------------------------------------------------------------------------------------------------
# configuration (the fields of Isaac's cfg classes that the ported functions read)


@dataclass
class HfCfg:
    function: str
    proportion: float = 1.0
    size: tuple = (8.0, 8.0)
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    # random_uniform
    noise_range: tuple | None = None
    noise_step: float | None = None
    downsampled_scale: float | None = None
    # pyramid sloped / stairs
    slope_range: tuple | None = None
    step_height_range: tuple | None = None
    step_width: float | None = None
    platform_width: float = 1.0
    inverted: bool = False
    # discrete obstacles
    obstacle_height_mode: str = "choice"
    obstacle_width_range: tuple | None = None
    obstacle_height_range: tuple | None = None
    num_obstacles: int | None = None


@dataclass
class MeshCfg:
    function: str
    proportion: float = 1.0
    size: tuple = (8.0, 8.0)
    border_width: float = 0.0
    step_height_range: tuple | None = None
    step_width: float | None = None
    platform_width: float = 1.0
    holes: bool = False
    grid_width: float | None = None
    grid_height_range: tuple | None = None


@dataclass
class GeneratorCfg:
    sub_terrains: dict
    size: tuple = (8.0, 8.0)
    border_width: float = 0.0
    border_height: float = 1.0
    num_rows: int = 1
    num_cols: int = 1
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = 0.75
    curriculum: bool = False
    difficulty_range: tuple = (0.0, 1.0)
    seed: int | None = None


def rough_terrains_cfg() -> GeneratorCfg:
    """``isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG`` (v2.3.2), key order preserved (it sets the
    column assignment)."""
    return GeneratorCfg(
        size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20, horizontal_scale=0.1, vertical_scale=0.005,
        slope_threshold=0.75,
        sub_terrains={
            "pyramid_stairs": MeshCfg("mesh_pyramid_stairs", proportion=0.2, step_height_range=(0.05, 0.23),
                                      step_width=0.3, platform_width=3.0, border_width=1.0, holes=False),
            "pyramid_stairs_inv": MeshCfg("mesh_inverted_pyramid_stairs", proportion=0.2, step_height_range=(0.05, 0.23),
                                          step_width=0.3, platform_width=3.0, border_width=1.0, holes=False),
            "boxes": MeshCfg("mesh_random_grid", proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2),
                             platform_width=2.0),
            "random_rough": HfCfg("hf_random_uniform", proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02,
                                  border_width=0.25),
            "hf_pyramid_slope": HfCfg("hf_pyramid_sloped", proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0,
                                      border_width=0.25),
            "hf_pyramid_slope_inv": HfCfg("hf_pyramid_sloped", proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0,
                                          border_width=0.25, inverted=True),
        })


# --------------------------------------------------------------------------------------------------
# random streams


class TorchUniform:
    """``torch.Tensor.uniform_`` on a fresh 1-D float32 tensor, as Isaac's random_grid calls it
    (``h_noise[:, 2].uniform_(-h, h)`` with ``h_noise`` of shape (num_boxes, 3)).

    ``device="cpu"``: torch's own CPU generator seeded with ``torch.manual_seed(seed)``.
    ``device="cuda"``: torch's CUDA generator reproduced bit-for-bit: ``curand_init(seed, idx, offset)``
    Philox4x32-10, ``curand_uniform4`` (``x * 2^-32 + 2^-33``), element ``i`` served by thread ``i``
    (grid-stride kernel, 256-thread blocks, unroll 4), the Philox offset advancing by 4 per call for
    fewer than 256*grid*4 elements, and ``value = fma(rand, to - from, from)`` in float32 (nvcc
    contracts the expression; ``value == to`` maps to ``from``)."""

    def __init__(self, seed: int, device: str = "cuda"):
        self.device = device
        self.seed = int(seed)
        self.offset = 0
        if device == "cpu":
            import torch
            self.gen = torch.Generator().manual_seed(self.seed)

    def __call__(self, n: int, lo: float, hi: float) -> np.ndarray:
        if self.device == "cpu":
            import torch
            h = torch.zeros((n, 3))
            h[:, 2].uniform_(lo, hi, generator=self.gen)
            return h[:, 2].numpy().copy()
        block, unroll = 256, 4
        grid = (n + block - 1) // block        # smaller than any GPU's resident-block cap for these sizes
        assert grid <= 48, "grid capped by the device's SM count: not reproduced"
        counter_offset = ((n - 1) // (block * grid * unroll) + 1) * 4
        assert counter_offset == 4, "more than one curand4 call per thread: not reproduced"
        idx = np.arange(n, dtype=np.uint64)
        ctr = [np.full(n, self.offset // 4, np.uint32), np.zeros(n, np.uint32),
               (idx & 0xFFFFFFFF).astype(np.uint32), (idx >> np.uint64(32)).astype(np.uint32)]
        x = philox4x32_10(ctr, (self.seed & 0xFFFFFFFF, (self.seed >> 32) & 0xFFFFFFFF))[0]
        self.offset += counter_offset
        rand = x.astype(np.float32) * np.float32(2.0 ** -32) + np.float32(2.0 ** -33)   # exact (power-of-2 scale)
        frm, to = np.float32(lo), np.float32(hi)
        rng = np.float32(to - frm)
        out = np.array([_fma_f32(r, rng, frm) for r in rand], np.float32)
        out[out == to] = frm
        return out


def _round_f32(q: Fraction) -> np.float32:
    """Round an exact rational to the nearest float32 (ties to even)."""
    f = np.float32(float(q))
    best = f
    for c in (np.nextafter(f, np.float32(-np.inf)), np.nextafter(f, np.float32(np.inf))):
        dc, db = abs(Fraction(float(c)) - q), abs(Fraction(float(best)) - q)
        if dc < db or (dc == db and (int(c.view(np.uint32)) & 1) == 0):
            best = c
    return np.float32(best)


def _fma_f32(a, b, c) -> np.float32:
    return _round_f32(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)))


def philox4x32_10(ctr, key):
    """Random123 / cuRAND Philox4x32-10 on uint32 numpy arrays: ``ctr`` 4 arrays, ``key`` 2 ints."""
    M0, M1, W0, W1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57), 0x9E3779B9, 0xBB67AE85
    c0, c1, c2, c3 = (np.asarray(c, np.uint32).astype(np.uint64) for c in ctr)
    k0, k1 = int(key[0]) & 0xFFFFFFFF, int(key[1]) & 0xFFFFFFFF
    mask = np.uint64(0xFFFFFFFF)
    for r in range(10):
        p0 = M0 * c0; p1 = M1 * c2
        hi0, lo0 = p0 >> np.uint64(32), p0 & mask
        hi1, lo1 = p1 >> np.uint64(32), p1 & mask
        c0, c1, c2, c3 = hi1 ^ c1 ^ np.uint64(k0), lo1, hi0 ^ c3 ^ np.uint64(k1), lo0
        k0 = (k0 + W0) & 0xFFFFFFFF; k1 = (k1 + W1) & 0xFFFFFFFF
    return [c.astype(np.uint32) for c in (c0, c1, c2, c3)]


# --------------------------------------------------------------------------------------------------
# height-field terrains (hf_terrains.py); ``rs`` replaces Isaac's global ``np.random``


def hf_random_uniform(difficulty, cfg: HfCfg, rs):
    import scipy.interpolate as interpolate
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError("Downsampled scale must be larger than or equal to the horizontal scale")
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)
    height_range = np.arange(height_min, height_max + height_step, height_step)
    height_field_downsampled = rs.choice(height_range, size=(width_downsampled, length_downsampled))
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    return np.rint(z_upsampled).astype(np.int16)


def hf_pyramid_sloped(difficulty, cfg: HfCfg, rs=None):
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)
    hf_raw = height_max * xx * yy
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))
    return np.rint(hf_raw).astype(np.int16)


def hf_pyramid_stairs(difficulty, cfg: HfCfg, rs=None):
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.inverted:
        step_height *= -1
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    step_width = int(cfg.step_width / cfg.horizontal_scale)
    step_height = int(step_height / cfg.vertical_scale)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    hf_raw = np.zeros((width_pixels, length_pixels))
    current_step_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        start_x += step_width; stop_x -= step_width
        start_y += step_width; stop_y -= step_width
        current_step_height += step_height
        hf_raw[start_x:stop_x, start_y:stop_y] = current_step_height
    return np.rint(hf_raw).astype(np.int16)


def hf_discrete_obstacles(difficulty, cfg: HfCfg, rs):
    obs_height = cfg.obstacle_height_range[0] + difficulty * (cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0])
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    obs_height = int(obs_height / cfg.vertical_scale)
    obs_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(0, length_pixels, 4)
    hf_raw = np.zeros((width_pixels, length_pixels))
    for _ in range(cfg.num_obstacles):
        if cfg.obstacle_height_mode == "choice":
            height = rs.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        else:
            raise ValueError(f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'.")
        width = int(rs.choice(obs_width_range))
        length = int(rs.choice(obs_length_range))
        x_start = int(rs.choice(obs_x_range))
        y_start = int(rs.choice(obs_y_range))
        if x_start + width > width_pixels:
            x_start = width_pixels - width
        if y_start + length > length_pixels:
            y_start = length_pixels - length
        hf_raw[x_start: x_start + width, y_start: y_start + length] = height
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    return np.rint(hf_raw).astype(np.int16)


HF_FUNCTIONS = {"hf_random_uniform": hf_random_uniform, "hf_pyramid_sloped": hf_pyramid_sloped,
                "hf_pyramid_stairs": hf_pyramid_stairs, "hf_discrete_obstacles": hf_discrete_obstacles}


def height_field_terrain(difficulty, cfg: HfCfg, rs):
    """``height_field_to_mesh``'s wrapper: returns ``(heights int16 (W+1, L+1) incl. border, origin)``."""
    if cfg.border_width > 0 and cfg.border_width < cfg.horizontal_scale:
        raise ValueError("The border width must be greater than or equal to the horizontal scale.")
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
    border_pixels = int(cfg.border_width / cfg.horizontal_scale) + 1
    heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    sub_terrain_size = [width_pixels - 2 * border_pixels, length_pixels - 2 * border_pixels]
    sub_terrain_size = [dim * cfg.horizontal_scale for dim in sub_terrain_size]
    terrain_size = copy.deepcopy(cfg.size)
    cfg.size = tuple(sub_terrain_size)
    z_gen = HF_FUNCTIONS[cfg.function](difficulty, cfg, rs)
    heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = z_gen
    cfg.size = terrain_size
    x1 = int((cfg.size[0] * 0.5 - 1) / cfg.horizontal_scale)
    x2 = int((cfg.size[0] * 0.5 + 1) / cfg.horizontal_scale)
    y1 = int((cfg.size[1] * 0.5 - 1) / cfg.horizontal_scale)
    y2 = int((cfg.size[1] * 0.5 + 1) / cfg.horizontal_scale)
    origin_z = np.max(heights[x1:x2, y1:y2]) * cfg.vertical_scale
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_z])
    return heights, origin


def height_field_vertex_moves(height_field, horizontal_scale, vertical_scale, slope_threshold):
    """``convert_height_field_to_mesh``'s slope-threshold correction as integer vertex moves (dx, dy)
    in grid cells: vertex (i, j) of Isaac's mesh sits at ((i + dx) * hs, (j + dy) * hs)."""
    num_rows, num_cols = height_field.shape
    hf = height_field.copy()
    dx = np.zeros((num_rows, num_cols)); dy = np.zeros((num_rows, num_cols))
    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols)); move_y = np.zeros((num_rows, num_cols)); move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        move_corners[1:num_rows, 1:num_cols] -= hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        dx = move_x + move_corners * (move_x == 0)
        dy = move_y + move_corners * (move_y == 0)
    return dx.astype(np.int64), dy.astype(np.int64)


def height_field_triangles(num_rows, num_cols):
    """Isaac's triangle indices for a (num_rows, num_cols) height field: (i, j) -> (i+1, j+1) diagonal."""
    tris = []
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1; ind2 = ind0 + num_cols; ind3 = ind2 + 1
        tris.append(np.stack([np.stack([ind0, ind3, ind1], 1), np.stack([ind0, ind2, ind3], 1)], 1).reshape(-1, 3))
    return np.concatenate(tris)


def top_surface_of_triangles(P, Z, tris, shape, tol=1e-9):
    """Highest point of a triangle soup straight above each integer grid point (downward ray from above;
    a point on a shared edge or vertex takes the highest triangle there). ``P`` (V, 2) vertex positions in
    grid units, ``Z`` (V,) heights; returns (shape) float64 with -inf where nothing is hit."""
    out = np.full(shape, -np.inf)
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    keep = np.abs(area) > 1e-12                                # vertical faces never face a vertical ray
    tris, a, b, c, area = tris[keep], a[keep], b[keep], c[keep], area[keep]
    lo = np.ceil(np.minimum(np.minimum(a, b), c) - 1e-6).astype(np.int64)
    hi = np.floor(np.maximum(np.maximum(a, b), c) + 1e-6).astype(np.int64)
    lo = np.maximum(lo, 0); hi = np.minimum(hi, np.array(shape) - 1)
    span = np.maximum(hi - lo + 1, 0)
    za, zb, zc = Z[tris[:, 0]], Z[tris[:, 1]], Z[tris[:, 2]]

    def accumulate(sel, ox, oy):
        px = lo[sel, 0] + ox; py = lo[sel, 1] + oy
        ok = (px <= hi[sel, 0]) & (py <= hi[sel, 1])
        A, B, C, ar = a[sel], b[sel], c[sel], area[sel]
        wa = ((B[:, 0] - px) * (C[:, 1] - py) - (B[:, 1] - py) * (C[:, 0] - px)) / ar
        wb = ((C[:, 0] - px) * (A[:, 1] - py) - (C[:, 1] - py) * (A[:, 0] - px)) / ar
        wc = ((A[:, 0] - px) * (B[:, 1] - py) - (A[:, 1] - py) * (B[:, 0] - px)) / ar
        ok &= (wa >= -tol) & (wb >= -tol) & (wc >= -tol)
        z = wa * za[sel] + wb * zb[sel] + wc * zc[sel]
        # exact vertex heights where the point is a vertex (the weights are then exactly 1, 0, 0)
        np.maximum.at(out, (px[ok], py[ok]), z[ok])

    small = (span[:, 0] <= 6) & (span[:, 1] <= 6)
    idx = np.nonzero(small)[0]
    for ox in range(6):
        for oy in range(6):
            accumulate(idx, ox, oy)
    for t in np.nonzero(~small)[0]:                           # large faces (box tops, borders): one by one
        gx, gy = np.meshgrid(np.arange(span[t, 0]), np.arange(span[t, 1]), indexing="ij")
        sel = np.full(gx.size, t)
        accumulate(sel, gx.ravel(), gy.ravel())
    return out


def height_field_surface(heights, cfg: HfCfg):
    """Top surface of Isaac's mesh for this height field at its own grid points (float32)."""
    dx, dy = height_field_vertex_moves(heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold)
    n0, n1 = heights.shape
    ii, jj = np.meshgrid(np.arange(n0), np.arange(n1), indexing="ij")
    P = np.stack([(ii + dx).ravel(), (jj + dy).ravel()], 1).astype(np.float64)
    Z = (heights.flatten() * cfg.vertical_scale).astype(np.float32).astype(np.float64)   # Isaac's float32 vertices
    if not dx.any() and not dy.any():
        return Z.reshape(n0, n1).astype(np.float32)
    s = top_surface_of_triangles(P, Z, height_field_triangles(n0, n1), (n0, n1))
    return s.astype(np.float32)


# --------------------------------------------------------------------------------------------------
# trimesh terrains (mesh_terrains.py) as axis-aligned boxes (centre, extents), in Isaac's float order


def _make_border(size, inner_size, height, position):
    thickness_x = (size[0] - inner_size[0]) / 2.0
    thickness_y = (size[1] - inner_size[1]) / 2.0
    dims = (size[0], thickness_y, height)
    out = [((position[0], position[1] + inner_size[1] / 2.0 + thickness_y / 2.0, position[2]), dims),
           ((position[0], position[1] - inner_size[1] / 2.0 - thickness_y / 2.0, position[2]), dims)]
    dims = (thickness_x, inner_size[1], height)
    out += [((position[0] - inner_size[0] / 2.0 - thickness_x / 2.0, position[1], position[2]), dims),
            ((position[0] + inner_size[0] / 2.0 + thickness_x / 2.0, position[1], position[2]), dims)]
    return out


def mesh_pyramid_stairs(difficulty, cfg: MeshCfg, trand=None, inverted=False):
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))
    total_height = (num_steps + 1) * step_height
    boxes = []
    if cfg.border_width > 0.0 and not cfg.holes:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], (-0.5 * step_height) if inverted else (-step_height / 2)]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        boxes += _make_border(cfg.size, border_inner_size, step_height, border_center)
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    for k in range(num_steps):
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        if inverted:
            box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
            box_height = total_height - (k + 1) * step_height
        else:
            box_z = terrain_center[2] + k * step_height / 2.0
            box_height = (k + 2) * step_height
        box_offset = (k + 0.5) * cfg.step_width
        dims = (box_size[0], cfg.step_width, box_height)
        boxes.append(((terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z), dims))
        boxes.append(((terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z), dims))
        dims = (cfg.step_width, box_size[1], box_height) if cfg.holes else (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        boxes.append(((terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z), dims))
        boxes.append(((terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z), dims))
    if inverted:
        dims = (terrain_size[0] - 2 * num_steps * cfg.step_width, terrain_size[1] - 2 * num_steps * cfg.step_width, step_height)
        pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
        origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])
    else:
        dims = (terrain_size[0] - 2 * num_steps * cfg.step_width, terrain_size[1] - 2 * num_steps * cfg.step_width,
                (num_steps + 2) * step_height)
        pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_steps * step_height / 2)
        origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])
    boxes.append((pos, dims))
    return boxes, origin


def mesh_inverted_pyramid_stairs(difficulty, cfg: MeshCfg, trand=None):
    return mesh_pyramid_stairs(difficulty, cfg, trand, inverted=True)


def mesh_random_grid(difficulty, cfg: MeshCfg, trand):
    if cfg.size[0] != cfg.size[1]:
        raise ValueError(f"The terrain must be square. Received size: {cfg.size}.")
    if cfg.holes:
        raise NotImplementedError("random_grid with holes")
    grid_height = cfg.grid_height_range[0] + difficulty * (cfg.grid_height_range[1] - cfg.grid_height_range[0])
    num_boxes_x = int(cfg.size[0] / cfg.grid_width)
    num_boxes_y = int(cfg.size[1] / cfg.grid_width)
    terrain_height = 1.0
    border_width = cfg.size[0] - min(num_boxes_x, num_boxes_y) * cfg.grid_width
    if not border_width > 0:
        raise RuntimeError("Border width must be greater than 0! Adjust the parameter 'cfg.grid_width'.")
    boxes = _make_border(cfg.size, (cfg.size[0] - border_width, cfg.size[1] - border_width), terrain_height,
                         (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2))
    # grid cells: template box [0, w]^2 x [-1, 0], offset by float32(w * index + border / 2) (torch float32
    # arithmetic), top raised by the float32 noise drawn with torch's uniform_
    xx, yy = np.meshgrid(np.arange(num_boxes_x), np.arange(num_boxes_y), indexing="ij")
    off = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float32) * np.float32(cfg.grid_width) + np.float32(border_width / 2)
    noise = trand(num_boxes_x * num_boxes_y, -grid_height, grid_height)
    tmpl = 0.5 * cfg.grid_width
    for (ox, oy), h in zip(off.astype(np.float64), noise.astype(np.float64)):
        boxes.append(((tmpl + ox, tmpl + oy, (h - 1.0) / 2), (cfg.grid_width, cfg.grid_width, 0.0 + h - (-1.0)), h))
    dim = (cfg.platform_width, cfg.platform_width, terrain_height + grid_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2 + grid_height / 2)
    boxes.append((pos, dim))
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], grid_height])
    return boxes, origin


MESH_FUNCTIONS = {"mesh_pyramid_stairs": mesh_pyramid_stairs, "mesh_inverted_pyramid_stairs": mesh_inverted_pyramid_stairs,
                  "mesh_random_grid": mesh_random_grid}


def boxes_surface(boxes, n, hs, tol=1e-6):
    """Top surface of a union of axis-aligned boxes at the (n, n) grid points ``(i * hs, j * hs)``: the
    highest box top whose closed footprint contains the point (-inf outside all boxes). A box is
    ``(centre, extents)`` (top = centre_z + extents_z / 2, trimesh's vertex arithmetic) or
    ``(centre, extents, top)`` for random-grid cells whose top vertex is exactly the noise value."""
    g = np.arange(n) * hs
    out = np.full((n, n), -np.inf)
    for b in boxes:
        (cx, cy, cz), (ex, ey, ez) = b[0], b[1]
        top = b[2] if len(b) > 2 else (ez * 0.5 + cz)
        x0, x1 = cx - 0.5 * ex, cx + 0.5 * ex
        y0, y1 = cy - 0.5 * ey, cy + 0.5 * ey
        i = np.nonzero((g >= x0 - tol) & (g <= x1 + tol))[0]
        j = np.nonzero((g >= y0 - tol) & (g <= y1 + tol))[0]
        if len(i) and len(j):
            blk = out[i[0]:i[-1] + 1, j[0]:j[-1] + 1]
            np.maximum(blk, top, out=blk)
    return out.astype(np.float32)


# --------------------------------------------------------------------------------------------------
# generator


@dataclass
class SubTerrain:
    row: int
    col: int
    name: str
    difficulty: float
    origin: np.ndarray            # world frame, float64 (Isaac's terrain_origins entry)
    surface: np.ndarray           # (n+1, n+1) float32 top surface on the local grid
    heights: np.ndarray | None = None      # int16 height field (height-field terrains)
    boxes: list | None = None              # boxes (trimesh terrains)


def sub_terrain(difficulty, cfg, rs, trand, hs):
    """One sub-terrain, as ``TerrainGenerator._get_terrain_mesh`` builds it: ``(local surface, local
    origin (centred), heights, boxes)``."""
    cfg = copy.deepcopy(cfg)
    n = int(round(cfg.size[0] / hs)) + 1
    if isinstance(cfg, HfCfg):
        heights, origin = height_field_terrain(float(difficulty), cfg, rs)
        surf, boxes = height_field_surface(heights, cfg), None
    else:
        boxes, origin = MESH_FUNCTIONS[cfg.function](float(difficulty), cfg, trand)
        surf, heights = boxes_surface(boxes, n, hs), None
    origin = origin + np.array([-cfg.size[0] * 0.5, -cfg.size[1] * 0.5, 0.0])
    return surf, origin, heights, boxes


class TerrainGenerator:
    """``isaaclab.terrains.TerrainGenerator`` for height-field / box terrains, producing the top surface
    on the ``horizontal_scale`` grid instead of a trimesh."""

    def __init__(self, cfg: GeneratorCfg, seed: int = 0, torch_device: str = "cuda"):
        self.cfg = cfg = copy.deepcopy(cfg)
        for sub in cfg.sub_terrains.values():
            sub.size = cfg.size
            if isinstance(sub, HfCfg):
                sub.horizontal_scale = cfg.horizontal_scale
                sub.vertical_scale = cfg.vertical_scale
                sub.slope_threshold = cfg.slope_threshold
        # Isaac: cfg.seed if set, else np.random.get_state()[1][0] of the global stream, which is the env
        # seed right after configure_seed(seed) -> np.random.seed(seed)
        seed = cfg.seed if cfg.seed is not None else int(np.random.RandomState(seed).get_state()[1][0])
        self.seed = seed
        self.np_rng = np.random.default_rng(seed)
        self.rs = np.random.RandomState(seed)             # Isaac's global np.random after np.random.seed(seed)
        self.trand = TorchUniform(seed, torch_device)     # Isaac's torch generator after torch.manual_seed(seed)
        hs = cfg.horizontal_scale
        self.n_cell = int(round(cfg.size[0] / hs))
        assert abs(self.n_cell * hs - cfg.size[0]) < 1e-9 and abs(cfg.size[0] - cfg.size[1]) < 1e-12
        self.nb = int(round(cfg.border_width / hs))
        assert abs(self.nb * hs - cfg.border_width) < 1e-9
        self.terrain_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.sub_terrains: list[SubTerrain] = []
        nx = cfg.num_rows * self.n_cell + 2 * self.nb + 1
        ny = cfg.num_cols * self.n_cell + 2 * self.nb + 1
        self.surface = np.full((nx, ny), -np.inf, np.float32)     # Isaac frame: [x index, y index]
        if cfg.curriculum:
            self._generate_curriculum_terrains()
        else:
            self._generate_random_terrains()
        self._add_terrain_border()
        # centre the terrain: origins shifted by -(size * num / 2)
        self.terrain_origins += np.array([-cfg.size[0] * cfg.num_rows * 0.5, -cfg.size[1] * cfg.num_cols * 0.5, 0.0])
        self.x0 = -cfg.size[0] * cfg.num_rows * 0.5 - cfg.border_width
        self.y0 = -cfg.size[1] * cfg.num_cols * 0.5 - cfg.border_width

    def _proportions(self):
        p = np.array([s.proportion for s in self.cfg.sub_terrains.values()])
        return p / np.sum(p)

    def _generate_random_terrains(self):
        proportions = self._proportions()
        cfgs = list(self.cfg.sub_terrains.items())
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            sub_row, sub_col = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            sub_index = self.np_rng.choice(len(proportions), p=proportions)
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            self._add_sub_terrain(difficulty, cfgs[sub_index], int(sub_row), int(sub_col))

    def _generate_curriculum_terrains(self):
        proportions = self._proportions()
        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_indices.append(np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0]))
        sub_indices = np.array(sub_indices, dtype=np.int32)
        cfgs = list(self.cfg.sub_terrains.items())
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                self._add_sub_terrain(difficulty, cfgs[sub_indices[sub_col]], sub_row, sub_col)

    def _add_sub_terrain(self, difficulty, named_cfg, row, col):
        name, cfg = named_cfg
        surf, origin, heights, boxes = sub_terrain(difficulty, cfg, self.rs, self.trand, self.cfg.horizontal_scale)
        # Isaac: mesh translated by ((row + 0.5) * size_x, (col + 0.5) * size_y); origin likewise
        origin = origin + np.array([(row + 0.5) * self.cfg.size[0], (col + 0.5) * self.cfg.size[1], 0.0])
        self.terrain_origins[row, col] = origin
        i0 = self.nb + row * self.n_cell; j0 = self.nb + col * self.n_cell
        blk = self.surface[i0:i0 + self.n_cell + 1, j0:j0 + self.n_cell + 1]
        np.maximum(blk, surf, out=blk)           # neighbours share their boundary grid line: highest wins
        self.sub_terrains.append(SubTerrain(row, col, name, float(difficulty), origin, surf, heights, boxes))

    def _add_terrain_border(self):
        """Isaac's border: boxes of height ``border_height`` whose top is at z = 0 around the sub-terrains
        (its inner edge coincides with the outer sub-terrain boundary)."""
        nb, nx, ny = self.nb, *self.surface.shape
        ring = np.ones((nx, ny), bool)
        ring[nb + 1:nx - nb - 1, nb + 1:ny - nb - 1] = False
        top = np.float32(-self.cfg.border_height / 2 + abs(self.cfg.border_height) * 0.5)
        self.surface[ring] = np.maximum(self.surface[ring], top)
        assert np.isfinite(self.surface).all()


# --------------------------------------------------------------------------------------------------


def isaac_rough_terrain_generator(seed: int = 0, num_rows: int = 10, num_cols: int = 20, torch_device: str = "cuda",
                                  curriculum: bool = True, **overrides) -> TerrainGenerator:
    """Isaac-Velocity-Rough-G1-v0's terrain: ROUGH_TERRAINS_CFG with the curriculum on, generated for env
    seed ``seed``."""
    cfg = rough_terrains_cfg()
    cfg = replace(cfg, num_rows=num_rows, num_cols=num_cols, curriculum=curriculum, **overrides)
    return TerrainGenerator(cfg, seed=seed, torch_device=torch_device)
