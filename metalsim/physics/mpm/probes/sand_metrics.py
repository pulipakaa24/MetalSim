"""Measurements for the sand-fidelity probes (research_notes/sand_fidelity/PLAN_2026-10-14.md). Pure numpy, CPU only,
shared by ``sand_repose.py``, ``tilt_box.py``, ``bearing_fidelity.py``, ``sphere_drop.py`` and the batch evaluator
(``scripts/sand_fidelity/evaluate_batch_01.py``); tested in ``tests/test_sand_metrics.py``.

Conventions: z up, positions in metres, angles in degrees. Stress: the solver's ``mpm:particle_stress`` (a symmetric
3x3 tensor per particle). Only scale-free ratios of it are used (sqrt(J2) / (mu p), the Lode parameter), so its
units and sign convention do not matter: the sign is chosen so that the median pressure is compressive (positive).
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# free surface
# ---------------------------------------------------------------------------------------------------------------------


def column_tops(q: np.ndarray, col: float):
    """Highest particle per xy column of width ``col`` (``repose_lift.heap_metrics``' surface).
    Returns (column centres (C, 2), column top z (C,), column index of every particle (P,))."""
    ij = np.floor(q[:, :2] / col).astype(np.int64)
    key = ij[:, 0] * 1_000_003 + ij[:, 1]
    uniq, inv = np.unique(key, return_inverse=True)
    zmax = np.full(uniq.shape[0], -np.inf)
    np.maximum.at(zmax, inv, q[:, 2])
    first = np.zeros(uniq.shape[0], np.int64)
    first[inv[::-1]] = np.arange(q.shape[0])[::-1]
    cxy = (ij[first] + 0.5) * col
    return cxy, zmax, inv


def radial_profile(r: np.ndarray, z: np.ndarray, edges: np.ndarray, min_count: int = 2):
    """Mean column radius and mean column top per radial bin (bins with fewer than ``min_count`` columns dropped)."""
    rc, h = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() >= min_count:
            rc.append(float(np.mean(r[m])))
            h.append(float(np.mean(z[m])))
    return np.asarray(rc), np.asarray(h)


def heap_profiles(q: np.ndarray, particle_volume: float, sectors: int = 8, bins: int = 30) -> dict:
    """Per-sector radial profiles h(r) of a heap centred at the origin, as ``repose_lift.heap_metrics`` builds them."""
    col = 2.0 * particle_volume ** (1.0 / 3.0)
    cxy, z, _ = column_tops(q, col)
    r = np.hypot(cxy[:, 0], cxy[:, 1])
    th = np.arctan2(cxy[:, 1], cxy[:, 0])
    edges = np.linspace(0.0, np.quantile(r, 0.995), bins + 1)
    rc, h = radial_profile(r, z, edges)
    out = {"edges": edges.tolist(), "all": {"r": rc.tolist(), "h": h.tolist()}, "sectors": []}
    for s in range(sectors):
        lo = -math.pi + 2 * math.pi * s / sectors
        sel = (th >= lo) & (th < lo + 2 * math.pi / sectors)
        rs, hs = radial_profile(r[sel], z[sel], edges)
        out["sectors"].append({"r": rs.tolist(), "h": hs.tolist()})
    return out


HEIGHT_BANDS = ((0.05, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.95))


def flank_angles_by_height(profiles: dict, voxel: float, bands=HEIGHT_BANDS) -> dict:
    """Flank angle [deg] fitted separately in bands of height (fractions of the heap's maximum height), per sector,
    then averaged over sectors. A convex heap (rounded crest, steeper lower flank) gives larger angles in the low
    bands. Also the steepest 3-bin local slope of the mean profile. Heights are also given in cells (``voxel``)."""
    h_all = np.asarray(profiles["all"]["h"])
    hmax = float(h_all.max())
    res = {"h_max": hmax, "h_max_cells": hmax / voxel, "bands": []}
    for lo, hi in bands:
        angs = []
        for sec in profiles["sectors"]:
            rs, hs = np.asarray(sec["r"]), np.asarray(sec["h"])
            # descending flank only: outside the radius of the sector's highest bin
            if len(rs) < 3:
                continue
            outer = rs >= rs[int(np.argmax(hs))]
            m = outer & (hs >= lo * hmax) & (hs < hi * hmax)
            if m.sum() >= 3:
                slope, _ = np.polyfit(rs[m], hs[m], 1)
                angs.append(math.degrees(math.atan(-slope)))
        res["bands"].append({"band": [lo, hi], "cells": [lo * hmax / voxel, hi * hmax / voxel],
                             "angle_deg": float(np.mean(angs)) if angs else float("nan"),
                             "sd_deg": float(np.std(angs)) if angs else float("nan"), "sectors": len(angs)})
    rc = np.asarray(profiles["all"]["r"])
    local = []
    for k in range(len(rc) - 2):
        if h_all[k + 2] < h_all[k] and h_all[k] < 0.95 * hmax and h_all[k + 2] > 0.05 * hmax:
            slope, _ = np.polyfit(rc[k:k + 3], h_all[k:k + 3], 1)
            local.append(math.degrees(math.atan(-slope)))
    res["max_local_angle_deg"] = float(max(local)) if local else float("nan")
    return res


# ---------------------------------------------------------------------------------------------------------------------
# stress
# ---------------------------------------------------------------------------------------------------------------------


def stress_invariants(S: np.ndarray):
    """Pressure p (compression positive after the sign choice), sqrt(J2), and the Lode parameter cos(3 theta) =
    (3 sqrt(3) / 2) J3 / J2^1.5 in [-1, 1] (0: pure shear, as in simple shear; +-1: triaxial compression /
    extension meridians) of stresses ``S`` (P, 3, 3). The sign is flipped when the median of -tr(S)/3 is negative."""
    S = 0.5 * (S + np.swapaxes(S, -1, -2))
    tr = np.trace(S, axis1=-2, axis2=-1)
    p = -tr / 3.0
    sign = 1.0
    if np.median(p) < 0:
        p, sign = -p, -1.0
    s = sign * S - (sign * tr / 3.0)[:, None, None] * np.eye(3)       # deviator
    # (the Lode parameter's sign follows the compression-positive convention only through ``sign``: J3 of -s is -J3)
    J2 = 0.5 * np.einsum("pij,pij->p", s, s)
    J3 = np.linalg.det(s)
    rJ2 = np.sqrt(np.maximum(J2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        lode = np.where(J2 > 0, 1.5 * math.sqrt(3.0) * J3 / np.maximum(J2, 1e-300) ** 1.5, 0.0)
    return p, rJ2, np.clip(lode, -1.0, 1.0), sign


def flank_stress(q: np.ndarray, S: np.ndarray, mu: float, voxel: float, particle_volume: float, r_apex: float,
                 r_toe: float, h_max: float, density: float = 1500.0, shells=(0.0, 1.0, 2.0, 4.0, 1e9)) -> dict:
    """Yield ratio sqrt(J2) / (mu p) and Lode parameter of the heap's flank particles, by depth below the local free
    surface (shells in cells). Flank: column radius between r_apex + 1.5 voxels and r_toe - 1.5 voxels and column top
    between 25 and 75 % of h_max (the window of the fitted angle). Also the pressure at the heap's base centre over
    rho g h_max (a scale check of the exported stress)."""
    col = 2.0 * particle_volume ** (1.0 / 3.0)
    cxy, ztop, inv = column_tops(q, col)
    p, rJ2, lode, sign = stress_invariants(S)
    rcol = np.hypot(cxy[:, 0], cxy[:, 1])[inv]
    top = ztop[inv]
    depth = (top - q[:, 2]) / voxel
    flank = (rcol >= r_apex + 1.5 * voxel) & (rcol <= r_toe - 1.5 * voxel) & (top >= 0.25 * h_max) & (top <= 0.75 * h_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p > 0, rJ2 / (mu * np.maximum(p, 1e-30)), np.inf)
    out = {"sign": sign, "flank_particles": int(flank.sum()), "shells": []}
    for a, b in zip(shells[:-1], shells[1:]):
        m = flank & (depth >= a) & (depth < b)
        n = int(m.sum())
        rec = {"depth_cells": [a, b if b < 1e8 else None], "n": n}
        if n:
            rr = ratio[m][np.isfinite(ratio[m])]
            rec.update(ratio_median=float(np.median(rr)) if rr.size else float("nan"),
                       ratio_p90=float(np.quantile(rr, 0.9)) if rr.size else float("nan"),
                       frac_at_yield=float(np.mean(rr >= 0.95)) if rr.size else float("nan"),
                       frac_tension=float(np.mean(p[m] <= 0)), lode_median=float(np.median(lode[m])),
                       lode_mean=float(np.mean(lode[m])))
        out["shells"].append(rec)
    base = (np.hypot(q[:, 0], q[:, 1]) < 3 * voxel) & (q[:, 2] < 2 * voxel)
    out["p_base_over_rho_g_h"] = float(np.median(p[base]) / (density * 9.81 * h_max)) if base.any() else float("nan")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# tilting box
# ---------------------------------------------------------------------------------------------------------------------


def layer_surface(q: np.ndarray, x_lo: float, x_hi: float, col: float, min_depth: float):
    """Surface of a layer in the box frame (x along the slope, z normal to the floor): highest particle per x column
    (over all y). Returns (x centres, top z) of the columns inside [x_lo, x_hi] at least ``min_depth`` deep."""
    i = np.floor(q[:, 0] / col).astype(np.int64)
    uniq, inv = np.unique(i, return_inverse=True)
    top = np.full(uniq.shape[0], -np.inf)
    np.maximum.at(top, inv, q[:, 2])
    xc = (uniq + 0.5) * col
    m = (xc >= x_lo) & (xc <= x_hi) & (top >= min_depth)
    return xc[m], top[m]


def surface_angle_world(q: np.ndarray, tilt_deg: float, x_lo: float, x_hi: float, col: float, min_depth: float):
    """Free-surface angle [deg] to the horizontal of a layer in a box tilted by ``tilt_deg`` (floor descending towards
    +x). In the box frame the surface rises towards +x with slope s once material has moved downslope, so the angle
    to the horizontal is tilt - atan(s). Returns (angle, box-frame slope angle, columns used)."""
    xc, top = layer_surface(q, x_lo, x_hi, col, min_depth)
    if xc.size < 4:
        return float("nan"), float("nan"), int(xc.size)
    # After a full avalanche the material piles against the downslope wall and the upslope part is bare or thin: a
    # line through every column then fits a step (batch 02b t1b: box-frame slope 89.9 deg). Fit the free face only:
    # the columns between 20 and 80 % of the way from the lowest to the highest top, when that range is >= 2 columns
    # wide in height; otherwise (a layer that did not avalanche) every column.
    lo, hi = float(top.min()), float(top.max())
    face = (top >= lo + 0.2 * (hi - lo)) & (top <= lo + 0.8 * (hi - lo))
    if hi - lo >= 2 * min_depth and face.sum() >= 4:
        xc, top = xc[face], top[face]
    s, _ = np.polyfit(xc, top, 1)
    a_box = math.degrees(math.atan(s))
    return tilt_deg - a_box, a_box, int(xc.size)


def surface_speed(q: np.ndarray, v: np.ndarray, x_lo: float, x_hi: float, col: float, skin: float,
                  quantile: float = 0.9) -> float:
    """``quantile`` of the downslope (+x) speed of the surface particles (within ``skin`` of their column's top) whose
    x lies in [x_lo, x_hi]."""
    i = np.floor(q[:, 0] / col).astype(np.int64)
    uniq, inv = np.unique(i, return_inverse=True)
    top = np.full(uniq.shape[0], -np.inf)
    np.maximum.at(top, inv, q[:, 2])
    m = (q[:, 0] >= x_lo) & (q[:, 0] <= x_hi) & (q[:, 2] >= top[inv] - skin)
    return float(np.quantile(v[m, 0], quantile)) if m.any() else float("nan")


def first_sustained(t: np.ndarray, x: np.ndarray, threshold: float, window: float):
    """Index of the first sample from which ``x > threshold`` holds for at least ``window`` seconds (None if never)."""
    above = x > threshold
    start = None
    for k in range(len(t)):
        if above[k]:
            if start is None:
                start = k
            if t[k] - t[start] >= window:
                return start
        else:
            start = None
    return None


# ---------------------------------------------------------------------------------------------------------------------
# bodies on sand
# ---------------------------------------------------------------------------------------------------------------------


def quat_wxyz_to_mat(qw: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(c) for c in qw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def box_overlap(q: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray, half: float) -> dict:
    """Particles inside a cube of half-size ``half`` at ``pos`` / ``quat`` (MuJoCo's w-first): count and the deepest
    penetration below the cube's bottom face (cube frame), i.e. how far the material overlaps the body."""
    R = quat_wxyz_to_mat(quat_wxyz)
    local = (q - pos) @ R
    inside = np.all(np.abs(local) < half, axis=1)
    depth = (local[inside, 2] + half) if inside.any() else np.zeros(0)
    return {"inside": int(inside.sum()), "max_depth_m": float(depth.max()) if depth.size else 0.0,
            "p90_depth_m": float(np.quantile(depth, 0.9)) if depth.size else 0.0}


def sphere_overlap(q: np.ndarray, centre: np.ndarray, radius: float) -> dict:
    d = radius - np.linalg.norm(q - centre, axis=1)
    m = d > 0
    return {"inside": int(m.sum()), "max_depth_m": float(d[m].max()) if m.any() else 0.0}


def ring_heave(q: np.ndarray, centre_xy, r_in: float, r_out: float, col: float, depth0: float) -> float:
    """Mean column top in the annulus [r_in, r_out] around ``centre_xy`` minus the initial bed depth [m]."""
    cxy, top, _ = column_tops(q, col)
    r = np.hypot(cxy[:, 0] - centre_xy[0], cxy[:, 1] - centre_xy[1])
    m = (r >= r_in) & (r <= r_out)
    return float(np.mean(top[m]) - depth0) if m.any() else float("nan")


def uehara_depth(mu: float, D: float, H: float, density_ratio: float = 1.0) -> float:
    """Uehara et al. 2003 (arXiv:cond-mat/0301549): final penetration d = 0.14 / mu (rho_ball / rho_grain)^(1/2)
    D^(2/3) H^(1/3) [m], H = drop height + d (total drop), mu = tan(draining repose angle)."""
    return 0.14 / mu * math.sqrt(density_ratio) * D ** (2.0 / 3.0) * H ** (1.0 / 3.0)


def uehara_solve(mu: float, D: float, h: float, density_ratio: float = 1.0) -> float:
    """d from Uehara's law with H = h + d (fixed point; converges in a few iterations for d << h)."""
    d = 0.0
    for _ in range(50):                                   # bounded
        d_new = uehara_depth(mu, D, h + d, density_ratio)
        if abs(d_new - d) < 1e-9:
            return d_new
        d = d_new
    return d


def impact_phase(r: dict, window: float = 0.3, g: float = 9.81) -> dict:
    """Impact-phase penetration of a drop record (``bearing_fidelity`` / ``bearing`` JSON): the deepest penetration
    within ``window`` s after free-fall contact (t_c = sqrt(2 h / g)), and the free-fall impact speed sqrt(2 g h).
    The record's own ``impact_speed_m_s`` is read when the body crosses the particle surface, after the grid contact
    (active about a voxel above the particles) has already slowed it, and its whole-run peak can be a later event."""
    drop = float(r.get("drop", 0.15))
    v = math.sqrt(2 * g * drop)
    if "bottom" not in r:
        return {"peak_m": float(r["peak_penetration_m"]), "t_peak_s": float(r.get("t_peak_s", float("nan"))),
                "v_impact": v}
    pen = float(r["depth"]) - np.asarray(r["bottom"], float)
    t = (np.arange(pen.size) + 1) / float(r["fps"])
    m = t <= math.sqrt(2 * drop / g) + window
    k = int(np.argmax(np.where(m, pen, -np.inf)))
    return {"peak_m": float(pen[k]), "t_peak_s": float(t[k]), "v_impact": v}


def linear_fit(x, y):
    """Least-squares y = a + b x; returns (a, b)."""
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(a), float(b)
