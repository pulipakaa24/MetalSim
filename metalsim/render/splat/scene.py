"""Splat scene preparation: transform into the sim frame, crop / outlier removal, spatial ordering, GPU packing and an
on-disk cache of the packed arrays.

Transform: an affine map x_world = A x_file + t is applied exactly to every Gaussian (mean -> A mu + t, covariance ->
A Sigma A^T, evaluated on the GPU from the file-frame rotation/scale and A), and view directions for the SH lookup are
mapped back into the file frame (normalize(A^-1 d)), so SH stay in the frame they were fitted in. ``A`` may include a
shear or a non-uniform scale (e.g. levelling a tilted floor plane, ``floor_plane_transform``), which is exact for the
Gaussians and first-order for SH directions. ``make_transform`` builds (A, t) from the common pieces (file up axis,
uniform scale, yaw, translation).

Packed GPU layout (``PackedSplats``), Morton-ordered so threadgroups touch spatially coherent splats:
* ``g0`` (N, 4) float32: world mean xyz, opacity;
* ``g1`` (N, 8) float16: file-frame quaternion wxyz, file-frame linear scales xyz, conservative world-space culling
  radius (opacity-aware: sqrt(2 ln(255 o)) x max scale x ||A||);
* ``sh`` (N, S) uint8 rows: half4 (dc r, g, b, 0) then the higher bands, uint8 (SPZ quantisation, S = 64) or float16
  (S = 8 + 2 x 3K rounded to 16 bytes).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from metalsim.render.splat.formats import GaussianCloud, load, sh_dim

CACHE_DIR = os.path.expanduser(os.environ.get("METALSIM_SPLAT_CACHE", "~/.cache/metalsim/splat"))
ALPHA_MIN = 1.0 / 255.0


UP_AXES = {
    # file frame -> MetalSim's z-up frame (x stays x; right-handed): (x, y, z)_sim = R (x, y, z)_file
    "z": np.eye(3),                                                   # already z-up (InteriorGS, most robotics data)
    "y": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]),          # +y up (OpenGL / RUB, SPZ, many NuRec volumes)
    "-y": np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),         # -y up (COLMAP / OpenCV RDF, original 3DGS training)
}


def make_transform(up: str = "z", scale: float = 1.0, yaw_deg: float = 0.0, translate=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """File -> sim affine map ``x_sim = A x_file + t`` from the usual pieces, applied in this order: rotate the file's
    up axis onto +z (``up`` in ``UP_AXES``), scale uniformly (file units -> metres), rotate by ``yaw_deg`` about +z,
    translate. Returns (A (3, 3), t (3,)) for ``load_scene`` / ``prepare``."""
    if up not in UP_AXES:
        raise ValueError(f"up={up!r}: one of {sorted(UP_AXES)} (or build A, t yourself)")
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return float(scale) * Rz @ UP_AXES[up], np.asarray(translate, np.float64).reshape(3).copy()


def floor_plane_transform(normal, point, up: str = "z") -> tuple[np.ndarray, np.ndarray]:
    """Level a scene whose floor is a slightly tilted plane: after ``up`` -> +z, the plane through ``point`` with unit
    ``normal`` (both in that z-up frame) becomes z = 0, by the shear z' = z - plane_height(x, y). Exact for the Gaussians
    (A Sigma A^T); SH directions are mapped back through A^-1, which is first order in the tilt. Use it when a collision
    height field is defined as "height above the fitted floor"."""
    n, m = np.asarray(normal, float), np.asarray(point, float)
    if abs(n[2]) < 1e-6:
        raise ValueError("floor normal must not be horizontal")
    S = np.eye(3); S[2, 0] = n[0] / n[2]; S[2, 1] = n[1] / n[2]
    t = np.array([0.0, 0.0, -(m[2] + (n[0] * m[0] + n[1] * m[1]) / n[2])])
    return S @ UP_AXES[up], t


@dataclass
class PackedSplats:
    g0: np.ndarray
    g1: np.ndarray
    sh: np.ndarray
    sh_u8: bool
    sh_degree: int
    antialiased: bool
    A: np.ndarray                      # file -> world linear part
    t: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.g0.shape[0])

    @property
    def sh_stride(self) -> int:
        return int(self.sh.shape[1])

    def nbytes(self) -> int:
        return self.g0.nbytes + self.g1.nbytes + self.sh.nbytes

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(tmp, g0=self.g0, g1=self.g1, sh=self.sh, A=self.A, t=self.t,
                 meta=np.array(json.dumps(dict(sh_u8=self.sh_u8, sh_degree=self.sh_degree, antialiased=self.antialiased, info=self.info))))
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "PackedSplats":
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        return PackedSplats(z["g0"], z["g1"], z["sh"], meta["sh_u8"], meta["sh_degree"], meta["antialiased"], z["A"], z["t"], meta["info"])


def _morton3(q: np.ndarray) -> np.ndarray:
    """q (N, 3) uint32 in [0, 1024) -> 30-bit Morton codes."""
    def spread(v):
        v = v.astype(np.uint64) & 0x3FF
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v
    return spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2)


def auto_crop_box(means_w: np.ndarray, q=0.01, grow=1.0, margin=1.0) -> np.ndarray:
    """Robust scene box: the 1-99 % quantile box of the means grown by its largest extent (+ 1 m) on every side.
    Far floaters (captures often carry a sparse shell of splats hundreds of metres out) are dropped; content seen through
    windows and doors stays inside the grown box. A tight box (grow 0.1) cuts background seen through windows; numbers
    on InteriorGS in SPLAT.md §6."""
    lo = np.quantile(means_w, q, axis=0); hi = np.quantile(means_w, 1 - q, axis=0)
    g = grow * float((hi - lo).max()) + margin            # isotropic: the largest extent on every axis (z too: views through glass)
    return np.stack([lo - g, hi + g])


def prepare(cloud: GaussianCloud, A=None, t=None, crop="auto", min_opacity: float = ALPHA_MIN, sh_degree: int | None = None,
            log=print) -> PackedSplats:
    """Cloud -> PackedSplats. ``crop``: None (keep all), "auto" (0.5-99.5 % quantile box of the world means + 1 m),
    or a (2, 3) world box [lo, hi]. Splats with opacity < 1/255 are dropped: their alpha can never reach the
    rasterizer's 1/255 threshold, so this is lossless. ``sh_degree``: clamp the SH degree (None = file's)."""
    t0 = time.perf_counter()
    A = np.eye(3) if A is None else np.asarray(A, np.float64)
    t = np.zeros(3) if t is None else np.asarray(t, np.float64)
    n0 = cloud.n
    mw = (cloud.means.astype(np.float64) @ A.T + t).astype(np.float32)
    keep = cloud.opacity >= min_opacity
    n_alpha = int((~keep).sum())
    box = None
    if crop is not None:
        box = auto_crop_box(mw[keep]) if (isinstance(crop, str) and crop == "auto") else np.asarray(crop, np.float64)
        keep &= np.all((mw >= box[0]) & (mw <= box[1]), axis=1)
    idx = np.nonzero(keep)[0]
    mw = mw[idx]
    # Morton order over the kept box
    lo, hi = mw.min(0), mw.max(0)
    qn = ((mw - lo) / np.maximum(hi - lo, 1e-6) * 1023.0).astype(np.uint32)
    order = np.argsort(_morton3(qn), kind="stable")
    idx = idx[order]; mw = mw[order]
    deg = cloud.sh_degree if sh_degree is None else min(sh_degree, cloud.sh_degree)
    K = sh_dim(deg) if deg > 0 else 0
    N = len(idx)
    g0 = np.empty((N, 4), np.float32); g0[:, :3] = mw; g0[:, 3] = cloud.opacity[idx]
    sc = np.exp(cloud.log_scales[idx])
    g1 = np.empty((N, 8), np.float16)
    g1[:, 0:4] = cloud.quats[idx]
    g1[:, 4:7] = np.minimum(sc, 6e4)
    a_norm = float(np.linalg.norm(A, 2))
    op = np.maximum(cloud.opacity[idx], ALPHA_MIN * 1.0001)
    r = np.sqrt(2.0 * np.log(255.0 * op)) * sc.max(1) * a_norm
    g1[:, 7] = np.minimum(r * 1.01 + 1e-4, 6e4)          # rounded up so fp16 never undercuts
    if cloud.sh_rest_u8 or K == 0:
        stride = 64 if K <= 15 else 8 + ((3 * K + 15) // 16) * 16
        sh = np.zeros((N, stride), np.uint8)
        sh_u8 = True
        if K:
            sh[:, 8:8 + 3 * K] = cloud.sh_rest[idx, :K, :].reshape(N, 3 * K)
    else:
        stride = 8 + ((2 * 3 * K + 15) // 16) * 16
        sh = np.zeros((N, stride), np.uint8)
        sh_u8 = False
        sh[:, 8:8 + 6 * K] = cloud.sh_rest[idx, :K, :].astype(np.float16).reshape(N, 3 * K).view(np.uint8)
    dc = np.zeros((N, 4), np.float16); dc[:, :3] = cloud.sh_dc[idx]
    sh[:, 0:8] = dc.view(np.uint8)
    info = dict(source=cloud.source, n_file=n0, n_kept=N, dropped_alpha=n_alpha, dropped_crop=int(n0 - n_alpha - N),
                crop_box=None if box is None else np.asarray(box).tolist(), sh_degree_file=cloud.sh_degree,
                antialiased_flag=cloud.antialiased, prep_s=round(time.perf_counter() - t0, 2), meta=cloud.meta)
    log(f"[splat] prepared {N:,} / {n0:,} splats (alpha<1/255: {n_alpha:,}, cropped: {info['dropped_crop']:,}) "
        f"SH deg {deg} ({'u8' if sh_u8 else 'f16'}), {(g0.nbytes + g1.nbytes + sh.nbytes) / 1e6:.0f} MB, {info['prep_s']} s")
    return PackedSplats(g0, g1, sh, sh_u8, deg, bool(cloud.antialiased), A, t, info)


def _cache_key(path, A, t, crop, min_opacity, sh_degree) -> str:
    st = os.stat(path)
    h = hashlib.sha1(json.dumps([os.path.abspath(path), st.st_size, int(st.st_mtime), np.asarray(A).round(12).tolist(),
                                 np.asarray(t).round(12).tolist(), crop if (crop is None or isinstance(crop, str)) else np.asarray(crop).tolist(),
                                 min_opacity, sh_degree, 5]).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{os.path.splitext(os.path.basename(path))[0]}_{h}.npz")


def load_scene(path: str, A=None, t=None, crop="auto", min_opacity: float = ALPHA_MIN, sh_degree: int | None = None,
               cache: bool = True, log=print) -> PackedSplats:
    """Load + prepare a splat file, with the packed arrays cached on disk (keyed by file, size, mtime and options).
    A ``.npz`` written by ``PackedSplats.save`` is loaded as is (already in the sim frame: A / t must be identity; used for
    scenes whose file -> sim map is not affine, e.g. a NuRec scene with a ground warp, built by a scene-import script)."""
    A = np.eye(3) if A is None else np.asarray(A, np.float64)
    t = np.zeros(3) if t is None else np.asarray(t, np.float64)
    if path.endswith(".npz"):
        if not (np.allclose(A, np.eye(3)) and np.allclose(t, 0)):
            raise ValueError(f"{path}: a prepared PackedSplats .npz is already in its final frame; pass no transform")
        t0 = time.perf_counter()
        p = PackedSplats.load(path)
        log(f"[splat] prepared scene {path} ({p.n:,} splats, {time.perf_counter() - t0:.2f} s)")
        return p
    key = _cache_key(path, A, t, crop, min_opacity, sh_degree)
    if cache and os.path.exists(key):
        t0 = time.perf_counter()
        p = PackedSplats.load(key)
        log(f"[splat] cache hit {key} ({p.n:,} splats, {time.perf_counter() - t0:.2f} s)")
        return p
    cloud = load(path)
    p = prepare(cloud, A, t, crop, min_opacity, sh_degree, log=log)
    del cloud
    if cache:
        p.save(key)
    return p
