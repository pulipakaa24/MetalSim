"""Camera lens models for the renderers (shaders/lens.h): pinhole, Isaac/RTX f-theta polynomial fisheye
("fisheyePolynomial", Isaac Lab ``FisheyeCameraCfg``'s default) and Kannala-Brandt (OpenCV fisheye, RTX
"fisheyeKannalaBrandtK3").

Tier 2 and the ray-traced depth camera cast one ray per pixel along ``lens_ray``; tier 0 renders cube faces with its
raster and resamples them (``metalsim.render.fisheye.FisheyeTier0``). ``Lens.ray_dirs`` is the numpy reference of the
shader mapping (tests); ``Lens.project`` maps camera-frame points to pixels (the forward model).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PINHOLE, FTHETA, KANNALA_BRANDT = 0, 1, 2


@dataclass
class Lens:
    model: int = PINHOLE
    a: tuple = (0.0, 0.0, 0.0, 0.0)
    b: tuple = (0.0, 0.0, 0.0, 0.0)
    c: tuple = (0.0, 0.0, 0.0, 0.0)
    meta: dict = field(default_factory=dict)

    # -- constructors ------------------------------------------------------------------------------------------------

    @classmethod
    def pinhole(cls):
        return cls(PINHOLE)

    @classmethod
    def ftheta(cls, nominal_width=1936.0, nominal_height=1216.0, optical_centre_x=970.94244, optical_centre_y=600.37482,
               max_fov=200.0, poly=(0.0, 0.00245, 0.0, 0.0, 0.0, 0.0)):
        """f-theta polynomial fisheye: theta(r) = sum poly[k] r^k (r in nominal-resolution pixels from the optical
        centre); ``max_fov`` full angle in degrees. Defaults: Isaac Lab ``FisheyeCameraCfg``."""
        p = tuple(float(x) for x in poly) + (0.0,) * (6 - len(poly))
        return cls(FTHETA, (float(nominal_width), float(nominal_height), float(optical_centre_x), float(optical_centre_y)),
                   p[:4], (p[4], p[5], float(np.deg2rad(max_fov)), 0.0),
                   dict(kind="ftheta", nominal=(nominal_width, nominal_height), centre=(optical_centre_x, optical_centre_y),
                        max_fov=max_fov, poly=p))

    @classmethod
    def kannala_brandt(cls, fx, fy, cx, cy, k=(0.0, 0.0, 0.0, 0.0), max_fov=180.0, image_size=None, render_size=None):
        """Kannala-Brandt / OpenCV fisheye (Isaac Lab ``OpenCvFisheyeDistortionCfg``, USD
        ``omni:lensdistortion:opencvFisheye``): fx, fy, cx, cy in pixels of ``image_size`` (W, H), scaled to
        ``render_size`` when both are given (as RTX scales them); ``max_fov`` full angle in degrees."""
        if image_size is not None and render_size is not None:
            sx, sy = render_size[0] / image_size[0], render_size[1] / image_size[1]
            fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
        k = tuple(float(x) for x in k) + (0.0,) * (4 - len(k))
        return cls(KANNALA_BRANDT, (float(fx), float(fy), float(cx), float(cy)), k[:4], (0.0, 0.0, float(np.deg2rad(max_fov)), 0.0),
                   dict(kind="kannala_brandt"))

    @classmethod
    def from_isaac(cls, projection_type="fisheyePolynomial", **cfg):
        """From Isaac Lab ``FisheyeCameraCfg`` field names (``fisheye_nominal_width``, ``fisheye_polynomial_a`` ...)."""
        if projection_type in ("pinhole",):
            return cls.pinhole()
        if projection_type == "fisheyePolynomial":
            g = lambda k, d: cfg.get(f"fisheye_{k}", d)            # noqa: E731
            return cls.ftheta(g("nominal_width", 1936.0), g("nominal_height", 1216.0), g("optical_centre_x", 970.94244),
                              g("optical_centre_y", 600.37482), g("max_fov", 200.0),
                              tuple(g(f"polynomial_{x}", d) for x, d in zip("abcdef", (0.0, 0.00245, 0.0, 0.0, 0.0, 0.0))))
        if projection_type == "opencvFisheye":       # Isaac Lab (develop) OpenCvFisheyeDistortionCfg fields
            return cls.kannala_brandt(cfg["fx"], cfg["fy"], cfg["cx"], cfg["cy"], tuple(cfg.get(f"k{i}", 0.0) for i in range(1, 5)),
                                      np.rad2deg(cfg.get("max_fov", np.pi)), cfg.get("image_size"), cfg.get("render_size"))
        # fisheyeKannalaBrandtK3 / fisheyeRadTanThinPrism: Isaac Lab writes only the projection token (no coefficients);
        # fisheyeSpherical and the other legacy tokens are unsupported in Isaac Sim 5 (docs/RENDER_COVERAGE.md)
        raise NotImplementedError(f"lens projection {projection_type!r}: use Lens.ftheta / Lens.kannala_brandt")

    # -- GPU record and numpy reference ---------------------------------------------------------------------------------

    def pack(self) -> np.ndarray:
        out = np.zeros(16, np.float32)
        out.view(np.uint32)[0] = self.model
        out[4:8] = self.a; out[8:12] = self.b; out[12:16] = self.c
        return out

    def ray_dirs(self, width, height, intrinsics=None):
        """(H, W, 3) unit camera-frame directions at pixel centres and an (H, W) validity mask (numpy twin of lens_ray)."""
        with np.errstate(all="ignore"):
            return self._ray_dirs(width, height, intrinsics)

    def _ray_dirs(self, width, height, intrinsics=None):
        y, x = np.mgrid[0:height, 0:width].astype(np.float64) + 0.5
        if self.model == PINHOLE:
            fx, fy, cx, cy = intrinsics
            d = np.stack([(x - cx) / fx, -(y - cy) / fy, -np.ones_like(x)], -1)
            return d / np.linalg.norm(d, axis=-1, keepdims=True), np.ones(x.shape, bool)
        if self.model == FTHETA:
            dx = x * self.a[0] / width - self.a[2]; dy = y * self.a[1] / height - self.a[3]
            r = np.hypot(dx, dy)
            p = tuple(self.b) + tuple(self.c[:2])
            th = sum(p[k] * r ** k for k in range(6))
            u, v = dx, dy
        else:
            u = (x - self.a[2]) / self.a[0]; v = (y - self.a[3]) / self.a[1]
            r = np.hypot(u, v); th = r.copy()
            k1, k2, k3, k4 = self.b
            for _ in range(12):
                t2 = th * th
                f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - r
                fp = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
                th = th - f / np.maximum(fp, 1e-6)
        valid = np.isfinite(th) & (th >= 0) & (th <= 0.5 * self.c[2]) & (th <= np.pi)
        th = np.where(valid, th, 0.0)
        s = np.sin(th); rr = np.where(r > 1e-12, r, 1.0)
        d = np.stack([np.where(r > 1e-12, s * u / rr, 0.0), np.where(r > 1e-12, -s * v / rr, 0.0), -np.cos(th)], -1)
        return d, valid

    def project(self, p_cam: np.ndarray, width, height, intrinsics=None):
        """Camera-frame points (..., 3) -> pixel coordinates (..., 2) (continuous; inverse of ray_dirs)."""
        p = np.asarray(p_cam, np.float64)
        if self.model == PINHOLE:
            fx, fy, cx, cy = intrinsics
            return np.stack([cx + fx * p[..., 0] / -p[..., 2], cy - fy * p[..., 1] / -p[..., 2]], -1)
        th = np.arctan2(np.hypot(p[..., 0], p[..., 1]), -p[..., 2])
        phi_x, phi_y = p[..., 0], -p[..., 1]
        n = np.maximum(np.hypot(phi_x, phi_y), 1e-12)
        if self.model == FTHETA:
            coef = tuple(self.b) + tuple(self.c[:2])
            r = np.zeros_like(th)                          # invert theta(r) by Newton (monotone polynomials)
            for _ in range(50):
                f = sum(coef[k] * r ** k for k in range(6)) - th
                fp = sum(k * coef[k] * r ** (k - 1) for k in range(1, 6))
                r = r - f / np.where(np.abs(fp) > 1e-12, fp, 1e-12)
            xn = self.a[2] + r * phi_x / n; yn = self.a[3] + r * phi_y / n
            return np.stack([xn * width / self.a[0], yn * height / self.a[1]], -1)
        k1, k2, k3, k4 = self.b
        t2 = th * th
        td = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        return np.stack([self.a[2] + self.a[0] * td * phi_x / n, self.a[3] + self.a[1] * td * phi_y / n], -1)

    def max_angle(self) -> float:
        """Half field of view (rad) actually covered (bounded by max_fov and by the image corners)."""
        return float(min(0.5 * self.c[2], np.pi)) if self.model != PINHOLE else float(np.pi / 2)
