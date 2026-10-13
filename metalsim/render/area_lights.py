"""USD area lights (RectLight, DiskLight, CylinderLight, SphereLight with a radius) as true area emitters.

``metalsim.scene.usd_to_mjcf`` imports each as a MuJoCo spot / point light with the emitter's far-field radiometry
(what tier 0 shades) and records its shape in the custom text ``usd_area_lights``: shape, size, radiance (UsdLux:
intensity x 2^exposure x color, / area with ``inputs:normalize``), the prim's X axis in the light's body frame. Tier 2
(``Tier2Renderer(area_lights=True)``, the default when the model has the record) samples the emitter instead:
next-event estimation from a uniform point on the rectangle / disk / cylinder side (area measure) or a uniform
direction in the cone the sphere subtends (solid-angle measure), with a shadow ray to it (pathtrace.metal, AREA_LIGHTS).
Soft shadows and the near-field irradiance follow.

Light record fields (scene_tables.build_lights layout; ModelSync keeps the .w components from the template):
specular.w = shape (1 rect, 2 disk, 3 cylinder, 4 sphere), ambient.w = width / radius, diffuse.w = height / length,
diffuse.xyz = radiance (locked against the per-world light_diffuse). The emitters' world X axes, per env and light,
follow the light records in the renderer's light buffer (``AreaLights.encode`` from ``xmat``, or ``write_host``).

Analytic irradiance for the tests: Lambert's polygon formula (``polygon_irradiance``), the coaxial disk and the sphere.
"""
from __future__ import annotations

import json

import mujoco
import numpy as np

SHAPES = {"rect": 1, "disk": 2, "cylinder": 3, "sphere": 4}


def model_area_lights(model) -> list:
    """The ``usd_area_lights`` records of a model (empty when it has none)."""
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "usd_area_lights")
    if tid < 0:
        return []
    raw = bytes(model.text_data[model.text_adr[tid]:model.text_adr[tid] + model.text_size[tid]]).split(b"\0")[0]
    return json.loads(raw.decode())


class AreaLights:
    """Per-light area records of a model and the per-env world X axes of the emitters."""

    def __init__(self, model, records=None):
        self.m = model
        recs = model_area_lights(model) if records is None else records
        self.lights = []                         # (light index, record)
        for r in recs:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, r["light"])
            if i < 0:
                raise ValueError(f"area light {r['light']!r} is not a light of the model")
            self.lights.append((int(i), r))
        nl = max(model.nlight, 1)
        self.x_body = np.zeros((nl, 3), np.float32); self.x_body[:, 0] = 1.0
        self.body = np.zeros(nl, np.int32)
        for i, r in self.lights:
            self.x_body[i] = np.asarray(r.get("x_axis", (1.0, 0.0, 0.0)), np.float32)
            self.body[i] = int(model.light_bodyid[i])

    def __bool__(self):
        return bool(self.lights)

    def apply(self, lights: np.ndarray, lock: np.ndarray):
        """Write the area fields into (max(nlight, 1), 20) light records and lock their radiance."""
        for i, r in self.lights:
            shape = r["shape"]
            a = r["width"] if shape == "rect" else r["radius"]
            b = r["height"] if shape == "rect" else (r["length"] if shape == "cylinder" else 0.0)
            lights[i, 4:7] = np.asarray(r["radiance"], np.float32)
            lights[i, 7] = b
            lights[i, 11] = SHAPES[shape]
            lights[i, 15] = a
            lock[i] |= 1
        return lights, lock

    def tail_bytes(self, n_envs: int) -> int:
        """Room the emitters' axes take after the light records (ModelSync(light_tail_bytes=...))."""
        return n_envs * max(self.m.nlight, 1) * 16

    def attach(self, ctx, light_buf, n_envs: int):
        """Write the initial axes after the (n_envs, max(nlight, 1)) records of ``light_buf`` and prepare the per-frame
        update (``encode`` / ``write_host``)."""
        import mujoco as _mj
        self.ctx, self.light_buf, self.n = ctx, light_buf, n_envs
        nl = max(self.m.nlight, 1)
        self.off = n_envs * nl * 80
        d0 = _mj.MjData(self.m); _mj.mj_kinematics(self.m, d0)
        ax0 = np.repeat(self.axes_host([d0]), n_envs, axis=0)
        ctx.write_buffer(light_buf, ax0, offset=self.off)
        xb = np.zeros((nl, 4), np.float32); xb[:, :3] = self.x_body
        self.xb_buf = ctx.buffer(xb.nbytes, xb, "area_xbody")
        self.body_buf = ctx.buffer(max(4 * nl, 16), self.body.astype(np.int32), "area_body")
        self.pipe = ctx.compute_pipeline(ctx.library("area_lights"), "area_axes")

    def encode(self, cb, sim):
        """The emitters' world X axes per env from the sim's body orientations (``sim.d.xmat``); a sim without it keeps
        the initial axes."""
        import Metal
        from metalsim.interop import warp_metal as wm
        xm = getattr(getattr(sim, "d", None), "xmat", None)
        if xm is None:
            return
        v = wm.buffer_of(xm)
        nl = max(self.m.nlight, 1)
        worlds = int(xm.shape[0]) if xm.ndim > 1 else 1
        k = np.array([self.n, nl, self.m.nbody, 1 if worlds > 1 else 0], np.uint32)
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.pipe)
        ce.setBuffer_offset_atIndex_(v.buffer, v.offset, 0)
        ce.setBuffer_offset_atIndex_(self.body_buf, 0, 1)
        ce.setBuffer_offset_atIndex_(self.xb_buf, 0, 2)
        ce.setBuffer_offset_atIndex_(self.light_buf, self.off, 3)
        ce.setBytes_length_atIndex_(k.tobytes(), k.nbytes, 4)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n * nl, 1, 1), Metal.MTLSize(64, 1, 1))
        ce.endEncoding()

    def write_host(self, datas):
        self.ctx.write_buffer(self.light_buf, self.axes_host(datas), offset=self.off)

    def axes_host(self, datas) -> np.ndarray:
        """(n_envs, max(nlight, 1), 4) world X axes from MjData xmat."""
        nl = max(self.m.nlight, 1)
        out = np.zeros((len(datas), nl, 4), np.float32)
        for e, d in enumerate(datas):
            R = np.asarray(d.xmat, np.float64).reshape(-1, 3, 3)[self.body]
            out[e, :, :3] = np.einsum("lij,lj->li", R, self.x_body)
        return out


# -- analytic irradiance (validation) ------------------------------------------------------------------------------------

def polygon_irradiance(L, verts, p, n):
    """Irradiance at point p with unit normal n from a Lambertian polygon of radiance L (vertices (K, 3), either
    winding, fully above p's horizon): Lambert's formula E = (L / 2) |sum_i Theta_i (Gamma_i . n)| (Arvo 1994)."""
    v = np.asarray(verts, np.float64) - np.asarray(p, np.float64)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    w = np.roll(v, -1, axis=0)
    c = np.cross(v, w)
    nc = np.linalg.norm(c, axis=1)
    ok = nc > 1e-15
    theta = np.arctan2(nc[ok], (v[ok] * w[ok]).sum(1))
    s = float((theta * (c[ok] @ np.asarray(n, np.float64)) / nc[ok]).sum())
    return 0.5 * L * abs(s)


def rect_vertices(center, x_axis, y_axis, width, height):
    c, x, y = (np.asarray(v, np.float64) for v in (center, x_axis, y_axis))
    hx, hy = 0.5 * width * x, 0.5 * height * y
    return np.array([c - hx - hy, c + hx - hy, c + hx + hy, c - hx + hy])


def disk_vertices(center, x_axis, y_axis, radius, k=4096):
    """A k-gon of the disk's area (circumradius scaled so the areas match): Lambert's formula on it converges to the
    disk's irradiance as k^-2."""
    t = 2 * np.pi * np.arange(k) / k
    r = radius * np.sqrt(2 * np.pi / (k * np.sin(2 * np.pi / k)))
    c, x, y = (np.asarray(v, np.float64) for v in (center, x_axis, y_axis))
    return c + r * (np.cos(t)[:, None] * x + np.sin(t)[:, None] * y)


def disk_coaxial_irradiance(L, radius, d):
    """On the axis of a disk at distance d, receiver facing it: pi L R^2 / (R^2 + d^2)."""
    return np.pi * L * radius ** 2 / (radius ** 2 + d ** 2)


def sphere_irradiance(L, radius, d):
    """Receiver facing a sphere of radiance L at centre distance d (the whole sphere above the horizon): pi L (R/d)^2."""
    return np.pi * L * (radius / d) ** 2
