"""Per-world MuJoCo Warp model fields in the renderers (shaders/model_sync.metal).

MuJoCo Warp batches ``geom_rgba``, ``geom_matid``, ``mat_rgba`` and the ``light_*`` fields per world when a BatchSim
lists them in ``per_world_fields`` (a colour or light randomization then writes ``sim.tm.<field>``). Each frame, in
the render command buffer after the physics wait, ``ModelSync.encode`` turns that world's values into the renderer's
per-env tables:

* ``inst_col`` (n_envs, G, 4): the renderer's DR colour (``t_colors``) x the model colour (MuJoCo's rule: ``geom_rgba``,
  or ``mat_rgba`` of the geom's material when the geom's rgba is the default), bound where the shaders read the
  per-instance colour; the renderer's material table then carries rgba = 1 (``unit_rgba``).
* ``lights`` (n_envs, nlight) light records (tier0.metal ``Light``): type, castshadow, cutoff, exponent, ambient,
  diffuse, specular, attenuation of that world; an inactive light (``light_active`` false) emits nothing.

Without a sim (the host paths) the same kernels read the compiled ``MjModel`` values, so both paths shade through one
code path. A sim may supply the fields through ``sim.render_fields`` instead of ``sim.m`` (``model_fields(sim)``; the
Newton bridge writes per-world ``geom_rgba`` from Newton's ``shape_color`` there); a field it does not supply as a Warp
array keeps the compiled value. Not covered: a per-world ``geom_matid`` switches the colour but not the texture / PBR parameters (the slot's
compiled material keeps those), and per-world ``mat_emission`` / ``mat_specular`` / ``mat_shininess``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import Metal

from metalsim.interop import warp_metal as wm

COLOR_FIELDS = ("geom_rgba", "geom_matid", "mat_rgba")
LIGHT_FIELDS = ("light_type", "light_castshadow", "light_active", "light_cutoff", "light_exponent", "light_ambient",
                "light_diffuse", "light_specular", "light_attenuation")


def model_fields(sim):
    """Where a renderer reads per-world model fields: ``sim.render_fields`` if the sim has one, else ``sim.m``."""
    rf = getattr(sim, "render_fields", None)
    return rf if rf is not None else getattr(sim, "m", None)


def unit_rgba(materials: np.ndarray) -> np.ndarray:
    """The (G, 16) material table the shaders read when ``ModelSync`` supplies the colour: rgba = 1."""
    m = np.array(materials, np.float32, copy=True).reshape(-1, 16)
    m[:, 0:4] = 1.0
    return m


class ModelSync:
    def __init__(self, ctx, model, n_envs: int, tables, color_buf, light_template: np.ndarray, light_lock=None,
                 opaque: bool = False, light_tail_bytes: int = 0):
        """``color_buf``: the renderer's ``t_colors`` buffer (n_envs, G, 4). ``light_template``: (max(nlight, 1), 20)
        light records (``scene_tables.build_lights`` plus a renderer's overrides); ``light_lock`` (nlight,) uint32 bit0
        keeps the template's diffuse (tier 2's USD light intensities). ``opaque``: alpha of the model colour forced to 1
        (a renderer's ``transparency=False``). ``light_tail_bytes``: room after the light records in the same buffer
        (tier 2's area-light axes, metalsim.render.area_lights)."""
        self.ctx, self.m, self.n, self.t = ctx, model, n_envs, tables
        self.opaque = bool(opaque)
        G, nl = tables.G, max(model.nlight, 1)
        self.G, self.nl = G, nl
        self.color_buf = color_buf
        c = ctx
        G0 = len(tables.geoms)                               # geom slots; extras (sites, flexes, ...) follow
        sg = np.zeros(G, np.int32); sg[:G0] = np.asarray(tables.geoms, np.int32)
        self.slot_geom = c.buffer(4 * G, sg, "sync_slot_geom")
        static = np.asarray(tables.materials, np.float32).reshape(G, 16)[:, 0:4].copy()
        src = np.zeros(G, np.int32); src[:G0] = 1             # extra slots keep the scene tables' colour
        if tables.sized is not None:
            sz = np.asarray(tables.sized, bool)[:G0]
            src[:len(sz)][sz] = 0                             # terrain slots: the scene tables' terrain colour
        self.rgba0, self.src0 = static.copy(), src.copy()
        self.static_buf = c.buffer(static.nbytes, static, "sync_static_rgba")
        self.src_buf = c.buffer(src.nbytes, src, "sync_src")
        self.inst_col = c.buffer(n_envs * G * 16, label="sync_inst_col")
        self.lights = c.buffer(n_envs * nl * 80 + int(light_tail_bytes), label="sync_lights")
        self.set_light_template(light_template, light_lock)
        # host defaults: the compiled model's values (one world, stride 0)
        m = model
        f32 = lambda a, k: np.ascontiguousarray(np.asarray(a, np.float32).reshape(-1, k))   # noqa: E731
        self.host = {
            "geom_rgba": f32(m.geom_rgba, 4), "geom_matid": np.asarray(m.geom_matid, np.int32),
            "mat_rgba": f32(m.mat_rgba, 4) if m.nmat else np.zeros((1, 4), np.float32),
            "light_type": np.asarray(m.light_type if m.nlight else [1], np.int32),
            "light_castshadow": np.asarray(m.light_castshadow if m.nlight else [0], np.uint8),
            "light_active": np.asarray(m.light_active if m.nlight else [0], np.uint8),
            "light_cutoff": np.asarray(m.light_cutoff if m.nlight else [0], np.float32),
            "light_exponent": np.asarray(m.light_exponent if m.nlight else [0], np.float32),
            **{k: (f32(getattr(m, k), 3) if m.nlight else np.zeros((1, 3), np.float32))
               for k in ("light_ambient", "light_diffuse", "light_specular", "light_attenuation")},
        }
        self.host_bufs = {k: SimpleNamespace(buffer=c.buffer(max(v.nbytes, 16), v, f"sync_host_{k}"), offset=0, worlds=1)
                          for k, v in self.host.items()}
        self.consts = np.zeros(20, np.uint32)
        self.consts[:5] = (n_envs, G, model.ngeom, model.nmat, nl)
        self.consts_buf = c.buffer(self.consts.nbytes, self.consts, "sync_consts")
        lib = c.library("model_sync")
        self.p_col = c.compute_pipeline(lib, "sync_colors")
        self.p_light = c.compute_pipeline(lib, "sync_lights")

    def set_light_template(self, template, lock=None):
        tmpl = np.zeros((self.nl, 20), np.float32)
        tmpl[:len(template)] = np.asarray(template, np.float32)[:self.nl]
        lk = np.zeros(self.nl, np.uint32) if lock is None else np.asarray(lock, np.uint32)
        if not hasattr(self, "tmpl_buf"):
            self.tmpl_buf = self.ctx.buffer(tmpl.nbytes, tmpl, "sync_light_template")
            self.lock_buf = self.ctx.buffer(max(lk.nbytes, 16), lk, "sync_light_lock")
        else:
            self.ctx.write_buffer(self.tmpl_buf, tmpl); self.ctx.write_buffer(self.lock_buf, lk)

    def set_static_rgba(self, materials):
        """After a material-table change (``set_materials``): a slot whose rgba was changed from its compiled colour
        keeps that colour (the explicit host override wins over the model's per-world colour); terrain slots always
        use the table's."""
        static = np.asarray(materials, np.float32).reshape(self.G, 16)[:, 0:4].copy()
        src = np.where(np.any(np.abs(static - self.rgba0) > 1e-7, axis=1), 0, self.src0).astype(np.int32)
        self.ctx.write_buffer(self.static_buf, static)
        self.ctx.write_buffer(self.src_buf, src)

    def _views(self, sim):
        if sim is None:
            return self.host_bufs
        import warp as wp
        src = model_fields(sim)
        out = dict(self.host_bufs)
        for k in COLOR_FIELDS + (LIGHT_FIELDS if self.m.nlight else ()):
            a = getattr(src, k, None)
            if not isinstance(a, wp.array) or a.size == 0:
                continue                      # not supplied per world: the compiled value
            v = wm.buffer_of(a)
            out[k] = SimpleNamespace(buffer=v.buffer, offset=v.offset, worlds=int(a.shape[0]) if a.ndim > 1 else 1)
        return out

    def encode(self, cb, sim=None):
        v = self._views(sim)
        for k, w in v.items():
            if w.worlds > 1 and w.worlds != self.n:
                raise ValueError(f"model field {k} has {w.worlds} worlds, renderer {self.n}")
        st = lambda k: 1 if v[k].worlds > 1 else 0          # noqa: E731
        self.consts[5:8] = (st("geom_rgba"), st("geom_matid"), st("mat_rgba"))
        self.consts[8:19] = (st("light_type"), st("light_castshadow"), st("light_active"), st("light_cutoff"),
                             st("light_exponent"), st("light_ambient"), st("light_diffuse"), st("light_specular"),
                             st("light_attenuation"), 1 if self.m.nlight else 0, 1 if self.opaque else 0)
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.p_col)
        for i, (b, o) in enumerate(((self.slot_geom, 0), (self.color_buf, 0), (v["geom_rgba"].buffer, v["geom_rgba"].offset),
                                    (v["geom_matid"].buffer, v["geom_matid"].offset), (v["mat_rgba"].buffer, v["mat_rgba"].offset),
                                    (self.static_buf, 0), (self.src_buf, 0), (self.inst_col, 0))):
            ce.setBuffer_offset_atIndex_(b, o, i)
        ce.setBytes_length_atIndex_(self.consts.tobytes(), self.consts.nbytes, 8)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n * self.G, 1, 1), Metal.MTLSize(64, 1, 1))
        ce.setComputePipelineState_(self.p_light)
        ce.setBuffer_offset_atIndex_(self.tmpl_buf, 0, 0); ce.setBuffer_offset_atIndex_(self.lock_buf, 0, 1)
        for i, k in enumerate(LIGHT_FIELDS):
            ce.setBuffer_offset_atIndex_(v[k].buffer, v[k].offset, 2 + i)
        ce.setBuffer_offset_atIndex_(self.lights, 0, 11)
        ce.setBytes_length_atIndex_(self.consts.tobytes(), self.consts.nbytes, 12)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n * self.nl, 1, 1), Metal.MTLSize(64, 1, 1))
        ce.endEncoding()
