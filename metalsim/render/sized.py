"""Per-world geom sizes in the renderers: every primitive (``scene_tables.SIZE_*``), including the rough terrain's box
slots (``scene_tables.per_world_size_geoms``).

MuJoCo Warp's ``Model.geom_size`` is per world when the BatchSim has it in ``per_world_fields`` (``BoxWindow`` writes
the terrain slots' sizes; a scale randomization writes object sizes). The scene tables draw sized geoms from a unit
mesh (capsules: the compiled mesh); this pass writes a renderer-owned copy of ``geom_xmat`` with the per-world scale
folded into those geoms' columns, which the raster pass and the ray-tracing instance refit read in place of the physics
buffer (one dispatch over n_envs x ngeom, encoded into the render command buffer after the physics wait)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import Metal

from metalsim.interop import warp_metal as wm


class SizedGeoms:
    def __init__(self, ctx, model, n_envs: int, geom_ids=None, modes=None):
        """``modes``: (ngeom,) ``scene_tables.SIZE_*`` per model geom (``SceneTables.geom_size_modes``); or
        ``geom_ids`` = the geoms drawn as diag(size) unit boxes (the terrain slots; the previous interface)."""
        self.ctx, self.m, self.n = ctx, model, n_envs
        if modes is None:
            modes = np.zeros(max(model.ngeom, 1), np.int32); modes[[int(g) for g in geom_ids]] = 1
        self.modes = np.asarray(modes, np.int32)
        self.ids = [int(g) for g in np.flatnonzero(self.modes)]
        self.flag_buf = ctx.buffer(self.modes.nbytes, self.modes, "sized_modes")
        s0 = np.ascontiguousarray(model.geom_size, np.float32).reshape(-1)
        self.size0_buf = ctx.buffer(max(s0.nbytes, 16), s0 if s0.size else None, "sized_size0")
        self.out = ctx.buffer(n_envs * model.ngeom * 36, label="sized_geom_xmat")
        self.consts = np.array([n_envs, model.ngeom, 0, 0], np.uint32)
        self.consts_buf = ctx.buffer(16, self.consts, "sized_consts")
        self.pipe = ctx.compute_pipeline(ctx.library("sized"), "scale_geom_xmat")
        self._host_size = None

    def encode(self, cb, xmat_buf, xmat_off, geom_size):
        """``geom_size``: MuJoCo Warp's ``Model.geom_size`` (per world when BatchSim has ``per_world_fields`` with
        it, else shared). Returns the (buffer, offset) to bind as geom_xmat."""
        import warp as wp
        if not isinstance(geom_size, wp.array) or geom_size.size == 0:        # not supplied per world (e.g. the Newton bridge): compiled sizes
            gs = SimpleNamespace(buffer=self.size0_buf, offset=0); stride = 0
        else:
            gs = wm.buffer_of(geom_size)
            stride = 1 if geom_size.shape[0] > 1 else 0
        if stride and geom_size.shape[0] != self.n:
            raise ValueError(f"geom_size has {geom_size.shape[0]} worlds, renderer {self.n}")
        if self.consts[2] != stride:
            self.consts[2] = stride
            self.ctx.write_buffer(self.consts_buf, self.consts)
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.pipe)
        ce.setBuffer_offset_atIndex_(xmat_buf, xmat_off, 0)
        ce.setBuffer_offset_atIndex_(gs.buffer, gs.offset, 1)
        ce.setBuffer_offset_atIndex_(self.flag_buf, 0, 2)
        ce.setBuffer_offset_atIndex_(self.out, 0, 3)
        ce.setBuffer_offset_atIndex_(self.consts_buf, 0, 4)
        ce.setBuffer_offset_atIndex_(self.size0_buf, 0, 5)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n * self.m.ngeom, 1, 1), Metal.MTLSize(64, 1, 1))
        ce.endEncoding()
        return self.out, 0

    def scale_host(self, gm: np.ndarray, sizes: np.ndarray | None = None) -> np.ndarray:
        """Host path: ``gm`` (n, ngeom, 9) from MjData; ``sizes`` (n, ngeom, 3) per-world sizes (default the model's)."""
        from metalsim.render.scene_tables import size_scale
        gm = gm.copy()
        s = np.broadcast_to(self.m.geom_size if sizes is None else sizes, (self.n, self.m.ngeom, 3))
        for g in self.ids:
            sc = np.stack([size_scale(int(self.modes[g]), s[e, g], self.m.geom_size[g]) for e in range(self.n)])
            gm[:, g] = (gm[:, g].reshape(-1, 3, 3) * sc[:, None, :]).reshape(-1, 9)
        return gm
