"""Per-world geom sizes in the renderers (the rough terrain's box slots; ``scene_tables.per_world_size_geoms``).

The slots are static world geoms whose size and position change per world at run time (``BoxWindow`` writes MuJoCo
Warp's per-world ``Model.geom_size`` and ``Data.geom_xpos``). The scene tables draw them as one unit box; this pass
writes a renderer-owned copy of ``geom_xmat`` with ``diag(geom_size)`` folded into those geoms' columns, which the
raster pass and the ray-tracing instance refit read in place of the physics buffer (one dispatch over
n_envs x ngeom, encoded into the render command buffer after the physics wait)."""
from __future__ import annotations

import numpy as np
import Metal

from metalsim.interop import warp_metal as wm


class SizedGeoms:
    def __init__(self, ctx, model, n_envs: int, geom_ids):
        self.ctx, self.m, self.n = ctx, model, n_envs
        self.ids = [int(g) for g in geom_ids]
        flag = np.zeros(max(model.ngeom, 1), np.int32); flag[self.ids] = 1
        self.flag_buf = ctx.buffer(flag.nbytes, flag, "sized_flags")
        self.out = ctx.buffer(n_envs * model.ngeom * 36, label="sized_geom_xmat")
        self.consts = np.array([n_envs, model.ngeom, 0, 0], np.uint32)
        self.consts_buf = ctx.buffer(16, self.consts, "sized_consts")
        self.pipe = ctx.compute_pipeline(ctx.library("sized"), "scale_geom_xmat")
        self._host_size = None

    def encode(self, cb, xmat_buf, xmat_off, geom_size):
        """``geom_size``: MuJoCo Warp's ``Model.geom_size`` (per world when BatchSim has ``per_world_fields`` with
        it, else shared). Returns the (buffer, offset) to bind as geom_xmat."""
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
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n * self.m.ngeom, 1, 1), Metal.MTLSize(64, 1, 1))
        ce.endEncoding()
        return self.out, 0

    def scale_host(self, gm: np.ndarray, sizes: np.ndarray | None = None) -> np.ndarray:
        """Host path: ``gm`` (n, ngeom, 9) from MjData; ``sizes`` (n, ngeom, 3) per-world sizes (default the model's)."""
        gm = gm.copy()
        s = np.broadcast_to(self.m.geom_size if sizes is None else sizes, (self.n, self.m.ngeom, 3))
        for g in self.ids:
            gm[:, g] = (gm[:, g].reshape(-1, 3, 3) * s[:, g][:, None, :]).reshape(-1, 9)
        return gm
