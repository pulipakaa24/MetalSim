"""Per-env material and texture swaps in every renderer (shaders/appearance.h ``MatEntry``, model_sync.metal
``resolve_mat_ids``).

Two sources choose, per (env, slot), a material entry other than the slot's own:

* **the model**: a MuJoCo Warp model whose ``geom_matid`` is batched per world (a material randomization writing
  ``sim.tm.geom_matid``). A geom whose world material differs from its compiled one is drawn with that material's whole
  block (texture with MuJoCo's mapping for this geom, metallic / roughness / specular / emission / reflectance), as
  ``mujoco.Renderer`` draws a model compiled with that material. Its colour already followed (``model_sync``).
* **a texture pool**, as Isaac Lab's ``randomize_visual_texture_material`` event term: ``set_texture_pool(images)``
  makes one entry per image (and rotation) that draws the image as the diffuse texture of a new default material
  (OmniPBR's defaults: metallic 0, roughness 0.5, specular 0.5), mapped by OmniPBR's ``project_uvw`` box projection of
  the geom-frame position (``projection="uvw"``, Isaac's choice) or by the surface's own coordinates
  (``projection="uv"``); ``texture_scale`` / ``rotations_deg`` as Replicator's ``texture_scale`` / ``texture_rotate``.
  Choose per env and slot through ``t_material_ids`` (an (N, G) int32 tensor on the GPU; -1 = the slot's own) or
  ``set_material_ids``. Unlike Isaac Lab (which needs ``replicate_physics=False`` and redraws every env on each call),
  each env's choice is its own and costs nothing per frame beyond a table lookup.

Entries live in one table (``entries``, (K, 32) float32: material row, extension row, pool mapping). The resolved
per-(env, slot) index is written each frame into the model sync's per-instance buffer (``ModelSync.ids_off``). With
neither source active nothing is dispatched and the shaders skip the lookup (scenes without them render byte-identically).
"""
from __future__ import annotations

import numpy as np
import Metal

ENTRY_WORDS = 32
_MAX_MODEL_ENTRIES = 1 << 18


class EnvMaterials:
    def __init__(self, ctx, model, tables, n_envs: int, sync, device="metal:0"):
        import warp as wp
        from metalsim.interop import torch_bridge as tb
        from metalsim.interop import warp_metal as wm
        self.ctx, self.m, self.t, self.n, self.sync = ctx, model, tables, int(n_envs), sync
        G = tables.G
        self.G = G
        G0 = len(tables.geoms)
        self.entries = np.zeros((0, ENTRY_WORDS), np.float32)
        self.slot_base = np.full(G, -1, np.int32)
        compiled = np.full(G, -2, np.int32)
        sg = np.zeros(G, np.int32)
        sg[:G0] = tables.geoms
        compiled[:G0] = [int(model.geom_matid[g]) for g in tables.geoms]
        sized = np.asarray(tables.sized, bool) if tables.sized is not None else np.zeros(G0, bool)
        if model.nmat and G0 * (model.nmat + 1) <= _MAX_MODEL_ENTRIES:
            from metalsim.render.scene_tables import material_block
            rows = []
            for s, g in enumerate(tables.geoms):
                if s < len(sized) and sized[s]:
                    continue                      # terrain slots keep the terrain's material
                self.slot_base[s] = len(rows)
                for mid in range(-1, model.nmat):
                    row, ext = material_block(model, g, mid, tables.atlas)
                    e = np.zeros(ENTRY_WORDS, np.float32)
                    e[0:16] = row; e[0:4] = 1.0; e[16:24] = ext
                    rows.append(e)
            if rows:
                self.entries = np.asarray(rows, np.float32)
        self.n_model = len(self.entries)
        c = ctx
        self.slot_geom_buf = c.buffer(4 * G, sg, "envmat_slot_geom")
        self.compiled_buf = c.buffer(4 * G, compiled, "envmat_compiled")
        self.base_buf = c.buffer(4 * G, self.slot_base, "envmat_slot_base")
        self._user_wp = wp.full((self.n, G), -1, dtype=wp.int32, device=device)
        self.t_material_ids = tb.mps_tensor(self._user_wp)        # (N, G) int32, GPU-writable
        self.user_buf = wm.buffer_of(self._user_wp).buffer
        self.user_active = False
        self.pool_tex = c.texture2d(1, 1, Metal.MTLPixelFormatRGBA8Unorm, Metal.MTLTextureUsageShaderRead, array_length=1,
                                    label="texture_pool", storage=Metal.MTLStorageModeShared)
        self.pool_size = 0
        self.version = 0                   # bumped when the entry table changes (renderers re-upload it)
        self.entries_buf = c.buffer(max(self.entries.nbytes, 16 * 4 * 8), self.entries if len(self.entries) else None,
                                    "envmat_entries")
        self.p_resolve = c.compute_pipeline(c.library("model_sync"), "resolve_mat_ids")
        self.consts = np.zeros(8, np.uint32)

    # -- the texture pool (Isaac Lab randomize_visual_texture_material) ----------------------------------------------

    def set_texture_pool(self, images, *, projection: str = "uvw", texture_scale=(1.0, 1.0), rotations_deg=(0.0,),
                         size: int | None = None, srgb: bool = True):
        """Replace the texture pool with ``images`` ((H, W, 3|4) uint8 each; resized to ``size`` x ``size``, default the
        largest side up to 1024, bilinear) and add one entry per (image, rotation). Returns (n_images, n_rotations)
        int32 entry ids for ``t_material_ids`` / ``set_material_ids``."""
        if projection not in ("uvw", "uv"):
            raise ValueError(f"projection must be 'uvw' or 'uv', got {projection!r}")
        imgs = [np.asarray(im) for im in images]
        if not imgs:
            raise ValueError("empty texture pool")
        S = int(size or min(1024, max(max(im.shape[:2]) for im in imgs)))
        c = self.ctx
        tex = c.texture2d(S, S, Metal.MTLPixelFormatRGBA8Unorm, Metal.MTLTextureUsageShaderRead, array_length=len(imgs),
                          label="texture_pool", storage=Metal.MTLStorageModeShared)
        for k, im in enumerate(imgs):
            c.upload_texture(tex, _rgba_resized(im, S), k)
        self.pool_tex, self.pool_size = tex, len(imgs)
        rots = [float(r) for r in rotations_deg]
        ent = np.zeros((len(imgs) * len(rots), ENTRY_WORDS), np.float32)
        ent[:, 0:4] = 1.0
        ent[:, 4:8] = (0.0, 0.5, 0.5, 0.0)                     # OmniPBR defaults: metallic, roughness, specular
        ent[:, 12:14] = texture_scale
        ent[:, 14] = 1.0
        ent[:, 16 + 3] = 1.0 if srgb else 0.0                  # MaterialExt e0.w: sRGB decode
        ent[:, 20:24] = (0.0, 0.0, 0.0, 0.0)
        ent[:, 20:23] = 1.0; ent[:, 23] = 1.0                  # e1: cube scale / faces (unused)
        ids = np.zeros((len(imgs), len(rots)), np.int32)
        for k in range(len(imgs)):
            for j, r in enumerate(rots):
                q = k * len(rots) + j
                ent[q, 24:28] = (1.0, float(k), 1.0 if projection == "uvw" else 0.0, np.deg2rad(r))
                ids[k, j] = self.n_model + q
        self.entries = np.concatenate([self.entries[:self.n_model], ent])
        self.entries_buf = c.buffer(self.entries.nbytes, self.entries, "envmat_entries")
        self.version += 1
        return ids

    def set_material_ids(self, ids):
        """(N, G) int32 entry per env and slot (-1: the slot's own material); host write (``t_material_ids`` on the GPU)."""
        import torch
        import warp as wp
        a = np.asarray(ids, np.int32).reshape(self.n, self.G)
        if a.max(initial=-1) >= len(self.entries):
            raise ValueError(f"material id {int(a.max())} >= {len(self.entries)} entries")
        wp.synchronize_device(); torch.mps.synchronize()
        self._user_wp.assign(a)
        wp.synchronize_device()
        self.user_active = True

    # -- per frame ----------------------------------------------------------------------------------------------------

    def _model_matid(self, sim):
        """(buffer, offset, worlds) of a per-world geom_matid of ``sim``'s model, or None."""
        if sim is None or not self.n_model:
            return None
        import warp as wp
        from metalsim.interop import warp_metal as wm
        from metalsim.render.model_sync import model_fields
        a = getattr(model_fields(sim), "geom_matid", None)
        if not isinstance(a, wp.array) or a.ndim < 2 or a.shape[0] <= 1:
            return None
        if a.shape[0] != self.n:
            raise ValueError(f"geom_matid has {a.shape[0]} worlds, renderer {self.n}")
        v = wm.buffer_of(a)
        return v.buffer, v.offset

    def active(self, sim=None) -> bool:
        return bool(self.user_active or self.pool_size or self._model_matid(sim) is not None)

    def encode(self, cb, sim=None) -> bool:
        """Write the resolved (env, slot) entries into the model sync's buffer; returns whether any source is active
        (False: nothing encoded, the shaders skip the lookup)."""
        mm = self._model_matid(sim)
        use_user = self.user_active or self.pool_size > 0
        if mm is None and not use_user:
            return False
        self.consts[:6] = (self.n, self.G, self.m.ngeom, 1, 1 if mm is not None else 0, 1 if use_user else 0)
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.p_resolve)
        ce.setBuffer_offset_atIndex_(self.user_buf, 0, 0)
        if mm is not None:
            ce.setBuffer_offset_atIndex_(mm[0], mm[1], 1)
        else:
            ce.setBuffer_offset_atIndex_(self.compiled_buf, 0, 1)
        ce.setBuffer_offset_atIndex_(self.slot_geom_buf, 0, 2)
        ce.setBuffer_offset_atIndex_(self.compiled_buf, 0, 3)
        ce.setBuffer_offset_atIndex_(self.base_buf, 0, 4)
        ce.setBuffer_offset_atIndex_(self.sync.inst_col, self.sync.ids_off, 5)
        ce.setBytes_length_atIndex_(self.consts.tobytes(), self.consts.nbytes, 6)
        n = self.n * self.G
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(n, 1, 1), Metal.MTLSize(min(n, 64), 1, 1))
        ce.endEncoding()
        return True

    def host_ids(self):
        """(N, G) int32 resolved entries of the last frame (synchronize first)."""
        a = self.ctx.buffer_array(self.sync.inst_col, np.int32, (self.sync.ids_off // 4 + self.n * self.G,))
        return a[self.sync.ids_off // 4:].reshape(self.n, self.G)


def _rgba_resized(im: np.ndarray, S: int) -> np.ndarray:
    """(S, S, 4) uint8 bilinear resize of an (H, W, 3|4) image (row 0 = t = 0, as the atlas stores textures)."""
    im = np.asarray(im)
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, -1)
    if im.shape[-1] == 3:
        im = np.concatenate([im, np.full(im.shape[:2] + (1,), 255, im.dtype)], -1)
    H, W = im.shape[:2]
    if (H, W) == (S, S):
        return np.ascontiguousarray(im.astype(np.uint8))
    y = (np.arange(S) + 0.5) * H / S - 0.5
    x = (np.arange(S) + 0.5) * W / S - 0.5
    y0 = np.clip(np.floor(y).astype(int), 0, H - 1); y1 = np.clip(y0 + 1, 0, H - 1); fy = np.clip(y - y0, 0, 1)[:, None, None]
    x0 = np.clip(np.floor(x).astype(int), 0, W - 1); x1 = np.clip(x0 + 1, 0, W - 1); fx = np.clip(x - x0, 0, 1)[None, :, None]
    f = im.astype(np.float64)
    out = (f[y0][:, x0] * (1 - fy) * (1 - fx) + f[y0][:, x1] * (1 - fy) * fx + f[y1][:, x0] * fy * (1 - fx) + f[y1][:, x1] * fy * fx)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
