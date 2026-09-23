"""Domain randomization over the batched scene. Everything the renderer reads per env (instance
colours, camera pose delta, DR light, ambient, background) is a GPU tensor, so randomizing is a
handful of torch kernels on the MPS device and free at render time. Material table changes
(roughness/metallic/texture assignment) are per batch on the host, matching Isaac Replicator's
per-frame `rep.randomizer.materials`/`texture` semantics at dataset rates.
"""
from __future__ import annotations

import numpy as np
import torch


class Randomizer:
    def __init__(self, renderer, seed: int = 0, device="mps"):
        self.r = renderer
        self.dev = torch.device(device)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.rng = np.random.default_rng(seed)

    def _u(self, lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=self.gen, device=self.dev)

    def _mask(self, mask):
        n = self.r.n
        if mask is None:
            return torch.ones(n, dtype=torch.bool, device=self.dev)
        return mask.to(self.dev, torch.bool)

    # -- per-env GPU randomizers ---------------------------------------------------------------------

    def colors(self, mask=None, lo=0.1, hi=0.95, jitter=0.12, slots=None, fixed: dict | None = None):
        """Per-env base colour with per-slot jitter (Isaac `rep.randomizer.color`). ``fixed`` maps
        slot -> rgb kept constant (e.g. the target object)."""
        n, G = self.r.n, self.r.G
        m = self._mask(mask).view(n, 1, 1).float()
        base = self._u(lo, hi, n, 1, 3)
        cols = torch.clamp(base + self._u(-jitter, jitter, n, G, 3), 0, 1)
        if fixed:
            for slot, rgb in fixed.items():
                cols[:, slot] = torch.tensor(rgb, device=self.dev)
        c = self.r.t_colors
        if slots is not None:
            keep = torch.ones(G, device=self.dev); keep[list(slots)] = 0
            cols = keep.view(1, G, 1) * c[:, :, :3] + (1 - keep.view(1, G, 1)) * cols
        c[:, :, :3] = m * cols + (1 - m) * c[:, :, :3]

    def camera(self, mask=None, pos_jitter=0.015, rot_jitter=0.01):
        n = self.r.n
        m = self._mask(mask).view(n, 1).float()
        cs = self.r.t_cam_spec
        rot = torch.cat([torch.ones(n, 1, device=self.dev), self._u(-rot_jitter, rot_jitter, n, 3)], 1)
        rot = rot / rot.norm(dim=1, keepdim=True)
        cs[:, 4:7] = m * self._u(-pos_jitter, pos_jitter, n, 3) + (1 - m) * cs[:, 4:7]
        cs[:, 8:12] = m * rot + (1 - m) * cs[:, 8:12]

    def intrinsics(self, mask=None, focal_scale=(0.9, 1.1), principal_jitter_px=2.0):
        """Focal length and principal point jitter (Isaac randomizes camera focal/aperture)."""
        n = self.r.n
        m = self._mask(mask).view(n, 1).float()
        cs = self.r.t_cam_spec
        base = torch.tensor([float(self.r.intrinsics.fx), float(self.r.intrinsics.fy), float(self.r.intrinsics.cx), float(self.r.intrinsics.cy)], dtype=torch.float32, device=self.dev)
        f = self._u(focal_scale[0], focal_scale[1], n, 1)
        new = torch.cat([base[:2] * f, base[2:] + self._u(-principal_jitter_px, principal_jitter_px, n, 2)], 1)
        cs[:, 0:4] = m * new + (1 - m) * cs[:, 0:4]

    def light(self, mask=None, base_dir=(0.3, 0.3, -0.9), jitter=0.15, intensity=(0.8, 1.2)):
        n = self.r.n
        m = self._mask(mask).view(n, 1).float()
        cs = self.r.t_cam_spec
        d = torch.tensor(base_dir, device=self.dev) + self._u(-jitter, jitter, n, 3)
        cs[:, 12:15] = m * d + (1 - m) * cs[:, 12:15]
        cs[:, 15:16] = m * self._u(intensity[0], intensity[1], n, 1) + (1 - m) * cs[:, 15:16]

    def ambient(self, mask=None, scale=(0.6, 1.4)):
        n = self.r.n
        m = self._mask(mask).view(n, 1).float()
        cs = self.r.t_cam_spec
        cs[:, 16:19] = m * self._u(scale[0], scale[1], n, 1).expand(n, 3) + (1 - m) * cs[:, 16:19]

    # -- host-side (per batch) randomizers ------------------------------------------------------------

    def backgrounds(self, env_ids=None, images=None, procedural=True):
        """Background images per env (photos of the deployment scene, or procedural fills)."""
        env_ids = list(range(self.r.bg_layers)) if env_ids is None else list(env_ids)
        T = self.r.th
        if images is None:
            images = []
            for _ in env_ids:
                kind = self.rng.integers(0, 3)
                if kind == 0:
                    im = np.full((T, self.r.tw, 3), self.rng.integers(0, 256, 3), np.uint8)
                elif kind == 1:
                    a = self.rng.integers(0, 256, 3).astype(np.float32); b = self.rng.integers(0, 256, 3).astype(np.float32)
                    t = np.linspace(0, 1, T)[:, None, None]
                    im = np.broadcast_to((a * (1 - t) + b * t).astype(np.uint8), (T, self.r.tw, 3)).copy()
                else:
                    small = self.rng.integers(0, 256, (8, 8, 3)).astype(np.uint8)
                    im = small[np.ix_(np.linspace(0, 7, T).astype(int), np.linspace(0, 7, self.r.tw).astype(int))]
                images.append(im)
        self.r.set_backgrounds(env_ids, images)

    def materials(self, roughness=(0.1, 1.0), metallic=(0.0, 0.3), slots=None):
        """Per-slot PBR parameters (shared across envs); host write of the material table."""
        mats = self.r.tables.materials
        idx = range(self.r.G) if slots is None else slots
        for s in idx:
            mats[s, 5] = self.rng.uniform(*roughness)
            mats[s, 4] = self.rng.uniform(*metallic)
        self.r.ctx.write_buffer(self.r.mat_buf, mats)
