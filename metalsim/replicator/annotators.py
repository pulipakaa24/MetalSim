"""Annotators: the renderer's outputs plus derived labels, computed on the GPU.

Available: rgb, depth (distance to image plane), normals, instance_segmentation (renderer slot
ids), semantic_segmentation (class id per body via the semantic table), bounding_box_2d_tight
(from instance masks), bounding_box_3d (oriented, from physics geom poses and MuJoCo's geom
AABBs), camera_params (intrinsics, extrinsics per env). Mirrors Isaac Replicator's annotator
names where the semantics match.
"""
from __future__ import annotations

import numpy as np
import torch


class Annotators:
    def __init__(self, renderer, sim=None, class_of_body=None):
        """``class_of_body``: optional mapping body id -> class id; default class = body id."""
        self.r = renderer
        self.sim = sim
        t = renderer.tables
        self.dev = torch.device("mps")
        self.slot_body = torch.as_tensor(t.semantic[:, 1], device=self.dev)
        self.slot_geom = torch.as_tensor(t.semantic[:, 0], device=self.dev)
        nb = int(renderer.m.nbody)
        cls = np.arange(nb) if class_of_body is None else np.array([class_of_body.get(b, b) for b in range(nb)])
        self.body_class = torch.as_tensor(cls, device=self.dev)
        aabb = renderer.m.geom_aabb.reshape(-1, 2, 3)   # per geom: center, half-size (geom frame)
        self.geom_aabb = torch.as_tensor(aabb.astype(np.float32), device=self.dev)

    # -- pixel annotators ------------------------------------------------------------------------------

    def rgb(self):
        return self.r.out.rgb

    def depth(self):
        return self.r.out.depth

    def normals(self):
        return self.r.out.normal

    def instance_segmentation(self):
        """(N,H,W) int32: 0 background, slot+1 otherwise (renderer seg_mode SEG_SLOT)."""
        return self.r.out.seg

    def semantic_segmentation(self):
        seg = self.r.out.seg
        lut = torch.cat([torch.tensor([-1], device=self.dev), self.body_class[self.slot_body]]) + 1
        return lut[seg.long()].to(torch.int32)      # 0 background, class+1 otherwise

    def bounding_box_2d_tight(self):
        """(N, G, 4) float32 [x0, y0, x1, y1] in pixels per env and slot; -1 where absent."""
        seg = self.r.out.seg
        n, h, w = seg.shape
        G = self.r.G
        ys = torch.arange(h, device=self.dev).view(1, h, 1).expand(n, h, w)
        xs = torch.arange(w, device=self.dev).view(1, 1, w).expand(n, h, w)
        idx = (seg.long() - 1).clamp(min=0)
        present = seg > 0
        big = torch.full((n, G + 1), 1e9, device=self.dev)
        x0 = big.clone().scatter_reduce(1, idx.reshape(n, -1) + (~present).reshape(n, -1).long() * G, torch.where(present, xs, torch.full_like(xs, 10**9)).float().reshape(n, -1), reduce="amin")
        y0 = big.clone().scatter_reduce(1, idx.reshape(n, -1) + (~present).reshape(n, -1).long() * G, torch.where(present, ys, torch.full_like(ys, 10**9)).float().reshape(n, -1), reduce="amin")
        neg = torch.full((n, G + 1), -1.0, device=self.dev)
        x1 = neg.clone().scatter_reduce(1, idx.reshape(n, -1) + (~present).reshape(n, -1).long() * G, torch.where(present, xs, torch.full_like(xs, -1)).float().reshape(n, -1), reduce="amax")
        y1 = neg.clone().scatter_reduce(1, idx.reshape(n, -1) + (~present).reshape(n, -1).long() * G, torch.where(present, ys, torch.full_like(ys, -1)).float().reshape(n, -1), reduce="amax")
        box = torch.stack([x0, y0, x1 + 1, y1 + 1], -1)[:, :G]
        absent = box[..., 2] <= 0
        box[absent] = -1
        return box

    # -- state annotators ---------------------------------------------------------------------------------

    def bounding_box_3d(self):
        """(N, G, 10): world center xyz, half-size xyz (geom frame), quaternion wxyz of the geom.
        Requires a BatchSim (reads geom_xpos / geom_xmat)."""
        assert self.sim is not None
        gid = self.slot_geom
        xpos = self.sim.t.geom_xpos[:, gid]                    # (N, G, 3)
        xmat = self.sim.t.geom_xmat[:, gid]                    # (N, G, 3, 3)
        c = self.geom_aabb[gid, 0]; h = self.geom_aabb[gid, 1]  # (G, 3)
        center = xpos + torch.einsum("ngij,gj->ngi", xmat, c)
        m = xmat
        w = torch.sqrt(torch.clamp(1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2], min=1e-8)) / 2
        x = (m[..., 2, 1] - m[..., 1, 2]) / (4 * w); y = (m[..., 0, 2] - m[..., 2, 0]) / (4 * w); z = (m[..., 1, 0] - m[..., 0, 1]) / (4 * w)
        return torch.cat([center, h.expand(center.shape[0], -1, -1), torch.stack([w, x, y, z], -1)], -1)

    def camera_params(self):
        """Intrinsics (N,4) fx fy cx cy and extrinsics (N,3), (N,3,3) from the sim's camera state."""
        cs = self.r.t_cam_spec
        intr = cs[:, 0:4].clone()
        if self.sim is None:
            return {"intrinsics": intr}
        pos = self.sim.t.cam_xpos[:, self.r.cam_id] + cs[:, 4:7]
        return {"intrinsics": intr, "position": pos, "rotation": self.sim.t.cam_xmat[:, self.r.cam_id]}
