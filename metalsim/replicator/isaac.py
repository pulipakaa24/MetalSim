"""Isaac Sim Replicator annotators over MetalSim's renderer outputs and physics state.

Everything is batched over envs on the MPS device (one "render product" per env, as an Isaac Lab
``TiledCamera``); the structured numpy records Isaac returns per render product are produced only
by ``records(...)`` / the writers, on the CPU, after one synchronization.

Sources (no renderer changes):

==========================  ==================================================================
annotator                   derived from
==========================  ==================================================================
rgb / LdrColor              renderer ``rgb`` (RGBA with alpha 255)
distance_to_image_plane     renderer ``depth`` (linearized z); background 0 -> inf
distance_to_camera          depth x per-pixel ray length (from the per-env intrinsics)
normals                     renderer ``normal`` (raster world-frame shading normal); if the renderer
                            has no normal output: ``normals_from_depth`` (depth-derived, flagged)
semantic_segmentation       renderer id buffer (``seg`` in slot or geom mode) -> semantic LUT
instance_segmentation       id buffer -> labelled-prim (instance) LUT
instance_id_segmentation    id buffer -> render prim (geom) ids
bounding_box_2d_tight       min/max of instance pixel coordinates (GPU scatter-reduce)
bounding_box_2d_loose       tight box of the *unoccluded* instance silhouette (one extra seg-only
                            render per instance through a private renderer; ``occlusion=True``),
                            else the projected render-mesh vertices clipped to the image
occlusionRatio              1 - visible pixels / unoccluded pixels (``occlusion=True``)
bounding_box_3d             instance extents in its body frame (geom AABBs from the model) + body pose
pointcloud                  depth back-projected to world, with rgb / normal / semantic / instance
motion_vectors              state-derived: each pixel's surface point moved with its geom from the
                            previous state and reprojected with the previous camera
camera_params               per-env intrinsics and camera pose (USD conventions)
==========================  ==================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from metalsim.replicator.semantics import BACKGROUND, UNLABELLED, Semantics

SEG_SLOT, SEG_GEOM, SEG_BODY = 0, 1, 2
BBOX2D_DTYPE = np.dtype([("semanticId", "<u4"), ("x_min", "<i4"), ("y_min", "<i4"), ("x_max", "<i4"),
                         ("y_max", "<i4"), ("occlusionRatio", "<f4")])
BBOX3D_DTYPE = np.dtype([("semanticId", "<u4"), ("x_min", "<f4"), ("y_min", "<f4"), ("z_min", "<f4"),
                         ("x_max", "<f4"), ("y_max", "<f4"), ("z_max", "<f4"), ("transform", "<f4", (4, 4)),
                         ("occlusionRatio", "<f4")])


def _qmat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


@dataclass
class State:
    """Poses the state annotators read: (N, ngeom, 3/3x3), (N, nbody, ...), (N, ncam, ...)."""
    geom_xpos: torch.Tensor
    geom_xmat: torch.Tensor
    xpos: torch.Tensor
    xmat: torch.Tensor
    cam_xpos: torch.Tensor
    cam_xmat: torch.Tensor

    @classmethod
    def from_sim(cls, sim, clone=False):
        n = sim.n
        t = sim.t
        f = (lambda x: x.clone()) if clone else (lambda x: x)
        return cls(f(t.geom_xpos.reshape(n, -1, 3)), f(t.geom_xmat.reshape(n, -1, 3, 3)), f(t.xpos.reshape(n, -1, 3)),
                   f(t.xmat.reshape(n, -1, 3, 3)), f(t.cam_xpos.reshape(n, -1, 3)), f(t.cam_xmat.reshape(n, -1, 3, 3)))

    @classmethod
    def from_datas(cls, datas, device="mps"):
        def g(name, shape):
            return torch.as_tensor(np.stack([getattr(d, name) for d in datas]).astype(np.float32).reshape((len(datas),) + shape), device=device)
        return cls(g("geom_xpos", (-1, 3)), g("geom_xmat", (-1, 3, 3)), g("xpos", (-1, 3)), g("xmat", (-1, 3, 3)),
                   g("cam_xpos", (-1, 3)), g("cam_xmat", (-1, 3, 3)))


class IsaacAnnotators:
    def __init__(self, renderer, semantics: Semantics | None = None, sim=None, semantic_types=("class",),
                 seg_mode: int | None = None, decimate_faces: int = 0, impl: str = "warp", loose_hull: bool = True):
        """``renderer``: a Tier0Renderer (or anything with ``out``, ``tables``, ``t_cam_spec``,
        ``cam_id``, ``tw``, ``th``, ``n``) whose outputs include ``seg`` in slot or geom mode.
        ``sim``: a BatchSim (state read live); or call ``set_state`` (host path). ``decimate_faces``
        must equal the renderer's (the occlusion pass renders the same meshes).

        ``impl``: "warp" (default) runs the per-pixel annotators and the 2-D extents as fused Warp
        kernels (``kernels.py``); "torch" keeps the first implementation (torch ops on MPS), rejected
        for cost but kept for comparison (same outputs; see docs/research/replicator_2026-09-25.md,
        decisions). ``loose_hull``: project only convex-hull vertices for the render-free loose box
        (exact; False projects every render-mesh vertex)."""
        if impl not in ("warp", "torch"):
            raise ValueError(impl)
        self.impl = impl
        self.loose_hull = loose_hull
        self.decimate_faces = decimate_faces
        self.r = renderer
        self.m = renderer.m
        self.sem = semantics or Semantics.from_body_names(self.m)
        self.sim = sim
        self._state = None
        self.dev = torch.device("mps")
        self.n, self.H, self.W = renderer.n, renderer.th, renderer.tw
        self.seg_mode = getattr(renderer, "seg_mode", SEG_SLOT) if seg_mode is None else seg_mode
        if self.seg_mode == SEG_BODY:
            raise ValueError("instance annotators need the slot or geom id buffer (seg_mode SEG_SLOT or SEG_GEOM)")
        t = renderer.tables
        m = self.m
        self.slot_geom = np.asarray(t.semantic[:, 0], np.int64)
        E = len(self.sem.entities)
        self.E = E
        ent_sem, self.id_to_labels = self.sem.semantic_table(semantic_types)
        # raw id-buffer value -> geom (-1 background)
        if self.seg_mode == SEG_SLOT:
            raw_geom = np.concatenate([[-1], self.slot_geom])
        else:
            raw_geom = np.arange(-1, m.ngeom)
        ge = self.sem.geom_entity
        raw_ent = np.array([-1 if g < 0 else ge[g] for g in raw_geom], np.int64)          # -1 background/unlabelled
        # instance ids: 0 background, 1 unlabelled, entity k -> k + 2
        raw_inst = np.where(raw_geom < 0, BACKGROUND, np.where(raw_ent < 0, UNLABELLED, raw_ent + 2))
        raw_semid = np.where(raw_geom < 0, BACKGROUND, np.where(raw_ent < 0, UNLABELLED, ent_sem[np.maximum(raw_ent, 0)] if E else UNLABELLED))
        self.lut_geom = torch.as_tensor(raw_geom, device=self.dev)
        self.lut_inst = torch.as_tensor(raw_inst, dtype=torch.int32, device=self.dev)
        self.lut_sem = torch.as_tensor(raw_semid, dtype=torch.int32, device=self.dev)
        self.lut_iid = torch.as_tensor(raw_geom + 1, dtype=torch.int32, device=self.dev)   # instance_id: geom + 1
        self.ent_sem = torch.as_tensor(ent_sem, dtype=torch.int64, device=self.dev)
        self.ent_body = torch.as_tensor([e.body for e in self.sem.entities], dtype=torch.long, device=self.dev)
        self.inst_to_path = {"0": "BACKGROUND", "1": "UNLABELLED"}
        self.inst_to_sem = {"0": {"class": "BACKGROUND"}, "1": {"class": "UNLABELLED"}}
        for k, e in enumerate(self.sem.entities):
            self.inst_to_path[str(k + 2)] = e.path
            self.inst_to_sem[str(k + 2)] = {t_: v for t_, v in e.labels.items() if t_ in semantic_types}
        self.iid_to_path = {"0": "BACKGROUND"}
        for g in (raw_geom[1:] if self.seg_mode == SEG_SLOT else range(m.ngeom)):
            self.iid_to_path[str(int(g) + 1)] = self.sem.geom_paths[int(g)]
        self._local_extents = self._entity_local_extents()
        self._slot_verts = None
        # pixel-centre grid
        ys, xs = torch.meshgrid(torch.arange(self.H, device=self.dev, dtype=torch.float32) + 0.5,
                                torch.arange(self.W, device=self.dev, dtype=torch.float32) + 0.5, indexing="ij")
        self.u, self.v = xs, ys
        self._occ = None
        import warp as wp
        from metalsim.replicator.kernels import Bridge
        self.kb = Bridge(getattr(renderer, "device", "metal:0"))
        self._w_raw_geom = wp.array(raw_geom.astype(np.int32), dtype=wp.int32, device=self.kb.device)
        self._w_raw_inst = wp.array(raw_inst.astype(np.int32), dtype=wp.int32, device=self.kb.device)
        # the occlusion renderer's id buffer is in geom mode: geom + 1 -> instance id
        geom_inst = np.where(ge < 0, UNLABELLED, ge + 2) if m.ngeom else np.zeros(0, np.int64)
        self._w_geom_inst = wp.array(np.concatenate([[BACKGROUND], geom_inst]).astype(np.int32), dtype=wp.int32, device=self.kb.device)
        wp.synchronize_device(self.kb.device)

    # -- state -------------------------------------------------------------------------------------------

    def set_state(self, state: State):
        self._state = state

    @property
    def state(self) -> State:
        if self._state is not None:
            return self._state
        if self.sim is None:
            raise ValueError("no state: pass sim= or call set_state()")
        return State.from_sim(self.sim)

    def camera_pose(self, state: State | None = None):
        """(N,3) world position and (N,3,3) camera-to-world rotation (MuJoCo/USD camera: -Z forward,
        +Y up), including the renderer's per-env camera deltas."""
        s = state or self.state
        cs = self.r.t_cam_spec
        pos = s.cam_xpos[:, self.r.cam_id] + cs[:, 4:7]
        R = s.cam_xmat[:, self.r.cam_id] @ _qmat(cs[:, 8:12] / cs[:, 8:12].norm(dim=1, keepdim=True))
        return pos, R

    def intrinsics(self):
        cs = self.r.t_cam_spec
        return cs[:, 0], cs[:, 1], cs[:, 2], cs[:, 3]

    # -- pixel annotators -------------------------------------------------------------------------------------

    def rgb(self, alpha=True):
        rgb = self.r.out.rgb
        if not alpha:
            return rgb
        return torch.cat([rgb, torch.full(rgb.shape[:-1] + (1,), 255, dtype=torch.uint8, device=self.dev)], -1)

    def distance_to_image_plane(self):
        d = self.r.out.depth
        return torch.where(d > 0, d, torch.full_like(d, math.inf))

    def _ray_scale(self):
        fx, fy, cx, cy = self.intrinsics()
        a = (self.u[None] - cx[:, None, None]) / fx[:, None, None]
        b = (self.v[None] - cy[:, None, None]) / fy[:, None, None]
        return a, b

    def _w(self, name):
        """The renderer's output ``name`` as its Warp array (tier 0: ``_out_arrays``, tier 2: ``_arrays``)."""
        arrs = getattr(self.r, "_out_arrays", None) or getattr(self.r, "_arrays", {})
        return arrs.get(name)

    def _w_cam_spec(self):
        a = getattr(self.r, "_cam_spec_wp", None)
        return a if a is not None else self.r._cam_wp

    def _w_camera(self, state: State | None = None, tag="cur"):
        import warp as wp
        pos, R = self.camera_pose(state)
        wpos, tpos = self.kb.buffer(f"campos_{tag}", (self.n,), wp.vec3)
        wrot, trot = self.kb.buffer(f"camrot_{tag}", (self.n,), wp.mat33)
        tpos.copy_(pos.reshape(tpos.shape)); trot.copy_(R.reshape(trot.shape))
        return wpos, wrot

    def _pixel_pass(self):
        """One fused pass: distance_to_camera, world points, unit normals (see kernels.pixel_kernel)."""
        import warp as wp
        from metalsim.replicator import kernels as K
        n, H, W = self.n, self.H, self.W
        wd, td = self.kb.buffer("dist_cam", (n, H, W))
        ww, tw = self.kb.buffer("world", (n, H, W, 3))
        wn, tn = self.kb.buffer("normals", (n, H, W, 4))
        normal = self._w("normal")
        wpos, wrot = self._w_camera()
        dummy = normal if normal is not None else self.kb.buffer("dummy_n", (1, 1, 1, 3))[0]
        self.kb.run(K.pixel_kernel, (n, H, W), [self._w("depth"), dummy, self._w_cam_spec(), wpos, wrot,
                                                 1 if normal is not None else 0, wd, ww, wn])
        return td, tw, tn

    def distance_to_camera(self):
        """(N,H,W) float32 Euclidean distance to the camera centre; inf background (fused kernel)."""
        if self.impl == "torch":
            return self._torch_distance_to_camera()
        return self._pixel_pass()[0]

    def camera_points(self):
        """(N,H,W,3) camera-frame points (x right, y up, -z forward); background -> 0."""
        d = self.r.out.depth
        a, b = self._ray_scale()
        return torch.stack([d * a, -d * b, -d], -1)

    def world_points(self, state: State | None = None):
        """(N,H,W,3) world-frame surface points; background -> 0 (fused kernel)."""
        if self.impl == "torch":
            return self._torch_world_points(state)
        return self._pixel_pass()[1]

    def normals(self):
        """(N,H,W,4) world-frame unit normals, w = 1 on geometry, 0 background (raster shading normal,
        stored by the renderer in fp16: direction within ~5e-4 rad). Falls back to
        ``normals_from_depth`` when the renderer has no normal output."""
        if self._w("normal") is None:
            return self.normals_from_depth()
        if self.impl == "torch":
            return self._torch_normals()
        return self._pixel_pass()[2]

    def normals_from_depth(self, frame="world"):
        """DEPTH-DERIVED normals (N,H,W,4): cross product of the back-projected neighbours' differences,
        using central differences where both neighbours hit the same render prim, one-sided otherwise;
        oriented towards the camera. Face interiors are exact up to depth-buffer precision (< 5e-3 rad in
        tests/test_replicator_annotators.py); pixels on a crease between two faces of one geom, on
        silhouettes and on curved surfaces are finite-difference estimates (a crease pixel mixes the two
        faces). w = 1 where a normal was formed."""
        if self.impl == "torch":
            return self._torch_normals_from_depth(frame)
        from metalsim.replicator import kernels as K
        n, H, W = self.n, self.H, self.W
        wo, to = self.kb.buffer("normals_depth", (n, H, W, 4))
        _, wrot = self._w_camera()
        self.kb.run(K.normals_from_depth_kernel, (n, H, W), [self._w("depth"), self._w("seg"), self._w_cam_spec(), wrot,
                                                             1 if frame == "world" else 0, wo])
        return to

    def semantic_segmentation(self):
        """(N,H,W) int32 semantic ids (0 BACKGROUND, 1 UNLABELLED) and ``{"idToLabels": ...}``."""
        return self.lut_sem[self.r.out.seg.long()], {"idToLabels": self.id_to_labels}

    def instance_segmentation(self):
        """(N,H,W) int32 instance ids of the labelled prims (0 BACKGROUND, 1 UNLABELLED), with
        ``idToLabels`` (id -> prim path) and ``idToSemantics`` (id -> labels)."""
        return self.lut_inst[self.r.out.seg.long()], {"idToLabels": self.inst_to_path, "idToSemantics": self.inst_to_sem}

    def instance_id_segmentation(self):
        """(N,H,W) int32 per render prim (geom) ids, ``idToLabels`` id -> prim path."""
        return self.lut_iid[self.r.out.seg.long()], {"idToLabels": self.iid_to_path}

    # -- boxes -------------------------------------------------------------------------------------------------

    def _extents(self, seg_w, lut_w, name, only=-1, reset=True):
        """(N,E,4) int32 inclusive pixel extents [x_min, y_min, x_max, y_max] and (N,E) pixel counts of
        every instance in an id buffer (fused kernel: per-row runs, then atomics)."""
        import warp as wp
        from metalsim.replicator import kernels as K
        wb, tbx = self.kb.buffer(f"box_{name}", (self.n, max(self.E, 1), 4), wp.int32)
        wc, tc = self.kb.buffer(f"cnt_{name}", (self.n, max(self.E, 1)), wp.int32)
        if reset:
            self.kb.run(K.reset_extents, (self.n, max(self.E, 1)), [wb, wc])
        self.kb.run(K.extents_kernel, (self.n, self.H), [seg_w, lut_w, only, wb, wc])
        return tbx[:, :self.E], tc[:, :self.E]

    def bounding_box_2d_tight(self, occlusion: bool = False):
        """GPU tensors: ``box`` (N,E,4) inclusive pixel extents, ``count`` (N,E) visible pixels,
        ``valid`` (N,E), ``semanticId`` (E,), ``occlusionRatio`` (N,E) (-1 unless ``occlusion``)."""
        if self.impl == "torch":
            inst, _ = self.instance_segmentation()
            box, cnt = self._torch_extents_from_ids(inst)
        else:
            box, cnt = self._extents(self._w("seg"), self._w_raw_inst, "tight")
            box, cnt = box.clone(), cnt.clone()
        occ = torch.full(cnt.shape, -1.0, device=self.dev)
        if occlusion:
            full = self.unoccluded()
            occ = torch.where(full["count"] > 0, 1 - cnt.float() / full["count"].clamp(min=1).float(), occ)
        return {"box": box, "count": cnt, "valid": cnt > 0, "semanticId": self.ent_sem, "occlusionRatio": occ}

    def bounding_box_2d_loose(self, occlusion: bool = True):
        """Loose box: extents of the whole instance regardless of occluders. With ``occlusion`` (default)
        the tight box of the unoccluded silhouette (extra seg-only render per instance); otherwise the
        render-mesh vertices projected and clipped to the image (a superset for curved meshes)."""
        if occlusion:
            full = self.unoccluded()
            vis = self.bounding_box_2d_tight(occlusion=False)
            occ = torch.where(full["count"] > 0, 1 - vis["count"].float() / full["count"].clamp(min=1).float(), torch.full(vis["count"].shape, -1.0, device=self.dev))
            return {"box": full["box"], "count": full["count"], "valid": vis["valid"], "semanticId": self.ent_sem, "occlusionRatio": occ}
        uv, infront = self.project_entity_vertices()
        lo = torch.where(infront.unsqueeze(-1), uv, torch.full_like(uv, math.inf)).amin(2)
        hi = torch.where(infront.unsqueeze(-1), uv, torch.full_like(uv, -math.inf)).amax(2)
        box = torch.stack([torch.ceil(lo[..., 0] - 0.5), torch.ceil(lo[..., 1] - 0.5),
                           torch.floor(hi[..., 0] - 0.5), torch.floor(hi[..., 1] - 0.5)], -1)
        box = torch.stack([box[..., 0].clamp(0, self.W - 1), box[..., 1].clamp(0, self.H - 1),
                           box[..., 2].clamp(0, self.W - 1), box[..., 3].clamp(0, self.H - 1)], -1).long()
        vis = self.bounding_box_2d_tight(occlusion=False)
        return {"box": box, "count": vis["count"], "valid": vis["valid"], "semanticId": self.ent_sem,
                "occlusionRatio": torch.full(vis["count"].shape, -1.0, device=self.dev)}

    def _vertices(self):
        """Per entity, the render-mesh vertices of its geoms in the geom frames: (V,3), geom id (V,),
        entity (V,) (static; built once)."""
        if self._slot_verts is None:
            t = self.r.tables
            vs, gs, es = [], [], []
            for slot, g in enumerate(self.slot_geom):
                e = self.sem.geom_entity[g]
                if e < 0:
                    continue
                mesh = t.meshes[int(t.geom_mesh[slot])]
                idx = np.unique(t.indices[mesh["i_off"]:mesh["i_off"] + mesh["i_count"]])
                v = t.vertices[idx, :3]
                # the extremes of a perspective projection of a point set lie on its convex hull
                # (projection maps segments to segments), so only hull vertices are kept: exact
                if self.loose_hull:
                    try:
                        from scipy.spatial import ConvexHull
                        v = v[ConvexHull(v.astype(np.float64)).vertices]
                    except Exception:
                        pass
                vs.append(v); gs.append(np.full(len(v), g)); es.append(np.full(len(v), e))
            self._slot_verts = tuple(torch.as_tensor(np.concatenate(a) if a else np.zeros((0,) + ((3,) if i == 0 else ())), device=self.dev)
                                     for i, a in enumerate((vs, gs, es)))
        return self._slot_verts

    def project_entity_vertices(self):
        """(N,E,Vmax,2) pixel coordinates of each instance's render-mesh vertices, (N,E,Vmax) in front
        of the camera (padding counts as behind)."""
        v, g, e = self._vertices()
        if not hasattr(self, "_vert_layout"):      # static: padded (entity, rank) slot of every vertex
            el = e.long().cpu()
            counts = torch.bincount(el, minlength=self.E)
            order = torch.argsort(el, stable=True)
            start = torch.cumsum(counts, 0) - counts
            rank = torch.arange(len(el)) - start[el[order]]
            self._vert_layout = (max(int(counts.max()) if len(el) else 0, 1), order.to(self.dev), el[order].to(self.dev), rank.to(self.dev))
        vmax, order, ent_sorted, rank = self._vert_layout
        s = self.state
        pw = torch.einsum("nvij,vj->nvi", s.geom_xmat[:, g.long()], v.float()) + s.geom_xpos[:, g.long()]
        pos, R = self.camera_pose(s)
        pc = torch.einsum("nji,nvj->nvi", R, pw - pos[:, None])
        fx, fy, cx, cy = self.intrinsics()
        z = -pc[..., 2]
        front = z > 1e-6
        u = cx[:, None] + fx[:, None] * pc[..., 0] / z.clamp(min=1e-6)
        vv = cy[:, None] - fy[:, None] * pc[..., 1] / z.clamp(min=1e-6)
        uv = torch.zeros(self.n, self.E, vmax, 2, device=self.dev)
        infr = torch.zeros(self.n, self.E, vmax, dtype=torch.bool, device=self.dev)
        uv[:, ent_sorted, rank] = torch.stack([u, vv], -1)[:, order]
        infr[:, ent_sorted, rank] = front[:, order]
        return uv, infr

    def _entity_local_extents(self):
        """(E,6) x_min..z_max of each instance in its body frame, from the geoms' model AABBs
        (``geom_aabb``: centre and half-size in the geom frame) transformed by the static geom-in-body
        pose (``geom_pos``/``geom_quat``)."""
        m = self.m
        out = np.zeros((len(self.sem.entities), 6), np.float64)
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)
        for k, ent in enumerate(self.sem.entities):
            pts = []
            for g in ent.geoms:
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                corners = c + signs * h
                R = np.zeros(9); import mujoco; mujoco.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
                p = corners @ R.T + m.geom_pos[g]
                b = int(m.geom_bodyid[g])
                # geoms of descendant bodies: chain to the entity body with the model's rest pose
                while b != ent.body and b != 0:
                    Rb = np.zeros(9); mujoco.mju_quat2Mat(Rb, m.body_quat[b]); p = p @ Rb.reshape(3, 3).T + m.body_pos[b]
                    b = int(m.body_parentid[b])
                pts.append(p)
            if pts:
                P = np.concatenate(pts)
                out[k] = np.concatenate([P.min(0), P.max(0)])
        return torch.as_tensor(out.astype(np.float32), device=self.dev)

    def bounding_box_3d(self, occlusion: bool = False):
        """``extents`` (E,6) local x_min..z_max, ``transform`` (N,E,4,4) local-to-world in USD's
        row-vector convention (rotation in the upper 3x3 transposed, translation in the last row),
        ``semanticId`` (E,), ``valid`` (N,E) (instance visible), ``occlusionRatio`` (N,E)."""
        s = self.state
        R = s.xmat[:, self.ent_body]
        p = s.xpos[:, self.ent_body]
        T = torch.zeros(self.n, self.E, 4, 4, device=self.dev)
        T[..., :3, :3] = R.transpose(-1, -2)
        T[..., 3, :3] = p
        T[..., 3, 3] = 1
        tight = self.bounding_box_2d_tight(occlusion=occlusion)
        return {"extents": self._local_extents, "transform": T, "semanticId": self.ent_sem, "valid": tight["valid"],
                "occlusionRatio": tight["occlusionRatio"]}

    # -- occlusion (unoccluded silhouettes through a private seg-only renderer) ---------------------------------

    def unoccluded(self):
        """Per instance, its silhouette rendered alone: (N,E,4) box and (N,E) pixel count. One seg-only
        render per instance (every other geom moved beyond the far plane) on the GPU path."""
        from metalsim.render.tier0 import Tier0Renderer
        import warp as wp
        from metalsim.interop import torch_bridge as tb
        if self._occ is None:
            r = self.r
            pr = Tier0Renderer(self.m, self.n, width=self.W, height=self.H, camera=r.cam_id, outputs=("seg",), seg_mode=SEG_GEOM,
                               shadows=False, decimate_faces=getattr(self, "decimate_faces", 0), ctx=r.ctx)
            ng, nc = self.m.ngeom, max(self.m.ncam, 1)

            class _D:
                pass
            d = _D()
            d.geom_xpos = wp.zeros((self.n, ng), dtype=wp.vec3, device=r.device)
            d.geom_xmat = wp.zeros((self.n, ng), dtype=wp.mat33, device=r.device)
            d.cam_xpos = wp.zeros((self.n, nc), dtype=wp.vec3, device=r.device)
            d.cam_xmat = wp.zeros((self.n, nc), dtype=wp.mat33, device=r.device)
            nl = max(self.m.nlight, 1)
            d.light_xpos = wp.zeros((self.n, nl), dtype=wp.vec3, device=r.device)
            d.light_xdir = wp.zeros((self.n, nl), dtype=wp.vec3, device=r.device)
            wp.synchronize_device(r.device)

            class _S:
                pass
            fake = _S(); fake.d = d
            from metalsim.interop import warp_metal as wm
            fake.event = wm.SharedEvent(r.device, "replicator.occlusion")
            self._occ = (pr, fake, {k: tb.mps_tensor(getattr(d, k)) for k in ("geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat")})
            ge = torch.as_tensor(self.sem.geom_entity, device=self.dev)
            self._geom_entity_t = ge
            self._occ_lut = torch.as_tensor(np.concatenate([[-1], self.sem.geom_entity]), device=self.dev)
        pr, fake, T = self._occ
        s = self.state
        pr.t_cam_spec.copy_(self.r.t_cam_spec)
        T["geom_xmat"].copy_(s.geom_xmat.reshape(T["geom_xmat"].shape)); T["cam_xpos"].copy_(s.cam_xpos.reshape(T["cam_xpos"].shape))
        T["cam_xmat"].copy_(s.cam_xmat.reshape(T["cam_xmat"].shape))
        far = torch.tensor([1e6, 1e6, 1e6], device=self.dev)
        from metalsim.interop import torch_bridge as tb
        boxes, counts = [], []
        for k in range(self.E):
            keep = (self._geom_entity_t == k).view(1, -1, 1)
            T["geom_xpos"].copy_(torch.where(keep, s.geom_xpos, far))
            v = fake.event.next_value()
            tb.signal_event(fake.event, v)
            rv = pr.render(fake, v)
            pr.after(rv)
            if self.impl == "torch":
                inst = torch.where(self._occ_lut[pr.out.seg.long()] == k, k + 2, 0)
                b, c = self._torch_extents_from_ids(inst)
                boxes.append(b[:, k]); counts.append(c[:, k])
            else:
                box, cnt = self._extents(pr._out_arrays["seg"], self._w_geom_inst, "occ", only=k, reset=(k == 0))
        if self.impl == "torch" and self.E:
            return {"box": torch.stack(boxes, 1), "count": torch.stack(counts, 1)}
        if self.E == 0:
            return {"box": torch.zeros(self.n, 0, 4, dtype=torch.int32, device=self.dev),
                    "count": torch.zeros(self.n, 0, dtype=torch.int32, device=self.dev)}
        return {"box": box.clone(), "count": cnt.clone()}

    # -- point cloud, motion vectors, camera --------------------------------------------------------------------

    def pointcloud(self, include_unlabelled: bool = False):
        """Dense GPU form: ``points`` (N,H,W,3) world, ``valid`` (N,H,W), ``rgb`` (N,H,W,4),
        ``normals`` (N,H,W,4), ``semantic`` / ``instance`` (N,H,W). ``records`` compacts per env."""
        inst, _ = self.instance_segmentation()
        sem, _ = self.semantic_segmentation()
        valid = self.r.out.depth > 0
        if not include_unlabelled:
            valid &= inst > UNLABELLED
        if self.impl == "torch":
            world, nrm = self.world_points(), self.normals()
        else:
            _, world, nrm = self._pixel_pass()
            if self._w("normal") is None:
                nrm = self.normals_from_depth()
        out = {"points": world, "valid": valid, "semantic": sem, "instance": inst, "normals": nrm}
        if self.r.out.rgb is not None:
            out["rgb"] = self.rgb()
        return out

    def motion_vectors(self, prev: State, state: State | None = None):
        """(N,H,W,4) state-derived motion: xy = pixel position of each pixel's surface point in the
        previous frame minus its position now (in pixels; Isaac: +x = motion to the left, +y = motion
        up), z = 0, w = 1 where defined. Each point moves rigidly with its geom; cameras from both
        states (fused kernel)."""
        if self.impl == "torch":
            return self._torch_motion_vectors(prev, state)
        import warp as wp
        from metalsim.replicator import kernels as K
        s = state or self.state
        n, H, W = self.n, self.H, self.W
        ng = self.m.ngeom
        bufs = {}
        for tag, st in (("cur", s), ("prev", prev)):
            wg, tg = self.kb.buffer(f"gpos_{tag}", (n, ng), wp.vec3)
            wr, tr = self.kb.buffer(f"grot_{tag}", (n, ng), wp.mat33)
            tg.copy_(st.geom_xpos.reshape(tg.shape)); tr.copy_(st.geom_xmat.reshape(tr.shape))
            bufs[tag] = (wg, wr) + self._w_camera(st, tag)
        wo, to = self.kb.buffer("motion", (n, H, W, 4))
        gc, rc, cpc, crc = bufs["cur"]
        gp, rp, cpp, crp = bufs["prev"]
        self.kb.run(K.motion_kernel, (n, H, W), [self._w("depth"), self._w("seg"), self._w_raw_geom, self._w_cam_spec(),
                                                  cpc, crc, cpp, crp, gc, rc, gp, rp, wo])
        return to

    def camera_params(self, focal_length_mm: float | None = None):
        """Per env (tensors): ``cameraViewTransform`` (N,4,4) world->camera, row-vector convention (as
        Isaac serialises it), ``cameraProjection`` (N,4,4) (OpenGL clip, row-vector), ``cameraFocalLength``
        (mm), ``cameraAperture`` (N,2) mm, ``cameraApertureOffset`` (N,2), ``cameraNearFar``,
        ``renderProductResolution``, ``metersPerSceneUnit``, ``cameraModel``."""
        pos, R = self.camera_pose()
        fx, fy, cx, cy = self.intrinsics()
        N = self.n
        V = torch.zeros(N, 4, 4, device=self.dev)
        V[:, :3, :3] = R            # row-vector form of world->camera: p_c^T = p_w^T R + t
        V[:, 3, :3] = -torch.einsum("ni,nij->nj", pos, R)
        V[:, 3, 3] = 1
        near, far = 0.01, 10.0
        W, H = float(self.W), float(self.H)
        P = torch.zeros(N, 4, 4, device=self.dev)   # column-vector clip matrix, then transposed
        P[:, 0, 0] = 2 * fx / W; P[:, 1, 1] = 2 * fy / H
        P[:, 0, 2] = 1 - 2 * cx / W; P[:, 1, 2] = 2 * cy / H - 1
        P[:, 2, 2] = far / (near - far); P[:, 2, 3] = near * far / (near - far); P[:, 3, 2] = -1
        m = self.m
        cam = self.r.cam_id
        if focal_length_mm is None:
            focal_length_mm = float(m.cam_intrinsic[cam][0] * 1000) if m.cam_sensorsize[cam][0] > 0 else 24.0
        f = torch.full((N,), focal_length_mm, device=self.dev)
        ap = torch.stack([W / fx * f, H / fy * f], -1)
        off = torch.stack([(cx - W / 2) / fx * f, (cy - H / 2) / fy * f], -1)
        return {"cameraViewTransform": V, "cameraProjection": P.transpose(1, 2), "cameraFocalLength": f,
                "cameraAperture": ap, "cameraApertureOffset": off,
                "cameraNearFar": torch.tensor([near, far], device=self.dev).expand(N, 2),
                "renderProductResolution": torch.tensor([self.W, self.H], device=self.dev).expand(N, 2),
                "metersPerSceneUnit": 1.0, "cameraModel": "pinhole"}


    # -- first implementation (torch ops on MPS): rejected for cost, kept behind impl="torch" ---------------
    # 1024 envs x 128^2, SO-101 scene, GPU shared with a training job (runs/replicator_bench_1024.log;
    # python -m metalsim.replicator.bench), torch vs warp ms/frame: distance_to_camera 9.1 vs 3.7, normals
    # 48.5 vs 2.7, normals_from_depth 231 vs 5.9, bounding_box_2d_tight 19.9 vs 1.4 (bincount counts:
    # 2374, a CPU fallback), bounding_box_3d 23.6 vs 1.8, pointcloud 209 vs 12.3, motion_vectors 429 vs
    # 4.8, occlusion pass 347 vs 153. Outputs identical (tests run both).

    def _torch_distance_to_camera(self):
        a, b = self._ray_scale()
        return self.distance_to_image_plane() * torch.sqrt(1 + a * a + b * b)

    def _torch_world_points(self, state=None):
        pos, R = self.camera_pose(state)
        pc = self.camera_points()
        valid = (self.r.out.depth > 0).unsqueeze(-1)
        return torch.where(valid, torch.einsum("nij,nhwj->nhwi", R, pc) + pos[:, None, None, :], torch.zeros_like(pc))

    def _torch_normals(self):
        n = self.r.out.normal
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        valid = (self.r.out.depth > 0).float().unsqueeze(-1)
        return torch.cat([n * valid, valid], -1)

    def _torch_normals_from_depth(self, frame="world"):
        P = self.camera_points()
        seg = self.r.out.seg
        valid = self.r.out.depth > 0

        def diff(axis):
            sl_p = [slice(None)] * 3; sl_m = [slice(None)] * 3
            fwd = torch.zeros_like(P); bwd = torch.zeros_like(P)
            okf = torch.zeros_like(valid); okb = torch.zeros_like(valid)
            a = axis + 1
            sl_p[a] = slice(1, None); sl_m[a] = slice(None, -1)
            dP = P[tuple(sl_p)] - P[tuple(sl_m)]
            same = (seg[tuple(sl_p)] == seg[tuple(sl_m)]) & valid[tuple(sl_p)] & valid[tuple(sl_m)]
            fwd[tuple(sl_m)] = dP; okf[tuple(sl_m)] = same
            bwd[tuple(sl_p)] = dP; okb[tuple(sl_p)] = same
            both = okf & okb
            d = torch.where(both.unsqueeze(-1), 0.5 * (fwd + bwd), torch.where(okf.unsqueeze(-1), fwd, bwd))
            return d, okf | okb
        dx, okx = diff(1)
        dy, oky = diff(0)
        n = torch.cross(dx, dy, dim=-1)
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        n = torch.where(((n * P).sum(-1, keepdim=True) > 0), -n, n)
        ok = (okx & oky & valid).unsqueeze(-1).float()
        n = n * ok
        if frame == "world":
            _, R = self.camera_pose()
            n = torch.einsum("nij,nhwj->nhwi", R, n)
        return torch.cat([n, ok], -1)

    def _torch_extents_from_ids(self, inst, counts="scatter_add"):
        """``counts="bincount"`` is the first variant (CPU fallback on MPS, 268 ms at 1024 envs)."""
        N, E = self.n, self.E
        k = (inst.long() - 2)
        valid = k >= 0
        key = (torch.arange(N, device=self.dev).view(N, 1, 1) * (E + 1) + torch.where(valid, k, torch.full_like(k, E))).reshape(-1)
        xs = torch.arange(self.W, device=self.dev).view(1, 1, -1).expand(N, self.H, self.W).reshape(-1)
        ys = torch.arange(self.H, device=self.dev).view(1, -1, 1).expand(N, self.H, self.W).reshape(-1)
        size = N * (E + 1)
        big = torch.full((size,), 1 << 30, dtype=torch.long, device=self.dev)
        neg = torch.full((size,), -1, dtype=torch.long, device=self.dev)
        x0 = big.scatter_reduce(0, key, xs, "amin"); y0 = big.clone().scatter_reduce(0, key, ys, "amin")
        x1 = neg.scatter_reduce(0, key, xs, "amax"); y1 = neg.clone().scatter_reduce(0, key, ys, "amax")
        if counts == "bincount":
            cnt = torch.bincount(key, minlength=size)
        else:
            cnt = torch.zeros(size, dtype=torch.int32, device=self.dev).scatter_add_(0, key, torch.ones_like(key, dtype=torch.int32))
        box = torch.stack([x0, y0, x1, y1], -1).view(N, E + 1, 4)[:, :E]
        return box, cnt.view(N, E + 1)[:, :E]

    def _torch_motion_vectors(self, prev, state=None):
        s = state or self.state
        geom = self.lut_geom[self.r.out.seg.long()]
        ok = (geom >= 0) & (self.r.out.depth > 0)
        g = geom.clamp(min=0)
        n = torch.arange(self.n, device=self.dev).view(-1, 1, 1)
        pos, R = self.camera_pose(s)
        pw = torch.einsum("nij,nhwj->nhwi", R, self.camera_points()) + pos[:, None, None, :]
        Rc, pc = s.geom_xmat[n, g], s.geom_xpos[n, g]
        local = torch.einsum("nhwji,nhwj->nhwi", Rc, pw - pc)
        Rp, pp = prev.geom_xmat[n, g], prev.geom_xpos[n, g]
        pw0 = torch.einsum("nhwij,nhwj->nhwi", Rp, local) + pp
        cpos, cR = self.camera_pose(prev)
        c = torch.einsum("nji,nhwj->nhwi", cR, pw0 - cpos[:, None, None])
        fx, fy, cx, cy = self.intrinsics()
        z = (-c[..., 2]).clamp(min=1e-6)
        u0 = cx[:, None, None] + fx[:, None, None] * c[..., 0] / z
        v0 = cy[:, None, None] - fy[:, None, None] * c[..., 1] / z
        mv = torch.stack([u0 - self.u, v0 - self.v, torch.zeros_like(z), ok.float()], -1)
        return mv * ok.unsqueeze(-1).float()

    # -- CPU records (Isaac's per-render-product structured outputs) ----------------------------------------

    def records(self, name: str, out: dict, env: int):
        """Isaac's structured numpy record array + info for one env from a batched box output."""
        if name in ("bounding_box_2d_tight", "bounding_box_2d_loose"):
            valid = out["valid"][env].cpu().numpy()
            box = out["box"][env].cpu().numpy(); occ = out["occlusionRatio"][env].cpu().numpy()
            ids = np.flatnonzero(valid)
            rec = np.zeros(len(ids), BBOX2D_DTYPE)
            sem = out["semanticId"].cpu().numpy()
            rec["semanticId"] = sem[ids]
            rec["x_min"], rec["y_min"], rec["x_max"], rec["y_max"] = box[ids].T
            rec["occlusionRatio"] = occ[ids]
        elif name == "bounding_box_3d":
            valid = out["valid"][env].cpu().numpy()
            ids = np.flatnonzero(valid)
            ext = out["extents"].cpu().numpy(); T = out["transform"][env].cpu().numpy()
            rec = np.zeros(len(ids), BBOX3D_DTYPE)
            rec["semanticId"] = out["semanticId"].cpu().numpy()[ids]
            for i, f in enumerate(("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")):
                rec[f] = ext[ids, i]
            rec["transform"] = T[ids]
            rec["occlusionRatio"] = out["occlusionRatio"][env].cpu().numpy()[ids]
        else:
            raise ValueError(name)
        info = {"bboxIds": (ids + 2).astype(np.uint32),
                "idToLabels": {str(int(s)): self.id_to_labels[str(int(s))] for s in np.unique(rec["semanticId"])},
                "primPaths": [self.sem.entities[i].path for i in ids]}
        return rec, info
