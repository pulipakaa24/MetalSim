"""Fisheye cameras on the tier-0 raster: cube faces resampled through a lens model (``metalsim.render.lens``).

A rasterizer cannot draw a non-linear projection directly, so ``FisheyeTier0`` renders the faces of a cube around the
camera that the lens's field of view touches (up to six 90-degree pinhole views, each a ``Tier0Renderer`` sharing one
Metal context, camera pose and the same per-env tables) and resamples them per output pixel with a host-built lookup
table (face, face pixel, z-depth factor); shaders/fisheye.metal. The face resolution matches the lens's finest angular
sampling, so the resample neither aliases nor blurs at the image centre. Cost: one tier-0 render per used face
(5 for Isaac Lab's default 200-degree f-theta lens).

Outputs follow ``Tier0Renderer``: rgb (bilinear within a face), depth = z-depth along the optical axis (negative behind
the image plane for fields of view over 180 degrees; 0 outside the lens's field of view or on background),
segmentation and world normals (nearest). Tier 2 and the ray-traced depth camera trace the lens exactly instead
(``Tier2Renderer(lens=...)``, ``RayTracer.make_depth_camera(..., lens=...)``).
Limits: a per-env camera pose / intrinsics delta (``t_cam_spec``) is not applied (the faces own the rotation delta),
and per-env backgrounds are per face.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch
import warp as wp
import Metal

from metalsim.interop import torch_bridge as tb
from metalsim.interop import warp_metal as wm
from metalsim.render.lens import Lens
from metalsim.render.metal_context import MetalContext
from metalsim.render.tier0 import RenderOutputs, Tier0Renderer

# cube faces in the camera frame (x right, y up, looking along -z): forward, up
FACES = {"front": ((0, 0, -1), (0, 1, 0)), "right": ((1, 0, 0), (0, 1, 0)), "left": ((-1, 0, 0), (0, 1, 0)),
         "up": ((0, 1, 0), (0, 0, 1)), "down": ((0, -1, 0), (0, 0, -1)), "back": ((0, 0, 1), (0, 1, 0))}


def face_rotation(name):
    """Columns: the face camera's x, y, z axes in the main camera frame."""
    f, u = (np.asarray(v, np.float64) for v in FACES[name])
    z = -f; y = u; x = np.cross(y, z)
    return np.stack([x, y, z], 1)


def build_lut(lens: Lens, width, height, face_res=None, intrinsics=None):
    """(H, W) lookup table rows (face index or -1, u, v, kz), the used face names and the face resolution."""
    d, valid = lens.ray_dirs(width, height, intrinsics)
    if face_res is None:          # finest angular step between neighbouring pixels -> 90 deg / step pixels per face
        ang = np.arccos(np.clip((d[:, 1:] * d[:, :-1]).sum(-1), -1, 1))[valid[:, 1:] & valid[:, :-1]]
        step = np.percentile(ang, 5) if ang.size else np.pi / 2 / 64
        face_res = int(np.clip(np.ceil((np.pi / 2) / max(step, 1e-6) * 1.0), 16, 2048))
    names = list(FACES)
    Rs = np.stack([face_rotation(n) for n in names])                    # (6, 3, 3)
    local = np.einsum("fji,hwj->hwfi", Rs, d)                          # direction in each face frame
    depth_f = -local[..., 2]                                           # along each face's forward
    face = np.argmax(depth_f, -1)
    lf = np.take_along_axis(local, face[..., None, None], 2)[..., 0, :]
    dz = np.maximum(-lf[..., 2], 1e-9)
    half = face_res / 2.0
    u = half + half * lf[..., 0] / dz
    v = half - half * lf[..., 1] / dz
    kz = (-d[..., 2]) / dz
    used = sorted(set(face[valid].tolist()))
    remap = {f: i for i, f in enumerate(used)}
    lut = np.zeros((height, width, 4), np.float32)
    lut.view(np.int32)[..., 0] = np.where(valid, np.vectorize(lambda f: remap.get(f, -1))(face), -1)
    lut[..., 1] = u; lut[..., 2] = v; lut[..., 3] = kz
    return lut, [names[f] for f in used], face_res


class FisheyeTier0:
    def __init__(self, model: mujoco.MjModel, n_envs: int, lens: Lens, *, width=128, height=128, camera=None,
                 outputs=("rgb",), face_res: int | None = None, ctx: MetalContext | None = None, device="metal:0", **tier0_kw):
        """``lens``: f-theta / Kannala-Brandt (``Lens.ftheta()`` = Isaac Lab's ``FisheyeCameraCfg`` default).
        ``face_res``: cube-face resolution (default from the lens's finest angular sampling). Other keywords go to each
        face's ``Tier0Renderer`` (seg_mode, far, tier, ...)."""
        self.m, self.n, self.tw, self.th, self.lens = model, n_envs, width, height, lens
        self.ctx = ctx or MetalContext(device)
        self.device = device
        intr = None
        if lens.model == 0:
            from metalsim.render.scene_tables import CameraIntrinsics
            cam_id = 0 if camera is None else (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) if isinstance(camera, str) else int(camera))
            ci = CameraIntrinsics.from_model(model, cam_id, width, height); intr = (ci.fx, ci.fy, ci.cx, ci.cy)
        lut, self.face_names, self.face_res = build_lut(lens, width, height, face_res, intr)
        fr = self.face_res
        self.faces = []
        for name in self.face_names:
            r = Tier0Renderer(model, n_envs, width=fr, height=fr, camera=camera, outputs=outputs, ctx=self.ctx, device=device, **tier0_kw)
            r.cam_spec[:, 0:4] = (fr / 2.0, fr / 2.0, fr / 2.0, fr / 2.0)        # 90-degree square pinhole
            q = np.zeros(4); mujoco.mju_mat2Quat(q, face_rotation(name).reshape(-1))
            r.cam_spec[:, 8:12] = q.astype(np.float32)
            r._sync_cam_spec()
            self.faces.append(r)
        c = self.ctx
        self.lut_buf = c.buffer(lut.nbytes, lut, "fisheye_lut")
        self.p_res = c.compute_pipeline(c.library("fisheye"), "fisheye_resample")
        self.out = RenderOutputs()
        self._out_arrays = {}
        specs = {"rgb": ((n_envs, height, width, 3), wp.uint8), "depth": ((n_envs, height, width), wp.float32),
                 "seg": ((n_envs, height, width), wp.int32), "normal": ((n_envs, height, width, 3), wp.float32)}
        for name in outputs:
            a = wp.zeros(specs[name][0], dtype=specs[name][1], device=device)
            self._out_arrays[name] = a
            setattr(self.out, name, tb.mps_tensor(a))
        wp.synchronize_device(device)
        self._mask = sum(1 << i for i, k in enumerate(("rgb", "depth", "seg", "normal")) if k in outputs)
        self.G = self.faces[0].G
        self.tables = self.faces[0].tables

    def _encode_resample(self, cb):
        dummy = self.lut_buf
        for i, r in enumerate(self.faces):
            ce = cb.computeCommandEncoder()
            ce.setComputePipelineState_(self.p_res)
            ce.setBuffer_offset_atIndex_(self.lut_buf, 0, 0)
            for k, name in enumerate(("rgb", "depth", "seg", "normal")):
                if name in r._out_bufs:
                    b, o = r._out_bufs[name]; ce.setBuffer_offset_atIndex_(b, o, 1 + k)
                else:
                    ce.setBuffer_offset_atIndex_(dummy, 0, 1 + k)
                if name in self._out_arrays:
                    v = wm.buffer_of(self._out_arrays[name]); ce.setBuffer_offset_atIndex_(v.buffer, v.offset, 5 + k)
                else:
                    ce.setBuffer_offset_atIndex_(dummy, 0, 5 + k)
            consts = np.array([self.n, self.tw, self.th, self.face_res, i, self._mask, 1 if i == 0 else 0, 0], np.uint32)
            ce.setBytes_length_atIndex_(consts.tobytes(), consts.nbytes, 9)
            ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.tw, self.th, self.n), Metal.MTLSize(8, 8, 1))
            ce.endEncoding()

    def render(self, sim, after_value: int | None = None) -> int:
        """GPU path: every face from ``sim``'s state, then the resample; returns the context event value."""
        if after_value is None:
            after_value = sim.event.value
        for r in self.faces:
            r.render(sim, after_value)
        cb = self.ctx.command_buffer()
        self._encode_resample(cb)
        v = self.ctx.signal(cb)
        self.ctx.commit(cb)
        return v

    def after(self, value: int) -> None:
        tb.wait_event(self.ctx.event, value)

    def render_host(self, datas, **kw):
        for r in self.faces:
            r.render_host(datas, **kw)
        cb = self.ctx.command_buffer()
        self._encode_resample(cb)
        self.ctx.commit(cb); self.ctx.synchronize()
        return {k: a.numpy().copy() for k, a in self._out_arrays.items()}
