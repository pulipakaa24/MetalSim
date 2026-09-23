"""Tier 2: progressive path tracer on Metal hardware ray tracing (see shaders/pathtrace.metal).

Same scene tables, materials, lights and camera model as tiers 0/1; the sensor layer's
acceleration structures (refit from the physics state each frame). Intended for dataset
generation and validation, not rollouts: cost is spp x bounces rays per pixel.

Validation lives in tests/test_render_tier2.py: analytic Lambertian irradiance under a directional
light, white-furnace energy conservation, convergence with sample count, and agreement of the
direct-light term with tier 0.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp
import Metal

from orchard.interop import torch_bridge as tb
from orchard.interop import warp_metal as wm
from orchard.render.metal_context import MetalContext
from orchard.render.scene_tables import CameraIntrinsics
from orchard.render.tier0 import RenderOutputs
from orchard.sensors.raytrace import RayTracer


class Tier2Renderer:
    def __init__(self, model, n_envs: int, *, width=256, height=256, camera=None, spp=4, max_bounces=4,
                 max_group=3, include_planes=True, decimate_faces=0, exposure=1.0, seed=0,
                 ctx: MetalContext | None = None, device="metal:0"):
        import mujoco
        self.m, self.n, self.tw, self.th = model, n_envs, width, height
        self.device = device
        self.ctx = ctx or MetalContext(device)
        self.rt = RayTracer(model, n_envs, max_range=1000.0, max_group=max_group, include_planes=include_planes,
                            decimate_faces=decimate_faces, ctx=self.ctx, device=device)
        self.tables = self.rt.tables
        self.G = self.tables.G
        cam_id = 0 if camera is None else (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) if isinstance(camera, str) else int(camera))
        self.cam_id = cam_id
        self.intrinsics = CameraIntrinsics.from_model(model, cam_id, width, height)
        c = self.ctx
        t = self.tables
        # per-env camera spec (same layout as tier 0)
        self.cam_spec = np.zeros((n_envs, 28), np.float32)
        self.cam_spec[:, 0:4] = (self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.cx, self.intrinsics.cy)
        self.cam_spec[:, 8:12] = (1, 0, 0, 0)
        self.cam_spec[:, 16:20] = (1, 1, 1, 0)
        sky = t.params[4]
        self.cam_spec[:, 20:24] = (sky[0], sky[1], sky[2], 1.0) if sky[3] > 0 else (0, 0, 0, 1)
        self.cam_spec.view(np.int32)[:, 24] = cam_id
        self._cam_wp = wp.array(self.cam_spec, dtype=wp.float32, device=device)
        self.t_cam_spec = tb.mps_tensor(self._cam_wp)
        self.cam_buf = wm.buffer_of(self._cam_wp).buffer
        self.colors = np.ones((n_envs, self.G, 4), np.float32)
        self._colors_wp = wp.array(self.colors, dtype=wp.float32, device=device)
        self.t_colors = tb.mps_tensor(self._colors_wp)
        self.color_buf = wm.buffer_of(self._colors_wp).buffer
        self.light_buf = c.buffer(t.lights.nbytes, t.lights, "pt_lights")
        self.params_buf = c.buffer(t.params.nbytes, t.params, "pt_params")
        ai = t.atlas.image
        self.atlas_tex = c.texture2d(ai.shape[1], ai.shape[0], Metal.MTLPixelFormatRGBA8Unorm, Metal.MTLTextureUsageShaderRead,
                                     label="pt_atlas", storage=Metal.MTLStorageModeShared)
        c.upload_texture(self.atlas_tex, ai)
        self.samp = c.sampler(linear=True, repeat=True)
        self.consts = np.zeros(16, np.uint32)
        self.consts[:4] = (n_envs, width, height, max(model.ncam, 1))
        self.consts[4:8] = (max(model.nlight, 1) if model.nlight else 0, self.G, spp, max_bounces)
        self.consts[8:12] = (0, self.rt.tpr, 1, seed)
        cf = self.consts.view(np.float32); cf[12] = self.rt.stride; cf[13] = exposure
        self.consts_buf = c.buffer(self.consts.nbytes, self.consts, "pt_consts")
        self.spp, self.max_bounces = spp, max_bounces
        self.frame = 0
        n, h, w = n_envs, height, width
        self._accum = wp.zeros((n, h, w, 4), dtype=wp.float32, device=device)
        self._arrays = {"rgb": wp.zeros((n, h, w, 3), dtype=wp.uint8, device=device), "depth": wp.zeros((n, h, w), dtype=wp.float32, device=device),
                        "seg": wp.zeros((n, h, w), dtype=wp.int32, device=device), "normal": wp.zeros((n, h, w, 3), dtype=wp.float32, device=device)}
        wp.synchronize_device(device)
        self._views = {k: wm.buffer_of(a) for k, a in self._arrays.items()}
        self._accum_view = wm.buffer_of(self._accum)
        self.out = RenderOutputs(**{k: tb.mps_tensor(a) for k, a in self._arrays.items()})
        self.t_accum = tb.mps_tensor(self._accum)
        lib = c.library("pathtrace")
        self.pipe = c.compute_pipeline(lib, "path_trace")
        nl = max(model.nlight, 1)
        self._host_light_xpos = c.buffer(n * nl * 12); self._host_light_xdir = c.buffer(n * nl * 12)
        self._host_cam_xpos = c.buffer(n * max(model.ncam, 1) * 12); self._host_cam_xmat = c.buffer(n * max(model.ncam, 1) * 36)
        lp = np.zeros((n, nl, 3), np.float32); ld = np.zeros((n, nl, 3), np.float32); ld[:, :, 2] = -1
        for i in range(model.nlight):
            lp[:, i] = model.light_pos[i]; ld[:, i] = model.light_dir[i]
        c.write_buffer(self._host_light_xpos, lp); c.write_buffer(self._host_light_xdir, ld)
        self.event = self.ctx.event

    def _encode(self, cb, reset: bool, cam_xpos, cxo, cam_xmat, cmo, light_xpos, lxo, light_xdir, ldo):
        self.consts[8] = self.frame
        self.consts[10] = 1 if reset else 0
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.pipe)
        rt = self.rt
        rt._use_resources(ce)
        ce.setAccelerationStructure_atBufferIndex_(rt.inst_as, 0)
        ce.setBuffer_offset_atIndex_(rt.inst_tab, 0, 1); ce.setBuffer_offset_atIndex_(rt.slot_mesh, 0, 2)
        ce.setBuffer_offset_atIndex_(rt.mesh_info, 0, 3); ce.setBuffer_offset_atIndex_(rt.vbuf, 0, 4)
        ce.setBuffer_offset_atIndex_(rt.ibuf, 0, 5); ce.setBuffer_offset_atIndex_(rt.inst_desc, 0, 6)
        ce.setBuffer_offset_atIndex_(rt.mat_buf, 0, 7); ce.setBuffer_offset_atIndex_(self.color_buf, 0, 8)
        ce.setBuffer_offset_atIndex_(self.light_buf, 0, 9)
        ce.setBuffer_offset_atIndex_(light_xpos, lxo, 10); ce.setBuffer_offset_atIndex_(light_xdir, ldo, 11)
        ce.setBuffer_offset_atIndex_(self.params_buf, 0, 12); ce.setBuffer_offset_atIndex_(self.cam_buf, 0, 13)
        ce.setBuffer_offset_atIndex_(cam_xpos, cxo, 14); ce.setBuffer_offset_atIndex_(cam_xmat, cmo, 15)
        ce.setBytes_length_atIndex_(self.consts.tobytes(), self.consts.nbytes, 16)   # copied at encode time (per pass)
        ce.setBuffer_offset_atIndex_(self._accum_view.buffer, self._accum_view.offset, 17)
        for i, k in enumerate(("rgb", "depth", "seg", "normal")):
            v = self._views[k]; ce.setBuffer_offset_atIndex_(v.buffer, v.offset, 18 + i)
        ce.setTexture_atIndex_(self.atlas_tex, 0); ce.setSamplerState_atIndex_(self.samp, 0)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.tw, self.th, self.n), Metal.MTLSize(8, 8, 1))
        ce.endEncoding()
        self.frame += 1

    def render(self, sim, after_value=None, reset=True, passes=1) -> int:
        """GPU path from a BatchSim: refit + rebuild the acceleration structure, then ``passes`` path
        tracing passes of ``spp`` samples each (progressive; ``reset`` restarts accumulation)."""
        if after_value is None:
            after_value = sim.event.value
        names = ["geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat"] + (["light_xpos", "light_xdir"] if self.m.nlight else [])
        v = {k: wm.buffer_of(getattr(sim.d, k)) for k in names}
        cb = self.ctx.command_buffer()
        self.ctx.wait_for(cb, sim.event, after_value)
        self.rt._encode_refit_and_build(cb, v["geom_xpos"].buffer, v["geom_xpos"].offset, v["geom_xmat"].buffer, v["geom_xmat"].offset)
        lx = v.get("light_xpos"); ld = v.get("light_xdir")
        lxb, lxo, ldb, ldo = (lx.buffer, lx.offset, ld.buffer, ld.offset) if lx else (self._host_light_xpos, 0, self._host_light_xdir, 0)
        for p in range(passes):
            self._encode(cb, reset and p == 0, v["cam_xpos"].buffer, v["cam_xpos"].offset, v["cam_xmat"].buffer, v["cam_xmat"].offset, lxb, lxo, ldb, ldo)
        val = self.ctx.signal(cb)
        self.ctx.commit(cb)
        return val

    def after(self, value):
        """Order torch's subsequent kernels after the render that signalled ``value``."""
        tb.wait_event(self.ctx.event, value)

    def wait_sim_after_render(self, sim, value):
        """Order the sim's next step after the render (it reads the sim's pose buffers)."""
        sim.wait(self.ctx.event, value)

    def set_colors(self, colors):
        self.colors[:] = colors
        self.t_colors.copy_(torch.as_tensor(np.asarray(colors, np.float32)))
        torch.mps.synchronize()

    def render_host(self, datas, passes=1, reset=True, cam_pos=None, cam_quat=None):
        """Host path from MjData lists (tests/tools); synchronizes; returns numpy outputs."""
        from orchard.render.scene_tables import quats_to_mats
        n, ng, nc, nl = self.n, self.m.ngeom, max(self.m.ncam, 1), max(self.m.nlight, 1)
        c = self.ctx; rt = self.rt
        c.write_buffer(rt._host_geom_xpos, np.stack([d.geom_xpos for d in datas]).astype(np.float32).reshape(n, ng, 3))
        c.write_buffer(rt._host_geom_xmat, np.stack([d.geom_xmat for d in datas]).astype(np.float32).reshape(n, ng, 9))
        if cam_pos is not None:
            cx = np.zeros((n, nc, 3), np.float32); cm = np.zeros((n, nc, 9), np.float32)
            cx[:, self.cam_id] = cam_pos; cm[:, self.cam_id] = quats_to_mats(np.asarray(cam_quat, np.float64)).reshape(n, 9)
        else:
            cx = np.stack([d.cam_xpos for d in datas]).astype(np.float32).reshape(n, nc, 3)
            cm = np.stack([d.cam_xmat for d in datas]).astype(np.float32).reshape(n, nc, 9)
        c.write_buffer(self._host_cam_xpos, cx); c.write_buffer(self._host_cam_xmat, cm)
        if self.m.nlight:
            c.write_buffer(self._host_light_xpos, np.stack([d.light_xpos for d in datas]).astype(np.float32).reshape(n, nl, 3))
            c.write_buffer(self._host_light_xdir, np.stack([d.light_xdir for d in datas]).astype(np.float32).reshape(n, nl, 3))
        cb = c.command_buffer()
        rt._encode_refit_and_build(cb, rt._host_geom_xpos, 0, rt._host_geom_xmat, 0)
        for p in range(passes):
            self._encode(cb, reset and p == 0, self._host_cam_xpos, 0, self._host_cam_xmat, 0, self._host_light_xpos, 0, self._host_light_xdir, 0)
        c.commit(cb); c.synchronize()
        out = {k: a.numpy().copy() for k, a in self._arrays.items()}
        acc = self._accum.numpy()
        out["hdr"] = acc[..., :3] / np.maximum(acc[..., 3:4], 1.0)
        out["samples"] = acc[..., 3].copy()
        return out
