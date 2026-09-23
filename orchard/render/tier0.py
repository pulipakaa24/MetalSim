"""Tier-0 batched renderer on native Metal.

Renders N environments as one tile atlas in one command buffer: per-env background quad, then one
instanced indexed draw per unique mesh, then an untile kernel that writes RGB / depth / segmentation
/ normals straight into learner-owned buffers (Warp arrays on ``metal:0``, aliased as MPS tensors).

Instance transforms and cameras are read from the physics buffers of a ``BatchSim`` (MuJoCo Warp
``geom_xpos``/``geom_xmat``/``cam_xpos``/``cam_xmat``) with no host work per frame. Ordering: the
render command buffer waits for the sim event value passed to ``render`` and signals
``renderer.ctx.event``; call ``renderer.after(value)`` before torch reads the outputs.

A host-side path (``render_host``) renders from ``mujoco.MjData`` lists for tests and tools.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
import warp as wp
import Metal

from orchard.interop import torch_bridge as tb
from orchard.interop import warp_metal as wm
from orchard.render.metal_context import MetalContext
from orchard.render.scene_tables import CameraIntrinsics, SceneTables, build_scene_tables, quats_to_mats

SEG_SLOT, SEG_GEOM, SEG_BODY = 0, 1, 2


@dataclass
class RenderOutputs:
    rgb: torch.Tensor | None = None       # (N, H, W, 3) uint8
    depth: torch.Tensor | None = None     # (N, H, W) float32 metric
    seg: torch.Tensor | None = None       # (N, H, W) int32
    normal: torch.Tensor | None = None    # (N, H, W, 3) float32


class Tier0Renderer:
    def __init__(self, model: mujoco.MjModel, n_envs: int, *, width=128, height=128, camera: str | int | None = None,
                 max_group=3, include_planes=True, backgrounds=False, outputs=("rgb",), seg_mode=SEG_SLOT,
                 decimate_faces: int = 0, shadows: bool = True, tier: int = 0, rt_samples: int = 4,
                 ctx: MetalContext | None = None, device="metal:0"):
        """tier 0: raster + shadow maps. tier 1: raster G-buffer with ray-traced shadows, ambient
        occlusion and mirror reflections from the fragment stage (Metal hardware ray tracing)."""
        self.m = model
        self.tier = tier
        self.rt_samples = rt_samples
        self.shadows = shadows and tier == 0
        self.rt = None
        self.n = n_envs
        self.tw, self.th = width, height
        self.tpr = int(np.ceil(np.sqrt(n_envs)))
        self.aw = self.tpr * width
        self.ah = int(np.ceil(n_envs / self.tpr)) * height
        self.ctx = ctx or MetalContext(device)
        self.device = device
        self.tables: SceneTables = build_scene_tables(model, n_envs, max_group=max_group, include_planes=include_planes,
                                                      decimate_faces=decimate_faces)
        self.G = self.tables.G
        self.seg_mode = seg_mode
        self.use_bg = backgrounds
        self.bg_layers = min(n_envs, 2048)
        if camera is None:
            cam_id = 0
        elif isinstance(camera, str):
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cam_id < 0:
                raise ValueError(f"camera {camera!r} not found")
        else:
            cam_id = int(camera)
        if model.ncam == 0:
            raise ValueError("model has no cameras")
        self.cam_id = cam_id
        self.intrinsics = CameraIntrinsics.from_model(model, cam_id, width, height)
        self._build_resources()
        self._build_pipelines()
        self._alloc_outputs(outputs)
        self._host_sim = None
        if tier >= 1:
            from orchard.sensors.raytrace import RayTracer
            self.rt = RayTracer(model, n_envs, max_range=50.0, max_group=max_group, include_planes=include_planes,
                                decimate_faces=decimate_faces, ctx=self.ctx, device=device)
            ro = np.zeros(4, np.uint32); ro[0] = self.rt.tpr; ro.view(np.float32)[1] = self.rt.stride; ro[2] = rt_samples
            self.rt_offset_buf = self.ctx.buffer(16, ro, "rt_offset")

    # -- resources --------------------------------------------------------------------------------

    def _build_resources(self):
        c, t = self.ctx, self.tables
        self.vbuf = c.buffer(t.vertices.nbytes, t.vertices, "vertices")
        self.ibuf = c.buffer(t.indices.nbytes, t.indices, "indices")
        self.inst_buf = c.buffer(t.inst_table.nbytes, t.inst_table, "instances")
        self.slot_geom = c.buffer(4 * self.G, np.asarray(t.geoms, np.int32), "slot_geom")
        self.mat_buf = c.buffer(t.materials.nbytes, t.materials, "materials")
        self.sem_buf = c.buffer(t.semantic.nbytes, t.semantic, "semantic")
        self.env_buf = c.buffer(self.n * (2 * 64 + 8 * 16), label="env_params")
        # per-env camera spec (host-written); defaults: model camera, no delta, white light from above
        # CameraSpec layout (28 floats = 112 bytes): intrinsics 0:4, pos_delta 4:8, rot_delta 8:12,
        # light 12:16 (w intensity), ambient 16:20 (w background mode), clear_color 20:24, cam_id 24 (int)
        self.cam_spec = np.zeros((self.n, 28), np.float32)
        self.cam_spec_i = self.cam_spec.view(np.int32)
        self.cam_spec[:, 0:4] = (self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.cx, self.intrinsics.cy)
        self.cam_spec[:, 8:12] = (1, 0, 0, 0)
        self.cam_spec[:, 12:16] = (0.0, 0.0, 0.0, 0.0)   # DR light off: light the scene with the model's lights
        self.cam_spec[:, 16:20] = (1.0, 1.0, 1.0, 1.0 if self.use_bg else 0.0)   # ambient scale
        sky = t.params[4]
        self.cam_spec[:, 20:24] = (sky[0], sky[1], sky[2], 1.0) if sky[3] > 0 else (0.0, 0.0, 0.0, 1.0)   # default background: sky
        self.cam_spec_i[:, 24] = self.cam_id
        self._cam_spec_wp = wp.array(self.cam_spec, dtype=wp.float32, device=self.device)
        self.t_cam_spec = tb.mps_tensor(self._cam_spec_wp)      # (N, 28) float32, GPU-writable
        self.cam_buf = wm.buffer_of(self._cam_spec_wp).buffer
        # per-instance color modulation (domain randomization), default 1
        self.colors = np.ones((self.n, self.G, 4), np.float32)
        self._colors_wp = wp.array(self.colors, dtype=wp.float32, device=self.device)
        self.t_colors = tb.mps_tensor(self._colors_wp)          # (N, G, 4) float32, GPU-writable
        self.color_buf = wm.buffer_of(self._colors_wp).buffer
        wp.synchronize_device(self.device)
        # lights and scene params (headlight, bounds)
        self.n_light = self.m.nlight
        self.light_buf = c.buffer(t.lights.nbytes, t.lights, "lights")
        self.params_buf = c.buffer(t.params.nbytes, t.params, "scene_params")
        # shadow atlas: one tile per env, twice the render tile, atlas capped at 4096
        st = min(2 * max(self.tw, self.th), max(16, 4096 // self.tpr))
        self.sw = self.sh = int(st)
        self.saw = self.tpr * self.sw
        self.sah = int(np.ceil(self.n / self.tpr)) * self.sh
        # consts
        flags = (1 if self.use_bg else 0) | (2 if self.shadows else 0)
        self.consts = np.array([self.n, self.m.ngeom, self.G, self.tpr, self.tw, self.th, self.aw, self.ah,
                                max(self.m.ncam, 1), self.bg_layers, flags, max(self.n_light, 1),
                                self.sw, self.sh, self.saw, self.sah], np.uint32)
        self.consts_buf = c.buffer(self.consts.nbytes, self.consts, "consts")
        # textures
        ai = t.atlas.image
        self.atlas_tex = c.texture2d(ai.shape[1], ai.shape[0], Metal.MTLPixelFormatRGBA8Unorm,
                                     Metal.MTLTextureUsageShaderRead, label="atlas", storage=Metal.MTLStorageModeShared)
        c.upload_texture(self.atlas_tex, ai)
        self.bg_tex = c.texture2d(self.tw, self.th, Metal.MTLPixelFormatRGBA8Unorm, Metal.MTLTextureUsageShaderRead,
                                  array_length=self.bg_layers, label="backgrounds", storage=Metal.MTLStorageModeShared)
        rt = Metal.MTLTextureUsageRenderTarget | Metal.MTLTextureUsageShaderRead
        self.color_tex = c.texture2d(self.aw, self.ah, Metal.MTLPixelFormatRGBA8Unorm, rt, label="color")
        self.seg_tex = c.texture2d(self.aw, self.ah, Metal.MTLPixelFormatR32Uint, rt, label="seg")
        self.normal_tex = c.texture2d(self.aw, self.ah, Metal.MTLPixelFormatRGBA16Float, rt, label="normal")
        self.depth_tex = c.texture2d(self.aw, self.ah, Metal.MTLPixelFormatDepth32Float, rt, label="depth")
        self.shadow_tex = c.texture2d(self.saw, self.sah, Metal.MTLPixelFormatDepth32Float, rt, label="shadow")
        self.samp = c.sampler(linear=True, repeat=True)
        self.samp_bg = c.sampler(linear=True, repeat=False)
        sd = Metal.MTLSamplerDescriptor.new()
        sd.setMinFilter_(Metal.MTLSamplerMinMagFilterLinear); sd.setMagFilter_(Metal.MTLSamplerMinMagFilterLinear)
        sd.setCompareFunction_(Metal.MTLCompareFunctionLessEqual)
        sd.setSAddressMode_(Metal.MTLSamplerAddressModeClampToEdge); sd.setTAddressMode_(Metal.MTLSamplerAddressModeClampToEdge)
        self.samp_shadow = c.device.newSamplerStateWithDescriptor_(sd)
        # host-path camera buffers (cam_xpos/cam_xmat) and geom buffers are bound from the sim in the GPU path
        self._host_cam_xpos = c.buffer(self.n * max(self.m.ncam, 1) * 12, label="host_cam_xpos")
        self._host_cam_xmat = c.buffer(self.n * max(self.m.ncam, 1) * 36, label="host_cam_xmat")
        self._host_geom_xpos = c.buffer(self.n * self.m.ngeom * 12, label="host_geom_xpos")
        self._host_geom_xmat = c.buffer(self.n * self.m.ngeom * 36, label="host_geom_xmat")
        nl = max(self.n_light, 1)
        self._host_light_xpos = c.buffer(self.n * nl * 12, label="host_light_xpos")
        self._host_light_xdir = c.buffer(self.n * nl * 12, label="host_light_xdir")
        # default light state for the host path: model light positions/directions
        lp = np.zeros((self.n, nl, 3), np.float32); ld = np.zeros((self.n, nl, 3), np.float32); ld[:, :, 2] = -1
        for i in range(self.m.nlight):
            lp[:, i] = self.m.light_pos[i]; ld[:, i] = self.m.light_dir[i]
        c.write_buffer(self._host_light_xpos, lp); c.write_buffer(self._host_light_xdir, ld)

    def _build_pipelines(self):
        c = self.ctx
        lib = c.library("tier0")
        self.p_env = c.compute_pipeline(lib, "build_env_params")
        self.p_untile = c.compute_pipeline(lib, "untile")
        vd = Metal.MTLVertexDescriptor.new()
        for i, (fmt, off) in enumerate(((Metal.MTLVertexFormatFloat3, 0), (Metal.MTLVertexFormatFloat3, 12),
                                        (Metal.MTLVertexFormatFloat2, 24))):
            a = vd.attributes().objectAtIndexedSubscript_(i)
            a.setFormat_(fmt); a.setOffset_(off); a.setBufferIndex_(0)
        lay = vd.layouts().objectAtIndexedSubscript_(0)
        lay.setStride_(32); lay.setStepFunction_(Metal.MTLVertexStepFunctionPerVertex)

        def make(vs, fs, depth_write, depth_cmp, with_vd):
            d = Metal.MTLRenderPipelineDescriptor.new()
            d.setVertexFunction_(lib.newFunctionWithName_(vs))
            d.setFragmentFunction_(lib.newFunctionWithName_(fs))
            if with_vd:
                d.setVertexDescriptor_(vd)
            ca = d.colorAttachments()
            ca.objectAtIndexedSubscript_(0).setPixelFormat_(Metal.MTLPixelFormatRGBA8Unorm)
            ca.objectAtIndexedSubscript_(1).setPixelFormat_(Metal.MTLPixelFormatR32Uint)
            ca.objectAtIndexedSubscript_(2).setPixelFormat_(Metal.MTLPixelFormatRGBA16Float)
            d.setDepthAttachmentPixelFormat_(Metal.MTLPixelFormatDepth32Float)
            p = c.render_pipeline(d)
            ds = Metal.MTLDepthStencilDescriptor.new()
            ds.setDepthWriteEnabled_(depth_write)
            ds.setDepthCompareFunction_(depth_cmp)
            return p, c.device.newDepthStencilStateWithDescriptor_(ds)

        self.geo_pipe, self.geo_ds = make("geom_vs", "geom_fs_rt" if self.tier >= 1 else "geom_fs", True, Metal.MTLCompareFunctionLess, True)
        self.bg_pipe, self.bg_ds = make("bg_vs", "bg_fs", False, Metal.MTLCompareFunctionAlways, False)
        d = Metal.MTLRenderPipelineDescriptor.new()
        d.setVertexFunction_(lib.newFunctionWithName_("shadow_vs"))
        d.setVertexDescriptor_(vd)
        d.setDepthAttachmentPixelFormat_(Metal.MTLPixelFormatDepth32Float)
        self.shadow_pipe = c.render_pipeline(d)
        self.shadow_ds = self.geo_ds

    def _alloc_outputs(self, outputs):
        """Output tensors live in Warp arrays on metal:0 so that they are simultaneously Warp arrays,
        Metal buffers for the untile kernel, and zero-copy MPS tensors."""
        self.out = RenderOutputs()
        self._out_arrays = {}
        self._out_bufs = {}
        n, h, w = self.n, self.th, self.tw
        specs = {"rgb": ((n, h, w, 3), wp.uint8), "depth": ((n, h, w), wp.float32),
                 "seg": ((n, h, w), wp.int32), "normal": ((n, h, w, 3), wp.float32)}
        for name in outputs:
            shape, dtype = specs[name]
            a = wp.zeros(shape, dtype=dtype, device=self.device)
            self._out_arrays[name] = a
            view = wm.buffer_of(a)
            self._out_bufs[name] = (view.buffer, view.offset)
            setattr(self.out, name, tb.mps_tensor(a))
        wp.synchronize_device(self.device)
        self._outputs_mask = np.array([1 if "rgb" in outputs else 0, 1 if "depth" in outputs else 0,
                                       (self.seg_mode + 1) if "seg" in outputs else 0,
                                       1 if "normal" in outputs else 0], np.uint32)
        self.outputs_buf = self.ctx.buffer(16, self._outputs_mask, "outputs")

    # -- per-env parameters (host-written; cheap, done at reset time) ---------------------------------

    def _sync_cam_spec(self):
        """Host write of the whole camera spec (synchronizes the GPU first; use ``t_cam_spec`` in loops)."""
        wp.synchronize_device(self.device)
        torch.mps.synchronize()
        self._cam_spec_wp.assign(self.cam_spec)
        wp.synchronize_device(self.device)

    def set_lights(self, dirs: np.ndarray, intensity=1.0):
        """Per-env domain-randomization directional light (adds to the model lights and becomes the
        shadow caster). intensity 0 disables it."""
        self.cam_spec[:, 12:15] = np.asarray(dirs, np.float32)
        self.cam_spec[:, 15] = intensity
        self._sync_cam_spec()

    def set_ambient(self, rgb: np.ndarray):
        """Per-env scale on the model's ambient light (1 = as authored)."""
        self.cam_spec[:, 16:19] = np.asarray(rgb, np.float32)
        self._sync_cam_spec()

    def set_clear_color(self, rgb: np.ndarray):
        self.cam_spec[:, 20:23] = np.asarray(rgb, np.float32)
        self._sync_cam_spec()

    def set_camera_delta(self, pos_delta: np.ndarray | None = None, quat_delta: np.ndarray | None = None):
        if pos_delta is not None:
            self.cam_spec[:, 4:7] = np.asarray(pos_delta, np.float32)
        if quat_delta is not None:
            q = np.asarray(quat_delta, np.float64)
            self.cam_spec[:, 8:12] = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
        self._sync_cam_spec()

    def set_colors(self, colors: np.ndarray):
        """(N, G, 4) per-env per-slot color modulation (host write; use ``t_colors`` in loops)."""
        self.colors[:] = colors
        wp.synchronize_device(self.device)
        torch.mps.synchronize()
        self._colors_wp.assign(self.colors)
        wp.synchronize_device(self.device)

    def set_backgrounds(self, env_ids, images):
        self.use_bg = True
        self.consts[10] = 1
        self.ctx.write_buffer(self.consts_buf, self.consts)
        self.cam_spec[:, 19] = 1.0
        self._sync_cam_spec()
        for eid, im in zip(env_ids, images):
            rgba = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
            self.ctx.upload_texture(self.bg_tex, rgba, int(eid) % self.bg_layers)

    # -- rendering ----------------------------------------------------------------------------------

    def _encode(self, cb, geom_xpos, geom_xpos_off, geom_xmat, geom_xmat_off, cam_xpos, cam_xpos_off, cam_xmat, cam_xmat_off,
                light_xpos=None, light_xpos_off=0, light_xdir=None, light_xdir_off=0):
        c = self.ctx
        if light_xpos is None:
            light_xpos, light_xpos_off, light_xdir, light_xdir_off = self._host_light_xpos, 0, self._host_light_xdir, 0
        # 1. per-env params from the camera and light state
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.p_env)
        ce.setBuffer_offset_atIndex_(self.env_buf, 0, 0)
        ce.setBuffer_offset_atIndex_(self.cam_buf, 0, 1)
        ce.setBuffer_offset_atIndex_(cam_xpos, cam_xpos_off, 2)
        ce.setBuffer_offset_atIndex_(cam_xmat, cam_xmat_off, 3)
        ce.setBuffer_offset_atIndex_(self.consts_buf, 0, 4)
        ce.setBuffer_offset_atIndex_(self.light_buf, 0, 5)
        ce.setBuffer_offset_atIndex_(light_xdir, light_xdir_off, 6)
        ce.setBuffer_offset_atIndex_(self.params_buf, 0, 7)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n, 1, 1), Metal.MTLSize(min(self.n, 64), 1, 1))
        ce.endEncoding()
        # 1b. shadow pass: depth from the caster light into the shadow atlas
        if self.shadows:
            sp = Metal.MTLRenderPassDescriptor.new()
            da = sp.depthAttachment()
            da.setTexture_(self.shadow_tex); da.setLoadAction_(Metal.MTLLoadActionClear)
            da.setStoreAction_(Metal.MTLStoreActionStore); da.setClearDepth_(1.0)
            se = cb.renderCommandEncoderWithDescriptor_(sp)
            se.setRenderPipelineState_(self.shadow_pipe)
            se.setDepthStencilState_(self.shadow_ds)
            se.setCullMode_(Metal.MTLCullModeNone)
            se.setDepthBias_slopeScale_clamp_(2.0, 2.0, 0.02)
            se.setVertexBuffer_offset_atIndex_(self.vbuf, 0, 0)
            se.setVertexBuffer_offset_atIndex_(self.env_buf, 0, 1)
            se.setVertexBuffer_offset_atIndex_(self.inst_buf, 0, 2)
            se.setVertexBuffer_offset_atIndex_(self.slot_geom, 0, 3)
            se.setVertexBuffer_offset_atIndex_(geom_xpos, geom_xpos_off, 4)
            se.setVertexBuffer_offset_atIndex_(geom_xmat, geom_xmat_off, 5)
            se.setVertexBuffer_offset_atIndex_(self.color_buf, 0, 6)
            se.setVertexBuffer_offset_atIndex_(self.consts_buf, 0, 7)
            for i_off, i_count, first, count in self.tables.draws:
                se.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance_(
                    Metal.MTLPrimitiveTypeTriangle, i_count, Metal.MTLIndexTypeUInt32, self.ibuf, i_off * 4, count, 0, first)
            se.endEncoding()
        # 1c. tier 1: refit instance descriptors from the same physics buffers and rebuild the instance AS
        if self.rt is not None:
            self.rt._encode_refit_and_build(cb, geom_xpos, geom_xpos_off, geom_xmat, geom_xmat_off)
        # 2. render pass
        rp = Metal.MTLRenderPassDescriptor.new()
        for i, (tex, clear) in enumerate(((self.color_tex, (0, 0, 0, 1)), (self.seg_tex, (0, 0, 0, 0)),
                                          (self.normal_tex, (0, 0, 0, 0)))):
            a = rp.colorAttachments().objectAtIndexedSubscript_(i)
            a.setTexture_(tex)
            a.setLoadAction_(Metal.MTLLoadActionClear)
            a.setStoreAction_(Metal.MTLStoreActionStore)
            a.setClearColor_(Metal.MTLClearColor(*clear))
        da = rp.depthAttachment()
        da.setTexture_(self.depth_tex)
        da.setLoadAction_(Metal.MTLLoadActionClear)
        da.setStoreAction_(Metal.MTLStoreActionStore)
        da.setClearDepth_(1.0)
        re = cb.renderCommandEncoderWithDescriptor_(rp)
        re.setCullMode_(Metal.MTLCullModeNone)
        # background quads
        re.setRenderPipelineState_(self.bg_pipe)
        re.setDepthStencilState_(self.bg_ds)
        re.setVertexBuffer_offset_atIndex_(self.env_buf, 0, 1)
        re.setFragmentBuffer_offset_atIndex_(self.env_buf, 0, 1)
        re.setFragmentBuffer_offset_atIndex_(self.consts_buf, 0, 5)
        re.setFragmentTexture_atIndex_(self.bg_tex, 1)
        re.setFragmentSamplerState_atIndex_(self.samp_bg, 0)
        re.drawPrimitives_vertexStart_vertexCount_instanceCount_(Metal.MTLPrimitiveTypeTriangle, 0, 6, self.n)
        # geometry
        re.setRenderPipelineState_(self.geo_pipe)
        re.setDepthStencilState_(self.geo_ds)
        re.setVertexBuffer_offset_atIndex_(self.vbuf, 0, 0)
        re.setVertexBuffer_offset_atIndex_(self.env_buf, 0, 1)
        re.setVertexBuffer_offset_atIndex_(self.inst_buf, 0, 2)
        re.setVertexBuffer_offset_atIndex_(self.slot_geom, 0, 3)
        re.setVertexBuffer_offset_atIndex_(geom_xpos, geom_xpos_off, 4)
        re.setVertexBuffer_offset_atIndex_(geom_xmat, geom_xmat_off, 5)
        re.setVertexBuffer_offset_atIndex_(self.color_buf, 0, 6)
        re.setVertexBuffer_offset_atIndex_(self.consts_buf, 0, 7)
        re.setFragmentBuffer_offset_atIndex_(self.env_buf, 0, 1)
        re.setFragmentBuffer_offset_atIndex_(self.mat_buf, 0, 2)
        re.setFragmentBuffer_offset_atIndex_(self.sem_buf, 0, 3)
        re.setFragmentBuffer_offset_atIndex_(self.color_buf, 0, 4)
        re.setFragmentBuffer_offset_atIndex_(self.consts_buf, 0, 5)
        re.setFragmentBuffer_offset_atIndex_(self.light_buf, 0, 6)
        re.setFragmentBuffer_offset_atIndex_(light_xpos, light_xpos_off, 7)
        re.setFragmentBuffer_offset_atIndex_(light_xdir, light_xdir_off, 8)
        re.setFragmentBuffer_offset_atIndex_(self.params_buf, 0, 9)
        re.setFragmentTexture_atIndex_(self.atlas_tex, 0)
        re.setFragmentTexture_atIndex_(self.shadow_tex, 2)
        re.setFragmentSamplerState_atIndex_(self.samp, 0)
        re.setFragmentSamplerState_atIndex_(self.samp_shadow, 1)
        if self.rt is not None:
            rt = self.rt
            re.useResource_usage_stages_(rt.inst_as, Metal.MTLResourceUsageRead, Metal.MTLRenderStageFragment)
            for a in rt.prim_as:
                re.useResource_usage_stages_(a, Metal.MTLResourceUsageRead, Metal.MTLRenderStageFragment)
            re.setFragmentAccelerationStructure_atBufferIndex_(rt.inst_as, 10)
            re.setFragmentBuffer_offset_atIndex_(self.rt_offset_buf, 0, 11)
            re.setFragmentBuffer_offset_atIndex_(rt.inst_tab, 0, 12)
            re.setFragmentBuffer_offset_atIndex_(rt.slot_mesh, 0, 13)
            re.setFragmentBuffer_offset_atIndex_(rt.mesh_info, 0, 14)
            re.setFragmentBuffer_offset_atIndex_(rt.vbuf, 0, 15)
            re.setFragmentBuffer_offset_atIndex_(rt.ibuf, 0, 16)
            re.setFragmentBuffer_offset_atIndex_(rt.inst_desc, 0, 17)
        for i_off, i_count, first, count in self.tables.draws:
            re.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance_(
                Metal.MTLPrimitiveTypeTriangle, i_count, Metal.MTLIndexTypeUInt32, self.ibuf, i_off * 4, count, 0, first)
        re.endEncoding()
        # 3. untile into learner buffers
        ue = cb.computeCommandEncoder()
        ue.setComputePipelineState_(self.p_untile)
        ue.setTexture_atIndex_(self.color_tex, 0)
        ue.setTexture_atIndex_(self.depth_tex, 1)
        ue.setTexture_atIndex_(self.seg_tex, 2)
        ue.setTexture_atIndex_(self.normal_tex, 3)
        for i, name in enumerate(("rgb", "depth", "seg", "normal")):
            if name in self._out_bufs:
                buf, off = self._out_bufs[name]
                ue.setBuffer_offset_atIndex_(buf, off, i)
        ue.setBuffer_offset_atIndex_(self.sem_buf, 0, 4)
        ue.setBuffer_offset_atIndex_(self.env_buf, 0, 5)
        ue.setBuffer_offset_atIndex_(self.consts_buf, 0, 6)
        ue.setBuffer_offset_atIndex_(self.outputs_buf, 0, 7)
        ue.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.tw, self.th, self.n), Metal.MTLSize(16, 16, 1))
        ue.endEncoding()

    def render(self, sim, after_value: int | None = None) -> int:
        """GPU path: render the current physics state of ``sim`` (a BatchSim). Waits for
        ``sim.event >= after_value`` (default: the latest value), signals ``self.ctx.event``;
        returns that value. Nothing synchronizes the host."""
        if after_value is None:
            after_value = sim.event.value
        names = ["geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat"] + (["light_xpos", "light_xdir"] if self.n_light else [])
        views = {name: wm.buffer_of(getattr(sim.d, name)) for name in names}
        cb = self.ctx.command_buffer()
        self.ctx.wait_for(cb, sim.event, after_value)
        lx = views.get("light_xpos"); ld = views.get("light_xdir")
        self._encode(cb, views["geom_xpos"].buffer, views["geom_xpos"].offset, views["geom_xmat"].buffer,
                     views["geom_xmat"].offset, views["cam_xpos"].buffer, views["cam_xpos"].offset,
                     views["cam_xmat"].buffer, views["cam_xmat"].offset,
                     lx.buffer if lx else None, lx.offset if lx else 0, ld.buffer if ld else None, ld.offset if ld else 0)
        v = self.ctx.signal(cb)
        self.ctx.commit(cb)
        return v

    def after(self, value: int) -> None:
        """Order torch's subsequent kernels after the render that signalled ``value``."""
        tb.wait_event(self.ctx.event, value)

    def wait_sim_after_render(self, sim, value: int) -> None:
        """Order the sim's next step after the render that signalled ``value`` (so physics does not
        overwrite poses the renderer is still reading)."""
        sim.wait(self.ctx.event, value)

    # -- host path (tests, tools): render from MjData ------------------------------------------------

    def render_host(self, datas, cam_pos=None, cam_quat=None):
        """Render from a list of N ``mujoco.MjData`` (mj_forward'd). Optional per-env camera pose
        overrides ``(N,3)``/``(N,4)`` replace the model camera. Synchronizes; returns numpy outputs."""
        n, ng, nc = self.n, self.m.ngeom, max(self.m.ncam, 1)
        gx = np.stack([d.geom_xpos for d in datas]).astype(np.float32).reshape(n, ng, 3)
        gm = np.stack([d.geom_xmat for d in datas]).astype(np.float32).reshape(n, ng, 9)
        if cam_pos is not None:
            cx = np.zeros((n, nc, 3), np.float32); cm = np.zeros((n, nc, 9), np.float32)
            cx[:, self.cam_id] = cam_pos
            cm[:, self.cam_id] = quats_to_mats(np.asarray(cam_quat, np.float64)).reshape(n, 9)
        else:
            cx = np.stack([d.cam_xpos for d in datas]).astype(np.float32).reshape(n, nc, 3)
            cm = np.stack([d.cam_xmat for d in datas]).astype(np.float32).reshape(n, nc, 9)
        c = self.ctx
        c.write_buffer(self._host_geom_xpos, gx); c.write_buffer(self._host_geom_xmat, gm)
        c.write_buffer(self._host_cam_xpos, cx); c.write_buffer(self._host_cam_xmat, cm)
        if self.m.nlight:
            c.write_buffer(self._host_light_xpos, np.stack([d.light_xpos for d in datas]).astype(np.float32).reshape(n, -1, 3))
            c.write_buffer(self._host_light_xdir, np.stack([d.light_xdir for d in datas]).astype(np.float32).reshape(n, -1, 3))
        cb = c.command_buffer()
        self._encode(cb, self._host_geom_xpos, 0, self._host_geom_xmat, 0, self._host_cam_xpos, 0, self._host_cam_xmat, 0)
        c.commit(cb)
        c.synchronize()
        res = {}
        for name, a in self._out_arrays.items():
            res[name] = a.numpy().copy()
        return res
