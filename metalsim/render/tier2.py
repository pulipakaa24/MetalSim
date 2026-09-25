"""Tier 2: progressive path tracer on Metal hardware ray tracing (see shaders/pathtrace.metal).

Same scene tables, materials, lights and camera model as tiers 0/1; the sensor layer's
acceleration structures (refit from the physics state each frame). Intended for dataset
generation and validation, not rollouts: cost is spp x bounces rays per pixel.

Validation lives in tests/test_render_tier2.py: analytic Lambertian irradiance under a directional
light, white-furnace energy conservation, convergence with sample count, and agreement of the
direct-light term with tier 0.

Physical (USD / RTX) mode, all off by default (the defaults reproduce the original renderer bit for bit):
  ``material_model``  'legacy' | 'usd_preview' / 'omnipbr' (MDL-style layering: GGX layer over Lambert, the base
                      attenuated by the layer's directional albedo; OmniPBR's custom_curve_layer 0.08 -> 1 weighted by
                      specular_level) | archived variants 'usd_preview_storm' (Hydra Storm's per-light (1 - F(V.H))
                      diffuse weight) and 'omnipbr_curve' (curve-weighted base); or a dict {geom name: model}, key
                      None = default
  ``usd_lights``      per model light: dict(intensity, color, angle_deg) in USD units; a DistantLight's
                      irradiance normal to it is pi * intensity * color (the stack's E = pi * diffuse
                      convention), its angle the sun disk's angular diameter (soft penumbra, same mean)
  ``dome``            (intensity, color): uniform environment radiance intensity * color (a DomeLight
                      without a texture); replaces the MuJoCo skybox colour
  ``headlight``       False turns MuJoCo's headlight off (Isaac has none)
  ``tonemap``         'linear' (clamp) | 'rtx' (Kit's default: exposure, ACES, sRGB); ``exposure`` is the
                      linear scale applied before it (``rtx_exposure()`` gives Kit's default camera)
  ``clip_far``        camera far clipping distance (z-depth; beyond it the background shows, as in a USD camera's
                      clippingRange); 0 = none
  ``center_sample``   first sample of each pass through the pixel centre, so depth / normal / segmentation are
                      pixel-centre values like a rasterizer's and Isaac's (jittered depth at grazing angles varied by
                      metres on the far ground)
  ``denoise``         None | 'oidn' (Open Image Denoise on its Metal device, shared buffers) | 'atrous'
                      (edge-avoiding a-trous on albedo-demodulated radiance, in the render command buffer)
See docs/research/rendering_vs_rtx_2026-09-25.md.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp
import Metal

from metalsim.interop import torch_bridge as tb
from metalsim.interop import warp_metal as wm
from metalsim.render.metal_context import MetalContext
from metalsim.render.scene_tables import CameraIntrinsics
from metalsim.render.tier0 import RenderOutputs
from metalsim.sensors.raytrace import RayTracer


class Tier2Renderer:
    def __init__(self, model, n_envs: int, *, width=256, height=256, camera=None, spp=4, max_bounces=4,
                 max_group=3, include_planes=True, decimate_faces=0, exposure=1.0, seed=0,
                 material_model="legacy", usd_lights=None, dome=None, headlight=None, tonemap="linear",
                 denoise=None, denoise_quality="high", atrous_iters=5, aux=False, center_sample=False, clip_far=0.0,
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
        lights = t.lights.copy(); params = t.params.copy()
        env_scale = 1.0
        if usd_lights is not None:
            for i, ul in enumerate(usd_lights):
                if ul is None or i >= model.nlight:
                    continue
                lights[i, 4:7] = float(ul.get("intensity", 1.0)) * np.asarray(ul.get("color", (1, 1, 1)), np.float32)
                lights[i, 7] = np.deg2rad(float(ul.get("angle_deg", 0.0))) / 2      # angular radius
        if dome is not None:
            params[4, :3] = np.asarray(dome[1], np.float32); params[4, 3] = 1.0; env_scale = float(dome[0])
        if headlight is False:
            params[3, 3] = 0.0
        self.light_buf = c.buffer(lights.nbytes, lights, "pt_lights")
        self.params_buf = c.buffer(params.nbytes, params, "pt_params")
        self.mat_model = material_model
        self.mat_buf = None
        if material_model != "legacy":
            mats = t.materials.copy().reshape(self.G, 16)
            ids = {"legacy": 0, "usd_preview_storm": 1, "omnipbr_curve": 2, "usd_preview": 3, "omnipbr": 4,
                   "usd_preview_v1": 1, "omnipbr_v1": 2, "usd_preview_fv": 5, "omnipbr_fv": 6}
            spec_ = material_model if isinstance(material_model, dict) else {None: material_model}
            for slot, g in enumerate(t.geoms):
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                mats[slot, 15] = ids[spec_.get(nm, spec_.get(None, "legacy"))]
            self.mat_buf = c.buffer(mats.nbytes, mats, "pt_materials")
        ai = t.atlas.image
        self.atlas_tex = c.texture2d(ai.shape[1], ai.shape[0], Metal.MTLPixelFormatRGBA8Unorm, Metal.MTLTextureUsageShaderRead,
                                     label="pt_atlas", storage=Metal.MTLStorageModeShared)
        c.upload_texture(self.atlas_tex, ai)
        self.samp = c.sampler(linear=True, repeat=True)
        self.consts = np.zeros(16, np.uint32)
        self.consts[:4] = (n_envs, width, height, max(model.ncam, 1))
        self.consts[4:8] = (max(model.nlight, 1) if model.nlight else 0, self.G, spp, max_bounces)
        self.consts[8:12] = (0, self.rt.tpr, 1, seed)
        cf = self.consts.view(np.float32); cf[12] = self.rt.stride; cf[13] = exposure; cf[14] = env_scale; cf[15] = clip_far
        self.denoise = denoise
        self.aux = bool(aux or denoise)
        self.base_flags = (2 if material_model != "legacy" else 0) | (4 if tonemap == "rtx" else 0) | (8 if self.aux else 0) | (16 if center_sample else 0)
        self.consts_buf = c.buffer(self.consts.nbytes, self.consts, "pt_consts")
        self.spp, self.max_bounces = spp, max_bounces
        self.frame = 0
        n, h, w = n_envs, height, width
        self._accum = wp.zeros((n, h, w, 4), dtype=wp.float32, device=device)
        self._arrays = {"rgb": wp.zeros((n, h, w, 3), dtype=wp.uint8, device=device), "depth": wp.zeros((n, h, w), dtype=wp.float32, device=device),
                        "seg": wp.zeros((n, h, w), dtype=wp.int32, device=device), "normal": wp.zeros((n, h, w, 3), dtype=wp.float32, device=device)}
        wp.synchronize_device(device)
        if self.aux:
            # denoiser buffers from the Metal context: hazard-tracked shared buffers (OIDN's Metal device rejects
            # Warp's untracked allocations: "Metal buffers without hazard tracking are not supported")
            shapes = {"hdr": (n, h, w, 4), "aux_accum": (n, h, w, 8), "albedo": (n, h, w, 3), "nrm": (n, h, w, 3), "hdr_out": (n, h, w, 4), "tmp": (n, h, w, 4)}
            self._aux_shapes = shapes
            self._aux_bufs = {k: c.buffer(int(np.prod(sh)) * 4, label=f"pt_{k}") for k, sh in shapes.items()}
            self._aux = None
        else:
            self._aux = {"hdr": wp.zeros(4, dtype=wp.float32, device=device), "aux_accum": wp.zeros(8, dtype=wp.float32, device=device),
                         "albedo": wp.zeros(4, dtype=wp.float32, device=device), "nrm": wp.zeros(4, dtype=wp.float32, device=device)}
        wp.synchronize_device(device)
        self._views = {k: wm.buffer_of(a) for k, a in self._arrays.items()}
        if self.aux:
            from types import SimpleNamespace
            self._aux_views = {k: SimpleNamespace(buffer=bf, offset=0) for k, bf in self._aux_bufs.items()}
        else:
            self._aux_views = {k: wm.buffer_of(a) for k, a in self._aux.items()}
        self._accum_view = wm.buffer_of(self._accum)
        self.out = RenderOutputs(**{k: tb.mps_tensor(a) for k, a in self._arrays.items()})
        self.t_accum = tb.mps_tensor(self._accum)
        lib = c.library("pathtrace")
        self.pipe = c.compute_pipeline(lib, "path_trace")
        self.tm_consts = np.zeros(8, np.uint32); self.tm_consts[:3] = (n, width, height); self.tm_consts[3] = self.base_flags
        self.tm_consts.view(np.float32)[4] = exposure
        self._oidn = None
        if denoise is not None:
            dl = c.library("denoise")
            self.pipe_tonemap = c.compute_pipeline(dl, "tonemap_hdr")
            if denoise == "atrous":
                self.pipe_atrous = c.compute_pipeline(dl, "atrous_step")
                self.atrous_iters = int(atrous_iters)
            elif denoise == "oidn":
                from metalsim.render.denoise import OIDNDenoiser
                av = self._aux_views
                self._oidn = OIDNDenoiser(c.queue, n, width, height, (av["hdr"].buffer, av["hdr"].offset), (av["hdr_out"].buffer, av["hdr_out"].offset),
                                          (av["albedo"].buffer, av["albedo"].offset), (av["nrm"].buffer, av["nrm"].offset), quality=denoise_quality)
            else:
                raise ValueError(f"unknown denoiser {denoise!r}")
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
        self.consts[10] = (1 if reset else 0) | self.base_flags
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.pipe)
        rt = self.rt
        rt._use_resources(ce)
        ce.setAccelerationStructure_atBufferIndex_(rt.inst_as, 0)
        ce.setBuffer_offset_atIndex_(rt.inst_tab, 0, 1); ce.setBuffer_offset_atIndex_(rt.slot_mesh, 0, 2)
        ce.setBuffer_offset_atIndex_(rt.mesh_info, 0, 3); ce.setBuffer_offset_atIndex_(rt.vbuf, 0, 4)
        ce.setBuffer_offset_atIndex_(rt.ibuf, 0, 5); ce.setBuffer_offset_atIndex_(rt.inst_desc, 0, 6)
        ce.setBuffer_offset_atIndex_(self.mat_buf or rt.mat_buf, 0, 7); ce.setBuffer_offset_atIndex_(self.color_buf, 0, 8)
        ce.setBuffer_offset_atIndex_(self.light_buf, 0, 9)
        ce.setBuffer_offset_atIndex_(light_xpos, lxo, 10); ce.setBuffer_offset_atIndex_(light_xdir, ldo, 11)
        ce.setBuffer_offset_atIndex_(self.params_buf, 0, 12); ce.setBuffer_offset_atIndex_(self.cam_buf, 0, 13)
        ce.setBuffer_offset_atIndex_(cam_xpos, cxo, 14); ce.setBuffer_offset_atIndex_(cam_xmat, cmo, 15)
        ce.setBytes_length_atIndex_(self.consts.tobytes(), self.consts.nbytes, 16)   # copied at encode time (per pass)
        ce.setBuffer_offset_atIndex_(self._accum_view.buffer, self._accum_view.offset, 17)
        for i, k in enumerate(("rgb", "depth", "seg", "normal")):
            v = self._views[k]; ce.setBuffer_offset_atIndex_(v.buffer, v.offset, 18 + i)
        for i, k in enumerate(("hdr", "aux_accum", "albedo", "nrm")):
            v = self._aux_views[k]; ce.setBuffer_offset_atIndex_(v.buffer, v.offset, 22 + i)
        ce.setTexture_atIndex_(self.atlas_tex, 0); ce.setSamplerState_atIndex_(self.samp, 0)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.tw, self.th, self.n), Metal.MTLSize(8, 8, 1))
        ce.endEncoding()
        self.frame += 1

    def _encode_atrous(self, cb):
        """Edge-avoiding a-trous wavelet filter (Dammertz et al. 2010) on albedo-demodulated radiance, the
        spatial half of SVGF (Schied et al. 2017): 5x5 B3 kernel at steps 1, 2, 4, ..., weights from the
        normal, depth and luminance differences; all envs in one dispatch per iteration."""
        av = self._aux_views; dv = self._views["depth"]
        N = self.atrous_iters
        pair = (av["hdr_out"], av["tmp"]) if N % 2 else (av["tmp"], av["hdr_out"])   # the last iteration lands in hdr_out
        src = av["hdr"]
        for it in range(N):
            out = pair[it % 2]
            ce = cb.computeCommandEncoder(); ce.setComputePipelineState_(self.pipe_atrous)
            k = np.array([self.n, self.tw, self.th, 1 << it, 1 if it == 0 else 0, 1 if it == N - 1 else 0, 0, 0], np.uint32)
            ce.setBytes_length_atIndex_(k.tobytes(), k.nbytes, 0)
            ce.setBuffer_offset_atIndex_(src.buffer, src.offset, 1); ce.setBuffer_offset_atIndex_(out.buffer, out.offset, 2)
            ce.setBuffer_offset_atIndex_(av["albedo"].buffer, av["albedo"].offset, 3); ce.setBuffer_offset_atIndex_(av["nrm"].buffer, av["nrm"].offset, 4)
            ce.setBuffer_offset_atIndex_(dv.buffer, dv.offset, 5)
            ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.tw, self.th, self.n), Metal.MTLSize(8, 8, 1))
            ce.endEncoding()
            src = out

    def _encode_tonemap(self, cb):
        av = self._aux_views; v = self._views["rgb"]
        ce = cb.computeCommandEncoder(); ce.setComputePipelineState_(self.pipe_tonemap)
        ce.setBytes_length_atIndex_(self.tm_consts.tobytes(), self.tm_consts.nbytes, 0)
        ce.setBuffer_offset_atIndex_(av["hdr_out"].buffer, av["hdr_out"].offset, 1); ce.setBuffer_offset_atIndex_(v.buffer, v.offset, 2)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.tw, self.th, self.n), Metal.MTLSize(8, 8, 1))
        ce.endEncoding()

    def _postprocess(self, cb):
        """Denoise + tone map. a-trous: encoded into ``cb``. OIDN: ``cb`` is committed, OIDN runs on the same
        queue, then a new command buffer tone-maps; returns the command buffer to finish on."""
        if self.denoise == "atrous":
            self._encode_atrous(cb); self._encode_tonemap(cb); return cb
        if self.denoise == "oidn":
            self.ctx.commit(cb)
            self._oidn.denoise(sync=False)
            cb2 = self.ctx.command_buffer(); self._encode_tonemap(cb2); return cb2
        return cb

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
        cb = self._postprocess(cb)
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
        from metalsim.render.scene_tables import quats_to_mats
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
        cb = self._postprocess(cb)
        c.commit(cb); c.synchronize()
        out = {k: a.numpy().copy() for k, a in self._arrays.items()}
        if self.aux:
            A = lambda k: self.ctx.buffer_array(self._aux_bufs[k], np.float32, self._aux_shapes[k]).copy()
            out["hdr_mean"] = A("hdr")[..., :3]
            out["albedo"] = A("albedo"); out["aux_normal"] = A("nrm")
            if self.denoise:
                out["hdr_denoised"] = A("hdr_out")[..., :3]
        acc = self._accum.numpy()
        out["hdr"] = acc[..., :3] / np.maximum(acc[..., 3:4], 1.0)
        out["samples"] = acc[..., 3].copy()
        return out


def rtx_exposure(film_iso=100.0, shutter=50.0, f_number=5.0, cm2_factor=1.0):
    """Kit's default physical camera exposure (see docs/research/rendering_vs_rtx_2026-09-25.md)."""
    return cm2_factor * (film_iso / 100.0) / (shutter * f_number ** 2)


def usd_scene_kwargs(meta, material_model=None, tonemap=True):
    """Tier-2 keyword arguments that reproduce Isaac's parity scene in USD units from its meta.json:
    sun DistantLight (intensity, colour, angle), textureless DomeLight, no headlight, RTX tone mapping."""
    L = meta["lights"]
    sun = {"intensity": L["sun"]["intensity"], "color": L["sun"]["color"], "angle_deg": L["sun"].get("angle", 0.53)}
    kw = dict(usd_lights=[sun], dome=(L["dome"]["intensity"], L["dome"]["color"]), headlight=False, center_sample=True)
    if material_model is not None:
        kw["material_model"] = material_model
    if tonemap:
        kw.update(tonemap="rtx", exposure=rtx_exposure())
    return kw


def mujoco_scene_kwargs(model, sun_angle_deg=0.53, exposure=1.0, material_model="omnipbr"):
    """The physical tier-2 path for a MuJoCo-authored scene (no USD intensities): the model's lights keep their
    MuJoCo radiometry (irradiance = pi * diffuse), directional lights become 0.53-degree sun disks, materials use
    the MDL-layered OmniPBR BRDF (F0 = 0.08 * specular), and the output goes through RTX's display transform
    (exposure, ACES, sRGB). The headlight and skybox stay as authored."""
    import mujoco
    lights = []
    for i in range(model.nlight):
        directional = int(model.light_type[i]) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL) if hasattr(model, "light_type") else bool(model.light_directional[i])
        lights.append({"intensity": 1.0, "color": tuple(float(c) for c in model.light_diffuse[i]), "angle_deg": sun_angle_deg if directional else 0.0})
    return dict(material_model=material_model, usd_lights=lights, tonemap="rtx", exposure=exposure)
