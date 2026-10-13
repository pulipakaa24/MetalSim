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
  ``firefly_clamp``   0 (off, the default and the unbiased estimator) or a luminance cap applied to every path
                      sample before accumulation: removes fireflies from rare high-energy paths at the cost of a
                      downward bias on bright indirect light (opt-in; docs/DECISIONS.md)
See docs/research/rendering_vs_rtx_2026-09-25.md.

Particles (``set_particles``): simulated particles (MPM sand, ...) drawn as render grains, instances of a few small
irregular grain meshes in the same instance acceleration structure as the scene, rebuilt from the particle buffer on
the GPU every frame, with a per-grain albedo palette (metalsim/render/particles.py, PARTICLES.md).

Environment map (``set_environment``): an equirectangular HDR image replaces the sky colour as the miss
radiance and as an importance-sampled light (next-event estimation with multiple importance sampling
against the BSDF sampler, as PBRT's InfiniteAreaLight). Orientation follows a textured USD DomeLight as
Kit/RTX renders it on Isaac's z-up stages (pole +z; docs/PARITY.md 1.7); radiance = map x dome intensity x
2^exposure x color (USD LightAPI). ``set_environment_from_model`` reads the DomeLight that
``metalsim.scene.usd_to_mjcf`` records in the model's custom text. Tests: a furnace under a constant map and
an analytic bright-window plane (importance-sampled estimate unbiased within 2 %, uniform table > 3x noisier).
"""
from __future__ import annotations

import math
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


def env_sampling_table(hdr, grid=(512, 256)):
    """Importance-sampling table of an equirectangular map on a (gw, gh) grid (PBRT 3rd ed. 12.6 / 14.2.4,
    Distribution2D over luminance x sin(theta)): pixel weight = area-averaged luminance x sin(theta), floored
    so no cell has zero probability (the MIS estimator needs full support). Returns
    ([marginal CDF over rows (gh) | per-row conditional CDFs (gh*gw) | density over (u, v) in [0,1]^2 (gh*gw)],
    gw, gh) as float32. The density is the sampler's own (u, v) density, so any grid is unbiased; finer grids
    only lower the variance."""
    hdr = np.asarray(hdr, np.float32)[..., :3]
    H, W = hdr.shape[:2]
    gw, gh = min(int(grid[0]), W), min(int(grid[1]), H)
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    # area average onto the grid: block sums along each axis (blocks of floor/ceil(W/gw) pixels)
    ex = np.linspace(0, W, gw + 1).astype(int); ey = np.linspace(0, H, gh + 1).astype(int)
    small = np.add.reduceat(np.add.reduceat(lum, ey[:-1], axis=0), ex[:-1], axis=1)
    small /= np.outer(np.diff(ey), np.diff(ex))
    sin_t = np.sin((np.arange(gh) + 0.5) / gh * np.pi)[:, None]
    w = np.maximum(small, 1e-6 * max(float(small.max()), 1e-6)) * sin_t
    row = w.sum(1)
    marg = np.cumsum(row) / row.sum()
    cond = np.cumsum(w, 1) / row[:, None]
    marg[-1] = 1.0; cond[:, -1] = 1.0
    density = w / w.sum() * (gw * gh)
    return np.concatenate([marg, cond.ravel(), density.ravel()]).astype(np.float32), gw, gh


# Kit/RTX maps an equirectangular DomeLight with pole +z and the image centre (u = 0.5) facing -x (measured on Isaac Sim
# 5.1 and 6.1 path-traced frames with marked maps, 2026-09-26, docs/research/isaac_side_session_2026-09-26.md); our shader's
# centre faces -y, so stages authored for Kit get this constant added to their dome yaw. The direct set_environment() API
# keeps the shader's own convention (its tests are written against it).
KIT_DOME_AZIMUTH = -math.pi / 2

class Tier2Renderer:
    def __init__(self, model, n_envs: int, *, width=256, height=256, camera=None, spp=4, max_bounces=4,
                 max_group=3, include_planes=True, decimate_faces=0, exposure=1.0, seed=0,
                 material_model="legacy", usd_lights=None, dome=None, headlight=None, tonemap="linear",
                 denoise=None, denoise_quality="high", atrous_iters=5, aux=False, center_sample=False, clip_far=0.0,
                 terrain_slots: bool = True, draw_hfields: bool = True, firefly_clamp=0.0,
                 appearance: str = "mujoco", skybox: bool = True, transparency: bool = True, transparent_ids: str = "surface",
                 visual=None, deformables=(), tube_slices=None, cap_stacks=None, dyn_rebuild_every: int = 32,
                 infinite_planes: bool = True, hfield_skirts: bool = True, per_world_size: bool = True,
                 model_fields: bool = True, lens=None, dyn_layout: str | None = None, env_layout: str | None = None,
                 haze: bool = True, fog: bool = False, tendons: str = "mujoco", rolling_shutter=None,
                 area_lights: bool = True, ctx: MetalContext | None = None, device="metal:0"):
        """``haze`` / ``fog``: MuJoCo's haze over the first infinite plane (drawn with the skybox texture) and its linear
        fog (off by default, as ``mujoco.Renderer``), composited on the camera ray at the first opaque surface
        (``metalsim.render.haze``; see ``Tier0Renderer``). ``tendons`` / ``rolling_shutter``: see ``Tier0Renderer``
        (rolling shutter here: per band the structure is refit at the band's poses and only its rows are traced).
        ``set_texture_pool`` / ``t_material_ids``: per-env material and texture swaps (``metalsim.render.env_materials``).
        ``terrain_slots`` / ``draw_hfields`` / ``appearance``: see ``Tier0Renderer`` (False / "legacy" = the
        previous behaviour). ``skybox``: the model's skybox texture is the environment (camera background and the
        radiance of every escaping ray, MuJoCo's cube mapping, unlit texel values x env scale) instead of its mean
        colour; a ``dome`` or ``set_environment`` map replaces it.

        ``transparency``: a geom's (material-resolved) rgba alpha < 1 is surface presence, as a USD opacity in Isaac's
        path tracer: rays meet it with probability alpha and otherwise pass unchanged, shadow rays are attenuated by
        (1 - alpha) per surface (pathtrace.metal ``transmit``); False = every geom opaque (the previous behaviour).
        ``transparent_ids``: annotations of pixels whose first surface is transparent, as ``Tier0Renderer``: "surface"
        (default, MuJoCo's; "mujoco" is an alias) or "behind" (the first opaque surface behind it).
        ``visual`` / ``deformables`` / ``tube_slices`` / ``cap_stacks``: sites, flexes, skins, spatial tendons and user
        deformable meshes, see ``Tier0Renderer``; ``dyn_rebuild_every`` / ``dyn_layout`` / ``env_layout``: see
        ``RayTracer`` (default "sbvh": one box per env, the env's triangles traversed in software; the compact env grid).
        ``infinite_planes`` / ``hfield_skirts`` / ``per_world_size`` /
        ``model_fields``: see ``Tier0Renderer`` (False = the previous behaviour). With ``usd_lights``, those lights keep
        their USD intensity (a per-world ``light_diffuse`` does not apply to them). ``lens``: a
        ``metalsim.render.lens.Lens`` (f-theta / Kannala-Brandt fisheye; default pinhole from the model camera and the
        per-env intrinsics); pixels outside its field of view are black with depth 0.

        ``area_lights``: the USD Rect / Disk / Cylinder / Sphere lights the importer recorded (``usd_area_lights``) are
        sampled as true area emitters (UsdLux radiance; soft shadows, exact near-field irradiance;
        metalsim.render.area_lights); False = the far-field spot / point lights of the import (the previous
        behaviour)."""
        from metalsim.render.lens import Lens
        import mujoco
        self.m, self.n, self.tw, self.th = model, n_envs, width, height
        self.device = device
        self.ctx = ctx or MetalContext(device)
        cam0 = 0 if camera is None else (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) if isinstance(camera, str) else int(camera))
        tube_view = None
        if not (tube_slices or cap_stacks) and model.ncam:          # tube tessellation for this camera (rendering only)
            from metalsim.render.dynamic_mesh import camera_view
            ci = CameraIntrinsics.from_model(model, cam0, width, height)
            tube_view = camera_view(model, cam0, max(ci.fx, ci.fy))
        self.rt = RayTracer(model, n_envs, max_range=1000.0, max_group=max_group, include_planes=include_planes,
                            decimate_faces=decimate_faces, terrain_slots=terrain_slots, draw_hfields=draw_hfields,
                            appearance=appearance,
                            visual=mujoco.MjvOption() if visual is None else visual, deformables=deformables,
                            tube_slices=tube_slices, cap_stacks=cap_stacks, tube_view=tube_view,
                            dyn_rebuild_every=dyn_rebuild_every, infinite_planes=infinite_planes, hfield_skirts=hfield_skirts,
                            per_world_size=per_world_size, dyn_layout=dyn_layout, env_layout=env_layout,
                            tendons=tendons, ctx=self.ctx, device=device)

        self.model_fields = bool(model_fields)
        self.lens = lens or Lens.pinhole()
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
        lock = np.zeros(max(model.nlight, 1), np.uint32)
        env_scale = 1.0
        if usd_lights is not None:
            for i, ul in enumerate(usd_lights):
                if ul is None or i >= model.nlight:
                    continue
                lights[i, 4:7] = float(ul.get("intensity", 1.0)) * np.asarray(ul.get("color", (1, 1, 1)), np.float32)
                lights[i, 7] = np.deg2rad(float(ul.get("angle_deg", 0.0))) / 2      # angular radius
                lock[i] = 1
        self._dome = None if dome is None else (float(dome[0]), tuple(float(v) for v in dome[1]))
        if dome is not None:
            params[4, :3] = np.asarray(dome[1], np.float32); params[4, 3] = 1.0; env_scale = float(dome[0])
        if transparent_ids not in ("mujoco", "behind", "surface"):
            raise ValueError(f"transparent_ids must be 'surface', 'behind' or 'mujoco', got {transparent_ids!r}")
        self.transparency = transparency
        if (dome is not None or not skybox) and params.shape[0] > 6:
            params[6, 1] = 0.0                # the skybox texture is not drawn (uniform dome / sky colour)
        if headlight is False:
            params[3, 3] = 0.0
        from metalsim.render.area_lights import AreaLights
        self.area = AreaLights(model) if (area_lights and model.nlight) else None
        if self.area is not None and not self.area:
            self.area = None
        if self.area is not None:
            lights, lock = self.area.apply(lights, lock)
        from metalsim.render.model_sync import ModelSync, unit_rgba
        self.sync = ModelSync(c, model, n_envs, t, self.color_buf, lights, lock, opaque=not transparency,
                              light_tail_bytes=self.area.tail_bytes(n_envs) if self.area is not None else 0)
        if self.area is not None:     # the emitters' world X axes after the light records
            self.area.attach(c, self.sync.lights, n_envs)
        self.light_buf = self.sync.lights                    # (n_envs, max(nlight, 1)) records
        self.params_buf = c.buffer(params.nbytes, params, "pt_params")
        self.mat_model = material_model
        # the renderer's own material table (rgba = 1: the model sync supplies the colour per world; column 15 is
        # the reflectance in legacy mode, the per-slot BRDF model id otherwise)
        mats = unit_rgba(t.materials)
        if material_model != "legacy":
            ids = {"legacy": 0, "usd_preview_storm": 1, "omnipbr_curve": 2, "usd_preview": 3, "omnipbr": 4,
                   "usd_preview_v1": 1, "omnipbr_v1": 2, "usd_preview_fv": 5, "omnipbr_fv": 6}
            spec_ = material_model if isinstance(material_model, dict) else {None: material_model}
            for slot, g in enumerate(t.geoms):
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                mats[slot, 15] = ids[spec_.get(nm, spec_.get(None, "legacy"))]
            for j, ex in enumerate(t.extras):    # extra slots (sites, flexes, ...) by their own name
                mats[len(t.geoms) + j, 15] = ids[spec_.get(ex["name"], spec_.get(None, "legacy"))]
        # transparency=False: the model sync writes alpha 1 (ModelSync(opaque=True))
        self.mat_buf = c.buffer(4 * (mats.size + self.G * 8), self._with_ext(mats), "pt_materials")
        from metalsim.render.env_materials import EnvMaterials   # per-env material / texture swaps (after the tables)
        self.env_mats = EnvMaterials(c, model, t, n_envs, self.sync, device=device)
        self.t_material_ids = self.env_mats.t_material_ids
        self._env_mats_on, self._env_mats_version = False, -1
        ai = t.atlas.image
        self.atlas_tex = c.texture2d(ai.shape[1], ai.shape[0], Metal.MTLPixelFormatRGBA8Unorm, Metal.MTLTextureUsageShaderRead,
                                     label="pt_atlas", storage=Metal.MTLStorageModeShared)
        c.upload_texture(self.atlas_tex, ai)
        self.samp = c.sampler(linear=True, repeat=True)
        # PTConsts (shaders/pathtrace.metal): 16 words as before + [16:18] env grid, [20] env_yaw, [21] clamp_lum, [24:28] env_mul
        # [28:32] ray_limit, [32:48] lens, [48:76] haze (HazeParams); the EnvGrid follows at encode time
        self.consts = np.zeros(76, np.uint32)
        self.consts[:4] = (n_envs, width, height, max(model.ncam, 1))
        self.consts[4:8] = (max(model.nlight, 1) if model.nlight else 0, self.G, spp, max_bounces)
        self.consts[8:12] = (0, self.rt.tpr, 1, seed)
        cf = self.consts.view(np.float32); cf[12] = self.rt.stride; cf[13] = exposure; cf[14] = env_scale; cf[15] = clip_far
        cf[21] = float(firefly_clamp); cf[24:27] = 1.0
        self.consts[23] = {"mujoco": 0, "surface": 0, "behind": 1}[transparent_ids]   # PTConsts.trans_ids
        cf[28] = self.rt.ray_limit                                   # PTConsts.ray_limit
        self.consts[32:48] = self.lens.pack().view(np.uint32)      # PTConsts.lens
        from metalsim.render import haze as hz
        hp = hz.haze_params(model, t.geoms, haze=haze, fog=fog, sky_drawn=bool(params.shape[0] > 6 and params[6, 1] > 0.5))
        hp[16:19] *= env_scale; hp[24:27] *= env_scale            # haze / fog colours in the skybox's radiance units
        self.haze_params = hp
        self.consts[48:48 + hz.WORDS] = hp.view(np.uint32)         # PTConsts.haze
        self.denoise = denoise
        self.aux = bool(aux or denoise)
        self.base_flags = (2 if material_model != "legacy" else 0) | (4 if tonemap == "rtx" else 0) | (8 if self.aux else 0) | (16 if center_sample else 0)
        self.env_flags = 0            # FLAG_ENVMAP (32) once set_environment has a map
        self.consts_buf = c.buffer(self.consts.nbytes, self.consts, "pt_consts")
        # environment map resources (placeholders until set_environment; the shader ignores them without FLAG_ENVMAP)
        self._env_res = {}
        self._set_env_resources(np.zeros((1, 1, 4), np.float32), np.zeros(3, np.float32))
        sd = Metal.MTLSamplerDescriptor.new()
        sd.setMinFilter_(Metal.MTLSamplerMinMagFilterLinear); sd.setMagFilter_(Metal.MTLSamplerMinMagFilterLinear)
        sd.setSAddressMode_(Metal.MTLSamplerAddressModeRepeat); sd.setTAddressMode_(Metal.MTLSamplerAddressModeClampToEdge)
        self.env_samp = c.device.newSamplerStateWithDescriptor_(sd)
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
        self._haze_on = hz.active(self.haze_params)
        self._build_pipeline()
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
        # rolling shutter: the previous frame's pose inputs per env (metalsim.render.rolling)
        self.rolling, self.history = rolling_shutter, None
        if rolling_shutter is not None:
            from metalsim.render.rolling import PoseHistory
            rolling_shutter.bands(height)
            nc = max(model.ncam, 1)
            f = {"geom_xpos": ("pos", 3 * model.ngeom), "geom_xmat": ("mat", 9 * model.ngeom),
                 "cam_xpos": ("pos", 3 * nc), "cam_xmat": ("mat", 9 * nc)}
            if model.nlight:
                f.update(light_xpos=("pos", 3 * model.nlight), light_xdir=("dir", 3 * model.nlight))
            if self.rt.pose is not None:
                f.update(site_xpos=("pos", 3 * model.nsite), site_xmat=("mat", 9 * model.nsite))
            self.history = PoseHistory(self.ctx, n_envs, f)
        self.grains = None            # GrainField once set_particles is called (metalsim/render/particles.py)
        self._ias = None
        self._grain_dummy = c.buffer(304, label="pt_grain_placeholder")

    # -- particles (render grains) ----------------------------------------------------------------------

    def set_particles(self, positions, radius=None, material=None, *, grains_per_particle=8, grain_radius=None,
                      radius_jitter=0.3, n_meshes=4, seed=0, grain_shape=None, ids=None, cull_z=-np.inf, grain_eps=None,
                      structure="two_level", grain_chunk=None):
        """Draw particles as render grains (granular media: MPM sand, ...). ``positions``: (n_envs, P, 3) or (P, 3)
        numpy [m], or a Warp array on the Metal device (bound directly each frame: the GPU path). ``radius``: the
        particle radius [m] (half the particle spacing): each particle's grains are spread over its cube of half side
        ``radius``. ``grain_radius``: volume-equivalent grain radius [m] (default ``radius / grains_per_particle^(1/3)``,
        about touching grains). ``material``: ``particles.GrainMaterial`` (default sand). ``positions=None`` removes
        the particles. ``grain_shape``: ``particles.grain_meshes`` options (irregularity, elongation, subdiv);
        ``cull_z``: particles below it get no grains; non-finite positions are culled. ``ids``: stable particle
        ids ((n_envs, P) or (P,) uint32) when the particle set is culled or reordered between frames (grains hash the id). Later calls with the same
        radius and grains_per_particle only replace the positions (or material); other
        arguments apply on the first call or when ``radius`` / ``grains_per_particle`` change. ``structure``:
        ``"two_level"`` (default: the grains are their own instance acceleration structure, one instance of the
        scene's) or ``"flat"`` (one top-level instance per grain); same image, see ``particles.InstanceStructure``.
        ``grain_chunk``: grains per two-level grain structure (default Metal's standard limit, 2^24; tests use less)."""
        from metalsim.render.particles import GrainField, InstanceStructure
        if positions is None:
            self.grains, self._ias = None, None
            return
        g = self.grains
        if g is None or (radius is not None and float(radius) != g.offset) or int(grains_per_particle) != g.k:
            if radius is None:
                raise ValueError("set_particles: radius is required the first time")
            gr = grain_radius if grain_radius is not None else float(radius) / float(grains_per_particle) ** (1 / 3)
            g = GrainField(self.ctx, self.n, grains_per_particle=grains_per_particle, radius=gr,
                           radius_jitter=radius_jitter, offset=float(radius), n_meshes=n_meshes, material=material,
                           seed=seed, cull_z=cull_z, shape=grain_shape)
            self.grains, self._ias = g, InstanceStructure(self.rt, g, structure, grain_chunk)
        elif material is not None:
            g.set_material(material)
        if self._ias.mode != structure or (grain_chunk is not None and grain_chunk != self._ias.chunk):
            self._ias = InstanceStructure(self.rt, g, structure, grain_chunk)
        g.cull_z = float(cull_z)
        g.set_positions(positions)
        g.set_ids(ids)
        cf = self.consts.view(np.float32)
        # flat: first grain instance, first grain mesh in the top-level list; two-level: grain count, meshes from 0
        self.consts[18:20] = (self.rt.n_inst, len(self.rt.prim_as)) if structure == "flat" else (g.n_grains, 0)
        cf[22] = grain_eps if grain_eps is not None else max(0.01 * g.radius, 1e-5)

    def _encode_scene(self, cb, gx, gxo, gm, gmo):
        """Refit the scene instances (and with particles: write the grain instances) and build the structure."""
        if self.grains is None or self.grains.n_grains == 0:
            self.rt._encode_refit_and_build(cb, gx, gxo, gm, gmo)
        else:
            self._ias.encode(cb, gx, gxo, gm, gmo)

    def _build_pipeline(self):
        """path_trace specialized on the scene: dynamic layout and infinite-plane count (RayTracer.shader_defines),
        haze / fog, per-env material entries (code that a scene does not use is compiled out: the kernel's register
        pressure sets its speed)."""
        defs = dict(self.rt.shader_defines())
        defs["HAZE_ON"] = 1 if self._haze_on else 0
        defs["ENV_MATS"] = 1 if self._env_mats_on else 0
        if self.area is not None:
            defs["AREA_LIGHTS"] = 1           # scenes with USD area lights (metalsim.render.area_lights)
        self.pipe = self.ctx.compute_pipeline(self.ctx.library("pathtrace", defs), "path_trace")

    def _encode_env_mats(self, cb, sim):
        """Per-env material entries: resolve them, keep the entry table behind the material table, and specialize the
        kernel when the feature switches on or off."""
        on = self.env_mats.encode(cb, sim)
        if on and self.env_mats.version != self._env_mats_version:
            head = self.ctx.buffer_array(self.mat_buf, np.float32, (self.G * 24,)).copy()
            ent = self.env_mats.entries.reshape(-1)
            self.mat_buf = self.ctx.buffer(4 * (head.size + ent.size), np.concatenate([head, ent]), "pt_materials")
            self._env_mats_version = self.env_mats.version
        if on != self._env_mats_on:
            self._env_mats_on = on
            self._build_pipeline()

    def set_texture_pool(self, images, **kw):
        """Per-env texture randomization (Isaac Lab randomize_visual_texture_material); see ``Tier0Renderer``."""
        return self.env_mats.set_texture_pool(images, **kw)

    def set_material_ids(self, ids):
        """(N, G) material entry per env and slot (-1 = the slot's own); host write."""
        self.env_mats.set_material_ids(ids)

    def _with_ext(self, mats):
        """(G, 16) material table followed by the (G, 8) MaterialExt table, the layout path_trace reads."""
        ext = self.tables.material_ext if self.tables.material_ext is not None else np.zeros((self.G, 8), np.float32)
        return np.concatenate([np.asarray(mats, np.float32).reshape(-1), np.asarray(ext, np.float32).reshape(-1)])

    def _encode(self, cb, reset: bool, cam_xpos, cxo, cam_xmat, cmo, light_xpos, lxo, light_xdir, ldo, rows=None):
        self.consts[8] = self.frame
        r0, r1 = rows if rows is not None else (0, self.th)
        self.consts[29] = r0                       # PTConsts._l1: the first row traced
        grains = self.grains is not None and self.grains.n_grains > 0
        two = grains and self._ias.mode == "two_level"
        if grains:
            self._ias.ensure()          # (the extended-limits flag is known once the structure is sized)
        ext = grains and self._ias.mode == "flat" and self._ias.extended
        if ext and self.rt.dyn_bvh is not None:
            raise ValueError("a flat grain structure past 2^24 instances needs the extended_limits intersector, which "
                             "the sbvh dynamic-object layout's queries cannot use: set_particles(structure='two_level') "
                             "(default) or Tier2Renderer(dyn_layout='per_env')")
        self.consts[10] = ((1 if reset else 0) | self.base_flags | self.env_flags | (64 if grains else 0) | (128 if two else 0)
                           | (256 if ext else 0))
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.pipe)
        rt = self.rt
        if grains:
            self._ias.use_resources(ce)
            ce.setAccelerationStructure_atBufferIndex_(self._ias.inst_as, 0)
        else:
            rt._use_resources(ce)
            ce.setAccelerationStructure_atBufferIndex_(rt.inst_as, 0)
        ce.setBuffer_offset_atIndex_(rt.inst_tab, 0, 1); ce.setBuffer_offset_atIndex_(rt.slot_mesh, 0, 2)
        ce.setBuffer_offset_atIndex_(rt.mesh_info, 0, 3); ce.setBuffer_offset_atIndex_(rt.vbuf, 0, 4)
        ce.setBuffer_offset_atIndex_(rt.ibuf, 0, 5); ce.setBuffer_offset_atIndex_(self._ias.inst_desc if grains else rt.inst_desc, 0, 6)
        ce.setBuffer_offset_atIndex_(self.mat_buf, 0, 7)   # + MaterialExt tail
        ce.setBuffer_offset_atIndex_(self.sync.inst_col, 0, 8)   # t_colors x the per-world model colour
        ce.setBuffer_offset_atIndex_(self.light_buf, 0, 9)
        ce.setBuffer_offset_atIndex_(light_xpos, lxo, 10); ce.setBuffer_offset_atIndex_(light_xdir, ldo, 11)
        ce.setBuffer_offset_atIndex_(self.params_buf, 0, 12); ce.setBuffer_offset_atIndex_(self.cam_buf, 0, 13)
        ce.setBuffer_offset_atIndex_(cam_xpos, cxo, 14); ce.setBuffer_offset_atIndex_(cam_xmat, cmo, 15)
        cbytes = self.consts.tobytes() + rt.grid.words().tobytes()     # PTConsts: 28 words, then the EnvGrid
        if rt.dyn_bvh is not None:      # PTConsts.dyn_args: the BVH argument struct's address (the 31 buffer slots are taken)
            cbytes += np.array([rt.dyn_bvh.args.gpuAddress(), 0], np.uint64).tobytes()
        ce.setBytes_length_atIndex_(cbytes, len(cbytes), 16)   # copied at encode time (per pass)
        ce.setBuffer_offset_atIndex_(self._accum_view.buffer, self._accum_view.offset, 17)
        for i, k in enumerate(("rgb", "depth", "seg", "normal")):
            v = self._views[k]; ce.setBuffer_offset_atIndex_(v.buffer, v.offset, 18 + i)
        for i, k in enumerate(("hdr", "aux_accum", "albedo", "nrm")):
            v = self._aux_views[k]; ce.setBuffer_offset_atIndex_(v.buffer, v.offset, 22 + i)
        ce.setBuffer_offset_atIndex_(self._env_cdf, 0, 26)
        g = self.grains if grains else None
        for i, (b, off) in enumerate(((g.vbuf, 0), (g.ibuf, 0), self._ias.grain_binding(), (g.mat_buf, 0)) if g
                                     else ((self._grain_dummy, 0),) * 4):
            ce.setBuffer_offset_atIndex_(b, off, 27 + i)
        ce.setTexture_atIndex_(self.atlas_tex, 0); ce.setSamplerState_atIndex_(self.samp, 0)
        ce.setTexture_atIndex_(self._env_tex, 1); ce.setSamplerState_atIndex_(self.env_samp, 1)
        ce.setTexture_atIndex_(self.env_mats.pool_tex, 2)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.tw, r1 - r0, self.n), Metal.MTLSize(8, 8, 1))
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

    def render(self, sim, after_value=None, reset=True, passes=1, sources=None) -> int:
        """GPU path from a BatchSim: refit + rebuild the acceleration structure, then ``passes`` path
        tracing passes of ``spp`` samples each (progressive; ``reset`` restarts accumulation).
        ``sources``: vertex arrays of the dynamic objects not in ``sim.d`` (see ``Tier0Renderer.render``)."""
        ev = getattr(sim, "event", None)
        if after_value is None and ev is not None:
            after_value = ev.value
        names = ["geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat"] + (["light_xpos", "light_xdir"] if self.m.nlight else [])
        v = {k: wm.buffer_of(getattr(sim.d, k)) for k in names}
        self.rt.set_sources(sim, sources)
        cb = self.ctx.command_buffer()
        if ev is not None:
            self.ctx.wait_for(cb, ev, after_value)
        self.sync.encode(cb, sim if self.model_fields else None)
        if self.area is not None:
            self.area.encode(cb, sim)
        if self.rt.dyn is not None:      # tendons at their limits: MuJoCo's limit colour over the synced colour
            self.rt.dyn.encode_tint(cb, self.rt._dyn_src, self.color_buf, self.sync.inst_col)
        self._encode_env_mats(cb, sim if self.model_fields else None)
        if self.rolling is not None:      # row bands at interpolated poses (metalsim.render.rolling)
            cur = {k: (x.buffer, x.offset) for k, x in v.items()}
            if self.rt.pose is not None:
                for k in ("site_xpos", "site_xmat"):
                    x = wm.buffer_of(getattr(sim.d, k)); cur[k] = (x.buffer, x.offset)
            self._encode_rolling(cb, cur, sim, reset, passes)
        else:
            gm_buf, gm_off = self.rt.encode_geom_xmat(cb, sim, v["geom_xmat"].buffer, v["geom_xmat"].offset)
            gx_buf, gx_off, gm_buf, gm_off = self.rt.encode_pose(cb, sim, v["geom_xpos"].buffer, v["geom_xpos"].offset, gm_buf, gm_off)
            self._encode_scene(cb, gx_buf, gx_off, gm_buf, gm_off)
            lx = v.get("light_xpos"); ld = v.get("light_xdir")
            lxb, lxo, ldb, ldo = (lx.buffer, lx.offset, ld.buffer, ld.offset) if lx else (self._host_light_xpos, 0, self._host_light_xdir, 0)
            for p in range(passes):
                self._encode(cb, reset and p == 0, v["cam_xpos"].buffer, v["cam_xpos"].offset, v["cam_xmat"].buffer, v["cam_xmat"].offset, lxb, lxo, ldb, ldo)
        cb = self._postprocess(cb)
        val = self.ctx.signal(cb)
        self.ctx.commit(cb)
        return val

    def _encode_rolling(self, cb, cur, sim, reset, passes):
        """Rolling shutter: per row band, the pose inputs at the band's time, the structure refit to them and only the
        band's rows traced; the history takes the current poses at the end."""
        for r0, r1, tau in self.rolling.bands(self.th):
            b = self.history.encode_interp(cb, cur, tau)
            gm_buf, gm_off = self.rt.encode_geom_xmat(cb, sim, *b["geom_xmat"])
            sites = (b["site_xpos"], b["site_xmat"]) if self.rt.pose is not None else None
            gx_buf, gx_off, gm_buf, gm_off = self.rt.encode_pose(cb, sim, *b["geom_xpos"], gm_buf, gm_off, sites=sites)
            self._encode_scene(cb, gx_buf, gx_off, gm_buf, gm_off)
            lx, ld = b.get("light_xpos"), b.get("light_xdir")
            lxb, ldb = (lx[0], ld[0]) if lx else (self._host_light_xpos, self._host_light_xdir)
            for p in range(passes):
                self._encode(cb, reset and p == 0, b["cam_xpos"][0], 0, b["cam_xmat"][0], 0, lxb, 0, ldb, 0, rows=(r0, r1))
        self.history.encode_snapshot(cb, cur)

    def _render_host_rolling(self, datas, prev, passes, reset, sources):
        c = self.ctx
        n = self.n
        arr = lambda ds: {k: np.asarray(np.stack([getattr(d, k) for d in ds]), np.float32).reshape(n, -1)   # noqa: E731
                          for k in self.history.fields}
        if prev is not None:
            self.history.write_prev_host(arr(prev))
        cur_np = arr(datas)
        if not hasattr(self, "_host_rolling_bufs"):
            self._host_rolling_bufs = {k: c.buffer(x.nbytes, label=f"pt_host_roll_{k}") for k, x in cur_np.items()}
        for k, x in cur_np.items():
            c.write_buffer(self._host_rolling_bufs[k], x)
        self.rt.set_sources(sources=sources, datas=datas)
        cb = c.command_buffer()
        self.sync.encode(cb, None)
        if self.area is not None:
            self.area.write_host(datas)
        if self.rt.dyn is not None:
            self.rt.dyn.encode_tint(cb, self.rt._dyn_src, self.color_buf, self.sync.inst_col)
        self._encode_env_mats(cb, None)
        self._encode_rolling(cb, {k: (bf, 0) for k, bf in self._host_rolling_bufs.items()}, None, reset, passes)
        cb = self._postprocess(cb)
        c.commit(cb); c.synchronize()
        out = {k: a.numpy().copy() for k, a in self._arrays.items()}
        return out

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

    # -- environment map and per-episode scene edits --------------------------------------------------

    def _set_env_resources(self, rgba, cdf):
        c = self.ctx
        self._env_tex = c.texture2d(rgba.shape[1], rgba.shape[0], Metal.MTLPixelFormatRGBA32Float, Metal.MTLTextureUsageShaderRead,
                                    label="pt_envmap", storage=Metal.MTLStorageModeShared)
        c.upload_texture(self._env_tex, np.ascontiguousarray(rgba, np.float32))
        self._env_cdf = c.buffer(cdf.nbytes, np.ascontiguousarray(cdf, np.float32), "pt_env_cdf")

    def set_environment(self, hdr=None, *, yaw=0.0, intensity=None, exposure=0.0, color=None, grid=(512, 256), key=None):
        """Equirectangular HDR environment ``hdr`` (H, W, 3) linear radiance: the miss radiance of every ray and
        an importance-sampled light (a textured USD DomeLight). Orientation as Kit/RTX renders a DomeLight on a
        z-up stage: pole +z, the image centre (u = 0.5) faces -y, u = 0.25 faces +x; ``yaw`` rotates the dome
        about +z (rad, a rotateZ on the prim). Radiance = map x ``intensity`` x 2^``exposure`` x ``color``
        (USD LightAPI); ``intensity`` / ``color`` default to the ``dome`` kwarg's (a DomeLight's intensity and
        colour apply to its texture) or 1. ``key`` caches the GPU texture and sampling table per map, so a
        randomizer can switch maps per episode without re-uploading (``hdr`` may then be None for a cached key).
        ``hdr=None`` without a key restores the sky colour."""
        if hdr is None and key is None:
            self.env_flags = 0
            return
        if key is not None and key in self._env_res:
            self._env_tex, self._env_cdf, gw, gh = self._env_res[key]
        else:
            if hdr is None:
                raise KeyError(f"environment map {key!r} is not cached; pass hdr")
            hdr = np.asarray(hdr, np.float32)[..., :3]
            rgba = np.concatenate([hdr, np.ones(hdr.shape[:2] + (1,), np.float32)], -1)
            cdf, gw, gh = env_sampling_table(hdr, grid)
            self._set_env_resources(rgba, cdf)
            if key is not None:
                self._env_res[key] = (self._env_tex, self._env_cdf, gw, gh)
        self.consts[16:18] = (gw, gh)
        self.env_flags = 32
        self.set_environment_pose(yaw, intensity, exposure, color)

    def set_environment_pose(self, yaw=0.0, intensity=None, exposure=0.0, color=None):
        """Dome rotation about +z (rad) and the USD light scale on the map: intensity x 2^exposure x color
        (``None``: the ``dome`` kwarg's intensity / colour, else 1)."""
        cf = self.consts.view(np.float32)
        cf[20] = float(yaw)
        if intensity is None:
            intensity = self._dome[0] if self._dome is not None else 1.0
        if color is None:
            color = self._dome[1] if self._dome is not None else (1.0, 1.0, 1.0)
        cf[24:27] = float(intensity) * 2.0 ** float(exposure) * np.asarray(color, np.float32)

    def set_environment_from_model(self, model, grid=(512, 256)):
        """Apply the textured USD DomeLight that ``metalsim.scene.usd_to_mjcf`` recorded in the model's custom
        text ``usd_dome`` (file, intensity, exposure, color, yaw); returns its record or None when the model has
        none (or the dome has no texture: then the ``dome`` kwarg already carries it)."""
        import json
        import mujoco
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "usd_dome")
        if tid < 0:
            return None
        adr, size = int(model.text_adr[tid]), int(model.text_size[tid])
        rec = json.loads(bytes(model.text_data[adr:adr + size]).split(b"\0")[0].decode())
        if not rec.get("file"):
            return rec
        from metalsim.render.hdr import load_hdr
        key = rec["file"]
        hdr = None if key in self._env_res else load_hdr(rec["file"])
        self.set_environment(hdr, yaw=float(rec.get("yaw", 0.0)) + KIT_DOME_AZIMUTH, intensity=float(rec.get("intensity", 1.0)),
                             exposure=float(rec.get("exposure", 0.0)), color=tuple(rec.get("color", (1, 1, 1))), grid=grid, key=key)
        return rec

    def set_firefly_clamp(self, clamp_luminance=0.0):
        """Opt-in per-path luminance cap (0 = off, unbiased). Biased: see the class docstring / DECISIONS.md."""
        self.consts.view(np.float32)[21] = float(clamp_luminance)

    def set_fovy(self, fovy_deg, env=None):
        """Change the vertical field of view (MuJoCo convention) for all envs or one (fx = fy)."""
        f = 0.5 * self.th / np.tan(np.radians(fovy_deg) / 2)
        sl = slice(None) if env is None else slice(env, env + 1)
        self.cam_spec[sl, 0:2] = f
        self.t_cam_spec.copy_(torch.as_tensor(self.cam_spec))
        torch.mps.synchronize()

    def set_materials(self, materials):
        """Replace the (G, 16) material table (layout: SceneTables.MATERIAL_LAYOUT) and rewrite the GPU buffer;
        the per-slot BRDF model ids of a non-legacy ``material_model`` are kept."""
        from metalsim.render.model_sync import unit_rgba
        mats = np.asarray(materials, np.float32).reshape(self.G, 16).copy()
        self.tables.materials[:] = mats.reshape(self.tables.materials.shape)
        self.ctx.write_buffer(self.rt.mat_buf, mats)            # the lidar's reflectance (column 15)
        if self.mat_model != "legacy":
            mats[:, 15] = self.ctx.buffer_array(self.mat_buf, np.float32, (self.G, 16))[:, 15]
        self.ctx.write_buffer(self.mat_buf, unit_rgba(mats))
        self.sync.set_static_rgba(mats)

    def render_host(self, datas, passes=1, reset=True, cam_pos=None, cam_quat=None, geom_size=None, chunk=None, sources=None, prev=None):
        """Host path from MjData lists (tests/tools); synchronizes; returns numpy outputs. ``geom_size`` (N, ngeom, 3):
        per-world sizes of the terrain slots (default the model's). ``chunk``: commit and wait every ``chunk`` passes
        (long offline renders: bounded command buffers; same result). ``sources``: as ``render`` (numpy arrays)."""
        if self.rolling is not None:      # ``prev``: the previous frame's MjData (default the last render's state)
            return self._render_host_rolling(datas, prev, passes, reset, sources)
        from metalsim.render.scene_tables import quats_to_mats
        n, ng, nc, nl = self.n, self.m.ngeom, max(self.m.ncam, 1), max(self.m.nlight, 1)
        c = self.ctx; rt = self.rt
        gx = np.stack([d.geom_xpos for d in datas]).astype(np.float32).reshape(n, ng, 3)
        gm = np.stack([d.geom_xmat for d in datas]).astype(np.float32).reshape(n, ng, 9)
        if rt.sized is not None:
            gm = rt.sized.scale_host(gm, geom_size).astype(np.float32)
        if rt.pose is not None:
            gx, gm = rt.pose.host(gx, gm, datas)
        c.write_buffer(rt._host_geom_xpos, gx)
        c.write_buffer(rt._host_geom_xmat, gm)
        rt.set_sources(sources=sources, datas=datas)
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
        if self.area is not None:
            self.area.write_host(datas)
        cb = c.command_buffer()
        self.sync.encode(cb, None)
        if rt.dyn is not None:
            rt.dyn.encode_tint(cb, rt._dyn_src, self.color_buf, self.sync.inst_col)
        self._encode_env_mats(cb, None)
        self._encode_scene(cb, rt._host_geom_xpos, 0, rt._host_geom_xmat, 0)
        for p in range(passes):
            if chunk and p and p % int(chunk) == 0:
                c.commit(cb); c.synchronize(); cb = c.command_buffer()
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
