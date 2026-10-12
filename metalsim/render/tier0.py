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

from metalsim.interop import torch_bridge as tb
from metalsim.interop import warp_metal as wm
from metalsim.render.metal_context import MetalContext
from metalsim.render.scene_tables import CameraIntrinsics, SceneTables, build_scene_tables, quats_to_mats

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
                 terrain_slots: bool = True, draw_hfields: bool = True, hfield_cull: bool = True, hfield_tile: int = 16,
                 appearance: str = "mujoco", skybox: bool = True, shadow_lights: str = "all", max_lights: int | None = None,
                 transparency: bool = True, transparent_ids: str = "surface",
                 visual=None, deformables=(), tube_slices=None, cap_stacks=None,
                 far: float = 10.0, near: float = 0.01, reverse_z: bool = True, infinite_planes: bool = True,
                 hfield_skirts: bool = True, per_world_size: bool = True, model_fields: bool = True,
                 ctx: MetalContext | None = None, device="metal:0"):
        """tier 0: raster + shadow maps. tier 1: raster G-buffer with ray-traced shadows, ambient
        occlusion and mirror reflections from the fragment stage (Metal hardware ray tracing).

        ``terrain_slots``: draw the rough terrain's per-world box slots at their per-world ``geom_size``
        (``scene_tables.per_world_size_geoms``; False = the previous behaviour, the slots baked at their compiled
        2 mm size). ``draw_hfields``: heightfields as their exact surface (False = not drawn, the previous behaviour).
        ``hfield_cull``: heightfields drawn as ``hfield_tile`` x ``hfield_tile``-cell tiles culled per env on the GPU
        against the env's view frustum (and, for the shadow pass, the frustum swept toward the shadow caster); False =
        every triangle once per env (the full rasterization path).

        ``appearance``: "mujoco" (default) draws geometry, normals and textures as MuJoCo's renderer does
        (``scene_tables.build_scene_tables``); "legacy" = the tables before 2026-10. ``skybox``: draw the model's skybox
        texture as the background (MuJoCo's cube mapping) where no background image is set; ``set_clear_color`` turns it
        off.

        ``shadow_lights``: "all" (default): every model light with ``castshadow`` casts shadows; tier 0 renders a shadow
        map per extra light (directional: orthographic, spot: the cone, point: six cube faces) besides the caster's,
        tier 1 traces a shadow ray to each. "caster" = the previous behaviour (only the domain-randomization light or the
        first shadow-casting directional light). Heightfields drawn with ``hfield_cull`` cast only the caster's shadow
        in tier 0 (tier 1 traces them). ``max_lights``: lights shaded per fragment; None (default) = all model lights,
        as Isaac / RTX (no light limit); 8 = MuJoCo's effective cap (fixed-function OpenGL lights GL_LIGHT0..7).

        ``transparency``: geoms whose (material-resolved) rgba alpha is below 0.995 are blended over what is behind
        them as MuJoCo draws them (sorted per env by distance, front-to-back then back-to-front, back faces culled, no
        depth write; see tier0.metal sort_transparent); False = drawn opaque (the previous behaviour). They still cast
        full shadows (as in MuJoCo). ``transparent_ids`` (annotations of a pixel covered by a transparent geom):
        "surface" (default; "mujoco" is an alias): segmentation, depth and normal of the transparent surface, as
        ``mujoco.Renderer``'s segmentation and depth images (measured: depth on the glass) and Isaac's depth on glass;
        "behind": all three from what is behind it (Isaac Sim's RTX Real-Time drops partially opaque objects from depth
        and segmentation). The colour pass itself writes no depth on the first (front-to-back) blend, as MuJoCo.

        ``visual``: the non-geom objects MuJoCo's own renderer draws, under this ``mujoco.MjvOption`` (None: MuJoCo's
        default option, i.e. what ``mujoco.Renderer`` shows; False: geoms only): sites in the visible site groups,
        flexes (cloth / soft volumes as their offset skin, cables as edge cylinders), skins and spatial tendons
        (``metalsim.render.extras``). ``deformables``: user ``extras.DeformableMesh`` objects (XPBD / PhysX-port cloth
        or any (N, V, 3) vertex array; pass the arrays as ``render(..., sources={name: array})``). The deforming ones are
        regenerated on the GPU each frame (``metalsim.render.dynamic_mesh``); ``tube_slices`` / ``cap_stacks`` set the
        cylinder / capsule tessellation (default MuJoCo's ``vis.quality``). Segmentation ids of the extra objects in
        geom mode: ``ngeom + 1 + j`` (``tables.extras[j]["seg_id"]``).

        ``far``: camera far plane in metres; geometry beyond renders as background with depth 0 (MuJoCo's own far
        plane is ``vis.map.zfar * stat.extent``; Isaac Lab's ``PinholeCameraCfg.clipping_range`` far default is 1e6).
        ``near``: near plane (m). ``reverse_z``: reversed-z depth buffer (float32 depth error ~1e-7 x distance at any
        ``far``; False = the previous convention, error ~6e-8 x distance^2 / near).
        ``infinite_planes``: MuJoCo's infinite planes (a size of 0) drawn to the far plane under each env's camera with
        MuJoCo's plane texture mapping (False = the previous 10 x 10 m square). ``hfield_skirts``: heightfield sides
        and base as MuJoCo draws them (False = top surface only). ``per_world_size``: every primitive drawn at the
        per-world ``geom_size`` of MuJoCo Warp's model in the GPU path (False = only the terrain slots).
        ``model_fields``: per-world ``geom_rgba`` / ``mat_rgba`` / ``geom_matid`` colours and ``light_*`` fields of
        MuJoCo Warp's model read each frame in the GPU path (``metalsim.render.model_sync``; False = the compiled
        model's)."""
        self.m = model
        self.tier = tier
        self.rt_samples = rt_samples
        self.shadows = shadows and tier == 0
        if shadow_lights not in ("all", "caster"):
            raise ValueError(f"shadow_lights must be 'all' or 'caster', got {shadow_lights!r}")
        self.shadow_lights = shadow_lights
        self.max_lights = max_lights
        self._skybox_req = skybox
        if transparent_ids not in ("mujoco", "behind", "surface"):
            raise ValueError(f"transparent_ids must be 'surface', 'behind' or 'mujoco', got {transparent_ids!r}")
        self.transparent_ids = "surface" if transparent_ids == "mujoco" else transparent_ids
        self.transparency = transparency
        self.rt = None
        self.n = n_envs
        self.tw, self.th = width, height
        self.tpr = int(np.ceil(np.sqrt(n_envs)))
        self.aw = self.tpr * width
        self.ah = int(np.ceil(n_envs / self.tpr)) * height
        self.ctx = ctx or MetalContext(device)
        self.device = device
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
        if not (far > near > 0):
            raise ValueError(f"need far > near > 0, got near={near}, far={far}")
        self.near, self.far, self.reverse_z = float(near), float(far), bool(reverse_z)
        self.model_fields = bool(model_fields)
        intr = CameraIntrinsics.from_model(model, cam_id, width, height)
        self.tables: SceneTables = build_scene_tables(model, n_envs, max_group=max_group, include_planes=include_planes,
                                                      decimate_faces=decimate_faces, sized_geoms="auto" if terrain_slots else (),
                                                      draw_hfields=draw_hfields, appearance=appearance,
                                                      visual=visual, deformables=deformables, hfield_skirts=hfield_skirts,
                                                      plane_ring=self.plane_disc(intr, far) if infinite_planes else None,
                                                      per_world_size=per_world_size)
        self.G = self.tables.G
        self._extras_kw = dict(visual=mujoco.MjvOption() if visual is None else visual, deformables=deformables,
                               tube_slices=tube_slices, cap_stacks=cap_stacks)   # the camera default: everything MuJoCo draws
        hf_tris = sum(self.tables.meshes[k]["i_count"] // 3 for g, k in zip(self.tables.geoms, self.tables.geom_mesh)
                      if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_HFIELD))
        self.hfield_cull, self.hfield_tile = bool(hfield_cull and hf_tris), int(hfield_tile)
        if hf_tris * n_envs > 2e8 and not self.hfield_cull:
            # the raster draws every triangle once per env: Isaac's rough terrain (4.8 M) at 1024 envs is 2.9 s per frame
            # (runs/terrain_render/cost.log); the ray tracer (tier 2) is unaffected (158 ms)
            import warnings
            warnings.warn(f"tier 0 draws the heightfield in full per env: {hf_tris * n_envs / 1e6:.0f} M triangles per frame "
                          f"(~{hf_tris * n_envs / 1.7e9:.1f} s at 1024 envs on the M4 Max); hfield_cull=True culls it per env")
        self.seg_mode = seg_mode
        self.use_bg = backgrounds
        self.bg_layers = min(n_envs, 2048)
        self.cam_id = cam_id
        self.intrinsics = CameraIntrinsics.from_model(model, cam_id, width, height, near=self.near, far=self.far)
        self._build_resources()
        self._build_hfield_tiles()
        self._build_pipelines()
        self._alloc_outputs(outputs)
        self._host_sim = None
        self.sized = None
        modes = self.tables.geom_size_modes(model.ngeom)
        if modes.any():
            from metalsim.render.sized import SizedGeoms
            self.sized = SizedGeoms(self.ctx, model, n_envs, modes=modes)
        if tier >= 1:
            from metalsim.sensors.raytrace import RayTracer
            self.rt = RayTracer(model, n_envs, max_range=50.0, max_group=max_group, include_planes=include_planes,
                                decimate_faces=decimate_faces, terrain_slots=terrain_slots, draw_hfields=draw_hfields,
                                appearance=appearance, infinite_planes=infinite_planes, hfield_skirts=hfield_skirts,
                                per_world_size=per_world_size, ctx=self.ctx, device=device, **self._extras_kw)
            ro = np.zeros(4, np.uint32); ro[0] = self.rt.tpr; ro.view(np.float32)[1] = self.rt.stride; ro[2] = rt_samples
            ro.view(np.float32)[3] = self.rt.ray_limit
            self.rt_offset_buf = self.ctx.buffer(16, ro, "rt_offset")
        # extra objects: pose rows of the site slots; vertices of the dynamic slots (shared with the ray tracer at tier 1,
        # whose records carry the env grid offset that dyn_vs subtracts)
        self.dyn, self.pose, self._dyn_src = None, None, None
        if self.tables.dyn:
            from metalsim.render.dynamic_mesh import DynamicMeshes
            self.dyn = self.rt.dyn if self.rt is not None else DynamicMeshes(self.ctx, model, self.tables, n_envs, tube_slices=tube_slices,
                                                                             cap_stacks=cap_stacks)
        if self.tables.site_ids:
            from metalsim.render.dynamic_mesh import PoseGather
            self.pose = PoseGather(self.ctx, model, self.tables, n_envs)

    @staticmethod
    def plane_disc(intr: CameraIntrinsics, far: float) -> float:
        """Apothem of the infinite-plane disc drawn under the camera: every point within z-depth ``far`` of the image
        lies within far x (corner-ray length per unit depth) horizontally; x 2 margin for intrinsics / pose DR."""
        kx = max(intr.cx, intr.width - intr.cx) / intr.fx
        ky = max(intr.cy, intr.height - intr.cy) / intr.fy
        return 2.0 * float(far) * float(np.sqrt(1.0 + kx * kx + ky * ky))

    # -- resources --------------------------------------------------------------------------------

    def _build_resources(self):
        c, t = self.ctx, self.tables
        self.vbuf = c.buffer(t.vertices.nbytes, t.vertices, "vertices")
        self.ibuf = c.buffer(t.indices.nbytes, t.indices, "indices")
        self.inst_buf = c.buffer(t.inst_table.nbytes, t.inst_table, "instances")
        self.slot_geom = c.buffer(4 * self.G, t.instanced_slot_pose(), "slot_geom")
        self.pose_rows = t.pose_rows(self.m)        # geom pose rows per env, then the drawn sites'
        self.mat_buf = c.buffer(t.materials.nbytes, t.materials, "materials")
        mext = t.material_ext if t.material_ext is not None else np.zeros((self.G, 8), np.float32)
        self.mat_ext_buf = c.buffer(mext.nbytes, np.ascontiguousarray(mext, np.float32), "material_ext")
        self.sem_buf = c.buffer(t.semantic.nbytes, t.semantic, "semantic")
        self.env_buf = c.buffer(self.n * (2 * 64 + 9 * 16), label="env_params")
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
        # lights and scene params (headlight, bounds); per-env light records and instance colours from the model sync
        self.n_light = self.m.nlight
        from metalsim.render.model_sync import ModelSync, unit_rgba
        self.sync = ModelSync(c, self.m, self.n, t, self.color_buf, t.lights)
        self.light_buf = self.sync.lights                    # (n_envs, max(nlight, 1)) records
        self.inst_col_buf = self.sync.inst_col               # t_colors x the model colour, per env and slot
        self.mat_buf = c.buffer(t.materials.nbytes, unit_rgba(t.materials), "materials")
        self.params_buf = c.buffer(t.params.nbytes, t.params, "scene_params")
        self.plane_buf = c.buffer(t.plane.nbytes, t.plane, "slot_plane")
        self.clip = np.array([self.near, self.far, 1.0 if self.reverse_z else 0.0, 0.0], np.float32)
        self.clip_buf = c.buffer(16, self.clip, "clip_params")
        # shadow atlas: one tile per env, twice the render tile, atlas capped at 4096
        st = min(2 * max(self.tw, self.th), max(16, 4096 // self.tpr))
        self.sw = self.sh = int(st)
        self.saw = self.tpr * self.sw
        self.sah = int(np.ceil(self.n / self.tpr)) * self.sh
        # consts
        self.skybox = bool(self._skybox_req and t.params.shape[0] > 6 and t.params[6, 1] > 0.5)
        self._build_shadow_views()
        self._build_transparency()
        flags = ((1 if self.use_bg else 0) | (2 if self.shadows else 0) | (4 if self.skybox else 0)
                 | (8 if self.shadows and self.n_views else 0))
        self.consts = np.array([self.n, self.pose_rows, self.G, self.tpr, self.tw, self.th, self.aw, self.ah,
                                max(self.m.ncam, 1), self.bg_layers, flags, max(self.n_light, 1),
                                self.sw, self.sh, self.saw, self.sah, self.n_views, 0, 0, 0], np.uint32)
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
        self._host_geom_xpos = c.buffer(self.n * self.pose_rows * 12, label="host_geom_xpos")
        self._host_geom_xmat = c.buffer(self.n * self.pose_rows * 36, label="host_geom_xmat")
        nl = max(self.n_light, 1)
        self._host_light_xpos = c.buffer(self.n * nl * 12, label="host_light_xpos")
        self._host_light_xdir = c.buffer(self.n * nl * 12, label="host_light_xdir")
        # default light state for the host path: model light positions/directions
        lp = np.zeros((self.n, nl, 3), np.float32); ld = np.zeros((self.n, nl, 3), np.float32); ld[:, :, 2] = -1
        for i in range(self.m.nlight):
            lp[:, i] = self.m.light_pos[i]; ld[:, i] = self.m.light_dir[i]
        c.write_buffer(self._host_light_xpos, lp); c.write_buffer(self._host_light_xdir, ld)

    def _build_transparency(self):
        """Transparent slots, their bounding radii (planes -1: sorted last), mesh ranges for vertex pulling."""
        c, t, m = self.ctx, self.tables, self.m
        tr = np.flatnonzero(t.transparent) if (self.transparency and t.transparent is not None) else np.zeros(0, int)
        self.n_trans = len(tr)
        if not self.n_trans:
            return
        rb = np.array([-1.0 if int(m.geom_type[t.geoms[s_]]) == int(mujoco.mjtGeom.mjGEOM_PLANE) else float(m.geom_rbound[t.geoms[s_]])
                       for s_ in tr], np.float32)
        self.trans_slot_buf = c.buffer(4 * self.n_trans, tr.astype(np.int32), "trans_slots")
        self.trans_rb_buf = c.buffer(4 * self.n_trans, rb, "trans_rbound")
        self.slot_mesh_buf = c.buffer(4 * self.G, np.asarray(t.geom_mesh, np.int32), "slot_mesh")
        mr = np.zeros((len(t.meshes), 4), np.uint32)
        for k, mesh in enumerate(t.meshes):
            mr[k, 0], mr[k, 1] = mesh["i_off"], mesh["i_count"]
        self.mesh_range_buf = c.buffer(mr.nbytes, mr, "mesh_ranges")
        self.trans_max_index = int(max(t.meshes[int(t.geom_mesh[s_])]["i_count"] for s_ in tr))
        self.trans_consts_buf = c.buffer(16, np.array([self.n_trans, self.trans_max_index, 0, 0], np.uint32), "trans_consts")
        self.trans_order_buf = c.buffer(4 * self.n * self.n_trans, label="trans_order")

    def _build_shadow_views(self):
        """Extra shadow views (tier0.metal build_shadow_views): one per shadow-casting model light other than the
        caster (the first shadow-casting directional light), six for a point light."""
        c, L = self.ctx, self.tables.lights
        nl = max(self.n_light, 1)
        light_sv = np.full((nl, 4), -1, np.int32)
        views = []
        if self.shadows and self.shadow_lights == "all":
            cap = self.n_light if self.max_lights is None else min(self.n_light, int(self.max_lights))
            first_dir = next((i for i in range(cap) if L[i, 0] == 0 and L[i, 1] > 0.5), -1)
            for i in range(cap):
                if L[i, 1] <= 0.5 or i == first_dir:
                    continue
                kind = int(L[i, 0])
                nv = 6 if kind == 1 else 1
                light_sv[i] = (len(views), nv, kind, 0)
                views += [(i, f if kind == 1 else -1, kind, 0) for f in range(nv)]
        self.n_views = len(views)
        vl = np.asarray(views if views else [(0, -1, 0, 0)], np.int32)
        self.view_light_buf = c.buffer(vl.nbytes, vl, "shadow_view_light")
        self.light_sv_buf = c.buffer(light_sv.nbytes, light_sv, "light_shadow_views")
        self.sviews_buf = c.buffer(max(self.n * self.n_views, 1) * 80, label="shadow_views")
        d = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            Metal.MTLPixelFormatDepth32Float, self.saw if self.n_views else 1, self.sah if self.n_views else 1, False)
        d.setTextureType_(Metal.MTLTextureType2DArray)
        d.setArrayLength_(max(self.n_views, 1))
        d.setUsage_(Metal.MTLTextureUsageRenderTarget | Metal.MTLTextureUsageShaderRead)
        d.setStorageMode_(Metal.MTLStorageModePrivate)
        self.smaps_tex = c.device.newTextureWithDescriptor_(d)
        self.smaps_tex.setLabel_("extra_shadow_maps")

    SKIRT_CHUNK = 96          # indices (32 triangles) per culled heightfield-side chunk

    def _build_hfield_tiles(self):
        """Tile tables for GPU culling of heightfields (see hf_cull in tier0.metal)."""
        t, m, T = self.tables, self.m, self.hfield_tile
        hf_mesh = {k for k, mesh in enumerate(t.meshes) if "hfield" in mesh}
        src = t.draws_opaque if (self.n_trans and t.draws_opaque is not None) else t.draws
        self._draws = [d for k, d in enumerate(src) if not (self.hfield_cull and k in hf_mesh) and d[3] > 0]
        self._shadow_draws = [d for k, d in enumerate(t.draws) if not (self.hfield_cull and k in hf_mesh)]   # all geoms cast
        self.n_skirt = 0
        if self.hfield_cull:            # the tiles cover the top surface; the sides / base are culled per env in chunks
            CH = self.SKIRT_CHUNK
            chunks, cboxes = [], []
            for slot, (g, k) in enumerate(zip(t.geoms, t.geom_mesh)):
                if k not in hf_mesh:
                    continue
                mesh = t.meshes[k]
                top = mesh.get("top_count", mesh["i_count"])
                a, b = mesh["i_off"] + top, mesh["i_off"] + mesh["i_count"]
                for s0 in range(a, b, CH):
                    cnt = min(CH, b - s0)
                    P = t.vertices[t.indices[s0:s0 + cnt], :3].astype(np.float64)
                    chunks.append([slot, s0, cnt, 0])
                    cboxes.append(list(P.min(0) - 1e-4) + [0] + list(P.max(0) + 1e-4) + [0])
            self.n_skirt = len(chunks)
            if self.n_skirt:        # (heightfield sides cast no tier-0 shadow on the culled path)
                c = self.ctx
                self.skirt_chunks = c.buffer(16 * self.n_skirt, np.asarray(chunks, np.int32), "hf_skirt_chunks")
                self.skirt_boxes = c.buffer(32 * self.n_skirt, np.asarray(cboxes, np.float32), "hf_skirt_boxes")
                self.skirt_vis = c.buffer(8 * self.n * self.n_skirt, label="hf_skirt_visible")
                self.skirt_args = c.buffer(16, np.zeros(4, np.uint32), "hf_skirt_args")
        self.n_tiles = 0
        if not self.hfield_cull:
            return
        tiles, boxes, fields = [], [], []
        for slot, (g, k) in enumerate(zip(t.geoms, t.geom_mesh)):
            if k not in hf_mesh:
                continue
            hid = t.meshes[k]["hfield"]; vbase = t.meshes[k]["v_off"]
            nrow, ncol = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
            sx, sy, sz, _ = (float(v) for v in m.hfield_size[hid])
            Z = m.hfield_data[int(m.hfield_adr[hid]):int(m.hfield_adr[hid]) + nrow * ncol].astype(np.float64).reshape(nrow, ncol) * sz
            xs = np.linspace(-sx, sx, ncol); ys = np.linspace(-sy, sy, nrow)
            fi = len(fields)
            fields.append([-sx, -sy, Z.min(), 0, sx, sy, Z.max(), 0])
            for r0 in range(0, nrow - 1, T):
                for c0 in range(0, ncol - 1, T):
                    r1, c1 = min(r0 + T, nrow - 1), min(c0 + T, ncol - 1)
                    blk = Z[r0:r1 + 1, c0:c1 + 1]
                    tiles.append([slot, vbase, nrow, ncol, r0, c0, fi, 0])
                    boxes.append([xs[c0], ys[r0], blk.min(), 0, xs[c1], ys[r1], blk.max(), 0])
        c = self.ctx
        # float32 boxes widened by one ulp-scale margin so the float32 vertices never poke out of their tile's box
        B = np.asarray(boxes, np.float64); F = np.asarray(fields, np.float64)
        B[:, :3] -= 1e-4; B[:, 4:7] += 1e-4
        self.n_tiles = len(tiles)
        self.hf_tiles = c.buffer(len(tiles) * 32, np.asarray(tiles, np.int32), "hf_tiles")
        self.hf_boxes = c.buffer(len(tiles) * 32, B.astype(np.float32), "hf_boxes")
        self.hf_fields = c.buffer(len(fields) * 32, F.astype(np.float32), "hf_fields")
        cap = self.n * self.n_tiles
        self.hf_vis = c.buffer(cap * 8, label="hf_visible")
        self.hf_svis = c.buffer(cap * 8, label="hf_shadow_visible")
        self.hf_args = c.buffer(64, np.zeros(16, np.uint32), "hf_draw_args")
        r = float(t.params[0, 3])
        cc = np.zeros(8, np.uint32); cc[:4] = (self.n, self.n_tiles, T, 1 if self.shadows else 0)
        cc.view(np.float32)[4] = 2.0 * (2.0 * r / self.sw)       # two shadow texels (hardware 2x2 filtered compare)
        self.hf_cc = c.buffer(32, cc, "hf_cull_consts")
        li = []
        for i in range(T):
            for j in range(T):
                a, b, cc_, d = i * (T + 1) + j, i * (T + 1) + j + 1, (i + 1) * (T + 1) + j + 1, (i + 1) * (T + 1) + j
                li += [a, b, cc_, a, cc_, d]                # MuJoCo's cell split, as hfield_mesh
        li = np.asarray(li, np.uint32)
        self.hf_ibuf = c.buffer(li.nbytes, li, "hf_tile_indices")

    def hfield_visible_counts(self) -> tuple:
        """(render, shadow) visible (env, tile) pairs of the last frame (synchronizes)."""
        self.ctx.synchronize()
        a = self.ctx.buffer_array(self.hf_args, np.uint32, (16,))
        return int(a[1]), int(a[9])

    def _build_pipelines(self):
        c = self.ctx
        lib = c.library("tier0", None if self.max_lights is None else {"MAX_LIGHTS": int(self.max_lights)})
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

        depth_cmp = Metal.MTLCompareFunctionGreater if self.reverse_z else Metal.MTLCompareFunctionLess
        self.geo_pipe, self.geo_ds = make("geom_vs", "geom_fs_rt" if self.tier >= 1 else "geom_fs", True, depth_cmp, True)
        self.bg_pipe, self.bg_ds = make("bg_vs", "bg_fs", False, Metal.MTLCompareFunctionAlways, False)
        d = Metal.MTLRenderPipelineDescriptor.new()
        d.setVertexFunction_(lib.newFunctionWithName_("shadow_vs"))
        d.setVertexDescriptor_(vd)
        d.setDepthAttachmentPixelFormat_(Metal.MTLPixelFormatDepth32Float)
        self.shadow_pipe = c.render_pipeline(d)
        ds = Metal.MTLDepthStencilDescriptor.new()          # the shadow map keeps the standard depth convention
        ds.setDepthWriteEnabled_(True); ds.setDepthCompareFunction_(Metal.MTLCompareFunctionLess)
        self.shadow_ds = c.device.newDepthStencilStateWithDescriptor_(ds)
        if self.n_trans:
            self.p_sort_trans = c.compute_pipeline(lib, "sort_transparent")
            d = Metal.MTLRenderPipelineDescriptor.new()
            d.setVertexFunction_(lib.newFunctionWithName_("trans_vs"))
            d.setFragmentFunction_(lib.newFunctionWithName_("geom_fs_rt" if self.tier >= 1 else "geom_fs"))
            ca = d.colorAttachments()
            a0 = ca.objectAtIndexedSubscript_(0)
            a0.setPixelFormat_(Metal.MTLPixelFormatRGBA8Unorm)
            a0.setBlendingEnabled_(True)
            a0.setRgbBlendOperation_(Metal.MTLBlendOperationAdd); a0.setAlphaBlendOperation_(Metal.MTLBlendOperationAdd)
            a0.setSourceRGBBlendFactor_(Metal.MTLBlendFactorSourceAlpha)
            a0.setDestinationRGBBlendFactor_(Metal.MTLBlendFactorOneMinusSourceAlpha)
            a0.setSourceAlphaBlendFactor_(Metal.MTLBlendFactorZero); a0.setDestinationAlphaBlendFactor_(Metal.MTLBlendFactorOne)
            ids = self.transparent_ids != "behind"
            for k, fmt in ((1, Metal.MTLPixelFormatR32Uint), (2, Metal.MTLPixelFormatRGBA16Float)):
                ak = ca.objectAtIndexedSubscript_(k)
                ak.setPixelFormat_(fmt)
                ak.setWriteMask_(Metal.MTLColorWriteMaskAll if ids else Metal.MTLColorWriteMaskNone)
            d.setDepthAttachmentPixelFormat_(Metal.MTLPixelFormatDepth32Float)
            self.trans_pipe = c.render_pipeline(d)
            dss = []
            for write in (False, self.transparent_ids == "surface"):     # pass 1 never writes depth; pass 2 for "surface"
                ds = Metal.MTLDepthStencilDescriptor.new()
                ds.setDepthWriteEnabled_(write)
                ds.setDepthCompareFunction_(depth_cmp)          # reversed z: greater
                dss.append(c.device.newDepthStencilStateWithDescriptor_(ds))
            self.trans_ds = dss
        if self.n_views:
            d = Metal.MTLRenderPipelineDescriptor.new()
            d.setVertexFunction_(lib.newFunctionWithName_("shadow_view_vs"))
            d.setFragmentFunction_(lib.newFunctionWithName_("shadow_view_fs"))
            d.setVertexDescriptor_(vd)
            d.setDepthAttachmentPixelFormat_(Metal.MTLPixelFormatDepth32Float)
            self.sview_pipe = c.render_pipeline(d)
            self.p_sviews = c.compute_pipeline(lib, "build_shadow_views")
            if self.tables.dyn:
                d = Metal.MTLRenderPipelineDescriptor.new()
                d.setVertexFunction_(lib.newFunctionWithName_("dyn_shadow_view_vs"))
                d.setFragmentFunction_(lib.newFunctionWithName_("shadow_view_fs"))
                d.setDepthAttachmentPixelFormat_(Metal.MTLPixelFormatDepth32Float)
                self.dyn_sview_pipe = c.render_pipeline(d)
        if self.tables.dyn:
            self.dyn_pipe, _ = make("dyn_vs", "geom_fs_rt" if self.tier >= 1 else "geom_fs", True, depth_cmp, False)
            d = Metal.MTLRenderPipelineDescriptor.new()
            d.setVertexFunction_(lib.newFunctionWithName_("dyn_shadow_vs"))
            d.setDepthAttachmentPixelFormat_(Metal.MTLPixelFormatDepth32Float)
            self.dyn_shadow_pipe = c.render_pipeline(d)
        if self.hfield_cull:
            self.hf_pipe, _ = make("hf_tile_vs", "geom_fs_rt" if self.tier >= 1 else "geom_fs", True, depth_cmp, False)
            d = Metal.MTLRenderPipelineDescriptor.new()
            d.setVertexFunction_(lib.newFunctionWithName_("hf_tile_shadow_vs"))
            d.setDepthAttachmentPixelFormat_(Metal.MTLPixelFormatDepth32Float)
            self.hf_shadow_pipe = c.render_pipeline(d)
            self.p_hf_reset = c.compute_pipeline(lib, "hf_cull_reset")
            self.p_hf_cull = c.compute_pipeline(lib, "hf_cull")
            if self.n_skirt:
                self.skirt_pipe, _ = make("hf_skirt_vs", "geom_fs_rt" if self.tier >= 1 else "geom_fs", True, depth_cmp, False)
                self.p_skirt_reset = c.compute_pipeline(lib, "hf_skirt_reset")
                self.p_skirt_cull = c.compute_pipeline(lib, "hf_skirt_cull")

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
        """Per-env background colour; replaces the skybox (as a background colour randomization expects)."""
        self.cam_spec[:, 20:23] = np.asarray(rgb, np.float32)
        self.consts[10] &= ~np.uint32(4)
        self.ctx.write_buffer(self.consts_buf, self.consts)
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

    def set_materials(self, materials):
        """Replace the (G, 16) material table (layout: SceneTables.MATERIAL_LAYOUT; rgba is the slot's compiled colour,
        which the model sync supplies per world) and rewrite the GPU buffer."""
        mats = np.asarray(materials, np.float32).reshape(self.G, 16)
        self.tables.materials[:] = mats.reshape(self.tables.materials.shape)
        from metalsim.render.model_sync import unit_rgba
        self.ctx.write_buffer(self.mat_buf, unit_rgba(mats))
        self.sync.set_static_rgba(mats)

    def set_backgrounds(self, env_ids, images):
        self.use_bg = True
        self.consts[10] |= 1                 # keep the shadow / skybox bits
        self.ctx.write_buffer(self.consts_buf, self.consts)
        self.cam_spec[:, 19] = 1.0
        self._sync_cam_spec()
        for eid, im in zip(env_ids, images):
            rgba = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
            self.ctx.upload_texture(self.bg_tex, rgba, int(eid) % self.bg_layers)

    # -- rendering ----------------------------------------------------------------------------------

    def _encode(self, cb, geom_xpos, geom_xpos_off, geom_xmat, geom_xmat_off, cam_xpos, cam_xpos_off, cam_xmat, cam_xmat_off,
                light_xpos=None, light_xpos_off=0, light_xdir=None, light_xdir_off=0, sim=None):
        c = self.ctx
        # 0. per-env light records and instance colours from the model (MuJoCo Warp's per-world fields with a sim)
        self.sync.encode(cb, sim if self.model_fields else None)
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
        ce.setBuffer_offset_atIndex_(self.clip_buf, 0, 8)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n, 1, 1), Metal.MTLSize(min(self.n, 64), 1, 1))
        ce.endEncoding()
        # 1-. dynamic slots' vertices (tier 1: generated with the ray tracer's structures in step 1c)
        if self.dyn is not None and self.rt is None:
            if self._dyn_src is None:
                raise ValueError("dynamic objects need their sources (render / render_host bind them)")
            self.dyn.encode(cb, self._dyn_src)
        # 1a'. transparent geoms: per-env order by camera distance (sort_transparent)
        if self.n_trans:
            ce = cb.computeCommandEncoder()
            ce.setComputePipelineState_(self.p_sort_trans)
            for i, (b, o) in enumerate(((self.env_buf, 0), (self.trans_slot_buf, 0), (self.trans_rb_buf, 0), (self.slot_geom, 0),
                                        (geom_xpos, geom_xpos_off), (self.consts_buf, 0), (self.trans_consts_buf, 0),
                                        (self.trans_order_buf, 0))):
                ce.setBuffer_offset_atIndex_(b, o, i)
            ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n, 1, 1), Metal.MTLSize(min(self.n, 64), 1, 1))
            ce.endEncoding()
        # 1a. heightfield tiles: per-env visible lists and indirect draw arguments (serial dispatches)
        if self.hfield_cull:
            ce = cb.computeCommandEncoder()
            ce.setComputePipelineState_(self.p_hf_reset)
            ce.setBuffer_offset_atIndex_(self.hf_args, 0, 0); ce.setBuffer_offset_atIndex_(self.hf_cc, 0, 1)
            ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(2, 1, 1), Metal.MTLSize(2, 1, 1))
            ce.setComputePipelineState_(self.p_hf_cull)
            for i, (b, o) in enumerate(((self.env_buf, 0), (self.hf_tiles, 0), (self.hf_boxes, 0), (self.hf_fields, 0),
                                        (self.slot_geom, 0), (geom_xpos, geom_xpos_off), (geom_xmat, geom_xmat_off),
                                        (self.consts_buf, 0), (self.hf_cc, 0), (self.hf_vis, 0), (self.hf_svis, 0), (self.hf_args, 0))):
                ce.setBuffer_offset_atIndex_(b, o, i)
            ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n * self.n_tiles, 1, 1), Metal.MTLSize(256, 1, 1))
            if self.n_skirt:
                ce.setComputePipelineState_(self.p_skirt_reset)
                ce.setBuffer_offset_atIndex_(self.skirt_args, 0, 0)
                ce.setBytes_length_atIndex_(np.array([self.SKIRT_CHUNK], np.uint32).tobytes(), 4, 1)
                ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(1, 1, 1), Metal.MTLSize(1, 1, 1))
                ce.setComputePipelineState_(self.p_skirt_cull)
                for i, (b, o) in enumerate(((self.env_buf, 0), (self.skirt_chunks, 0), (self.skirt_boxes, 0), (self.slot_geom, 0),
                                            (geom_xpos, geom_xpos_off), (geom_xmat, geom_xmat_off), (self.consts_buf, 0))):
                    ce.setBuffer_offset_atIndex_(b, o, i)
                ce.setBytes_length_atIndex_(np.array([self.n_skirt], np.uint32).tobytes(), 4, 7)
                ce.setBuffer_offset_atIndex_(self.skirt_vis, 0, 8); ce.setBuffer_offset_atIndex_(self.skirt_args, 0, 9)
                ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n * self.n_skirt, 1, 1), Metal.MTLSize(256, 1, 1))
            ce.endEncoding()
        grains = getattr(self, "_grain_raster", None)     # particles (set_particles, metalsim/render/particles.py)
        if grains is not None:
            grains.encode_instances(cb)
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
            se.setVertexBuffer_offset_atIndex_(self.inst_col_buf, 0, 6)
            se.setVertexBuffer_offset_atIndex_(self.consts_buf, 0, 7)
            se.setVertexBuffer_offset_atIndex_(self.plane_buf, 0, 12)
            for i_off, i_count, first, count in self._shadow_draws:
                se.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance_(
                    Metal.MTLPrimitiveTypeTriangle, i_count, Metal.MTLIndexTypeUInt32, self.ibuf, i_off * 4, count, 0, first)
            if self.hfield_cull:
                se.setRenderPipelineState_(self.hf_shadow_pipe)
                self._bind_hf(se, self.hf_svis)
                se.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset_(
                    Metal.MTLPrimitiveTypeTriangle, Metal.MTLIndexTypeUInt32, self.hf_ibuf, 0, self.hf_args, 32)
            if self.dyn is not None:
                se.setRenderPipelineState_(self.dyn_shadow_pipe)
                self._draw_dyn(se)
            if grains is not None:
                grains.draw_shadow(se)
            se.endEncoding()
        # 1b'. extra shadow views (spot / point / further directional lights): matrices, then one depth pass each
        if self.shadows and self.n_views:
            ce = cb.computeCommandEncoder()
            ce.setComputePipelineState_(self.p_sviews)
            for i, (b, o) in enumerate(((self.sviews_buf, 0), (self.view_light_buf, 0), (self.light_buf, 0),
                                        (light_xpos, light_xpos_off), (light_xdir, light_xdir_off), (self.consts_buf, 0),
                                        (self.params_buf, 0))):
                ce.setBuffer_offset_atIndex_(b, o, i)
            ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n * self.n_views, 1, 1),
                                                      Metal.MTLSize(min(self.n * self.n_views, 64), 1, 1))
            ce.endEncoding()
            for v in range(self.n_views):
                sp = Metal.MTLRenderPassDescriptor.new()
                da = sp.depthAttachment()
                da.setTexture_(self.smaps_tex); da.setSlice_(v); da.setLoadAction_(Metal.MTLLoadActionClear)
                da.setStoreAction_(Metal.MTLStoreActionStore); da.setClearDepth_(1.0)
                se = cb.renderCommandEncoderWithDescriptor_(sp)
                se.setRenderPipelineState_(self.sview_pipe)
                se.setDepthStencilState_(self.shadow_ds)
                se.setCullMode_(Metal.MTLCullModeNone)
                se.setDepthBias_slopeScale_clamp_(2.0, 2.0, 0.02)
                for i, (b, o) in enumerate(((self.vbuf, 0), (self.env_buf, 0), (self.inst_buf, 0), (self.slot_geom, 0),
                                            (geom_xpos, geom_xpos_off), (geom_xmat, geom_xmat_off), (self.sviews_buf, 0),
                                            (self.consts_buf, 0))):
                    se.setVertexBuffer_offset_atIndex_(b, o, i)
                vb = np.array([v], np.uint32).tobytes()
                se.setVertexBytes_length_atIndex_(vb, 4, 8)
                se.setFragmentBuffer_offset_atIndex_(self.sviews_buf, 0, 6)
                se.setFragmentBuffer_offset_atIndex_(self.consts_buf, 0, 7)
                se.setFragmentBytes_length_atIndex_(vb, 4, 8)
                se.setFragmentBuffer_offset_atIndex_(light_xpos, light_xpos_off, 9)
                for i_off, i_count, first, count in self._shadow_draws:
                    se.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance_(
                        Metal.MTLPrimitiveTypeTriangle, i_count, Metal.MTLIndexTypeUInt32, self.ibuf, i_off * 4, count, 0, first)
                if self.dyn is not None:            # deformables, tendons and skins cast every light's shadow too
                    se.setRenderPipelineState_(self.dyn_sview_pipe)
                    self._draw_dyn(se)
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
        da.setClearDepth_(0.0 if self.reverse_z else 1.0)
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
        re.setFragmentBuffer_offset_atIndex_(self.params_buf, 0, 9)
        re.setFragmentTexture_atIndex_(self.atlas_tex, 0)
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
        re.setVertexBuffer_offset_atIndex_(self.inst_col_buf, 0, 6)
        re.setVertexBuffer_offset_atIndex_(self.consts_buf, 0, 7)
        re.setVertexBuffer_offset_atIndex_(self.plane_buf, 0, 12)
        re.setFragmentBuffer_offset_atIndex_(self.env_buf, 0, 1)
        re.setFragmentBuffer_offset_atIndex_(self.mat_buf, 0, 2)
        re.setFragmentBuffer_offset_atIndex_(self.sem_buf, 0, 3)
        re.setFragmentBuffer_offset_atIndex_(self.inst_col_buf, 0, 4)
        re.setFragmentBuffer_offset_atIndex_(self.consts_buf, 0, 5)
        re.setFragmentBuffer_offset_atIndex_(self.light_buf, 0, 6)
        re.setFragmentBuffer_offset_atIndex_(light_xpos, light_xpos_off, 7)
        re.setFragmentBuffer_offset_atIndex_(light_xdir, light_xdir_off, 8)
        re.setFragmentBuffer_offset_atIndex_(self.params_buf, 0, 9)
        re.setFragmentBuffer_offset_atIndex_(self.mat_ext_buf, 0, 18)
        re.setFragmentBuffer_offset_atIndex_(self.sviews_buf, 0, 19)
        re.setFragmentBuffer_offset_atIndex_(self.light_sv_buf, 0, 20)
        re.setFragmentTexture_atIndex_(self.smaps_tex, 3)
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
        for i_off, i_count, first, count in self._draws:
            re.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance_(
                Metal.MTLPrimitiveTypeTriangle, i_count, Metal.MTLIndexTypeUInt32, self.ibuf, i_off * 4, count, 0, first)
        if self.hfield_cull:
            re.setRenderPipelineState_(self.hf_pipe)
            self._bind_hf(re, self.hf_vis)
            re.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset_(
                Metal.MTLPrimitiveTypeTriangle, Metal.MTLIndexTypeUInt32, self.hf_ibuf, 0, self.hf_args, 0)
            if self.n_skirt:
                re.setRenderPipelineState_(self.skirt_pipe)
                re.setVertexBuffer_offset_atIndex_(self.skirt_vis, 0, 2)
                re.setVertexBuffer_offset_atIndex_(self.skirt_chunks, 0, 6)
                re.setVertexBuffer_offset_atIndex_(self.ibuf, 0, 8)
                re.drawPrimitives_indirectBuffer_indirectBufferOffset_(Metal.MTLPrimitiveTypeTriangle, self.skirt_args, 0)
        if self.dyn is not None:
            re.setRenderPipelineState_(self.dyn_pipe)
            self._draw_dyn(re)
        if grains is not None:
            grains.draw(re, light_xpos, light_xpos_off, light_xdir, light_xdir_off)
        if self.n_trans:          # transparent geoms last: front to back, then back to front (MuJoCo's mjr_render)
            if grains is not None:     # the grain pass rebinds fragment buffers the geom fragment shaders read
                re.setFragmentBuffer_offset_atIndex_(self.mat_buf, 0, 2)
                re.setFragmentBuffer_offset_atIndex_(self.sem_buf, 0, 3)
                re.setFragmentBuffer_offset_atIndex_(self.inst_col_buf, 0, 4)
                re.setFragmentTexture_atIndex_(self.atlas_tex, 0)
                re.setFragmentSamplerState_atIndex_(self.samp, 0)
                re.setFragmentSamplerState_atIndex_(self.samp_shadow, 1)
                if self.rt is not None:
                    re.setFragmentAccelerationStructure_atBufferIndex_(self.rt.inst_as, 10)
                    re.setFragmentBuffer_offset_atIndex_(self.rt_offset_buf, 0, 11)
            re.setRenderPipelineState_(self.trans_pipe)
            re.setFrontFacingWinding_(Metal.MTLWindingCounterClockwise)
            re.setCullMode_(Metal.MTLCullModeBack)
            for i, (b, o) in enumerate(((self.vbuf, 0), (self.env_buf, 0), (self.trans_order_buf, 0), (self.slot_geom, 0),
                                        (geom_xpos, geom_xpos_off), (geom_xmat, geom_xmat_off), (self.slot_mesh_buf, 0),
                                        (self.consts_buf, 0), (self.trans_consts_buf, 0), (self.mesh_range_buf, 0),
                                        (self.ibuf, 0))):
                re.setVertexBuffer_offset_atIndex_(b, o, i)
            for rev in (0, 1):
                re.setDepthStencilState_(self.trans_ds[rev])
                re.setVertexBytes_length_atIndex_(np.array([rev], np.uint32).tobytes(), 4, 11)
                re.drawPrimitives_vertexStart_vertexCount_instanceCount_(Metal.MTLPrimitiveTypeTriangle, 0, self.trans_max_index,
                                                                         self.n_trans * self.n)
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
        ue.setBuffer_offset_atIndex_(self.clip_buf, 0, 8)
        ue.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.tw, self.th, self.n), Metal.MTLSize(16, 16, 1))
        ue.endEncoding()

    def _draw_dyn(self, enc):
        """One draw per dynamic object (instance = env) from its generated records; buffer 1 (env params) stays bound."""
        dyn = self.dyn
        enc.setVertexBuffer_offset_atIndex_(dyn.vbuf, 0, 0)
        enc.setVertexBuffer_offset_atIndex_(self.env_buf, 0, 1)
        for i_off, i_count, rbase, nvert, slot in dyn.draws:
            if not i_count:
                continue
            dd = np.zeros(8, np.uint32)
            dd[:4] = (rbase, nvert, slot, dyn.tpr)
            dd.view(np.float32)[4] = dyn.stride
            enc.setVertexBytes_length_atIndex_(dd.tobytes(), dd.nbytes, 2)
            enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance_(
                Metal.MTLPrimitiveTypeTriangle, i_count, Metal.MTLIndexTypeUInt32, dyn.local_ibuf, i_off * 4, self.n, 0, 0)

    def _bind_hf(self, enc, vis):
        """Vertex buffers of the tile draw (1, 3, 4, 5, 7 stay bound from the preceding draws)."""
        enc.setVertexBuffer_offset_atIndex_(self.vbuf, 0, 0)
        enc.setVertexBuffer_offset_atIndex_(vis, 0, 2)
        enc.setVertexBuffer_offset_atIndex_(self.hf_tiles, 0, 6)
        enc.setVertexBuffer_offset_atIndex_(self.hf_cc, 0, 8)

    def set_particles(self, positions, radius=None, material=None, *, grains_per_particle=8, grain_radius=None,
                      radius_jitter=0.3, n_meshes=4, seed=0, grain_shape=None, ids=None, cull_z=float("-inf")):
        """Draw particles as render grains, as ``Tier2Renderer.set_particles`` (same arguments, same grains and
        per-grain colours; metalsim/render/particles.py, PARTICLES.md). Tier-0 shading; with ``tier=1`` the grains
        are lit without shadows. ``positions=None`` removes them."""
        from metalsim.render.particles import GrainField, GrainRaster
        if positions is None:
            self._grain_raster = None
            return
        gr = getattr(self, "_grain_raster", None)
        f = gr.field if gr is not None else None
        if f is None or (radius is not None and float(radius) != f.offset) or int(grains_per_particle) != f.k:
            if radius is None:
                raise ValueError("set_particles: radius is required the first time")
            g_r = grain_radius if grain_radius is not None else float(radius) / float(grains_per_particle) ** (1 / 3)
            f = GrainField(self.ctx, self.n, grains_per_particle=grains_per_particle, radius=g_r, radius_jitter=radius_jitter,
                           offset=float(radius), n_meshes=n_meshes, material=material, seed=seed, cull_z=cull_z, shape=grain_shape)
            self._grain_raster = GrainRaster(self, f) if gr is None else gr.rebind(f)
        elif material is not None:
            f.set_material(material)
        f.cull_z = float(cull_z)
        f.set_positions(positions)
        f.set_ids(ids)

    def _set_sources(self, sim=None, sources=None, datas=None):
        """Per-frame sources of the dynamic slots (see ``RayTracer.set_sources``)."""
        if self.rt is not None:
            self.rt.set_sources(sim, sources, datas)
        elif self.dyn is not None:
            self._dyn_src = self.dyn.write_host(datas, sources) if datas is not None else self.dyn.sources_from_sim(sim.d, sources)
        elif sources:
            raise ValueError("this scene has no dynamic objects")

    def render(self, sim, after_value: int | None = None, sources=None) -> int:
        """GPU path: render the current physics state of ``sim`` (a BatchSim). Waits for
        ``sim.event >= after_value`` (default: the latest value), signals ``self.ctx.event``;
        returns that value. Nothing synchronizes the host.

        ``sources``: per-frame vertex arrays of the dynamic objects that do not come from ``sim.d``: ``flexvert_xpos``
        from another flex simulator (``XPBDSim.x``) and one (N, V, 3) Warp array per user ``DeformableMesh``
        (``PhysXClothSim.x``). A ``sim`` without an ``event`` (``DeformableSim``, ``XPBDSim``) must be synchronized by
        the caller before the render."""
        ev = getattr(sim, "event", None)
        if after_value is None and ev is not None:
            after_value = ev.value
        names = ["geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat"] + (["light_xpos", "light_xdir"] if self.n_light else [])
        views = {name: wm.buffer_of(getattr(sim.d, name)) for name in names}
        self._set_sources(sim, sources)
        cb = self.ctx.command_buffer()
        if ev is not None:
            self.ctx.wait_for(cb, ev, after_value)
        lx = views.get("light_xpos"); ld = views.get("light_xdir")
        gm_buf, gm_off = views["geom_xmat"].buffer, views["geom_xmat"].offset
        if self.sized is not None:          # per-world slot sizes folded into a copy of geom_xmat
            from metalsim.render.model_sync import model_fields
            gm_buf, gm_off = self.sized.encode(cb, gm_buf, gm_off, getattr(model_fields(sim), "geom_size", None))
        gx_buf, gx_off = views["geom_xpos"].buffer, views["geom_xpos"].offset
        if self.pose is not None:           # site slots: their pose rows after the geoms'
            sx, sm = wm.buffer_of(sim.d.site_xpos), wm.buffer_of(sim.d.site_xmat)
            (gx_buf, gx_off), (gm_buf, gm_off) = self.pose.encode(cb, (gx_buf, gx_off), (gm_buf, gm_off),
                                                                  (sx.buffer, sx.offset), (sm.buffer, sm.offset))
        self._encode(cb, gx_buf, gx_off, gm_buf,
                     gm_off, views["cam_xpos"].buffer, views["cam_xpos"].offset,
                     views["cam_xmat"].buffer, views["cam_xmat"].offset,
                     lx.buffer if lx else None, lx.offset if lx else 0, ld.buffer if ld else None, ld.offset if ld else 0,
                     sim=sim)
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

    def render_host(self, datas, cam_pos=None, cam_quat=None, geom_size=None, sources=None):
        """Render from a list of N ``mujoco.MjData`` (mj_forward'd). Optional per-env camera pose
        overrides ``(N,3)``/``(N,4)`` replace the model camera; ``geom_size`` (N, ngeom, 3): per-world sizes of the
        terrain slots (default the model's); ``sources``: as ``render`` (numpy (N, V, 3) arrays). Synchronizes;
        returns numpy outputs."""
        n, ng, nc = self.n, self.m.ngeom, max(self.m.ncam, 1)
        gx = np.stack([d.geom_xpos for d in datas]).astype(np.float32).reshape(n, ng, 3)
        gm = np.stack([d.geom_xmat for d in datas]).astype(np.float32).reshape(n, ng, 9)
        if self.sized is not None:
            gm = self.sized.scale_host(gm, geom_size).astype(np.float32)
        if self.pose is not None:
            gx, gm = self.pose.host(gx, gm, datas)
        self._set_sources(sources=sources, datas=datas)
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
