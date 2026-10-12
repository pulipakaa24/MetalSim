"""Ray-traced sensors on Metal hardware ray tracing (WS5).

``RayTracer`` builds one primitive acceleration structure per unique mesh (from the renderer's
scene tables) and, each frame, refits an instance descriptor buffer from the physics state
(``geom_xpos/geom_xmat``) and rebuilds one instance acceleration structure for all envs. Envs are
spatially separated in the structure by a grid offset larger than the scene extent plus the
maximum sensor range, so a ray of env e never meets another env's geometry and the BVH stays
shallow. Sensors are compute kernels issuing ``intersector`` queries:

* ``Lidar``: beams (azimuth, elevation) in a site frame; outputs range, point, normal, hit slot and
  a reflectance-based intensity per beam as MPS tensors.
* ``raycast_depth``: pinhole depth by ray casting (validation of the rasterizer and of the
  acceleration structures; also the basis for tier-1 ray queries).

Ordering follows the renderer: ``trace(sim, after_value)`` waits for the sim event, signals its own.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp
import Metal

from metalsim.interop import torch_bridge as tb
from metalsim.interop import warp_metal as wm
from metalsim.render.metal_context import MetalContext
from metalsim.render.scene_tables import CameraIntrinsics, build_scene_tables


class RayTracer:
    def __init__(self, model, n_envs: int, *, max_range: float = 10.0, max_group=3, include_planes=True,
                 decimate_faces: int = 0, terrain_slots: bool = True, draw_hfields: bool = True,
                 appearance: str = "mujoco", visual=None, deformables=(), tube_slices=None, cap_stacks=None,
                 dyn_rebuild_every: int = 32, ctx: MetalContext | None = None,
                 device="metal:0"):
        """``terrain_slots`` / ``draw_hfields`` / ``appearance``: see ``Tier0Renderer`` (False / "legacy" = the
        previous behaviour).

        ``visual`` / ``deformables`` / ``tube_slices`` / ``cap_stacks``: the non-geom objects MuJoCo's renderer draws
        (sites, flexes, skins, spatial tendons) and user deformable meshes; see ``Tier0Renderer``. Each dynamic object
        has one primitive acceleration structure per env, in the env's own frame and placed by its instance's translation
        as the rigid meshes are, refit each frame after the vertices are regenerated and rebuilt every
        ``dyn_rebuild_every`` frames (1: every frame, 0: refit only after the first build)."""
        self.m = model
        self.n = n_envs
        self.device = device
        self.ctx = ctx or MetalContext(device)
        if not self.ctx.device.supportsRaytracing():
            raise RuntimeError("this GPU has no Metal ray tracing support")
        if visual is None:
            # a sensor's default: what MuJoCo draws except sites (visual markers, not surfaces: mj_ray and MuJoCo's
            # rangefinder skip them, and a lidar's own mounting site would occlude every beam). The cameras built on this
            # ray tracer (tier 1 / tier 2) pass MuJoCo's full default option.
            import mujoco
            visual = mujoco.MjvOption()
            visual.sitegroup[:] = 0
        self.tables = build_scene_tables(model, n_envs, max_group=max_group, include_planes=include_planes,
                                         decimate_faces=decimate_faces, sized_geoms="auto" if terrain_slots else (),
                                         draw_hfields=draw_hfields, appearance=appearance,
                                         visual=visual, deformables=deformables)
        t = self.tables
        self.visual = visual
        self.sized = None
        if t.sized.any():
            from metalsim.render.sized import SizedGeoms
            self.sized = SizedGeoms(self.ctx, model, n_envs, t.sized_geom_ids())
        self.G = t.G
        self.tpr = int(np.ceil(np.sqrt(n_envs)))
        extent = float(model.stat.extent)
        self.max_range = max_range
        self.stride = 2.0 * (extent + max_range) + 1.0
        c = self.ctx
        # non-geom objects (metalsim.render.extras): sites are instanced slots posed from extra pose rows; each dynamic
        # object (flex, skin, tendon, user mesh) has its records after the static vertices, its every-env triangles
        # after the static indices, a mesh entry, and one identity instance
        self.dyn, self.pose, self._dyn_src = None, None, None
        self._dyn_frames, self.dyn_rebuild_every = 0, int(dyn_rebuild_every)
        inst_tab, mesh_list = t.inst_table, list(t.meshes)
        verts, inds = t.vertices, t.indices
        n_inst_slots = len(t.instanced_slot_pose())
        slot_mesh = np.full(self.G, -1, np.int64)
        slot_mesh[:n_inst_slots] = t.geom_mesh if t.slot_mesh_inst is None else t.slot_mesh_inst
        if t.dyn:
            from metalsim.render.dynamic_mesh import DynamicMeshes
            nv_static = len(t.vertices)
            # records in the env's own (world) frame, no grid offset: one primitive structure per env holding every
            # dynamic object as one geometry (in object order), placed in the env grid by its instance's translation as
            # the rigid meshes are. (Offsets baked into one structure over many envs put far envs' vertices at tens of
            # km, where float32 spacing reaches millimetres, and made rays 3-10x slower than the same triangles as static
            # instanced meshes; one structure per object and env made the per-frame refits host-bound.) A hit's slot is
            # the instance's first slot + its geometry index, its mesh the instance's structure index + the geometry
            # index: the env structure is listed once per object in the instanced-structure list.
            self.dyn = DynamicMeshes(self.ctx, model, t, n_envs, tpr=self.tpr, stride=0.0, rbase=nv_static,
                                     vbuf=False, tube_slices=tube_slices, cap_stacks=cap_stacks)
            exp = self.dyn.expanded_indices()
            verts = np.zeros((nv_static + self.dyn.n_records, 8), np.float32)
            verts[:nv_static] = t.vertices
            objs = self.dyn.objects
            slots = [ob["slot"] for ob in objs]
            if any(len(ob["tris"]) == 0 for ob in objs) or slots != list(range(slots[0], slots[0] + len(objs))):
                raise ValueError("dynamic objects must have triangles and consecutive slots")   # extras.extend_tables
            obj_off, i_off = [], len(t.indices)
            for ob, ix in zip(objs, exp):
                obj_off.append(i_off)
                i_off += ix.size
            self.n_dyn_obj = len(objs)
            dyn_inst = []
            for e in range(n_envs):               # mesh entries env-major: the env's structure, then its geometries
                for k, ob in enumerate(objs):
                    per_env = 3 * len(ob["tris"])
                    mesh_list.append({"i_off": obj_off[k] + e * per_env, "i_count": per_env, "dynamic": k, "env": e})
                dyn_inst.append((e, slots[0]))
            for k, ob in enumerate(objs):
                slot_mesh[ob["slot"]] = len(t.meshes) + k            # env 0's (only used by code that ignores envs)
            inds = np.concatenate([t.indices] + exp).astype(np.uint32)
            inst_tab = np.concatenate([t.inst_table, np.asarray(dyn_inst, np.uint32).reshape(-1, 2)])
        self.vbuf = c.buffer(verts.nbytes, verts, "rt_vertices")
        if self.dyn is not None:
            self.dyn.vbuf = self.vbuf
        self.ibuf = c.buffer(inds.nbytes, inds, "rt_indices")
        self.inst_tab = c.buffer(inst_tab.nbytes, inst_tab, "rt_inst_tab")
        self.slot_geom = c.buffer(4 * self.G, t.instanced_slot_pose(), "rt_slot_geom")
        self.slot_mesh = c.buffer(4 * self.G, slot_mesh.astype(np.int32), "rt_slot_mesh")
        self.meshes_all = mesh_list
        mi = np.zeros((len(mesh_list), 4), np.uint32)
        for k, mesh in enumerate(mesh_list):
            mi[k, 0] = mesh["i_off"]; mi[k, 1] = mesh["i_count"]
        self.mesh_info = c.buffer(mi.nbytes, mi, "rt_mesh_info")
        # material table (G x 16) followed by the MaterialExt table (G x 8; scene_tables.MATERIAL_EXT_LAYOUT), as the
        # path tracer reads it; kernels that index the material table alone are unaffected by the tail
        mext = t.material_ext if t.material_ext is not None else np.zeros((self.G, 8), np.float32)
        mt = np.concatenate([np.asarray(t.materials, np.float32).reshape(-1), np.asarray(mext, np.float32).reshape(-1)])
        self.mat_buf = c.buffer(mt.nbytes, mt, "rt_materials")
        self.n_inst_static = len(t.inst_table)     # instances refit from the pose rows (geoms, then sites)
        self.n_inst = len(inst_tab)                # + one identity instance per dynamic object (the scene's instances)
        self.inst_desc = c.buffer(64 * self.n_inst, label="rt_instance_descriptors")
        self.consts = np.zeros(12, np.uint32)
        self.consts[:4] = (self.n, t.pose_rows(model), n_inst_slots, self.tpr)
        if t.site_ids:
            from metalsim.render.dynamic_mesh import PoseGather
            self.pose = PoseGather(self.ctx, model, t, n_envs)
        self.consts_f = self.consts.view(np.float32)
        self.consts_f[4] = self.stride; self.consts_f[5] = max_range
        self.consts[7] = max(model.nsite, 1)
        self.consts_buf = c.buffer(self.consts.nbytes, self.consts, "rt_consts")
        lib = c.library("rt")
        self.p_refit = c.compute_pipeline(lib, "refit_instances")
        self.p_lidar = c.compute_pipeline(lib, "lidar")
        self.p_lidar_ext = c.compute_pipeline(lib, "lidar_ext")
        self.p_depth = c.compute_pipeline(lib, "raycast_depth")
        self._build_primitive_structures()
        self._alloc_instance_structure()
        self.event = self.ctx.event
        rows = t.pose_rows(model)
        self._host_geom_xpos = c.buffer(self.n * rows * 12, label="rt_host_geom_xpos")
        self._host_geom_xmat = c.buffer(self.n * rows * 36, label="rt_host_geom_xmat")
        self._host_site_xpos = c.buffer(self.n * max(model.nsite, 1) * 12, label="rt_host_site_xpos")
        self._host_site_xmat = c.buffer(self.n * max(model.nsite, 1) * 36, label="rt_host_site_xmat")
        self._host_cam_xpos = c.buffer(self.n * max(model.ncam, 1) * 12, label="rt_host_cam_xpos")
        self._host_cam_xmat = c.buffer(self.n * max(model.ncam, 1) * 36, label="rt_host_cam_xmat")

    # -- acceleration structures ------------------------------------------------------------------

    def _build_primitive_structures(self):
        c, t = self.ctx, self.tables
        self.prim_as = []
        descs = []
        for mesh in t.meshes:
            g = Metal.MTLAccelerationStructureTriangleGeometryDescriptor.new()
            g.setVertexBuffer_(self.vbuf); g.setVertexBufferOffset_(0); g.setVertexStride_(32)
            g.setVertexFormat_(Metal.MTLAttributeFormatFloat3)
            g.setIndexBuffer_(self.ibuf); g.setIndexBufferOffset_(int(mesh["i_off"]) * 4)
            g.setIndexType_(Metal.MTLIndexTypeUInt32)
            g.setTriangleCount_(int(mesh["i_count"]) // 3)
            g.setOpaque_(True)
            d = Metal.MTLPrimitiveAccelerationStructureDescriptor.new()
            d.setGeometryDescriptors_([g])
            descs.append(d)
        sizes = [c.device.accelerationStructureSizesWithDescriptor_(d) for d in descs]
        scratch_size = max(s.buildScratchBufferSize for s in sizes) if sizes else 16
        scratch = c.buffer(scratch_size, label="rt_scratch")
        cb = c.command_buffer()
        enc = cb.accelerationStructureCommandEncoder()
        for d, s in zip(descs, sizes):
            a = c.device.newAccelerationStructureWithSize_(s.accelerationStructureSize)
            enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(a, d, scratch, 0)
            self.prim_as.append(a)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        if cb.error() is not None:
            raise RuntimeError(f"primitive acceleration structure build failed: {cb.error()}")
        self._build_dynamic_structures()

    def _build_dynamic_structures(self):
        """One primitive structure per dynamic object over every env's triangles, built on the first frame, then refit
        (and rebuilt every ``dyn_rebuild_every`` frames) after the vertices are regenerated."""
        self.dyn_as, self.dyn_desc = [], []
        self.prim_as_unique = list(self.prim_as)
        if self.dyn is None:
            return
        c = self.ctx
        scratch = 16
        dm = self.meshes_all[len(self.tables.meshes):]
        D = self.n_dyn_obj
        sizes = []
        for e in range(self.n):
            geos = []
            for mesh in dm[e * D:(e + 1) * D]:
                g = Metal.MTLAccelerationStructureTriangleGeometryDescriptor.new()
                g.setVertexBuffer_(self.vbuf); g.setVertexBufferOffset_(0); g.setVertexStride_(32)
                g.setVertexFormat_(Metal.MTLAttributeFormatFloat3)
                g.setIndexBuffer_(self.ibuf); g.setIndexBufferOffset_(int(mesh["i_off"]) * 4)
                g.setIndexType_(Metal.MTLIndexTypeUInt32)
                g.setTriangleCount_(int(mesh["i_count"]) // 3)
                g.setOpaque_(True)
                geos.append(g)
            d = Metal.MTLPrimitiveAccelerationStructureDescriptor.new()
            d.setGeometryDescriptors_(geos)
            self._dyn_refittable = self.dyn_rebuild_every != 1
            if self._dyn_refittable:                 # refit between rebuilds: the structure must allow it
                d.setUsage_(Metal.MTLAccelerationStructureUsageRefit)
            s = c.device.accelerationStructureSizesWithDescriptor_(d)
            scratch = max(scratch, int(s.buildScratchBufferSize), int(s.refitScratchBufferSize))
            self.dyn_desc.append(d)
            sizes.append(int(s.accelerationStructureSize))
        # the env structures live in one heap: residency is one useHeap call per encoder, not one call per env
        sa = [c.device.heapAccelerationStructureSizeAndAlignWithSize_(z) for z in sizes]
        hd = Metal.MTLHeapDescriptor.new()
        hd.setSize_(sum((int(x.size) + int(x.align) - 1) // int(x.align) * int(x.align) for x in sa) + 65536)
        hd.setStorageMode_(Metal.MTLStorageModePrivate)
        hd.setHazardTrackingMode_(Metal.MTLHazardTrackingModeTracked)   # refits / builds ordered like any resource
        self.dyn_heap = c.device.newHeapWithDescriptor_(hd)
        if self.dyn_heap is None:
            raise RuntimeError("could not allocate the dynamic objects' acceleration-structure heap")
        for z in sizes:
            a = self.dyn_heap.newAccelerationStructureWithSize_(z)
            if a is None:
                raise RuntimeError("acceleration-structure heap too small")
            self.dyn_as.append(a)
        self.dyn_scratch_stride = (scratch + 255) // 256 * 256      # one scratch region per structure (no sharing)
        self.dyn_scratch = c.buffer(self.dyn_scratch_stride * len(self.dyn_as), label="rt_dyn_scratch")
        # one list for the instance structure (particles.py appends grains): each env structure once per geometry, so
        # as_index + geometry index is the mesh entry of the hit object
        self.prim_as_unique = list(self.prim_as)    # residency: the static structures one by one, the dynamic heap once
        self.prim_as = list(self.prim_as) + [a for a in self.dyn_as for _ in range(D)]
        self._dyn_built = False

    def _encode_dynamic(self, cb, src):
        """Regenerate the dynamic objects' vertices from ``src``, then rebuild or refit their structures."""
        self.dyn.encode(cb, src)
        rebuild = (not self._dyn_built) or not self._dyn_refittable or (self.dyn_rebuild_every > 0 and self._dyn_frames % self.dyn_rebuild_every == 0)
        ae = cb.accelerationStructureCommandEncoder()
        sc, st = self.dyn_scratch, self.dyn_scratch_stride      # one call per structure: keep the loop lean (host time)
        if rebuild:
            f = ae.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_
            for k, (a, d) in enumerate(zip(self.dyn_as, self.dyn_desc)):
                f(a, d, sc, k * st)
        else:
            f = ae.refitAccelerationStructure_descriptor_destination_scratchBuffer_scratchBufferOffset_
            for k, (a, d) in enumerate(zip(self.dyn_as, self.dyn_desc)):
                f(a, d, None, sc, k * st)
        ae.endEncoding()
        self._dyn_built = True
        self._dyn_frames += 1

    def _alloc_instance_structure(self):
        c = self.ctx
        d = Metal.MTLInstanceAccelerationStructureDescriptor.new()
        d.setInstancedAccelerationStructures_(self.prim_as)
        d.setInstanceCount_(self.n_inst)
        d.setInstanceDescriptorBuffer_(self.inst_desc)
        d.setInstanceDescriptorBufferOffset_(0)
        d.setInstanceDescriptorStride_(64)
        d.setInstanceDescriptorType_(Metal.MTLAccelerationStructureInstanceDescriptorTypeDefault)
        self.inst_desc_obj = d
        s = c.device.accelerationStructureSizesWithDescriptor_(d)
        self.inst_as = c.device.newAccelerationStructureWithSize_(s.accelerationStructureSize)
        self.inst_scratch = c.buffer(max(int(s.buildScratchBufferSize), 16), label="rt_inst_scratch")
        if self.n_inst > self.n_inst_static:   # dynamic objects: one instance per env, translated into the env grid
            desc = np.zeros((self.n_inst - self.n_inst_static, 16), np.uint32)
            f = desc.view(np.float32)
            f[:, 0] = f[:, 4] = f[:, 8] = 1.0                 # rotation columns (1,0,0), (0,1,0), (0,0,1)
            env = np.arange(self.n)
            f[:, 9] = (env % self.tpr) * self.stride          # translation: the env's grid offset (rt.metal env_offset)
            f[:, 10] = (env // self.tpr) * self.stride
            desc[:, 12] = 4                                   # opaque
            desc[:, 13] = 0xFFFFFFFF
            desc[:, 15] = len(self.tables.meshes) + np.arange(len(desc)) * self.n_dyn_obj   # env e's first listing
            c.write_buffer(self.inst_desc, desc, offset=64 * self.n_inst_static)

    def encode_geom_xmat(self, cb, sim, gm_buf, gm_off):
        """The geom_xmat to refit from: the physics buffer, or its copy with the per-world slot sizes folded in."""
        if self.sized is None:
            return gm_buf, gm_off
        return self.sized.encode(cb, gm_buf, gm_off, sim.m.geom_size)

    def encode_pose(self, cb, sim, gx_buf, gx_off, gm_buf, gm_off):
        """The pose rows to refit from: the geoms' (as given), then the drawn sites' when the scene has site slots."""
        if self.pose is None:
            return gx_buf, gx_off, gm_buf, gm_off
        sx, sm = wm.buffer_of(sim.d.site_xpos), wm.buffer_of(sim.d.site_xmat)
        (xb, xo), (mb, mo) = self.pose.encode(cb, (gx_buf, gx_off), (gm_buf, gm_off), (sx.buffer, sx.offset), (sm.buffer, sm.offset))
        return xb, xo, mb, mo

    def set_sources(self, sim=None, sources=None, datas=None):
        """Bind the per-frame sources of the dynamic objects: from a MuJoCo Warp ``sim`` (its ``d``) or, on the host
        path, from ``datas`` (MjData list); ``sources``: {name: array} overrides (``flexvert_xpos`` from another
        simulator, e.g. ``XPBDSim.x``; one (N, V, 3) array per ``DeformableMesh``)."""
        if self.dyn is None:
            if sources:
                raise ValueError("this scene has no dynamic objects")
            return
        self._dyn_src = self.dyn.write_host(datas, sources) if datas is not None else self.dyn.sources_from_sim(sim.d, sources)

    def _encode_refit_and_build(self, cb, geom_xpos, gx_off, geom_xmat, gm_off):
        self._encode_refit(cb, geom_xpos, gx_off, geom_xmat, gm_off)
        ae = cb.accelerationStructureCommandEncoder()
        ae.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(self.inst_as, self.inst_desc_obj, self.inst_scratch, 0)
        ae.endEncoding()

    def _encode_refit(self, cb, geom_xpos, gx_off, geom_xmat, gm_off, inst_desc=None):
        """Scene instance descriptors from the physics state into ``inst_desc`` (default: this tracer's own buffer;
        the particle renderer passes a larger buffer whose tail holds grain instances, metalsim/render/particles.py).
        The scene instances are the ``n_inst_static`` rigid ones (geoms, sites) refit here, then one identity instance
        per dynamic object (``n_inst`` in all), whose vertices are regenerated and structures refit first."""
        if self.dyn is not None:
            if self._dyn_src is None:
                raise ValueError("dynamic objects need their sources (set_sources; render / trace call it)")
            self._encode_dynamic(cb, self._dyn_src)
            if inst_desc is not None and inst_desc is not self.inst_desc:     # the dynamic objects' identity instances
                be = cb.blitCommandEncoder()
                be.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
                    self.inst_desc, 64 * self.n_inst_static, inst_desc, 64 * self.n_inst_static, 64 * (self.n_inst - self.n_inst_static))
                be.endEncoding()
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.p_refit)
        ce.setBuffer_offset_atIndex_(inst_desc if inst_desc is not None else self.inst_desc, 0, 0)
        ce.setBuffer_offset_atIndex_(self.inst_tab, 0, 1)
        ce.setBuffer_offset_atIndex_(self.slot_geom, 0, 2)
        ce.setBuffer_offset_atIndex_(self.slot_mesh, 0, 3)
        ce.setBuffer_offset_atIndex_(geom_xpos, gx_off, 4)
        ce.setBuffer_offset_atIndex_(geom_xmat, gm_off, 5)
        ce.setBuffer_offset_atIndex_(self.consts_buf, 0, 6)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n_inst_static, 1, 1), Metal.MTLSize(64, 1, 1))
        ce.endEncoding()

    def _use_resources(self, ce):
        ce.useResource_usage_(self.inst_as, Metal.MTLResourceUsageRead)
        for a in self.prim_as_unique:       # static meshes, then the dynamic objects' heap
            ce.useResource_usage_(a, Metal.MTLResourceUsageRead)
        if self.dyn_as:
            ce.useHeap_(self.dyn_heap)

    # -- sensors ------------------------------------------------------------------------------------

    def make_lidar(self, site: str | int, azimuths: np.ndarray, elevations: np.ndarray, **extras):
        """``extras``: beam divergence / multi-return / reflectance options of ``Lidar``."""
        return Lidar(self, site, azimuths, elevations, **extras)

    def default_slot_reflectance(self) -> np.ndarray:
        """Per-slot reflectance used by the default lidar kernel: MuJoCo material reflectance, 0.5 if unset."""
        w = self.tables.materials.reshape(self.G, -1)[:, 15]
        return np.where(w > 0, w, 0.5).astype(np.float32)

    def slot_reflectance(self, reflectance=None) -> np.ndarray:
        """Per-slot reflectance: the default, overridden by {geom name or regex: value} or an (ngeom,) array."""
        import mujoco
        import re
        refl = self.default_slot_reflectance()
        if reflectance is None:
            return refl
        geoms = np.asarray(self.tables.geoms)
        if isinstance(reflectance, dict):
            names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in geoms]
            names += [e["name"] or "" for e in self.tables.extras]     # extra slots (sites, flexes, ...) follow the geoms
            for key, val in reflectance.items():
                hit = [k for k, nm in enumerate(names) if nm == key or re.fullmatch(key, nm)]
                if not hit:
                    raise ValueError(f"no traced geom matches {key!r}")
                refl[hit] = val
            return refl
        if len(geoms) == self.G:
            return np.asarray(reflectance, np.float32)[geoms]
        refl[:len(geoms)] = np.asarray(reflectance, np.float32)[geoms]
        return refl

    def make_depth_camera(self, camera: str | int, width: int, height: int):
        return RayDepthCamera(self, camera, width, height)

    # -- frame drivers -------------------------------------------------------------------------------

    def trace(self, sim, sensors, after_value: int | None = None, sources=None) -> int:
        """GPU path: refit + rebuild from ``sim`` state, then run each sensor's kernel. Waits for the
        sim event, signals the context event; returns the value. ``sources``: see ``set_sources`` (a ``sim``
        without an event, e.g. ``DeformableSim`` / ``XPBDSim``, must be synchronized by the caller)."""
        import mujoco  # noqa
        ev = getattr(sim, "event", None)
        if after_value is None and ev is not None:
            after_value = ev.value
        gx, gm = wm.buffer_of(sim.d.geom_xpos), wm.buffer_of(sim.d.geom_xmat)
        self.set_sources(sim, sources)
        cb = self.ctx.command_buffer()
        if ev is not None:
            self.ctx.wait_for(cb, ev, after_value)
        gm_buf, gm_off = self.encode_geom_xmat(cb, sim, gm.buffer, gm.offset)
        gx_buf, gx_off, gm_buf, gm_off = self.encode_pose(cb, sim, gx.buffer, gx.offset, gm_buf, gm_off)
        self._encode_refit_and_build(cb, gx_buf, gx_off, gm_buf, gm_off)
        for s in sensors:
            s.encode(cb, sim=sim)
        v = self.ctx.signal(cb)
        self.ctx.commit(cb)
        return v

    def trace_host(self, datas, sensors, sources=None):
        """Host path from ``mujoco.MjData`` lists (tests/tools); synchronizes. ``sources``: see ``set_sources``."""
        n, ng, ns, nc = self.n, self.m.ngeom, max(self.m.nsite, 1), max(self.m.ncam, 1)
        c = self.ctx
        gx = np.stack([d.geom_xpos for d in datas]).astype(np.float32).reshape(n, ng, 3)
        gm = np.stack([d.geom_xmat for d in datas]).astype(np.float32).reshape(n, ng, 9)
        gm = self.sized.scale_host(gm).astype(np.float32) if self.sized is not None else gm
        if self.pose is not None:
            gx, gm = self.pose.host(gx, gm, datas)
        c.write_buffer(self._host_geom_xpos, gx)
        c.write_buffer(self._host_geom_xmat, gm)
        self.set_sources(sources=sources, datas=datas)
        if self.m.nsite:
            c.write_buffer(self._host_site_xpos, np.stack([d.site_xpos for d in datas]).astype(np.float32).reshape(n, ns, 3))
            c.write_buffer(self._host_site_xmat, np.stack([d.site_xmat for d in datas]).astype(np.float32).reshape(n, ns, 9))
        if self.m.ncam:
            c.write_buffer(self._host_cam_xpos, np.stack([d.cam_xpos for d in datas]).astype(np.float32).reshape(n, nc, 3))
            c.write_buffer(self._host_cam_xmat, np.stack([d.cam_xmat for d in datas]).astype(np.float32).reshape(n, nc, 9))
        cb = c.command_buffer()
        self._encode_refit_and_build(cb, self._host_geom_xpos, 0, self._host_geom_xmat, 0)
        for s in sensors:
            s.encode(cb, sim=None)
        c.commit(cb)
        c.synchronize()
        return {s.name: s.numpy() for s in sensors}


class Lidar:
    """Beams given as (azimuth, elevation) pairs in the site frame (radians).

    Default (no extras): one infinitesimal ray per beam, first hit; outputs (N, beams[, 3]),
    identical to ``mj_ray`` up to tessellation.

    Extras (Isaac Sim RTX lidar attributes in brackets; see ``lidar_ext`` in rt.metal for the model):

    * ``divergence_deg=(hor, ver)`` full-angle beam divergence [``divergenceHorDeg``,
      ``divergenceVerDeg``], sampled by ``spot_rays`` sub-rays (sunflower pattern over the elliptical
      spot; ``beam_profile`` "uniform" [``UNIFORM_BEAM``] or "gaussian" (divergence = 1/e^2 full
      width) [``GAUSSIAN_BEAM``]).
    * ``max_returns`` [``maxReturns``]: echoes per beam; each sub-ray is re-cast past the object it
      hit up to ``hits_per_ray`` times (default ``max_returns``), with power scaled by
      ``transmission`` per pass; hits closer than ``min_echo_sep`` [``minDistBetweenEchosM``] merge
      into one echo with the power-weighted mean range (mixed pixel).
    * ``reflectance``: per-geom reflectance, {geom name or regex: value} or an (ngeom,) array;
      unset geoms keep the default (MuJoCo material reflectance, else 0.5). intensity = sum over
      the echo's hits of weight * reflectance * cos(incidence) / r^2.

    With extras, outputs gain a trailing returns dimension when ``max_returns > 1``
    ("range" (N, beams, R), "points" (N, beams, R, 3), ...) and "n_returns" (N, beams).
    """

    def __init__(self, rt: RayTracer, site, azimuths, elevations, name="lidar", *, divergence_deg=(0.0, 0.0),
                 spot_rays: int = 1, beam_profile: str = "uniform", max_returns: int = 1, hits_per_ray: int | None = None,
                 min_echo_sep: float = 0.4, transmission: float = 1.0, reflectance=None):
        import mujoco
        self.rt, self.name = rt, name
        m = rt.m
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site) if isinstance(site, str) else int(site)
        if self.site < 0:
            raise ValueError(f"site {site!r} not found")
        az, el = np.broadcast_arrays(np.asarray(azimuths, np.float32), np.asarray(elevations, np.float32))
        self.beams = np.stack([az.reshape(-1), el.reshape(-1)], 1).astype(np.float32)
        self.n_beams = len(self.beams)
        self.beam_buf = rt.ctx.buffer(self.beams.nbytes, self.beams, "lidar_beams")
        self.consts = rt.consts.copy()
        self.consts[6] = self.n_beams; self.consts[8] = self.site
        self.consts_buf = rt.ctx.buffer(self.consts.nbytes, self.consts, "lidar_consts")
        hits_per_ray = int(hits_per_ray or max_returns)
        self.extended = (spot_rays != 1 or max_returns != 1 or hits_per_ray != 1 or reflectance is not None
                         or tuple(divergence_deg) != (0.0, 0.0))
        self.max_returns = int(max_returns)
        n = rt.n
        if not self.extended:
            shp = (n, self.n_beams)
        else:
            if spot_rays * hits_per_ray > 32 or max_returns > 32 or spot_rays < 1 or max_returns < 1:
                raise ValueError("spot_rays * hits_per_ray and max_returns must be <= 32 (LIDAR_MAX_CAND)")
            self._setup_ext(divergence_deg, spot_rays, beam_profile, hits_per_ray, min_echo_sep, transmission, reflectance)
            shp = (n, self.n_beams) if self.max_returns == 1 else (n, self.n_beams, self.max_returns)
        self._arrays = {
            "range": wp.zeros(shp, dtype=wp.float32, device=rt.device),
            "points": wp.zeros(shp + (3,), dtype=wp.float32, device=rt.device),
            "normals": wp.zeros(shp + (3,), dtype=wp.float32, device=rt.device),
            "slot": wp.zeros(shp, dtype=wp.int32, device=rt.device),
            "intensity": wp.zeros(shp, dtype=wp.float32, device=rt.device),
        }
        if self.extended:
            self._arrays["n_returns"] = wp.zeros((n, self.n_beams), dtype=wp.int32, device=rt.device)
        wp.synchronize_device(rt.device)
        self._views = {k: wm.buffer_of(a) for k, a in self._arrays.items()}
        self.out = {k: tb.mps_tensor(a) for k, a in self._arrays.items()}

    @staticmethod
    def spot_pattern(divergence_deg, spot_rays: int, beam_profile: str = "uniform") -> np.ndarray:
        """(K, 4) sub-ray table: tangent-plane slopes along azimuth / elevation, power weight, 0.
        Sunflower points over the unit disk scaled to the elliptical spot (half-angles = divergence/2);
        K = 1 is the beam axis."""
        K = int(spot_rays)
        th, tv = np.tan(np.deg2rad(np.asarray(divergence_deg, np.float64) / 2.0))
        i = np.arange(K)
        rho = np.sqrt((i + 0.5) / K) if K > 1 else np.zeros(1)
        phi = i * np.pi * (3.0 - np.sqrt(5.0))
        x, y = rho * np.cos(phi), rho * np.sin(phi)
        if beam_profile == "uniform":
            w = np.ones(K)
        elif beam_profile == "gaussian":
            w = np.exp(-2.0 * rho ** 2)
        else:
            raise ValueError("beam_profile must be 'uniform' or 'gaussian'")
        w = w / w.sum()
        return np.stack([x * th, y * tv, w, np.zeros(K)], 1).astype(np.float32)

    def _setup_ext(self, divergence_deg, spot_rays, beam_profile, hits_per_ray, min_echo_sep, transmission, reflectance):
        rt = self.rt
        self.sub = self.spot_pattern(divergence_deg, spot_rays, beam_profile)
        self.sub_buf = rt.ctx.buffer(self.sub.nbytes, self.sub, "lidar_sub")
        ext = np.zeros(8, np.uint32)
        ext[0], ext[1], ext[2] = spot_rays, self.max_returns, hits_per_ray
        ext.view(np.float32)[4] = min_echo_sep; ext.view(np.float32)[5] = transmission
        self.ext_buf = rt.ctx.buffer(ext.nbytes, ext, "lidar_ext")
        refl = rt.slot_reflectance(reflectance)
        self.slot_reflectance = refl.astype(np.float32)
        self.refl_buf = rt.ctx.buffer(self.slot_reflectance.nbytes, self.slot_reflectance, "lidar_refl")

    def encode(self, cb, sim=None):
        rt = self.rt
        if sim is not None:
            sx, sm = wm.buffer_of(sim.d.site_xpos), wm.buffer_of(sim.d.site_xmat)
            sxb, sxo, smb, smo = sx.buffer, sx.offset, sm.buffer, sm.offset
        else:
            sxb, sxo, smb, smo = rt._host_site_xpos, 0, rt._host_site_xmat, 0
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(rt.p_lidar_ext if self.extended else rt.p_lidar)
        rt._use_resources(ce)
        ce.setAccelerationStructure_atBufferIndex_(rt.inst_as, 0)
        ce.setBuffer_offset_atIndex_(self.beam_buf, 0, 1)
        ce.setBuffer_offset_atIndex_(sxb, sxo, 2)
        ce.setBuffer_offset_atIndex_(smb, smo, 3)
        ce.setBuffer_offset_atIndex_(rt.inst_tab, 0, 4)
        ce.setBuffer_offset_atIndex_(rt.slot_mesh, 0, 5)
        ce.setBuffer_offset_atIndex_(rt.mesh_info, 0, 6)
        ce.setBuffer_offset_atIndex_(rt.vbuf, 0, 7)
        ce.setBuffer_offset_atIndex_(rt.ibuf, 0, 8)
        ce.setBuffer_offset_atIndex_(self.refl_buf if self.extended else rt.mat_buf, 0, 9)
        ce.setBuffer_offset_atIndex_(rt.inst_desc, 0, 10)
        ce.setBuffer_offset_atIndex_(self.consts_buf, 0, 11)
        if self.extended:
            ce.setBuffer_offset_atIndex_(self.sub_buf, 0, 12)
            ce.setBuffer_offset_atIndex_(self.ext_buf, 0, 13)
            keys, base = ("range", "points", "normals", "slot", "intensity", "n_returns"), 14
        else:
            keys, base = ("range", "points", "normals", "slot", "intensity"), 12
        for i, k in enumerate(keys):
            v = self._views[k]
            ce.setBuffer_offset_atIndex_(v.buffer, v.offset, base + i)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n_beams, rt.n, 1), Metal.MTLSize(64, 1, 1))
        ce.endEncoding()

    def numpy(self):
        return {k: a.numpy().copy() for k, a in self._arrays.items()}


class RayDepthCamera:
    def __init__(self, rt: RayTracer, camera, width, height, name="ray_depth"):
        import mujoco
        self.rt, self.name, self.w, self.h = rt, name, width, height
        m = rt.m
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera) if isinstance(camera, str) else int(camera)
        ci = CameraIntrinsics.from_model(m, self.cam, width, height)
        self.intr = np.array([ci.fx, ci.fy, ci.cx, ci.cy], np.float32)
        self.intr_buf = rt.ctx.buffer(16, self.intr, "raydepth_intr")
        self.dims = np.array([width, height, max(m.ncam, 1), self.cam], np.uint32)
        self.dims_buf = rt.ctx.buffer(16, self.dims, "raydepth_dims")
        self._array = wp.zeros((rt.n, height, width), dtype=wp.float32, device=rt.device)
        wp.synchronize_device(rt.device)
        self._view = wm.buffer_of(self._array)
        self.out = tb.mps_tensor(self._array)

    def encode(self, cb, sim=None):
        rt = self.rt
        if sim is not None:
            cx, cm = wm.buffer_of(sim.d.cam_xpos), wm.buffer_of(sim.d.cam_xmat)
            cxb, cxo, cmb, cmo = cx.buffer, cx.offset, cm.buffer, cm.offset
        else:
            cxb, cxo, cmb, cmo = rt._host_cam_xpos, 0, rt._host_cam_xmat, 0
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(rt.p_depth)
        rt._use_resources(ce)
        ce.setAccelerationStructure_atBufferIndex_(rt.inst_as, 0)
        ce.setBuffer_offset_atIndex_(cxb, cxo, 1)
        ce.setBuffer_offset_atIndex_(cmb, cmo, 2)
        ce.setBuffer_offset_atIndex_(self.intr_buf, 0, 3)
        ce.setBuffer_offset_atIndex_(self.dims_buf, 0, 4)
        ce.setBuffer_offset_atIndex_(rt.consts_buf, 0, 5)
        ce.setBuffer_offset_atIndex_(self._view.buffer, self._view.offset, 6)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.w, self.h, rt.n), Metal.MTLSize(16, 16, 1))
        ce.endEncoding()

    def numpy(self):
        return self._array.numpy().copy()
