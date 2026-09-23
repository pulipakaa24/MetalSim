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
                 decimate_faces: int = 0, ctx: MetalContext | None = None, device="metal:0"):
        self.m = model
        self.n = n_envs
        self.device = device
        self.ctx = ctx or MetalContext(device)
        if not self.ctx.device.supportsRaytracing():
            raise RuntimeError("this GPU has no Metal ray tracing support")
        self.tables = build_scene_tables(model, n_envs, max_group=max_group, include_planes=include_planes,
                                         decimate_faces=decimate_faces)
        t = self.tables
        self.G = t.G
        self.tpr = int(np.ceil(np.sqrt(n_envs)))
        extent = float(model.stat.extent)
        self.max_range = max_range
        self.stride = 2.0 * (extent + max_range) + 1.0
        c = self.ctx
        self.vbuf = c.buffer(t.vertices.nbytes, t.vertices, "rt_vertices")
        self.ibuf = c.buffer(t.indices.nbytes, t.indices, "rt_indices")
        self.inst_tab = c.buffer(t.inst_table.nbytes, t.inst_table, "rt_inst_tab")
        self.slot_geom = c.buffer(4 * self.G, np.asarray(t.geoms, np.int32), "rt_slot_geom")
        self.slot_mesh = c.buffer(4 * self.G, np.asarray(t.geom_mesh, np.int32), "rt_slot_mesh")
        mi = np.zeros((len(t.meshes), 4), np.uint32)
        for k, mesh in enumerate(t.meshes):
            mi[k, 0] = mesh["i_off"]; mi[k, 1] = mesh["i_count"]
        self.mesh_info = c.buffer(mi.nbytes, mi, "rt_mesh_info")
        self.mat_buf = c.buffer(t.materials.nbytes, t.materials, "rt_materials")
        self.n_inst = len(t.inst_table)
        self.inst_desc = c.buffer(64 * self.n_inst, label="rt_instance_descriptors")
        self.consts = np.zeros(12, np.uint32)
        self.consts[:4] = (self.n, model.ngeom, self.G, self.tpr)
        self.consts_f = self.consts.view(np.float32)
        self.consts_f[4] = self.stride; self.consts_f[5] = max_range
        self.consts[7] = max(model.nsite, 1)
        self.consts_buf = c.buffer(self.consts.nbytes, self.consts, "rt_consts")
        lib = c.library("rt")
        self.p_refit = c.compute_pipeline(lib, "refit_instances")
        self.p_lidar = c.compute_pipeline(lib, "lidar")
        self.p_depth = c.compute_pipeline(lib, "raycast_depth")
        self._build_primitive_structures()
        self._alloc_instance_structure()
        self.event = self.ctx.event
        self._host_geom_xpos = c.buffer(self.n * model.ngeom * 12, label="rt_host_geom_xpos")
        self._host_geom_xmat = c.buffer(self.n * model.ngeom * 36, label="rt_host_geom_xmat")
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

    def _encode_refit_and_build(self, cb, geom_xpos, gx_off, geom_xmat, gm_off):
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.p_refit)
        ce.setBuffer_offset_atIndex_(self.inst_desc, 0, 0)
        ce.setBuffer_offset_atIndex_(self.inst_tab, 0, 1)
        ce.setBuffer_offset_atIndex_(self.slot_geom, 0, 2)
        ce.setBuffer_offset_atIndex_(self.slot_mesh, 0, 3)
        ce.setBuffer_offset_atIndex_(geom_xpos, gx_off, 4)
        ce.setBuffer_offset_atIndex_(geom_xmat, gm_off, 5)
        ce.setBuffer_offset_atIndex_(self.consts_buf, 0, 6)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n_inst, 1, 1), Metal.MTLSize(64, 1, 1))
        ce.endEncoding()
        ae = cb.accelerationStructureCommandEncoder()
        ae.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(self.inst_as, self.inst_desc_obj, self.inst_scratch, 0)
        ae.endEncoding()

    def _use_resources(self, ce):
        ce.useResource_usage_(self.inst_as, Metal.MTLResourceUsageRead)
        for a in self.prim_as:
            ce.useResource_usage_(a, Metal.MTLResourceUsageRead)

    # -- sensors ------------------------------------------------------------------------------------

    def make_lidar(self, site: str | int, azimuths: np.ndarray, elevations: np.ndarray):
        return Lidar(self, site, azimuths, elevations)

    def make_depth_camera(self, camera: str | int, width: int, height: int):
        return RayDepthCamera(self, camera, width, height)

    # -- frame drivers -------------------------------------------------------------------------------

    def trace(self, sim, sensors, after_value: int | None = None) -> int:
        """GPU path: refit + rebuild from ``sim`` state, then run each sensor's kernel. Waits for the
        sim event, signals the context event; returns the value."""
        import mujoco  # noqa
        if after_value is None:
            after_value = sim.event.value
        gx, gm = wm.buffer_of(sim.d.geom_xpos), wm.buffer_of(sim.d.geom_xmat)
        cb = self.ctx.command_buffer()
        self.ctx.wait_for(cb, sim.event, after_value)
        self._encode_refit_and_build(cb, gx.buffer, gx.offset, gm.buffer, gm.offset)
        for s in sensors:
            s.encode(cb, sim=sim)
        v = self.ctx.signal(cb)
        self.ctx.commit(cb)
        return v

    def trace_host(self, datas, sensors):
        """Host path from ``mujoco.MjData`` lists (tests/tools); synchronizes."""
        n, ng, ns, nc = self.n, self.m.ngeom, max(self.m.nsite, 1), max(self.m.ncam, 1)
        c = self.ctx
        c.write_buffer(self._host_geom_xpos, np.stack([d.geom_xpos for d in datas]).astype(np.float32).reshape(n, ng, 3))
        c.write_buffer(self._host_geom_xmat, np.stack([d.geom_xmat for d in datas]).astype(np.float32).reshape(n, ng, 9))
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
    """Beams given as (azimuth, elevation) pairs in the site frame (radians)."""

    def __init__(self, rt: RayTracer, site, azimuths, elevations, name="lidar"):
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
        n = rt.n
        self._arrays = {
            "range": wp.zeros((n, self.n_beams), dtype=wp.float32, device=rt.device),
            "points": wp.zeros((n, self.n_beams, 3), dtype=wp.float32, device=rt.device),
            "normals": wp.zeros((n, self.n_beams, 3), dtype=wp.float32, device=rt.device),
            "slot": wp.zeros((n, self.n_beams), dtype=wp.int32, device=rt.device),
            "intensity": wp.zeros((n, self.n_beams), dtype=wp.float32, device=rt.device),
        }
        wp.synchronize_device(rt.device)
        self._views = {k: wm.buffer_of(a) for k, a in self._arrays.items()}
        self.out = {k: tb.mps_tensor(a) for k, a in self._arrays.items()}

    def encode(self, cb, sim=None):
        rt = self.rt
        if sim is not None:
            sx, sm = wm.buffer_of(sim.d.site_xpos), wm.buffer_of(sim.d.site_xmat)
            sxb, sxo, smb, smo = sx.buffer, sx.offset, sm.buffer, sm.offset
        else:
            sxb, sxo, smb, smo = rt._host_site_xpos, 0, rt._host_site_xmat, 0
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(rt.p_lidar)
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
        ce.setBuffer_offset_atIndex_(rt.mat_buf, 0, 9)
        ce.setBuffer_offset_atIndex_(rt.inst_desc, 0, 10)
        ce.setBuffer_offset_atIndex_(self.consts_buf, 0, 11)
        for i, k in enumerate(("range", "points", "normals", "slot", "intensity")):
            v = self._views[k]
            ce.setBuffer_offset_atIndex_(v.buffer, v.offset, 12 + i)
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
