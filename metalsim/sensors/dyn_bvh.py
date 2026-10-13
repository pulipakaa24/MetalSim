"""Dynamic objects in the ray tracers as one bounding box per env ("sbvh" layout of ``RayTracer``).

The hardware instance structure holds the rigid instances plus one instance of a single bounding-box structure with
one box per env (the env's dynamic triangles at its grid offset). A ray that enters env e's box traverses e's own BVH
in software (``shaders/dyn_bvh.h``, through ``intersection_query``), in the env's frame as the per-env hardware
structures were. The BVH topology is shared by every env: built once on the host (median splits over the triangle
centroids of env 0 at the first frame, leaves of up to ``LEAF`` triangles); each frame one compute pass refits every
env's node bounds (and copies its triangles in leaf order) level by level from the regenerated vertex records, writes
the boxes, and the box structure (one per scene, ``n_envs`` primitives) is rebuilt. The number of hardware structures
does not grow with the env count.

As the objects deform, refit bounds loosen. Every ``RayTracer.dyn_rebuild_every`` frames the leaf assignment is
rebuilt from env 0's current triangles on a host thread (median splits give the same tree shape for the same triangle
count, so only the leaf order changes) and swapped in through a second order buffer and argument struct, so frames in
flight keep a consistent set.

Hits report the box instance (identity transform; inst_tab x = D objects, y = first dynamic slot), the object as the
geometry, the triangle within the object's env mesh as the primitive, and the env: slot, mesh, normal, uv and
barycentrics resolve exactly as for the per-env structures.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import Metal

LEAF = 4
MAX_DEPTH = 30       # the traversal stack holds 32 entries


def build_topology(tri_pos: np.ndarray, leaf: int = LEAF):
    """Median-split BVH over triangles (T, 3, 3): returns topo (N, 4) uint32 (left, right, first, count; count 0 =
    inner), order (T,) triangle indices in leaf order, depth (N,)."""
    T = len(tri_pos)
    cent = tri_pos.mean(axis=1)
    topo, depth = [], []
    order = np.empty(T, np.int64)
    filled = 0
    topo.append([0, 0, 0, 0]); depth.append(0)
    stack = [(0, np.arange(T), 0)]
    while stack:
        node, idx, dep = stack.pop()
        if len(idx) <= leaf or dep >= MAX_DEPTH:
            topo[node] = [0, 0, filled, len(idx)]
            order[filled:filled + len(idx)] = idx
            filled += len(idx)
            continue
        c = cent[idx]
        ext = c.max(axis=0) - c.min(axis=0)
        ax = int(np.argmax(ext))
        h = len(idx) // 2
        part = np.argpartition(c[:, ax], h)
        left, right = idx[part[:h]], idx[part[h:]]
        lc, rc = len(topo), len(topo) + 1
        topo.append([0, 0, 0, 0]); depth.append(dep + 1)
        topo.append([0, 0, 0, 0]); depth.append(dep + 1)
        topo[node] = [lc, rc, 0, 0]
        stack.append((rc, right, dep + 1))
        stack.append((lc, left, dep + 1))
    assert filled == T
    return np.asarray(topo, np.uint32), order, np.asarray(depth, np.int64)


class DynBVH:
    def __init__(self, rt):
        self.rt = rt
        c = self.ctx = rt.ctx
        self.n = rt.n
        self.D = rt.n_dyn_obj
        if self.D >= 256:
            raise ValueError("sbvh layout: at most 255 dynamic objects")
        self.mesh_base = int(rt.dyn_mesh_index[0, 0])
        assert np.array_equal(rt.dyn_mesh_index, self.mesh_base + np.arange(self.n * self.D).reshape(self.n, self.D))
        self.tris_per_obj = [int(rt.meshes_all[self.mesh_base + k]["i_count"]) // 3 for k in range(self.D)]
        lib = c.library("dyn_bvh")
        self.p_refit = c.compute_pipeline(lib, "dyn_bvh_refit")
        self.p_boxes = c.compute_pipeline(lib, "dyn_bvh_boxes")
        # one box per env, one structure (the rt shaders intersect within Metal's standard limits: as_limits)
        from metalsim.render import as_limits
        as_limits.check("primitives", self.n, "dynamic objects' box structure")
        self.boxes = c.buffer(24 * self.n, label="dyn_bvh_boxes")
        g = Metal.MTLAccelerationStructureBoundingBoxGeometryDescriptor.new()
        g.setBoundingBoxBuffer_(self.boxes); g.setBoundingBoxBufferOffset_(0)
        g.setBoundingBoxStride_(24); g.setBoundingBoxCount_(self.n)
        d = Metal.MTLPrimitiveAccelerationStructureDescriptor.new()
        d.setGeometryDescriptors_([g])
        d.setUsage_(Metal.MTLAccelerationStructureUsagePreferFastBuild)
        self.box_desc = d
        s = c.device.accelerationStructureSizesWithDescriptor_(d)
        self.box_as = c.device.newAccelerationStructureWithSize_(s.accelerationStructureSize)
        self.box_scratch = c.buffer(max(int(s.buildScratchBufferSize), 16), label="dyn_bvh_box_scratch")
        self.topo = None
        self.argss = [c.buffer(176, label="dyn_bvh_args0"), c.buffer(176, label="dyn_bvh_args1")]
        self.cur = 0
        self._pool, self._pending, self._frames, self._last_retopo = None, None, 0, 0
        self.retopologies = 0

    @property
    def args(self):
        return self.argss[self.cur]

    @property
    def order(self):
        return self.orders[self.cur]
    # -- topology (first frame) -----------------------------------------------------------------------

    def _ensure_topology(self, cb, src):
        if self.topo is not None:
            return
        rt, c = self.rt, self.ctx
        cb0 = c.command_buffer()           # env 0's vertices of this frame, synchronously (once)
        for ev, val in c.waits_of(cb):
            cb0.encodeWaitForEvent_value_(ev, val)
        rt.dyn.encode(cb0, src)
        cb0.commit(); cb0.waitUntilCompleted()
        verts = c.buffer_array(rt.vbuf, np.float32, (rt.vbuf.length() // 32, 8))
        inds = c.buffer_array(rt.ibuf, np.uint32, (rt.ibuf.length() // 4,))
        tri_pos, ent = [], []
        for k in range(self.D):
            mesh = rt.meshes_all[self.mesh_base + k]
            ix = inds[mesh["i_off"]: mesh["i_off"] + mesh["i_count"]].reshape(-1, 3)
            tri_pos.append(verts[ix, :3].astype(np.float64))
            ent.append((k << 24) | np.arange(len(ix), dtype=np.int64))
        tri_pos = np.concatenate(tri_pos)
        ent = np.concatenate(ent)
        topo, order, depth = build_topology(tri_pos)
        self.n_nodes = len(topo)
        self.topo_np = topo
        self.topo = c.buffer(topo.nbytes, topo, "dyn_bvh_topo")
        self.ents = ent[order].astype(np.uint32)
        self.orders = [c.buffer(4 * len(order), self.ents, "dyn_bvh_order0"), c.buffer(4 * len(order), self.ents, "dyn_bvh_order1")]
        lv = []
        self.levels = []
        off = 0
        for dp in range(int(depth.max()), -1, -1):      # deepest first
            nodes = np.nonzero(depth == dp)[0].astype(np.uint32)
            lv.append(nodes)
            self.levels.append((off, len(nodes)))
            off += len(nodes)
        lvl = np.concatenate(lv)
        self.level_nodes = c.buffer(lvl.nbytes, lvl, "dyn_bvh_levels")
        self.bounds = c.device.newBufferWithLength_options_(max(32 * self.n_nodes * self.n, 16), Metal.MTLResourceStorageModePrivate)
        if self.bounds is None:
            raise RuntimeError(f"dyn_bvh: {32 * self.n_nodes * self.n / 2**20:.0f} MiB of node bounds could not be allocated")
        self.bounds.setLabel_("dyn_bvh_bounds")
        self.n_tris = int(len(order))
        # shared: env 0's triangles are read back (unsynchronized) for the leaf rebuilds
        self.tris = c.device.newBufferWithLength_options_(max(36 * self.n_tris * self.n, 16), Metal.MTLResourceStorageModeShared)
        if self.tris is None:
            raise RuntimeError(f"dyn_bvh: {36 * self.n_tris * self.n / 2**20:.0f} MiB of triangles could not be allocated")
        self.tris.setLabel_("dyn_bvh_tris")
        for i in (0, 1):
            self._write_args(i)
        rt._grid_listeners.append(lambda: [self._write_args(i) for i in (0, 1)])
        self.stats = {"nodes": self.n_nodes, "triangles": int(len(order)), "depth": int(depth.max()),
                      "bounds_mb": 32 * self.n_nodes * self.n / 2**20, "tris_mb": 36 * self.n_tris * self.n / 2**20}

    def _write_args(self, i):
        rt = self.rt
        a = np.zeros(44, np.uint32)       # DynBVHArgs: 7 pointers, 12 words, the EnvGrid at byte 96
        p = a.view(np.uint64)
        p[0], p[1], p[2] = rt.vbuf.gpuAddress(), rt.ibuf.gpuAddress(), rt.mesh_info.gpuAddress()
        p[3], p[4], p[5] = self.topo.gpuAddress(), self.orders[i].gpuAddress(), self.bounds.gpuAddress()
        p[6] = self.tris.gpuAddress()
        a[14:18] = (self.n, self.n_nodes, self.D, self.mesh_base)
        a[18] = rt.n_inst_static
        a[19] = self.n_tris
        a[24:44] = rt.grid.words()
        self.ctx.write_buffer(self.argss[i], a)

    # -- leaf rebuilds ----------------------------------------------------------------------------------

    def _maybe_retopology(self):
        every = int(self.rt.dyn_rebuild_every)
        if self._pending is not None and self._pending.done():
            ents_snap, (topo, order, _) = self._pending.result()
            self._pending = None
            if np.array_equal(topo, self.topo_np):          # same shape (median splits): swap the leaf order in
                nxt = 1 - self.cur
                self.ents = ents_snap[order]
                self.ctx.write_buffer(self.orders[nxt], self.ents)
                self.cur = nxt
                self.retopologies += 1
        if every <= 0 or self._pending is not None or self._frames - self._last_retopo < every:
            return
        self._last_retopo = self._frames
        tri = np.array(self.ctx.buffer_array(self.tris, np.float32, (self.n_tris, 9)), np.float64).reshape(-1, 3, 3)
        ents = self.ents.copy()
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dyn_bvh")
        self._pending = self._pool.submit(lambda: (ents, build_topology(tri)))

    # -- per frame ------------------------------------------------------------------------------------

    def _refit_args(self, level_off=0, level_count=0):
        a = np.zeros(28, np.uint32)       # RefitArgs: 8 words, the EnvGrid at byte 32
        a[:7] = (self.n, self.n_nodes, self.D, self.mesh_base, level_off, level_count, self.n_tris)
        a[8:28] = self.rt.grid.words()
        return a.tobytes()

    def encode(self, cb, src, rebuild=False):
        """Refit every env's BVH from the records generated earlier in ``cb``, then the boxes and the box structure."""
        self._ensure_topology(cb, src)
        self._maybe_retopology()
        self._frames += 1
        rt = self.rt
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.p_refit)
        for i, b in enumerate((rt.vbuf, rt.ibuf, rt.mesh_info, self.topo, self.order, self.bounds, self.level_nodes)):
            ce.setBuffer_offset_atIndex_(b, 0, i)
        ce.setBuffer_offset_atIndex_(self.tris, 0, 8)
        for off, cnt in self.levels:
            a = self._refit_args(off, cnt)
            ce.setBytes_length_atIndex_(a, len(a), 7)
            ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(cnt, self.n, 1), Metal.MTLSize(min(cnt, 64), 1, 1))
        ce.setComputePipelineState_(self.p_boxes)
        ce.setBuffer_offset_atIndex_(self.bounds, 0, 0)
        ce.setBuffer_offset_atIndex_(self.boxes, 0, 1)
        a = self._refit_args()
        ce.setBytes_length_atIndex_(a, len(a), 2)
        ce.setBuffer_offset_atIndex_(rt.grid_flag, 0, 3)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n, 1, 1), Metal.MTLSize(64, 1, 1))
        ce.endEncoding()
        ae = cb.accelerationStructureCommandEncoder()
        ae.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(self.box_as, self.box_desc, self.box_scratch, 0)
        ae.endEncoding()

    def resources(self):
        """Buffers the ray kernels read through the argument struct's pointers (residency)."""
        rt = self.rt
        return [self.args, rt.vbuf, rt.ibuf, rt.mesh_info, self.topo, self.order, self.bounds, self.tris]

    def __del__(self):
        if getattr(self, "_pool", None) is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
