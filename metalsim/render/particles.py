"""Particle / granular rendering: render grains as instances in the renderer's instance acceleration structure.

Each simulated particle (MPM sand, Newton's ``State.particle_q``) is drawn as ``grains_per_particle`` small irregular
grains. A grain is an instance of one of a few grain meshes (perturbed icospheres, each its own primitive acceleration
structure), with a hashed offset inside the particle's cube, rotation, mesh and scale (``shaders/particles.metal``),
so the instances are rebuilt on the GPU from the particle buffer every frame and a grain keeps its identity over
time. Per-grain albedo comes from a small palette (most grains the sand colour, a few darker / lighter mineral grains)
with a brightness jitter, evaluated in the path tracer from the grain's index. Design and measurements:
``metalsim/render/PARTICLES.md``.

Users go through ``Tier2Renderer.set_particles`` (and ``Tier0Renderer.set_particles`` for raster cameras).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import Metal


@dataclass
class GrainMaterial:
    """Per-grain appearance. ``palette``: up to 8 (r, g, b, weight) linear base colours picked per grain with the
    given relative weights; ``brightness_jitter``: per-grain relative brightness variation (uniform +-);
    ``roughness`` / ``specular`` / ``metallic``: as the scene materials (``SceneTables.MATERIAL_LAYOUT``); ``model``:
    BRDF model id used when the renderer runs a non-legacy ``material_model`` (4 = OmniPBR)."""
    palette: list = field(default_factory=lambda: [(0.60, 0.49, 0.35, 0.74), (0.76, 0.68, 0.55, 0.14),
                                                   (0.40, 0.30, 0.21, 0.08), (0.14, 0.13, 0.12, 0.04)])
    brightness_jitter: float = 0.18
    roughness: float = 0.85
    specular: float = 0.5
    metallic: float = 0.0
    model: int = 4

    @classmethod
    def uniform(cls, rgb, roughness=1.0, specular=0.0):
        """One colour, no jitter (analytic tests)."""
        return cls(palette=[(*rgb, 1.0)], brightness_jitter=0.0, roughness=roughness, specular=specular)

    def pack(self) -> np.ndarray:
        """GrainMat (shaders/pathtrace.metal): pal[8] (rgb, cumulative weight), p0 (n, jitter), mrse, p1 (model)."""
        if not 1 <= len(self.palette) <= 8:
            raise ValueError("palette: 1 to 8 entries")
        g = np.zeros((11, 4), np.float32)
        w = np.array([p[3] if len(p) > 3 else 1.0 for p in self.palette], np.float64)
        cw = np.cumsum(w) / w.sum()
        for i, p in enumerate(self.palette):
            g[i, :3] = p[:3]; g[i, 3] = cw[i]
        g[len(self.palette) - 1, 3] = 1.0
        g[8] = (len(self.palette), self.brightness_jitter, 0, 0)
        g[9] = (self.metallic, self.roughness, self.specular, 0.0)
        g[10] = (self.model, 0, 0, 0)
        return g


SAND = GrainMaterial()


def grain_meshes(n: int = 4, subdiv: int = 1, irregularity: float = 0.18, elongation: float = 0.25, seed: int = 0):
    """``n`` irregular grain meshes: icospheres (``subdiv`` levels: 1 = 80 triangles) with smooth random radial
    noise and random axis ratios, rescaled to unit volume-equivalent radius. Returns a list of
    (vertices (V, 8) float32 [pos, normal, uv], indices (T*3,) uint32)."""
    rng = np.random.default_rng(seed)
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], np.float64)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
                  [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(subdiv):
        verts = list(v); cache = {}
        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = verts[a] + verts[b]; verts.append(m / np.linalg.norm(m)); cache[k] = len(verts) - 1
            return cache[k]
        nf = []
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v, f = np.array(verts), np.array(nf)
    out = []
    for _ in range(n):
        # smooth radial noise: a sum of a few random lobes, plus random axis ratios
        lobes = rng.normal(size=(6, 3)); lobes /= np.linalg.norm(lobes, axis=1, keepdims=True)
        amp = rng.uniform(-1, 1, 6) * irregularity
        r = 1.0 + (np.clip(v @ lobes.T, 0, None) ** 3 * amp).sum(1)
        axes = 1.0 - rng.uniform(0, elongation, 3); axes[rng.integers(3)] = 1.0
        p = v * r[:, None] * axes
        tri = p[f]
        vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
        p /= (vol / (4 / 3 * np.pi)) ** (1 / 3)
        tri = p[f]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        vn = np.zeros_like(p)
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True)
        uv = np.stack([np.arctan2(v[:, 1], v[:, 0]) / (2 * np.pi) + 0.5, np.arccos(np.clip(v[:, 2], -1, 1)) / np.pi], 1)
        out.append((np.concatenate([p, vn, uv], 1).astype(np.float32), f.astype(np.uint32).ravel()))
    return out


class GrainField:
    """GPU side of one particle set: grain meshes (vertex / index buffers and primitive acceleration structures), the
    particle position buffer and the instance kernel. Owned by a renderer, which places the grain instances after its
    scene instances in one instance acceleration structure."""

    def __init__(self, ctx, n_envs: int, *, grains_per_particle: int = 8, radius: float = 0.002,
                 radius_jitter: float = 0.3, offset: float = 0.005, n_meshes: int = 4, subdiv: int = 1,
                 material: GrainMaterial | None = None, seed: int = 0, cull_z: float = -np.inf):
        self.ctx, self.n_envs = ctx, n_envs
        self.k = int(grains_per_particle)
        self.radius, self.radius_jitter, self.offset = float(radius), float(radius_jitter), float(offset)
        self.seed, self.cull_z = int(seed), float(cull_z)
        self.material = material or SAND
        meshes = grain_meshes(n_meshes, subdiv, seed=seed)
        verts = np.concatenate([m[0] for m in meshes])
        idx, info, v0, i0 = [], np.zeros((len(meshes), 4), np.uint32), 0, 0
        for k, (vv, ii) in enumerate(meshes):
            idx.append(ii + v0); info[k, :2] = (i0, len(ii)); v0 += len(vv); i0 += len(ii)
        idx = np.concatenate(idx).astype(np.uint32)
        self.n_meshes = len(meshes)
        self.vbuf = ctx.buffer(verts.nbytes, verts, "grain_vertices")
        self.ibuf = ctx.buffer(idx.nbytes, idx, "grain_indices")
        self.mesh_info = ctx.buffer(info.nbytes, info, "grain_mesh_info")
        self.mat_buf = ctx.buffer(176, self.material.pack(), "grain_material")
        self._build_meshes(info)
        self.pipe = ctx.compute_pipeline(ctx.library("particles"), "grain_instances")
        self.n_particles = 0
        self._pos_buf, self._pos_off, self._own_pos = None, 0, None

    def _build_meshes(self, info):
        c = self.ctx
        descs = []
        for i_off, i_count, _, _ in info:
            g = Metal.MTLAccelerationStructureTriangleGeometryDescriptor.new()
            g.setVertexBuffer_(self.vbuf); g.setVertexBufferOffset_(0); g.setVertexStride_(32)
            g.setVertexFormat_(Metal.MTLAttributeFormatFloat3)
            g.setIndexBuffer_(self.ibuf); g.setIndexBufferOffset_(int(i_off) * 4)
            g.setIndexType_(Metal.MTLIndexTypeUInt32); g.setTriangleCount_(int(i_count) // 3); g.setOpaque_(True)
            d = Metal.MTLPrimitiveAccelerationStructureDescriptor.new()
            d.setGeometryDescriptors_([g])
            descs.append(d)
        sizes = [c.device.accelerationStructureSizesWithDescriptor_(d) for d in descs]
        scratch = c.buffer(max(s.buildScratchBufferSize for s in sizes), label="grain_scratch")
        cb = c.command_buffer(); enc = cb.accelerationStructureCommandEncoder()
        self.prim_as = []
        for d, s in zip(descs, sizes):
            a = c.device.newAccelerationStructureWithSize_(s.accelerationStructureSize)
            enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(a, d, scratch, 0)
            self.prim_as.append(a)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if cb.error() is not None:
            raise RuntimeError(f"grain acceleration structure build failed: {cb.error()}")

    @property
    def n_grains(self) -> int:
        return self.n_envs * self.n_particles * self.k

    def set_material(self, material: GrainMaterial):
        self.material = material
        self.ctx.write_buffer(self.mat_buf, material.pack())

    def set_positions(self, positions):
        """``positions``: (n_envs, P, 3) or (P, 3) (one env, or the same particles in every env) as numpy (uploaded
        into a shared buffer), or a Warp array on the Metal device (bound directly: the GPU path, no host copy; it must
        stay alive and hold n_envs * P * 3 float32). Returns the particle count per env."""
        import warp as wp
        if isinstance(positions, wp.array):
            from metalsim.interop import warp_metal as wm
            n = positions.size * (3 if positions.dtype == wp.vec3 else 1) // 3
            if n % self.n_envs:
                raise ValueError(f"{n} positions do not split into {self.n_envs} envs")
            v = wm.buffer_of(positions)
            self._pos_buf, self._pos_off, self._own_pos = v.buffer, v.offset, positions
            self.n_particles = n // self.n_envs
            return self.n_particles
        p = np.ascontiguousarray(np.asarray(positions, np.float32))
        if p.ndim == 2:
            p = np.broadcast_to(p, (self.n_envs,) + p.shape)
        if p.shape[0] != self.n_envs or p.shape[-1] != 3:
            raise ValueError(f"positions {p.shape}: expected (n_envs={self.n_envs}, P, 3) or (P, 3)")
        p = np.ascontiguousarray(p)
        if self._own_pos is None or isinstance(self._own_pos, (wp.array,)) or self._own_pos.length() < p.nbytes:
            self._own_pos = self.ctx.buffer(p.nbytes, label="grain_particles")
        self.ctx.write_buffer(self._own_pos, p)
        self._pos_buf, self._pos_off = self._own_pos, 0
        self.n_particles = p.shape[1]
        return self.n_particles

    def consts(self, as_base: int, inst_base: int, tiles_per_row: int, stride: float) -> np.ndarray:
        c = np.zeros(16, np.uint32)
        c[:8] = (self.n_envs, self.n_particles, self.k, self.n_meshes, as_base, tiles_per_row, self.seed, inst_base)
        cf = c.view(np.float32)
        cf[8:13] = (stride, self.radius, self.radius_jitter, self.offset, self.cull_z)
        return c

    def encode_instances(self, cb, inst_desc, as_base: int, inst_base: int, tiles_per_row: int, stride: float):
        if self.n_grains == 0:
            return
        k = self.consts(as_base, inst_base, tiles_per_row, stride)
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.pipe)
        ce.setBuffer_offset_atIndex_(inst_desc, 0, 0)
        ce.setBuffer_offset_atIndex_(self._pos_buf, self._pos_off, 1)
        ce.setBytes_length_atIndex_(k.tobytes(), k.nbytes, 2)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n_grains, 1, 1), Metal.MTLSize(256, 1, 1))
        ce.endEncoding()

    def use_resources(self, ce):
        for a in self.prim_as:
            ce.useResource_usage_(a, Metal.MTLResourceUsageRead)


class InstanceStructure:
    """The scene's instances followed by grain instances in one instance acceleration structure, rebuilt every frame
    (scene refit kernel, grain kernel, build). Reallocated when the grain count grows."""

    def __init__(self, rt, field: GrainField):
        self.rt, self.field = rt, field
        self.capacity = -1
        self.n_total = -1

    def ensure(self):
        rt, f = self.rt, self.field
        n_total = rt.n_inst + f.n_grains
        if n_total == self.n_total:
            return
        c = rt.ctx
        if n_total > self.capacity:
            self.capacity = n_total
            self.inst_desc = c.buffer(64 * n_total, label="pt_instances_with_grains")
        d = Metal.MTLInstanceAccelerationStructureDescriptor.new()
        d.setInstancedAccelerationStructures_(list(rt.prim_as) + list(f.prim_as))
        d.setInstanceCount_(n_total)
        d.setInstanceDescriptorBuffer_(self.inst_desc); d.setInstanceDescriptorBufferOffset_(0)
        d.setInstanceDescriptorStride_(64)
        d.setInstanceDescriptorType_(Metal.MTLAccelerationStructureInstanceDescriptorTypeDefault)
        s = c.device.accelerationStructureSizesWithDescriptor_(d)
        self.inst_as = c.device.newAccelerationStructureWithSize_(s.accelerationStructureSize)
        if self.inst_as is None:
            raise RuntimeError(f"instance acceleration structure of {n_total} instances "
                               f"({s.accelerationStructureSize / 2**20:.0f} MiB) could not be allocated")
        self.scratch = c.buffer(max(int(s.buildScratchBufferSize), 16), label="pt_inst_scratch_grains")
        self.desc, self.n_total = d, n_total
        self.as_bytes = int(s.accelerationStructureSize)

    def encode(self, cb, geom_xpos, gx_off, geom_xmat, gm_off):
        """Scene refit into the first rt.n_inst descriptors, grain instances after them, one build."""
        self.ensure()
        rt = self.rt
        rt._encode_refit(cb, geom_xpos, gx_off, geom_xmat, gm_off, inst_desc=self.inst_desc)
        self.field.encode_instances(cb, self.inst_desc, len(rt.prim_as), rt.n_inst, rt.tpr, rt.stride)
        ae = cb.accelerationStructureCommandEncoder()
        ae.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(self.inst_as, self.desc, self.scratch, 0)
        ae.endEncoding()

    def use_resources(self, ce):
        ce.useResource_usage_(self.inst_as, Metal.MTLResourceUsageRead)
        for a in self.rt.prim_as:
            ce.useResource_usage_(a, Metal.MTLResourceUsageRead)
        self.field.use_resources(ce)
