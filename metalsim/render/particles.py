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
                 material: GrainMaterial | None = None, seed: int = 0, cull_z: float = -np.inf, shape: dict | None = None):
        """``shape``: grain_meshes keyword arguments (irregularity, elongation; ``subdiv`` / ``n_meshes`` as given)."""
        self.ctx, self.n_envs = ctx, n_envs
        self.k = int(grains_per_particle)
        self.radius, self.radius_jitter, self.offset = float(radius), float(radius_jitter), float(offset)
        self.seed, self.cull_z = int(seed), float(cull_z)
        self.material = material or SAND
        sh = dict(shape or {})
        meshes = grain_meshes(n_meshes, sh.pop("subdiv", subdiv), seed=seed, **sh)
        verts = np.concatenate([m[0] for m in meshes])
        idx, info, v0, i0 = [], np.zeros((len(meshes), 4), np.uint32), 0, 0
        for k, (vv, ii) in enumerate(meshes):
            idx.append(ii + v0); info[k, :2] = (i0, len(ii)); v0 += len(vv); i0 += len(ii)
        idx = np.concatenate(idx).astype(np.uint32)
        self.n_meshes = len(meshes)
        self.vbuf = ctx.buffer(verts.nbytes, verts, "grain_vertices")
        self.ibuf = ctx.buffer(idx.nbytes, idx, "grain_indices")
        self.mesh_info = ctx.buffer(info.nbytes, info, "grain_mesh_info")
        if len(meshes) > 8:
            raise ValueError("at most 8 grain meshes")
        # GrainMat (shaders/pathtrace.metal): the material (176 bytes) followed by 8 grain mesh index ranges
        mb = np.zeros(304, np.uint8)
        mb[:176] = self.material.pack().view(np.uint8).ravel()
        mi = np.zeros((8, 4), np.uint32); mi[:len(meshes)] = info
        mb[176:] = mi.view(np.uint8).ravel()
        self.mat_buf = ctx.buffer(mb.nbytes, mb, "grain_material")
        self._build_meshes(info)
        self.pipe = ctx.compute_pipeline(ctx.library("particles"), "grain_instances")
        self.n_particles = 0
        self._pos_buf, self._pos_off, self._own_pos = None, 0, None
        self._ids_buf, self.has_ids = ctx.buffer(16, label="grain_ids_placeholder"), False
        self.keys = ctx.buffer(16, label="grain_keys")     # per-grain identity key, written with the instances

    def _build_meshes(self, info):
        from metalsim.render import as_limits
        c = self.ctx
        descs = []
        for i_off, i_count, _, _ in info:
            as_limits.check("primitives", int(i_count) // 3, "grain mesh")
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

    def set_ids(self, ids):
        """Stable particle ids (n_envs, P) or (P,) uint32: a grain's shape, offset and colour hash the id instead of the
        particle's index, so a caller can drop / reorder particles (surface culling, compaction) without grains
        changing. None: index = identity."""
        if ids is None:
            self.has_ids = False
            return
        i = np.asarray(ids, np.uint32)
        if i.ndim == 1:
            i = np.broadcast_to(i, (self.n_envs, i.shape[0]))
        i = np.ascontiguousarray(i)
        if self._ids_buf.length() < i.nbytes:
            self._ids_buf = self.ctx.buffer(i.nbytes, label="grain_ids")
        self.ctx.write_buffer(self._ids_buf, i)
        self.has_ids = True

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

    def consts(self, as_base: int, inst_base: int, tiles_per_row: int, stride: float, user_ids: bool = False) -> np.ndarray:
        c = np.zeros(16, np.uint32)
        c[:8] = (self.n_envs, self.n_particles, self.k, self.n_meshes, as_base, tiles_per_row, self.seed, inst_base)
        cf = c.view(np.float32)
        cf[8:13] = (stride, self.radius, self.radius_jitter, self.offset, self.cull_z)
        c[13] = 1 if self.has_ids else 0
        c[14] = 1 if user_ids else 0
        return c

    def encode_instances(self, cb, inst_desc, as_base: int, inst_base: int, tiles_per_row: int, stride: float,
                         keys=None, keys_off: int = 0, user_ids: bool = False):
        """Grain i's instance descriptor to ``inst_desc[inst_base + i]`` (mesh ``as_base + k`` of the structure's
        instanced list) and its identity key to ``keys`` at byte ``keys_off`` (default: this field's own key buffer).
        ``user_ids``: UserID descriptors (68 bytes) with user id 0x80000000 | i (two-level structures)."""
        if self.n_grains == 0:
            return
        k = self.consts(as_base, inst_base, tiles_per_row, stride, user_ids)
        if keys is None:
            if self.keys.length() < 4 * self.n_grains:
                self.keys = self.ctx.buffer(4 * self.n_grains, label="grain_keys")
            keys, keys_off = self.keys, 0
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.pipe)
        ce.setBuffer_offset_atIndex_(inst_desc, 0, 0)
        ce.setBuffer_offset_atIndex_(self._pos_buf, self._pos_off, 1)
        ce.setBytes_length_atIndex_(k.tobytes(), k.nbytes, 2)
        ce.setBuffer_offset_atIndex_(self._ids_buf, 0, 3)
        ce.setBuffer_offset_atIndex_(keys, keys_off, 4)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(self.n_grains, 1, 1), Metal.MTLSize(256, 1, 1))
        ce.endEncoding()

    def use_resources(self, ce):
        for a in self.prim_as:
            ce.useResource_usage_(a, Metal.MTLResourceUsageRead)


def instance_descriptor(prim_list, count, desc_buf, offset=0, label="instances", extended_ok=False, user_ids=False):
    """The descriptor of an instance acceleration structure of ``count`` instances over ``prim_list`` (descriptors at
    byte ``offset`` of ``desc_buf``). Past Metal's standard 2^24 instances it gets the extended limits when the caller
    intersects it with the ``extended_limits`` tag (``extended_ok``), else ValueError (metalsim/render/as_limits.py).
    Returns (descriptor, extended)."""
    from metalsim.render import as_limits
    extended = as_limits.check("instances", count, label, extended_ok=extended_ok)
    d = Metal.MTLInstanceAccelerationStructureDescriptor.new()
    d.setInstancedAccelerationStructures_(list(prim_list))
    d.setInstanceCount_(int(count))
    d.setInstanceDescriptorBuffer_(desc_buf); d.setInstanceDescriptorBufferOffset_(int(offset))
    d.setInstanceDescriptorStride_(68 if user_ids else 64)
    d.setInstanceDescriptorType_(Metal.MTLAccelerationStructureInstanceDescriptorTypeUserID if user_ids
                                 else Metal.MTLAccelerationStructureInstanceDescriptorTypeDefault)
    as_limits.apply(d, extended)
    return d, extended


def _instance_structure(ctx, prim_list, count, desc_buf, label, offset=0, extended_ok=False, user_ids=False):
    """(descriptor, structure, scratch, bytes, extended) of an instance acceleration structure over ``prim_list``."""
    d, extended = instance_descriptor(prim_list, count, desc_buf, offset, label, extended_ok, user_ids)
    s = ctx.device.accelerationStructureSizesWithDescriptor_(d)
    a = ctx.device.newAccelerationStructureWithSize_(s.accelerationStructureSize)
    if a is None:
        raise RuntimeError(f"{label}: instance acceleration structure of {count} instances "
                           f"({s.accelerationStructureSize / 2**20:.0f} MiB) could not be allocated")
    scratch = ctx.buffer(max(int(s.buildScratchBufferSize), 16), label=f"{label}_scratch")
    return d, a, scratch, int(s.accelerationStructureSize), extended


def _identity_descriptor(as_index: int) -> np.ndarray:
    """An identity, opaque, all-rays instance of ``as_index`` (MTLAccelerationStructureInstanceDescriptor, 64 bytes)."""
    top = np.zeros(16, np.float32)
    top[0] = top[4] = top[8] = 1.0
    tu = top.view(np.uint32)
    tu[12] = 4; tu[13] = 0xFFFFFFFF; tu[14] = 0; tu[15] = as_index
    return top


class InstanceStructure:
    """The renderer's instance acceleration structure with grains, rebuilt every frame from the physics state and the
    particle buffer, in one of two layouts:

    * ``"two_level"`` (default): the grains are their own instance acceleration structures (one instance per grain),
      in chunks of at most ``chunk`` grains (Metal's standard limit is 2^24 instances per structure), and each chunk
      is one identity instance of the scene's top-level structure next to the scene geometry. The path tracer
      intersects through two instance levels (``max_levels<2>``). Metal reports only the innermost instance of a hit
      (``instance_count`` stays 1; scripts/particle_demo/multilevel_probe2.py), so the grain structures use UserID
      descriptors whose user id is 0x80000000 | grain index, and the scene's Default descriptors read user id 0. Every
      structure stays within the standard limits, and the top level stays as small as the scene.
    * ``"flat"``: the scene's instances followed by one top-level instance per grain, one build; past 2^24 instances it
      is built with Metal's extended limits and intersected with the ``extended_limits`` tag (FLAG_GRAINS_EXT).

    Before 2026-10-13 the flat layout was the only one and set no usage flags, so 25.1 M grain instances were built
    outside Metal's supported range; on macOS 26 that build (with large ground planes) never finished and the GPU work
    outlived its process (PARTICLES.md).

    Grain descriptors and identity keys: two-level, one buffer of the 68-byte UserID descriptors followed by the keys
    (the path tracer's buffer 29); flat, the descriptors follow the scene's in the top-level buffer and the keys are the
    field's own.
    """

    MODES = ("two_level", "flat")
    CHUNK = 1 << 24

    def __init__(self, rt, field: GrainField, mode: str = "two_level", chunk: int | None = None):
        if mode not in self.MODES:
            raise ValueError(f"grain structure {mode!r}: one of {self.MODES}")
        self.rt, self.field, self.mode = rt, field, mode
        self.chunk = int(chunk or self.CHUNK)
        if not 1 <= self.chunk <= self.CHUNK:
            raise ValueError(f"chunk {self.chunk}: 1 to {self.CHUNK}")
        self.capacity = -1
        self.n_grains = -1
        self.extended = False

    @property
    def n_chunks(self) -> int:
        return max(1, -(-self.field.n_grains // self.chunk))

    def ensure(self):
        rt, f = self.rt, self.field
        G = f.n_grains
        if G == self.n_grains:
            return
        c = rt.ctx
        if self.mode == "flat":
            n_total = rt.n_inst + G
            if n_total > self.capacity:
                self.capacity = n_total
                self.inst_desc = c.buffer(64 * n_total, label="pt_instances_with_grains")
            self.desc, self.inst_as, self.scratch, self.as_bytes, self.extended = _instance_structure(
                c, list(rt.prim_as) + list(f.prim_as), n_total, self.inst_desc, "pt_flat_grains", extended_ok=True)
            self.grain_as_bytes = 0
        else:
            if G > self.capacity:
                self.capacity = G
                self.grain_buf = c.buffer(72 * G, label="pt_grain_instances_and_keys")   # UserID descriptors, then keys
            nc = self.n_chunks
            self.chunks = []        # (descriptor, structure, scratch, bytes) per chunk of <= chunk grains
            for k in range(nc):
                n = min(self.chunk, G - k * self.chunk)
                d, a, sc, nb, _ = _instance_structure(c, f.prim_as, n, self.grain_buf, f"pt_grains_{k}",
                                                      offset=68 * k * self.chunk, user_ids=True)
                self.chunks.append((d, a, sc, nb))
            self.grain_as_bytes = sum(ch[3] for ch in self.chunks)
            self.inst_desc = c.buffer(64 * (rt.n_inst + nc), label="pt_instances_scene_and_grains")
            for k in range(nc):     # chunk k: identity instance after the scene's, instancing structure n_prim + k
                c.write_buffer(self.inst_desc, _identity_descriptor(len(rt.prim_as) + k), offset=64 * (rt.n_inst + k))
            self.desc, self.inst_as, self.scratch, top_bytes, _ = _instance_structure(
                c, list(rt.prim_as) + [ch[1] for ch in self.chunks], rt.n_inst + nc, self.inst_desc, "pt_scene_and_grains")
            self.as_bytes = top_bytes + self.grain_as_bytes
        self.n_grains = G

    @property
    def n_total(self) -> int:
        """Instances in the structure(s): scene instances + grains (+ one top-level instance per grain chunk)."""
        return self.rt.n_inst + self.n_grains + (len(self.chunks) if self.mode == "two_level" else 0)

    def grain_binding(self):
        """(buffer, byte offset) for the path tracer's buffer 29."""
        if self.mode == "flat":
            return self.field.keys, 0
        return self.grain_buf, 0

    def encode(self, cb, geom_xpos, gx_off, geom_xmat, gm_off):
        """Scene refit, grain instances, the build(s)."""
        self.ensure()
        rt, f = self.rt, self.field
        rt._encode_refit(cb, geom_xpos, gx_off, geom_xmat, gm_off, inst_desc=self.inst_desc)
        if self.mode == "flat":
            f.encode_instances(cb, self.inst_desc, len(rt.prim_as), rt.n_inst, rt.tpr, rt.stride)
        else:
            f.encode_instances(cb, self.grain_buf, 0, 0, rt.tpr, rt.stride, keys=self.grain_buf, keys_off=68 * f.n_grains,
                               user_ids=True)
            ae = cb.accelerationStructureCommandEncoder()       # the grain chunks first (the top level instances them)
            for d, a, sc, _ in self.chunks:
                ae.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(a, d, sc, 0)
            ae.endEncoding()
        ae = cb.accelerationStructureCommandEncoder()
        ae.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(self.inst_as, self.desc, self.scratch, 0)
        ae.endEncoding()

    def use_resources(self, ce):
        ce.useResource_usage_(self.inst_as, Metal.MTLResourceUsageRead)
        if self.mode == "two_level":
            for ch in self.chunks:
                ce.useResource_usage_(ch[1], Metal.MTLResourceUsageRead)
        for a in self.rt.prim_as:
            ce.useResource_usage_(a, Metal.MTLResourceUsageRead)
        self.field.use_resources(ce)


class GrainRaster:
    """Tier-0 (raster) drawing of a GrainField: grain instances written by the same kernel (no env offset: every env
    renders in its own tile), drawn as one instanced draw in tier 0's shadow and geometry passes with
    ``shaders/particles_tier0.metal`` compiled appended to ``tier0.metal`` (tier 0's structs, shading and attachments).
    Grains get segmentation id ``n_slots + 1`` (slot mode), geom ``ngeom`` / body ``nbody`` (geom / body modes): the
    renderer's semantic table is extended by that one entry."""

    def __init__(self, r0, field: GrainField):
        from metalsim.render.metal_context import SHADER_DIR, _inline_includes
        self.r0, self.field = r0, field
        c = r0.ctx
        src = _inline_includes((SHADER_DIR / "tier0.metal").read_text() + "\n" + (SHADER_DIR / "particles_tier0.metal").read_text())
        opts = Metal.MTLCompileOptions.new(); opts.setFastMathEnabled_(True)
        lib, err = c.device.newLibraryWithSource_options_error_(src, opts, None)
        if lib is None:
            raise RuntimeError(f"Metal compile failed for particles_tier0: {err}")
        d = Metal.MTLRenderPipelineDescriptor.new()
        d.setVertexFunction_(lib.newFunctionWithName_("grain_vs")); d.setFragmentFunction_(lib.newFunctionWithName_("grain_fs"))
        ca = d.colorAttachments()
        ca.objectAtIndexedSubscript_(0).setPixelFormat_(Metal.MTLPixelFormatRGBA8Unorm)
        ca.objectAtIndexedSubscript_(1).setPixelFormat_(Metal.MTLPixelFormatR32Uint)
        ca.objectAtIndexedSubscript_(2).setPixelFormat_(Metal.MTLPixelFormatRGBA16Float)
        d.setDepthAttachmentPixelFormat_(Metal.MTLPixelFormatDepth32Float)
        self.pipe = c.render_pipeline(d)
        d = Metal.MTLRenderPipelineDescriptor.new()
        d.setVertexFunction_(lib.newFunctionWithName_("grain_shadow_vs"))
        d.setDepthAttachmentPixelFormat_(Metal.MTLPixelFormatDepth32Float)
        self.shadow_pipe = c.render_pipeline(d)
        self.inst_desc, self.capacity = None, 0
        # semantic table + 1 entry for the grains (untile reads sem[seg - 1])
        sem = c.buffer_array(r0.sem_buf, np.int32, (r0.G, 4)).copy()
        m = r0.m
        sem = np.concatenate([sem, np.array([[m.ngeom, m.nbody, m.nbody, 0]], np.int32)])
        r0.sem_buf = c.buffer(sem.nbytes, sem, "sem_with_grains")
        self.rebind(field)

    def rebind(self, field: GrainField):
        """Same pipelines and semantic entry, another grain field (new radius / grains per particle)."""
        self.field = field
        self.inst_desc, self.capacity = None, 0
        info = self.r0.ctx.buffer_array(field.mesh_info, np.uint32, (field.n_meshes, 4))
        self.i_count = int(info[0, 1])
        self.verts_per_mesh = int(field.vbuf.length() // 32 // field.n_meshes)
        return self

    def encode_instances(self, cb):
        f = self.field
        if f.n_grains > self.capacity:
            self.capacity = f.n_grains
            self.inst_desc = self.r0.ctx.buffer(64 * f.n_grains, label="t0_grain_instances")
        f.encode_instances(cb, self.inst_desc, 0, 0, 1, 0.0)

    def _bind(self, enc):
        f, r0 = self.field, self.r0
        k = np.array([f.n_particles * f.k, self.verts_per_mesh, 0, r0.G + 1], np.uint32)
        enc.setVertexBuffer_offset_atIndex_(f.vbuf, 0, 0)
        enc.setVertexBuffer_offset_atIndex_(r0.env_buf, 0, 1)
        enc.setVertexBuffer_offset_atIndex_(self.inst_desc, 0, 2)
        enc.setVertexBytes_length_atIndex_(k.tobytes(), k.nbytes, 3)
        return k

    def _draw(self, enc):
        enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance_(
            Metal.MTLPrimitiveTypeTriangle, self.i_count, Metal.MTLIndexTypeUInt32, self.field.ibuf, 0, self.field.n_grains, 0, 0)

    def draw_shadow(self, se):
        if self.field.n_grains == 0:
            return
        se.setRenderPipelineState_(self.shadow_pipe)
        self._bind(se)
        self._draw(se)

    def draw(self, re, light_xpos, light_xpos_off, light_xdir, light_xdir_off):
        if self.field.n_grains == 0:
            return
        r0 = self.r0
        re.setRenderPipelineState_(self.pipe)
        re.setDepthStencilState_(r0.geo_ds)
        k = self._bind(re)
        re.setFragmentBuffer_offset_atIndex_(r0.env_buf, 0, 1)
        re.setFragmentBuffer_offset_atIndex_(self.field.mat_buf, 0, 2)
        re.setFragmentBytes_length_atIndex_(k.tobytes(), k.nbytes, 3)
        re.setFragmentBuffer_offset_atIndex_(r0.consts_buf, 0, 5)
        re.setFragmentBuffer_offset_atIndex_(r0.light_buf, 0, 6)
        re.setFragmentBuffer_offset_atIndex_(light_xpos, light_xpos_off, 7)
        re.setFragmentBuffer_offset_atIndex_(light_xdir, light_xdir_off, 8)
        re.setFragmentBuffer_offset_atIndex_(r0.params_buf, 0, 9)
        re.setFragmentTexture_atIndex_(r0.shadow_tex, 2)
        re.setFragmentSamplerState_atIndex_(r0.samp_shadow, 1)
        re.setFragmentBuffer_offset_atIndex_(self.field.keys, 0, 10)
        self._draw(re)
