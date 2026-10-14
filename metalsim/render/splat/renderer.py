"""Batched 3D Gaussian splat renderer on native Metal (compute, tile-based). Design + measurements: ``render/SPLAT.md``.

Outputs follow ``Tier0Renderer``: Warp arrays on ``metal:0`` aliased as MPS tensors in ``self.out`` (``rgb`` (N, H, W, 3)
uint8, ``depth`` (N, H, W) float32 planar metric, 0 = nothing; plus optional ``depth_expected`` and ``alpha``).
Cameras use tier0's conventions: MuJoCo camera frames (``cam_xpos`` / ``cam_xmat``: looks along -z, +y up) and the
same 28-float ``CameraSpec`` per env (intrinsics fx, fy, cx, cy; pos/rot deltas; ``cam_id``), so a learner or harness
that drives a Tier0Renderer drives this one identically.

``mesh=Tier0Renderer(...)`` composites mesh-rendered geometry (robot, dynamic objects) with correct occlusion: per
pixel the splat blend stops at the mesh depth and the mesh colour fills the remaining transmittance.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import warp as wp
import Metal

from metalsim.interop import torch_bridge as tb
from metalsim.interop import warp_metal as wm
from metalsim.render.metal_context import MetalContext
from metalsim.render.scene_tables import quats_to_mats
from metalsim.render.splat.scene import PackedSplats

TILE = 16
SCAN_PARTS = 256
REC_BYTES = 32
CAM_BYTES = 144
MAX_BATCH = 32          # cameras per sub-batch (per-splat camera bitmask is 32 bits)


def _sz(n):
    return Metal.MTLSize(int(n), 1, 1)


class SplatRenderer:
    def __init__(self, splats: PackedSplats, n_envs: int, *, width=640, height=360, fx=None, fy=None, cx=None, cy=None,
                 hfov_deg=None, outputs=("rgb", "depth"), near=0.05, far=500.0, clear_color=(0.0, 0.0, 0.0),
                 antialiased: bool | None = None, eps2d: float | None = None, batch: int | None = None, cap_records=6_000_000,
                 cap_keys=16_000_000, scratch_budget_gb: float = 3.0, depth_bits: int = 24, blend: str = "1px", mesh=None, ctx: MetalContext | None = None, device="metal:0", defines: dict | None = None):
        self.s = splats
        self.n, self.w, self.h = int(n_envs), int(width), int(height)
        self.mesh = mesh
        self.ctx = ctx or (mesh.ctx if mesh is not None else MetalContext(device))
        self.device = device
        if mesh is not None:
            assert (mesh.n, mesh.tw, mesh.th) == (self.n, self.w, self.h), "mesh renderer must have the same envs and size"
            fx, fy, cx, cy = mesh.intrinsics.fx, mesh.intrinsics.fy, mesh.intrinsics.cx, mesh.intrinsics.cy
            assert "rgb" in mesh._out_bufs and "depth" in mesh._out_bufs, "mesh renderer needs outputs rgb + depth"
        if fx is None:
            hf = 90.0 if hfov_deg is None else hfov_deg
            fx = (self.w / 2) / math.tan(math.radians(hf) / 2)
        fy = fx if fy is None else fy
        cx = self.w / 2 if cx is None else cx
        cy = self.h / 2 if cy is None else cy
        self.near, self.far = float(near), float(far)
        self.antialiased = bool(splats.antialiased if antialiased is None else antialiased)
        # 2D dilation (px^2): None = the scene's hint (formats: NuRec/3DGUT models are trained without it -> 0) else 3DGS's 0.3
        self.eps2d = float(eps2d if eps2d is not None else (splats.info.get("meta") or {}).get("eps2d_hint", 0.3))
        self.clear_color = np.asarray(clear_color, np.float32)
        self.tiles_x, self.tiles_y = (self.w + TILE - 1) // TILE, (self.h + TILE - 1) // TILE
        self.tiles_per_cam = self.tiles_x * self.tiles_y
        self.max_batch = min(MAX_BATCH, 65536 // self.tiles_per_cam)
        if self.max_batch < 1:
            raise ValueError("image too large for 16-bit tile ids")
        self.batch = min(self.n, self.max_batch if batch is None else min(int(batch), self.max_batch))
        self.cap_rec, self.cap_keys = min(int(cap_records), 1 << 24), int(cap_keys)
        self.scratch_budget = float(scratch_budget_gb) * 1e9
        if depth_bits not in (16, 24):
            raise ValueError("depth_bits: 24 (default) or 16 (one radix pass less; ties within ~1.4e-4 z sort by splat index)")
        self.depth_bits = depth_bits
        if blend not in ("2px", "1px"):
            raise ValueError("blend: '1px' (default) or '2px' (two pixels per thread; bit-identical, SPLAT.md §6)")
        self.blend = blend
        self._last_cb = None
        self.n_blocks = (splats.n + 255) // 256
        self._lib = self.ctx.library("splat", defines)          # defines: shader experiments (SPLAT.md ablations)
        self._p = {k: self.ctx.compute_pipeline(self._lib, k) for k in (
            "splat_cams", "splat_cull", "splat_project", "splat_emit", "scan_reduce", "scan_top", "scan_down", "splat_args",
            "rs_hist", "rs_scatter", "splat_blend", "splat_blend2", "splat_reset")}
        self._upload_scene()
        self._init_cameras(fx, fy, cx, cy)
        self._alloc_outputs(outputs)
        self._alloc_scratch()
        self._batches = None
        self.last_stats = None

    # ---------------------------------------------------------------------------------------------- setup

    def _upload_scene(self):
        c, s = self.ctx, self.s
        self.g0 = c.buffer(s.g0.nbytes, s.g0, "splat_g0")
        self.g1 = c.buffer(s.g1.nbytes, s.g1, "splat_g1")
        self.shb = c.buffer(s.sh.nbytes, s.sh, "splat_sh")
        # per-threadgroup (256 consecutive, Morton-ordered splats) bounding spheres incl. the per-splat cull radii
        nb = (s.n + 255) // 256
        pad = nb * 256 - s.n
        p = np.concatenate([s.g0[:, :3], np.repeat(s.g0[-1:, :3], pad, 0)]).reshape(nb, 256, 3).astype(np.float64)
        r = np.concatenate([s.g1[:, 7].astype(np.float64), np.repeat(s.g1[-1:, 7].astype(np.float64), pad)]).reshape(nb, 256)
        lo, hi = p.min(1), p.max(1); ctr = 0.5 * (lo + hi)
        rad = (np.linalg.norm(p - ctr[:, None], axis=2) + r).max(1)
        ch = np.concatenate([ctr, rad[:, None] * 1.0001 + 1e-4], 1).astype(np.float32)
        self.chunks = c.buffer(ch.nbytes, ch, "splat_chunks")
        self.scene_bytes = s.g0.nbytes + s.g1.nbytes + s.sh.nbytes + ch.nbytes

    def _init_cameras(self, fx, fy, cx, cy):
        self.cam_spec = np.zeros((self.n, 28), np.float32)
        self.cam_spec_i = self.cam_spec.view(np.int32)
        self.cam_spec[:, 0:4] = (fx, fy, cx, cy)
        self.cam_spec[:, 8:12] = (1, 0, 0, 0)
        if self.mesh is not None:                         # GPU path: the same MuJoCo camera as the mesh renderer
            self.cam_spec_i[:, 24] = self.mesh.cam_spec_i[:, 24]
        self._saved_cam_ids = None                        # per-env cam ids set aside while render_host uses host cameras
        self._cam_spec_wp = wp.array(self.cam_spec, dtype=wp.float32, device=self.device)
        self.t_cam_spec = tb.mps_tensor(self._cam_spec_wp)
        self.cam_spec_buf = wm.buffer_of(self._cam_spec_wp).buffer
        wp.synchronize_device(self.device)
        self.cams = self.ctx.buffer(self.n * CAM_BYTES, label="splat_cams")
        self._host_cam_xpos = self.ctx.buffer(self.n * 12, label="splat_host_cam_xpos")
        self._host_cam_xmat = self.ctx.buffer(self.n * 36, label="splat_host_cam_xmat")
        self._cams_cc = self.ctx.buffer(16, np.array([self.n, 1, self.w, self.h], np.uint32))
        self._cams_cf = self.ctx.buffer(16, np.array([self.near, 0, 0, 0], np.float32))

    @property
    def intrinsics(self):
        return tuple(float(v) for v in self.cam_spec[0, 0:4])

    def set_intrinsics(self, fx, fy, cx, cy, env_ids=None):
        self.cam_spec[slice(None) if env_ids is None else env_ids, 0:4] = (fx, fy, cx, cy)
        self._sync_cam_spec()

    def set_camera_delta(self, pos_delta=None, quat_delta=None):
        if pos_delta is not None:
            self.cam_spec[:, 4:7] = np.asarray(pos_delta, np.float32)
        if quat_delta is not None:
            q = np.asarray(quat_delta, np.float64)
            self.cam_spec[:, 8:12] = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
        self._sync_cam_spec()

    def _sync_cam_spec(self):
        wp.synchronize_device(self.device)
        torch.mps.synchronize()
        self._cam_spec_wp.assign(self.cam_spec)
        wp.synchronize_device(self.device)

    def _alloc_outputs(self, outputs):
        from metalsim.render.tier0 import RenderOutputs
        self.out = RenderOutputs()
        self._out_arrays, self._out_bufs = {}, {}
        n, h, w = self.n, self.h, self.w
        specs = {"rgb": ((n, h, w, 3), wp.uint8), "depth": ((n, h, w), wp.float32),
                 "depth_expected": ((n, h, w), wp.float32), "alpha": ((n, h, w), wp.float32)}
        for name in outputs:
            if name not in specs:
                raise ValueError(f"splat renderer output {name!r} not supported (have {sorted(specs)})")
            a = wp.zeros(specs[name][0], dtype=specs[name][1], device=self.device)
            self._out_arrays[name] = a
            v = wm.buffer_of(a)
            self._out_bufs[name] = (v.buffer, v.offset)
            setattr(self.out, name, tb.mps_tensor(a))
        wp.synchronize_device(self.device)
        self._dummy = self.ctx.buffer(16, label="splat_dummy")
        self._outs = self.ctx.buffer(16, np.array([int(k in outputs) for k in ("rgb", "depth", "depth_expected", "alpha")], np.uint32))

    def _alloc_scratch(self):
        c = self.ctx
        B, nb = self.max_batch, self.n_blocks
        self.mask = c.buffer(4 * self.s.n, label="splat_mask")
        self.bcount = c.buffer(4 * B * nb, label="splat_bcount")
        self.boff = c.buffer(4 * B * nb, label="splat_boff")
        self.parts = c.buffer(4 * SCAN_PARTS, label="splat_scan_parts")
        self._alloc_caps()
        # LSD passes: the value's top byte (low 8 bits of the 24-bit depth), then the key's 4 bytes
        passes = ((24, 1), (0, 0), (8, 0), (16, 0), (24, 0)) if self.depth_bits == 24 else ((0, 0), (8, 0), (16, 0), (24, 0))
        self._shift_bufs = [c.buffer(16, np.array([s, src, 0, 0], np.uint32)) for s, src in passes]
        self._sorted = len(passes) % 2                      # ping-pong buffer holding the sorted keys / values
        self._which = [c.buffer(16, np.array([w, 0, 0, 0], np.uint32)) for w in (0, 1)]

    def _alloc_caps(self):
        c = self.ctx
        self.recs = c.buffer(REC_BYTES * self.cap_rec, label="splat_records")
        self.ntile = c.buffer(4 * self.cap_rec, label="splat_ntile")
        self.koff = c.buffer(4 * self.cap_rec, label="splat_key_offsets")
        self.keys = [c.buffer(4 * self.cap_keys, label="splat_keys_a"), c.buffer(4 * self.cap_keys, label="splat_keys_b")]
        self.vals = [c.buffer(4 * self.cap_keys, label="splat_vals_a"), c.buffer(4 * self.cap_keys, label="splat_vals_b")]
        mb = (self.cap_keys + 4095) // 4096
        self.hist = c.buffer(4 * 256 * mb, label="splat_hist")
        self.hoff = c.buffer(4 * 256 * mb, label="splat_hist_off")
        self._hist_cap = 256 * mb
        self._batches = None

    def scratch_bytes(self) -> int:
        nb = self.n_blocks
        return (4 * self.s.n + 8 * self.max_batch * nb + REC_BYTES * self.cap_rec + 8 * self.cap_rec
                + 16 * self.cap_keys + 8 * 256 * ((self.cap_keys + 4095) // 4096))

    def _build_batches(self):
        """Per sub-batch constant / counter buffers (host-written once; counters reset on the GPU each frame)."""
        A = np.asarray(self.s.A, np.float64)
        D = np.linalg.inv(A)
        self._batches = []
        for cam0 in range(0, self.n, self.batch):
            nc = min(self.batch, self.n - cam0)
            k = np.zeros(48, np.uint32); f = k.view(np.float32)
            flags = (1 if self.antialiased else 0) | (2 if self.s.sh_u8 else 0) | (4 if self.mesh is not None else 0)
            k[0:16] = (self.s.n, self.n, self.w, self.h, self.tiles_x, self.tiles_y, self.tiles_per_cam, cam0,
                       nc, self.s.sh_degree, flags, self.s.sh_stride, self.cap_rec, self.cap_keys, self.n_blocks, self.depth_bits)
            lo = math.log(self.near); f[16:20] = (self.near, lo, 16777215.0 / (math.log(self.far) - lo), self.eps2d)
            for r in range(3):
                f[20 + 4 * r:23 + 4 * r] = A[r]
                f[32 + 4 * r:35 + 4 * r] = D[r]
            f[44:47] = self.clear_color[:3]; f[47] = 1.0
            b = dict(cam0=cam0, nc=nc, consts=self.ctx.buffer(k.nbytes, k, f"splat_consts_{cam0}"),
                     ctr=self.ctx.buffer(16, label=f"splat_ctr_{cam0}"), totals=self.ctx.buffer(16, label=f"splat_totals_{cam0}"),
                     args=self.ctx.buffer(48, label=f"splat_args_{cam0}"), nblk=self.ctx.buffer(16, label=f"splat_nblk_{cam0}"),
                     n_bc=self.ctx.buffer(16, np.array([nc * self.n_blocks, 0, 0, 0], np.uint32)),
                     cap_bc=self.ctx.buffer(16, np.array([nc * self.n_blocks, 0, 0, 0], np.uint32)),
                     cap_rec=self.ctx.buffer(16, np.array([self.cap_rec, 0, 0, 0], np.uint32)),
                     cap_hist=self.ctx.buffer(16, np.array([self._hist_cap, 0, 0, 0], np.uint32)),
                     n_reset=self.ctx.buffer(16, np.array([4, 0, 0, 0], np.uint32)))
            self._batches.append(b)

    # ---------------------------------------------------------------------------------------------- encoding

    def _scan(self, e, inp, inp_off, n_ptr, n_ptr_off, cap_buf, out, total, total_off):
        e.setComputePipelineState_(self._p["scan_reduce"])
        e.setBuffer_offset_atIndex_(inp, inp_off, 0); e.setBuffer_offset_atIndex_(n_ptr, n_ptr_off, 1)
        e.setBuffer_offset_atIndex_(self.parts, 0, 2); e.setBuffer_offset_atIndex_(cap_buf, 0, 3)
        e.dispatchThreadgroups_threadsPerThreadgroup_(_sz(SCAN_PARTS), _sz(1024))
        e.setComputePipelineState_(self._p["scan_top"])
        e.setBuffer_offset_atIndex_(total, total_off, 4)
        e.dispatchThreadgroups_threadsPerThreadgroup_(_sz(1), _sz(1024))
        e.setComputePipelineState_(self._p["scan_down"])
        e.setBuffer_offset_atIndex_(out, 0, 5)
        e.dispatchThreadgroups_threadsPerThreadgroup_(_sz(SCAN_PARTS), _sz(1024))

    def _encode(self, cb, cam_xpos, cam_xpos_off, cam_xmat, cam_xmat_off, n_cam=1, profile=None):
        """``profile``: a list -> every stage gets its own command buffer (committed here) and (stage, cb) is appended,
        so GPU times per stage can be read after synchronizing (``profile_frame``). Only for measurements."""
        if self._batches is None:
            self._build_batches()
        if n_cam != 1:
            self.ctx.write_buffer(self._cams_cc, np.array([self.n, n_cam, self.w, self.h], np.uint32))
        e = cb.computeCommandEncoder()
        P = self._p
        state = dict(cb=cb, e=e)

        def stage(name):
            if profile is None:
                return state["e"]
            state["e"].endEncoding()
            self.ctx.commit(state["cb"])
            profile.append((name, state["cb"]))
            state["cb"] = self.ctx.command_buffer()
            state["e"] = state["cb"].computeCommandEncoder()
            return state["e"]
        e.setComputePipelineState_(P["splat_cams"])
        e.setBuffer_offset_atIndex_(self.cams, 0, 0); e.setBuffer_offset_atIndex_(self.cam_spec_buf, 0, 1)
        e.setBuffer_offset_atIndex_(cam_xpos, cam_xpos_off, 2); e.setBuffer_offset_atIndex_(cam_xmat, cam_xmat_off, 3)
        e.setBuffer_offset_atIndex_(self._cams_cc, 0, 4); e.setBuffer_offset_atIndex_(self._cams_cf, 0, 5)
        e.dispatchThreads_threadsPerThreadgroup_(_sz(self.n), _sz(min(self.n, 64)))
        e = stage("cams")
        nb = self.n_blocks
        bg_rgb = bg_dep = self._dummy
        bg_rgb_off = bg_dep_off = 0
        if self.mesh is not None:
            bg_rgb, bg_rgb_off = self.mesh._out_bufs["rgb"]
            bg_dep, bg_dep_off = self.mesh._out_bufs["depth"]
        for b in self._batches:
            K = b["consts"]
            e.setComputePipelineState_(P["splat_reset"])
            e.setBuffer_offset_atIndex_(b["ctr"], 0, 0); e.setBuffer_offset_atIndex_(b["n_reset"], 0, 1)
            e.dispatchThreads_threadsPerThreadgroup_(_sz(4), _sz(4))
            # 1. conservative cull + per-(camera, block) counts
            e.setComputePipelineState_(P["splat_cull"])
            for i, buf in enumerate((self.g0, self.g1, self.cams, K, self.mask, self.bcount, self.chunks)):
                e.setBuffer_offset_atIndex_(buf, 0, i)
            e.dispatchThreadgroups_threadsPerThreadgroup_(_sz(nb), _sz(256))
            e = stage("cull")
            # 2. record offsets
            self._scan(e, self.bcount, 0, b["n_bc"], 0, b["cap_bc"], self.boff, b["totals"], 0)
            e = stage("scan_records")
            # 3. projection
            e.setComputePipelineState_(P["splat_project"])
            for i, buf in enumerate((self.g0, self.g1, self.shb, self.cams, K, self.mask, self.boff, self.recs, self.ntile, b["ctr"])):
                e.setBuffer_offset_atIndex_(buf, 0, i)
            e.dispatchThreadgroups_threadsPerThreadgroup_(_sz(nb), _sz(256))
            e = stage("project")
            e.setComputePipelineState_(P["splat_args"])
            for i, buf in enumerate((b["ctr"], b["totals"], K, b["args"], b["nblk"], self._which[0])):
                e.setBuffer_offset_atIndex_(buf, 0, i)
            e.dispatchThreads_threadsPerThreadgroup_(_sz(1), _sz(1))
            # 4. key offsets (n = ctr.n_rec at byte 8)
            self._scan(e, self.ntile, 0, b["ctr"], 8, b["cap_rec"], self.koff, b["totals"], 4)
            e.setComputePipelineState_(P["splat_args"])
            for i, buf in enumerate((b["ctr"], b["totals"], K, b["args"], b["nblk"], self._which[1])):
                e.setBuffer_offset_atIndex_(buf, 0, i)
            e.dispatchThreads_threadsPerThreadgroup_(_sz(1), _sz(1))
            e = stage("scan_keys")
            # 5. keys
            e.setComputePipelineState_(P["splat_emit"])
            for i, buf in enumerate((self.recs, self.koff, self.ntile, K, self.keys[0], self.vals[0], b["ctr"], self.boff)):
                e.setBuffer_offset_atIndex_(buf, 0, i)
            e.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup_(b["args"], 0, _sz(256))
            e = stage("emit")
            # 6. radix sort, 5 x 8 bits, ping-pong (result in keys[1] / vals[1])
            for p, shift in enumerate(self._shift_bufs):
                src, dst = p % 2, 1 - p % 2
                e.setComputePipelineState_(P["rs_hist"])
                for i, buf in enumerate((self.keys[src], b["ctr"], b["nblk"], self.hist, shift, self.vals[src])):
                    e.setBuffer_offset_atIndex_(buf, 0, i)
                e.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup_(b["args"], 12, _sz(256))
                self._scan(e, self.hist, 0, b["nblk"], 4, b["cap_hist"], self.hoff, b["totals"], 8)
                e.setComputePipelineState_(P["rs_scatter"])
                for i, buf in enumerate((self.keys[src], self.vals[src], b["ctr"], b["nblk"], self.hoff, self.keys[dst], self.vals[dst], shift)):
                    e.setBuffer_offset_atIndex_(buf, 0, i)
                e.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup_(b["args"], 12, _sz(256))
            e = stage("sort")
            # 8. blend into the learner buffers
            e.setComputePipelineState_(P["splat_blend2" if self.blend == "2px" else "splat_blend"])
            ob = lambda name: self._out_bufs.get(name, (self._dummy, 0))
            for i, (buf, off) in enumerate(((self.recs, 0), (self.vals[self._sorted], 0), (self.keys[self._sorted], 0), (K, 0), ob("rgb"), ob("depth"),
                                            ob("depth_expected"), (bg_rgb, bg_rgb_off), (bg_dep, bg_dep_off), ob("alpha"), (self._outs, 0),
                                            (b["ctr"], 0))):
                e.setBuffer_offset_atIndex_(buf, off, i)
            e.dispatchThreadgroups_threadsPerThreadgroup_(_sz(b["nc"] * self.tiles_per_cam), Metal.MTLSize(TILE, TILE // 2 if self.blend == "2px" else TILE, 1))
            e = stage("blend")
        e.endEncoding()
        return state["cb"]

    # ---------------------------------------------------------------------------------------------- stats / capacity

    def stats(self, synchronize=True) -> dict:
        """Records (visible splat-camera pairs), keys (tile intersections) and overflow per sub-batch of the last frame."""
        if synchronize:
            self.ctx.synchronize()
        rows = []
        for b in self._batches or []:
            v = self.ctx.buffer_array(b["ctr"], np.uint32, (4,))
            t = self.ctx.buffer_array(b["totals"], np.uint32, (4,))
            rows.append(dict(cam0=b["cam0"], n_cams=b["nc"], records=int(t[0]), keys=int(t[1]),
                             overflow=bool(v[0] or v[1] or t[0] > self.cap_rec or t[1] > self.cap_keys)))
        self.last_stats = rows
        return dict(batches=rows, records=sum(r["records"] for r in rows), keys=sum(r["keys"] for r in rows),
                    overflow=any(r["overflow"] for r in rows))

    def _fix_overflow(self, st) -> bool:
        """After an overflowing frame: size the per-camera need from its counters (x1.25), then pick the largest sub-batch
        that fits the 24-bit record index and the scratch budget, and reallocate. True if anything changed."""
        if not st["overflow"]:
            return False
        rows = st["batches"]
        per_rec = max(r["records"] / r["n_cams"] for r in rows) * 1.25 + 1024
        per_key = max(r["keys"] / r["n_cams"] for r in rows) * 1.25 + 4096
        budget = self.scratch_budget - 4 * self.s.n
        fit_budget = int(budget // (per_rec * (REC_BYTES + 8) + per_key * 18))
        batch = max(1, min(self.batch, int((1 << 24) // per_rec), fit_budget))
        self.cap_rec = max(self.cap_rec, min(int(per_rec * batch), 1 << 24))
        self.cap_keys = max(self.cap_keys, int(per_key * batch))
        changed_batch = batch != self.batch
        self.batch = batch
        warnings.warn(f"splat renderer: capacity exceeded (per camera ~{per_rec:,.0f} records, {per_key:,.0f} keys); sub-batch "
                      f"{self.batch} cameras, capacities {self.cap_rec:,} / {self.cap_keys:,} ({self.scratch_bytes() / 1e9:.2f} GB scratch)"
)
        self._alloc_caps()
        return True

    # ---------------------------------------------------------------------------------------------- rendering

    def render(self, sim, after_value: int | None = None, cam_id: int | None = None) -> int:
        """GPU path: cameras from ``sim.d.cam_xpos / cam_xmat`` (``cam_spec`` cam_id per env; ``cam_id`` overrides),
        waits for ``sim.event >= after_value``, signals ``self.ctx.event``; returns that value. Nothing synchronizes.
        With ``mesh``, the mesh renderer's GPU render is encoded first on the same queue."""
        if after_value is None:
            after_value = sim.event.value
        if self._last_cb is not None and self._last_cb.status() >= Metal.MTLCommandBufferStatusCompleted:
            st = self.stats(synchronize=False)            # previous frame's counters, no host wait
            self._last_cb = None
            if self._fix_overflow(st):
                warnings.warn("splat renderer (GPU path): the previous frame overflowed and is incomplete; capacities fixed from now on")
        xp, xm = wm.buffer_of(sim.d.cam_xpos), wm.buffer_of(sim.d.cam_xmat)
        n_cam = int(sim.m.ncam) if hasattr(sim.m, "ncam") else 1
        if cam_id is not None:
            self.cam_spec_i[:, 24] = cam_id
            self._saved_cam_ids = None
            self._sync_cam_spec()
        elif self._saved_cam_ids is not None:            # a render_host call zeroed them: restore for the GPU path
            self.cam_spec_i[:, 24] = self._saved_cam_ids
            self._saved_cam_ids = None
            self._sync_cam_spec()
        if self.mesh is not None:
            self.mesh.render(sim, after_value)
        cb = self.ctx.command_buffer()
        self.ctx.wait_for(cb, sim.event, after_value)
        cb = self._encode(cb, xp.buffer, xp.offset, xm.buffer, xm.offset, n_cam=n_cam)
        v = self.ctx.signal(cb)
        self.ctx.commit(cb)
        self._last_cb = cb
        return v

    def after(self, value: int) -> None:
        tb.wait_event(self.ctx.event, value)

    def wait_sim_after_render(self, sim, value: int) -> None:
        sim.wait(self.ctx.event, value)

    def profile_frame(self, cam_pos, cam_mat) -> dict:
        """GPU ms per stage (summed over sub-batches) for one host-path frame, each stage in its own command buffer."""
        self.ctx.write_buffer(self._host_cam_xpos, np.asarray(cam_pos, np.float32).reshape(self.n, 3))
        self.ctx.write_buffer(self._host_cam_xmat, np.asarray(cam_mat, np.float32).reshape(self.n, 9))
        prof = []
        cb = self._encode(self.ctx.command_buffer(), self._host_cam_xpos, 0, self._host_cam_xmat, 0, profile=prof)
        self.ctx.commit(cb)
        self.ctx.synchronize()
        out = {}
        for name, c in prof:
            out[name] = out.get(name, 0.0) + (c.GPUEndTime() - c.GPUStartTime()) * 1e3
        out["total"] = sum(out.values())
        return out

    def bench(self, cam_pos, cam_mat, iters=20) -> float:
        """Throughput: ``iters`` frames encoded back to back at fixed host-written cameras, one synchronize at the end (no
        output copy, like the GPU path). Returns ms per frame (all N cameras). Run ``render_host`` once first (capacities)."""
        import time
        self.ctx.write_buffer(self._host_cam_xpos, np.asarray(cam_pos, np.float32).reshape(self.n, 3))
        self.ctx.write_buffer(self._host_cam_xmat, np.asarray(cam_mat, np.float32).reshape(self.n, 9))
        self.ctx.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            cb = self._encode(self.ctx.command_buffer(), self._host_cam_xpos, 0, self._host_cam_xmat, 0)
            self.ctx.commit(cb)
        self.ctx.synchronize()
        return (time.perf_counter() - t) * 1e3 / iters

    def render_host(self, cam_pos, cam_quat=None, cam_mat=None, mesh_datas=None, retry=True) -> dict:
        """Host path: per-env MuJoCo camera poses (``cam_pos`` (N, 3), and ``cam_quat`` (N, 4) wxyz or ``cam_mat``
        (N, 3, 3) with the camera looking along -z, +y up). ``mesh_datas``: list of N MjData for the mesh composite.
        Synchronizes; returns numpy outputs. Grows capacities and re-renders if a frame overflowed."""
        pos = np.asarray(cam_pos, np.float32).reshape(self.n, 3)
        if cam_mat is None:
            cam_mat = quats_to_mats(np.asarray(cam_quat, np.float64).reshape(self.n, 4))
        mat = np.asarray(cam_mat, np.float32).reshape(self.n, 9)
        if self.mesh is not None:
            if mesh_datas is None:
                raise ValueError("composite renderer: pass mesh_datas (N MjData) for the mesh pass")
            self.mesh.render_host(mesh_datas, cam_pos=pos, cam_quat=cam_quat if cam_quat is not None else _mats_to_quats(mat))
        self.ctx.write_buffer(self._host_cam_xpos, pos)
        self.ctx.write_buffer(self._host_cam_xmat, mat)
        if not np.all(self.cam_spec_i[:, 24] == 0):
            self._saved_cam_ids = self.cam_spec_i[:, 24].copy()
            self.cam_spec_i[:, 24] = 0
            self._sync_cam_spec()
        for _ in range(6):
            cb = self._encode(self.ctx.command_buffer(), self._host_cam_xpos, 0, self._host_cam_xmat, 0)
            self.ctx.commit(cb)
            st = self.stats(synchronize=True)
            if not (retry and self._fix_overflow(st)):
                break
        return {name: a.numpy().copy() for name, a in self._out_arrays.items()}


def _mats_to_quats(mats):
    import mujoco
    out = np.zeros((len(mats), 4))
    for i, m in enumerate(np.asarray(mats, np.float64).reshape(-1, 9)):
        mujoco.mju_mat2Quat(out[i], m)
    return out
