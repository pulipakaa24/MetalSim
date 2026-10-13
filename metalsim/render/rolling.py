"""Rolling shutter for the batched renderers (tier 0 / 1 raster, tier 2 path tracer), from per-env pose history.

What Isaac does (research notes "Render sensor fidelity/gaps_and_perf.md"): Isaac Lab's camera configs have no rolling
shutter; Omniverse RTX (Kit 109) does, through a per-pixel time texture on the camera (``shutterTimeTexturePath``,
values in [0, 1]: 0 = the start of the previous frame, 1 = the start of the current one; each pixel exposed
instantaneously, no blur; readout order whatever the texture holds, top to bottom in NVIDIA's example).

MetalSim follows that convention. A ``RollingShutter`` gives each image row a time in [0, 1] of the interval between the
previous rendered frame and the current one. The renderer keeps the previous frame's pose inputs per env (``PoseHistory``:
geom, site, camera and light poses, copied on the GPU at the end of each frame) and renders the image in row bands: for
each band, the poses at the band's time (positions linear, orientations slerp, ``shaders/rolling.metal``), then only
that band's rows (tier 0 / 1: the frame re-rendered and its rows written out; tier 2: only those rows traced). With one
band per row the skew is exact for motion that is linear over the frame interval; with fewer bands each band shows its
mean time (a staircase of at most half a band's time span). Deforming meshes (flexes, skins, tendons, user meshes) and
particles are drawn at the current state (not interpolated).

    rs = RollingShutter.top_to_bottom(height=96, readout=0.5)   # rows read over the last half of the frame interval
    r = Tier0Renderer(m, n, width=128, height=96, rolling_shutter=rs)
    r.render(sim)                                               # the first frame has no motion (no history yet)
"""
from __future__ import annotations

import numpy as np
import Metal


class RollingShutter:
    def __init__(self, row_times, bands: int | None = None):
        """``row_times``: (H,) time of each image row in [0, 1] (0: the previous frame's state, 1: the current one).
        ``bands``: at most this many row bands per frame (default: one per run of equal times, at most the row count)."""
        t = np.asarray(row_times, np.float64).reshape(-1)
        if t.size == 0 or np.any(t < 0) or np.any(t > 1):
            raise ValueError("row times must be in [0, 1]")
        self.row_times = t
        self.max_bands = None if bands is None else max(1, int(bands))

    @classmethod
    def top_to_bottom(cls, height: int, readout: float, end: float = 1.0, bands: int | None = None):
        """Rows read top to bottom over ``readout`` (fraction of the frame interval), the last row at ``end``: row r at
        end - readout (H - 1 - r) / (H - 1)."""
        H = int(height)
        r = np.arange(H, dtype=np.float64)
        t = np.full(H, float(end)) if H == 1 else end - readout * (H - 1 - r) / (H - 1)
        return cls(t, bands)

    @classmethod
    def from_readout_time(cls, height: int, readout_s: float, frame_dt: float, bands: int | None = None):
        """A sensor reading its rows over ``readout_s`` seconds, frames ``frame_dt`` apart (the last row at the frame)."""
        return cls.top_to_bottom(height, float(readout_s) / float(frame_dt), 1.0, bands)

    def bands(self, height: int):
        """[(row0, row1, tau)]: contiguous row bands and the time each is rendered at."""
        t = self.row_times
        if len(t) != height:
            raise ValueError(f"rolling shutter has {len(t)} row times, the image {height} rows")
        runs, r0 = [], 0
        for r in range(1, height + 1):
            if r == height or t[r] != t[r0]:
                runs.append((r0, r, float(t[r0])))
                r0 = r
        if self.max_bands is None or len(runs) <= self.max_bands:
            return runs
        B = self.max_bands
        edges = np.linspace(0, height, B + 1).round().astype(int)
        return [(int(a), int(b), float(t[a:b].mean())) for a, b in zip(edges[:-1], edges[1:]) if b > a]


class PoseHistory:
    """The previous frame's pose inputs on the GPU and their interpolation to a band's time.

    ``fields``: {name: (kind, floats per env)} with kind "pos" (3 per item), "mat" (9) or "dir" (3)."""

    def __init__(self, ctx, n_envs: int, fields: dict):
        self.ctx, self.n = ctx, int(n_envs)
        self.fields = dict(fields)
        c = ctx
        self.prev = {k: c.buffer(4 * self.n * w, label=f"hist_prev_{k}") for k, (_, w) in self.fields.items()}
        self.out = {k: c.buffer(4 * self.n * w, label=f"hist_band_{k}") for k, (_, w) in self.fields.items()}
        self.primed = False
        self.pipe = c.compute_pipeline(c.library("rolling"), "interp_poses")
        self._dummy = c.buffer(16, label="hist_dummy")

    def encode_snapshot(self, cb, cur: dict):
        """Copy the current pose inputs ({name: (buffer, offset)}) into the history (end of the frame)."""
        be = cb.blitCommandEncoder()
        for k, (_, w) in self.fields.items():
            b, o = cur[k]
            be.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(b, o, self.prev[k], 0, 4 * self.n * w)
        be.endEncoding()

    def encode_interp(self, cb, cur: dict, tau: float) -> dict:
        """The pose inputs at time ``tau`` into this history's band buffers; returns {name: (buffer, 0)}."""
        if not self.primed:              # no previous frame yet: the current state (no motion)
            self.encode_snapshot(cb, cur)
            self.primed = True
        ce = cb.computeCommandEncoder()
        ce.setComputePipelineState_(self.pipe)
        d = self._dummy
        for k, (kind, w) in self.fields.items():
            items = self.n * w // (9 if kind == "mat" else 3)
            b, o = cur[k]
            slots = {"pos": (0, 1, 2), "mat": (3, 4, 5), "dir": (6, 7, 8)}[kind]
            for i in range(9):
                ce.setBuffer_offset_atIndex_(d, 0, i)
            ce.setBuffer_offset_atIndex_(self.prev[k], 0, slots[0])
            ce.setBuffer_offset_atIndex_(b, o, slots[1])
            ce.setBuffer_offset_atIndex_(self.out[k], 0, slots[2])
            cs = np.zeros(8, np.uint32)
            cs[{"pos": 0, "mat": 1, "dir": 2}[kind]] = items
            cs.view(np.float32)[4] = tau
            ce.setBytes_length_atIndex_(cs.tobytes(), cs.nbytes, 9)
            ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(items, 1, 1), Metal.MTLSize(min(items, 64), 1, 1))
        ce.endEncoding()
        return {k: (self.out[k], 0) for k in self.fields}

    def write_prev_host(self, arrays: dict):
        """Host path: set the history from numpy arrays ({name: (N, ...)})."""
        for k, a in arrays.items():
            self.ctx.write_buffer(self.prev[k], np.ascontiguousarray(a, np.float32))
        self.primed = True
