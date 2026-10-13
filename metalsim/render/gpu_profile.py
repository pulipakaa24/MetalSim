"""Per-pass GPU timing of the renderers (a diagnostic, not used by them).

``PassProfiler(ctx)`` replaces ``ctx.command_buffer`` with a proxy that puts every encoder (compute, render,
acceleration-structure, blit) of a frame into its own Metal command buffer, committed in order on the same queue, and
reads each one's ``GPUStartTime`` / ``GPUEndTime`` after completion. A pass is labelled by the pipelines it used
(compute kernels by function name, render pipelines by vertex / fragment function) or its kind. The renderers' code
runs unchanged: event waits go to the first command buffer of the frame, signals to the last.

Splitting a frame this way serializes passes that the GPU could otherwise overlap and adds one command buffer's
scheduling gap per pass (measured ~10-30 us), so the per-pass times add up to more than the frame's normal GPU time;
they are for attribution and A/B comparison, the headline numbers come from unsplit frames.

    from metalsim.render import gpu_profile
    gpu_profile.install()                       # before creating renderers (names their render pipelines)
    prof = gpu_profile.PassProfiler(renderer.ctx)
    with prof:
        for _ in range(10): renderer.render(sim)
        renderer.ctx.synchronize()
    print(prof.table())
"""
from __future__ import annotations

from collections import defaultdict

import Metal

_RENDER_NAMES = {}


def install():
    """Record the vertex / fragment function names of render pipelines created from now on."""
    from metalsim.render.metal_context import MetalContext
    if getattr(MetalContext, "_profile_installed", False):
        return
    orig = MetalContext.render_pipeline

    def render_pipeline(self, desc):
        p = orig(self, desc)
        vf, ff = desc.vertexFunction(), desc.fragmentFunction()
        _RENDER_NAMES[id(p)] = "/".join(str(f.name()) for f in (vf, ff) if f is not None)
        return p
    MetalContext.render_pipeline = render_pipeline
    MetalContext._profile_installed = True


class _Enc:
    """An encoder that records the pipelines set on it."""

    def __init__(self, enc, seg, prof):
        self._e, self._seg, self._p = enc, seg, prof

    def __getattr__(self, name):
        return getattr(self._e, name)

    def setComputePipelineState_(self, p):
        self._seg["names"].append(self._p.compute_name(p))
        return self._e.setComputePipelineState_(p)

    def setRenderPipelineState_(self, p):
        self._seg["names"].append(_RENDER_NAMES.get(id(p), "render"))
        return self._e.setRenderPipelineState_(p)

    def endEncoding(self):
        return self._e.endEncoding()


class _CB:
    """Command-buffer proxy: one real command buffer per encoder."""

    def __init__(self, prof):
        self._p = prof
        self._segs = []          # dicts: cb, kind, names
        self._cur = None
        self._waits = []
        self._done = False

    def _new(self, kind):
        if self._cur is not None and self._cur["used"]:
            self._cur["cb"].commit()
            self._cur["cb"].waitUntilCompleted()      # passes strictly one after the other: no overlap, no queued wait
            self._cur = None
        if self._cur is None:
            cb = self._p.ctx.queue.commandBuffer()
            for ev, v in self._waits:          # every split command buffer waits as the frame's would
                cb.encodeWaitForEvent_value_(ev, v)
            self._cur = {"cb": cb, "kind": kind, "names": [], "used": False, "frame": self._p.frame}
            self._segs.append(self._cur)
        self._cur["kind"], self._cur["used"] = kind, True
        return self._cur

    def computeCommandEncoder(self):
        s = self._new("compute")
        return _Enc(s["cb"].computeCommandEncoder(), s, self._p)

    def renderCommandEncoderWithDescriptor_(self, d):
        s = self._new("render")
        return _Enc(s["cb"].renderCommandEncoderWithDescriptor_(d), s, self._p)

    def accelerationStructureCommandEncoder(self):
        s = self._new("accel")
        return _Enc(s["cb"].accelerationStructureCommandEncoder(), s, self._p)

    def blitCommandEncoder(self):
        s = self._new("blit")
        return _Enc(s["cb"].blitCommandEncoder(), s, self._p)

    def encodeWaitForEvent_value_(self, ev, v):
        self._waits.append((ev, v))

    def encodeSignalEvent_value_(self, ev, v):
        if self._cur is None:
            self._new("signal")
        self._cur["cb"].encodeSignalEvent_value_(ev, v)

    def commit(self):
        if self._cur is None:
            self._new("empty")
        self._cur["cb"].commit()
        self._cur = None
        self._done = True
        self._p.segments.extend(self._segs)

    def waitUntilCompleted(self):
        for s in self._segs:
            s["cb"].waitUntilCompleted()

    def status(self):
        return min((int(s["cb"].status()) for s in self._segs), default=int(Metal.MTLCommandBufferStatusCompleted))

    def error(self):
        for s in self._segs:
            if s["cb"].error() is not None:
                return s["cb"].error()
        return None


class PassProfiler:
    def __init__(self, ctx):
        self.ctx = ctx
        self.segments = []
        self.frame = 0
        self._orig = None
        self._cnames = None

    def compute_name(self, p):
        if self._cnames is None:
            self._cnames = {id(v): k[2] for k, v in self.ctx._pipelines.items() if k[0] == "c"}
        n = self._cnames.get(id(p))
        if n is None:
            self._cnames = {id(v): k[2] for k, v in self.ctx._pipelines.items() if k[0] == "c"}
            n = self._cnames.get(id(p), "kernel")
        return n

    def __enter__(self):
        self._orig = self.ctx.command_buffer
        orig = self._orig

        def command_buffer():
            orig()                      # keeps the context's bookkeeping (pending list) as before
            return _CB(self)
        self.ctx.command_buffer = command_buffer
        return self

    def __exit__(self, *a):
        self.ctx.command_buffer = self._orig
        return False

    def next_frame(self):
        self.frame += 1

    def times(self, skip_frames=0):
        """{label: [ms per occurrence]} over the recorded passes (synchronize first)."""
        out = defaultdict(list)
        for s in self.segments:
            if s["frame"] < skip_frames or not s["used"]:
                continue
            cb = s["cb"]
            cb.waitUntilCompleted()
            names = s["names"]
            label = s["kind"] + ":" + ("+".join(dict.fromkeys(names)) if names else "-")
            out[label].append(1e3 * (float(cb.GPUEndTime()) - float(cb.GPUStartTime())))
        return out

    def table(self, skip_frames=1):
        """Rows (label, ms per frame, share) sorted by cost, averaged over the recorded frames after ``skip_frames``."""
        t = self.times(skip_frames)
        nf = max(self.frame - skip_frames, 1)
        rows = sorted(((k, sum(v) / nf) for k, v in t.items()), key=lambda r: -r[1])
        tot = sum(r[1] for r in rows) or 1.0
        return [(k, round(v, 3), round(v / tot, 3)) for k, v in rows]
