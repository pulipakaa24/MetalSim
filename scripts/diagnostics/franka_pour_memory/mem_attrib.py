"""GPU memory breakdown of Isaac Lab Franka-Pour on Metal (the task's PourPhysics, capture_unit "sim").

Every Metal allocation Warp makes through its allocator is recorded with its size, the phase (build / eager first
policy step / capture / replays) and an owner taken from the Python stack (subsystem + the innermost non-Warp-core
function). Allocations made during a capture whose pointer equals a buffer freed earlier in the same capture are
counted as reuse (Warp's capture memory pool), not as new memory. The device's own tally
(MTLDevice.currentAllocatedSize, via PyObjC) and the process footprint are recorded at each phase.

    mem_attrib.py NUM_ENVS [REPLAYS] [JSON]
"""
from __future__ import annotations

import collections
import ctypes
import json
import os
import sys
import time
import traceback

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
REPLAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
OUT = sys.argv[3] if len(sys.argv) > 3 else None

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True
wp.init()
import warp._src.context as ctx  # noqa: E402

try:
    import Metal  # noqa: E402

    _MTL = Metal.MTLCreateSystemDefaultDevice()
except Exception:  # noqa: BLE001
    _MTL = None


def rusage_v2():
    class RU(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(40)]
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    ru = RU()
    lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(ru))
    return ru


def footprint_bytes() -> int:
    """phys_footprint of this process (proc_pid_rusage, RUSAGE_INFO_V2)."""
    class RU(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(40)]
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    ru = RU()
    if lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(ru)) != 0:
        return -1
    return int(ru.f7)  # ri_phys_footprint: the 8th 64-bit field after the uuid (rusage_info_v2)


def device_bytes() -> int:
    return int(_MTL.currentAllocatedSize()) if _MTL is not None else -1


SUBSYSTEMS = [  # priority order: the first subsystem found anywhere on the stack names the owner
    ("implicit_mpm", "MPM"), ("warp/_src/fem", "MPM(fem)"), ("mujoco_warp", "MuJoCo Warp"),
    ("solvers/mujoco", "SolverMuJoCo"), ("collision", "collision"), ("narrow_phase", "collision"),
    ("broad_phase", "collision"), ("contact", "collision"), ("geometry/", "collision"),
    ("experimental/coupled", "coupler"), ("newton/_src/sim", "model/state"), ("metalsim/learn", "task"),
]
CORE = ("warp/_src/context.py", "warp/_src/types.py", "warp/_src/utils.py", "warp/_src/fem/cache.py",
        "mem_attrib.py", "warp/_src/fem/linalg.py")


def owner_of(stack) -> str:
    files = [fr.filename.replace("\\", "/") for fr in stack]
    sub = "other"
    for key, name in SUBSYSTEMS:
        if any(key in f for f in files):
            sub = name
            break
    site = "?"
    for fr in reversed(stack):
        f = fr.filename.replace("\\", "/")
        if not any(c in f for c in CORE):
            site = f"{os.path.basename(f)}:{fr.name}"
            break
    return f"{sub} | {site}"


class Tally:
    def __init__(self):
        self.phase = "build"
        self.live = {}  # ptr -> (size, owner, phase, capture_id)
        self.capture_id = 0
        self.freed_in_capture = {}  # ptr -> size (current capture)
        self.events = collections.Counter()
        self.bytes_alloc = collections.Counter()  # (phase, owner) -> bytes allocated (new memory)
        self.bytes_reused = collections.Counter()
        self.capture_freed = collections.Counter()  # owner -> bytes allocated and freed inside a capture (graph-held)
        self.samples = []

    def sample(self, label):
        self.samples.append({"label": label, "device_bytes": device_bytes(), "footprint_bytes": footprint_bytes(),
                             "warp_live_bytes": sum(v[0] for v in self.live.values()),
                             "native_tracked_bytes": int(_core.wp_alloc_tracker_get_current_bytes()),
                             "resident_bytes": int(rusage_v2().f6)})
        s = self.samples[-1]
        print(f"[mem] {label:28s} device {s['device_bytes']/1e9:7.2f} GB  footprint {s['footprint_bytes']/1e9:7.2f} GB"
              f"  resident {s['resident_bytes']/1e9:7.2f} GB  warp-live {s['warp_live_bytes']/1e9:7.2f} GB  native-tracked {s['native_tracked_bytes']/1e9:7.2f} GB", flush=True)


_core = ctx.runtime.core
_core.wp_alloc_tracker_enable.argtypes = [ctypes.c_int]
_core.wp_alloc_tracker_get_current_bytes.restype = ctypes.c_size_t
_core.wp_alloc_tracker_report.restype = ctypes.c_size_t
_core.wp_alloc_tracker_report.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
_core.wp_alloc_tracker_enable(1)


def native_report(max_items=25) -> str:
    buf = ctypes.create_string_buffer(1 << 22)
    n = _core.wp_alloc_tracker_report(buf, len(buf), 0, max_items)
    return buf.value[:n].decode(errors="replace")


T = Tally()
_orig_alloc = ctx.MetalDefaultAllocator.allocate
_orig_free = ctx.MetalDefaultAllocator.deallocate


def _alloc(self, size):
    ptr = _orig_alloc(self, size)
    capturing = bool(getattr(self.device, "is_capturing", False))
    owner = owner_of(traceback.extract_stack()[:-1])
    reused = capturing and ptr in T.freed_in_capture
    if reused:
        T.freed_in_capture.pop(ptr)
        T.bytes_reused[(T.phase, owner)] += size
    else:
        T.bytes_alloc[(T.phase, owner)] += size
    T.live[ptr] = (size, owner, T.phase, T.capture_id if capturing else None)
    return ptr


GRAPH_HELD = collections.Counter()  # (kind, owner) -> bytes the graph keeps alive (see wp_free_metal)


def _free(self, ptr, size):
    rec = T.live.pop(ptr, None)
    capturing = bool(getattr(self.device, "is_capturing", False))
    if rec is not None:
        sz, owner, phase, cid = rec
        if capturing and cid == T.capture_id:
            T.freed_in_capture[ptr] = sz
            T.capture_freed[owner] += sz
            if sz >= 64 << 10:
                GRAPH_HELD[("capture pool", owner)] += sz  # reusable by later allocations of the same capture
            else:
                GRAPH_HELD[("retained: small capture temp", owner)] += sz
        elif capturing:
            GRAPH_HELD[("retained: allocated before capture, freed during", owner)] += sz
        elif cid is not None:
            GRAPH_HELD[("retained: allocated in capture, freed after", owner)] += sz
    return _orig_free(self, ptr, size)


ctx.MetalDefaultAllocator.allocate = _alloc
ctx.MetalDefaultAllocator.deallocate = _free

_orig_begin = ctx.capture_begin if hasattr(ctx, "capture_begin") else None


def _begin(*a, **k):
    T.capture_id += 1
    T.freed_in_capture = {}
    return _orig_begin(*a, **k)


if _orig_begin is not None:
    ctx.capture_begin = _begin
    wp.capture_begin = _begin

# TemporaryStore pools: requested vs handed-out bytes of the borrows outstanding at once (fragmentation)
import warp._src.fem.cache as _fc  # noqa: E402

POOLSTAT = collections.defaultdict(lambda: {"out_req": 0, "out_cap": 0, "peak_req": 0, "peak_cap": 0, "n": 0})
_OUT = {}
EVENTS = []
_orig_borrow = _fc.TemporaryStore.Pool.borrow
_orig_redeem = _fc.TemporaryStore.Pool.redeem


def _pborrow(self, shape, dtype, requires_grad):
    t = _orig_borrow(self, shape, dtype, requires_grad)
    if t.ptr:
        req = int(np.prod(shape if not isinstance(shape, int) else (shape,))) * self._dtype_size
        st = POOLSTAT[str(self.dtype.__name__ if hasattr(self.dtype, "__name__") else self.dtype)]
        st["out_req"] += req; st["out_cap"] += t.capacity; st["n"] += 1
        st["peak_req"] = max(st["peak_req"], st["out_req"]); st["peak_cap"] = max(st["peak_cap"], st["out_cap"])
        _OUT[t.ptr] = (st, req, t.capacity)
        EVENTS.append(("b", str(self.dtype), req, t.ptr, T.phase))
    return t


def _predeem(self, ptr):
    rec = _OUT.pop(ptr, None)
    EVENTS.append(("r", str(self.dtype), 0, ptr, T.phase))
    if rec is not None:
        rec[0]["out_req"] -= rec[1]; rec[0]["out_cap"] -= rec[2]
    return _orig_redeem(self, ptr)


_fc.TemporaryStore.Pool.borrow = _pborrow
_fc.TemporaryStore.Pool.redeem = _predeem

VOLUMES = []
for _name in ("allocate_by_voxels", "allocate_by_tiles"):
    _orig_v = getattr(wp.Volume, _name)

    def _wrap(*a, _orig_v=_orig_v, _name=_name, **k):
        n0 = int(_core.wp_alloc_tracker_get_current_bytes())
        v = _orig_v(*a, **k)
        n1 = int(_core.wp_alloc_tracker_get_current_bytes())
        caps = {kk: k.get(kk) for kk in ("max_active_voxels", "max_tiles", "max_leaf_nodes", "max_lower_nodes",
                                          "max_upper_nodes", "rebuildable")}
        site = owner_of(traceback.extract_stack()[:-1])
        VOLUMES.append({"fn": _name, "bytes": n1 - n0, "caps": caps, "owner": site, "phase": T.phase})
        print(f"[vol] {_name} +{(n1 - n0) / 1e9:.3f} GB caps {caps} at {site}", flush=True)
        return v

    setattr(wp.Volume, _name, staticmethod(_wrap) if isinstance(wp.Volume.__dict__.get(_name), staticmethod)
            else classmethod(lambda cls, *a, _w=_wrap, **k: _w(*a, **k)) if isinstance(wp.Volume.__dict__.get(_name), classmethod)
            else _wrap)

from metalsim.learn import franka_pour_scene as S  # noqa: E402

t0 = time.perf_counter()
T.sample("start")
cfg = S.PourPhysicsConfig(num_envs=N, rigid_deterministic=bool(int(os.environ.get("RIGID_DET", "0"))))
phys = S.PourPhysics(cfg)
wp.synchronize_device()
T.sample("built")
import gc  # noqa: E402

gc.collect()
wp.synchronize_device()
T.sample("built, after gc")
open((OUT or "mem") + ".native_built.txt", "w").write(native_report())
T.phase = "eager"
# the first policy step: eager steps, then the capture of one sim step (capture_unit "sim")
orig_capture = phys.capture


def capture_tagged():
    T.phase = "capture"
    orig_capture()
    T.phase = "replay"


phys.capture = capture_tagged
phys.policy_step()
wp.synchronize_device()
T.sample("after first policy step")
for _ in range(REPLAYS):
    phys.policy_step()
wp.synchronize_device()
T.sample(f"after {REPLAYS} replayed policy steps")
gc.collect()
wp.synchronize_device()
T.sample("after gc")

by_owner_live = collections.Counter()
by_phase_live = collections.Counter()
for size, owner, phase, cid in T.live.values():
    by_owner_live[(phase, owner)] += size
    by_phase_live[phase] += size
res = {
    "num_envs": N, "rigid_deterministic": cfg.rigid_deterministic, "seconds": time.perf_counter() - t0,
    "samples": T.samples,
    "live_by_phase": dict(by_phase_live),
    "live_by_owner": {f"{p} || {o}": b for (p, o), b in by_owner_live.most_common()},
    "allocated_by_owner": {f"{p} || {o}": b for (p, o), b in T.bytes_alloc.most_common()},
    "reused_in_capture_by_owner": {f"{p} || {o}": b for (p, o), b in T.bytes_reused.most_common()},
    "freed_inside_capture_by_owner": dict(T.capture_freed.most_common()),
    "freed_inside_capture_total": sum(T.capture_freed.values()),
    "reused_total": sum(T.bytes_reused.values()),
}
sub_live = collections.Counter(); sub_cap = collections.Counter(); sub_freed = collections.Counter()
for (p, o), b in by_owner_live.items():
    sub_live[(p, o.split(" | ")[0])] += b
for (p, o), b in T.bytes_alloc.items():
    if p == "capture":
        sub_cap[o.split(" | ")[0]] += b
for o, b in T.capture_freed.items():
    sub_freed[o.split(" | ")[0]] += b
res["live_by_subsystem"] = {f"{p} || {o}": b for (p, o), b in sub_live.most_common()}
res["capture_new_by_subsystem"] = dict(sub_cap.most_common())
res["capture_freed_by_subsystem"] = dict(sub_freed.most_common())
print("live by phase / subsystem:")
for k, b in res["live_by_subsystem"].items():
    print(f"  {b/1e9:8.3f} GB  {k}")
print("new memory taken during capture, by subsystem:", {k: round(v / 1e9, 3) for k, v in sub_cap.items()})
print("freed inside capture (graph-held unless reused), by subsystem:", {k: round(v / 1e9, 3) for k, v in sub_freed.items()})
print(json.dumps({k: v for k, v in res.items() if k in ("live_by_phase", "freed_inside_capture_total", "reused_total")},
                 indent=1))
for k, b in list(res["live_by_owner"].items())[:40]:
    print(f"  live {b/1e9:8.3f} GB  {k}")
print("allocated during capture (new memory), top:")
for k, b in [(k, b) for k, b in res["allocated_by_owner"].items() if k.startswith("capture")][:30]:
    print(f"  cap  {b/1e9:8.3f} GB  {k}")
res["volumes"] = VOLUMES
json.dump(EVENTS, open((OUT or "mem") + ".pool_events.json", "w"))
res["temporary_pool_stats"] = {k: dict(v) for k, v in POOLSTAT.items()}
print("TemporaryStore pools: peak requested / peak handed out (GB):",
      {k: (round(v["peak_req"] / 1e9, 3), round(v["peak_cap"] / 1e9, 3)) for k, v in POOLSTAT.items()})
# graph-held memory: pooled capture buffers net of reuse (a reused buffer is live again, counted there)
pool_net = sum(b for (k, o), b in GRAPH_HELD.items() if k == "capture pool") - res["reused_total"]
res["graph_held_by_kind"] = {}
for (k, o), b in GRAPH_HELD.items():
    res["graph_held_by_kind"].setdefault(k, 0)
    res["graph_held_by_kind"][k] += b
res["graph_held_pool_net_of_reuse"] = pool_net
res["graph_held_by_owner"] = {f"{k} || {o}": b for (k, o), b in GRAPH_HELD.most_common(40)}
print("graph-held by kind (GB):", {k: round(v / 1e9, 3) for k, v in res["graph_held_by_kind"].items()},
      "pool net of reuse", round(pool_net / 1e9, 3))
# individual live allocations (largest first) and per-site size histograms
res["live_allocations"] = sorted(([v[0], v[1], v[2]] for v in T.live.values()), key=lambda r: -r[0])[:400]
hist = collections.defaultdict(collections.Counter)
for size, owner, phase, _ in T.live.values():
    hist[owner][size] += 1
res["live_size_histogram"] = {o: sorted(([s, c] for s, c in h.items()), key=lambda r: -r[0] * r[1])[:20]
                              for o, h in sorted(hist.items(), key=lambda kv: -sum(s * c for s, c in kv[1].items()))[:30]}
# TemporaryStore pools (warp.fem): capacity held, available vs borrowed
try:
    import gc

    import warp._src.fem.cache as fc

    stores = [o for o in gc.get_objects() if isinstance(o, fc.TemporaryStore)]
    res["temporary_stores"] = []
    for st in stores:
        pools = getattr(st, "_temporaries", None) or getattr(st, "_pools", None) or {}
        tot = avail = 0
        for pool in (pools.values() if isinstance(pools, dict) else pools):
            for ptr, (cap, _d) in getattr(pool, "_allocs", {}).items():
                tot += cap
            avail += sum(getattr(pool, "_pool_capacities", []))
        res["temporary_stores"].append({"held_bytes": tot, "available_bytes": avail})
    print("TemporaryStores:", [(round(x["held_bytes"] / 1e9, 3), round(x["available_bytes"] / 1e9, 3))
                               for x in res["temporary_stores"]])
except Exception as e:  # noqa: BLE001
    res["temporary_stores_error"] = repr(e)
open((OUT or "mem") + ".native_end.txt", "w").write(native_report())
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
