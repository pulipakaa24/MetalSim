"""Peak Metal allocation and peak memory footprint of a Python command, run in this process.

usage: python scripts/diagnostics/peak_memory.py OUT.json -m module args...  (or: OUT.json script.py args...)

peak_metal_gb: MTLDevice.currentAllocatedSize, the quantity Metal compares against recommendedMaxWorkingSetSize,
sampled every 10 ms in a thread (a peak shorter than that can be missed). peak_footprint_gb: the process's lifetime
peak physical footprint (proc_pid_rusage), what macOS charges against RAM, GPU memory included; peak_rss_gb misses
GPU-written memory (a filled 2 GB Warp array: footprint 2.27 GB, RSS 0.11 GB). Memory peaks at start-up, graph
capture and the first iteration, so a short run gives the peak. The result is written even if the command raises or
calls sys.exit.
"""
import json
import os
import resource
import runpy
import sys
import threading
import time

import Metal

out, rest = sys.argv[1], sys.argv[2:]
dev = Metal.MTLCreateSystemDefaultDevice()
peak = {"metal": 0, "t": 0.0}
stop = threading.Event()
t0 = time.time()


def sample():
    while not stop.is_set():
        v = int(dev.currentAllocatedSize())
        if v > peak["metal"]:
            peak["metal"], peak["t"] = v, time.time() - t0
        time.sleep(0.01)


th = threading.Thread(target=sample, daemon=True)
th.start()
status = "ok"
try:
    if rest[0] == "-m":
        sys.argv = [rest[1]] + rest[2:]
        runpy.run_module(rest[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = rest
        sys.path.insert(0, os.path.dirname(os.path.abspath(rest[0])))
        runpy.run_path(rest[0], run_name="__main__")
except SystemExit as e:
    status = f"exit {e.code}"
except BaseException as e:  # noqa: BLE001
    status = f"{type(e).__name__}: {str(e)[:300]}"
finally:
    stop.set()
    th.join()
    peak["metal"] = max(peak["metal"], int(dev.currentAllocatedSize()))
    # rusage_info_v4: 16-byte uuid, then uint64 fields; [7] ri_phys_footprint, [28] ri_lifetime_max_phys_footprint
    import ctypes
    buf = (ctypes.c_uint64 * 48)()
    libc = ctypes.CDLL(None)
    ok = libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) == 0
    u64 = (ctypes.c_uint64 * 46).from_buffer(buf, 16)
    rec = dict(cmd=" ".join(rest), status=status, wall_s=round(time.time() - t0, 1),
               peak_metal_gb=round(peak["metal"] / 1e9, 2), peak_metal_at_s=round(peak["t"], 1),
               peak_footprint_gb=round(u64[28] / 1e9, 2) if ok else None,
               footprint_now_gb=round(u64[7] / 1e9, 2) if ok else None,
               peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
               working_set_gb=round(int(dev.recommendedMaxWorkingSetSize()) / 1e9, 2), device=str(dev.name()))
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("PEAKMEM", json.dumps(rec), file=sys.stderr)
