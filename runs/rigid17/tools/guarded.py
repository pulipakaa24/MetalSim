"""Run ONE probe configuration with crash guards (the Mac panicked on 2026-10-11 after a GPU hang starved WindowServer).

    python3 guarded.py --tree new --log out.log [--max-rss-gb 40] [--timeout 900] [--stall 60] -- scale_pour args...

Guards:
  * refuses to start unless >= --min-free-gb of memory is free/inactive (vm_stat) and no other python3.12 probe runs;
  * refuses --worlds > 4096, and --det on / --pipeline-det above 2048 worlds;
  * kills the job (SIGKILL) when its resident memory exceeds --max-rss-gb, when the wall time exceeds --timeout, or
    when its CPU time has not advanced for --stall seconds (a GPU hang: the host sits in waitUntilCompleted).
Exit code: the job's, or 90 (refused), 91 (RSS), 92 (timeout), 93 (stall / GPU hang).
"""
import argparse
import os
import re
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--max-rss-gb", type=float, default=40.0)
ap.add_argument("--min-free-gb", type=float, default=24.0)
ap.add_argument("--timeout", type=float, default=900.0)
ap.add_argument("--stall", type=float, default=120.0)
ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the job")
ap.add_argument("rest", nargs=argparse.REMAINDER)
a = ap.parse_args()
rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
T = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[guarded {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = sum(int(re.search(rf"{k}:\s+(\d+)", out).group(1)) for k in ("Pages free", "Pages inactive", "Pages speculative"))
    return pages * page / 2**30


def ps(pid):
    """(rss GB, cpu seconds) of pid, or None if gone."""
    out = subprocess.run(["ps", "-o", "rss=,time=", "-p", str(pid)], capture_output=True, text=True).stdout.split()
    if len(out) < 2:
        return None
    t = out[1].split(":")
    secs = float(t[-1]) + 60 * int(t[-2]) + (3600 * int(t[-3]) if len(t) > 2 else 0)
    return int(out[0]) / 2**20, secs


def arg(name, default=None):
    return rest[rest.index(name) + 1] if name in rest else default


worlds = int(arg("--worlds", "1"))
det = "--det" in rest
if worlds > 4096 or (det and worlds > 2048):
    log(f"refused: worlds={worlds} det={det}")
    sys.exit(90)
# running probe interpreters only (not gpu_run.sh / gpu_lock.py waiters whose command lines mention the probe)
others = [ln.split()[0] for ln in subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True,
                                                    text=True).stdout.splitlines()
          if "scale_pour.py" in ln and ln.split()[1].split("/")[-1].startswith("python")
          and "gpu_lock.py" not in ln and "gpu_run.sh" not in ln]
if others and worlds >= 2048:  # big configs run alone (timing slot); small check jobs may share the GPU
    log(f"refused: other scale_pour processes running {others}")
    sys.exit(90)
fg = free_gb()
if fg < a.min_free_gb:
    log(f"refused: only {fg:.1f} GB free/inactive")
    sys.exit(90)

env = dict(os.environ)
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v
cmd = ["bash", "-c", f'source {T}/env.sh {a.tree}; exec $PY $SP "$@"', "guarded"] + rest
log(f"start tree={a.tree} free={fg:.1f}GB: {' '.join(rest)}")
with open(a.log, "w") as fh:
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
t0 = time.time()
last_cpu, last_change, peak = -1.0, time.time(), 0.0
code = None
while True:
    rc = p.poll()
    if rc is not None:
        code = rc
        break
    s = ps(p.pid)
    now = time.time()
    if s:
        rss, cpu = s
        peak = max(peak, rss)
        if cpu > last_cpu + 0.05:
            last_cpu, last_change = cpu, now
        if rss > a.max_rss_gb:
            log(f"KILL: rss {rss:.1f} GB > {a.max_rss_gb}")
            p.kill()
            code = 91
            break
        if now - last_change > a.stall:
            log(f"KILL: cpu time stalled {now - last_change:.0f}s at {cpu:.1f}s (GPU hang?)")
            subprocess.run(["sample", str(p.pid), "1", "-file", a.log + ".sample"], capture_output=True)
            p.kill()
            code = 93
            break
    if now - t0 > a.timeout:
        log(f"KILL: timeout {a.timeout}s")
        p.kill()
        code = 92
        break
    time.sleep(2)
p.wait()
log(f"end code={code} wall={time.time() - t0:.0f}s peak_rss={peak:.1f}GB")
sys.exit(code)
