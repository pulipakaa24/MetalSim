"""Run a command under a GPU-hang watchdog (a hung Metal command buffer once panicked the Mac, MPM.md 8c).

    python3 watch.py --log out.log [--timeout 1800] [--stall 120] -- cmd args...

Kills the job (SIGKILL, whole process group) when the wall time exceeds --timeout or when the CPU time of the job and
its children has not advanced for --stall seconds (a GPU hang: the host sits in waitUntilCompleted). Exit code: the
job's, 92 (timeout) or 93 (stall).
"""
import argparse
import os
import signal
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--timeout", type=float, default=1800.0)
ap.add_argument("--stall", type=float, default=120.0)
ap.add_argument("rest", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest


def log(msg):
    print(f"[watch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tree_cpu(pgid):
    """Total CPU seconds of the processes in the job's process group."""
    out = subprocess.run(["ps", "-g", str(pgid), "-o", "time="], capture_output=True, text=True).stdout.split()
    total = 0.0
    for t in out:
        p = t.split(":")
        total += float(p[-1]) + 60 * int(p[-2]) + (3600 * int(p[-3]) if len(p) > 2 else 0)
    return total


log(f"start: {' '.join(cmd)}")
with open(a.log, "w") as fh:
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
t0 = time.time()
last_cpu, last_change, code = -1.0, time.time(), None
while True:
    rc = p.poll()
    if rc is not None:
        code = rc
        break
    now = time.time()
    cpu = tree_cpu(p.pid)
    if cpu > last_cpu + 0.05:
        last_cpu, last_change = cpu, now
    if now - last_change > a.stall:
        log(f"KILL: cpu time stalled {now - last_change:.0f}s at {cpu:.1f}s (GPU hang?)")
        subprocess.run(["sample", str(p.pid), "1", "-file", a.log + ".sample"], capture_output=True)
        os.killpg(p.pid, signal.SIGKILL)
        code = 93
        break
    if now - t0 > a.timeout:
        log(f"KILL: timeout {a.timeout}s")
        os.killpg(p.pid, signal.SIGKILL)
        code = 92
        break
    time.sleep(2)
p.wait()
log(f"end code={code} wall={time.time() - t0:.0f}s")
sys.exit(code)
