"""Run a command and kill it if it stalls or overruns (a hung GPU kernel once took a Mac down).

    python3 scripts/particle_demo/watchdog.py --timeout 3600 --stall 300 --log out.log -- cmd args...

The child's stdout/stderr go to ``--log``; "stall" = no new output for ``--stall`` seconds. The process group is
killed (SIGTERM, then SIGKILL after 10 s). Exit code: the child's, or 124 on a timeout / stall.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--timeout", type=float, default=3600)
ap.add_argument("--stall", type=float, default=300)
ap.add_argument("--log", required=True)
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
with open(a.log, "ab") as log:
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    last_size, last_change = 0, t0
    reason = None
    while p.poll() is None:
        time.sleep(2)
        size = os.path.getsize(a.log)
        now = time.time()
        if size != last_size:
            last_size, last_change = size, now
        if now - t0 > a.timeout:
            reason = f"timeout {a.timeout:.0f} s"
        elif now - last_change > a.stall:
            reason = f"no output for {a.stall:.0f} s"
        if reason:
            os.killpg(p.pid, signal.SIGTERM)
            for _ in range(10):
                if p.poll() is not None:
                    break
                time.sleep(1)
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            break
with open(a.log, "a") as log:
    log.write(f"\n[watchdog] {'killed: ' + reason if reason else 'exit ' + str(p.returncode)} after {time.time() - t0:.0f} s\n")
sys.exit(124 if reason else p.returncode)
