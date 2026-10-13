#!/usr/bin/env python3
"""Run a command under a watchdog: kill its process group on a timeout or when its CPU time stops advancing.

    python3 scripts/diagnostics/watchdog_run.py [--timeout 900] [--stall 90] -- command args...

A Metal GPU hang shows up as a process blocked in a command-buffer wait (its CPU time stops advancing); one took the
whole machine down once, so jobs that run new GPU kernels go through this. Every 5 s the CPU time of the command's
process group is sampled with ``ps``; no progress for ``--stall`` seconds, or ``--timeout`` seconds in total, sends
SIGTERM to the group, then SIGKILL 5 s later. Exit code: the command's, 124 on timeout, 125 on a stall.
Standard library only.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def _cpu(pgid: int) -> float:
    out = subprocess.run(["ps", "-A", "-o", "pgid=,time="], capture_output=True, text=True).stdout
    tot = 0.0
    for line in out.splitlines():
        f = line.split()
        if len(f) != 2 or int(f[0]) != pgid:
            continue
        t = f[1]
        days = 0
        if "-" in t:
            d, t = t.split("-", 1)
            days = int(d)
        s = 0.0
        for x in t.split(":"):
            s = s * 60 + float(x)
        tot += s + 86400 * days
    return tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--stall", type=float, default=90.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    p = subprocess.Popen(cmd, start_new_session=True)
    pgid = p.pid
    t0 = last_t = time.time()
    last_cpu = -1.0
    rc = None
    while rc is None:
        try:
            rc = p.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.time()
        cpu = _cpu(pgid)
        if cpu > last_cpu + 0.01:
            last_cpu, last_t = cpu, now
        why = 124 if now - t0 > a.timeout else (125 if now - last_t > a.stall else 0)
        if why:
            print(f"[watchdog] {'timeout' if why == 124 else 'stall'} after {now - t0:.0f} s: killing {cmd[0]}", file=sys.stderr, flush=True)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                time.sleep(5)
            p.wait()
            return why
    return rc


if __name__ == "__main__":
    sys.exit(main())
