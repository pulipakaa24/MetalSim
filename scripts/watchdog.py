#!/usr/bin/env python3
"""Run a command under a watchdog: kill its process tree on a CPU stall, a memory ceiling or a timeout.

    python3 scripts/watchdog.py [--stall 120] [--rss-gb 40] [--timeout 3600] [--cwd DIR] -- command args...

A GPU hang on Metal shows up as a process whose CPU time stops advancing (it blocks in a command-buffer wait);
it once took the whole machine down, so every large-world or new-kernel job runs under this. The tree's CPU time
(user + system, summed over the command and its descendants) is sampled every 5 s with ``ps``; when it has not
advanced for ``--stall`` seconds, or the tree's resident memory exceeds ``--rss-gb``, or ``--timeout`` seconds have
passed, the tree gets SIGTERM, then SIGKILL 10 s later. Exit code: the command's, or 124 (timeout), 125 (stall),
126 (memory). Standard library only (the system python has no psutil).
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def _tree(root: int) -> list[int]:
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    kids: dict[int, list[int]] = {}
    for line in out.splitlines():
        p, pp = (int(x) for x in line.split())
        kids.setdefault(pp, []).append(p)
    seen, todo = [], [root]
    while todo:
        p = todo.pop()
        seen.append(p)
        todo.extend(kids.get(p, []))
    return seen


def _cpu_seconds(t: str) -> float:
    # ps "time" is [[dd-]hh:]mm:ss.cc
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    parts = [float(x) for x in t.split(":")]
    s = 0.0
    for x in parts:
        s = s * 60 + x
    return s + 86400 * days


def _sample(pids: list[int]) -> tuple[float, float]:
    out = subprocess.run(["ps", "-o", "rss=,time=", "-p", ",".join(map(str, pids))], capture_output=True,
                         text=True).stdout
    cpu, rss_kb = 0.0, 0
    for line in out.splitlines():
        f = line.split()
        if len(f) == 2:
            rss_kb += int(f[0])
            cpu += _cpu_seconds(f[1])
    return cpu, rss_kb / 2**20


def _kill(pids: list[int]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in reversed(pids):
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        time.sleep(10 if sig == signal.SIGTERM else 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stall", type=float, default=120.0, help="seconds without CPU-time progress before killing")
    ap.add_argument("--rss-gb", type=float, default=40.0)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--cwd", default=None, help="working directory of the command")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    proc = subprocess.Popen(cmd, cwd=a.cwd)

    def forward(signum, _frame):     # the queue wrapper stops a job by signalling this process: stop the tree too
        _kill(_tree(proc.pid))
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)
    t0 = last_progress = time.time()
    last_cpu = -1.0
    reason = None
    while proc.poll() is None:
        time.sleep(5)
        pids = _tree(proc.pid)
        cpu, rss = _sample(pids)
        now = time.time()
        if cpu > last_cpu + 0.05:
            last_cpu, last_progress = cpu, now
        if now - last_progress > a.stall:
            reason, code = f"CPU stall {now - last_progress:.0f} s (cpu {cpu:.1f} s)", 125
        elif rss > a.rss_gb:
            reason, code = f"RSS {rss:.1f} GB > {a.rss_gb}", 126
        elif now - t0 > a.timeout:
            reason, code = f"timeout {a.timeout:.0f} s", 124
        if reason:
            print(f"[watchdog] killing {cmd[:3]}: {reason}", file=sys.stderr, flush=True)
            _kill(pids)
            proc.wait()
            return code
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
