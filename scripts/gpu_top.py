#!/usr/bin/env python3
"""Per-process GPU use on Apple Silicon without sudo: samples the AGX driver's per-client ``accumulatedGPUTime``
(ns of GPU time, IORegistry ``AGXDeviceUserClient`` -> ``AppUsage``) twice and prints each process's share of the
interval, plus the device's own "Device Utilization %". Used in scripts/gpu_run.sh's log header.

    python3 scripts/gpu_top.py [--interval 1.0] [--min 1.0]
"""
import argparse, plistlib, re, subprocess, time


def sample():
    raw = subprocess.run(["ioreg", "-a", "-l", "-w", "0", "-c", "AGXDeviceUserClient"], capture_output=True).stdout
    per = {}
    if raw.strip():
        def walk(o):
            if isinstance(o, dict):
                cr = o.get("IOUserClientCreator")
                if isinstance(cr, str):
                    m = re.match(r"pid (\d+), (.*)", cr)
                    if m:
                        t = sum(int(u.get("accumulatedGPUTime", 0)) for u in o.get("AppUsage", []) or [] if isinstance(u, dict))
                        pid = int(m.group(1)); p = per.setdefault(pid, [m.group(2), 0]); p[1] += t
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(plistlib.loads(raw))
    return per


def device_util():
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
    m = re.search(r'"Device Utilization %"=(\d+)', out)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--interval", type=float, default=1.0); ap.add_argument("--min", type=float, default=1.0)
    a = ap.parse_args()
    s0 = sample(); t0 = time.monotonic(); time.sleep(a.interval); s1 = sample(); dt = time.monotonic() - t0
    rows = []
    for pid, (name, t) in s1.items():
        d = t - s0.get(pid, [name, t])[1]
        share = 100.0 * d / (dt * 1e9)
        if share >= a.min:
            rows.append((share, pid, name))
    print(f"GPU (ioreg): device utilization {device_util()} %; per-process GPU time over {dt:.1f} s:"
          + ("".join(f"\n  pid {pid:>6} {name:<32} {share:5.1f} %" for share, pid, name in sorted(rows, reverse=True)) or " none >= %.0f %%" % a.min))


if __name__ == "__main__":
    main()
