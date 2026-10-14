"""Run a command and sample the GPU's IOAccelerator "Device Utilization %" (ioreg, no root) every --interval s while
it runs (research_notes/gpu_latency). Device utilization counts time with work queued, not ALU occupancy: a stream of
short latency-bound dispatches reads ~100 %. Package power needs `sudo powermetrics --samplers gpu_power` (printed as a
hint; not run here).

    python3 gpu_busy_sampler.py --out samples.json [--interval 0.2] -- cmd args...
"""
import argparse
import json
import re
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--interval", type=float, default=0.2)
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd


def util():
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
    m = re.search(r'"Device Utilization %"=(\d+)', out)
    return int(m.group(1)) if m else None


print("hint: package power: in another terminal, `sudo powermetrics --samplers gpu_power -i 500` while this runs",
      file=sys.stderr)
p = subprocess.Popen(cmd)
t0 = time.time()
samples = []
while p.poll() is None:
    samples.append((round(time.time() - t0, 2), util()))
    time.sleep(a.interval)
rc = p.wait()
vals = [u for _, u in samples if u is not None]
res = {"cmd": cmd, "rc": rc, "seconds": round(time.time() - t0, 2), "samples": samples,
       "util_mean": sum(vals) / len(vals) if vals else None,
       "util_ge_90_fraction": sum(v >= 90 for v in vals) / len(vals) if vals else None}
json.dump(res, open(a.out, "w"), indent=1)
print(f"device utilization mean {res['util_mean']}, >=90 % in {res['util_ge_90_fraction']} of samples", file=sys.stderr)
sys.exit(rc)
