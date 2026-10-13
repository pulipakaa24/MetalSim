"""Summarize scale_pour JSON outputs: python3 summ.py 'out/*.json'"""
import glob
import json
import sys

for pat in sys.argv[1:]:
    for f in sorted(glob.glob(pat)):
        try:
            r = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f, "unreadable", e)
            continue
        name = f.split("/")[-1]
        if "kprof" in r:
            k = r["kprof"]
            print(name, f"{k['phase']} wall/rep {k['wall_ms_per_rep']:.2f}", k.get("per_rep_counters"),
                  "contacts", r.get("rigid_contacts"), "kernel total", round(k.get("kernel_total_ms", 0), 2))
            for row in k.get("kernels", [])[:int(sys.argv[0] and 12)]:
                print(f"   {row['ms']:9.3f} {row['launches']:7.1f}  {row['kernel'][:105]}")
            continue
        keys = ["collide_ms", "substep_ms", "tick_ms", "tick_ms_captured", "rigid_contacts", "nefc_mean", "capture"]
        out = {k: (round(r[k]["median"], 2) if isinstance(r.get(k), dict) and "median" in r[k] else r.get(k)) for k in keys}
        print(name, out)
        for rk in ("repeat_check", "repeat_check_captured"):
            if rk in r:
                print("   ", rk, r[rk])
