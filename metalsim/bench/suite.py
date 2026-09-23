"""Reproduce-it-yourself benchmark suite: raw throughput plus the plan's compute-normalized figure.

Each row reports steps/s (or env-frames/s) and the same number divided by the hardware figure
that bounds the workload (peak FP32 TFLOPS for physics and shading, memory bandwidth for the
untile/copy stages), so results on different chips compare on equal terms. Hardware figures are
vendor-published peaks, taken from `HARDWARE` below; the M4 Max entry is what this machine has.

Usage: python -m metalsim.bench.suite [--quick] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time

import mujoco
import numpy as np
import torch
import warp as wp

from metalsim.physics.batch import BatchSim, BatchSimOptions
from metalsim.render.tier0 import Tier0Renderer
from metalsim.bench.render_bench import PRIMS

# vendor peak figures (TFLOPS FP32, GB/s). Apple figures from Apple's published specs; NVIDIA from
# NVIDIA's spec sheets. Ray-throughput normalization needs a traversal microbenchmark on both machines
# and is reported raw until one exists.
HARDWARE = {
    "Apple M4 Max (40-core GPU)": {"fp32_tflops": 18.4, "bandwidth_gbs": 546},
    "Apple M4 Max (32-core GPU)": {"fp32_tflops": 14.7, "bandwidth_gbs": 410},
    "Apple M3 Max (40-core GPU)": {"fp32_tflops": 14.2, "bandwidth_gbs": 400},
    "NVIDIA RTX 4090": {"fp32_tflops": 82.6, "bandwidth_gbs": 1008},
    "NVIDIA RTX 5090": {"fp32_tflops": 104.8, "bandwidth_gbs": 1792},
    "NVIDIA A100 80GB": {"fp32_tflops": 19.5, "bandwidth_gbs": 2039},
}


def gpu_name():
    try:
        out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=20).stdout
        chip = next(l.split(":")[1].strip() for l in out.splitlines() if "Chipset Model" in l)
        cores = next(l.split(":")[1].strip() for l in out.splitlines() if "Total Number of Cores" in l)
        return f"{chip} ({cores}-core GPU)"
    except Exception:
        return platform.processor()


def physics_rows(model, name, ns, substeps=1, steps=100):
    rows = []
    for n in ns:
        sim = BatchSim(model, n, options=BatchSimOptions(substeps=substeps))
        sim.step(); sim.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            sim.step()
        sim.synchronize()
        dt = time.perf_counter() - t0
        rows.append({"workload": f"physics {name}", "N": n, "value": n * steps * substeps / dt, "unit": "steps/s", "bound": "fp32"})
        del sim
    return rows


def render_rows(model, cam, name, grid, decimate=0, reps=50):
    rows = []
    for n, res in grid:
        sim = BatchSim(model, n)
        rend = Tier0Renderer(model, n, width=res, height=res, camera=cam, outputs=("rgb",), decimate_faces=decimate)
        sim.forward(); sim.synchronize(); v = sim.event.value
        for _ in range(3):
            rend.render(sim, v)
        rend.ctx.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            rend.render(sim, v)
        rend.ctx.synchronize()
        dt = time.perf_counter() - t0
        rows.append({"workload": f"render tier-0 {name} {res}px", "N": n, "value": n * reps / dt, "unit": "env-frames/s", "bound": "fp32"})
        del rend, sim
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    wp.config.quiet = True
    hw = gpu_name()
    spec = HARDWARE.get(hw, {"fp32_tflops": float("nan"), "bandwidth_gbs": float("nan")})
    rows = []
    prims = mujoco.MjModel.from_xml_string(PRIMS)
    so101 = mujoco.MjModel.from_xml_path("assets/so101/scene_box_rl.xml")
    ns = [64, 1024] if args.quick else [64, 256, 1024, 4096]
    rows += physics_rows(so101, "SO-101 lift", ns)
    rows += render_rows(prims, "cam", "primitives", [(64, 64), (1024, 64), (64, 128), (1024, 128)] if not args.quick else [(1024, 64)])
    rows += render_rows(so101, "base_cam", "SO-101 (2000 faces/mesh LOD)", [(64, 128), (1024, 64)], decimate=2000)
    print(f"machine: {hw}; normalization: {spec}")
    print("| workload | N | raw | per TFLOPS FP32 |")
    print("|---|---|---|---|")
    for r in rows:
        r["normalized"] = r["value"] / spec["fp32_tflops"]
        print(f"| {r['workload']} | {r['N']} | {r['value']:,.0f} {r['unit']} | {r['normalized']:,.0f} |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"machine": hw, "hardware": spec, "rows": rows, "torch": torch.__version__, "warp": wp.__version__}, f, indent=1)


if __name__ == "__main__":
    main()
