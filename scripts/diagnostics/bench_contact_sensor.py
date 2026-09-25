"""Per-step cost of the contact sensor (metalsim.sensors.contact) on the G1 task's MuJoCo Warp
setting: 4096 worlds, 5 ms physics step, 4 substeps per control step, nconmax 32, njmax 256,
10 Newton / 20 line-search iterations. Each configuration is timed as graph replays of sim.step()
with a device synchronize around the timed block (A/B alternated 3 times; median reported).

    scripts/gpu_run.sh contact_sensor_bench timing 10 -- .venv/bin/python scripts/diagnostics/bench_contact_sensor.py
"""
import sys
import time

import numpy as np
import torch
import warp as wp

from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics.batch import BatchSim, BatchSimOptions
from metalsim.sensors.contact import ContactSensor

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
STEPS = 50


def timed(sim, steps=STEPS):
    for _ in range(5):
        sim.step()
    sim.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step()
    sim.synchronize()
    return (time.perf_counter() - t0) / steps * 1e3


def main():
    m, _ = build_g1_model("flat")
    sim = BatchSim(m, N, options=BatchSimOptions(substeps=4, njmax=256, nconmax=32, solver_iterations=10, ls_iterations=20))
    q = np.tile(m.key_qpos[0], (N, 1)).astype(np.float32)
    q[:, 0] = (np.arange(N) % 64) * 2.5; q[:, 1] = (np.arange(N) // 64) * 2.5
    def reset_state():
        sim.set_state(q, np.zeros((N, m.nv), np.float32))
        sim.t.ctrl.copy_(torch.as_tensor(np.tile(m.key_qpos[0][7:], (N, 1)).astype(np.float32)))
        torch.mps.synchronize()
    configs = {
        "feet+torso, history 3, air time": dict(bodies=["left_ankle_roll_link", "right_ankle_roll_link", "torso_link"], history_length=3, track_air_time=True),
        "+ filter [ground], force_mode total": dict(bodies=["left_ankle_roll_link", "right_ankle_roll_link", "torso_link"], history_length=3,
                                                   track_air_time=True, filter=["world"], force_mode="total"),
    }
    sensors = {k: ContactSensor(sim, **v, attach=False) for k, v in configs.items()}
    res = {"plain": []} | {k: [] for k in configs}
    for rep in range(3):
        for name in res:
            sim._substep_hooks = [] if name == "plain" else [sensors[name].launch]
            with wp.ScopedDevice(sim.device):
                sim._capture()
            reset_state()
            res[name].append(timed(sim))
    base = np.median(res["plain"])
    print(f"G1 MuJoCo Warp, {N} worlds, 4 substeps x 5 ms per step, graph replay, synchronized, median of 3 x {STEPS} steps")
    for name, v in res.items():
        med = np.median(v)
        extra = "" if name == "plain" else f"  (+{med - base:.3f} ms/step, +{(med - base) / 4 * 1e3:.0f} us/substep, {100 * (med - base) / base:+.1f} %)"
        print(f"  {name:40s} {med:8.3f} ms/step  runs {np.round(v, 3).tolist()}{extra}")
    # sensor kernels alone (4 substeps' worth), captured
    for name, s in sensors.items():
        with wp.ScopedCapture(device=sim.device) as cap:
            for _ in range(4):
                s.launch()
        sim.synchronize(); t0 = time.perf_counter()
        for _ in range(200):
            wp.capture_launch(cap.graph)
        sim.synchronize()
        print(f"  sensor kernels alone [{name}]: {(time.perf_counter() - t0) / 200 * 1e3:.3f} ms per 4 substeps")


if __name__ == "__main__":
    main()
