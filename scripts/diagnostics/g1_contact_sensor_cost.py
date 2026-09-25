"""Per-control-step cost of the ContactSensor in the G1 flat task (MuJoCo Warp, Isaac G1FlatEnvCfg set): the
same task timed with the sensor's substep hook attached (A) and detached (B, touch-site path), interleaved
A/B repeats, synchronized env-steps/s over benchmark_step's protocol (random actions, full env.step graph).

    bash scripts/gpu_run.sh g1_contact_sensor_cost timing 10 -- .venv/bin/python scripts/diagnostics/g1_contact_sensor_cost.py --envs 4096
"""
import argparse, time
import numpy as np, warp as wp

from metalsim.learn.g1_velocity import G1VelocityTask, benchmark_step


def rate(task, frames):
    benchmark_step(task, num_frames=10, warmup=0)                    # warm-up / capture
    t0 = time.perf_counter(); benchmark_step(task, num_frames=frames, warmup=0); task.sim.synchronize()
    return frames * task.n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--envs", type=int, default=4096); ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--physics_dt", type=float, default=0.0025)
    a = ap.parse_args(); wp.config.quiet = True
    task = G1VelocityTask(a.envs, terrain="flat", seed=0, physics_dt=a.physics_dt)
    sensor = task.contact
    hooks = (list(task.sim._substep_hooks), list(task.sim._reset_hooks))
    def attach(on):
        task.sim._substep_hooks[:] = hooks[0] if on else []
        task.sim._reset_hooks[:] = hooks[1] if on else []
        with wp.ScopedDevice(task.device):
            task.sim._capture()
        task.contact = sensor if on else None; task.use_sensor = 1 if on else 0
    res = {"sensor": [], "touch": []}
    for r in range(a.repeats):
        for on in (True, False):
            attach(on)
            res["sensor" if on else "touch"].append(rate(task, a.frames))
    for k, v in res.items():
        print(f"{k:6s}: {np.median(v):,.0f} env-steps/s (median of {len(v)}: {', '.join(f'{x:,.0f}' for x in v)}); "
              f"{1e3 * a.envs / np.median(v):.2f} ms per control step")
    d = 1e3 * a.envs / np.median(res["sensor"]) - 1e3 * a.envs / np.median(res["touch"])
    print(f"N={a.envs}, physics dt {task.physics_dt} (decimation {task.decimation}), history {sensor.T} substeps: "
          f"sensor costs {d:+.2f} ms per control step ({100 * (np.median(res['touch']) / np.median(res['sensor']) - 1):+.1f} %)")


if __name__ == "__main__":
    main()
