"""Step-only throughput of the G1 rough task, the measurement of `python -m metalsim.learn.g1_velocity N rough`
(metalsim/learn/g1_velocity.py __main__: benchmark_step 100 frames, then 100 more timed with a synchronize, reported as
110 * N / wall time) with explicit task settings from RR_TASK_SETTINGS (JSON kwargs of G1VelocityTask), for settings
the module's CLI cannot select (terrain_collision, scan_surface, effort_limit).

    RR_TASK_SETTINGS='{"terrain_collision": "hfield", "scan_surface": "grid", "contact_cfg": "default", "effort_limit": "actuator"}' \
        .venv/bin/python runs/rough_rerun_2026-10-31/timing16/step_only.py 4096"""
import json, os, sys, time
import warp as wp
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask, benchmark_step, PHYSICS_DT

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
kw = json.loads(os.environ.get("RR_TASK_SETTINGS", "{}") or "{}")
task = G1VelocityTask(n, terrain="rough", physics_dt=PHYSICS_DT, **kw)
print(f"G1 (rough, {task.engine}, physics dt {task.physics_dt}): settings {kw}; terrain_collision {task.terrain_collision} "
      f"scan {task.scan_surface} contact_cfg {task.contact_cfg} effort {task.effort_limit} cone {int(task.model.opt.cone)}")
r = benchmark_step(task, num_frames=100)
t0 = time.perf_counter(); r2 = benchmark_step(task, num_frames=100); task.sim.synchronize(); dt = time.perf_counter() - t0
print(f"N={n}: {110 * n / dt:,.0f} env-steps/s (110 steps, synchronized); per-call mean {r2['mean_step_ms']:.2f} ms")
