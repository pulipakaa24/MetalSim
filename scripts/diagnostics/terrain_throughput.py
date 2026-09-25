"""G1 rough-task step throughput at N envs on the Isaac-exact terrain ("new") or on the pre-port
re-implemented terrain ("old", metalsim/learn/terrain.py at commit 9e33cfe, loaded from git).
Same protocol as ``python -m metalsim.learn.g1_velocity N rough``: graph-captured env.step with random
actions, 100 frames after 10 warmup, then a synchronized 110-step measurement (the headline).

    scripts/gpu_run.sh terrain_throughput timing 15 -- .venv/bin/python scripts/diagnostics/terrain_throughput.py new 4096
"""
import subprocess
import sys
import time

import numpy as np
import warp as wp

which = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
wp.config.quiet = True
import metalsim.learn.terrain as terrain
if which == "old":
    src = subprocess.check_output(["git", "show", "9e33cfe:metalsim/learn/terrain.py"], text=True)
    import importlib.util, os, tempfile
    path = os.path.join(tempfile.mkdtemp(), "old_terrain.py"); open(path, "w").write(src)   # Warp kernels need a file
    spec = importlib.util.spec_from_file_location("old_terrain", path); old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    terrain.isaac_rough_terrain = old.isaac_rough_terrain
from metalsim.learn.g1_velocity import G1VelocityTask, benchmark_step

t0 = time.perf_counter()
task = G1VelocityTask(n, terrain="rough")
print(f"[{which}] build {time.perf_counter() - t0:.1f} s; hfield {task.hfield['nrow']} x {task.hfield['ncol']}"
      f" = {task.hfield['nrow'] * task.hfield['ncol']:,} samples", flush=True)
benchmark_step(task, num_frames=100)
rates = []
for _ in range(reps):
    t0 = time.perf_counter(); benchmark_step(task, num_frames=100); task.sim.synchronize(); dt = time.perf_counter() - t0
    rates.append(110 * n / dt)
    print(f"[{which}] N={n}: {rates[-1]:,.0f} env-steps/s (110 steps, synchronized)", flush=True)
q = task.sim.d.qpos.numpy()
print(f"[{which}] median {np.median(rates):,.0f} env-steps/s over {reps}; finite state: {bool(np.isfinite(q).all())}")
