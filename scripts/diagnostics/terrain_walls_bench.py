"""G1 rough env-step throughput per terrain collision surface (Isaac's benchmark protocol: 4096 envs, random uniform
actions, apply action + physics + rewards/resets + observations in one captured graph; synchronized rate over 110 steps,
the headline of PARITY 1.4). Each mode runs in its own process. Also reports the contact load (nacon at the end vs
naconmax) and the model setup time.

    scripts/gpu_run.sh terrain_walls_bench timing 12 -- python scripts/diagnostics/terrain_walls_bench.py hfield boxes
"""
import json, os, subprocess, sys, time

MODES = {"hfield": ("hfield", 0.025), "hfield@nconmax128": ("hfield", 0.025), "boxes": ("boxes", 0.025), "meshes": ("meshes", 0.025),
         "hfield_fine:0.05": ("hfield_fine", 0.05), "hfield_fine:0.025": ("hfield_fine", 0.025),
         "boxes_fine:0.025": ("boxes_fine", 0.025),
         "boxes_local": ("boxes_local", 0.025)}


def one(mode, n, physics_dt, out):
    import numpy as np, warp as wp
    wp.config.quiet = True
    import metalsim.learn.terrain as T
    from metalsim.learn.g1_velocity import G1VelocityTask, benchmark_step
    if mode.endswith("@njmax256"):
        # fine heightfields: the task asks for njmax 512 because put_data checks MuJoCo C's initial contact set at
        # qpos0 (C's hfield collider returns up to 50 contacts per foot there: 400 rows), and 512 runs out of Metal
        # memory at 4096 envs. For the measurement: njmax 256 (as every other mode) with the robot lifted 5 m for that
        # initial C check only (reset_all places every env before stepping; Warp makes <= 8 hfield contacts per foot)
        mode = mode[:-len("@njmax256")]
        import mujoco, metalsim.learn.g1_velocity as G
        _BSO = G.BatchSimOptions
        G.BatchSimOptions = lambda **kw: _BSO(**{**kw, "njmax": 256})
        _fwd = mujoco.mj_forward
        def _lifted(m, d, _fwd=_fwd):
            if m.nq and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE: d.qpos[2] += 5.0
            _fwd(m, d)
        mujoco.mj_forward = _lifted
    mode, _, scan = mode.partition("+")         # MODE+exact: the exact height scan on box-built cells
    coll, fr = MODES[mode]
    if mode.endswith("@nconmax128"):          # the baseline with the contact capacity the other modes need
        import metalsim.learn.g1_velocity as G
        _BSO = G.BatchSimOptions
        G.BatchSimOptions = lambda **kw: _BSO(**{**kw, "nconmax": 128})
    orig = T.isaac_rough_terrain
    T.isaac_rough_terrain = lambda **kw: orig(fine_res=fr, **kw)     # fine resolution for hfield_fine
    t0 = time.perf_counter()
    task = G1VelocityTask(n, terrain="rough", seed=0, physics_dt=physics_dt, reward_cfg="rough_isaac", terrain_collision=coll,
                          scan_surface=scan or "grid")
    setup = time.perf_counter() - t0
    if "_lifted" in locals(): mujoco.mj_forward = _fwd
    import gc
    arrs = [o for o in gc.get_objects() if type(o) is wp.array and str(getattr(o, "device", "")) == "metal:0" and getattr(o, "ptr", None)]
    seen = {}; [seen.setdefault(a.ptr, a.capacity or 0) for a in arrs]
    big = sorted(((a.capacity or 0, a.shape, str(a.dtype.__name__ if hasattr(a.dtype, "__name__") else a.dtype)) for a in arrs), key=lambda t: -t[0])[:6]
    print(f"live Warp arrays on metal:0: {sum(seen.values()) / 1e9:.2f} GB; largest {big}", flush=True)
    benchmark_step(task, num_frames=100)
    rates = []
    for _ in range(3):
        t0 = time.perf_counter(); benchmark_step(task, num_frames=100); task.sim.synchronize(); dt = time.perf_counter() - t0
        rates.append(110 * n / dt)
    d = task.sim.d
    nacon = int(d.nacon.numpy()[0]); naconmax = int(d.naconmax) if hasattr(d, "naconmax") else None
    r = {"mode": sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--one" else mode, "n": n, "physics_dt": physics_dt, "ngeom": int(task.model.ngeom), "hfield_samples": int(task.model.nhfielddata),
         "setup_s": round(setup, 1), "env_steps_per_s": [round(x) for x in rates], "best": round(max(rates)),
         "nacon_end": nacon, "naconmax": naconmax,
         "box_window_overflow": task.box_window.overflow.numpy().tolist() if getattr(task, "box_window", None) else None}
    print(json.dumps(r), flush=True)
    with open(out, "a") as f: f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]); sys.exit(0)
    n = int(os.environ.get("N", 4096)); pdt = float(os.environ.get("PHYSICS_DT", 0.0025))
    out = os.environ.get("OUT", "runs/terrain_walls/bench.jsonl")
    subprocess.run([sys.executable, "scripts/gpu_lock.py", "status"])
    for m in sys.argv[1:]:
        subprocess.run([sys.executable, __file__, "--one", m, str(n), str(pdt), out])
