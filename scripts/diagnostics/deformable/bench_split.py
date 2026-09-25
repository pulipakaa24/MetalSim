"""Where does a MuJoCo Warp flex step spend its time on Metal? 4096 worlds of the box scene, graph replay.
Variants: full step; contacts off (DSBL_CONTACT); constraints off; CG iterations 5; kinematics+collision only."""
import sys, time, json, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from metalsim.physics import deformable as dfm

def timed(fn, n=10, reps=3):
    fn(); wp.synchronize_device("metal:0")
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(n): fn()
        wp.synchronize_device("metal:0")
        best = min(best, (time.perf_counter() - t) / n)
    return best

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
base = dfm.scene_model("box")
out = {}
for name, mod in [("full", lambda m: None),
                  ("no_contact", lambda m: setattr(m.opt, "disableflags", m.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_CONTACT)),
                  ("no_constraint", lambda m: setattr(m.opt, "disableflags", m.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)),
                  ("cg5", lambda m: (setattr(m.opt, "iterations", 5), setattr(m.opt, "ls_iterations", 5)))]:
    m = mujoco.MjModel.from_xml_string(dfm.box_scene_xml()); mod(m)
    sim = dfm.DeformableSim(m, N, device="metal:0")
    sim.randomize(seed=1)
    for _ in range(80): sim.step()       # into contact
    t = timed(sim.step)
    out[name] = t
    print(json.dumps({"variant": name, "num_envs": N, "ms_per_step": t * 1e3, "env_steps_per_s": N / t}), flush=True)
    del sim; import gc; gc.collect(); wp.synchronize_device('metal:0')
# pieces with the full model: kinematics, collision, make_constraint, solve (eager, graph-captured separately)
m = mujoco.MjModel.from_xml_string(dfm.box_scene_xml())
sim = dfm.DeformableSim(m, N, device="metal:0", capture=False)
sim.randomize(seed=1)
for _ in range(80): sim.step(eager=True)
M, D = sim.m, sim.d
pieces = {"kinematics+flex": lambda: (mjw.kinematics(M, D), mjw.flex(M, D)),
          "fwd_position(kin+collision+constraints)": lambda: mjw.fwd_position(M, D),
          "collision": lambda: mjw.collision(M, D),
          "fwd_velocity": lambda: mjw.fwd_velocity(M, D),
          "fwd_acceleration+solve": lambda: (mjw.fwd_acceleration(M, D), mjw.solve(M, D))}
for name, fn in pieces.items():
    with wp.ScopedDevice("metal:0"):
        t = timed(lambda: fn(), n=5)   # eager launches (graphs per piece would each keep their own workspace)
    print(json.dumps({"piece": name, "num_envs": N, "ms": t * 1e3}), flush=True)
