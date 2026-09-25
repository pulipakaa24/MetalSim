"""Check for metalsim/parity/side_by_side.py's deterministic start: after copying Isaac's default qpos into the sim and
running forward kinematics, the body poses the renderer reads (d.xpos / d.xquat) must equal MuJoCo C's kinematics of
that state. Runs the start sequence with and without the MPS sync (the fix) on a 1-env G1, a few seconds of GPU.

    python scripts/diagnostics/side_by_side_step0_check.py
"""
import numpy as np, torch, mujoco, warp as wp
from metalsim.learn.g1_velocity import G1VelocityTask, build_g1_model
from metalsim.physics.batch import BatchSim, BatchSimOptions
from metalsim.parity.record_g1 import hero_spec
import json, glob

wp.config.quiet = True
meta = json.load(open(sorted(glob.glob("runs/parity/isaac/**/rt/meta.json", recursive=True))[0]))
for sync in (False, True):
    task = G1VelocityTask(1, terrain="flat", seed=0, physics_dt=0.0025)
    m, info = build_g1_model("flat", visuals=True, physics_dt=0.0025); hero = hero_spec(info["spec"], meta).compile()
    task.model = hero
    task.sim = BatchSim(hero, 1, options=BatchSimOptions(substeps=task.decimation, njmax=256, nconmax=32, solver_iterations=10, ls_iterations=20)); task.sim.synchronize()
    task.origins.assign(np.zeros((1, 3), np.float32)); task.reset_all()
    q0 = hero.key_qpos[0].astype(np.float32); task.sim.t.qpos.copy_(torch.as_tensor(q0)[None]); task.sim.t.qvel.zero_()
    if sync:
        torch.mps.synchronize()
    v = task.sim.forward(); task.sim.after(v); task.sim.synchronize()
    d = mujoco.MjData(hero); d.qpos[:] = task.sim.d.qpos.numpy()[0]; mujoco.mj_kinematics(hero, d)
    err = np.abs(task.sim.d.xpos.numpy()[0] - d.xpos).max()
    qerr = np.abs(np.abs((task.sim.d.xquat.numpy()[0] * d.xquat).sum(1)) - 1).max()
    state_is_q0 = np.abs(task.sim.d.qpos.numpy()[0] - q0).max()
    print(f"sync {sync}: qpos == Isaac default to {state_is_q0:.2e}; body pos vs MuJoCo C kinematics of the state max |diff| {err:.2e} m, "
          f"quat 1-|dot| max {qerr:.2e}")
