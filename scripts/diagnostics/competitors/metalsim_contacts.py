"""Contacts per world after 200 steps of the metalsim_step.py setup (generated from it: same model, controller, settings)."""
import sys, time, json, numpy as np, mujoco, torch, warp as wp
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *
STEPS = 200
from metalsim.physics.batch import BatchSim, BatchSimOptions
name, N, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]          # mode: matched | default
R = ROBOTS[name]
spec = mujoco.MjSpec.from_file(R["scene"])
LIM = force_limits(mujoco.MjModel.from_xml_path(R["scene"]))
for a, (lo, hi) in zip(spec.actuators, LIM):                         # motor -> PD position servo, force-limited
    a.set_to_position(kp=R["kp"], kv=R["kd"])
    a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE; a.forcerange = [lo, hi]
spec.option.timestep = DT
spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)   # MuJoCo Warp: MULTICCD + margins unsupported; Genesis has no MULTICCD either
for gm in spec.geoms:
    gm.margin = 0.0                                                  # MuJoCo Warp: non-zero margin unsupported with NATIVECCD (Go2 has 1 mm)
m = spec.compile()
home = m.key_qpos[0].copy() if m.nkey else m.qpos0.copy()
qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
opts = BatchSimOptions(solver_iterations=10, ls_iterations=20) if mode == "matched" else BatchSimOptions()
t0 = time.time(); sim = BatchSim(m, N, options=opts); sim.set_state(home); sim.forward(); sim.synchronize(); t_build = time.time() - t0
q_home = torch.tensor(home[qadr], device="mps", dtype=torch.float32)
g = torch.Generator(device="mps").manual_seed(0)
def resample():
    sim.t.ctrl.copy_(q_home + AMP * (2 * torch.rand((N, m.nu), device="mps", generator=g) - 1)); torch.mps.synchronize()
for i in range(WARMUP):
    if i % RESAMPLE == 0: resample()
    sim.step()
sim.synchronize()
t0 = time.perf_counter()
for i in range(STEPS):
    if i % RESAMPLE == 0: resample()
    sim.step()
sim.synchronize(); el = time.perf_counter() - t0
qz = sim.t.qpos[:, 2].float(); ok = torch.isfinite(sim.t.qpos).all().item()
nac = sim.t.nacon; print("CONTACTS", name, "metalsim per_world", round(float(nac.flatten()[0].item()) / N, 2))
