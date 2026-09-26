import sys, time, json, numpy as np, mujoco, torch, warp as wp
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *
from metalsim.physics.batch import BatchSim, BatchSimOptions
name, N, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]          # mode: matched | default
R = ROBOTS[name]
spec = mujoco.MjSpec.from_file(R["scene"])
LIM = force_limits(mujoco.MjModel.from_xml_path(R["scene"])) if R["kp"] is not None else []
for a, (lo, hi) in zip(spec.actuators, LIM):                         # motor -> PD position servo, force-limited
    a.set_to_position(kp=R["kp"], kv=R["kd"])
    a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE; a.forcerange = [lo, hi]
spec.option.timestep = DT
import os
if os.environ.get('CONE'): spec.option.cone = {'pyramidal': mujoco.mjtCone.mjCONE_PYRAMIDAL, 'elliptic': mujoco.mjtCone.mjCONE_ELLIPTIC}[os.environ['CONE']]
spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)   # MuJoCo Warp: MULTICCD + margins unsupported; Genesis has no MULTICCD either
for gm in spec.geoms:
    gm.margin = 0.0                                                  # MuJoCo Warp: non-zero margin unsupported with NATIVECCD (Go2 has 1 mm)
m = spec.compile()
home = m.key_qpos[0].copy() if m.nkey else m.qpos0.copy()
qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
JAC = os.environ.get("JAC")                                           # dense | sparse: MuJoCo Warp constraint-Jacobian layout (default: its auto rule, sparse for nv > 32)
opts = BatchSimOptions(solver_iterations=10, ls_iterations=20, jacobian=JAC, **({"njmax": int(os.environ["NJMAX"])} if os.environ.get("NJMAX") else {})) if mode == "matched" else BatchSimOptions(jacobian=JAC, **({"njmax": int(os.environ["NJMAX"])} if os.environ.get("NJMAX") else {}))
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
print("RESULT", json.dumps(dict(engine="metalsim", cone=int(m.opt.cone), robot=name, N=N, mode=mode, nv=m.nv, nu=m.nu, iters=int(sim.m.opt.iterations), ls=int(sim.m.opt.ls_iterations),
      build_s=round(t_build, 1), steps_per_s=round(N * STEPS / el), finite=ok, base_z_mean=round(qz.mean().item(), 3), overflow=sim.overflow_flags())))
