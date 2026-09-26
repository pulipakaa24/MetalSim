"""CPU oracle check of the elliptic-cone Newton Hessian variants: MuJoCo Warp on Warp's CPU device vs mj_step,
Go2 (dense Jacobian) and G1 (sparse, MuJoCo Warp's auto rule), 2 worlds, PD to random targets. No GPU needed.
usage: MJW_JTCJ_MODE=world|contact|capacity MJW_JTDAJ_ELLIPTIC_LANES=32|1 python elliptic_cpu_check.py [steps]"""
import os, sys, numpy as np, mujoco, warp as wp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
wp.config.quiet = True
import mujoco_warp as mjw
T = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MODELS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["go2", "g1"]
print(f"mujoco_warp {os.path.dirname(mjw.__file__)} MJW_JTCJ_MODE={os.environ.get('MJW_JTCJ_MODE')} LANES={os.environ.get('MJW_JTDAJ_ELLIPTIC_LANES')}")
with wp.ScopedDevice("cpu"):
  for name in MODELS:
    R = ROBOTS[name]
    spec = mujoco.MjSpec.from_file(R["scene"])
    LIM = force_limits(mujoco.MjModel.from_xml_path(R["scene"])) if R["kp"] is not None else []
    for a, (lo, hi) in zip(spec.actuators, LIM):
      a.set_to_position(kp=R["kp"], kv=R["kd"]); a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
      a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE; a.forcerange = [lo, hi]
    spec.option.timestep = DT; spec.option.cone = {"pyramidal": 0, "elliptic": 1}[os.environ.get("CONE", "elliptic")]
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    spec.option.iterations, spec.option.ls_iterations = 10, 20
    for gm in spec.geoms: gm.margin = 0.0
    m = spec.compile(); d = mujoco.MjData(m); (mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else None)
    mm = mjw.put_model(m); dd = mjw.put_data(m, d, nworld=2, naconmax=200, njmax=512)
    rng = np.random.default_rng(3); qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    errs = []
    for t in range(T):
      if t % RESAMPLE == 0: tgt = (m.key_qpos[0] if m.nkey else m.qpos0)[qadr] + AMP * rng.uniform(-1, 1, m.nu)
      d.ctrl[:] = tgt; dd.ctrl.assign(np.tile(tgt, (2, 1)).astype(np.float32))
      mjw.step(mm, dd); mujoco.mj_step(m, d)
      if (t + 1) in (1, 2, 5, 10, 20, 40, 80): errs.append((t + 1, float(np.abs(dd.qpos.numpy() - d.qpos[None]).max())))
    nacon = int(dd.nacon.numpy()[0]); print(f"{name}: sparse={mm.is_sparse} nv={m.nv} nacon={nacon} |dq| vs MuJoCo C: " + " ".join(f"t{k}:{e:.1e}" for k, e in errs), flush=True)
