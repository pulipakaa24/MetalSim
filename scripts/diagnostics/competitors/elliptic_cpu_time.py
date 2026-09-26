"""CPU-device step time of the Go2 with elliptic cones (dense Jacobian) per MJW_JTCJ_MODE, for the upstream PR draft.
usage: MJW_JTCJ_MODE=world|capacity python elliptic_cpu_time.py [nworld] [steps]"""
import os, sys, time, numpy as np, mujoco, warp as wp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
wp.config.quiet = True
import mujoco_warp as mjw
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 32; T = int(sys.argv[2]) if len(sys.argv) > 2 else 20
R = ROBOTS["go2"]
with wp.ScopedDevice("cpu"):
  spec = mujoco.MjSpec.from_file(R["scene"]); LIM = force_limits(mujoco.MjModel.from_xml_path(R["scene"]))
  for a, (lo, hi) in zip(spec.actuators, LIM):
    a.set_to_position(kp=R["kp"], kv=R["kd"]); a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE; a.forcerange = [lo, hi]
  spec.option.timestep = DT; spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD); spec.option.iterations, spec.option.ls_iterations = 10, 20
  for gm in spec.geoms: gm.margin = 0.0
  m = spec.compile(); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
  mm = mjw.put_model(m); dd = mjw.put_data(m, d, nworld=NW, naconmax=48 * NW, njmax=64)
  qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
  dd.ctrl.assign(np.tile(m.key_qpos[0][qadr], (NW, 1)).astype(np.float32))
  for _ in range(3): mjw.step(mm, dd)
  wp.synchronize(); t0 = time.perf_counter()
  for _ in range(T): mjw.step(mm, dd)
  wp.synchronize(); el = time.perf_counter() - t0
  print(f"cpu go2 elliptic MJW_JTCJ_MODE={os.environ.get('MJW_JTCJ_MODE')} nworld={NW} naconmax={dd.naconmax} ndof_tri={mm.dof_tri_row.size}: {1e3 * el / T:.1f} ms/step, {NW * T / el:.0f} steps/s, nacon={int(dd.nacon.numpy()[0])}")
