"""CPU stand-in for g1_state_dump.py: G1-task model states from MuJoCo C (64 worlds of PD hold at the standing keyframe
plus random joint targets resampled every 20 control steps, 8 substeps of 2.5 ms), same npz/mjb layout.
    python g1_state_dump_c.py OUT_PREFIX [--worlds 64] [--preset tau10_impact_hardlimits_ellip10]"""
import argparse, os, sys, numpy as np, mujoco
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import contact_tuning
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--worlds", type=int, default=64); ap.add_argument("--preset", default="tau10_impact_hardlimits_ellip10")
ap.add_argument("--at", default="2,10,30,60,100,150,200,300,399"); a = ap.parse_args()
m, _ = build_g1_model("flat", physics_dt=0.0025); contact_tuning.apply(m, a.preset); mujoco.mj_saveModel(m, a.out + ".mjb")
qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]; home = m.key_qpos[0]
AT = sorted(int(x) for x in a.at.split(",")); N = a.worlds; rng = np.random.default_rng(0)
out = {f"t{t}": {k: [] for k in ("qpos", "qvel", "ctrl", "qacc_warmstart", "act")} for t in AT}
for w in range(N):
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    for t in range(max(AT) + 1):
        if t % 20 == 0: d.ctrl[:] = home[qadr] + float(os.environ.get("AMP", "0.05")) * rng.uniform(-1, 1, m.nu)
        if t in AT:
            o = out[f"t{t}"]; o["qpos"].append(d.qpos.copy()); o["qvel"].append(d.qvel.copy()); o["ctrl"].append(d.ctrl.copy()); o["qacc_warmstart"].append(d.qacc_warmstart.copy()); o["act"].append(d.act.copy())
        for _ in range(8): mujoco.mj_step(m, d)
arrs = {f"{k}_{f}": np.array(v, dtype=np.float32) for k, dd in out.items() for f, v in dd.items()}
np.savez_compressed(a.out + ".npz", steps=np.array(AT), opt=np.array([m.opt.iterations, m.opt.ls_iterations, m.opt.cone, m.opt.impratio, m.opt.timestep]), njmax=256, naconmax=32 * N, nconmax=32, **arrs)
print("saved", a.out + ".npz", "worlds", N, "pelvis z mean at snapshots:", [round(float(arrs[f"t{t}_qpos"][:, 2].mean()), 3) for t in AT], "min", [round(float(arrs[f"t{t}_qpos"][:, 2].min()), 3) for t in AT])
