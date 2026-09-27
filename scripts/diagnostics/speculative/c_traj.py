"""MuJoCo C (float64) counterpart of cpu_traj.py: the same G1 protocol (2 worlds, drops from 1.0 / 0.9 m, the same seeded
random PD targets every 8 substeps, float32-cast), stepped by mj_step (--stock) or by the speculative-rule oracle
(c_oracle.SpecC, --mode cone|none|live), ls_iterations 50 (the elliptic cost review's safe setting), Newton cap of the preset.
    python scripts/diagnostics/speculative/c_traj.py PRESET OUT.npz [S] [--stock | --mode live]
    python scripts/diagnostics/speculative/c_traj.py --diff A.npz B.npz      (max |dq| at 0.05 / 0.1 / 0.25 / 0.5 / 1 s, per world)"""
import sys, os, copy, numpy as np
if sys.argv[1] == "--diff":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    for t in (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0):
        k = int(round(t / 0.0025)) - 1
        if k < len(a["qpos"]):
            print(f"  t {t:4.2f} s: max|dq| per world " + " ".join(f"{np.abs(a['qpos'][k, w].astype(float) - b['qpos'][k, w]).max():.2e}" for w in range(a["qpos"].shape[1])))
    sys.exit(0)
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c_oracle import SpecC
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import contact_tuning as ct
p, out = sys.argv[1], sys.argv[2]; S = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 400
mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "cone"; stock = "--stock" in sys.argv
m, _ = build_g1_model("flat", physics_dt=0.0025); ct.apply(m, p)
m.opt.iterations = ct.PRESETS[p].solver_iterations or 10; m.opt.ls_iterations = 50; m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND      # same solver path for --stock and the oracle (see c_oracle)
rng = np.random.default_rng(0); nw = 2
ds = []
for w, z in enumerate((1.0, 0.9)):
    d = mujoco.MjData(m); d.qpos[:] = m.key_qpos[0]; d.qpos[2] = z; d.ctrl[:] = m.key_qpos[0][7:]; mujoco.mj_forward(m, d); ds.append(d)
orc = None if stock else SpecC(m, mode, True, os.environ.get("MJW_SPEC_FRICIMP", "pos"))
Q, V = [], []
for s in range(S):
    if s % 8 == 0:
        ctrl = (np.tile(m.key_qpos[0][7:], (nw, 1)) + rng.uniform(-0.5, 0.5, (nw, m.nu))).astype(np.float32)
        for w in range(nw): ds[w].ctrl[:] = ctrl[w]
    for w in range(nw):
        if stock: mujoco.mj_step(m, ds[w])
        else: orc.step(ds[w])
    Q.append(np.array([d.qpos.copy() for d in ds])); V.append(np.array([d.qvel.copy() for d in ds]))
np.savez(out, qpos=np.array(Q), qvel=np.array(V))
print("wrote", out, "final z", Q[-1][:, 2])
