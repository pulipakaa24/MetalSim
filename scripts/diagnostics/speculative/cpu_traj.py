"""CPU-device trajectory dump for bitwise comparisons between MuJoCo Warp checkouts (speculative contacts, 2026-09-27).

G1 (Isaac's minimal USD, flat) with a contact preset: 2 worlds dropped from 1.0 m / 0.9 m, PD targets = keyframe + a
seeded random sequence (0.5 rad, new every 8 substeps), S substeps at 2.5 ms. Writes qpos / qvel / efc_force sums /
nefc per substep. MuJoCo Warp comes from PYTHONPATH; MJW_* switches from the environment.

    python scripts/diagnostics/speculative/cpu_traj.py PRESET OUT.npz [S]
    python scripts/diagnostics/speculative/cpu_traj.py --cmp A.npz B.npz
"""
import sys, numpy as np
if sys.argv[1] == "--cmp":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    for k in ("qpos", "qvel", "nefc", "fsum"):
        d = np.abs(a[k].astype(np.float64) - b[k].astype(np.float64))
        first = int(np.argmax(d.reshape(len(d), -1).max(1) > 0)) if d.max() > 0 else -1
        print(f"{k}: bitwise equal {bool((a[k] == b[k]).all())}, max |diff| {d.max():.3e}, first differing substep {first}")
    sys.exit(0)
import mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.physics import contact_tuning as ct
p, out = sys.argv[1], sys.argv[2]; S = int(sys.argv[3]) if len(sys.argv) > 3 else 400
m, _ = build_g1_model("flat", physics_dt=0.0025); ct.apply(m, p)
m.opt.iterations = ct.PRESETS[p].solver_iterations or 10; m.opt.ls_iterations = 20
d = mujoco.MjData(m); d.qpos[:] = m.key_qpos[0]; d.qpos[2] = 1.0; d.ctrl[:] = m.key_qpos[0][7:]; mujoco.mj_forward(m, d)
rng = np.random.default_rng(0); nw = 2
with wp.ScopedDevice("cpu"):
    wm = mjw.put_model(m); wd = mjw.put_data(m, d, nworld=nw, nconmax=64, njmax=512)
    q = wd.qpos.numpy(); q[1, 2] = 0.9; wd.qpos.assign(q)
    Q, V, NE, FS = [], [], [], []
    for s in range(S):
        if s % 8 == 0:
            ctrl = np.tile(m.key_qpos[0][7:], (nw, 1)) + rng.uniform(-0.5, 0.5, (nw, m.nu))
            wd.ctrl.assign(ctrl.astype(np.float32))
        mjw.step(wm, wd)
        Q.append(wd.qpos.numpy().copy()); V.append(wd.qvel.numpy().copy()); ne = wd.nefc.numpy().copy(); NE.append(ne)
        f = wd.efc.force.numpy(); FS.append(np.array([f[w, :ne[w]].sum() for w in range(nw)]))
np.savez(out, qpos=np.array(Q), qvel=np.array(V), nefc=np.array(NE), fsum=np.array(FS))
print("wrote", out, "final z", Q[-1][:, 2], "max nefc", np.array(NE).max())
