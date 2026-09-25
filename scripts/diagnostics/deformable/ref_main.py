"""Reference rollout with a given MuJoCo build (run with scratch/deformable/venv-mjmain for MuJoCo main).
usage: ref_main.py XML Q0.npy NSTEPS OUT.npz"""
import sys, numpy as np, mujoco
xml, q0f, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
d.qpos[:] = np.load(q0f); mujoco.mj_forward(m, d)
Q, V, X, E, N, CS = [], [], [], [], [], []
for k in range(n):
    Q.append(d.qpos.copy()); V.append(d.qvel.copy())       # state before step k
    mujoco.mj_step(m, d)
    X.append(d.flexvert_xpos.copy()); E.append(d.energy.copy()); N.append(d.ncon)
    c = d.contact[:d.ncon]
    CS.append(np.stack([c.geom[:, 0], c.flex[:, 1], c.elem[:, 1], c.vert[:, 1]], 1) if d.ncon else np.zeros((0, 4), int))
np.savez(out, qpos=np.array(Q), qvel=np.array(V), vert=np.array(X), energy=np.array(E), ncon=np.array(N),
         cs=np.array(CS, dtype=object), version=mujoco.__version__)
print("ref", mujoco.__version__, "steps", n, "max ncon", max(N))
