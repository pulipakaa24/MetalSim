"""Exact (hard-closure) planar reference trajectories for mechanisms.py, sampled every 1.25 ms (a common grid for
the 2.5 ms and 5 ms engine runs): fourbar passive 5 s, fourbar under world-0 sigma3 torques 5 s, leg passive 2 s
(knee limits are not modelled in the reference; the passive leg stays inside them for the first 2 s)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechanisms as M  # noqa: E402
from mechanisms import torque_table  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/closed_loops/ref"
os.makedirs(OUT, exist_ok=True)
for mech, torque, T in [("fourbar", "none", 5.0), ("fourbar", "sigma3", 5.0), ("leg", "none", 2.0)]:
    ref = M.PlanarRef(mech, damped=torque == "sigma3")
    te = np.arange(int(round(T / 0.00125)) + 1) * 0.00125
    if torque == "sigma3":
        tab = torque_table(mech, 1, T)[:, 0]
        act = M.GEOMS[mech]()["actuated"]
        tau_fn = lambda t, tab=tab, act=act: dict(zip(act, tab[min(int(t / 0.02 + 1e-9), len(tab) - 1)]))
    else:
        tau_fn = lambda t: {}
    # integrate piecewise between torque switches so the discontinuities are hit exactly
    ys = []; y0 = None
    edges = np.arange(0, T + 1e-12, 0.02) if torque == "sigma3" else np.array([0.0, T])
    if edges[-1] < T - 1e-12:
        edges = np.append(edges, T)
    from scipy.integrate import solve_ivp
    y = np.concatenate([ref.q0, np.zeros_like(ref.q0)])
    out = np.zeros((len(te), y.size)); out[0] = y
    for a, b in zip(edges[:-1], edges[1:]):
        sel = np.where((te > a + 1e-12) & (te <= b + 1e-12))[0]
        tau_c = tau_fn(a + 1e-6)
        s = solve_ivp(ref.rhs, (a, b), y, args=(lambda t, tc=tau_c: tc,), rtol=1e-10, atol=1e-12, method="DOP853",
                      t_eval=te[sel] if len(sel) else None, dense_output=False)
        assert s.success, s.message
        if len(sel):
            out[sel] = s.y.T
        y = s.y[:, -1] if not len(sel) or abs(s.t[-1] - b) < 1e-12 else solve_ivp(ref.rhs, (a, b), y, args=(lambda t, tc=tau_c: tc,), rtol=1e-10, atol=1e-12, method="DOP853").y[:, -1]
    nb = ref.nb
    ang = out[:, 2:3 * nb:3]
    E = M.energy(mech, ang, 0.00125)
    viol = max(np.abs(ref.constraints(out[k, :3 * nb], out[k, 3 * nb:])[0]).max() for k in range(0, len(te), 50))
    q = [ref.joint_angles(out[k, :3 * nb]) for k in range(len(te))]
    knee = np.array([x.get("knee", 0.0) for x in q])
    meta = dict(mech=mech, torque=torque, T=T, energy_start=float(E[0]), energy_end=float(E[-1]), pin_violation_max=float(viol),
                knee_rel_range=[float(knee.min()), float(knee.max())])
    print(json.dumps(meta), flush=True)
    np.savez(f"{OUT}/{mech}_{torque}.npz", t=te, ang=ang, meta=json.dumps(meta))
