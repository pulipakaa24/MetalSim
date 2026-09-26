"""Compare engine runs (npz from mjw_loops.py / kamino_probe.py: t, ang, closure) with the exact reference
(ref_traj.py) and with each other. Angles are body direction angles, unwrapped."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechanisms as M  # noqa: E402


def load(p):
    z = np.load(p)
    return dict(t=z["t"], ang=np.unwrap(z["ang"], axis=0), closure=z["closure"] if "closure" in z.files else np.zeros(len(z["t"])), meta=json.loads(str(z["meta"])))


def on_grid(run, t):
    """run's angles at times t (run samples are a superset of t when dt divides; nearest otherwise)."""
    idx = np.clip(np.rint(t / (run["t"][1] - run["t"][0])).astype(int), 0, len(run["t"]) - 1)
    return run["ang"][idx]


def compare(run, ref, horizons=(0.5, 1.0, 2.0, 5.0), thresh=0.1):
    T = min(run["t"][-1], ref["t"][-1])
    dt = run["t"][1] - run["t"][0]
    t = np.arange(0, T + 1e-9, dt)
    ra = on_grid(ref, t); ea = on_grid(run, t)
    err = np.abs(ea - ra).max(1)
    out = {}
    for h in horizons:
        if h <= T + 1e-9:
            out[f"maxerr_{h}s"] = float(err[t <= h + 1e-9].max())
    bad = np.where(err > thresh)[0]
    out[f"t_err>{thresh}"] = float(t[bad[0]]) if len(bad) else None
    mech = run["meta"]["mech"]
    E = M.energy(mech, run["ang"], dt)
    out["energy_drift_%"] = float(100 * (E[min(len(E) - 1, int(T / dt) - 2)] - E[0]) / abs(E[0])) if run["meta"].get("torque", "none") == "none" else None
    out["closure_max_mm"] = float(np.nanmax(run["closure"]) * 1e3)
    out["closure_mean_mm"] = float(np.nanmean(run["closure"]) * 1e3)
    return out


def pair(a, b):
    """max |angle difference| between two runs with the same dt, at horizons."""
    n = min(len(a["t"]), len(b["t"]))
    err = np.abs(a["ang"][:n] - b["ang"][:n]).max(1)
    t = a["t"][:n]
    return {f"maxdiff_{h}s": float(err[t <= h + 1e-9].max()) for h in (0.5, 1.0, 2.0, 5.0) if h <= t[-1] + 1e-9}


if __name__ == "__main__":
    ref = load(sys.argv[1])
    for p in sys.argv[2:]:
        r = load(p)
        print(os.path.basename(p), json.dumps(compare(r, ref)))
