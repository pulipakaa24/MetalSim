"""Gait of Isaac's checkpoints in PhysX (Isaac's own play recordings, runs/parity/isaac/parity_out/play/isaac_it*/traj.npz:
4 envs, 400 control steps, command 0.5 m/s, same protocol as the transfer test) vs in MuJoCo Warp (overtravel_probe.py
outputs with jpos/root). Steady state = control steps 100-400.
    python scripts/diagnostics/speculative/gait_vs_isaac.py DIR TAG[,TAG...] [--n 64]"""
import sys, os, numpy as np
D = sys.argv[1]; TAGS = sys.argv[2].split(","); N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 64
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LEG = {"hip_pitch": (0, 1), "hip_roll": (3, 4), "knee": (11, 12), "ankle_pitch": (15, 16), "ankle_roll": (19, 20)}
CT = 0.02; S0 = 100


def gait(jp, root):
    """jp (S, n, 37) Isaac order, root (S, n, 3)"""
    x = root[-1, :, 0]; vx = (root[-1, :, 0] - root[S0, :, 0]) / ((len(root) - 1 - S0) * CT)
    sig = jp[S0:, :, 0] - jp[S0:, :, 0].mean(0)                  # left hip pitch
    # cadence from the upward zero crossings of the detrended left hip pitch (linear interpolation of the crossing times)
    f = np.zeros(sig.shape[1])
    for e in range(sig.shape[1]):
        y = sig[:, e]; i = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
        tc = (i + y[i] / (y[i] - y[i + 1])) * CT
        f[e] = (len(tc) - 1) / (tc[-1] - tc[0])
    out = dict(x=x.mean(), x_se=x.std(ddof=1) / np.sqrt(len(x)), vx=vx.mean(), f=f.mean(), stride=(vx / f).mean(), z=root[S0:, :, 2].mean())
    for k, (l, r) in LEG.items():
        seg = jp[S0:, :, [l, r]]
        out[k + "_mean"] = seg.mean(); out[k + "_amp"] = seg.std(0).mean()    # time-std per env and side, averaged
        out[k + "_min"] = np.percentile(seg, 5); out[k + "_max"] = np.percentile(seg, 95)
    return out


keys = ["x", "vx", "f", "stride", "z"] + [f"{k}_{s}" for k in LEG for s in ("mean", "amp")] + ["ankle_pitch_min", "ankle_pitch_max"]
for it in (500, 1000, 1499):
    iz = np.load(os.path.join(ROOT, f"runs/parity/isaac/parity_out/play/isaac_it{it}/traj.npz"))
    ref = gait(iz["joint_pos"], iz["root_pos"])
    rows = {"Isaac PhysX (4 envs)": ref}
    for t in TAGS:
        f = os.path.join(D, f"{t}_n{N}.npz")
        if os.path.exists(f):
            z = np.load(f); rows[t] = gait(z[f"it{it}_jpos"], z[f"it{it}_root"])
    print(f"\n#### checkpoint {it}\n")
    print("| setting | " + " | ".join(keys) + " | gait distance to Isaac |"); print("|---|" + "---|" * (len(keys) + 1))
    for t, r in rows.items():
        # distance: RMS over the leg-joint means and amplitudes (rad) of the difference to Isaac
        dist = np.sqrt(np.mean([(r[k] - ref[k]) ** 2 for k in keys if k.endswith(("_mean", "_amp"))]))
        print(f"| {t} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + f" | {dist:.4f} |")
