"""Summarize the 2026-09-26 penetration sweep (sweep.sh rec) into the full protocol table against Isaac 5.1 PhysX.

usage: python summarize.py [--profile] DIR...     (DIR = runs/penetration/rec/<preset>[_dt...][_r2])

Penetration is MuJoCo's contact dist of the deepest ground contact per body (left foot, right foot, torso) in each substep,
from the pose at the substep's start; PhysX's reference (runs/parity3/isaac/penetration_il2, RigidContactView separations per
5 ms step, pose after the previous step) is the same definition (geometric shape-to-plane distance; restOffset 0), sampled at
5 ms: the "5 ms tick" columns keep only the poses at multiples of 5 ms. Positive depth = penetration. Momentum-derived impulses
and 20 ms mean forces: metalsim.parity.momentum (every env). Chatter: on/off transitions of the foot flag |F| > 1 N in A_hold +
C_drop, both feet, per substep and on 5 ms averages (scripts/diagnostics/contact_research/flicker.py's definition)."""
import json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts/diagnostics/contact_research"))
from flicker import phases
ISAAC = os.path.join(ROOT, "runs/parity/isaac/parity_out2/rt")
PHYSX = {"land": 1996.0, "torso": 2200.0, "hold": 2190.0}      # Isaac's momentum-derived 20 ms mean forces (PARITY §1.7)
KEYS = ("joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel_b", "root_ang_vel_b")


def depth(Z, dt):
    d = Z["sub_dist"]; T, S, n, _ = d.shape
    return -d.reshape(T * S, n, 3)                          # (substeps, envs, [L, R, torso]) mm-free, metres; NaN = no contact


def summarize(dr):
    meta = json.load(open(os.path.join(dr, "meta.json")))
    dt = meta["physics_dt"]; stride = int(round(0.005 / dt)); cap = meta.get("solver_iterations", 10)
    Z = {t: np.load(os.path.join(dr, f"{t}.npz")) for t in ("A_hold", "B_random", "C_drop")}
    n = Z["C_drop"]["sub_dist"].shape[2]
    out = {"dir": os.path.relpath(dr, ROOT), "preset": meta["contact_tuning"], "dt": dt, "cap": cap, "envs": n}
    P = {t: depth(Z[t], dt) for t in Z}
    with np.errstate(all="ignore"):
        feet = {t: np.nanmax(P[t][:, :, :2], axis=2) for t in P}          # (S, n) deepest foot
        torso = {t: P[t][:, :, 2] for t in P}
        pk = lambda x, ax=0: np.nan_to_num(np.nanmax(x, axis=ax), nan=0.0)
        out["drop_feet_peak_mm"] = (pk(feet["C_drop"]) * 1e3).tolist()          # per env
        out["drop_feet_peak_5ms_mm"] = (pk(feet["C_drop"][::stride]) * 1e3).tolist()
        out["drop_torso_peak_mm"] = (pk(torso["C_drop"]) * 1e3).tolist()
        for t, nm in (("A_hold", "hold"), ("B_random", "random")):
            out[f"{nm}_feet_peak_mm"] = (pk(feet[t]) * 1e3).tolist(); out[f"{nm}_torso_peak_mm"] = (pk(torso[t]) * 1e3).tolist()
        allb = np.nanmax(P["C_drop"], axis=2)                                   # deepest contact of any body per substep
        last = allb[-int(round(0.5 / dt)):]
        out["drop_settle_max_mm"] = (pk(last) * 1e3).tolist()
        out["drop_settle_mean_of_step_max_mm"] = (np.nanmean(last, axis=0) * 1e3).tolist()
        # landing: first-contact index, peak index, depth 55 ms after the peak, time from peak to < 0.1 mm
        f0 = feet["C_drop"][:, 0]; first = int(np.flatnonzero(~np.isnan(f0))[0]); win = f0[first:first + int(0.2 / dt)]
        ip = int(np.nanargmax(win)); a55 = win[min(ip + int(round(0.055 / dt)), len(win) - 1)]
        below = np.flatnonzero(np.nan_to_num(win[ip:], nan=0.0) < 1e-4)
        out["land_first_contact_ms"] = first * dt * 1e3; out["land_peak_after_ms"] = ip * dt * 1e3
        out["land_depth_55ms_after_peak_mm"] = float(a55 * 1e3) if not np.isnan(a55) else 0.0
        out["land_ms_to_below_0p1mm"] = float(below[0] * dt * 1e3) if len(below) else None
        out["land_profile_5ms_mm"] = [None if np.isnan(x) else round(float(x) * 1e3, 3)
                                      for x in feet["C_drop"][first - (first % stride) - stride: first + int(0.1 / dt): stride, 0]]
    # momentum-derived impulses and 20 ms mean forces, per env
    from metalsim.parity import momentum
    joints = meta["isaac_joints"]; mom = []
    for e in range(n):
        r = {}; JJ = {}
        for t in ("C_drop", "A_hold"):
            A = np.load(os.path.join(ISAAC, f"{t}.npz")); T = min(len(A["joint_pos"]), len(Z[t]["joint_pos"]))
            Ji = momentum.contact_impulse({k: A[k][:T] for k in KEYS}, joints, True, env=0)
            Jo = momentum.contact_impulse({k: Z[t][k][:T] for k in KEYS}, joints, False, env=e)
            r[t] = momentum.event_metrics(Ji, Jo, t); JJ[t] = (Ji, Jo)
        Jd, Jh = JJ["C_drop"], JJ["A_hold"]
        mom.append({"imp_land": r["C_drop"]["landing"]["impulse_Ns"]["ratio"], "imp_torso": r["C_drop"]["torso"]["impulse_Ns"]["ratio"],
                    "imp_hold": r["A_hold"]["torso"]["impulse_Ns"]["ratio"],
                    "f_land": r["C_drop"]["landing"]["max_20ms_mean_force_N"]["metalsim"], "f_torso": r["C_drop"]["torso"]["max_20ms_mean_force_N"]["metalsim"],
                    "f_hold": r["A_hold"]["torso"]["max_20ms_mean_force_N"]["metalsim"],
                    # phase-robust companion: largest 40 ms (two control steps) mean force in the event window, ours / Isaac's
                    **{f"f40_{ev}": _f40(r2[0], r2[1], lo, hi) for ev, r2, (lo, hi) in (("land", Jd, (5, 30)), ("torso", Jd, (55, 90)), ("hold", Jh, (55, 90)))}})
    out["momentum"] = mom
    # chatter (A_hold + C_drop, both feet), per env
    ch = []
    for e in range(n):
        c25 = c5 = 0
        for t in ("A_hold", "C_drop"):
            F = Z[t]["sub_force"]; F = F.reshape(-1, n, 3)[:, e, :2]
            for k in range(2):
                f = F[:, k]; c25 += phases(f > 1.0, dt)[0]; c5 += phases(f.reshape(-1, stride).mean(1) > 1.0, 0.005)[0]
        ch.append((c25, c5))
    out["chatter"] = ch
    # largest 5 ms mean contact force per event (sum over the contact bodies, per-substep sensor forces averaged over 5 ms
    # windows aligned with PhysX's steps), against the bounds on PhysX's (contact_discrepancies_2026-09-25.md, win5_bounds.py)
    w5 = []
    for e in range(n):
        r = {}
        for t, ev, (lo, hi) in (("C_drop", "land", (5, 30)), ("C_drop", "torso", (55, 90)), ("A_hold", "hold", (55, 90))):
            F = Z[t]["sub_force"]; S = F.shape[1]; F = F[:, :, e].sum(-1)            # (T, S)
            f5 = F[lo:hi].reshape(-1, stride).mean(1)
            r[ev] = float(f5.max())
        w5.append(r)
    out["max_5ms_force"] = w5
    out["drop_limit_exc_rad"] = Z["C_drop"]["limit_excursion"].max(0).tolist()
    it = np.concatenate([Z[t]["sub_niter"].reshape(-1) for t in Z])
    out["niter_at_cap_frac"] = float((it >= cap).mean()); out["niter_max"] = int(it.max()); out["niter_mean"] = float(it.mean())
    rep = os.path.join(dr, "report", "report.json")
    if os.path.exists(rep):
        R = json.load(open(rep)); R = R.get("physics", R)
        out["hold_joint_rmse_end_max"] = [R["A_hold"]["joint_rmse_rad"]["end"], R["A_hold"]["joint_rmse_rad"]["max"]]
        out["drop_root_z_end"] = R["C_drop"]["root_height"]["metalsim_end"]
    return out


def _f40(Ji, Jo, lo, hi):
    s2 = lambda J: (J[lo:hi][1:] + J[lo:hi][:-1]).max()
    return float(s2(Jo) / s2(Ji))


def rng(v, f="{:.2f}"):
    v = np.asarray(v, float)
    return f.format(v[0]) + (f" ({f.format(v.min())}–{f.format(v.max())})" if len(v) > 1 and v.max() - v.min() > 0 else "")


def row(s):
    M = s["momentum"]; g = lambda k: np.array([m[k] for m in M])
    c = np.array(s["chatter"])
    return (f"| {s['preset']}{'' if s['dt'] == 0.0025 else ' @ ' + str(s['dt'] * 1e3) + ' ms'}{' (repeat)' if '_r2' in s['dir'] else ''}{' [prototype fork]' if s['dir'].endswith('_spec') else ''} "
            f"| {rng(s['drop_feet_peak_mm'])} | {rng(s['drop_feet_peak_5ms_mm'])} | {rng(s['drop_torso_peak_mm'])} "
            f"| {s['hold_feet_peak_mm'][0]:.2f} / {s['hold_torso_peak_mm'][0]:.2f} | {s['random_feet_peak_mm'][0]:.2f} / {s['random_torso_peak_mm'][0]:.2f} "
            f"| {s['drop_settle_max_mm'][0]:.3f} / {s['drop_settle_mean_of_step_max_mm'][0]:.3f} | {s['land_depth_55ms_after_peak_mm']:.2f} "
            f"| {g('imp_land')[0]:.3f} / {g('imp_torso')[0]:.3f} / {g('imp_hold')[0]:.3f} "
            f"| {g('f_land')[0] / PHYSX['land']:.2f} / {rng(g('f_torso') / PHYSX['torso'])} / {rng(g('f_hold') / PHYSX['hold'])} "
            f"| {g('f40_land')[0]:.2f} / {rng(g('f40_torso'))} / {g('f40_hold')[0]:.2f} "
            f"| {c[0, 0]} / {c[0, 1]} (mean {c[:, 0].mean():.0f} / {c[:, 1].mean():.0f}) | {max(s['drop_limit_exc_rad']):.4f} "
            f"| {100 * s['niter_at_cap_frac']:.2f} % / {s['niter_max']} "
            f"| {s['max_5ms_force'][0]['land']:.0f} / {s['max_5ms_force'][0]['torso']:.0f} / {s['max_5ms_force'][0]['hold']:.0f} |")


HEAD = ("| setting | drop feet peak mm, substep (env range) | drop feet peak mm, 5 ms tick | drop torso peak mm | hold feet / torso mm "
        "| random feet / torso mm | drop settle max / mean mm | depth 55 ms after landing peak mm | impulse ratio land / drop-torso / hold-torso "
        "| 20 ms force / PhysX land / drop-torso / hold-torso | 40 ms force / PhysX land / drop-torso / hold-torso | chatter A+C 2.5 ms / 5 ms | drop limit exc. rad | substeps at cap / max iter | max 5 ms force N land / drop-torso / hold-torso |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| **Isaac 5.1 PhysX (GPU)** | – | **0.51** (L 0.51 / R 0.43) | 0.10 | – | – | 0.013 / 0.013 | < 0.1 (0.08 at 55 ms) | 1 | 1 (1996 / 2200 / 2190 N) | 1 | – | 0.004 | – | 2174–6522 / 2618–7854 / 2682–8045 (bounds) |\n")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    S = [summarize(d) for d in args]
    print(HEAD + "\n".join(row(s) for s in S))
    if "--profile" in sys.argv:
        for s in S:
            print(s["preset"], s["dir"], "first contact", s["land_first_contact_ms"], "ms; profile (5 ms ticks, L foot, mm):", s["land_profile_5ms_mm"][:16])
    js = os.path.join(ROOT, "runs/penetration/summary.jsonl")
    with open(js, "a") as f:
        for s in S: f.write(json.dumps(s) + "\n")
