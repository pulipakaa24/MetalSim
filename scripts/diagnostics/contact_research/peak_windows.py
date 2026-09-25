"""Contact-force peaks of MuJoCo's per-substep recordings under different reporting windows, next to Isaac's
recorded (PhysX, control-step-sampled, 5 ms-averaged) values. Per protocol, per body group (feet, torso),
and per event (foot landing, torso impact).

  inst      : max over all 2.5 ms substeps (MuJoCo's own resolution; each value is already the constraint
              impulse of that step / dt)
  ours_ctrl : the last substep of each control step (what metalsim.parity.record_g1 records; the tables' peaks)
  isaac_eq  : mean of the last 2 substeps (= the last 5 ms) of each control step: Isaac's report
              (get_net_contact_forces(dt = 5 ms) of the last PhysX step of the decimation), applied to our physics
  win5_max  : max over all aligned 5 ms windows (the peak PhysX's reporting could show at the best phase)
  win20     : mean over the whole 20 ms control step (impulse per control step / 20 ms)
  impulse   : integral of the vertical force over the event (N s), MuJoCo substeps; Isaac: sum of recorded
              control-step samples x 20 ms (an estimate that assumes each sample stands for its control step)

    python scripts/diagnostics/contact_research/peak_windows.py runs/contact_research/substep_c_default.npz [...]
"""
import sys, json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ISAAC = os.path.join(ROOT, "runs/parity/isaac/parity_out2/rt")
GROUPS = {"feet": ["left_ankle_roll_link", "right_ankle_roll_link"], "torso": ["torso_link"]}
EVENTS = {"C_drop": {"landing": (0, 40), "torso_impact": (40, 150)}, "A_hold": {"torso_impact": (0, 150)},
          "B_random": {"all": (0, 250)}}


def per_body_norm(F):
    return np.linalg.norm(F, axis=-1)


def analyse(fn):
    Z = np.load(fn); dec = int(Z["dec"]); dt = float(Z["dt"]); bodies = list(Z["bodies"])
    res = {"file": os.path.basename(fn)}
    for tag, evs in EVENTS.items():
        if f"{tag}_force" not in Z.files: continue
        F = Z[f"{tag}_force"]; T = F.shape[0] // dec
        Ia = np.load(os.path.join(ISAAC, f"{tag}.npz"))["contact"][:, 0]            # (T, B, 3)
        for g, names in GROUPS.items():
            idx = [bodies.index(n) for n in names]
            for ev, (t0, t1) in evs.items():
                n_sub = per_body_norm(F[:, idx]).reshape(T, dec, len(idx))[t0:t1]          # (steps, dec, nb)
                fz_sub = F[:, idx, 2].reshape(T, dec, len(idx))[t0:t1]
                ia = per_body_norm(Ia[t0:t1, idx])
                pairs = n_sub.reshape(-1, 2, len(idx)).mean(1) if dec % 2 == 0 else None     # aligned 5 ms windows
                r = {"inst": float(n_sub.max()), "ours_ctrl": float(n_sub[:, -1].max()),
                     "isaac_eq": float(n_sub[:, -2:].mean(1).max()), "win5_max": float(pairs.max()),
                     "win20": float(n_sub.mean(1).max()),
                     "impulse_Ns": float(fz_sub.sum((0, 1)).sum() * dt),
                     "isaac_recorded": float(ia.max()), "isaac_impulse_est_Ns": float(Ia[t0:t1, idx, 2].sum() * 0.02),
                     "inst_step": int(np.unravel_index(n_sub.argmax(), n_sub.shape)[0] + t0),
                     "isaac_step": int(ia.max(1).argmax() + t0)}
                res[f"{tag}/{g}/{ev}"] = r
    return res


if __name__ == "__main__":
    rows = [analyse(f) for f in sys.argv[1:]]
    keys = ["inst", "win5_max", "win20", "ours_ctrl", "isaac_eq", "isaac_recorded", "impulse_Ns", "isaac_impulse_est_Ns", "inst_step", "isaac_step"]
    for r in rows:
        print("==", r["file"]); print(f"{'protocol/group/event':34s}" + "".join(f"{k:>12s}" for k in keys))
        for k, v in r.items():
            if k == "file": continue
            print(f"{k:34s}" + "".join(f"{v[c]:12.1f}" if isinstance(v[c], float) else f"{v[c]:12d}" for c in keys))
    out = os.path.join(ROOT, "runs/contact_research/peak_windows.json")
    json.dump(rows, open(out, "w"), indent=1)
