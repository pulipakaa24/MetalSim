"""so101-film-bam: the arm's droop in the hold, seed 10051, 7v4_real against 7v4_bam, from the MetalSim evals' per-physics
step trajectories (traj.npz: held command cmd, joint position q) and, when given, the two takes' recovered joint angles
(take_measure.py JSON). Hold = lift phase, 3 s < t <= 6 s (so101-expert's contacts.py window).

    .venv/bin/python runs/readme_media/scratch/so101-film-bam/droop.py [take_measure_real.json take_measure_bam.json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.abspath(os.path.join(HERE, ".."))
TRAJ = dict(real=os.path.join(S, "so101-expert/metalsim/deploy_7v4_real/traj.npz"),
            bam=os.path.join(S, "so101-bam-gpu/metalsim/deploy_7v4_bam/traj.npz"))
SEED = 10051
J = ("pan", "lift", "elbow", "wflex", "wroll", "grip")
out = {}
for k, p in TRAJ.items():
    z = np.load(p)
    e = int(np.flatnonzero(z["seeds"] == SEED)[0])
    dt = float(z["dt"])
    n = np.arange(z["q"].shape[1])
    t = (n + 1) * dt
    hold = (t > 3.0) & (z["phase"][e] == 3)
    err = (z["cmd"][e] - z["q"][e])[hold]
    out[k] = dict(sag_mrad_median=(np.median(err, 0) * 1e3).round(1).tolist(),
                  q_hold_median=np.median(z["q"][e][hold], 0).round(4).tolist(),
                  cmd_hold_median=np.median(z["cmd"][e][hold], 0).round(4).tolist(),
                  q_frames={}, n_hold=int(hold.sum()))
    for f in (60, 110, 149, 199, 249, 299):            # take frame f = the state after physics step 4 (f + 1) - 1
        out[k]["q_frames"][f] = z["q"][e][4 * (f + 1) - 1].round(4).tolist()
    print(k, "sag (cmd - q) median in the hold, mrad", dict(zip(J, out[k]["sag_mrad_median"])))
    print("   cmd", dict(zip(J, out[k]["cmd_hold_median"])), "\n   q  ", dict(zip(J, out[k]["q_hold_median"])))
d = np.array(out["bam"]["q_hold_median"]) - np.array(out["real"]["q_hold_median"])
dc = np.array(out["bam"]["cmd_hold_median"]) - np.array(out["real"]["cmd_hold_median"])
print("q bam - real in the hold, mrad", dict(zip(J, (d * 1e3).round(1).tolist())))
print("cmd bam - real in the hold, mrad", dict(zip(J, (dc * 1e3).round(1).tolist())))
out["q_bam_minus_real_mrad"] = (d * 1e3).round(1).tolist()
out["cmd_bam_minus_real_mrad"] = (dc * 1e3).round(1).tolist()
if len(sys.argv) > 2:                                  # the takes: recovered joint angles against the evals
    for k, p in zip(("real", "bam"), sys.argv[1:3]):
        tm = list(json.load(open(p)).values())[0]
        q = np.array(tm["q"])
        dev = [float(np.abs(q[f, :5] - np.array(out[k]["q_frames"][f][:5])).max()) for f in out[k]["q_frames"]]
        out[k]["take_vs_eval_max_abs_rad_at_frames"] = [round(x, 5) for x in dev]
        out[k]["take_q_frames"] = {f: q[f].round(4).tolist() for f in out[k]["q_frames"]}
        out[k]["take_tilt_deg_frames"] = {f: tm["tilt_deg"][f] for f in out[k]["q_frames"]}
        out[k]["take_box_z_frames"] = {f: tm["box_z"][f] for f in out[k]["q_frames"]}
        print(k, "take vs eval, max |dq| over arm joints at frames", dict(zip(out[k]["q_frames"], out[k]["take_vs_eval_max_abs_rad_at_frames"])))
    qr, qb = (np.array(list(out[k]["take_q_frames"].values())) for k in ("real", "bam"))
    print("take q bam - real at frames (mrad, lift / elbow / wflex):")
    for f, a, b in zip(out["real"]["take_q_frames"], qr, qb):
        print(f"  frame {f}: {((b - a)[1:4] * 1e3).round(1).tolist()}  tilt real {out['real']['take_tilt_deg_frames'][f]} "
              f"bam {out['bam']['take_tilt_deg_frames'][f]}  box z real {out['real']['take_box_z_frames'][f]} "
              f"bam {out['bam']['take_box_z_frames'][f]}")
json.dump(out, open(os.path.join(HERE, "droop.json"), "w"), indent=1)
