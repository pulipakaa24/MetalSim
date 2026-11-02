"""so101-bam-gpu: how much of the servo's torque the deploy expert uses, per arm joint, from eval_metalsim_bam.py's
traj.npz (frc: actuator force per physics step; phase per step). Against the datasheet's 7.4 V stall (19.5 kg.cm =
1.912 N.m; the gripper against 7v4_real's cap, LeRobot's 50 % torque limit: 0.986 N.m) and the run's own clip
(episodes.json servo forcerange_Nm). Tells whether 7v4_bam's BAM stall (3.32 N.m) against the datasheet's matters
for this task: only the steps above 1.912 N.m (gripper: above 7v4_real's cap) can differ.

  .venv/bin/python stall_use.py <run dir> [...]
"""
import json
import os
import sys

import numpy as np

J = ("pan", "lift", "elbow", "wflex", "wroll", "grip")
PH = ("approach", "descend", "grasp", "lift")
DATASHEET = 19.5 * 0.0980665                       # kg.cm -> N.m, Feetech STS3215 7.4 V (so101.py KGCM)
for run in sys.argv[1:]:
    z = np.load(os.path.join(run, "traj.npz"))
    ep = json.load(open(os.path.join(run, "episodes.json")))
    fr = ep["servo"]["applied"] if "applied" in ep["servo"] else ep["servo"]
    f, ph = np.abs(z["frc"]), z["phase"]
    ok = ~np.isnan(f[..., 0])
    caps = np.array([DATASHEET] * 5 + [0.986])     # 7v4_real's grip cap: 50 % duty of 1.912 / 0.97
    clip = np.array(fr["forcerange_Nm"])
    print(f"== {run}  (own clip {clip.round(3).tolist()})")
    print("joint   max N.m  p99.9  at own clip  steps > datasheet cap +0.1 % (all / per phase approach, descend, grasp, lift)  episodes with any")
    for j, nm in enumerate(J):
        x = f[..., j][ok]
        atclip = float((x >= 0.999 * clip[j]).mean() * 100)
        above = f[..., j] > caps[j] * 1.001
        per = [int((above & ok & (ph == i)).sum()) for i in range(4)]
        eps = int((above & ok).any(1).sum())
        print(f"{nm:6s} {x.max():7.3f} {np.quantile(x, 0.999):7.3f}  {atclip:6.2f} %   {int((above & ok).sum()):6d} ({100 * (above & ok).sum() / ok.sum():.3f} %)"
              f"   {per}   {eps}")
