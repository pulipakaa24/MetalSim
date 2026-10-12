"""Snow Jp after step 1: L4 base vs Metal vs L4 nofma, in float32 ulps, and the implied exp() argument."""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm_per_material import C  # noqa: E402

W = "/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c/runs/mpm/closeout"
b = np.load(f"/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c/runs/parity/isaaclab3_mpm_closeout/steps_mm_base/steps.npz")["particle_Jp"]
m = np.load(sys.argv[1] if len(sys.argv) > 1 else f"{W}/steps/metal_rsq_mm/steps.npz")["particle_Jp"]
n = np.load(f"/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c/runs/parity/isaaclab3_mpm_closeout/steps_mm_nofma/steps.npz")["particle_Jp"]
s = slice(C[2], C[3])
for k in range(2):
    jb, jm, jn = b[k, s], m[k, s], n[k, s]
    print(f"step {k + 1}: snow Jp base: min {jb.min():.9g} max {jb.max():.9g}, != 1 on {(jb != 1).sum()}")
    d = jm.view(np.int32).astype(np.int64) - jb.view(np.int32).astype(np.int64)
    dn = jn.view(np.int32).astype(np.int64) - jb.view(np.int32).astype(np.int64)
    u, c = np.unique(d, return_counts=True)
    print("  metal - base (ulp):", dict(zip(u.tolist(), c.tolist())) if u.size < 30 else (u.min(), u.max(), (d != 0).sum()))
    u, c = np.unique(dn, return_counts=True)
    print("  nofma - base (ulp):", dict(zip(u.tolist(), c.tolist())) if u.size < 30 else (u.min(), u.max(), (dn != 0).sum()))
    if k == 0:
        sel = d != 0
        print("  Jp-1 where metal differs: ", np.percentile((jb[sel].astype(np.float64) - 1), [0, 50, 100]))
        print("  Jp-1 where equal & !=1: ", np.percentile((jb[(~sel) & (jb != 1)].astype(np.float64) - 1), [0, 50, 100]) if ((~sel) & (jb != 1)).any() else "none")
        print("  sample (base, metal):", [(float(x), float(y)) for x, y in zip(jb[sel][:6], jm[sel][:6])])
