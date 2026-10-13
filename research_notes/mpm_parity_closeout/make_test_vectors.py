"""Pick regression vectors for the Warp fork's Metal CUDA-libm test from the L4's results: inputs where the Metal
library's function differs from CUDA's (plus special values), with the L4's result bits."""
import numpy as np

W = "/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c/runs/mpm/closeout"
rng = np.random.default_rng(7)
for fn in ["log", "exp", "sinh"]:
    c = np.load(f"/Users/aditya/robosim/.claude/worktrees/agent-a8a12903da01bb68c/runs/parity/isaaclab3_mpm_closeout/libm_cuda/{fn}.npz")
    m = np.load(f"{W}/libm_metal_orig/{fn}.npz")
    x, yc, ym = c["sample_x"], c["sample_y"], m["sample_y"]
    xf = x.view(np.float32)
    fin = np.isfinite(yc.view(np.float32)) & np.isfinite(xf)
    diff = np.where(fin & (yc != ym))[0]
    same = np.where(fin & (yc == ym))[0]
    # spread over magnitudes: sort differing inputs by |x| and take evenly spaced ones
    order = diff[np.argsort(np.abs(xf[diff]))]
    pick = order[np.linspace(0, order.size - 1, 40).astype(int)]
    pick = np.concatenate([pick, rng.choice(same, 8, replace=False)])
    print(f"    \"{fn}\": [")
    for i in pick:
        print(f"        (0x{int(x[i]):08x}, 0x{int(yc[i]):08x}),  # {float(xf[i])!r}")
    print("    ],")
