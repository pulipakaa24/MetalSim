"""Which sinh inputs differ between the L4 and the Metal emulation (random sample)."""
import numpy as np

W = "runs/mpm/closeout"
a = np.load(f"runs/parity/isaaclab3_mpm_closeout/libm_cuda/sinh.npz")
b = np.load(f"{W}/libm_metal_emul/sinh.npz")
x = a["sample_x"].view(np.float32)
ya, yb = a["sample_y"].view(np.float32), b["sample_y"].view(np.float32)
d = ya.view(np.uint32) != yb.view(np.uint32)
ax = np.abs(x[d])
print("differ:", d.sum(), "|x| range", ax.min(), ax.max())
for lo, hi in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 90)]:
    s = (np.abs(x) >= lo) & (np.abs(x) < hi)
    print(f"  [{lo},{hi}): {d[s].mean() * 100:.1f} % of {s.sum()}")
i = np.where(d)[0][:10]
for j in i:
    print(f"x={x[j]!r:>14} cuda={ya[j]!r} metal={yb[j]!r} ref={np.sinh(np.float64(x[j]))!r}")
