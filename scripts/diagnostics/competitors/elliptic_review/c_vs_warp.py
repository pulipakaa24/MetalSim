"""Final qvel after 8 substeps: C warm / C cold vs Warp warm (float32), elliptic imp10 and pyramidal; plus C warm
with alpha==0 exits counted (solver stats: an iteration whose improvement is 0 at exit)."""
import numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
pre = "runs/competitors/g1_states/ellip10"; z = np.load(pre + ".npz"); W = 256
src = open("scripts/diagnostics/competitors/elliptic_review/warmstart_probe.py").read(); exec(src[src.index("def model("):src.index("def line(")])
for cone, imp in [(1, 10.0), (1, 1.0), (0, 1.0)]:
  _, cw = runC(cone, imp, False); _, cc = runC(cone, imp, True); _, ww = runW(cone, imp, False)
  for tag, a in (("C warm", cw), ("C cold", cc)):
    dv = np.abs(a - ww).max(1)
    print(f"cone {cone} imp {imp}: {tag} vs Warp warm: median {np.median(dv):.1e} p99 {np.percentile(dv,99):.1e} max {dv.max():.1e} worlds >1e-3 {(dv>1e-3).sum()} of {len(dv)}", flush=True)
