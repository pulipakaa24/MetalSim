"""bitcmp.py DIR_A DIR_B [run ...]: bitwise comparison of record_mpm.py outputs (every array of init/traj/solves npz)."""
import glob, os, sys
import numpy as np

a, b = sys.argv[1], sys.argv[2]
runs = sys.argv[3:] or sorted(os.path.basename(p) for p in glob.glob(os.path.join(a, "*")) if os.path.isdir(p))
ok_all = True
for r in runs:
    msgs = []
    for f in ("init.npz", "traj.npz", "solves.npz"):
        pa, pb = os.path.join(a, r, f), os.path.join(b, r, f)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            msgs.append(f"{f} missing"); continue
        za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
        if set(za.files) != set(zb.files):
            msgs.append(f"{f} keys differ"); continue
        for k in za.files:
            x, y = za[k], zb[k]
            if not (x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()):
                msgs.append(f"{f}:{k} differs")
    ok_all &= not msgs
    print(f"{r:24s} {'BITWISE EQUAL' if not msgs else 'DIFFERS: ' + '; '.join(msgs[:6])}")
sys.exit(0 if ok_all else 1)
