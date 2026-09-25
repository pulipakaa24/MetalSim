"""Foot contact flicker in the fidelity protocols (MuJoCo per-substep recordings from substep_forces.py): on/off
transitions of Isaac's contact flag (|net normal force| > 1 N) per foot, and how many of them survive when the
flag is evaluated like PhysX reports it (force averaged over 5 ms = 2 substeps, updated every 5 ms).
Short phases (< 20 ms) are flicker: they reset Isaac's current_air_time / current_contact_time."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def phases(c, dt):
    tr = np.flatnonzero(np.diff(c.astype(int)) != 0)
    ln = np.diff(np.r_[0, tr + 1, len(c)]) * dt; st = np.r_[c[0], c[tr + 1]]
    inner = slice(1, -1)                           # drop the first/last (unfinished) phases
    return len(tr), int(((~st[inner]) & (ln[inner] < 0.02)).sum()), int((st[inner] & (ln[inner] < 0.02)).sum())


def main():
  for p in sys.argv[1:] or ["default", "tau10_impact_hardlimits", "tau5_imp99_hardlimits"]:
      Z = np.load(os.path.join(ROOT, f"runs/contact_research/substep_c_{p}.npz")); b = list(Z["bodies"])
      for tag in ("A_hold", "C_drop", "B_random"):
          F = Z[f"{tag}_force"]; out = []
          for nm in ("left_ankle_roll_link", "right_ankle_roll_link"):
              f = np.linalg.norm(F[:, b.index(nm)], axis=-1)
              f5 = f.reshape(-1, 2).mean(1)
              n25, sa25, sc25 = phases(f > 1.0, 0.0025); n5, sa5, sc5 = phases(f5 > 1.0, 0.005)
              out.append(f"{nm.split('_')[0]:5s} 2.5 ms: {n25:3d} transitions ({sa25:2d} air / {sc25:2d} contact < 20 ms); "
                         f"5 ms avg: {n5:3d} ({sa5:2d} / {sc5:2d})")
          print(f"{p:24s} {tag:8s} " + " | ".join(out))


if __name__ == "__main__":
    main()
