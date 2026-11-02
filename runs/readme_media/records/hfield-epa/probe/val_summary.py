"""Summarize the V1 training smokes (stream hfield-epa): per-iteration time over iterations 4-8 from the cumulative
sps lines (t_N = steps_N / sps_N), EPA_HORIZON worlds (train_wrap.py's [rr-guard] lines), loaded MuJoCo Warp commit.
Usage: python val_summary.py <val dir>"""
import glob
import os
import re
import sys

import numpy as np

out = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(out, "*.out"))):
  tag = os.path.basename(f)[:-4]
  txt = open(f).read()
  logf = os.path.join(out, tag + ".log")
  if not os.path.exists(logf):
    continue
  t = {}
  for m in re.finditer(r"it\s+(\d+) steps\s+(\d+) sps\s+([\d,]+)", open(logf).read()):
    t[int(m.group(1))] = int(m.group(2)) / float(m.group(3).replace(",", ""))
  per = (t[8] - t[3]) / 5 if 8 in t and 3 in t else float("nan")
  epa = re.findall(r"EPA_HORIZON flagged in (\d+) worlds", txt)
  com = re.search(r"commit (\w+)", txt)
  pw = re.search(r"power: ([^|]+)\|", txt)
  rows[tag] = per
  print(f"{tag:14s} {com.group(1)[:8] if com else '?':8s} per-iteration (it 4-8) {per:.3f} s; EPA_HORIZON worlds {epa[-1] if epa else 0}; "
        f"{pw.group(1).strip() if pw else ''}")
for cfg in ("il3", "s0"):
  g = {k: np.mean([v for t, v in rows.items() if t.startswith(f"{cfg}_{k}")]) for k in ("pin", "base", "fix")}
  g = {k: v for k, v in g.items() if np.isfinite(v)}
  if "base" in g and "fix" in g:
    s = f"{cfg}: mean per-iteration " + ", ".join(f"{k} {v:.3f} s" for k, v in g.items())
    s += f"; fix / 93e5ce1 {g['fix'] / g['base']:.3f}"
    if "pin" in g:
      s += f"; 93e5ce1 / pinned {g['base'] / g['pin']:.3f}; fix / pinned {g['fix'] / g['pin']:.3f}"
    print(s)
