"""Merge fix: scale_coupled.build() reads a.deterministic (the branch's --deterministic), but main's probes
(one_world_reset.py, policy_step_ab.py, #33) build their argparse.Namespace without it. Default to 0.

    python3 fix_scale_coupled.py REPO
"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1]) / "metalsim/physics/mpm/probes/scale_coupled.py"
s = p.read_text()
old = "options=BatchSimOptions(substeps=3, deterministic=bool(a.deterministic)))"
new = "options=BatchSimOptions(substeps=3, deterministic=bool(getattr(a, \"deterministic\", 0))))"
if s.count(new) == 1:
    print("already")
elif s.count(old) == 1:
    p.write_text(s.replace(old, new))
    print("fixed")
else:
    sys.exit("build() line not found as expected")
