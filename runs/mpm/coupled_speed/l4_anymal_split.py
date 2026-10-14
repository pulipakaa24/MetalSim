"""Split the L4's per-kernel GPU time for ANYmal (nsys sums, 4 frames) into MPM kernels and the rest (robot: MuJoCo
Warp, Newton FK, policy). Name-pattern heuristic; prints both sums and the largest non-MPM kernels.
    python3 l4_anymal_split.py CSV [FRAMES]"""
import csv
import re
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
frames = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
mpm = re.compile(r"gs_solve|subgrid|integrate_velocity|advect|strain|collision_weight|rasterize|identity_field|"
                 r"dispatch_(bi)?linear|bsr|delassus|apply_stress|yield|postprocess|project_outside|quadrature|node|"
                 r"cell|partition|nanogrid|volume|fem|tiled_sum|update_condition|evaluate_strain|conditional|radix|"
                 r"onesweep|scan|reduce|runlength|select|memset|memcpy|copy|fill|particle|grid|color|pic|scatter|"
                 r"gather|compress|unique|sort|triplet|segment|sample|vb_|warp_", re.I)
tot = {"mpm": 0.0, "other": 0.0}
other = {}
for r in rows:
    t = float(r["Total Time (ns)"]) / 1e6 / frames
    if mpm.search(r["Name"]):
        tot["mpm"] += t
    else:
        tot["other"] += t
        other[r["Name"][:80]] = other.get(r["Name"][:80], 0.0) + t
print({k: round(v, 2) for k, v in tot.items()}, "ms/frame")
for n, t in sorted(other.items(), key=lambda x: -x[1])[:12]:
    print(f"{t:7.3f}  {n}")
