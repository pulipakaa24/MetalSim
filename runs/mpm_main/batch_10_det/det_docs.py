"""Deterministic-mode cost (batch_10) into #29: research note, GAPS row, PR body.

    python3 det_docs.py REPO RUNS_DIR PR_BODY
"""
import json
import pathlib
import statistics
import sys

repo, runs, body = (pathlib.Path(p) for p in sys.argv[1:4])
res = {}
for scene, key in (("g1", "mean_step_ms"), ("box", "ms_per_step_median")):
    for n in (64, 1024):
        for det in (0, 1):
            v = [json.load(open(p))[key] for p in sorted(runs.glob(f"{scene}_{n}_{det}_r*.json"))]
            if not v:
                print(f"missing {scene} {n} {det}")
                sys.exit(1)
            res[(scene, n, det)] = (statistics.median(v), v)
names = {"g1": "G1 flat velocity task, env step", "box": "coupled box on sand, coupled step"}
rows, cost = [], {}
for scene in ("g1", "box"):
    for n in (64, 1024):
        off, on = res[(scene, n, 0)][0], res[(scene, n, 1)][0]
        cost[(scene, n)] = 100 * (on / off - 1)
        rows.append(f"| {names[scene]} | {n:,} | {off:.2f} | {on:.2f} | {cost[(scene, n)]:+.1f} % | "
                    f"{', '.join(f'{x:.2f}' for x in res[(scene, n, 0)][1])} / {', '.join(f'{x:.2f}' for x in res[(scene, n, 1)][1])} |")
table = ("| scene | worlds | default, ms | deterministic, ms | cost | reps default / deterministic |\n"
         "|---|---|---|---|---|---|\n" + "\n".join(rows))
short = (f"G1 flat {cost[('g1', 64)]:+.0f} % / {cost[('g1', 1024)]:+.0f} %, coupled box scene "
         f"{cost[('box', 64)]:+.0f} % / {cost[('box', 1024)]:+.0f} % at 64 / 1,024 worlds")

p = repo / "research_notes/MPM in main simulator/deterministic_ccd_divergence.md"
p.write_text(p.read_text().rstrip("\n") + f"""

## Cost (batch_10, 16-inch, runs/mpm_main/batch_10_det)

`BatchSimOptions(deterministic=True)` against the default, each run its own process (the mode is process-wide), three
interleaved repetitions, timing class. G1: Isaac Lab's flat velocity task, one captured graph per env step (actions,
4 physics substeps, observations, rewards, resets; `scripts/diagnostics/det_cost_g1.py`, mean of 100 steps after 20).
Coupled box: `probes/scale_coupled.py --time` (3 MuJoCo substeps + 1 MPM step, median of 40 after 10). Medians:

{table}
""")
print("note updated")

p = repo / "docs/GAPS.md"
s = p.read_text()
old = "the default mode stays non-repeatable (cost of the deterministic mode not measured)"
if old not in s:
    print("GAPS anchor missing")
    sys.exit(1)
p.write_text(s.replace(old, f"the default mode stays non-repeatable; the deterministic mode costs {short} "
                            "(research_notes/MPM in main simulator/deterministic_ccd_divergence.md)"))
print("GAPS updated")

b = body.read_text()
marker = "\n🤖 Generated with"
bullet = (f"- Cost (batch_10, 16-inch, interleaved timing jobs): {short}. Table in the research note.\n")
if "- Cost (batch_10" not in b:
    i = b.index(marker)
    b = b[:i].rstrip("\n") + "\n" + bullet + b[i:]
    body.write_text(b)
print("PR body updated")
print(table)
