"""Table of the coupled-scene re-timing (run.sh): ms/frame per scene, mode and protocol, per repetition and the
median over repetitions, against the L4's recorded ms/frame (MPM.md sections 7 / 8, `stage_mpm_timing.sh`).

    python3 runs/mpm/coupled_retime/summarize.py [--json out.json]
L protocol: median of record_mpm.py's per-frame wall_s over the whole 2 s run (frame 0, which compiles, included in
the median like on the L4); also the median of frames 6-65 for comparison with S.
S protocol: speed_bench.py's ms_median (60 frames after 5 warmup).
"""
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# L4 (Isaac Lab 3.0-EA's Newton 1.5.2 / Warp 1.16, stage_mpm_timing.sh, 2 s, median ms/frame): MPM.md section 7.
L4 = {"twoway": 73.0, "twoway_capture": 31.0, "anymal": 58.9, "anymal_capture": 24.2}
SCENES = ["twoway", "twoway_capture", "anymal", "anymal_capture"]
MODES = ["off", "relaxed"]

rows = {}
for s in SCENES:
    for m in MODES:
        L, L6, S, waits = [], [], [], []
        for d in sorted(glob.glob(f"{HERE}/L_{s}_{m}_r*/meta.json")):
            w = json.load(open(d))["wall_s"]
            L.append(statistics.median(w) * 1e3)
            L6.append(statistics.median(w[5:65]) * 1e3)
        for j in sorted(glob.glob(f"{HERE}/S_{s}_{m}_r*.json")):
            r = json.load(open(j))
            S.append(r["ms_median"])
            waits.append(r["per_frame_counters"]["host_waits"])
        rows[(s, m)] = {"L": L, "L_frames6_65": L6, "S": S, "S_host_waits": waits}


def med(x):
    return statistics.median(x) if x else float("nan")


print("| scene | mode | L4 ms/frame | Metal, L4 protocol (2 s): per rep -> median | Metal / L4 | "
      "Metal, speed_suite protocol (frames 6-65): per rep -> median | host waits / frame |")
print("|---|---|---|---|---|---|---|")
out = {}
for s in SCENES:
    for m in MODES:
        r = rows[(s, m)]
        lm, sm = med(r["L"]), med(r["S"])
        out[f"{s}/{m}"] = {**r, "L_median": lm, "S_median": sm, "L4": L4[s], "ratio_L": lm / L4[s]}
        print(f"| {s} | {m} | {L4[s]:.1f} | {' / '.join(f'{v:.1f}' for v in r['L'])} -> **{lm:.1f}** | "
              f"**{lm / L4[s]:.2f}x** | {' / '.join(f'{v:.1f}' for v in r['S'])} -> {sm:.1f} | "
              f"{med(r['S_host_waits']):.1f} |")
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
