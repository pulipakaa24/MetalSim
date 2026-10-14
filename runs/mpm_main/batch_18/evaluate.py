"""batch_18 verdict from the probe JSONs and the G1 runs' exit codes.

    python3 evaluate.py OUT_DIR RC_G1_64 RC_G1_1024 NOTE_MD
Prints the verdict on its last line: M1_CONFIRMED | CAPTURE_ONLY | FIX_INCOMPLETE | INCONCLUSIVE, and appends the
results section to the research note.
"""
import json
import pathlib
import sys

o, rc64, rc1024, note = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])


def load(name):
    try:
        return json.load(open(o / f"{name}.json"))
    except Exception:  # noqa: BLE001
        return None


runs = {n: load(n) for n in ("pre_stack_eager", "pre_stack_capt", "pre_g1_64", "pre_g1_1024",
                             "fix_stack_eager", "fix_stack_capt", "fix_g1_64", "fix_g1_1024")}
lines = []
for n, r in runs.items():
    if r is None:
        lines.append(f"- `{n}`: no result")
        continue
    lines.append(f"- `{n}` ({r['scene']}, {r['worlds']} worlds, capture {r['capture']}, cap {r['iteration_cap']}): steps "
                 f"with cap worlds {len(r['steps_with_cap_worlds'])}/{r['steps']}, steps with trips "
                 f"{len(r['steps_with_hits'])}, trips only in cap steps {r['hits_only_in_cap_steps']}, kernels "
                 f"{r['kernels_hit']}, flag values {r['flag_values']}")
lines.append(f"- G1 deterministic, captured, everything on, with the fix (det_cost_g1.py, check_overflow): exit code "
             f"64 worlds {rc64}, 1,024 worlds {rc1024}")


def solve_done_only_kind2(r):
    return (r is not None and r["kernels_hit"] and all(k.startswith("_solve_done") for k in r["kernels_hit"])
            and set(r["flag_values"]) <= {"2"} and r["hits_only_in_cap_steps"])


def clean(r):
    return r is not None and not r["kernels_hit"]


pre_e, pre_c = runs["pre_stack_eager"], runs["pre_stack_capt"]
fixed = all(clean(runs[n]) for n in ("fix_stack_eager", "fix_stack_capt", "fix_g1_64", "fix_g1_1024")) and rc64 == "0" and rc1024 == "0"
if solve_done_only_kind2(pre_e) and solve_done_only_kind2(pre_c) and fixed:
    verdict = "M1_CONFIRMED"
    text = ("M1 confirmed: before the fix the uncaptured and captured cap-2 runs trip in `_solve_done` only, with kind 2 "
            "(a phase-1 counter call with no phase-0 record), only in steps where worlds reach the cap; with the fix "
            "every run is clean, including the G1 configuration batch_16 failed.")
elif clean(pre_e) and pre_c is not None and pre_c["kernels_hit"]:
    verdict = "CAPTURE_ONLY"
    text = "Capture-only trip (M2-M4): the uncaptured cap-2 run is clean, the captured one trips; see kernels and kinds."
elif (pre_e and pre_e["kernels_hit"]) and not fixed:
    verdict = "FIX_INCOMPLETE"
    text = "The M1 trip is reproduced but the fix does not clean every run: see the remaining kernels and kinds."
else:
    verdict = "INCONCLUSIVE"
    text = "The runs do not match any single mechanism's prediction: see the per-run lines."
note.write_text(note.read_text().rstrip("\n") + "\n\n## 6. Results (batch_18, 16-inch)\n\n" + text + "\n\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
print(verdict)
