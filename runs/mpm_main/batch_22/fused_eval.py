"""Decision for section 9 (batch_22): H-det vs H-default, from arms A (fused), B and C (unfused twice; the floor).

    python3 fused_eval.py DIR B18_FIX_G1_64_JSON B19_FIX_G1_64_JSON NOTE_MD
Prints the result lines and, last, the decision: H_DET, H_DEFAULT or INVALID. Appends "Result (batch_22)" to NOTE_MD.
"""
import json
import pathlib
import sys

import numpy as np

d, b18, b19, note = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])
J = {k: json.loads((d / f"{k}.json").read_text()) for k in "ABC"}
Z = {k: np.load(d / f"{k}.npz") for k in "ABC"}


def med_dq(x, y):   # per step: median over worlds of max |dqpos| per world
    return np.median(np.abs(x["qpos"] - y["qpos"]).max(axis=2), axis=1)


ab, bc = med_dq(Z["A"], Z["B"]), med_dq(Z["B"], Z["C"])
cA, cB, cC = (J[k]["steps_with_cap_worlds"] for k in "ABC")
mA, mB, mC = (J[k]["mean_niter"] for k in "ABC")
checks = {
    "cap steps": abs(cA - cB) <= 2 * abs(cB - cC) + 2,
    "mean solver_niter": abs(mA - mB) <= 2 * abs(mB - mC) + 0.05 * mB,
    "qpos (last step)": ab[-1] <= 10 * bc[-1] + 1e-6,
}
invalid = []
if not J["A"]["fused_kernel_ran"]:
    invalid.append("arm A did not run the fused kernel")
if J["B"]["fused_kernel_ran"] or J["C"]["fused_kernel_ran"]:
    invalid.append("arm B or C ran the fused kernel")
dec = "INVALID" if invalid else ("H_DET" if all(checks.values()) else "H_DEFAULT")
x18, x19 = json.load(open(b18)), json.load(open(b19))
lines = [
    f"- A, default mode, fused: steps with cap worlds {cA}/{J['A']['steps']}, cap world-steps {J['A']['cap_world_steps']}, "
    f"mean solver_niter {mA:.3f}, fused kernel ran {J['A']['fused_kernel_ran']}",
    f"- B, default mode, unfused: {cB}/{J['B']['steps']}, {J['B']['cap_world_steps']}, {mB:.3f}",
    f"- C, default mode, unfused (repeat): {cC}/{J['C']['steps']}, {J['C']['cap_world_steps']}, {mC:.3f}",
    f"- median over worlds of max |Δqpos|, last step: A-B {ab[-1]:.3g}, B-C (floor) {bc[-1]:.3g}; "
    f"step 10 / 30: A-B {ab[min(9, len(ab) - 1)]:.3g} / {ab[min(29, len(ab) - 1)]:.3g}, "
    f"B-C {bc[min(9, len(bc) - 1)]:.3g} / {bc[min(29, len(bc) - 1)]:.3g}",
    f"- D, deterministic, fused (batch_18 fix_g1_64): {len(x18['steps_with_cap_worlds'])}/{x18['steps']} steps with cap worlds",
    f"- E, deterministic, unfused (batch_19 fix_g1_64): {len(x19['steps_with_cap_worlds'])}/{x19['steps']}",
    "- checks (A within the B-C floor): " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in checks.items()),
]
head = {"H_DET": "**H-det holds:** in default mode the fused update converges like the unfused one (within Metal's "
                 "run-to-run floor), so the D vs E difference is specific to deterministic mode (consistent with the "
                 "in-launch counter hypothesis above, not proven by it); #29's pin already takes the unfused update "
                 "there. Nothing to change.",
        "H_DEFAULT": "**H-default:** the fused update changes default-mode convergence beyond the floor. Next round: "
                     "localise (CPU 60-step fused vs unfused on the default cone; per-iteration Metal comparison; MuJoCo "
                     "C's iteration counts on the saved G1 states).",
        "INVALID": "**Invalid run:** " + "; ".join(invalid) + "."}[dec]
text = "\n".join(["", "**Result (batch_22, 16-inch, main with MuJoCo Warp pin 8eb51fe):** "
                  + head, ""] + lines + [""])
s = note.read_text()
if "**Result (batch_22" not in s:
    note.write_text(s.rstrip("\n") + "\n" + text)
print(text)
print(dec)
