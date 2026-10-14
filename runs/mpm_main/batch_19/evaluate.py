"""batch_19 verdict on M1 (research_notes/MPM in main simulator/deterministic_g1_capture.md, sections 3, 7, 8).

    python3 evaluate.py OUT19 OUT18 RC_G1FIX_64 RC_G1FIX_1024 TESTPRE NOTE_MD

TESTPRE: 0 passed, 1 failed on the counter phase mismatch, 2 failed otherwise.

M1 (read-back of a suppressed increment in _solve_done) predicts, with the diagnostic Warp build:
  - pin, captured: trips only in _solve_done, flag 2 or 6 (6 = 2 | 4: the extra phase-1 call has no phase-0 record,
    and the thread's phase-0 / phase-1 call counts then differ), only in steps where some world reached the cap, and
    none in the stack scene's free-fall steps (no constraint rows, no cap);
  - fix (count in a local; unfused update in deterministic mode): no trips with the same cap steps, at 64 and 1,024;
  - uncaptured: the same as captured wherever _solve_done is a deterministic launch (counter consumed); where it is
    not one, no trip (that is coverage, recorded, not a contradiction);
  - test_iteration_cap_repeats fails on the pin (it would otherwise be vacuous) and the stack reaches the cap before
    its step 160.
Prints the per-run lines and the verdict as the last line: M1_CONFIRMED or INCONCLUSIVE.
"""
import json
import pathlib
import sys

o19, o18 = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rc64, rc1024, rc_testpre = (int(x) for x in sys.argv[3:6])
note = pathlib.Path(sys.argv[6])
M1_FLAGS = {"2", "6"}
SD = "_solve_done__locals__kernel"


def load(d, tag):
    p = d / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


lines, fails = [], []


def line(tag, r):
    if r is None:
        lines.append(f"- `{tag}`: no result")
        return
    lines.append(
        f"- `{tag}` ({r['scene']}, {r['worlds']} worlds, capture {r['capture']}, cap {r['iteration_cap']}): steps with "
        f"cap worlds {len(r['steps_with_cap_worlds'])}/{r['steps']} (first {r.get('first_cap_step')}), steps with trips "
        f"{len(r['steps_with_hits'])}, trips only in cap steps {r['hits_only_in_cap_steps']}, kernels {r['kernels_hit']}, "
        f"flag values {r['flag_values']}, fused update {r.get('fused_update')}, _solve_done deterministic launches "
        f"{r.get('det_launches', {}).get(SD, 0)}")


def expect_trips(tag, r):
    """M1's pre-fix prediction for a run that reached the cap with _solve_done as a deterministic launch."""
    if r is None:
        fails.append(f"{tag}: no result")
        return
    if not r["steps_with_cap_worlds"]:
        fails.append(f"{tag}: no world reached the cap (vacuous)")
        return
    if not r["steps_with_hits"]:
        fails.append(f"{tag}: cap reached, _solve_done launched deterministically, but no trip")
    if set(r["kernels_hit"]) - {SD}:
        fails.append(f"{tag}: trips outside _solve_done {r['kernels_hit']}")
    if set(r["flag_values"]) - M1_FLAGS:
        fails.append(f"{tag}: flag values {r['flag_values']} outside 2 / 6")
    if not r["hits_only_in_cap_steps"]:
        fails.append(f"{tag}: trips in steps without cap worlds")


def expect_clean(tag, r):
    if r is None:
        fails.append(f"{tag}: no result")
        return
    if not r["steps_with_cap_worlds"]:
        fails.append(f"{tag}: no world reached the cap (vacuous)")
    if r["steps_with_hits"]:
        fails.append(f"{tag}: trips with the fix {r['kernels_hit']} {r['flag_values']}")
    if r.get("fused_update"):
        fails.append(f"{tag}: the fused update ran in deterministic mode")


runs = {t: load(o19, t) for t in ("pre_stack_capt", "fix_stack_capt", "pre_stack_eager", "fix_stack_eager",
                                  "pre_g1_1024u", "fix_g1_64", "fix_g1_1024")}
for t, r in runs.items():
    line(t, r)
for t in ("pre_g1_64", "fix_g1_64"):
    line(f"batch_18 {t} (fused update)", load(o18, t))

# stack, captured: both kinds of steps present, so "only in cap steps" discriminates (G1 reaches the cap every step)
r = runs["pre_stack_capt"]
expect_trips("pre_stack_capt", r)
if r is not None:
    if r.get("steps_without_cap_worlds", 0) == 0:
        fails.append("pre_stack_capt: no step without cap worlds (cannot test 'only in cap steps')")
    if r.get("first_cap_step") is None or r["first_cap_step"] >= 160:
        fails.append(f"pre_stack_capt: first cap step {r.get('first_cap_step')}: test_iteration_cap_repeats (160 steps) "
                     "would be vacuous")
expect_clean("fix_stack_capt", runs["fix_stack_capt"])

# stack, uncaptured: M1 makes no capture distinction where _solve_done is a deterministic launch
r = runs["pre_stack_eager"]
eager_note = "no result"
if r is not None:
    if r.get("det_launches", {}).get(SD, 0):
        expect_trips("pre_stack_eager", r)
        eager_note = "_solve_done is a deterministic launch uncaptured; M1's trips appear there too" \
            if r["steps_with_hits"] else "_solve_done is a deterministic launch uncaptured but did not trip"
    else:
        eager_note = ("_solve_done is not a deterministic launch uncaptured (its counter is consumed only with the "
                      "captured early exit), so uncaptured runs cannot trip: explains batch_13's 'capture-only'")
        if r["steps_with_hits"]:
            fails.append(f"pre_stack_eager: trips {r['kernels_hit']} without _solve_done deterministic")
r = runs["fix_stack_eager"]
if r is None or r["steps_with_hits"]:
    fails.append(f"fix_stack_eager: {'no result' if r is None else r['kernels_hit']}")

# G1 1,024: the pin with the unfused update (the pin's fused plan cannot be built there), then the fix
expect_trips("pre_g1_1024u", runs["pre_g1_1024u"])
expect_clean("fix_g1_64", runs["fix_g1_64"])
expect_clean("fix_g1_1024", runs["fix_g1_1024"])
if rc64 or rc1024:
    fails.append(f"det_cost_g1 with the fix (check_overflow): rc 64 {rc64}, 1,024 {rc1024}")
if rc_testpre != 1:
    fails.append("test_iteration_cap_repeats on the pin: " + ("passed: it does not test the fix" if rc_testpre == 0
                 else "failed without a counter phase mismatch"))

verdict = "M1_CONFIRMED" if not fails else "INCONCLUSIVE"
lines.append(f"- G1 deterministic with the fix, captured (det_cost_g1.py, check_overflow): exit code 64 worlds {rc64}, "
             f"1,024 worlds {rc1024}")
lines.append(f"- test_iteration_cap_repeats on the current pin: {({0: 'passed', 1: 'failed on the counter phase mismatch'}).get(rc_testpre, 'failed otherwise')} "
             "(M1 predicts the phase-mismatch failure)")
lines.append(f"- Uncaptured: {eager_note}")
head = ("M1 confirmed: every prediction held." if not fails else
        "Not confirmed: " + "; ".join(fails) + ".")
text = "\n".join(["", "## 8. Results (batch_19, 16-inch)", "", head, ""] + lines + [""])
s = note.read_text()
if "## 8. Results (batch_19" not in s:
    note.write_text(s.rstrip("\n") + "\n" + text)
print(text)
print(verdict)
