"""Pin move for the deterministic BatchSim (batch_08): Warp and MuJoCo Warp pins in scripts and docs, the bullets and
rows that describe the deterministic option, the research note's validation results.

    python3 pin_docs.py REPO OLD_WARP NEW_WARP OLD_MJW NEW_MJW TRIPWIRE_CODE_SHA CCD_CODE_SHA

SHAs in full (40 hex). Pins: every occurrence in scripts/setup_warp.sh; elsewhere only on lines that state the current
pin (historical mentions stay). Prints every edit; exits 1 if an expected anchor is missing.
"""
import pathlib
import sys

repo, ow, nw, om, nm, trip, ccd = sys.argv[1:8]
R = pathlib.Path(repo)
W8, NW8, M7, NM7 = ow[:8], nw[:8], om[:7], nm[:7]
TRIP8, CCD7 = trip[:8], ccd[:7]
missing = []


def edit(path, fn):
    p = R / path
    s = p.read_text()
    t = fn(s)
    if t != s:
        p.write_text(t)
        print(f"edited {path}")
    else:
        print(f"unchanged {path}")


def lines_sub(s, markers, pairs, path):
    out, n = [], 0
    for line in s.split("\n"):
        if any(m in line for m in markers):
            new = line
            for a, b in pairs:
                new = new.replace(a, b)
            n += new != line
            line = new
        out.append(line)
    print(f"  {path}: {n} pin line(s)")
    return "\n".join(out)


def must_replace(s, old, new, path):
    if old not in s:
        missing.append(f"{path}: {old[:70]!r}")
        return s
    return s.replace(old, new, 1)


WPAIRS = [(ow, nw), (W8, NW8)]
MPAIRS = [(om, nm), (M7, NM7)]

# -- scripts/setup_warp.sh: every occurrence
def setup_warp(s):
    import re

    s = s.replace(ow, nw).replace(W8, NW8)
    t, n = re.subn(r"^(PIN=[0-9a-f]{40}\s+)# metalsim head, .*$",
                   r"\1# metalsim head, 2026-10-13 (deterministic counter phase-mismatch tripwire)", s, count=1, flags=re.M)
    if n != 1:
        missing.append("scripts/setup_warp.sh: PIN comment")
    return t


edit("scripts/setup_warp.sh", setup_warp)


# -- README
def readme(s):
    s = lines_sub(s, ("pinned by",), WPAIRS, "README.md (Warp)")
    s = lines_sub(s, ("the commit the install pins", "mujoco_warp@", "patches/mujoco_warp/metalsim-"), MPAIRS,
                  "README.md (MuJoCo Warp)")
    warp_bullet = (
        f"- Deterministic counters checked (2026-10-13, `{TRIP8}`; `research_notes/MPM in main simulator/"
        "deterministic_ccd_divergence.md`): a kernel whose slot-counter calls differ between the counting and the "
        "execution pass (the counting pass suppresses stores) raises \"Deterministic counter phase mismatch\" instead "
        "of silently handing slots to the wrong calls.\n\n"
        "Branch [`metalsim-flex`](https://github.com/pulipakaa24/warp/tree/metalsim-flex)")
    s = must_replace(s, "\nBranch [`metalsim-flex`](https://github.com/pulipakaa24/warp/tree/metalsim-flex)",
                     "\n" + warp_bullet, "README.md")
    mjw_bullet = (
        f"- Deterministic mode (Warp `RUN_TO_RUN`): non-heightfield CCD runs as a counter-free solve kernel and a "
        f"contact-write kernel (`{CCD7}`); the single kernel picked its EPA scratch row with a slot counter, so under "
        "Warp's two-pass counters a box-box contact could land in another world's slot. Default mode unchanged.\n\n"
        "Branch [`metalsim-flex`](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim-flex)")
    s = must_replace(s, "\nBranch [`metalsim-flex`](https://github.com/pulipakaa24/mujoco_warp/tree/metalsim-flex)",
                     "\n" + mjw_bullet, "README.md")
    return s


edit("README.md", readme)


# -- GUIDE
def guide(s):
    s = lines_sub(s, ("pinned commit `" + W8, "patches/warp/metalsim-" + W8), WPAIRS, "docs/GUIDE.md (Warp)")
    s = lines_sub(s, ("mujoco_warp@", "pinned commit `" + M7, "patches/mujoco_warp/metalsim-" + M7), MPAIRS,
                  "docs/GUIDE.md (MuJoCo Warp)")
    s = must_replace(
        s, "experimental `BatchSimOptions(deterministic=True)` (this branch; one mode per process; needs the MuJoCo Warp "
           "fork branch `metalsim-det-ccd` for convex (GJK/EPA) contacts such as box-box; no heightfield GJK/EPA "
           "contacts).",
        "opt-in `BatchSimOptions(deterministic=True)` (Warp's RUN_TO_RUN mode for MuJoCo Warp's kernels; one mode per "
        "process, chosen before MuJoCo Warp's first launch; bitwise run to run and across processes, e.g. two stacked "
        "boxes at 256 worlds; heightfield contacts that take GJK/EPA are refused).", "docs/GUIDE.md")
    return s


edit("docs/GUIDE.md", guide)

# -- MPM.md: current-pin lines and the repeatability paragraph
def mpm(s):
    s = lines_sub(s, ("shared Warp fork @" + W8, "pinned `" + W8), WPAIRS, "MPM.md (Warp)")
    s = must_replace(
        s, "An opt-in deterministic `BatchSim` (Warp's RUN_TO_RUN mode for MuJoCo Warp's kernels)\n"
           "  is being developed separately on branch `batchsim-deterministic` (open: a divergence in MuJoCo Warp's box-box CCD\n"
           "  kernel under the two-phase deterministic counters, 2 of 256 worlds from step 139).",
        "With `BatchSimOptions(deterministic=True)` (Warp's RUN_TO_RUN mode for MuJoCo Warp's\n"
        "  kernels, 2026-10-13) rigid contacts repeat too: two stacked boxes at 256 worlds are bitwise equal in one process\n"
        "  and across processes (`research_notes/MPM in main simulator/deterministic_ccd_divergence.md`).", "MPM.md")
    return s


edit("metalsim/physics/mpm/MPM.md", mpm)

# -- STATUS: the WS2 row's pins (its Warp pin was already stale)
def status(s):
    out = []
    for line in s.split("\n"):
        if line.startswith("| WS2 GPU physics |"):
            line = line.replace("pinned `95f6f121`", f"pinned `{NW8}`").replace(f"pinned `{W8}`", f"pinned `{NW8}`")
            line = line.replace(f"pinned `{M7}`", f"pinned `{NM7}`")
            line = line.replace("capacity overflow flags latched per substep (2026-10-13).",
                                "capacity overflow flags latched per substep (2026-10-13); opt-in bitwise-repeatable "
                                "`BatchSimOptions(deterministic=True)` (2026-10-13).")
        out.append(line)
    return "\n".join(out)


edit("docs/STATUS.md", status)


# -- GAPS: the determinism rows
def gaps(s):
    out = []
    for line in s.split("\n"):
        if line.startswith("| Determinism on Metal |"):
            new = line.replace(
                "MetalSim's own `BatchSim` (MuJoCo Warp) has no deterministic option and stays non-repeatable run to run",
                "MetalSim's own `BatchSim` (MuJoCo Warp) is non-repeatable by default; **since 2026-10-13** "
                "`BatchSimOptions(deterministic=True)` repeats bitwise (Warp RUN_TO_RUN for MuJoCo Warp's kernels; "
                "research_notes/MPM in main simulator/deterministic_ccd_divergence.md)")
            if new == line:
                missing.append("docs/GAPS.md: Determinism on Metal row text")
            line = new
        if line.startswith("| MuJoCo Warp BatchSim is not run-to-run repeatable"):
            line = ("| MuJoCo Warp BatchSim is not run-to-run repeatable on Metal by default (float atomics in arrival "
                    "order: a box resting on the floor diverges at step 18 at 1,024 worlds; the panda between two plain "
                    "runs) | **resolved 2026-10-13 (opt-in)**: `BatchSimOptions(deterministic=True)` runs MuJoCo Warp's "
                    "kernels in Warp's RUN_TO_RUN mode. The stacked-boxes divergence (2 of 256 worlds from step 139) was "
                    "MuJoCo Warp's CCD kernel choosing its EPA scratch with a slot counter: under Warp's two-pass "
                    "counters a box-box contact landed in another world's slot. MuJoCo Warp pin "
                    f"`{NM7}` splits CCD into a counter-free solve and a write kernel in deterministic mode; Warp pin "
                    f"`{NW8}` raises on any such counter phase mismatch. Two stacked boxes at 256 worlds repeat in one "
                    "process and across processes (cold / warm kernel cache), every world keeps its contact "
                    "(runs/mpm_main/batch_06; research_notes/MPM in main simulator/deterministic_ccd_divergence.md) | "
                    "the default mode stays non-repeatable (cost of the deterministic mode not measured); heightfield "
                    "contacts that take GJK/EPA (primitives, meshes over 256 vertices) are refused in deterministic "
                    "mode; Newton's `SolverMuJoCo(deterministic=True)` with MuJoCo contacts gets the CCD fix once its "
                    "MuJoCo Warp includes it |")
        out.append(line)
    return "\n".join(out)


edit("docs/GAPS.md", gaps)


# -- CHANGELOG: new section on top
def changelog(s):
    anchor = "\n## "
    i = s.index(anchor)
    sec = (
        "\n## Physics: deterministic BatchSim (2026-10-13, `research_notes/MPM in main simulator/"
        "deterministic_ccd_divergence.md`)\n\n"
        "- 10-13 `BatchSimOptions(deterministic=True)`: MuJoCo Warp's kernels in Warp's RUN_TO_RUN mode (set before "
        "MuJoCo Warp's first launch, one mode per process, record bound for `smooth`, post-capture check of every "
        "replayed module including the `module=\"unique\"` ones, deterministic overflow and phase-mismatch flags in "
        "`check_overflow`). Two stacked boxes at 256 worlds repeat bitwise in one process and across processes.\n"
        f"- 10-13 MuJoCo Warp pin {M7} -> {NM7}: deterministic-mode CCD as a counter-free solve kernel plus a "
        "contact-write kernel (the single kernel's EPA scratch row came from a slot counter, so its counting pass ran "
        "EPA on stale scratch and a contact could land in another world's slot). Warp pin "
        f"{W8} -> {NW8}: a slot-counter call that differs between the counting and the execution pass raises "
        "(\"Deterministic counter phase mismatch\").\n")
    return s[:i] + "\n" + sec.rstrip("\n") + "\n" + s[i:]


edit("CHANGELOG.md", changelog)


# -- research note: validation results
def note(s):
    old_start = s.index("## Validation")
    return s[:old_start] + (
        "## Validation (batch_06, 16-inch, runs/mpm_main/batch_06)\n\n"
        "With both fork branches first on `PYTHONPATH`: Warp's Metal deterministic tests 15 OK (the tripwire tests "
        "included); stacked boxes at 4 worlds take the split kernels (`ccd_det_kernels_builder` modules, RUN_TO_RUN); "
        "at 256 worlds and 200 steps, two runs in one process and three processes (two cold kernel caches, one warm) "
        "are bitwise identical; the contact audit gives every world the same box-box contact count at every step "
        "130-160 and no state difference; tests/test_batch_deterministic.py 7 passed. Tripwire A/B: the single CCD "
        "kernel (`MJW_CCD_DET_SPLIT=0`) at 256 worlds now stops with \"Warp deterministic-mode counter phase "
        "mismatch\" from `BatchSim.check_overflow` instead of diverging silently.\n\n"
        f"Pinned on 2026-10-13 (batch_08): MuJoCo Warp `{NM7}` (code `{CCD7}`), Warp `{NW8}` (code `{TRIP8}`, rebased "
        "onto the coupled-speed pin `31cf79d7`). Newton's `SolverMuJoCo(deterministic=True)` with MuJoCo contacts takes "
        "the same CCD path and gets the fix once its MuJoCo Warp includes it.\n")


edit("research_notes/MPM in main simulator/deterministic_ccd_divergence.md", note)

if missing:
    print("MISSING ANCHORS:\n  " + "\n  ".join(missing))
    sys.exit(1)
print("pin_docs: ok")
