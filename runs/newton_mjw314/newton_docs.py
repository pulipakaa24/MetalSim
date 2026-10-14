"""Newton 1.7 pins after the MuJoCo Warp 3.14 contact fix (batch_11).

    python3 newton_docs.py REPO OLD_NF NEW_NF OLD_HT NEW_HT

Full SHAs. NF = branch metalsim (.venv-newtonfork, scripts/setup_newton.sh), HT = metalsim-hashtable-metal (.venv's
newton extra, pyproject.toml). Exits 1 if an anchor is missing.
"""
import pathlib
import sys

repo, onf, nnf, oht, nht = sys.argv[1:6]
R = pathlib.Path(repo)
missing = []


def edit(path, pairs):
    p = R / path
    s = p.read_text()
    for old, new in pairs:
        if old not in s:
            missing.append(f"{path}: {old[:80]!r}")
            continue
        s = s.replace(old, new)
    p.write_text(s)
    print(f"edited {path}")


edit("pyproject.toml", [
    (oht, nht),
    (f"metalsim-hashtable-metal-{oht[:8]}.diff", f"metalsim-hashtable-metal-{nht[:8]}.diff"),
])
edit("scripts/setup_newton.sh", [
    (f"PIN={onf}                           # metalsim head, 2026-10-12",
     f"PIN={nnf}                           # metalsim head, 2026-10-13 (SolverMuJoCo on MuJoCo Warp 3.14)"),
    (f"metalsim-{onf[:8]}.diff", f"metalsim-{nnf[:8]}.diff"),
])
edit("README.md", [
    (f"[`{onf[:7]}`](https://github.com/pulipakaa24/newton/commit/{onf})",
     f"[`{nnf[:7]}`](https://github.com/pulipakaa24/newton/commit/{nnf})"),
    ("Base: newton-physics/newton `4545802`. Apache-2.0.",
     "Base: newton-physics/newton `4545802`. Apache-2.0. Both branches pass MuJoCo Warp 3.14's contact adhesion to "
     "`contact_force_fn` (the contact part of upstream's MuJoCo Warp 3.14 update `90f809a`, 2026-10-13); before, "
     "`SolverMuJoCo` did not compile against MetalSim's MuJoCo Warp fork."),
    (f"(`{oht[:7]}`) is upstream `4545802`", f"(`{nht[:7]}`) is upstream `4545802`"),
])
edit("metalsim/physics/mpm/MPM.md", [
    (f"`metalsim-hashtable-metal` `{oht[:8]}`", f"`metalsim-hashtable-metal` `{nht[:8]}`"),
    (f"`metalsim` `{onf[:8]}` (`.venv-newtonfork`)", f"`metalsim` `{nnf[:8]}` (`.venv-newtonfork`)"),
    ("(`contact_force_fn` signature).",
     "(`contact_force_fn` signature; fixed 2026-10-13 on both 1.7 branches, `metalsim` "
     f"`{nnf[:8]}` and `metalsim-hashtable-metal` `{nht[:8]}`: the contact part of upstream's `90f809a`)."),
])
p = R / "CHANGELOG.md"
s = p.read_text()
i = s.index("\n## ")
s = s[:i] + (
    "\n## Newton 1.7: SolverMuJoCo on MuJoCo Warp 3.14 (2026-10-13)\n\n"
    f"- 10-13 Newton pins {onf[:8]} -> {nnf[:8]} (`metalsim`, `.venv-newtonfork`) and {oht[:8]} -> {nht[:8]} "
    "(`metalsim-hashtable-metal`, `.venv`'s `newton` extra): `SolverMuJoCo` passes the contact adhesion that MuJoCo "
    "Warp 3.14's `contact_force_fn` takes (upstream Newton's `90f809a`). Before, `SolverMuJoCo`'s kernels did not "
    "compile against MetalSim's MuJoCo Warp fork, and the Newton-engine and Newton-bridge render tests skipped in `.venv`.\n"
) + s[i:]
p.write_text(s)
print("edited CHANGELOG.md")
if missing:
    print("MISSING ANCHORS:\n  " + "\n  ".join(missing))
    sys.exit(1)
print("newton_docs: ok")
