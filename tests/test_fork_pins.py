"""The MuJoCo Warp and Warp fork checkouts are at the commits this tree pins.

A checkout off its pin runs other physics than the docs describe, silently: on 2026-10-31 the shared
``upstream/warp-metalsim`` was found 70 commits behind its pin, and the MuJoCo Warp pin moved to the heightfield-contact
fix (docs/research/hfield_contacts_2026-10-31.md), which changes every heightfield contact.

Pins, from the sources scripts/fork_sync.sh reads: MuJoCo Warp = the commit of README.md's
``pip install ... mujoco_warp@<sha>`` line (docs/GUIDE.md must repeat it); Warp = ``PIN`` in scripts/setup_warp.sh.

Checked when present, skipped otherwise (a wheel or PyPI install has no checkout to compare):
- ``upstream/mujoco_warp``, ``upstream/warp-metalsim`` and (older layout) ``upstream/warp-innate`` of this checkout and,
  from a git worktree, of the main checkout, where scripts/setup_warp.sh and scripts/fork_sync.sh keep them;
- the git checkout the importable ``mujoco_warp`` / ``warp`` package comes from (found with importlib, nothing is
  imported): what actually runs, wherever it lives (``pip install -e git+...`` clones into the venv's ``src/``; on the
  author's Mac the editable installs point into ``upstream/``). A package is taken from a checkout only when its
  ``<package>/__init__.py`` is tracked there at the top level, so a venv inside this repository is not mistaken for one.
A checkout passes when HEAD is the pin and no tracked file is modified. CPU only; needs git.
"""
import importlib.util
import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ("move it with scripts/fork_sync.sh (scripts/fork_sync.sh -n reports what it would do); a checkout made by the "
       "README's `pip install -e git+...` line: run that line again")


def _git(path, *args):
    r = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _mjw_pin(doc="README.md"):
    m = re.search(r"mujoco_warp@([0-9a-f]{40})", (ROOT / doc).read_text())
    return m.group(1) if m else None


def _warp_pin():
    m = re.search(r"^PIN=([0-9a-f]{40})", (ROOT / "scripts" / "setup_warp.sh").read_text(), re.M)
    return m.group(1) if m else None


PINS = {"mujoco_warp": _mjw_pin, "warp": _warp_pin}


def test_pin_sources_agree():
    """README and GUIDE pin the same MuJoCo Warp commit, and each pin's whole-branch fallback diff is the one in
    patches/ and the one the docs name (a pin bump that misses one of them fails here)."""
    mjw, warp = _mjw_pin(), _warp_pin()
    assert mjw and warp, "pin not found (README.md pip line / scripts/setup_warp.sh PIN=)"
    assert _mjw_pin("docs/GUIDE.md") == mjw, "docs/GUIDE.md's pip line pins another MuJoCo Warp commit than README.md"
    diffs = sorted(p.name for p in (ROOT / "patches" / "mujoco_warp").glob("metalsim-*.diff"))
    assert diffs == [f"metalsim-{mjw[:7]}.diff"], f"patches/mujoco_warp whole-branch diffs {diffs}, pin {mjw[:7]}"
    for doc in ("README.md", "docs/GUIDE.md"):
        named = set(re.findall(r"patches/mujoco_warp/(metalsim-[0-9a-f]+\.diff)", (ROOT / doc).read_text()))
        assert named == {diffs[0]}, f"{doc} names {sorted(named)}"
    assert (ROOT / "patches" / "warp" / f"metalsim-{warp[:8]}.diff").is_file(), "no patches/warp diff for the Warp pin"


def _main_checkout():
    common = _git(ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if common and pathlib.Path(common).name == ".git":
        return pathlib.Path(common).parent.resolve()
    return ROOT


def _upstream_checkouts(name):
    dirs = []
    for root in dict.fromkeys((ROOT, _main_checkout())):
        d = root / "upstream" / name
        if (d / ".git").exists():
            dirs.append(d)
    return dirs


def _imported_checkout(pkg):
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.origin:
        return []
    pkg_dir = pathlib.Path(spec.origin).resolve().parent
    top = _git(pkg_dir, "rev-parse", "--show-toplevel")
    if top is None or _git(top, "ls-files", "--error-unmatch", f"{pkg}/__init__.py") is None:
        return []
    if pathlib.Path(top).resolve() != pkg_dir.parent:
        return []
    return [pathlib.Path(top)]


SLOTS = [("upstream/mujoco_warp", "mujoco_warp"), ("upstream/warp-metalsim", "warp"), ("upstream/warp-innate", "warp"),
         ("imported mujoco_warp", "mujoco_warp"), ("imported warp", "warp")]


def _relation(d, head, pin):
    if _git(d, "cat-file", "-e", f"{pin}^{{commit}}") is None:
        return "the pinned commit is not in this checkout (needs a fetch)"
    if _git(d, "merge-base", "--is-ancestor", head, pin) is not None:
        return f"behind the pin by {_git(d, 'rev-list', '--count', f'{head}..{pin}')} commit(s)"
    if _git(d, "merge-base", "--is-ancestor", pin, head) is not None:
        return f"ahead of the pin by {_git(d, 'rev-list', '--count', f'{pin}..{head}')} commit(s)"
    return "diverged from the pin"


@pytest.mark.parametrize("slot,repo", SLOTS, ids=[s for s, _ in SLOTS])
def test_fork_checkout_at_pin(slot, repo):
    if shutil.which("git") is None:
        pytest.skip("needs git")
    pin = PINS[repo]()
    assert pin, f"no {repo} pin found"
    kind, name = slot.split(" ")[0], slot.split("/")[-1].split(" ")[-1]
    dirs = _upstream_checkouts(name) if kind.startswith("upstream") else _imported_checkout(name)
    if not dirs:
        pytest.skip(f"{slot}: no git checkout (absent, or installed from a wheel)")
    bad = []
    for d in dirs:
        head = _git(d, "rev-parse", "HEAD")
        if head != pin:
            bad.append(f"{d}: HEAD {str(head)[:8]}, pin {pin[:8]}: {_relation(d, head, pin)}")
        elif _git(d, "status", "--porcelain", "--untracked-files=no"):
            bad.append(f"{d}: at the pin {pin[:8]}, but tracked files are modified")
        else:
            print(f"{slot}: {d} at the pin {pin[:8]}")
    assert not bad, f"{repo} checkout off its pin:\n" + "\n".join(bad) + f"\n{FIX}"
