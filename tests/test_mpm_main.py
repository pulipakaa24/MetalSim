"""MPM in MetalSim's main simulator (``.venv``): the vendored Newton 1.5.2-fork solver and its coupling to BatchSim.

Design: ``research_notes/MPM in main simulator/NOTE.md``; usage: ``metalsim/physics/mpm/MPM.md`` section 11.
"""
import os
import pathlib
import re
import subprocess
import sys
import urllib.error

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import vendor_mpm152  # noqa: E402


def _pin_of_setup_script() -> str:
    m = re.search(r"^PIN=([0-9a-f]{40})", (ROOT / "scripts" / "setup_newton152.sh").read_text(), re.M)
    assert m, "scripts/setup_newton152.sh has no PIN line"
    return m.group(1)


def test_vendored_pin_is_the_setup_pin():
    """The vendored solver comes from the same Newton fork commit that .venv-newton152 installs."""
    assert vendor_mpm152.PIN == _pin_of_setup_script()
    assert vendor_mpm152.PIN in (vendor_mpm152.DEST / "SOURCE.txt").read_text()


def _pinned_sources():
    """The pinned commit's solver files: the local fork checkout if it is at the pin, else GitHub's archive."""
    # setup_newton152.sh's checkout (METALSIM_NEWTON152_DIR, default upstream/newton-metalsim-1.5.2 of the clone; a
    # worktree of the clone looks in the enclosing checkouts too)
    env = os.environ.get("METALSIM_NEWTON152_DIR")
    cands = ([pathlib.Path(env)] if env else []) + [d / "upstream" / "newton-metalsim-1.5.2" for d in (ROOT, *ROOT.parents)]
    for cand in cands:
        if (cand / ".git").exists():
            try:
                head = subprocess.run(["git", "-C", str(cand), "rev-parse", "HEAD"], capture_output=True, text=True,
                                      check=True).stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
            dirty = subprocess.run(["git", "-C", str(cand), "status", "--porcelain", "--", vendor_mpm152.SRC_DIR],
                                   capture_output=True, text=True).stdout.strip()
            if head == vendor_mpm152.PIN and not dirty:
                return vendor_mpm152.sources(cand)
    try:
        return vendor_mpm152.sources(None)
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"pinned Newton fork commit unavailable (no checkout at the pin, no network): {e}")


def test_vendored_copy_regenerates_from_the_pin():
    """Regenerating metalsim/physics/mpm/implicit_mpm from the pinned fork commit gives the committed files exactly
    (no hand edits, no drift after a pin move without re-vendoring)."""
    files = vendor_mpm152.generate(_pinned_sources())
    dest = vendor_mpm152.DEST
    committed = {p.name for p in dest.iterdir() if p.is_file()}
    assert committed == set(files), f"file set differs: {sorted(committed ^ set(files))}"
    for name, text in files.items():
        assert (dest / name).read_text() == text, f"{name} differs from the pinned fork commit's"


def test_vendored_files_keep_the_licence_header():
    """Every vendored source starts with Newton's SPDX copyright and Apache-2.0 lines, then the modification notice."""
    for p in sorted(vendor_mpm152.DEST.glob("*.py")):
        lines = p.read_text().splitlines()
        assert lines[0].startswith("# SPDX-FileCopyrightText:") and "Newton Developers" in lines[0], p.name
        assert lines[1] == "# SPDX-License-Identifier: Apache-2.0", p.name
        assert lines[2].startswith("# Modified by MetalSim (Apache-2.0 section 4)"), p.name
