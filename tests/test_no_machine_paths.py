"""No machine-specific absolute paths in tracked files: a clone must run on any Mac, under any user name.

Fails on a home directory (Users or home followed by a user name, as an absolute path) or a macOS temporary directory
(private/tmp, private/var/folders, absolute) in any tracked text file, except where paths are kept as history: run logs and records
(``runs/``, ``*.log``), research notes and reports (``research_notes/``, ``reports/``, ``docs/research/``). Code, tests,
setup scripts and the user-facing docs resolve paths from their own location or take an environment variable instead.
CPU only; needs git.
"""
import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# built from pieces so this file does not match itself
PATTERN = re.compile("/" + "(?:Users|home)/[A-Za-z0-9._-]+/|/" + "private/(?:tmp|var/folders)/")
HISTORY = ("runs/", "research_notes/", "reports/", "docs/research/")


def _tracked():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def test_no_machine_specific_paths_in_tracked_files():
    hits = []
    for rel in _tracked():
        if rel.startswith(HISTORY) or rel.endswith(".log"):
            continue
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:   # binary
            continue
        for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{rel}:{n}: {line.strip()[:160]}")
    assert not hits, ("machine-specific absolute paths (resolve from the file's location or use an environment "
                      "variable):\n" + "\n".join(hits))
