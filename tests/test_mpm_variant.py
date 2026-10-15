"""Flag-gated implicit-MPM solver variants (scripts/vendor_mpm_variant.py; research_notes/sand_fidelity/PLAN_2026-10-14.md
section 9, P1). CPU only: the variant regenerates exactly from the vendored copy plus its patch, keeps the licence
header, is selected only by ``MPMConfig(solver_variant=...)``, and the default (parity) path never imports it. The
variant's own physics tests are tests/test_mpm_variant_pr4660.py (Newton PR #4660's tests, CPU and Metal)."""
import importlib.util
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import vendor_mpm_variant  # noqa: E402


@pytest.mark.skipif(shutil.which("patch") is None, reason="needs the patch tool")
def test_variant_regenerates_from_vendored_copy_and_patch():
    for name in vendor_mpm_variant.VARIANTS:
        assert vendor_mpm_variant.check(name) == [], f"{name}: committed files differ from a fresh generation"


def test_variant_files_keep_the_licence_header():
    for spec in vendor_mpm_variant.VARIANTS.values():
        for p in sorted(spec["dest"].glob("*.py")):
            lines = p.read_text().splitlines()
            assert lines[0].startswith("# SPDX-FileCopyrightText:") and "Newton Developers" in lines[0], p.name
            assert lines[1] == "# SPDX-License-Identifier: Apache-2.0", p.name
            assert lines[2].startswith("# Modified by MetalSim (Apache-2.0 section 4)"), p.name


def test_variant_changes_only_the_patched_files():
    """Files the patch does not touch are the vendored files byte for byte apart from the notice."""
    vend = ROOT / "metalsim/physics/mpm/implicit_mpm"
    var = ROOT / "metalsim/physics/mpm/implicit_mpm_pr4660"
    patched = {"implicit_mpm_solver_kernels.py", "rasterized_collisions.py", "rheology_solver_kernels.py",
               "solver_implicit_mpm.py"}

    def body(p):
        lines = p.read_text().splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.startswith("# Modified by MetalSim"))
        j = i
        while j < len(lines) and lines[j].startswith("#"):
            j += 1
        return lines[:i] + lines[j:]

    for p in sorted(vend.glob("*.py")):
        same = body(p) == body(var / p.name)
        assert same == (p.name not in patched), p.name


def test_solver_variant_selection():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from metalsim.physics.mpm import solver

    if solver.solver_kind() == "newton":
        pytest.skip("Newton 1.5.2 environment: parity runs Newton's own solver")
    parity = solver.solver_class()
    assert parity.__module__.startswith("metalsim.physics.mpm.implicit_mpm.")
    assert solver.solver_class("parity") is parity
    v = solver.solver_class("pr4660")
    assert v.__module__.startswith("metalsim.physics.mpm.implicit_mpm_pr4660.") and v is not parity
    with pytest.raises(ValueError):
        solver.solver_class("nope")


def test_mpm_config_default_and_validation():
    pytest.importorskip("warp")
    pytest.importorskip("mujoco")
    from metalsim.physics.mpm.coupled import MPMConfig, MPMCoupling

    assert MPMConfig().solver_variant == "parity"
    with pytest.raises(ValueError):
        MPMCoupling(None, MPMConfig(solver_variant="nope"))


def test_parity_path_never_imports_the_variant():
    """newton_compat.install's module list includes a variant only once it is imported (selected)."""
    pytest.importorskip("warp")
    if importlib.util.find_spec("newton") is None:
        pytest.skip("needs newton")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from metalsim.physics.mpm import newton_compat as nc\n"
        "names = [m.__name__ for m in nc.rheology_modules()]\n"
        "assert not any('implicit_mpm_pr4660' in n for n in sys.modules), 'variant imported on the parity path'\n"
        "assert names[-1] == nc.VENDORED_MODULE, names\n"
        "import metalsim.physics.mpm.implicit_mpm_pr4660\n"
        "assert nc.rheology_modules()[-1].__name__ == nc.VARIANT_MODULES[0]\n"
        "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout + r.stderr
