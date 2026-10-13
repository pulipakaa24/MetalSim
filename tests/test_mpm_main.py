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


# ---------------------------------------------------------------------------------------------------------------------
# GPU: the vendored solver and the BatchSim coupling (Metal, main environment)
# ---------------------------------------------------------------------------------------------------------------------

import importlib.util  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

RUNS = ROOT / "runs" / "mpm_main"


def _metal():
    wp = pytest.importorskip("warp")
    pytest.importorskip("newton")
    pytest.importorskip("mujoco_warp")
    if not wp.is_metal_available():
        pytest.skip("needs Metal")
    return wp


def _box_on_sand():
    spec = importlib.util.spec_from_file_location("box_on_sand", ROOT / "metalsim/physics/mpm/probes/box_on_sand.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vendored_column_bitwise_newton152():
    """The vendored solver in this environment reproduces Newton 1.5.2's own solver (.venv-newton152, recorded in
    runs/mpm_main/vendored_bitwise) bit for bit: every frame of a column collapse, positions and velocities."""
    wp = _metal()
    import hashlib

    from metalsim.physics.mpm import newton_compat, scenes

    ref = json.loads((RUNS / "vendored_bitwise" / "n152_newton_column.json").read_text())
    newton_compat.install(accumulate="float32", loop="cuda", rigid_deterministic=None, device_args=None)
    newton_compat.drain_log()
    frames = []
    with wp.ScopedDevice("metal:0"):
        sc = scenes.make_scene("column", aspect=1.0, seconds=0.25)
        for _ in range(sc.frames):
            sc.step()
            q, v = sc.state.particle_q.numpy(), sc.state.particle_qd.numpy()
            frames.append(hashlib.sha256(q.tobytes() + v.tobytes()).hexdigest()[:16])
    assert frames == ref["frames"]
    assert newton_compat.drain_log()["iterations"].tolist() == ref["iterations"]


@pytest.fixture(scope="module")
def box_runs():
    """Two independent box-on-sand runs (2 worlds, 60 steps of 1/120 s) of the coupled main simulator."""
    _metal()
    mod = _box_on_sand()

    class A:
        worlds, steps, voxel, cells, capture, mass, mode, mpm_substeps, particle_stride = 2, 60, 0.02, "auto", 1, \
            "effective", "staggered", 1, 1000

    return mod, [mod.run_metalsim(A) for _ in range(2)]


def test_box_sinks_as_isaac_coupling(box_runs):
    """A box dropped on sand follows Isaac Lab's coupling path (Newton 1.5.2 SolverCoupledProxy with SolverMuJoCo,
    recorded with box_on_sand.py): box height within 20 um and the sand's vertical force on the box within 1 N at
    every step of the impact (peaks ~120 N) and the settling; late in the run the sand carries the box's weight."""
    _, (r, _) = box_runs
    ref = json.loads((RUNS / "box_on_sand" / "box_w2b_summary.json").read_text())
    n = r["pos"].shape[0]
    zr = np.asarray(ref["newton_pos"])[:n, :, 2]
    fr = np.asarray(ref["newton_wrench"])[:n, :, 2]
    assert np.abs(r["pos"][:, :, 2] - zr).max() < 2e-5
    assert np.abs(r["wrench"][:, :, 2] - fr).max() < 1.0
    assert np.allclose(r["wrench"][n - 20:, :, 2].mean(0), 2.0 * 9.81, rtol=0.15)


def test_coupled_repeatable(box_runs):
    """Two runs of the coupled scene (captured steps) are bitwise identical: box state and every particle."""
    _, (a, b) = box_runs
    assert np.array_equal(a["pos"], b["pos"]) and np.array_equal(a["qvel"], b["qvel"])
    assert (a["phash"] == b["phash"]).all()


def test_free_fall_momentum():
    """Material in free fall above the floor gains the momentum gravity gives it: mean velocity -g t, nothing
    sideways (no collider reached)."""
    wp = _metal()
    import mujoco

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling

    m = mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.005"/><worldbody>'
                                       '<geom type="plane" size="0 0 1"/></worldbody></mujoco>')
    sim = BatchSim(m, 2, options=BatchSimOptions(substeps=2))
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=0.05, max_iterations=20))
    mpm.add_emitter(GridEmitter(lower=(-0.1, -0.1, 1.0), upper=(0.1, 0.1, 1.1)))
    mpm.finalize()
    steps = 10
    for _ in range(steps):
        mpm.step()
    v = mpm.particle_qd.numpy().reshape(2, -1, 3)
    t = steps * sim.dt
    assert np.abs(v[..., :2]).max() < 1e-5
    assert np.allclose(v[..., 2].mean(1), -9.81 * t, rtol=1e-3)


def test_main_sim_settings_untouched():
    """Building and stepping an MPM scene changes no process-wide Warp setting MuJoCo Warp compiles with (argument
    space, determinism, fast math, ...), so rigid kernels are those of a scene without MPM."""
    wp = _metal()
    import mujoco

    from metalsim.physics.batch import BatchSim
    from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling

    def snap():
        return {k: repr(getattr(wp.config, k)) for k in dir(wp.config) if not k.startswith("_")}

    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="0 0 1"/>'
                                       '<body pos="0 0 .3"><freejoint/><geom type="sphere" size=".05"/></body>'
                                       '</worldbody></mujoco>')
    sim = BatchSim(m, 1)
    before = snap()
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=0.05, max_iterations=10))
    mpm.add_emitter(GridEmitter(lower=(-0.1, -0.1, 0.0), upper=(0.1, 0.1, 0.05)))
    mpm.finalize()
    mpm.step()
    mpm.step()
    assert snap() == before
