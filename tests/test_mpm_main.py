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


def _probe_result(out):
    """The JSON a probe prints as its last stdout line; on anything else, fail with the probe's own output (stdout and
    stderr tails, untruncated) instead of a bare JSONDecodeError."""
    lines = out.stdout.strip().splitlines()
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        pytest.fail(f"probe exit code {out.returncode}, last stdout line not JSON\n--- stdout tail\n{out.stdout[-2000:]}"
                    f"\n--- stderr tail\n{out.stderr[-4000:]}", pytrace=False)


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


def _box_args(solver, steps=60):
    class A:
        worlds, voxel, cells, capture, mass, mode, mpm_substeps, particle_stride = 2, 0.02, "auto", 1, "effective", \
            "staggered", 1, 1000
    A.solver, A.steps = solver, steps
    return A


@pytest.fixture(scope="module")
def box_runs():
    """Box-on-sand runs of the coupled main simulator (2 worlds, 60 steps of 1/120 s): two with the default
    (Gauss-Seidel) solver, one with Jacobi (Isaac's Franka-Pour solver), all captured."""
    _metal()
    mod = _box_on_sand()
    return {"gs": [mod.run_metalsim(_box_args("auto")) for _ in range(2)],
            "jacobi": mod.run_metalsim(_box_args("jacobi"))}


def test_box_sinks_as_isaac_coupling(box_runs):
    """A box dropped on sand follows Isaac Lab's coupling path (Newton 1.5.2 SolverCoupledProxy: SolverMuJoCo +
    SolverImplicitMPM, staggered proxy with MuJoCo's effective mass; recorded by box_on_sand.py in .venv-newton152,
    runs/mpm_main/box_on_sand/box_jacobi_newton.json). Until the box reaches the sand every particle is bitwise
    Newton's; afterwards the rigid stacks differ (MuJoCo Warp 3.14 vs 3.11) by rounding only: box height within
    20 um and the sand's vertical force within 1 N at every step (impact peaks ~120 N); the sand ends up carrying
    the box's weight."""
    r = box_runs["jacobi"]
    ref = json.loads((RUNS / "box_on_sand" / "box_jacobi_newton.json").read_text())
    n = r["pos"].shape[0]
    fr = np.asarray(ref["wrench"])[:n]
    contact = int(np.argmax(np.abs(fr[:, :, 2]).max(1) > 0))
    assert contact > 5
    assert r["phash"][:contact].tolist() == ref["phash"][:contact]
    assert np.abs(r["pos"][:, :, 2] - np.asarray(ref["pos"])[:n, :, 2]).max() < 2e-5
    assert np.abs(r["wrench"][:, :, 2] - fr[:, :, 2]).max() < 1.0
    assert np.allclose(r["wrench"][n - 20:, :, 2].mean(0), 2.0 * 9.81, rtol=0.15)


def test_coupled_repeatable(box_runs):
    """Two runs of the coupled scene with Gauss-Seidel and separate worlds (captured steps) are bitwise identical:
    box state, sand wrench and every particle (needs the fork's environment-first coloring fix)."""
    a, b = box_runs["gs"]
    assert np.array_equal(a["pos"], b["pos"]) and np.array_equal(a["qvel"], b["qvel"])
    assert np.array_equal(a["wrench"], b["wrench"])
    assert (a["phash"] == b["phash"]).all()


def test_gs_coloring_valid_separate_worlds():
    """The Gauss-Seidel coloring with separate worlds on a sparse grid is a valid coloring: within a color no two
    strain nodes write the same velocity node, and every strain node is solved once per sweep."""
    _metal()
    out = subprocess.run([sys.executable, str(ROOT / "metalsim/physics/mpm/probes/gs_coloring_check.py"),
                          "--device", "metal:0", "--worlds", "3", "--cells", "auto"], capture_output=True, text=True)
    assert out.stdout.strip().endswith("VALID") and "INVALID" not in out.stdout, out.stdout + out.stderr[-2000:]


@pytest.mark.parametrize("worlds", [1, 64])
def test_repeatable_at_scale(worlds):
    """The coupled scene (box on a small bed, carried by the sand, no MuJoCo Warp contacts) repeats bitwise run to run
    (1,024 worlds: runs/mpm_main/scale/). MuJoCo Warp's own contact solve is not run-to-run repeatable on Metal
    (docs/GAPS.md), so scenes whose bodies touch MuJoCo geoms repeat only while they do not."""
    _metal()
    out = subprocess.run([sys.executable, str(ROOT / "metalsim/physics/mpm/probes/scale_coupled.py"), "--worlds",
                          str(worlds), "--steps", "40", "--runs", "2"], capture_output=True, text=True)
    res = _probe_result(out)
    assert res["repeatable"] and res["finite"], res


def test_render_sand():
    """The material renders from the solver's particle buffer (no host copy) in tier 0: grain pixels in the
    segmentation of every world."""
    _metal()
    import mujoco

    from metalsim.physics.batch import BatchSim
    from metalsim.physics.mpm.coupled import SAND, GridEmitter, MPMConfig, MPMCoupling
    from metalsim.render.tier0 import Tier0Renderer

    xml = """<mujoco><worldbody><light directional="true" pos="0 0 3" dir="0.2 0.3 -1"/>
    <camera name="cam" pos="0.4 -0.4 0.35" xyaxes="0.707 0.707 0 -0.4 0.4 0.82" fovy="45"/>
    <geom type="plane" size="1 1 .1" rgba=".3 .3 .3 1"/></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    sim = BatchSim(m, 2)
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=0.02, max_iterations=20))
    mpm.add_emitter(GridEmitter(lower=(-0.1, -0.1, 0.0), upper=(0.1, 0.1, 0.06), material=SAND))
    mpm.finalize()
    for _ in range(3):
        mpm.step()
    sim.synchronize()
    r0 = Tier0Renderer(m, 2, width=96, height=96, camera="cam", outputs=("rgb", "seg"))
    mpm.attach_renderer(r0, grains_per_particle=2)
    seg = r0.render_host([sim.get_world(w) for w in range(2)])["seg"]
    grains = seg == r0.G + 1
    assert grains[0].sum() > 500 and grains[1].sum() > 500


def _pile(friction, seconds=1.0):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pile.json")
        out = subprocess.run([sys.executable, str(ROOT / "metalsim/physics/mpm/probes/pile_repose.py"), "--engine",
                              "metalsim", "--friction", str(friction), "--seconds", str(seconds), "--json", path],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-3000:]
        return json.loads(open(path).read()), out


def test_pile_collapse_matches_newton():
    """A sand column collapsing on the MuJoCo floor of a BatchSim scene ends bitwise where the same column ends in
    Newton 1.5.2 alone on a Newton plane (runs/mpm_main/repose/collapse_newton_mu0.68.json, recorded in
    .venv-newton152): the main simulator's static colliders are Newton's."""
    _metal()
    ref = json.loads((RUNS / "repose" / "collapse_newton_mu0.68.json").read_text())
    res, out = _pile(0.68, ref["seconds"])
    assert res["hashes"] == ref["hashes"], out.stderr[-2000:]


def test_pile_collapse_responds_to_friction():
    """The collapse deposit of a higher-friction material is taller and runs out less (friction 0.4 vs 0.9)."""
    _metal()
    lo, _ = _pile(0.4)
    hi, _ = _pile(0.9)
    assert hi["deposit"]["h_max"] > 1.3 * lo["deposit"]["h_max"], (lo["deposit"], hi["deposit"])
    assert hi["deposit"]["runout"] < 0.8 * lo["deposit"]["runout"], (lo["deposit"], hi["deposit"])


def test_resting_box_carried_by_sand():
    """A 2 kg box placed at rest on the bed (bearing capacity 5-10x its 2 kPa) is carried by the sand with the default
    lagged exchange and a converged solve: settlement within about half a voxel and no lasting creep. (With Isaac's
    staggered exchange it sinks at g dt per step through the bed: research_notes/MPM in main simulator/
    bearing_and_sinking.md.)"""
    _metal()
    spec = importlib.util.spec_from_file_location("bearing", ROOT / "metalsim/physics/mpm/probes/bearing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.run(mode="lagged", rest=True, iterations=250, tolerance=1e-6, seconds=1.5)
    assert r["final_penetration_m"] < 0.015, r["final_penetration_m"]
    assert abs(r["creep_m_per_s"]) < 0.002, r["creep_m_per_s"]
    assert abs(r["force_over_weight_tail"] - 1.0) < 0.1, r["force_over_weight_tail"]


def _free_fall(air_drag):
    import mujoco

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling

    m = mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.005"/><worldbody>'
                                       '<geom type="plane" size="0 0 1"/></worldbody></mujoco>')
    sim = BatchSim(m, 2, options=BatchSimOptions(substeps=2))
    mpm = MPMCoupling(sim, MPMConfig(voxel_size=0.05, max_iterations=20, air_drag=air_drag))
    mpm.add_emitter(GridEmitter(lower=(-0.1, -0.1, 1.0), upper=(0.1, 0.1, 1.1)))
    mpm.finalize()
    steps = 10
    for _ in range(steps):
        mpm.step()
    return mpm.particle_qd.numpy().reshape(2, -1, 3), 9.81 * steps * sim.dt


def test_free_fall_momentum():
    """Material in free fall (no collider reached; equal particle masses) gains exactly the momentum gravity gives it
    when the solver's numerical air drag is off: mean vertical velocity -g t, no net lateral momentum (the block is
    symmetric about the origin). With Isaac's default drag (1 per s, on partly filled boundary nodes) the block falls
    slightly slower, never faster, and still has no net lateral momentum."""
    _metal()
    v, gt = _free_fall(0.0)
    assert np.allclose(v[..., 2].mean(1), -gt, rtol=1e-4), v[..., 2].mean(1)
    assert np.abs(v[..., :2].mean(1)).max() < 1e-6 * gt
    vd, _ = _free_fall(1.0)
    vz = vd[..., 2].mean(1)
    assert np.all(vz > -gt) and np.all(vz < -0.95 * gt), vz
    assert np.abs(vd[..., :2].mean(1)).max() < 1e-6 * gt


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


def test_policy_step_graph_matches_eager():
    """Resets and the coupled step captured into one policy-step graph (``sim.launch_reset``,
    ``mpm.launch_reset(from_sim_mask=True)``, ``mpm.launch_step``) give bitwise the robot state, sand wrench and
    particles of the eager API, with a quarter of the worlds reset per step (probes/policy_step_ab.py)."""
    _metal()
    out = subprocess.run([sys.executable, str(ROOT / "metalsim/physics/mpm/probes/policy_step_ab.py"), "--worlds", "4",
                          "--fraction", "0.25", "--hash-steps", "40", "--steps", "20", "--warmup", "2", "--rounds", "1"],
                         capture_output=True, text=True)
    res = _probe_result(out)
    assert res["bitwise_equal"], res
    assert res["resets_per_run"] > 0, res


def test_one_world_masked_reset():
    """With one world (no separate FEM environments, so the solver's own masked reset refuses grid warm starts) a
    masked reset of the selected world equals the solver's full reset bitwise, and of an unselected world changes
    nothing (probes/one_world_reset.py; before 2026-10-13 ``mpm.reset(mask)`` raised with one world)."""
    _metal()
    out = subprocess.run([sys.executable, str(ROOT / "metalsim/physics/mpm/probes/one_world_reset.py")],
                         capture_output=True, text=True)
    res = _probe_result(out)
    assert res["ok"], res
