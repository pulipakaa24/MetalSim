"""BatchSimOptions(deterministic=True): MuJoCo Warp's kernels in Warp's RUN_TO_RUN mode, so identical runs are bitwise
equal (the default path is not: batch_02 measured a box on the floor diverging at step 18 at 1,024 worlds). Each case
runs in its own process (the mode is process-wide), with a cold kernel cache (fresh WARP_CACHE_PATH) and warm, and the
refusal when MuJoCo Warp already ran in default mode. Design: research_notes/MPM in main simulator/deterministic_batchsim.md.

    scripts/gpu_run.sh test_det check 40 -- .venv/bin/python -m pytest tests/test_batch_deterministic.py
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest
import warp as wp

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIAG = ROOT / "metalsim/physics/mpm/probes/rigid_det_diag.py"


def _diag(scene, worlds, out, cache=None, steps=150):
    env = dict(os.environ)
    if cache is not None:
        env["WARP_CACHE_PATH"] = cache
    r = subprocess.run([sys.executable, str(DIAG), "--scene", scene, "--worlds", str(worlds), "--steps", str(steps),
                        "--json", out], capture_output=True, text=True, env=env)
    assert r.returncode in (0, 3), r.stderr[-3000:]
    return json.load(open(out)), r


@pytest.mark.parametrize("scene,worlds", [("box", 1024), ("stack", 4), ("stack", 256), ("panda", 16)])
def test_repeats_cold_and_warm_cache(scene, worlds):
    """Two runs in one process are bitwise equal, and so are three processes: two with a fresh (cold) kernel cache
    each, one with the warm cache the second left."""
    if scene == "panda" and not any(p.exists() for p in (ROOT / "upstream/mujoco_menagerie",
                                                          ROOT.parent.parent.parent / "upstream/mujoco_menagerie")):
        pytest.skip("MuJoCo Menagerie not present")
    with tempfile.TemporaryDirectory() as d:
        c1, c2 = os.path.join(d, "cache1"), os.path.join(d, "cache2")
        a, ra = _diag(scene, worlds, os.path.join(d, "a.json"), c1)
        b, rb = _diag(scene, worlds, os.path.join(d, "b.json"), c2)
        c, rc = _diag(scene, worlds, os.path.join(d, "c.json"), c2)
        for x, r in ((a, ra), (b, rb), (c, rc)):
            assert all(f is None for f in x["in_process_first_diff"]), (x["in_process_first_diff"], r.stdout[-500:])
        assert a["runs"][0]["per_step"] == b["runs"][0]["per_step"] == c["runs"][0]["per_step"]


def test_refuses_after_default_mode_launch():
    """A deterministic BatchSim refuses to start once MuJoCo Warp ran in default mode in the process, and a default
    one refuses after a deterministic one (one mode per process)."""
    code = """
import mujoco, sys
sys.path.insert(0, {root!r})
from metalsim.physics.batch import BatchSim, BatchSimOptions
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="0 0 1"/><body pos="0 0 .3"><freejoint/>'
                                   '<geom type="sphere" size=".05"/></body></worldbody></mujoco>')
first, second = {first}, {second}
s = BatchSim(m, 2, options=BatchSimOptions(deterministic=first)); s.step(); s.synchronize()
try:
    BatchSim(m, 2, options=BatchSimOptions(deterministic=second))
except RuntimeError as e:
    print("REFUSED", e)
"""
    for first, second in ((False, True), (True, False)):
        r = subprocess.run([sys.executable, "-c", code.format(root=str(ROOT), first=first, second=second)],
                           capture_output=True, text=True)
        assert "REFUSED" in r.stdout, (first, second, r.stdout[-1000:], r.stderr[-2000:])


def test_stack_uses_deterministic_ccd_split():
    """The stacked boxes' box-box contacts go through MuJoCo Warp's deterministic CCD split (solve + write kernels),
    built in RUN_TO_RUN mode, and not through the single CCD kernel, whose slot counter breaks repeatability
    (research_notes/MPM in main simulator/deterministic_ccd_divergence.md)."""
    with tempfile.TemporaryDirectory() as d:
        x, r = _diag("stack", 4, os.path.join(d, "s.json"), steps=160)
    mods = x["ccd_modules"]["modules"]
    assert any(n.startswith("ccd_det_kernels_builder") for n in mods), (mods, r.stdout[-500:])
    assert not any(n.startswith("ccd_kernel_builder") for n in mods), mods
    assert all(v == str(wp.DeterministicMode.RUN_TO_RUN) for v in mods.values()), mods


def test_ccd_split_matches_single_kernel():
    """The split computes the same contacts as the single CCD kernel (bitwise, on CPU, at sampled steps of the
    stacked-boxes scene)."""
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, WARP_CACHE_PATH=os.path.join(d, "cache"))
        r = subprocess.run([sys.executable, "-m", "metalsim.physics.mpm.probes.ccd_split_equiv", "--device", "cpu",
                            "--worlds", "8", "--steps", "200"], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])


def test_iteration_cap_repeats():
    """Worlds that reach the solver's iteration cap repeat too (captured, with MuJoCo Warp's early exit of the solver
    loop). MuJoCo Warp's _solve_done read the iteration count back after incrementing it; under Warp's two-pass slot
    counter (the early exit's) a world at the cap then made the counter call in one pass only, and Warp's tripwire
    stopped the run (the G1 task, batch_16 / 17)."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "cap.json")
        r = subprocess.run([sys.executable, str(DIAG), "--scene", "stack", "--worlds", "64", "--steps", "160",
                            "--iterations", "2", "--json", out], capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-3000:])
        x = json.load(open(out))
    assert all(f is None for f in x["in_process_first_diff"]), x["in_process_first_diff"]
