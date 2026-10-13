"""BatchSimOptions(deterministic=True): MuJoCo Warp's kernels in Warp's RUN_TO_RUN mode, so two identical runs are
bitwise equal (the default path is not: float atomics in the constraint solver accumulate in arrival order on Metal).
Run in its own process (the option sets process-wide module options):

    scripts/gpu_run.sh test_det check 20 -- .venv/bin/python -m pytest tests/test_batch_deterministic.py
"""
import hashlib
import pathlib

import numpy as np
import pytest
import warp as wp

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

ROOT = pathlib.Path(__file__).resolve().parents[1]
PANDA = ROOT.parent.parent.parent / "upstream/mujoco_menagerie/franka_emika_panda/scene.xml"
BOXES = """<mujoco><worldbody><geom type="plane" size="0 0 1"/>
<body pos="0 0 0.3" euler="20 30 10"><freejoint/><geom type="box" size=".05 .07 .04"/></body>
<body pos="0.02 0 0.5" euler="10 0 40"><freejoint/><geom type="box" size=".04 .04 .04"/></body></worldbody></mujoco>"""


def _run(model, n, steps=150, deterministic=True):
    from metalsim.physics.batch import BatchSim, BatchSimOptions

    sim = BatchSim(model, n, options=BatchSimOptions(deterministic=deterministic))
    rng = np.random.default_rng(0)
    h = hashlib.sha256()
    for s in range(steps):
        if model.nu:
            c = rng.uniform(-1, 1, (n, model.nu)).astype(np.float32)
            lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            sim.d.ctrl.assign(np.where(model.actuator_ctrllimited.astype(bool), lo + 0.5 * (c + 1) * (hi - lo), c)
                              .astype(np.float32))
        sim.step()
        if s % 10 == 9:
            h.update(sim.d.qpos.numpy().tobytes())
            h.update(sim.d.qvel.numpy().tobytes())
    assert np.isfinite(sim.d.qpos.numpy()).all()
    return h.hexdigest()


@pytest.mark.parametrize("n", [4, 256])
def test_stacked_boxes_repeat(n):
    """Two boxes landing on each other and on the floor, n worlds: two runs bitwise equal."""
    import mujoco

    m = mujoco.MjModel.from_xml_string(BOXES)
    assert _run(m, n) == _run(m, n)


def test_panda_repeats():
    """The Franka panda with seeded random controls (16 worlds), not repeatable by default, repeats bitwise."""
    import mujoco

    path = PANDA if PANDA.exists() else ROOT / "upstream/mujoco_menagerie/franka_emika_panda/scene.xml"
    if not path.exists():
        pytest.skip("MuJoCo Menagerie not present")
    m = mujoco.MjModel.from_xml_path(str(path))
    assert _run(m, 16) == _run(m, 16)
