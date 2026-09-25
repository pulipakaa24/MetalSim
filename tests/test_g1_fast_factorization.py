"""The G1 task's MuJoCo Warp factorization settings (throughput, docs/research/mjwarp_throughput_2026-09-25.md):
the solver Hessian's Cholesky in registers (Warp fork ``metal_register_cholesky_max`` 48 instead of 40, so the
G1's 43 dofs no longer take the barrier-per-column path) and M / M - dt*D factored by MuJoCo Warp's
tree-sparse L'DL instead of a dense tile Cholesky (``m_dense_max`` 0). Same matrices, different arithmetic:
the physics must still match MuJoCo C step for step to float noise, as the dense path does."""
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")


@pytest.fixture(scope="module")
def task():
    from metalsim.learn.g1_velocity import G1VelocityTask
    wp.config.quiet = True
    return G1VelocityTask(4, terrain="flat", physics_dt=0.0025)


def test_task_uses_the_fast_factorization(task):
    assert task.sim.opt.metal_register_cholesky_max == 48 and wp.config.metal_register_cholesky_max == 48
    assert task.sim.opt.m_dense_max == 0
    m = task.sim.m
    assert m.qLD_block_total == 0 and not m.M_tiles       # no dense tile block: M goes to the sparse L'DL
    assert (m.qLD_block_adr.numpy() == -1).all()          # every dof in the sparse region (Q_LD_BLOCK_SPARSE)


def test_first_steps_match_mujoco_c(task):
    """PD hold plus a fixed random target from Isaac's initial pose: after one and two control steps
    (8 substeps each) every world agrees with mj_step to float noise (the dense path measured 4e-6)."""
    m, sim, n = task.model, task.sim, task.n
    task.reset_all()
    rng = np.random.default_rng(7)
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    q0 = np.tile(m.key_qpos[0], (n, 1)).astype(np.float32); q0[:, :2] += task.origins.numpy()[:, :2]
    sim.t.qpos.copy_(torch.as_tensor(q0)); sim.t.qvel.zero_(); sim.t.qacc_warmstart.zero_()
    v = sim.forward(); sim.after(v); sim.synchronize()
    for step in range(2):
        target = m.key_qpos[0][7:] + 0.25 * rng.uniform(-1, 1, m.nu)
        d.ctrl[:] = target
        sim.t.ctrl.copy_(torch.as_tensor(np.tile(target, (n, 1)).astype(np.float32))); torch.mps.synchronize()
        sim.step()
        for _ in range(task.decimation):
            mujoco.mj_step(m, d)
        sim.synchronize()
        q = sim.d.qpos.numpy().astype(np.float64); q[:, :2] -= task.origins.numpy()[:, :2]
        err = np.abs(q - d.qpos[None]).max()
        assert np.isfinite(q).all() and err < 1e-4, (step, err)
