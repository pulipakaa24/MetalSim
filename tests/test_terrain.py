"""Rough terrain (Isaac's terrain generator layout) and the Metal height scanner (Isaac's
RayCaster height scan), checked against MuJoCo's mj_ray on the same heightfield."""
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from orchard.learn.terrain import HeightScanner, isaac_rough_terrain
from orchard.physics.batch import BatchSim

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")


def test_terrain_layout():
    hf = isaac_rough_terrain(num_rows=2, num_cols=4, seed=0)
    H = hf["H"]
    assert H.shape == (2 * 80 + 40, 4 * 80 + 40)
    assert hf["zmax"] - hf["zmin"] > 0.2                # stairs/boxes present
    assert np.abs(H[:20]).max() == 0 and np.abs(H[:, :20]).max() == 0   # flat border
    o = hf["origins"](64, 0)
    assert o.shape == (64, 3) and np.isfinite(o).all()
    V, F = hf["mesh"]()
    assert V.shape[0] == H.size and F.shape[1] == 3


class _Stub:
    def __init__(self, model, n, sim):
        self.model, self.n, self.sim = model, n, sim
        self.height_scan = wp.zeros((n, 187), dtype=float, device="metal:0")


def test_height_scan_matches_mj_ray():
    hf = isaac_rough_terrain(num_rows=2, num_cols=4, seed=1)
    spec = mujoco.MjSpec()
    h = spec.add_hfield(); h.name = "terrain"; h.nrow, h.ncol = hf["nrow"], hf["ncol"]; h.size = hf["size"]
    h.userdata = hf["data"].reshape(-1).tolist()
    g = spec.worldbody.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_HFIELD; g.hfieldname = "terrain"; g.pos = [0, 0, hf["zmin"]]
    b = spec.worldbody.add_body(); b.name = "torso_link"; b.add_freejoint()
    s = b.add_geom(); s.type = mujoco.mjtGeom.mjGEOM_SPHERE; s.size = [0.05, 0, 0]; s.contype = 0; s.conaffinity = 0
    m = spec.compile()
    n = 8
    sim = BatchSim(m, n)
    o = hf["origins"](n, 3)
    rng = np.random.default_rng(0)
    q = np.zeros((n, 7), np.float32); q[:, :2] = o[:, :2]; q[:, 2] = o[:, 2] + 0.8
    yaw = rng.uniform(-np.pi, np.pi, n); q[:, 3] = np.cos(yaw / 2); q[:, 6] = np.sin(yaw / 2)
    sim.t.qpos.copy_(torch.as_tensor(q)); v = sim.forward(); sim.after(v); sim.synchronize()
    stub = _Stub(m, n, sim)
    sc = HeightScanner(stub, hf)
    sc.launch(); sim.synchronize()
    ours = stub.height_scan.numpy()
    # reference: mj_ray straight down from 20 m above each grid point (grid in the body's yaw frame)
    d = mujoco.MjData(m)
    worst = []
    for e in range(n):
        d.qpos[:] = q[e]; mujoco.mj_forward(m, d)
        c, s_ = np.cos(yaw[e]), np.sin(yaw[e])
        for k, (gx, gy) in enumerate(sc.grid):
            wx = q[e, 0] + c * gx - s_ * gy; wy = q[e, 1] + s_ * gx + c * gy
            geomid = np.zeros(1, np.int32)
            dist = mujoco.mj_ray(m, d, np.array([wx, wy, q[e, 2] + 20.0]), np.array([0, 0, -1.0]), None, 1, -1, geomid)
            ref = q[e, 2] - (q[e, 2] + 20.0 - dist) - 0.5
            worst.append(abs(ours[e, k] - ref))
    worst = np.array(worst)
    print(f"height scan vs mj_ray over {len(worst)} rays: median {np.median(worst):.4f} m, 99th pct {np.percentile(worst, 99):.4f} m, max {worst.max():.4f}")
    assert np.median(worst) < 2e-3 and np.percentile(worst, 99) < 0.05   # triangulation of cells can differ on the diagonal


def test_g1_rough_task_runs():
    from orchard.learn.g1_velocity import G1VelocityTask, benchmark_step
    task = G1VelocityTask(16, terrain="rough")
    assert task.obs_dim == 12 + 3 * 37 + 187
    r = benchmark_step(task, num_frames=10, warmup=2)
    task.sim.synchronize()
    obs = task.obs.numpy(); q = task.sim.d.qpos.numpy()
    assert np.isfinite(obs).all() and np.isfinite(q).all()
    scan = obs[:, -187:]
    assert scan.std() > 0.0 and np.abs(scan).max() < 5.0


@pytest.mark.xfail(strict=True, reason="MuJoCo Warp (metal branch) HFIELD-MESH collision: inverted normals and 5 cm "
                                       "penetrations on contact, launching worlds; MuJoCo C on the same scene is stable. "
                                       "Rough-terrain results are not claimed until this passes (docs/PARITY.md).")
def test_hfield_mesh_contacts_match_mujoco_c():
    """G1 standing on Isaac's rough terrain layout under a PD hold: every heightfield contact normal
    must point up (z > 0), penetrations must stay below 1 cm, and the batched physics must not launch
    the robot (pelvis within 5 cm of MuJoCo C after 0.5 s)."""
    import mujoco
    from orchard.learn.g1_velocity import G1VelocityTask
    task = G1VelocityTask(4, terrain="rough", seed=0); m = task.model
    task.reset_all()
    org = task.origins.numpy()
    q = np.tile(m.key_qpos[0], (4, 1)).astype(np.float32); q[:, :2] = org[:, :2]; q[:, 2] = org[:, 2] + 0.74
    task.sim.t.qpos.copy_(torch.as_tensor(q)); task.sim.t.qvel.zero_()
    task.sim.t.ctrl.copy_(torch.as_tensor(np.tile(m.key_qpos[0][7:], (4, 1)).astype(np.float32)))
    v = task.sim.forward(); task.sim.after(v); task.sim.synchronize()
    d = task.sim.d
    worst_normal, worst_pen = 1.0, 0.0
    for t in range(25):
        task.sim.step(); task.sim.synchronize()
        na = int(d.nacon.numpy()[0])
        if na:
            fr = d.contact.frame.numpy()[:na]; dist = d.contact.dist.numpy()[:na]
            worst_normal = min(worst_normal, float(fr[:, 0, 2].min()) if fr.ndim == 3 else float(fr[:, 2].min()))
            worst_pen = max(worst_pen, float(-dist.min()))
    ours = d.qpos.numpy()[:, 2] - org[:, 2]
    ref = []
    for e in range(4):
        dc = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, dc, 0); dc.qpos[:] = q[e]; dc.ctrl[:] = m.key_qpos[0][7:]
        for _ in range(100):
            mujoco.mj_step(m, dc)
        ref.append(dc.qpos[2] - org[e, 2])
    print(f"hfield contacts over 0.5 s: min normal z {worst_normal:.3f}, max penetration {worst_pen:.4f} m; "
          f"pelvis z ours {ours.round(3)} vs C {np.round(ref, 3)}")
    assert worst_normal > 0.0 and worst_pen < 0.01
    assert np.all(np.abs(ours - np.array(ref)) < 0.05)
