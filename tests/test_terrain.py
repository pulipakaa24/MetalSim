"""Rough terrain (Isaac Lab's terrain generator, ported) checked against Isaac's own generator code,
and the height scanner (Isaac's RayCaster height scan) checked against MuJoCo's mj_ray on the same
heightfield."""
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metalsim.learn.terrain import HeightScanner, isaac_rough_terrain
from metalsim.physics.batch import BatchSim

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")


def test_terrain_layout():
    hf = isaac_rough_terrain(num_rows=2, num_cols=4, seed=0)
    H = hf["H"]                                         # [y index, x index]; Isaac: rows along x, 20 m border
    assert H.shape == (4 * 80 + 2 * 200 + 1, 2 * 80 + 2 * 200 + 1)
    assert hf["size"][0] == 28.0 and hf["size"][1] == 36.0     # centred like Isaac's terrain
    assert hf["zmax"] - hf["zmin"] > 0.2                # stairs/boxes present
    assert np.abs(H[:200]).max() == 0 and np.abs(H[:, :200]).max() == 0   # flat border
    assert hf["origin_table"]().shape == (2, 4, 3)
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
    from metalsim.learn.g1_velocity import G1VelocityTask, benchmark_step
    task = G1VelocityTask(16, terrain="rough")
    assert task.obs_dim == 12 + 3 * 37 + 187
    r = benchmark_step(task, num_frames=10, warmup=2)
    task.sim.synchronize()
    obs = task.obs.numpy(); q = task.sim.d.qpos.numpy()
    assert np.isfinite(obs).all() and np.isfinite(q).all()
    scan = obs[:, -187:]
    assert scan.std() > 0.0 and np.abs(scan).max() < 5.0


def test_hfield_mesh_contacts_match_mujoco_c():
    """G1 standing on Isaac's rough terrain layout under a PD hold: every heightfield contact normal
    must point up (z > 0), penetrations must stay in the range MuJoCo C reports for the same initial
    placement (the feet start intersecting terrain boxes by up to 4.4 cm), and the batched physics
    must not launch the robot (pelvis within 5 cm of MuJoCo C after 0.5 s).

    Before the plane-contact patch to MuJoCo Warp's heightfield kernel (patches/mujoco_warp-hfield-
    plane-contacts.patch) this failed: an inverted normal (z = -1), a 2.05 m penetration, one world
    launched to 2.3 m."""
    import mujoco
    from metalsim.learn.g1_velocity import G1VelocityTask
    task = G1VelocityTask(4, terrain="rough", seed=0, terrain_collision="hfield"); m = task.model   # the heightfield kernel
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
    ref = []; c_pen = 0.0
    for e in range(4):
        dc = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, dc, 0); dc.qpos[:] = q[e]; dc.ctrl[:] = m.key_qpos[0][7:]
        for t in range(100):
            mujoco.mj_step(m, dc)
            if t % 4 == 0 and dc.ncon:
                c_pen = max(c_pen, float(-min(c.dist for c in dc.contact[:dc.ncon])))
        ref.append(dc.qpos[2] - org[e, 2])
    print(f"hfield contacts over 0.5 s: min normal z {worst_normal:.3f}, max penetration {worst_pen:.4f} m (MuJoCo C: {c_pen:.4f}); "
          f"pelvis z ours {ours.round(3)} vs C {np.round(ref, 3)}")
    assert worst_normal > 0.0 and worst_pen < max(0.02, 1.5 * c_pen)   # the initial placement intersects terrain boxes
    assert np.all(np.abs(ours - np.array(ref)) < 0.05)


# --------------------------------------------------------------------------------------------------
# exactness against Isaac Lab's own terrain generator (v2.3.2 sources in assets/isaac/terrains/, run on
# the CPU through tests/isaac_terrain_ref.py)

def _isaac_ref():
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    import isaac_terrain_ref
    return isaac_terrain_ref


def _surface_of_isaac_mesh(mesh, shape, x0, y0, res=0.1):
    """Top surface of an Isaac trimesh at the grid points (x0 + i res, y0 + j res)."""
    from metalsim.learn.isaac_terrain import top_surface_of_triangles
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.faces)
    P = np.stack([(V[:, 0] - x0) / res, (V[:, 1] - y0) / res], 1)
    r = np.round(P); snap = np.abs(P - r) < 1e-3; P[snap] = r[snap]     # float32 vertices of grid points
    return top_surface_of_triangles(P, V[:, 2].astype(np.float32).astype(np.float64), F, shape).astype(np.float32)


def _warp_raycast(mesh, xy):
    """Isaac's RayCaster operation (isaaclab.utils.warp.raycast_mesh: wp.mesh_query_ray on the float32
    terrain mesh) straight down from 20 m, on Warp's CPU device."""
    m = wp.Mesh(points=wp.array(np.asarray(mesh.vertices, np.float32), dtype=wp.vec3, device="cpu"),
                indices=wp.array(np.asarray(mesh.faces, np.int32).ravel(), dtype=wp.int32, device="cpu"))
    out = wp.zeros(len(xy), dtype=float, device="cpu")
    wp.launch(_raycast_down, dim=len(xy), inputs=[m.id, wp.array(xy.astype(np.float32), dtype=wp.vec2, device="cpu"), out],
              device="cpu")
    return out.numpy()


@wp.kernel
def _raycast_down(mesh: wp.uint64, xy: wp.array(dtype=wp.vec2), out: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, wp.vec3(xy[i][0], xy[i][1], 20.0), wp.vec3(0.0, 0.0, -1.0), 1.0e6)
    out[i] = 20.0 - q.t if q.result else -1.0e9


def test_isaac_rough_terrain_exact():
    """Isaac-Velocity-Rough-G1-v0's terrain at env seed 0: Isaac's TerrainGenerator (its code, CPU torch)
    vs ours with torch_device="cpu": env origins bit-for-bit, heightfield = the top surface of Isaac's
    terrain mesh at every grid point bit-for-bit, and within float32 ray precision of Isaac's own
    ray cast (wp.mesh_query_ray) everywhere except on vertical walls lying on grid lines."""
    from metalsim.learn.isaac_terrain import isaac_rough_terrain_generator
    R = _isaac_ref()
    isaac = R.isaac_rough_generator(seed=0)
    ours = isaac_rough_terrain_generator(seed=0, torch_device="cpu")
    assert np.array_equal(isaac.terrain_origins, ours.terrain_origins)
    hf = isaac_rough_terrain(seed=0, torch_device="cpu")
    assert np.array_equal(hf["origin_table"](), isaac.terrain_origins.astype(np.float32))
    H = hf["H"].T                                       # back to Isaac's [x, y]
    assert H.shape == (1201, 2001) and hf["size"][:2] == [60.0, 100.0]
    ref = _surface_of_isaac_mesh(isaac.terrain_mesh, H.shape, -60.0, -100.0)
    assert np.array_equal(H.astype(np.float32), ref), f"max diff {np.abs(H - ref).max()}"
    # Isaac's own ray cast at every grid point
    ix, iy = np.meshgrid(np.arange(H.shape[0]), np.arange(H.shape[1]), indexing="ij")
    z = _warp_raycast(isaac.terrain_mesh, np.stack([ix.ravel() * 0.1 - 60.0, iy.ravel() * 0.1 - 100.0], 1)).reshape(H.shape)
    d = np.abs(z - H)
    wall = np.zeros(H.shape, bool)
    for a in (-1, 1):
        for ax in (0, 1):
            wall |= np.abs(np.roll(H, a, ax) - H) > 1e-3
    print(f"Isaac ray cast vs ours at {H.size} grid points: {np.mean(d < 1e-5):.4f} within 1e-5 m "
          f"(max {d[~wall].max():.1e} off walls); {int((d > 1e-5).sum())} differ, all on walls: {bool(np.all(wall[d > 1e-5]))}")
    assert d[~wall].max() < 1e-5 and np.all(wall[d > 1e-5]) and (d > 1e-5).mean() < 1e-3
    # Isaac on an NVIDIA GPU draws the random-grid box heights from torch's CUDA Philox generator: the
    # default torch_device="cuda" differs from the CPU run only in the four "boxes" columns (8-11)
    cuda = isaac_rough_terrain_generator(seed=0)
    diff_cols = np.nonzero(np.any(cuda.surface != ours.surface, axis=0))[0]
    assert diff_cols.min() >= 200 + 8 * 80 and diff_cols.max() <= 200 + 12 * 80
    assert np.array_equal(np.delete(cuda.terrain_origins, [8, 9, 10, 11], 1), np.delete(ours.terrain_origins, [8, 9, 10, 11], 1))


def test_torch_cuda_uniform_reproduction():
    """The Philox4x32-10 generator behind torch's CUDA uniform_: Random123 known-answer vectors, and
    torch.manual_seed(0); torch.rand(8, device="cuda") = [0.3990, 0.5167, 0.0249, 0.9401, 0.9459,
    0.7967, 0.4150, 0.8203] (the values torch prints on NVIDIA GPUs)."""
    from metalsim.learn.isaac_terrain import TorchUniform, philox4x32_10
    kat = [((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
           ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
           ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0), (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1))]
    for ctr, key, want in kat:
        got = philox4x32_10([np.array([c], np.uint32) for c in ctr], key)
        assert tuple(int(g[0]) for g in got) == want
    r = TorchUniform(0, "cuda")(8, 0.0, 1.0)
    assert np.allclose(r, [0.3990, 0.5167, 0.0249, 0.9401, 0.9459, 0.7967, 0.4150, 0.8203], atol=5e-5)


def _isaac_sub_cfg(R, ours_cfg):
    """Isaac's cfg object for one of our sub-terrain cfgs (same parameters)."""
    tg = R.load()
    from metalsim.learn.isaac_terrain import HfCfg
    cls = {"hf_random_uniform": "HfRandomUniformTerrainCfg", "hf_pyramid_stairs": "HfPyramidStairsTerrainCfg",
           "hf_discrete_obstacles": "HfDiscreteObstaclesTerrainCfg", "mesh_pyramid_stairs": "MeshPyramidStairsTerrainCfg",
           "mesh_inverted_pyramid_stairs": "MeshInvertedPyramidStairsTerrainCfg", "mesh_random_grid": "MeshRandomGridTerrainCfg",
           "hf_pyramid_sloped": "HfPyramidSlopedTerrainCfg"}[ours_cfg.function]
    fields = {k: v for k, v in vars(ours_cfg).items() if k != "function" and v is not None}
    if not isinstance(ours_cfg, HfCfg):
        for k in ("grid_width", "grid_height_range", "step_height_range", "step_width"):
            fields.pop(k, None) if getattr(ours_cfg, k) is None else None
    else:
        for k in ("obstacle_height_mode",):
            if ours_cfg.function != "hf_discrete_obstacles":
                fields.pop(k)
        if ours_cfg.function not in ("hf_pyramid_sloped", "hf_pyramid_stairs"):
            fields.pop("inverted")
        if ours_cfg.function == "hf_random_uniform":
            fields.pop("platform_width")
    return getattr(tg, cls)(**fields)


def _sub_cfgs():
    from metalsim.learn.isaac_terrain import HfCfg, MeshCfg, rough_terrains_cfg
    hf = dict(size=(8.0, 8.0), horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75)
    out = {name: c for name, c in rough_terrains_cfg().sub_terrains.items()}
    for c in out.values():
        if isinstance(c, HfCfg):
            c.slope_threshold = 0.75
    out["hf_pyramid_stairs"] = HfCfg("hf_pyramid_stairs", step_height_range=(0.05, 0.23), step_width=0.3, platform_width=3.0,
                                     border_width=0.25, **hf)
    out["hf_pyramid_stairs_inv"] = HfCfg("hf_pyramid_stairs", step_height_range=(0.05, 0.23), step_width=0.3,
                                         platform_width=3.0, border_width=0.25, inverted=True, **hf)
    out["hf_discrete_obstacles"] = HfCfg("hf_discrete_obstacles", obstacle_width_range=(0.5, 2.0),
                                         obstacle_height_range=(0.05, 0.3), num_obstacles=40, platform_width=2.0,
                                         border_width=0.25, **hf)
    return out


@pytest.mark.parametrize("name", ["pyramid_stairs", "pyramid_stairs_inv", "boxes", "random_rough", "hf_pyramid_slope",
                                  "hf_pyramid_slope_inv", "hf_pyramid_stairs", "hf_pyramid_stairs_inv", "hf_discrete_obstacles"])
def test_isaac_sub_terrain_exact(name):
    """Each ported sub-terrain function against Isaac's (same cfg, same random state) at several
    difficulties: origin bit-for-bit; height-field terrains: Isaac's mesh vertices (after its slope-
    threshold moves) bit-for-bit from our heights and moves; every terrain: top surface on the grid
    bit-for-bit."""
    import torch
    from metalsim.learn.isaac_terrain import HfCfg, TorchUniform, height_field_terrain, height_field_vertex_moves, sub_terrain
    R = _isaac_ref()
    cfg = _sub_cfgs()[name]
    cfg.size = (8.0, 8.0)
    for k, difficulty in enumerate([0.0, 0.13, 0.5, 0.77, 0.999]):
        seed = 17 + k
        icfg = _isaac_sub_cfg(R, cfg)
        np.random.seed(seed); torch.manual_seed(seed)
        meshes, iorigin = icfg.function(difficulty, icfg.copy())
        import trimesh
        imesh = trimesh.util.concatenate(meshes)
        rs = np.random.RandomState(seed); tr = TorchUniform(seed, "cpu")
        surf, origin, heights, boxes = sub_terrain(difficulty, cfg, rs, tr, 0.1)
        assert np.array_equal(origin, iorigin + np.array([-4.0, -4.0, 0.0])), (origin, iorigin)
        ref = _surface_of_isaac_mesh(imesh, surf.shape, 0.0, 0.0)
        assert np.array_equal(surf, ref), f"{name} d={difficulty}: max diff {np.abs(surf - ref).max()}"
        if isinstance(cfg, HfCfg):
            import copy
            c2 = copy.deepcopy(cfg)
            h2, _ = height_field_terrain(difficulty, c2, np.random.RandomState(seed))
            dx, dy = height_field_vertex_moves(h2, 0.1, 0.005, 0.75)
            lin = np.linspace(0, 80 * 0.1, 81)
            X = (lin[:, None] + dx * 0.1).astype(np.float32); Y = (lin[None, :] + dy * 0.1).astype(np.float32)
            ours_v = np.stack([X.ravel(), Y.ravel(), (h2.flatten() * 0.005).astype(np.float32)], 1)
            # trimesh merges coincident vertices (moved vertices land on their neighbours): compare as sets
            assert np.array_equal(np.unique(ours_v, axis=0), np.unique(np.asarray(meshes[0].vertices, np.float32), axis=0))


def test_height_scan_ray_order_matches_isaac_recording():
    """The height-scan rays are in Isaac's order (GridPatternCfg ordering "xy": x fastest). Oracle: Isaac Sim 5.1's own
    step-0 policy observation recorded by isaac_side/play_policy.py on the training terrain (env seed 0, rows 3 and 6,
    columns 0/4/9/14, robot in its default pose at the cell origin). Runs the scan kernel on Warp's CPU device. The
    legacy "ij" order (MetalSim before 2026-09-25) must differ on the random-rough cell, where the scan is not symmetric."""
    import glob, json, mujoco
    from metalsim.learn.terrain import height_scan_grid, scan_grid
    from metalsim.learn.g1_velocity import build_g1_model
    metas = sorted(glob.glob("runs/parity/isaac/rough/play_rough/L*/isaac_it500/meta.json"))
    if not metas:
        pytest.skip("Isaac recording not present (runs/parity/isaac/rough/play_rough)")
    hf = isaac_rough_terrain(seed=0)
    m, _ = build_g1_model("flat"); body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    H = wp.array(hf["H"].astype(np.float32), dtype=float, device="cpu")
    for path in metas:
        mt = json.load(open(path)); obs0 = np.array(mt["obs0"], np.float32); eo = np.array(mt["env_origins"], np.float32)
        xp, xm = [], []
        for o in eo:
            d = mujoco.MjData(m); d.qpos[:] = m.key_qpos[0]; d.qpos[:3] = o + m.key_qpos[0][:3]; mujoco.mj_kinematics(m, d)
            xp.append(d.xpos.copy()); xm.append(d.xmat.reshape(-1, 3, 3).copy())
        xp = wp.array(np.array(xp, np.float32), dtype=wp.vec3, device="cpu"); xm = wp.array(np.array(xm, np.float32), dtype=wp.mat33, device="cpu")
        err = {}
        for order in ("xy", "ij"):
            grid = wp.array(scan_grid(order), dtype=float, device="cpu"); out = wp.zeros((len(eo), 187), dtype=float, device="cpu")
            wp.launch(height_scan_grid, dim=(len(eo), 187), inputs=[xp, xm, body, grid, H, float(hf["res"]), float(hf["size"][0]),
                                                                    float(hf["size"][1]), 0.5, out], device="cpu")
            err[order] = np.abs(np.clip(out.numpy(), -1, 1) - obs0[:, -187:])
        assert err["xy"].max() < 1e-3, (path, err["xy"].max())
        assert err["ij"].max() > 0.02, path


# --------------------------------------------------------------------------------------------------
# collision surfaces for Isaac's vertical walls (metalsim.learn.terrain TERRAIN_COLLISION;
# docs/research/terrain_walls_2026-09-25.md)

def test_terrain_collision_surfaces_vs_isaac_mesh():
    """Off-grid, the top of the MuJoCo collision geometry of each option against Isaac's own ray cast on Isaac's own
    terrain mesh (its generator run here, CPU torch box heights) at random points in every sub-terrain type: "boxes"
    (Isaac's trimesh sub-terrains as its own boxes) is exact on the stair and box cells, where the 0.1 m heightfield
    ramps the walls; the height-scan grid ``H`` is the same in every mode."""
    from metalsim.learn.terrain import collision_top
    R = _isaac_ref()
    isaac = R.isaac_rough_generator(seed=0)
    rng = np.random.default_rng(0)
    base = isaac_rough_terrain(seed=0, torch_device="cpu")
    types = {}
    for s in base["generator"].sub_terrains:
        types.setdefault(s.name, []).append((s.row, s.col))
    pts = {}
    for name, cells in types.items():
        rc = np.array(cells)[rng.integers(0, len(cells), 4000)]
        pts[name] = np.stack([rc[:, 0] * 8.0 - 40.0 + rng.uniform(0, 8, 4000), rc[:, 1] * 8.0 - 80.0 + rng.uniform(0, 8, 4000)], 1)
    ref = {n: _warp_raycast(isaac.terrain_mesh, p) for n, p in pts.items()}
    err = {}
    for mode in ("hfield", "boxes"):
        hf = base if mode == "hfield" else isaac_rough_terrain(seed=0, torch_device="cpu", collision=mode)
        assert np.array_equal(hf["H"], base["H"])            # the height scan reads the same exact grid in every mode
        # within 0.1 mm of a wall take the better side (Isaac's float32 vertices vs the float64 box geoms)
        err[mode] = {n: np.min([np.abs(collision_top(hf, p[:, 0] + dx, p[:, 1] + dy) - ref[n])
                                for dx, dy in ((0, 0), (1e-4, 0), (-1e-4, 0), (0, 1e-4), (0, -1e-4))], 0) for n, p in pts.items()}
    for n in ("pyramid_stairs", "pyramid_stairs_inv", "boxes"):
        print(f"{n}: hfield mean {err['hfield'][n].mean():.4f} p99 {np.percentile(err['hfield'][n], 99):.3f} max "
              f"{err['hfield'][n].max():.3f} m; boxes max {err['boxes'][n].max():.1e} m")
        assert err["boxes"][n].max() < 1e-5 and np.percentile(err["hfield"][n], 99) > 0.05
    for n in ("random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv"):    # height-field cells: same heightfield in both
        assert np.allclose(err["boxes"][n], err["hfield"][n], atol=1e-6)     # (float32 data renormalised to a lower zmin)


def test_exact_height_scan_matches_isaac_raycast():
    """HeightScanner(surface="exact") off the grid (random torso positions and yaws on every sub-terrain type) against
    Isaac's ray cast on Isaac's own mesh: exact on the box-built sub-terrains (stairs, inverted stairs, boxes), where the
    default grid interpolation reads walls as 0.1 m ramps; identical to the grid scan elsewhere. Warp CPU device."""
    from metalsim.learn.terrain import box_cell_tops, height_scan_exact, height_scan_grid, scan_grid
    R = _isaac_ref()
    isaac = R.isaac_rough_generator(seed=0)
    hf = isaac_rough_terrain(seed=0, torch_device="cpu"); gen = hf["generator"]
    rng = np.random.default_rng(3); n = 600
    cells = [(s.row, s.col, s.boxes is not None) for s in gen.sub_terrains]
    pick = np.array(cells)[rng.integers(0, len(cells), n)]
    pos = np.stack([pick[:, 0] * 8.0 - 40.0 + rng.uniform(1.0, 7.0, n), pick[:, 1] * 8.0 - 80.0 + rng.uniform(1.0, 7.0, n),
                    np.full(n, 1.0)], 1).astype(np.float32)
    yaw = rng.uniform(-np.pi, np.pi, n)
    xm = np.zeros((n, 1, 3, 3), np.float32); xm[:, 0, 0, 0] = np.cos(yaw); xm[:, 0, 0, 1] = -np.sin(yaw)
    xm[:, 0, 1, 0] = np.sin(yaw); xm[:, 0, 1, 1] = np.cos(yaw); xm[:, 0, 2, 2] = 1.0
    dev = "cpu"
    xp = wp.array(pos[:, None], dtype=wp.vec3, device=dev); xmw = wp.array(xm, dtype=wp.mat33, device=dev)
    g = scan_grid("xy"); grid = wp.array(g, dtype=float, device=dev)
    H = wp.array(hf["H"].astype(np.float32), dtype=float, device=dev)
    blk, F = box_cell_tops(gen)
    common = [xp, xmw, 0, grid, H, float(hf["res"]), float(hf["size"][0]), float(hf["size"][1]), 0.5]
    out_e = wp.zeros((n, 187), dtype=float, device=dev); out_g = wp.zeros((n, 187), dtype=float, device=dev)
    wp.launch(height_scan_exact, dim=(n, 187), inputs=common + [wp.array(blk, dtype=int, device=dev), wp.array(F, dtype=float, device=dev),
                                                                8.0, -40.0, -80.0, 0.025, out_e], device=dev)
    wp.launch(height_scan_grid, dim=(n, 187), inputs=common + [out_g], device=dev)
    rx = pos[:, 0:1] + np.cos(yaw)[:, None] * g[None, :, 0] - np.sin(yaw)[:, None] * g[None, :, 1]
    ry = pos[:, 1:2] + np.sin(yaw)[:, None] * g[None, :, 0] + np.cos(yaw)[:, None] * g[None, :, 1]
    ref = 1.0 - _warp_raycast(isaac.terrain_mesh, np.stack([rx.ravel(), ry.ravel()], 1)).reshape(n, 187) - 0.5
    box = pick[:, 2].astype(bool)
    ee, eg = np.abs(out_e.numpy() - ref), np.abs(out_g.numpy() - ref)
    print(f"box-built cells, {int(box.sum()) * 187} rays: exact scan max {ee[box].max():.1e} m, >1 mm {np.mean(ee[box] > 1e-3):.5f}; "
          f"grid scan mean {eg[box].mean():.4f} p99 {np.percentile(eg[box], 99):.3f} max {eg[box].max():.3f} m")
    assert np.mean(ee[box] > 1e-3) < 1e-4 and np.percentile(eg[box], 99) > 0.05
    assert np.array_equal(out_e.numpy()[~box], out_g.numpy()[~box])


@pytest.mark.parametrize("mode", ["boxes", "meshes", "boxes_local"])
def test_step_edge_contacts_match_mujoco_c(mode):
    """The G1 foot collider at a 0.11 m riser of the inverted-stairs pit (scripts/diagnostics/terrain_step_edge.py):
    stubbing into the riser at 1 m/s and landing across the riser edge, MuJoCo Warp vs MuJoCo C on the same model.
    With Isaac's boxes the riser is a wall: the toe stops at it (soft-contact penetration only), the foot does not
    climb, a near-horizontal contact normal appears, and Warp follows C's trajectory ("boxes_local": Warp with the
    per-world box window against C on all boxes). The 0.1 m heightfield fails all of this (runs/terrain_walls/step_edge.jsonl:
    Warp's normal never below z 0.67, Warp vs C 0.21 m apart after 0.4 s)."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "diagnostics"))
    import terrain_step_edge as S
    r = S.run(mode)
    print(r)
    for side in ("C", "warp"):
        assert r[side]["stub_toe_x_max_minus_riser"] < 0.01 and r[side]["stub_sole_rise"] < 0.01
        assert r[side]["stub_min_abs_normal_z"] < 0.2
    assert max(r["warp_vs_C_pos_final"]) < 0.01 and r["warp_vs_C_pos_max"] < 0.02
