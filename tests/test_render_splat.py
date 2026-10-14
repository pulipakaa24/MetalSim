"""3D Gaussian splat renderer (metalsim/render/splat, SPLAT.md): format decoders, Metal vs the independent float64 reference,
batching / sub-batching / determinism, capacity growth, analytic single-Gaussian check and the mesh composite.
Decoder, transform and float64-reference tests run on any machine (CPU); the rest need Metal. All scenes are synthetic;
real-data checks on fetched public scenes are in test_render_splat_public.py."""
import gzip
import os
import struct

import mujoco
import numpy as np
import pytest
import warp as wp

from metalsim.render.splat import formats
from metalsim.render.splat import reference as ref
from metalsim.render.splat.formats import GaussianCloud
from metalsim.render.splat.scene import PackedSplats, floor_plane_transform, make_transform, prepare

needs_metal = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
if wp.is_metal_available():
    from metalsim.render.splat.renderer import SplatRenderer


def random_cloud(n=1500, seed=0, deg=3, box=1.5, u8=True, scale=(0.01, 0.15), aa=False):
    rng = np.random.default_rng(seed)
    means = rng.uniform(-box, box, (n, 3)).astype(np.float32)
    ls = np.log(rng.uniform(scale[0], scale[1], (n, 3))).astype(np.float32)
    q = rng.normal(size=(n, 4)).astype(np.float32); q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = rng.uniform(0.02, 1.0, n).astype(np.float32)
    dc = rng.normal(0, 1.0, (n, 3)).astype(np.float32)
    K = (deg + 1) ** 2 - 1
    rest = None
    if K:
        rest = rng.integers(64, 192, (n, K, 3)).astype(np.uint8) if u8 else rng.normal(0, 0.3, (n, K, 3)).astype(np.float16)
    return GaussianCloud(means, ls, q, op, dc, rest, u8, deg, aa, "synthetic")


def look_at(pos, target=(0, 0, 0)):
    p = np.asarray(pos, float); f = np.asarray(target, float) - p; f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(f, up); x /= np.linalg.norm(x); y = np.cross(x, f)
    return np.stack([x, y, -f], 1)          # MuJoCo camera frame columns


CAMS = [(0.2, -4.0, 1.0), (3.0, 2.0, 2.5), (-2.5, -0.5, 0.3), (0.0, 0.1, 4.0)]
W, H = 96, 64


def _levels(a, b):
    return np.abs(a.astype(int) - (np.clip(b, 0, 1) * 255 + 0.5).astype(int)).max(-1)


@needs_metal
@pytest.mark.parametrize("deg,u8,aa,seed", [(3, True, False, 0), (1, False, False, 1), (0, True, True, 2), (2, False, True, 3)])
def test_metal_matches_reference(deg, u8, aa, seed):
    s = prepare(random_cloud(seed=seed, deg=deg, u8=u8, aa=aa), crop=None, log=lambda *a: None)
    pos = np.array(CAMS); mats = np.array([look_at(p) for p in CAMS])
    r = SplatRenderer(s, len(CAMS), width=W, height=H, hfov_deg=75, outputs=("rgb", "depth", "depth_expected", "alpha"))
    out = r.render_host(pos, cam_mat=mats)
    for i in range(len(CAMS)):
        rr = ref.render(s, pos[i], mats[i], r.intrinsics, W, H)
        d = _levels(out["rgb"][i], rr["rgb"])
        assert d.max() <= 3 and (d > 1).mean() < 2e-3, (i, d.max(), (d > 1).mean())
        both = (out["depth"][i] > 0) & (rr["depth"] > 0)
        assert ((out["depth"][i] > 0) == (rr["depth"] > 0)).mean() > 0.999
        # median depth jumps where T crosses 0.5 within float32/float64 rounding: bound the fraction of such pixels
        assert (np.abs(out["depth"][i] - rr["depth"])[both] > 1e-3).mean() < 2e-3
        assert np.abs(out["alpha"][i] - rr["alpha"]).max() < 0.01


@needs_metal
def test_batching_subbatches_and_determinism():
    s = prepare(random_cloud(seed=5), crop=None, log=lambda *a: None)
    pos = np.array(CAMS); mats = np.array([look_at(p) for p in CAMS])
    full = SplatRenderer(s, 4, width=W, height=H, hfov_deg=75).render_host(pos, cam_mat=mats)
    sub = SplatRenderer(s, 4, width=W, height=H, hfov_deg=75, batch=3).render_host(pos, cam_mat=mats)
    single = SplatRenderer(s, 1, width=W, height=H, hfov_deg=75)
    for k in ("rgb", "depth"):
        assert np.array_equal(full[k], sub[k])
    for i in range(4):
        o = single.render_host(pos[i:i + 1], cam_mat=mats[i:i + 1])
        assert np.array_equal(o["rgb"][0], full["rgb"][i]) and np.array_equal(o["depth"][0], full["depth"][i])
    again = SplatRenderer(s, 4, width=W, height=H, hfov_deg=75).render_host(pos, cam_mat=mats)
    assert np.array_equal(again["rgb"], full["rgb"])


@needs_metal
def test_capacity_growth():
    s = prepare(random_cloud(seed=6), crop=None, log=lambda *a: None)
    pos = np.array(CAMS); mats = np.array([look_at(p) for p in CAMS])
    ok = SplatRenderer(s, 4, width=W, height=H, hfov_deg=75).render_host(pos, cam_mat=mats)
    small = SplatRenderer(s, 4, width=W, height=H, hfov_deg=75, cap_records=1000, cap_keys=2000)
    with pytest.warns(UserWarning, match="capacity exceeded"):
        o = small.render_host(pos, cam_mat=mats)
    assert np.array_equal(o["rgb"], ok["rgb"])


@needs_metal
def test_single_gaussian_analytic():
    """One isotropic Gaussian (sigma 0.1 m) 2 m in front of the camera, SH degree 0: pixel = 0.99-clamped opacity x
    exp(-r^2 / (2 (f sigma / z)^2 + 2 * 0.3)) x colour, on black."""
    c = GaussianCloud(np.array([[0, 0, -2.0]], np.float32), np.log(np.full((1, 3), 0.1, np.float32)), np.array([[1, 0, 0, 0]], np.float32),
                      np.array([0.8], np.float32), np.array([[(0.6 - 0.5) / formats.SH_C0] * 3], np.float32), None, False, 0, False)
    s = prepare(c, crop=None, log=lambda *a: None)
    r = SplatRenderer(s, 1, width=64, height=64, fx=100.0, outputs=("rgb", "alpha"))
    out = r.render_host(np.zeros((1, 3)), cam_mat=np.eye(3)[None])          # MuJoCo camera at the origin looking along -z
    var = (100.0 * 0.1 / 2.0) ** 2 + 0.3
    ys, xs = np.mgrid[0:64, 0:64] + 0.5
    g = 0.8 * np.exp(-((xs - 32) ** 2 + (ys - 32) ** 2) / (2 * var))
    a = np.where(g >= 1 / 255, np.minimum(g, 0.99), 0)
    assert np.abs(out["alpha"][0] - a).max() < 2e-3
    assert np.abs(out["rgb"][0][..., 0].astype(float) - (a * 0.6 * 255)).max() <= 1.0


def _spz_bytes(c: GaussianCloud, fb=12, version=3):
    """Encode a cloud as legacy gzip SPZ with the reference packing (load-spz.cc packGaussians)."""
    n = c.n
    fixed = np.round(c.means.astype(np.float64) * (1 << fb)).astype(np.int64).ravel()
    pos = np.stack([(fixed >> 0) & 255, (fixed >> 8) & 255, (fixed >> 16) & 255], 1).astype(np.uint8)
    alpha = np.clip(np.round(c.opacity * 255), 0, 255).astype(np.uint8)
    col = np.clip(np.round(c.sh_dc * (0.15 * 255) + 127.5), 0, 255).astype(np.uint8)
    scl = np.clip(np.round((c.log_scales + 10) * 16), 0, 255).astype(np.uint8)
    rot = np.zeros((n, 4), np.uint8)
    q = c.quats[:, [1, 2, 3, 0]].astype(np.float64)                    # xyzw
    for i in range(n):
        il = int(np.argmax(np.abs(q[i]))); neg = q[i, il] < 0; comp = il
        for j in range(4):
            if j != il:
                mag = int(511 * abs(q[i, j]) / np.sqrt(0.5) + 0.5); comp = (comp << 10) | ((int(q[i, j] < 0) ^ int(neg)) << 9) | mag
        rot[i] = [(comp >> s) & 255 for s in (0, 8, 16, 24)]
    sh = c.sh_rest.reshape(n, -1) if c.sh_rest is not None else np.zeros((n, 0), np.uint8)
    body = struct.pack("<IIIBBBB", 0x5053474E, version, n, c.sh_degree, fb, 0, 0) + pos.tobytes() + alpha.tobytes() + col.tobytes() + \
        scl.tobytes() + rot.tobytes() + sh.tobytes()
    return gzip.compress(body)


def test_spz_ply_splat_decoders(tmp_path):
    c = random_cloud(200, seed=7, deg=3)
    p = tmp_path / "t.spz"; p.write_bytes(_spz_bytes(c))
    d = formats.load(str(p))
    assert d.n == 200 and d.sh_degree == 3 and d.antialiased is False and d.sh_rest_u8
    assert np.abs(d.means - c.means).max() <= 0.5 / 4096 + 1e-6
    assert np.abs(d.opacity - c.opacity).max() <= 0.5 / 255 + 1e-6
    assert np.abs(d.log_scales - c.log_scales).max() <= 1 / 32 + 1e-6
    assert np.abs(d.sh_dc - c.sh_dc).max() <= 0.5 / (0.15 * 255) + 1e-4 or np.abs(c.sh_dc).max() > 3.3
    dots = np.abs((d.quats * c.quats).sum(1))
    assert dots.min() > 0.999
    assert np.array_equal(d.sh_rest, c.sh_rest)
    # PLY round trip (float), then .splat (degree 0)
    formats.save_ply(d, str(tmp_path / "t.ply"))
    e = formats.load(str(tmp_path / "t.ply"))
    assert np.allclose(e.means, d.means) and np.allclose(e.quats, d.quats, atol=1e-6) and np.allclose(e.opacity, d.opacity, atol=1e-6)
    assert np.allclose(e.sh_rest.astype(np.float32), d.sh_rest_float(), atol=1e-3)
    rec = np.zeros(200, np.dtype([("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)]))
    rec["p"] = d.means; rec["s"] = np.exp(d.log_scales)
    rec["c"][:, :3] = np.clip(np.round((0.5 + formats.SH_C0 * d.sh_dc) * 255), 0, 255); rec["c"][:, 3] = np.round(d.opacity * 255)
    rec["r"] = np.clip(np.round(d.quats * 128 + 128), 0, 255)
    rec.tofile(tmp_path / "t.splat")
    f = formats.load(str(tmp_path / "t.splat"))
    assert np.allclose(f.means, d.means) and f.sh_degree == 0 and np.abs((f.quats * d.quats).sum(1)).min() > 0.99


def test_prepare_transform_crop_and_cache(tmp_path, monkeypatch):
    c = random_cloud(500, seed=8)
    c.means[:3] = 500.0                                                   # far floaters
    c.opacity[3:8] = 0.5 / 255                                           # can never reach the 1/255 alpha threshold
    A = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]); t = np.array([0.5, -1.0, 2.0])
    s = prepare(c, A, t, crop="auto", log=lambda *a: None)
    assert s.n == 492 and s.info["dropped_alpha"] == 5 and s.info["dropped_crop"] == 3
    assert np.allclose(np.sort(s.g0[:, 0]), np.sort((c.means[8:] @ A.T + t)[:, 0]), atol=1e-5)
    p = tmp_path / "c.npz"; s.save(str(p)); s2 = PackedSplats.load(str(p))
    assert np.array_equal(s2.g0, s.g0) and np.array_equal(s2.sh, s.sh) and s2.sh_degree == s.sh_degree
    # load_scene: the packed arrays are cached on disk, keyed by file and options
    from metalsim.render.splat import scene as scene_mod
    monkeypatch.setattr(scene_mod, "CACHE_DIR", str(tmp_path / "cache"))
    formats.save_ply(c, str(tmp_path / "c.ply"))
    a = scene_mod.load_scene(str(tmp_path / "c.ply"), A, t, log=lambda *a: None)
    assert len(os.listdir(tmp_path / "cache")) == 1 and a.n == 492
    b = scene_mod.load_scene(str(tmp_path / "c.ply"), A, t, log=lambda *a: None)
    assert np.array_equal(a.g0, b.g0) and np.array_equal(a.g1, b.g1)
    scene_mod.load_scene(str(tmp_path / "c.ply"), log=lambda *a: None)
    assert len(os.listdir(tmp_path / "cache")) == 2                       # another transform: another entry
    if not wp.is_metal_available():
        return
    # rendering the transformed scene from the transformed camera == the original scene from the original camera
    s0 = prepare(c, crop="auto", log=lambda *a: None)
    cam0 = np.array([0.3, -3.5, 0.8]); R0 = look_at(cam0)
    r0 = SplatRenderer(s0, 1, width=W, height=H, hfov_deg=75).render_host(cam0[None], cam_mat=R0[None])
    r1 = SplatRenderer(s, 1, width=W, height=H, hfov_deg=75).render_host((A @ cam0 + t)[None], cam_mat=(A @ R0)[None])
    assert np.abs(r0["rgb"].astype(int) - r1["rgb"].astype(int)).max() <= 2
    assert np.abs(r0["depth"] - r1["depth"]).max() < 1e-3


COMPOSITE_XML = """
<mujoco>
  <visual><headlight ambient="0.6 0.6 0.6" diffuse="0.3 0.3 0.3"/></visual>
  <worldbody>
    <camera name="cam" pos="0 -4 1" quat="0.7071 0.7071 0 0" fovy="50"/>
    <geom name="box" type="box" pos="0.3 -1.5 1.0" size="0.25 0.25 0.25" rgba="0.9 0.1 0.1 1"/>
    <geom name="far" type="sphere" pos="-0.4 3.0 1.0" size="0.4" rgba="0.1 0.2 0.9 1"/>
  </worldbody>
</mujoco>
"""


@needs_metal
def test_mesh_composite_occlusion():
    from metalsim.render.tier0 import Tier0Renderer
    m = mujoco.MjModel.from_xml_string(COMPOSITE_XML); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    c = random_cloud(1500, seed=9, box=1.2)
    c.means[:, 2] += 1.0
    s = prepare(c, crop=None, log=lambda *a: None)
    mesh = Tier0Renderer(m, 1, width=W, height=H, camera="cam", outputs=("rgb", "depth"), shadows=False)
    r = SplatRenderer(s, 1, width=W, height=H, mesh=mesh, outputs=("rgb", "depth", "alpha"))
    pos = d.cam_xpos[0][None].copy(); mat = d.cam_xmat[0].reshape(1, 3, 3).copy()
    out = r.render_host(pos, cam_mat=mat, mesh_datas=[d])
    mo = mesh.render_host([d], cam_pos=pos, cam_quat=np.array([[0.7071, 0.7071, 0, 0]]))
    rr = ref.render(s, pos[0], mat[0], r.intrinsics, W, H, mesh_rgb=mo["rgb"][0], mesh_depth=mo["depth"][0])
    lv = _levels(out["rgb"][0], rr["rgb"])
    assert lv.max() <= 3 and (lv > 1).mean() < 5e-3
    box = (mo["depth"][0] > 0) & (mo["depth"][0] < 3.0)                   # the near box occludes the splat cloud
    assert box.sum() > 50
    assert (np.abs(out["depth"][0][box] - mo["depth"][0][box]) < 1e-3).mean() > 0.95
    far = (mo["depth"][0] > 5.0)                                          # the far sphere is hidden behind dense splats in places
    assert far.sum() > 10 and (out["depth"][0][far] < mo["depth"][0][far] - 0.1).any()


GPU_XML = """
<mujoco>
  <option gravity="0 0 -30"/>
  <worldbody>
    <body name="rig" pos="0 -4 1.2"><freejoint/><geom type="sphere" size="0.05" contype="0" conaffinity="0" group="5"/>
      <camera name="cam" pos="0 0 0" quat="0.7071 0.7071 0 0" fovy="60"/></body>
    <geom name="box" type="box" pos="0.3 -2.0 1.0" size="0.2 0.2 0.2" rgba="0.9 0.1 0.1 1"/>
  </worldbody>
</mujoco>
"""


@needs_metal
def test_gpu_path_matches_host_path_and_composites():
    """BatchSim -> splat (+ tier0 mesh composite) render ordered by events, no host sync; equals the host path at the
    same camera poses."""
    import torch
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.interop import warp_metal as wm
    from metalsim.render.tier0 import Tier0Renderer
    model = mujoco.MjModel.from_xml_string(GPU_XML)
    n = 5
    sim = BatchSim(model, n, options=BatchSimOptions(substeps=2))
    c = random_cloud(1500, seed=10, box=1.2); c.means[:, 2] += 1.0
    s = prepare(c, crop=None, log=lambda *a: None)
    mesh = Tier0Renderer(model, n, width=W, height=H, camera="cam", outputs=("rgb", "depth"), shadows=False)
    r = SplatRenderer(s, n, width=W, height=H, mesh=mesh, outputs=("rgb", "depth"))
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    assert (r.cam_spec_i[:, 24] == cam).all()                              # the GPU path follows the mesh renderer's camera
    sim.synchronize()
    c0 = wm.counters()
    means = []
    for k in range(25):
        vs = sim.step()
        vr = r.render(sim, vs)
        r.wait_sim_after_render(sim, vr)
        r.after(vr)
        means.append(r.out.rgb.float().mean())
    torch.mps.synchronize()
    assert (wm.counters() - c0).syncs == 0
    m = torch.stack(means).cpu().numpy()
    assert np.all(m > 0) and m.std() > 0
    sim.synchronize(); r.ctx.synchronize()
    gpu = {k: getattr(r.out, k).cpu().numpy().copy() for k in ("rgb", "depth")}
    pos = sim.d.cam_xpos.numpy()[:, 0].copy(); mat = sim.d.cam_xmat.numpy()[:, 0].reshape(n, 3, 3).copy()
    datas = []
    for i in range(n):
        d = mujoco.MjData(model); d.qpos[:] = sim.d.qpos.numpy()[i]; mujoco.mj_forward(model, d); datas.append(d)
    host = r.render_host(pos, cam_mat=mat, mesh_datas=datas)
    assert np.abs(host["rgb"].astype(int) - gpu["rgb"].astype(int)).max() <= 1
    assert np.abs(host["depth"] - gpu["depth"]).max() < 1e-4
    # after a host-path call the GPU path renders from the sim's camera again (same state -> same image)
    r.after(r.render(sim, sim.event.value)); torch.mps.synchronize()
    assert (r.cam_spec_i[:, 24] == cam).all()
    assert np.abs(r.out.rgb.cpu().numpy().astype(int) - gpu["rgb"].astype(int)).max() <= 1


@needs_metal
def test_blend_variants_bit_identical():
    s = prepare(random_cloud(seed=11), crop=None, log=lambda *a: None)
    pos = np.array(CAMS); mats = np.array([look_at(p) for p in CAMS])
    o1 = SplatRenderer(s, 4, width=W, height=H, hfov_deg=75, outputs=("rgb", "depth", "depth_expected", "alpha"), blend="1px").render_host(pos, cam_mat=mats)
    o2 = SplatRenderer(s, 4, width=W, height=H, hfov_deg=75, outputs=("rgb", "depth", "depth_expected", "alpha"), blend="2px").render_host(pos, cam_mat=mats)
    for k in o1:
        assert np.array_equal(o1[k], o2[k]), k


def _compressed_ply(c: GaussianCloud, path):
    """Encode as the SuperSplat compressed .ply (inverse of formats.load_ply_compressed / PlayCanvas packing)."""
    n = c.n; nc = (n + 255) // 256
    pad = lambda a: np.concatenate([a, np.repeat(a[-1:], nc * 256 - n, 0)]).reshape(nc, 256, -1)
    P, S = pad(c.means), pad(c.log_scales)
    rgb = np.clip(0.5 + formats.SH_C0 * c.sh_dc, 0, 1); Cc = pad(rgb)
    pmin, pmax, smin, smax, cmin, cmax = P.min(1), P.max(1), S.min(1), S.max(1), Cc.min(1), Cc.max(1)
    ci = np.arange(n) // 256
    norm = lambda x, lo, hi: np.where(hi > lo, (x - lo) / np.maximum(hi - lo, 1e-12), 0.0)
    q11 = lambda t, b: np.round(np.clip(t, 0, 1) * ((1 << b) - 1)).astype(np.uint32)
    def p111011(x, lo, hi):
        t = norm(x, lo[ci], hi[ci]); return (q11(t[:, 0], 11) << 21) | (q11(t[:, 1], 10) << 11) | q11(t[:, 2], 11)
    q = c.quats.astype(np.float64).copy()
    L = np.argmax(np.abs(q), 1); q *= np.sign(q[np.arange(n), L])[:, None]
    rot = L.astype(np.uint32) << 30
    for i in range(n):
        o = [k for k in range(4) if k != L[i]]
        v = [int(np.round(np.clip(q[i, k] / np.sqrt(2) + 0.5, 0, 1) * 1023)) for k in o]
        rot[i] |= np.uint32((v[0] << 20) | (v[1] << 10) | v[2])
    t = norm(rgb, cmin[ci], cmax[ci])
    col = (q11(t[:, 0], 8) << 24) | (q11(t[:, 1], 8) << 16) | (q11(t[:, 2], 8) << 8) | q11(c.opacity, 8)
    rest = c.sh_rest_float().transpose(0, 2, 1).reshape(n, -1)
    shb = np.clip(np.round((rest + 4) * 255 / 8), 0, 255).astype(np.uint8)
    ch = np.concatenate([pmin, pmax, smin, smax, cmin, cmax], 1).astype("<f4")
    names = [f"{a}_{x}" for a in ("min", "max") for x in "xyz"] + [f"{a}_scale_{x}" for a in ("min", "max") for x in "xyz"] + \
            [f"{a}_{x}" for a in ("min", "max") for x in "rgb"]
    # property order must match the column order of ch: min_x..max_z, min_scale.., max_scale.., min_r.., max_r..
    hdr = "ply\nformat binary_little_endian 1.0\n" + f"element chunk {nc}\n" + "".join(f"property float {k}\n" for k in names) + \
          f"element vertex {n}\n" + "".join(f"property uint {k}\n" for k in ("packed_position", "packed_rotation", "packed_scale", "packed_color")) + \
          f"element sh {n}\n" + "".join(f"property uchar f_rest_{i}\n" for i in range(shb.shape[1])) + "end_header\n"
    vtx = np.stack([p111011(c.means, pmin, pmax), rot, p111011(c.log_scales, smin, smax), col], 1).astype("<u4")
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(ch.tobytes()); f.write(vtx.tobytes()); f.write(shb.tobytes())


def test_supersplat_compressed_ply(tmp_path):
    c = random_cloud(700, seed=12, deg=3)
    c.sh_dc = np.clip(c.sh_dc, -1.7, 1.7)                      # colour within [0, 1] (the format clamps)
    p = tmp_path / "c.ply"; _compressed_ply(c, str(p))
    d = formats.load(str(p))
    assert d.n == 700 and d.sh_degree == 3 and d.meta["format"].startswith("ply-compressed")
    ext = c.means.max(0) - c.means.min(0)
    assert (np.abs(d.means - c.means) <= ext / 1023 + 1e-5).all()
    assert np.abs(d.log_scales - c.log_scales).max() <= (c.log_scales.max() - c.log_scales.min()) / 1023 + 1e-5
    assert np.abs((d.quats * c.quats).sum(1)).min() > 0.9995
    assert np.abs(d.opacity - c.opacity).max() <= 0.5 / 255 + 1e-6
    assert np.abs(d.sh_dc - c.sh_dc).max() <= (1 / 255) / formats.SH_C0 + 1e-4
    assert np.abs(d.sh_rest.astype(np.float32) - c.sh_rest_float()).max() <= 4 / 255 + 2e-3


def test_make_transform_and_floor_plane():
    for up, v in (("z", (0, 0, 1.0)), ("y", (0, 1.0, 0)), ("-y", (0, -1.0, 0))):
        A, t = make_transform(up=up)
        assert np.allclose(A @ np.array(v), (0, 0, 1)) and np.isclose(np.linalg.det(A), 1.0) and np.allclose(A @ A.T, np.eye(3))
    A, t = make_transform(up="y", scale=0.01, yaw_deg=90, translate=(1, 2, 3))
    assert np.allclose(A @ np.array([100.0, 0, 0]) + t, (1, 3, 3))       # file +x (1 m at cm units) -> sim +y after the yaw
    assert np.allclose(A @ np.array([0, 100.0, 0]) + t, (1, 2, 4))       # file up -> sim +z
    with pytest.raises(ValueError):
        make_transform(up="x")
    # a plane tilted by a few mrad through (1, 2, 0.5): its points map to z = 0, heights above it are kept to first order
    n = np.array([0.004, -0.003, 1.0]); n /= np.linalg.norm(n); m = np.array([1.0, 2.0, 0.5])
    A, t = floor_plane_transform(n, m)
    rng = np.random.default_rng(0)
    for _ in range(5):
        xy = rng.uniform(-5, 5, 2)
        p = np.array([xy[0], xy[1], m[2] - (n[0] * (xy[0] - m[0]) + n[1] * (xy[1] - m[1])) / n[2]])
        assert abs((A @ p + t)[2]) < 1e-9
        assert abs((A @ (p + np.array([0, 0, 1.3])) + t)[2] - 1.3) < 1e-9


def test_reference_single_gaussian_analytic_cpu():
    """The float64 reference alone (no GPU): one isotropic Gaussian 2 m in front of the camera matches the closed form."""
    c = GaussianCloud(np.array([[0, 0, -2.0]], np.float32), np.log(np.full((1, 3), 0.1, np.float32)), np.array([[1, 0, 0, 0]], np.float32),
                      np.array([0.8], np.float32), np.array([[(0.6 - 0.5) / formats.SH_C0] * 3], np.float32), None, False, 0, False)
    s = prepare(c, crop=None, log=lambda *a: None)
    rr = ref.render(s, np.zeros(3), np.eye(3), (100.0, 100.0, 32.0, 32.0), 64, 64)
    var = (100.0 * 0.1 / 2.0) ** 2 + 0.3
    ys, xs = np.mgrid[0:64, 0:64] + 0.5
    g = 0.8 * np.exp(-((xs - 32) ** 2 + (ys - 32) ** 2) / (2 * var))
    a = np.where(g >= 1 / 255, np.minimum(g, 0.99), 0)
    assert np.abs(rr["alpha"] - a).max() < 2e-3
    assert np.abs(rr["rgb"][..., 0] - a * 0.6).max() < 2e-3
    assert np.allclose(rr["depth"][a > 0.5], 2.0, atol=1e-3) and (rr["depth"][a < 0.4] == 0).all()
