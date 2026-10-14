"""NVIDIA NuRec (.nurec) splat loader and the 3DGUT kernel (SPLAT.md §7d): the msgpack decoder, a synthetic .nurec round trip
(layout, activations, SH order, degree, precision 16/32), load_scene on a prepared .npz, the renderer's eps2d hint, and the
float64 3DGUT reference (ray-max response) against the EWA reference on configurations where they must agree / differ."""
import gzip
import struct

import numpy as np
import pytest
import warp as wp

from metalsim.render.splat import formats
from metalsim.render.splat import reference as ref
from metalsim.render.splat.formats import GaussianCloud, load_nurec, msgpack_tree
from metalsim.render.splat.scene import load_scene, prepare


@pytest.fixture(autouse=True)
def _private_cache(tmp_path, monkeypatch):
    """The .nurec loader decompresses into METALSIM_SPLAT_CACHE: keep tests out of the user's cache."""
    monkeypatch.setenv("METALSIM_SPLAT_CACHE", str(tmp_path / "splat_cache"))


# ---- a minimal msgpack encoder (test-only; the loader has only a decoder)
def pack(o) -> bytes:
    if o is None: return b"\xc0"
    if o is True: return b"\xc3"
    if o is False: return b"\xc2"
    if isinstance(o, int):
        if 0 <= o < 128: return bytes([o])
        if -32 <= o < 0: return struct.pack("b", o)
        if 0 <= o < 1 << 16: return b"\xcd" + struct.pack(">H", o)
        if -(1 << 31) <= o < 1 << 31: return b"\xd2" + struct.pack(">i", o)
        return b"\xcf" + struct.pack(">Q", o)
    if isinstance(o, float): return b"\xcb" + struct.pack(">d", o)
    if isinstance(o, str):
        b = o.encode(); return (bytes([0xA0 | len(b)]) if len(b) < 32 else b"\xd9" + bytes([len(b)])) + b
    if isinstance(o, (bytes, bytearray)): return b"\xc6" + struct.pack(">I", len(o)) + bytes(o)
    if isinstance(o, (list, tuple)): return (bytes([0x90 | len(o)]) if len(o) < 16 else b"\xdc" + struct.pack(">H", len(o))) + b"".join(pack(v) for v in o)
    if isinstance(o, dict):
        return (bytes([0x80 | len(o)]) if len(o) < 16 else b"\xde" + struct.pack(">H", len(o))) + b"".join(pack(k) + pack(v) for k, v in o.items())
    raise TypeError(type(o))


def test_msgpack_tree_types():
    doc = {"a": [1, -5, 300, -70000, 1 << 40, 2.5, None, True, False], "s" * 40: "x" * 40, "b": b"\x01\x02\x03", "nested": {"k": [[], {}]}}
    t = msgpack_tree(pack(doc))
    assert t["a"] == [1, -5, 300, -70000, 1 << 40, 2.5, None, True, False]
    assert t["s" * 40] == "x" * 40 and t["nested"] == {"k": [[], {}]}
    blob = t["b"]; buf = pack(doc)
    assert buf[blob.off:blob.off + blob.n] == b"\x01\x02\x03"
    # float32 (0xca) and uint8 (0xcc) / int8 (0xd0) forms
    assert msgpack_tree(b"\x93\xca" + struct.pack(">f", 1.5) + b"\xcc\xc8\xd0\x80") == [1.5, 200, -128]
    with pytest.raises(ValueError):
        msgpack_tree(pack([1]) + b"\x00")                    # trailing bytes


def write_nurec(path, n=500, deg=3, precision=16, seed=0):
    rng = np.random.default_rng(seed)
    dt = {16: np.float16, 32: np.float32}[precision]
    pos = rng.uniform(-3, 3, (n, 3)); q = rng.normal(size=(n, 4)); sc = np.log(rng.uniform(0.01, 0.2, (n, 3)))
    dens = rng.normal(0, 2, (n, 1)); alb = rng.normal(0, 1, (n, 3)); K = (deg + 1) ** 2 - 1
    spec = rng.normal(0, 0.2, (n, K, 3))                              # coefficient-major, channel-last
    P = ".gaussians_nodes.gaussians."
    sd = {"._extra_state": {"obj_track_ids": {"gaussians": []}}}
    for k, v in (("positions", pos), ("rotations", q), ("scales", sc), ("densities", dens), ("extra_signal", np.zeros((n, 0))),
                 ("features_albedo", alb), ("features_specular", spec.reshape(n, 3 * K))):
        sd[P + k] = np.ascontiguousarray(v, dt).tobytes(); sd[P + k + ".shape"] = list(np.shape(v))
    sd[P + "n_active_features"] = np.array([deg], "<i8").tobytes(); sd[P + "n_active_features.shape"] = []
    cfg = {"layers": {"gaussians": {"name": "sh-gaussians", "density_activation": "sigmoid", "scale_activation": "exp",
                                    "rotation_activation": "normalize", "precision": precision,
                                    "particle": {"density_kernel_planar": False, "density_kernel_degree": 2, "density_kernel_min_response": 0.0113,
                                                 "radiance_sph_degree": 3}, "transmittance_threshold": 0.0001}},
           "renderer": {"name": "3dgut-nrend", "global_z_order": True, "render": {"mode": "kbuffer"}}}
    with gzip.open(path, "wb") as f:
        f.write(pack({"nre_data": {"version": "0.2.576", "model": "nre", "config": cfg, "state_dict": sd}}))
    return dict(pos=pos.astype(dt), q=q.astype(dt), sc=sc.astype(dt), dens=dens.astype(dt), alb=alb.astype(dt), spec=spec.astype(dt))


@pytest.mark.parametrize("precision,deg", [(16, 3), (32, 1)])
def test_nurec_round_trip(tmp_path, precision, deg):
    p = tmp_path / "scene.nurec"
    src = write_nurec(str(p), deg=deg, precision=precision)
    c = formats.load(str(p))
    assert isinstance(c, GaussianCloud) and c.n == 500 and c.sh_degree == deg and c.antialiased is False
    np.testing.assert_array_equal(c.means, src["pos"].astype(np.float32))
    np.testing.assert_array_equal(c.log_scales, src["sc"].astype(np.float32))
    q = src["q"].astype(np.float32); q /= np.linalg.norm(q, axis=1, keepdims=True)
    np.testing.assert_allclose(c.quats, q, atol=1e-6)                              # wxyz kept as stored, normalised
    np.testing.assert_allclose(c.opacity, 1 / (1 + np.exp(-src["dens"].astype(np.float64)[:, 0])), atol=1e-6)
    np.testing.assert_array_equal(c.sh_dc, src["alb"].astype(np.float32))
    np.testing.assert_array_equal(np.asarray(c.sh_rest, np.float32), src["spec"][:, :(deg + 1) ** 2 - 1].astype(np.float32))
    assert c.meta["format"] == "nurec" and c.meta["eps2d_hint"] == 0.0 and abs(c.meta["min_response"] - 0.0113) < 1e-12
    assert c.meta["renderer"] == "3dgut-nrend" and c.meta["precision"] == precision


def test_nurec_rejects_unsupported_activation(tmp_path):
    p = tmp_path / "bad.nurec"; write_nurec(str(p), n=10)
    raw = gzip.decompress(p.read_bytes()).replace(b"\xa7sigmoid", b"\xa7softexp")
    p.write_bytes(gzip.compress(raw))
    with pytest.raises(ValueError, match="density_activation"):
        load_nurec(str(p))


def test_load_scene_prepared_npz_and_eps2d_hint(tmp_path):
    p = tmp_path / "scene.nurec"; write_nurec(str(p))
    s = prepare(load_nurec(str(p)), crop=None, log=lambda *a: None)
    s.save(str(tmp_path / "packed.npz"))
    s2 = load_scene(str(tmp_path / "packed.npz"), log=lambda *a: None)
    np.testing.assert_array_equal(s.g0, s2.g0); assert s2.info["meta"]["eps2d_hint"] == 0.0
    with pytest.raises(ValueError):
        load_scene(str(tmp_path / "packed.npz"), A=2 * np.eye(3), log=lambda *a: None)
    if wp.is_metal_available():
        from metalsim.render.splat.renderer import SplatRenderer
        assert SplatRenderer(s2, 1, width=32, height=32).eps2d == 0.0              # NuRec -> no dilation by default
        assert SplatRenderer(s2, 1, width=32, height=32, eps2d=0.3).eps2d == 0.3


def _one(scale, dist, opacity=0.8):
    c = GaussianCloud(np.array([[0, 0, 0]], np.float32), np.log(np.array([scale], np.float32)), np.array([[1, 0, 0, 0]], np.float32),
                      np.array([opacity], np.float32), np.zeros((1, 3), np.float32))
    s = prepare(c, crop=None, log=lambda *a: None)
    pos = np.array([0.0, -dist, 0.0]); mat = np.stack([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]], 1)   # x right, y up = +z, z back = -y: looks along +y
    return s, pos, mat


def test_3dgut_reference_kernel():
    W = H = 41; K = (40.0, 40.0, W / 2, H / 2)
    # small, far, isotropic Gaussian: the ray-max response equals the EWA projection (eps2d 0) to first order
    s, pos, mat = _one([0.05, 0.05, 0.05], 5.0)
    g = ref.render(s, pos, mat, K, W, H, kernel="3dgut", min_response=0.0)
    e = ref.render(s, pos, mat, K, W, H, eps2d=0.0)
    assert np.abs(g["alpha"] - e["alpha"]).max() < 2e-3 and g["alpha"].max() > 0.7
    # min_response cuts the tail at ~3 sigma: no contribution beyond 3.1 sigma from the centre
    g3 = ref.render(s, pos, mat, K, W, H, kernel="3dgut", min_response=0.0113)
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    r_sig = np.hypot(xx - W / 2, yy - H / 2) / (40.0 * 0.05 / 5.0)
    assert g3["alpha"][r_sig > 3.1].max() == 0 and g3["alpha"][r_sig < 2.5].min() > 0
    # big Gaussian close to the camera, off-axis: EWA's linearisation differs from the exact 3D response (what SPLAT.md §7d measures)
    s, pos, mat = _one([0.6, 0.6, 0.6], 1.2)
    pos = pos + np.array([0.5, 0.0, 0.0])
    g = ref.render(s, pos, mat, K, W, H, kernel="3dgut"); e = ref.render(s, pos, mat, K, W, H, eps2d=0.0)
    assert np.abs(g["alpha"] - e["alpha"]).max() > 1e-2
