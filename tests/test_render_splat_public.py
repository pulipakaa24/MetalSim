"""Splat renderer on real, public scenes fetched by ``scripts/fetch_splat_scenes.sh`` (SPLAT.md §7b). Skipped with the
fetch command when the data is absent; nothing here is committed data.

* InteriorGS 0204_840158 (972,596 splats, SuperSplat compressed .ply, SH 3; non-commercial, not redistributable): the
  decoder on the real file, the robust crop, Metal vs the independent float64 reference at robot-camera views from the
  room polygons, depth coverage inside a closed room, determinism.
* dxgl modern_arm_chair (CC0, standard 3DGS .ply from nerfstudio): the plain-PLY path on a real trained object.
* optional: NVIDIA NuRec zh_lounge (CC-BY-4.0, ``fetch_splat_scenes.sh nurec_lounge``): the .nurec decoder on a released file.
"""
import os

import numpy as np
import pytest
import warp as wp

from metalsim.render.splat import load, prepare
from metalsim.render.splat import reference as ref
from metalsim.render.splat.views import DXGL_REF, INTERIORGS_REF, NUREC_LOUNGE, orbit_views, room_views

FETCH = "scripts/fetch_splat_scenes.sh"
IGS_PLY = os.path.join(INTERIORGS_REF, "3dgs_compressed.ply")
need_igs = pytest.mark.skipif(not os.path.exists(IGS_PLY), reason=f"InteriorGS reference scene absent: run {FETCH} interiorgs")
need_dxgl = pytest.mark.skipif(not os.path.exists(DXGL_REF), reason=f"dxgl CC0 object splat absent: run {FETCH} dxgl")
needs_metal = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
QUIET = dict(log=lambda *a: None)


def _psnr(a_u8, b_float):
    b = np.clip(b_float, 0, 1) * 255.0
    mse = np.mean((a_u8.astype(np.float64) - b) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


@pytest.fixture(scope="module")
def igs():
    c = load(IGS_PLY)
    return c, prepare(c, **QUIET)


@need_igs
def test_interiorgs_decode_and_prepare(igs):
    c, s = igs
    assert c.n == 972_596 and c.sh_degree == 3 and c.meta["format"].startswith("ply-compressed")
    assert np.isfinite(c.means).all() and np.allclose(np.linalg.norm(c.quats, axis=1), 1, atol=1e-5)
    assert 0 <= c.opacity.min() and c.opacity.max() <= 1
    # z-up metres, floor at z = 0 (InteriorGS README): the densest layer near the floor is at z = 0 and the ceiling below
    # 6 m (splats below the floor are expected: reflections of a glossy floor are fitted as a mirrored room)
    z = c.means[:, 2]
    h, e = np.histogram(z, bins=20, range=(-1.0, 1.0))
    assert abs(0.5 * (e[h.argmax()] + e[h.argmax() + 1])) < 0.25
    assert 2.0 < np.quantile(z, 0.99) < 6.0
    assert s.n > 0.99 * c.n                                                 # the auto crop keeps the room


@need_igs
@needs_metal
def test_interiorgs_metal_matches_reference(igs):
    from metalsim.render.splat.renderer import SplatRenderer
    _, s = igs
    pos, mats = room_views(os.path.join(INTERIORGS_REF, "structure.json"))
    pick = np.linspace(0, len(pos) - 1, 4).astype(int)
    W, H = 320, 180
    r = SplatRenderer(s, len(pick), width=W, height=H, hfov_deg=88.9, outputs=("rgb", "depth", "alpha"))
    out = r.render_host(pos[pick], cam_mat=mats[pick])
    again = r.render_host(pos[pick], cam_mat=mats[pick])
    assert np.array_equal(out["rgb"], again["rgb"]) and np.array_equal(out["depth"], again["depth"])
    for k, i in enumerate(pick):
        rr = ref.render(s, pos[i], mats[i], r.intrinsics, W, H)
        lv = np.abs(out["rgb"][k].astype(int) - (np.clip(rr["rgb"], 0, 1) * 255 + 0.5).astype(int)).max(-1)
        psnr = _psnr(out["rgb"][k], rr["rgb"])
        assert psnr > 45 and (lv > 3).mean() < 5e-3, (int(i), psnr, int(lv.max()), (lv > 3).mean())
        assert (out["depth"][k] > 0).mean() > 0.95                           # a closed room: every ray hits something
        both = (out["depth"][k] > 0) & (rr["depth"] > 0)
        assert np.median(np.abs(out["depth"][k] - rr["depth"])[both]) < 1e-3
        assert np.abs(out["alpha"][k] - rr["alpha"]).mean() < 1e-3


@need_dxgl
def test_dxgl_plain_ply_decode():
    c = load(DXGL_REF)
    assert c.meta["format"] == "ply" and c.n > 10_000 and c.sh_degree >= 1
    assert np.isfinite(c.means).all() and np.allclose(np.linalg.norm(c.quats, axis=1), 1, atol=1e-5)


@need_dxgl
@needs_metal
def test_dxgl_metal_matches_reference():
    from metalsim.render.splat.renderer import SplatRenderer
    c = load(DXGL_REF)
    s = prepare(c, **QUIET)
    lo, hi = np.quantile(s.g0[:, :3], [0.02, 0.98], axis=0)
    pos, mats = orbit_views(0.5 * (lo + hi), 1.6 * float(np.linalg.norm(hi - lo)), n=4)
    W, H = 160, 120
    r = SplatRenderer(s, 4, width=W, height=H, hfov_deg=60, outputs=("rgb", "depth", "alpha"))
    out = r.render_host(pos, cam_mat=mats)
    for k in range(4):
        rr = ref.render(s, pos[k], mats[k], r.intrinsics, W, H)
        assert (out["alpha"][k] > 0.5).mean() > 0.02                         # the object is in view
        assert _psnr(out["rgb"][k], rr["rgb"]) > 45


@pytest.mark.skipif(not os.path.exists(NUREC_LOUNGE), reason=f"optional NuRec lounge absent: run {FETCH} nurec_lounge")
def test_nurec_lounge_decode(tmp_path, monkeypatch):
    monkeypatch.setenv("METALSIM_SPLAT_CACHE", str(tmp_path))
    c = load(NUREC_LOUNGE)
    assert c.meta["format"] == "nurec" and c.n > 100_000 and c.meta["eps2d_hint"] == 0.0
    assert np.isfinite(c.means).all() and np.allclose(np.linalg.norm(c.quats, axis=1), 1, atol=1e-3)
    s = prepare(c, **QUIET)
    assert s.n > 0.9 * c.n


def test_fetch_script_explains_missing_login(tmp_path):
    """No Hugging Face login: the fetch script names the exact login step and the terms page (not a bare curl error) and
    downloads nothing. Offline: it exits before any request."""
    import shutil
    import subprocess
    if shutil.which("zsh") is None:
        pytest.skip("zsh not installed")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
    env.update(HF_TOKEN_PATH=str(tmp_path / "no_token"), METALSIM_SPLAT_DATA=str(tmp_path / "data"), TMPDIR=str(tmp_path))
    r = subprocess.run(["zsh", os.path.join(root, FETCH), "interiorgs"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 3, r.stderr
    assert "hf auth login" in r.stderr and "https://huggingface.co/datasets/spatialverse/InteriorGS" in r.stderr
    assert not (tmp_path / "data" / "interiorgs" / "0204_840158" / "3dgs_compressed.ply").exists()
