"""USD import visual coverage (docs/RENDER_COVERAGE.md 1d): area lights, GeomSubset per-face materials, mesh UVs.

Builds the synthetic stages of tests/test_scene_usd.py, imports them with ``metalsim.scene.usd_to_mjcf`` (or, with
``--importer FILE``, another copy of the importer, e.g. the previous revision for a before/after), and measures:

* area lights: lights imported, and the irradiance a MuJoCo light gives (renderer convention) over the exact area-light
  irradiance (quadrature) on axis at 1, 3 and 10 light sizes and 30 deg off axis at 10 sizes;
* GeomSubset: per face region (top red, sides blue, bottom green), the fraction of pixels whose dominant colour is the
  subset's, in MuJoCo's renderer and tier 0 (regions from MuJoCo's segmentation of the fixed import);
* UVs (vertex / faceVarying / indexed faceVarying st on a quad mesh): luminance std inside the plate and the fraction of
  interior pixels showing the texture's colour class there (right way up), in MuJoCo's renderer and tier 0.

    scripts/gpu_run.sh usd_import check 5 -- .venv/bin/python scripts/diagnostics/usd_import_visuals.py [--importer F]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_scene_usd as T   # noqa: E402  (stage builders and analytic helpers)


def importer(path):
    if path is None:
        from metalsim.scene import usd_to_mjcf
        return usd_to_mjcf
    spec = importlib.util.spec_from_file_location("usd_to_mjcf_alt", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def mj_render(m, d, cam, W, H, seg=False):
    r = mujoco.Renderer(m, H, W)
    if seg:
        r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam); o = r.render().copy(); r.close()
    return o


TIER0 = False


def t0_render(m, d, cam, W, H):
    if not TIER0:
        return None
    try:
        from metalsim.render.tier0 import Tier0Renderer
        r = Tier0Renderer(m, 1, width=W, height=H, camera=cam, outputs=("rgb",))
        return r.render_host([d])["rgb"][0]
    except Exception as e:      # no Metal device: MuJoCo numbers only
        return None


def lights(imp, tmp):
    specs = T.area_light_stage(os.path.join(tmp, "lights.usda"))
    try:
        m = imp.load_usd(os.path.join(tmp, "lights.usda")).compile()
    except Exception as e:
        return {"error": str(e)}
    out = {"nlight": int(m.nlight)}
    for k, s in specs.items():
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, k)
        if i < 0:
            out[k] = "dropped"; continue
        size = T.area_light_size(s); row = {}
        for n_size, off in ((1, 0), (3, 0), (10, 0), (10, 30)):
            p = s["c"] + n_size * size * np.array([0.0, np.sin(np.radians(off)), -np.cos(np.radians(off))])
            row[f"{n_size}size_{off}deg"] = round(T.mujoco_light_irradiance(m, i, p) / T.area_light_exact_irradiance(s, p) - 1, 4)
        out[k] = {"rel_err_mujoco_vs_exact": row}
    return out


def subsets(imp, tmp, ref_model=None):
    from metalsim.scene import usd_to_mjcf as cur
    path = os.path.join(tmp, "sub.usda"); T.subset_stage(path)
    W = H = 128
    mref = cur.load_usd(path).compile(); dref = mujoco.MjData(mref); mujoco.mj_forward(mref, dref)
    seg = mj_render(mref, dref, "cam", W, H, seg=True)
    regions = {}
    for name, cls in (("box_top", 0), ("box_sides", 1)):
        g = mujoco.mj_name2id(mref, mujoco.mjtObj.mjOBJ_GEOM, name)
        regions[name] = ((seg[..., 0] == g) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)), cls)
    m = imp.load_usd(path).compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    out = {"ngeom": int(m.ngeom)}
    for rname, img in (("mujoco", mj_render(m, d, "cam", W, H)), ("tier0", t0_render(m, d, "cam", W, H))):
        if img is None:
            continue
        cls = T.pixel_class(img)
        out[rname] = {k: {"px": int(mk.sum()), "right_colour": round(float((cls[mk] == c).mean()), 3)} for k, (mk, c) in regions.items()}
    return out


def uvs(imp, tmp):
    img = T.write_asym_texture(os.path.join(tmp, "asym.png"))
    out = {}
    W = H = 96
    for interp, indexed, tris in (("vertex", False, False), ("faceVarying", False, False), ("faceVarying", True, False), ("faceVarying", False, True)):
        path = os.path.join(tmp, f"plate_{interp}_{indexed}_{tris}.usda")
        T.uv_plate_stage(path, os.path.join(tmp, "asym.png"), interp, indexed, tris)
        m = imp.load_usd(path).compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        exp = T.plate_expected_class(m, d, 0, W, H, img); inner = T.plate_interior(exp)
        row = {"texcoords": bool(m.nmesh and m.mesh_texcoordadr[0] >= 0)}
        for rname, rgb in (("mujoco", mj_render(m, d, "cam", W, H)), ("tier0", t0_render(m, d, "cam", W, H))):
            if rgb is None:
                continue
            chroma = rgb.astype(np.float32)[..., 0] - rgb.astype(np.float32)[..., 2]   # the three colours share one luminance
            row[rname] = {"red_minus_blue_std": round(float(chroma[exp >= 0].std()), 1),
                          "texture_match": round(float((T.pixel_class(rgb)[inner] == exp[inner]).mean()), 3),
                          "texture_match_v_flipped": round(float((T.pixel_class(rgb)[inner] == np.flipud(exp)[inner]).mean()), 3)}
        out[f"{interp}{'_indexed' if indexed else ''}{'_triangles' if tris else '_quads'}"] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--importer", default=None, help="path to an alternative usd_to_mjcf.py (e.g. the previous revision)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tier0", action="store_true", help="also render with tier 0 (needs the GPU: run through gpu_run.sh)")
    a = ap.parse_args()
    global TIER0
    TIER0 = a.tier0
    imp = importer(a.importer)
    with tempfile.TemporaryDirectory() as tmp:
        res = {"importer": a.importer or "current", "area_lights": lights(imp, tmp), "geomsubset": subsets(imp, tmp),
               "uvs": uvs(imp, tmp)}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
