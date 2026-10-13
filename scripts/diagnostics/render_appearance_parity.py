"""Static-geometry appearance vs MuJoCo's renderer, before / after the 2026-10 render fixes (docs/RENDER_COVERAGE.md).

Renders the probe scenes of ``render_coverage_audit.py`` (synthetic MJCF only) with ``mujoco.Renderer`` (the
reference) and with MetalSim tier 0 / tier 1 / tier 2, once with ``appearance="legacy"`` (the tables before the fixes)
and once with the default ``appearance="mujoco"``, and reports per item:

* ``mesh_normals``: luminance std inside a box written as a mesh (MuJoCo 2.9: three flat faces) and the mean absolute
  RGB difference to MuJoCo inside the object.
* ``sdf``: coverage of an SDF geom (``type="sdf"``, plugin torus) in rgb / depth / segmentation.
* ``textures``: luminance correlation with MuJoCo per textured geom (2D checker on sphere / cylinder / capsule / box,
  cube texture on a box, an asymmetric image on a mesh with explicit UVs), which colour is at the top of the mesh, and
  the skybox background (luminance std and correlation with MuJoCo's gradient).
* ``transparency``, ``lights``, ``shadows`` (``--items``): see the probe functions.

    scripts/gpu_run.sh appearance check 15 -- .venv/bin/python scripts/diagnostics/render_appearance_parity.py --out runs/render_appearance
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import render_coverage_audit as rca   # noqa: E402

W, H = rca.W, rca.H


def _mask(seg, g, otype=mujoco.mjtObj.mjOBJ_GEOM):
    return (seg[..., 0] == g) & (seg[..., 1] == int(otype))


def _corr(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _variants(renderers=("tier0", "tier1", "tier2")):
    return [(r, a) for a in ("legacy", "mujoco") for r in renderers]


def mesh_normals():
    m = mujoco.MjModel.from_xml_string(rca.MESHBOX)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    seg = rca.ref_render(m, d, seg=True)
    ref = rca.ref_render(m, d).astype(np.float32)
    res = {"probe": "mesh normals: box written as a mesh vs primitive box (luminance std inside; MAD = mean |RGB - MuJoCo|)"}
    imgs = {f"{r}/{a}": rca.ours(m, d, r, appearance=a)["rgb"].astype(np.float32) for r, a in _variants()}
    for name in ("mesh_box", "prim_box"):
        k = _mask(seg, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name))
        row = {"mujoco_std": round(float(ref.mean(-1)[k].std()), 1)}
        for key, img in imgs.items():
            row[key] = {"std": round(float(img.mean(-1)[k].std()), 1), "mad": round(float(np.abs(img[k] - ref[k]).mean()), 1)}
        res[name] = row
    return res, (ref.astype(np.uint8), imgs)


def sdf():
    m = mujoco.MjModel.from_xml_string(rca.GEOMS)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    seg = rca.ref_render(m, d, seg=True); rdep = rca.ref_render(m, d, depth=True)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sdf")
    k = _mask(seg, g)
    res = {"probe": "SDF geom (plugin torus) coverage of MuJoCo's silhouette", "px": int(k.sum())}
    for r, a in _variants(("tier0", "tier2")):
        o = rca.ours(m, d, r, appearance=a)
        res[f"{r}/{a}"] = rca.coverage(k, o, rdep, g, o.get("geoms"))
    rt = rca.RayTracer(m, 1, max_range=100.0)
    cam = rt.make_depth_camera("cam", W, H)
    res["raydepth/mujoco"] = rca.coverage(k, {"depth": rt.trace_host([d], [cam])["ray_depth"][0]}, rdep, g)
    return res


def textures(tmpdir):
    xml = rca.texture_xml(tmpdir)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    ref = rca.ref_render(m, d).astype(np.float32)
    seg = rca.ref_render(m, d, seg=True)
    outs = {f"{r}/{a}": rca.ours(m, d, r, appearance=a)["rgb"].astype(np.float32) for r, a in _variants(("tier0", "tier2"))}
    res = {"probe": "textures: luminance correlation with MuJoCo inside each geom; skybox background", "objects": {}}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        k = _mask(seg, g)
        row = {"px": int(k.sum()), "mujoco_lum_std": round(float(ref.mean(-1)[k].std()), 1)}
        for key, img in outs.items():
            row[key] = {"corr": round(_corr(ref.mean(-1)[k], img.mean(-1)[k]), 3),
                        "mad": round(float(np.abs(img[k] - ref[k]).mean()), 1), "lum_std": round(float(img.mean(-1)[k].std()), 1)}
        res["objects"][name] = row
    bgm = seg[..., 0] < 0
    res["skybox_background"] = {"mujoco_lum_std": round(float(ref.mean(-1)[bgm].std()), 1)}
    for key, img in outs.items():
        res["skybox_background"][key] = {"lum_std": round(float(img.mean(-1)[bgm].std()), 1),
                                         "corr": round(_corr(ref.mean(-1)[bgm], img.mean(-1)[bgm]), 3),
                                         "mad": round(float(np.abs(img[bgm] - ref[bgm]).mean()), 1)}
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tex_mesh_uv")
    k = _mask(seg, g)
    ys = np.nonzero(k.any(1))[0]
    top, bot = ys[: max(2, len(ys) // 4)], ys[-max(2, len(ys) // 4):]

    def dom(img, rows):
        sub = img[rows][k[rows]].astype(np.float32).mean(0)
        return "red" if sub[0] > sub[2] else "blue"
    res["mesh_uv_orientation"] = {"mujoco": {"top": dom(ref, top), "bottom": dom(ref, bot)},
                                  **{key: {"top": dom(img, top), "bottom": dom(img, bot)} for key, img in outs.items()}}
    return res, (ref.astype(np.uint8), outs)


def transparency():
    """Red box behind 25 %-alpha blue glass (rca.TRANSPARENT): mean RGB and MAD vs MuJoCo where the box is behind the
    glass, and the annotations there (fraction segmented as glass, median depth; MuJoCo's depth stops at the box)."""
    m = mujoco.MjModel.from_xml_string(rca.TRANSPARENT)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    ref = rca.ref_render(m, d).astype(np.float32)
    seg_ref = rca.ref_render(m, d, seg=True)
    dep_ref = rca.ref_render(m, d, depth=True)
    m2 = mujoco.MjModel.from_xml_string(rca.TRANSPARENT.replace('rgba="0.1 0.1 0.95 0.25"', 'rgba="0.1 0.1 0.95 0"'))
    d2 = mujoco.MjData(m2); mujoco.mj_forward(m2, d2)
    back, glass = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("back", "glass"))
    k = _mask(rca.ref_render(m2, d2, seg=True), back)
    kg = _mask(seg_ref, glass) & ~k                       # glass over the background only
    res = {"probe": "transparency: red box behind 25 % blue glass", "px": int(k.sum()),
           "mujoco": {"mean_rgb": [round(float(x)) for x in ref[k].mean(0)], "seg_is_glass": round(float((seg_ref[..., 0][k] == glass).mean()), 3),
                      "box_without_glass_rgb": [round(float(x)) for x in rca.ref_render(m2, d2).astype(np.float32)[k].mean(0)],
                      "depth_median_m": round(float(np.median(dep_ref[k])), 3)}}
    # what the glass changes: mean RGB with the glass minus without it (separates blending from the box's shading)
    ref_nog = rca.ref_render(m2, d2).astype(np.float32)[k].mean(0)
    res["mujoco"]["glass_delta_rgb"] = [round(float(x)) for x in ref[k].mean(0) - ref_nog]
    nog = {r: rca.ours(m2, d2, r)["rgb"].astype(np.float32)[k].mean(0) for r in ("tier0", "tier1", "tier2")}
    variants = [("tier0", "before", dict(transparency=False)), ("tier0", "after", {}), ("tier1", "after", {}),
                ("tier2", "before", dict(transparency=False)), ("tier2", "after", {}),
                ("tier0", "after_ids_behind", dict(transparent_ids="behind")), ("tier2", "after_ids_behind", dict(transparent_ids="behind"))]
    imgs = {}
    for r, tag, kw in variants:
        o = rca.ours(m, d, r, **kw)
        s = o["seg"]
        if r == "tier2":
            s = np.concatenate([[0], np.asarray(o["geoms"]) + 1])[s]
        img = o["rgb"].astype(np.float32)
        imgs[f"{r}/{tag}"] = img
        res[f"{r}/{tag}"] = {"mean_rgb": [round(float(x)) for x in img[k].mean(0)], "mad": round(float(np.abs(img[k] - ref[k]).mean()), 1),
                             "glass_over_bg_mad": round(float(np.abs(img[kg] - ref[kg]).mean()), 1),
                             "glass_delta_rgb": [round(float(x)) for x in img[k].mean(0) - nog[r]],
                             "box_without_glass_rgb": [round(float(x)) for x in nog[r]],
                             "seg_is_glass": round(float((s[k] == glass + 1).mean()), 3),
                             "depth_median_m": round(float(np.median(o["depth"][k])), 3)}
    return res, (ref.astype(np.uint8), imgs)


def lights():
    """Ten spot lights over a floor (rca.LIGHTS10): luminance of the patch under each light. MuJoCo stops at 8."""
    m = mujoco.MjModel.from_xml_string(rca.LIGHTS10)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    fy = H / (2 * np.tan(np.deg2rad(m.cam_fovy[0]) / 2))
    imgs = {"mujoco": rca.ref_render(m, d), "tier0/before": rca.ours(m, d, "tier0", max_lights=8)["rgb"],
            "tier0/after": rca.ours(m, d, "tier0")["rgb"], "tier1/after": rca.ours(m, d, "tier1")["rgb"],
            "tier2/after": rca.ours(m, d, "tier2")["rgb"]}
    res = {"probe": "10 spot lights, patch luminance per light index (MuJoCo lights 8)"}
    for key, img in imgs.items():
        lum = img.astype(np.float32).mean(-1)
        vals = []
        for i in range(10):
            px = int(round(W / 2 + (-2.25 + 0.5 * i) / 6.0 * fy))
            vals.append(round(float(lum[H // 2 - 2:H // 2 + 3, max(px - 2, 0):px + 3].mean()), 1))
        res[key] = vals
    return res


def shadows():
    """Plate over a floor under one spot / point / directional light (rca.SHADOW): floor luminance in the plate's
    shadow footprint over the ring outside it (1 = no shadow), and the full-image MAD vs MuJoCo."""
    out = {"probe": "shadow ratio (footprint / ring floor luminance; 1 = no shadow) and image MAD vs MuJoCo"}
    lights_ = {"spot": '<light pos="0 0 1.5" dir="0 0 -1" cutoff="50" exponent="1" castshadow="true" diffuse="0.9 0.9 0.9"/>',
               "point": '<light type="point" pos="0 0 1.5" castshadow="true" diffuse="0.9 0.9 0.9"/>',
               "directional": '<light directional="true" pos="0 0 3" dir="0 0 -1" castshadow="true" diffuse="0.9 0.9 0.9"/>',
               "spot_plus_dir": '<light directional="true" pos="0 0 3" dir="0.3 0 -1" castshadow="true" diffuse="0.4 0.4 0.4"/>'
                                '<light pos="0 0 1.5" dir="0 0 -1" cutoff="50" exponent="1" castshadow="true" diffuse="0.6 0.6 0.6"/>'}
    sheets = []
    for kind, l in lights_.items():
        m = mujoco.MjModel.from_xml_string(rca.SHADOW.replace("{LIGHT}", l))
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        seg = rca.ref_render(m, d, seg=True)
        floor = _mask(seg, 0)
        dep = rca.ref_render(m, d, depth=True)
        fy = H / (2 * np.tan(np.deg2rad(m.cam_fovy[0]) / 2))
        yy, xx = np.mgrid[0:H, 0:W]
        dirs = np.stack([(xx + 0.5 - W / 2) / fy, -(yy + 0.5 - H / 2) / fy, -np.ones_like(xx, float)], -1)
        P = d.cam_xpos[0] + (dirs * dep[..., None]) @ d.cam_xmat[0].reshape(3, 3).T
        inside = floor & (np.abs(P[..., 0]) < 0.1) & (np.abs(P[..., 1]) < 0.1)
        rr = np.hypot(P[..., 0], P[..., 1])
        ring = floor & (rr > 0.3) & (rr < 0.4)
        ref = rca.ref_render(m, d)
        imgs = {"mujoco": ref, "tier0/before": rca.ours(m, d, "tier0", shadow_lights="caster")["rgb"],
                "tier0/after": rca.ours(m, d, "tier0")["rgb"], "tier1/before": rca.ours(m, d, "tier1", shadow_lights="caster")["rgb"],
                "tier1/after": rca.ours(m, d, "tier1")["rgb"], "tier2/after": rca.ours(m, d, "tier2")["rgb"]}
        row = {}
        for key, img in imgs.items():
            lum = img.astype(np.float32).mean(-1)
            row[key] = {"ratio": round(float(lum[inside].mean() / max(lum[ring].mean(), 1e-3)), 3),
                        "mad_vs_mujoco": round(float(np.abs(img.astype(np.float32) - ref.astype(np.float32)).mean()), 1)}
        out[kind] = row
        sheets.append([imgs["mujoco"], imgs["tier0/before"], imgs["tier0/after"], imgs["tier1/after"], imgs["tier2/after"]])
    return out, sheets


def sheet(path, rows):
    from PIL import Image
    Image.fromarray(np.concatenate([np.concatenate([np.clip(t, 0, 255).astype(np.uint8) for t in row], 1) for row in rows], 0)).save(path)


ITEMS = ("mesh_normals", "sdf", "textures", "transparency", "lights", "shadows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/render_appearance")
    ap.add_argument("--items", default=",".join(ITEMS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = a.items.split(",")
    results = []
    if "mesh_normals" in items:
        r, (ref, imgs) = mesh_normals(); results.append(r); print(json.dumps(r), flush=True)
        sheet(os.path.join(a.out, "mesh_normals.png"), [[ref, imgs["tier0/legacy"], imgs["tier0/mujoco"], imgs["tier2/mujoco"]]])
    if "sdf" in items:
        r = sdf(); results.append(r); print(json.dumps(r), flush=True)
    if "textures" in items:
        r, (ref, imgs) = textures(a.out); results.append(r); print(json.dumps(r), flush=True)
        sheet(os.path.join(a.out, "textures.png"), [[ref, imgs["tier0/legacy"]], [imgs["tier0/mujoco"], imgs["tier2/mujoco"]]])
    if "transparency" in items:
        r, (ref, imgs) = transparency(); results.append(r); print(json.dumps(r), flush=True)
        sheet(os.path.join(a.out, "transparency.png"), [[ref, imgs["tier0/before"], imgs["tier0/after"], imgs["tier1/after"], imgs["tier2/after"]]])
    if "lights" in items:
        r = lights(); results.append(r); print(json.dumps(r), flush=True)
    if "shadows" in items:
        r, rows = shadows(); results.append(r); print(json.dumps(r), flush=True)
        sheet(os.path.join(a.out, "shadows.png"), rows)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
