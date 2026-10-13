"""Render world-extent / per-environment audit (workstream C of docs/RENDER_COVERAGE.md): far plane, depth
precision, infinite ground planes, heightfield sides, per-world model fields, camera models.

Every probe is a synthetic MJCF built here. ``mujoco.Renderer`` (MuJoCo's own OpenGL renderer) is the reference for
what should be visible; for per-world fields the reference is MetalSim itself rendering a model *compiled* with that
world's values (so shading differences between MuJoCo and MetalSim cancel and the probe measures only whether the
per-world value reaches the image). Prints one JSON object per probe and writes them to ``--out``/results.json.

    scripts/gpu_run.sh render_world check 10 -- .venv/bin/python scripts/diagnostics/render_world_audit.py --out runs/render_world

Works on the code before and after the fixes (``far=`` and the other new keywords are passed only when the renderer
accepts them), so the same script gives the before / after numbers.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from metalsim.render.tier0 import Tier0Renderer, SEG_GEOM   # noqa: E402
from metalsim.render.tier2 import Tier2Renderer             # noqa: E402
from metalsim.sensors.raytrace import RayTracer             # noqa: E402

W, H = 160, 120


def _accepts(cls, name):
    return name in inspect.signature(cls.__init__).parameters


def ref_render(m, d, cam="cam", seg=False, depth=False, w=W, h=H):
    r = mujoco.Renderer(m, h, w)
    if seg:
        r.enable_segmentation_rendering()
    if depth:
        r.enable_depth_rendering()
    r.update_scene(d, camera=cam)
    out = r.render().copy()
    r.close()
    return out


def mujoco_far(m):
    return float(m.vis.map.zfar * m.stat.extent)


def ours(m, d, which, cam="cam", w=W, h=H, far=None, **kw):
    if which in ("tier0", "tier1"):
        if far is not None and _accepts(Tier0Renderer, "far"):
            kw["far"] = far
        r = Tier0Renderer(m, 1, width=w, height=h, camera=cam, outputs=("rgb", "depth", "seg"), seg_mode=SEG_GEOM,
                          tier=0 if which == "tier0" else 1, **kw)
        o = r.render_host([d])
        return {"rgb": o["rgb"][0], "depth": o["depth"][0], "seg": o["seg"][0]}
    if which == "tier2":
        r = Tier2Renderer(m, 1, width=w, height=h, camera=cam, spp=16, max_bounces=2, center_sample=True, **kw)
        o = r.render_host([d], passes=2)
        lut = np.concatenate([[0], r.tables.slot_seg_ids()])
        return {"rgb": o["rgb"][0], "depth": o["depth"][0], "seg": lut[o["seg"][0]]}
    if which == "raydepth":
        rt = RayTracer(m, 1, max_range=kw.get("max_range", 100.0))
        c = rt.make_depth_camera(cam, w, h)
        return {"depth": rt.trace_host([d], [c])["ray_depth"][0]}
    raise ValueError(which)


def coverage(mask, o, ref_depth, gid):
    n = int(mask.sum())
    res = {"px": n}
    if n == 0:
        return res
    if "rgb" in o:
        res["rgb"] = round(float((o["rgb"][mask].max(-1) > 3).mean()), 4)
    if "depth" in o:
        dv = o["depth"][mask]
        ok = dv > 0
        res["depth"] = round(float(ok.mean()), 4)
        if ok.any():
            rel = np.abs(dv[ok] - ref_depth[mask][ok]) / ref_depth[mask][ok]
            res["depth_relerr_med"] = float(f"{np.median(rel):.2e}")
    if "seg" in o:
        res["seg"] = round(float((o["seg"][mask] == gid + 1).mean()), 4)
    return res


def geom_mask(seg, g):
    return (seg[..., 0] == g) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))


# ---------------------------------------------------------------------------------------------------------------
# 1. far plane: boxes at 6, 15 and 40 m (MuJoCo far = zfar * extent = 200 * extent)

FAR = """
<mujoco>
  <visual><map zfar="200"/></visual>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0.3 -1"/>
    <camera name="cam" pos="0 0 1.5" xyaxes="1 0 0 0 0.05 1"/>
    <geom name="ground" type="plane" size="0 0 1" rgba="0.5 0.5 0.5 1"/>
    <geom name="box6"  type="box" pos="-1.0 6 1.5"  size="0.3 0.3 0.3" rgba="0.9 0.3 0.3 1"/>
    <geom name="box15" type="box" pos="0 15 1.5"    size="0.8 0.8 0.8" rgba="0.3 0.9 0.3 1"/>
    <geom name="box40" type="box" pos="3 40 2.0"    size="2 2 2"       rgba="0.3 0.3 0.9 1"/>
  </worldbody>
</mujoco>"""


def far_probe():
    m = mujoco.MjModel.from_xml_string(FAR)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    seg, rdep = ref_render(m, d, seg=True), ref_render(m, d, depth=True)
    far = mujoco_far(m)
    outs = {"tier0_default": ours(m, d, "tier0"), "tier0_far_mujoco": ours(m, d, "tier0", far=far),
            "tier1_far_mujoco": ours(m, d, "tier1", far=far), "tier2": ours(m, d, "tier2"), "raydepth": ours(m, d, "raydepth")}
    res = {"probe": "far plane (boxes at 6 / 15 / 40 m; ground = infinite plane)", "mujoco_far_m": round(far, 2),
           "tier0_accepts_far": _accepts(Tier0Renderer, "far"), "objects": {}}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        mask = geom_mask(seg, g)
        res["objects"][name] = {k: coverage(mask, o, rdep, g) for k, o in outs.items()}
    return res, [ref_render(m, d), outs["tier0_default"]["rgb"], outs["tier0_far_mujoco"]["rgb"], outs["tier2"]["rgb"]]


# ---------------------------------------------------------------------------------------------------------------
# 2. depth precision at long range: fronto-parallel walls at 2 / 20 / 200 / 2000 m, far = 1e4 (and 1e6 = Isaac's
#    default clipping range), z-depth error vs the exact value (camera looks along +y, walls are boxes facing it)

PREC = """
<mujoco>
  <visual><map zfar="1e5"/></visual>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0.3 -1"/>
    <camera name="cam" pos="0 0 1" xyaxes="1 0 0 0 0 1" fovy="30"/>
    <geom name="w2"    type="box" pos="-0.55 2.1 1.15"     size="0.5 0.1 0.1"   rgba="0.9 0.3 0.3 1"/>
    <geom name="w20"   type="box" pos="5.5 21 6.5"         size="5 1 1"         rgba="0.3 0.9 0.3 1"/>
    <geom name="w200"  type="box" pos="-55 210 -15"        size="50 10 10"      rgba="0.3 0.3 0.9 1"/>
    <geom name="w2000" type="box" pos="550 2100 -150"      size="500 100 100"   rgba="0.9 0.9 0.3 1"/>
  </worldbody>
</mujoco>"""


def precision_probe():
    m = mujoco.MjModel.from_xml_string(PREC)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    res = {"probe": "depth precision (z-depth of fronto-parallel walls; exact = wall front face distance)", "rows": {}}
    exact = {"w2": 2.0, "w20": 20.0, "w200": 200.0, "w2000": 2000.0}
    for far in (1e4, 1e6):
        if not _accepts(Tier0Renderer, "far"):
            break
        for label, rz in (("reversed_z", True), ("standard_z", False)):
            kw = {"far": far}
            if _accepts(Tier0Renderer, "reverse_z"):
                kw["reverse_z"] = rz
            elif not rz:
                continue
            r = Tier0Renderer(m, 1, width=W, height=H, camera="cam", outputs=("depth", "seg"), seg_mode=SEG_GEOM, **kw)
            o = r.render_host([d])
            row = {}
            for name, z in exact.items():
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
                sel = o["seg"][0] == g + 1
                if sel.sum() == 0:
                    row[name] = None
                    continue
                err = np.abs(o["depth"][0][sel] - z)
                row[name] = {"px": int(sel.sum()), "abs_err_med_m": float(f"{np.median(err):.3g}"), "abs_err_max_m": float(f"{err.max():.3g}"),
                             "rel_err_max": float(f"{err.max() / z:.2e}")}
            res["rows"][f"far={far:g} {label}"] = row
    return res


# ---------------------------------------------------------------------------------------------------------------
# 3. infinite ground: a Menagerie-style textured floor (size 0 0 0.05, checker, texuniform, texrepeat 5 5) seen
#    from a locomotion camera; ground coverage per renderer and texture correlation vs MuJoCo; a lidar ring

GROUND = """
<mujoco>
  <visual><headlight ambient="0.3 0.3 0.3"/><map zfar="40"/></visual>
  <statistic extent="2" center="0 0 0.5"/>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <texture type="2d" name="asym" builtin="gradient" rgb1="1 0.1 0.1" rgb2="0.1 0.1 1" width="64" height="64"/>
    <material name="asym" texture="asym" texuniform="{TEXUNIFORM}" texrepeat="1 1" emission="1" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <camera name="cam" pos="0 0 1.0" xyaxes="1 0 0 0 0.12 0.993"/>
    <camera name="down" pos="0.3 0.2 6" xyaxes="1 0 0 0 1 0" fovy="70"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="{MAT}"/>
    <site name="lidar" pos="0 0 1.0" group="4"/>
  </worldbody>
</mujoco>"""


def _corr(a, b, mask):
    la, lb = a[mask].astype(np.float32).mean(-1), b[mask].astype(np.float32).mean(-1)
    if la.std() < 1 or lb.std() < 1:
        return float("nan")
    return round(float(np.corrcoef(la, lb)[0, 1]), 4)


def ground_probe():
    res = {"probe": "infinite ground plane", "views": {}}
    sheets = []
    for view, mat, texu in (("loco_checker", "groundplane", "true"), ("topdown_gradient_texuniform", "asym", "true"),
                            ("topdown_gradient", "asym", "false")):
        cam = "cam" if view.startswith("loco") else "down"
        m = mujoco.MjModel.from_xml_string(GROUND.replace("{MAT}", mat).replace("{TEXUNIFORM}", texu))
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        seg, rdep, rgb = ref_render(m, d, cam, seg=True), ref_render(m, d, cam, depth=True), ref_render(m, d, cam)
        far = mujoco_far(m)
        outs = {"tier0_default": ours(m, d, "tier0", cam), "tier0_far_mujoco": ours(m, d, "tier0", cam, far=far),
                "tier1_far_mujoco": ours(m, d, "tier1", cam, far=far), "tier2": ours(m, d, "tier2", cam),
                "raydepth": ours(m, d, "raydepth", cam)}
        mask = geom_mask(seg, 0)
        row = {"mujoco_ground_px": int(mask.sum()), "mujoco_far_m": round(far, 1)}
        for k, o in outs.items():
            row[k] = coverage(mask, o, rdep, 0)
            if "rgb" in o:
                row[k]["tex_corr_vs_mujoco"] = _corr(rgb, o["rgb"], mask & (o["seg"] == 1))
                # within 8 m (beyond, the checker aliases differently in each renderer: no mip maps in tier 0)
                row[k]["tex_corr_within_8m"] = _corr(rgb, o["rgb"], mask & (o["seg"] == 1) & (rdep < 8.0))
        res["views"][view] = row
        sheets.append([rgb, outs["tier0_far_mujoco"]["rgb"], outs["tier2"]["rgb"]])
    # lidar: 64 beams at -30 deg (ground at 1.73 m) and 64 at -3.8 deg (ground at 15 m), max range 30 m
    m = mujoco.MjModel.from_xml_string(GROUND.replace("{MAT}", "groundplane").replace("{TEXUNIFORM}", "true"))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    rt = RayTracer(m, 1, max_range=30.0)
    az = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    el = np.concatenate([np.full(64, -np.deg2rad(30.0)), np.full(64, -np.arctan(1 / 15))])
    lid = rt.make_lidar("lidar", np.concatenate([az, az]), el)
    rng = rt.trace_host([d], [lid])["lidar"]["range"][0]
    res["lidar"] = {"returns_at_1.73m": float((np.abs(rng[:64] - 2.0) < 0.01).mean()),
                    "returns_at_15m": float((np.abs(rng[64:] - np.hypot(1, 15)) < 0.05).mean())}
    return res, sheets


# ---------------------------------------------------------------------------------------------------------------
# 4. heightfield sides and base

HFIELD = """
<mujoco>
  <asset>
    <hfield name="hf" nrow="8" ncol="8" size="0.3 0.3 0.15 0.08"
      elevation="0 0 0 0 0 0 0 0  0 1 1 1 1 1 1 0  0 1 2 2 2 2 1 0  0 1 2 3 3 2 1 0  0 1 2 3 3 2 1 0  0 1 2 2 2 2 1 0  0 1 1 1 1 1 1 0  0 0 0 0 0 0 0 0"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0.3 -1"/>
    <camera name="cam" pos="0 -1.4 0.55" xyaxes="1 0 0 0 0.3 0.954"/>
    <camera name="under" pos="0.1 -0.9 -0.4" xyaxes="1 0 0 0 -0.8 0.6"/>
    <geom name="hfield" type="hfield" pos="0 0 0.3" hfield="hf" rgba="0.6 0.6 0.95 1"/>
    <geom name="anchor" type="box" pos="2.5 3 2.5" size="0.01 0.01 0.01" rgba="1 1 1 1"/>
  </worldbody>
</mujoco>"""


def hfield_probe():
    m = mujoco.MjModel.from_xml_string(HFIELD)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    res = {"probe": "heightfield sides and base", "views": {}}
    for cam in ("cam", "under"):
        seg, rdep = ref_render(m, d, cam, seg=True), ref_render(m, d, cam, depth=True)
        mask = geom_mask(seg, 0)
        outs = {r: ours(m, d, r, cam) for r in ("tier0", "tier1", "tier2", "raydepth")}
        outs["tier0_nocull"] = ours(m, d, "tier0", cam, hfield_cull=False)
        res["views"][cam] = {r: coverage(mask, o, rdep, 0) for r, o in outs.items()}
    return res


# ---------------------------------------------------------------------------------------------------------------
# 5. per-world model fields through BatchSim (GPU path): geom_size, geom_rgba, mat_rgba, light_diffuse, light_active

PW = """
<mujoco>
  <visual><headlight ambient="0.15 0.15 0.15" diffuse="0.3 0.3 0.3"/></visual>
  <asset><material name="blue" rgba="0.2 0.4 0.9 1"/></asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 3" dir="0.2 0.3 -1" diffuse="0.5 0.5 0.5" castshadow="false"/>
    <light name="spot" pos="0.0 -0.3 1.6" dir="0 0.2 -1" diffuse="0.6 0.6 0.6" cutoff="60" castshadow="false"/>
    <camera name="cam" pos="0 -1.6 1.1" xyaxes="1 0 0 0 0.55 0.835"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.1" rgba="0.7 0.7 0.7 1"/>
    <geom name="box" type="box" pos="-0.55 0 0.15" size="0.1 0.12 0.15" rgba="0.9 0.3 0.3 1"/>
    <geom name="sphere" type="sphere" pos="-0.2 0 0.15" size="0.12" rgba="0.3 0.9 0.3 1"/>
    <geom name="cylinder" type="cylinder" pos="0.15 0 0.15" size="0.1 0.15" rgba="0.9 0.9 0.3 1"/>
    <geom name="capsule" type="capsule" pos="0.45 0 0.2" size="0.07 0.1" rgba="0.3 0.9 0.9 1"/>
    <geom name="ellipsoid" type="ellipsoid" pos="0.75 0 0.15" size="0.08 0.12 0.15" rgba="0.9 0.3 0.9 1"/>
    <geom name="matbox" type="box" pos="0 0.45 0.1" size="0.25 0.08 0.1" material="blue"/>
  </worldbody>
</mujoco>"""
PW_FIELDS = ("geom_size", "geom_rgba", "mat_rgba", "light_diffuse", "light_active")


def _pw_values(m, n):
    rng = np.random.default_rng(1)
    vals = []
    prims = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, s) for s in ("box", "sphere", "cylinder", "capsule", "ellipsoid", "matbox")]
    for e in range(n):
        gs = m.geom_size.copy(); gr = m.geom_rgba.copy(); mr = m.mat_rgba.copy()
        ld = m.light_diffuse.copy(); la = m.light_active.copy()
        k = (0.6, 1.4, 0.8, 1.2)[e % 4]
        for g in prims:
            gs[g] = m.geom_size[g] * np.array([k, 1.0 / k, k]) if int(m.geom_type[g]) in (6, 4) else m.geom_size[g] * k
        for g in prims[:-1]:
            gr[g, :3] = rng.uniform(0.1, 0.95, 3)
        mr[0, :3] = rng.uniform(0.1, 0.95, 3)
        ld[1] = (0.2 + 0.4 * e, 0.6, 1.2 - 0.3 * e)
        la[1] = e != 3
        vals.append(dict(geom_size=gs, geom_rgba=gr, mat_rgba=mr, light_diffuse=ld, light_active=la))
    return vals


def perworld_probe(n=4):
    import torch
    import warp as wp
    from metalsim.physics.batch import BatchSim, BatchSimOptions
    m = mujoco.MjModel.from_xml_string(PW)
    vals = _pw_values(m, n)
    sim = BatchSim(m, n, options=BatchSimOptions(per_world_fields=PW_FIELDS, njmax=64))
    for f in PW_FIELDS:
        arr = getattr(sim.m, f)
        cur = arr.numpy()
        host = np.stack([v[f] for v in vals]).astype(cur.dtype)
        arr.assign(host.reshape(cur.shape))
    wp.synchronize_device("metal:0")
    sim.forward(); sim.synchronize()
    res = {"probe": "per-world model fields via BatchSim (GPU path) vs the same renderer on a model compiled with that world's values",
           "fields": list(PW_FIELDS), "rows": {}}
    sheet = []
    for tier in ("tier0", "tier2"):
        if tier == "tier0":
            r = Tier0Renderer(m, n, width=W, height=H, camera="cam", outputs=("rgb", "seg"), seg_mode=SEG_GEOM)
        else:
            r = Tier2Renderer(m, n, width=W, height=H, camera="cam", spp=8, max_bounces=1, center_sample=True, seed=3)
        v = r.render(sim, sim.event.value); r.after(v); torch.mps.synchronize()
        rgb = r.out.rgb.cpu().numpy(); seg = r.out.seg.cpu().numpy()
        if tier == "tier2":
            seg = np.concatenate([[0], r.tables.slot_seg_ids()])[seg]
        rows = []
        for e in range(n):
            me = mujoco.MjModel.from_xml_string(PW)
            for f in PW_FIELDS:
                getattr(me, f)[:] = vals[e][f]
            de = mujoco.MjData(me); mujoco.mj_forward(me, de)
            if tier == "tier0":
                re = Tier0Renderer(me, 1, width=W, height=H, camera="cam", outputs=("rgb", "seg"), seg_mode=SEG_GEOM)
                oe = re.render_host([de]); ref_rgb, ref_seg = oe["rgb"][0], oe["seg"][0]
            else:
                re = Tier2Renderer(me, 1, width=W, height=H, camera="cam", spp=8, max_bounces=1, center_sample=True, seed=3)
                oe = re.render_host([de]); ref_rgb = oe["rgb"][0]
                ref_seg = np.concatenate([[0], re.tables.slot_seg_ids()])[oe["seg"][0]]
            mseg = ref_render(me, de, seg=True)
            ious = {}
            for g in range(1, me.ngeom):
                a = seg[e] == g + 1; b = geom_mask(mseg, g)
                ious[mujoco.mj_id2name(me, mujoco.mjtObj.mjOBJ_GEOM, g)] = round(float((a & b).sum() / max((a | b).sum(), 1)), 3)
            both = (seg[e] > 0) & (ref_seg > 0)
            rows.append({"env": e, "silhouette_iou_vs_mujoco": ious,
                         "rgb_mean_abs_diff_vs_compiled": round(float(np.abs(rgb[e].astype(np.float32) - ref_rgb.astype(np.float32))[both].mean()), 2),
                         "floor_lum_diff_vs_compiled": round(float(rgb[e][seg[e] == 1].astype(np.float32).mean() - ref_rgb[ref_seg == 1].astype(np.float32).mean()), 2)})
            if tier == "tier0":
                sheet.append([ref_render(me, de), rgb[e], ref_rgb])
        res["rows"][tier] = rows
    return res, sheet


def randomizer_probe():
    from metalsim.replicator.randomizers import Randomizer
    m = mujoco.MjModel.from_xml_string(PW)
    out = {"probe": "Randomizer.materials on renderers"}
    for name, mk in (("tier0", lambda: Tier0Renderer(m, 2, width=32, height=32, camera="cam")),
                     ("tier2_legacy", lambda: Tier2Renderer(m, 2, width=32, height=32, camera="cam", spp=1)),
                     ("tier2_omnipbr", lambda: Tier2Renderer(m, 2, width=32, height=32, camera="cam", spp=1, material_model="omnipbr"))):
        try:
            r = mk(); Randomizer(r, seed=0).materials(roughness=(0.2, 0.3), metallic=(0.5, 0.6))
            out[name] = "ok"
        except Exception as ex:    # noqa: BLE001
            out[name] = f"{type(ex).__name__}: {ex}"[:160]
    return out


# ---------------------------------------------------------------------------------------------------------------
# 6. fisheye: Isaac Lab's default FisheyeCameraCfg (f-theta, 200 deg) and a Kannala-Brandt lens; spheres at 0, 60
#    (left) and 95 degrees (right) off axis; blob centroids vs the lens's forward projection of the sphere centres, z-depth

FISH = """
<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 5" dir="0 0.3 -1"/>
    <camera name="cam" pos="0 0 1" xyaxes="1 0 0 0 0 1"/>
    <geom name="s0"  type="sphere" pos="0 4 1" size="0.3" rgba="0.9 0.3 0.3 1"/>
    <geom name="s60" type="sphere" pos="-3.4641 2 1" size="0.3" rgba="0.3 0.9 0.3 1"/>
    <geom name="s95" type="sphere" pos="3.9848 -0.3486 1" size="0.3" rgba="0.3 0.3 0.9 1"/>
  </worldbody>
</mujoco>"""
FISH_W, FISH_H = 192, 120


def fisheye_probe():
    from metalsim.render.lens import Lens
    from metalsim.render.fisheye import FisheyeTier0
    m = mujoco.MjModel.from_xml_string(FISH)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    R = d.cam_xmat[0].reshape(3, 3); cp = d.cam_xpos[0]
    lenses = {"ftheta_isaac_default": Lens.ftheta(),
              "kannala_brandt": Lens.kannala_brandt(38.0, 38.0, FISH_W / 2, FISH_H / 2, (0.05, -0.01, 0.0, 0.0), 200.0)}
    res = {"probe": "fisheye (blob centroid vs lens projection of the sphere centre, px; z-depth)", "lenses": {}}
    sheets = []
    for lname, lens in lenses.items():
        t2 = Tier2Renderer(m, 1, width=FISH_W, height=FISH_H, camera="cam", spp=4, max_bounces=1, center_sample=True, lens=lens)
        o2 = t2.render_host([d])
        seg2 = np.concatenate([[0], t2.tables.slot_seg_ids()])[o2["seg"][0]]
        f0 = FisheyeTier0(m, 1, lens, width=FISH_W, height=FISH_H, camera="cam", outputs=("rgb", "depth", "seg"), seg_mode=SEG_GEOM)
        o0 = f0.render_host([d])
        rt = RayTracer(m, 1, max_range=20.0)
        od = rt.trace_host([d], [rt.make_depth_camera("cam", FISH_W, FISH_H, lens=lens)])["ray_depth"][0]
        row = {"face_res": f0.face_res, "faces": f0.face_names}
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            pc = R.T @ (d.geom_xpos[g] - cp)
            target = lens.project(pc, FISH_W, FISH_H)
            r = {}
            for tag, seg, dep in (("tier2", seg2, o2["depth"][0]), ("tier0_cube", o0["seg"][0], o0["depth"][0]), ("raydepth", seg2, od)):
                mk = seg == g + 1
                if mk.sum() == 0:
                    r[tag] = None
                    continue
                yy, xx = np.nonzero(mk)
                cen = np.array([xx.mean() + 0.5, yy.mean() + 0.5])
                r[tag] = {"px": int(mk.sum()), "centroid_err_px": round(float(np.linalg.norm(cen - target)), 2),
                          "zdepth_nearest": round(float(dep[mk][np.argmin(np.abs(dep[mk]))]) if tag == "raydepth" else float(np.median(dep[mk])), 3)}
            r["zdepth_centre_exact"] = round(float(-pc[2]), 3)
            a, b = seg2 == g + 1, o0["seg"][0] == g + 1
            r["iou_tier0_vs_tier2"] = round(float((a & b).sum() / max((a | b).sum(), 1)), 3)
            row[name] = r
        res["lenses"][lname] = row
        sheets.append([o2["rgb"][0], o0["rgb"][0]])
    return res, sheets


def sheet(path, rows):
    from PIL import Image
    Image.fromarray(np.concatenate([np.concatenate([np.asarray(t, np.uint8) for t in r], 1) for r in rows], 0)).save(path)


PROBES = ("far", "precision", "ground", "hfield", "perworld", "randomizer", "fisheye")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/render_world")
    ap.add_argument("--probes", default=",".join(PROBES))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results = []
    for p in a.probes.split(","):
        t0 = time.time()
        try:
            r = globals()[f"{p}_probe"]()
        except Exception as ex:   # noqa: BLE001  (the "before" code lacks some features: record, keep going)
            import traceback
            traceback.print_exc()
            r = {"probe": p, "error": f"{type(ex).__name__}: {ex}"[:300]}
        if isinstance(r, tuple):
            r, imgs = r
            if imgs and isinstance(imgs[0], list):
                sheet(os.path.join(a.out, f"{p}.png"), imgs)
            elif imgs:
                sheet(os.path.join(a.out, f"{p}.png"), [imgs])
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(json.dumps(r), flush=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
