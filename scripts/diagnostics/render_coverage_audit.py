"""Render-coverage audit (docs/RENDER_COVERAGE.md): does every renderer draw what the simulator simulates?

For each probe scene, renders the same MjData with ``mujoco.Renderer`` (the reference: MuJoCo's own OpenGL renderer
draws every simulated visual: geoms, sites, tendons, flexes, skins) and with MetalSim's tier 0, tier 1, tier 2 and the
ray-traced depth camera, then measures per object (MuJoCo segmentation mask = the reference silhouette):

* ``rgb``: fraction of the reference mask where MetalSim's RGB is not the clear colour,
* ``depth``: fraction of the mask with a depth value > 0, and the median |depth - MuJoCo depth| there,
* ``seg``: fraction of the mask whose MetalSim id (geom mode) is the object's own id: geom + 1 for geoms,
  ``ngeom + 1 + j`` for the j-th extra object (sites, flexes, skins, tendons; ``tables.extras``), and any id.

Plus targeted probes (transparency, far plane, infinite plane, textures, lights > 8, spot/point shadows, mesh normals,
skybox). Prints one JSON object per probe and writes everything to ``--out`` (JSON) and contact sheets (PNG).

    scripts/gpu_run.sh render_coverage check 10 -- .venv/bin/python scripts/diagnostics/render_coverage_audit.py --out runs/render_coverage
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from metalsim.render.tier0 import Tier0Renderer, SEG_GEOM   # noqa: E402
from metalsim.render.tier2 import Tier2Renderer             # noqa: E402
from metalsim.sensors.raytrace import RayTracer             # noqa: E402

W, H = 160, 120
CAM = '<camera name="cam" pos="0 -2.2 0.75" xyaxes="1 0 0 0 0.25 0.968"/>'
ANCHOR = '<geom name="anchor" type="box" pos="2.5 3 2.5" size="0.01 0.01 0.01" rgba="1 1 1 1"/>'   # keeps "no drawable geoms" away


def ref_render(m, d, seg=False, depth=False, w=W, h=H):
    r = mujoco.Renderer(m, h, w)
    if seg:
        r.enable_segmentation_rendering()
    if depth:
        r.enable_depth_rendering()
    r.update_scene(d, camera="cam")
    out = r.render().copy()
    r.close()
    return out


def ours(m, d, which, w=W, h=H, **kw):
    if which in ("tier0", "tier1"):
        r = Tier0Renderer(m, 1, width=w, height=h, camera="cam", outputs=("rgb", "depth", "seg"), seg_mode=SEG_GEOM,
                          tier=0 if which == "tier0" else 1, **kw)
        o = r.render_host([d])
        return {"rgb": o["rgb"][0], "depth": o["depth"][0], "seg": o["seg"][0], "extras": getattr(r.tables, "extras", [])}
    if which == "tier2":
        r = Tier2Renderer(m, 1, width=w, height=h, camera="cam", spp=16, max_bounces=2, center_sample=True, **kw)
        o = r.render_host([d], passes=2)
        ids = r.tables.slot_seg_ids() if hasattr(r.tables, "slot_seg_ids") else np.asarray(r.tables.geoms) + 1
        return {"rgb": o["rgb"][0], "depth": o["depth"][0], "seg": o["seg"][0], "slot_ids": ids, "geoms": r.tables.geoms,
                "extras": getattr(r.tables, "extras", [])}
    if which == "raydepth":
        rt = RayTracer(m, 1, max_range=100.0)
        cam = rt.make_depth_camera("cam", w, h)
        o = rt.trace_host([d], [cam])
        return {"depth": o["ray_depth"][0]}
    raise ValueError(which)


def coverage(mask, o, ref_depth, geom_id=None, tier2_geoms=None, *, seg_id=None, slot_ids=None):
    """Per-object coverage of one MetalSim output against the reference mask. The object's id in a geom-mode
    segmentation buffer: ``seg_id``, or ``geom_id + 1`` for a geom (None: only "any id" is measured); tier 2 writes
    slots: ``slot_ids`` (slot -> id, ``tables.slot_seg_ids()``) or the older ``tier2_geoms`` (slot -> geom)."""
    if seg_id is None and geom_id is not None:
        seg_id = geom_id + 1
    if slot_ids is None and tier2_geoms is not None:
        slot_ids = np.asarray(tier2_geoms) + 1
    n = int(mask.sum())
    res = {"px": n}
    if n == 0:
        return res
    if "rgb" in o:
        res["rgb"] = round(float((o["rgb"][mask].max(-1) > 3).mean()), 3)
    if "depth" in o:
        dv = o["depth"][mask]
        res["depth"] = round(float((dv > 0).mean()), 3)
        ok = dv > 0
        if ok.any():
            res["depth_err_med_m"] = round(float(np.median(np.abs(dv[ok] - ref_depth[mask][ok]))), 4)
    if "seg" in o:
        s = o["seg"][mask]
        if slot_ids is not None:         # tier 2 writes slot + 1: map to the geom-mode id
            lut = np.concatenate([[0], np.asarray(slot_ids)])
            s = lut[s]
        res["seg_any"] = round(float((s > 0).mean()), 3)
        if seg_id is not None:
            res["seg_correct"] = round(float((s == seg_id).mean()), 3)
    return res


def object_audit(xml, name, objects, renderers=("tier0", "tier1", "tier2", "raydepth"), nstep=0):
    """objects: list of (label, mjtObj type, id or name)."""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    seg = ref_render(m, d, seg=True)
    rdep = ref_render(m, d, depth=True)
    outs = {r: ours(m, d, r) for r in renderers}
    res = {"probe": name, "objects": {}}
    for label, otype, oid in objects:
        if isinstance(oid, str):
            oid = mujoco.mj_name2id(m, otype, oid)
        mask = (seg[..., 0] == oid) & (seg[..., 1] == int(otype))

        def sid(o):
            if otype == mujoco.mjtObj.mjOBJ_GEOM:
                return oid + 1
            hit = [e["seg_id"] for e in o.get("extras", []) if e["objtype"] == int(otype) and e["objid"] == oid]
            return hit[0] if hit else None
        res["objects"][label] = {r: coverage(mask, o, rdep, seg_id=sid(o) if r != "raydepth" else None, slot_ids=o.get("slot_ids")) for r, o in outs.items()}
    return res, m, d, outs


# --------------------------------------------------------------------------------------------------------------------
# Probe scenes

GEOMS = f"""
<mujoco>
  <extension><plugin plugin="mujoco.sdf.torus"><instance name="torus">
    <config key="radius1" value="0.12"/><config key="radius2" value="0.04"/></instance></plugin></extension>
  <asset>
    <mesh name="tet" vertex="-0.12 -0.12 -0.1  0.12 -0.12 -0.1  0 0.12 -0.1  0 0 0.14"/>
    <mesh name="torus"><plugin instance="torus"/></mesh>
    <hfield name="hf" nrow="8" ncol="8" size="0.15 0.15 0.12 0.02"
      elevation="0 0 0 0 0 0 0 0  0 1 1 1 1 1 1 0  0 1 2 2 2 2 1 0  0 1 2 3 3 2 1 0  0 1 2 3 3 2 1 0  0 1 2 2 2 2 1 0  0 1 1 1 1 1 1 0  0 0 0 0 0 0 0 0"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0.3 -1"/>
    {CAM}{ANCHOR}
    <geom name="sphere"    type="sphere"    pos="-1.05 0 0.95" size="0.13"            rgba="0.9 0.3 0.3 1"/>
    <geom name="capsule"   type="capsule"   pos="-0.7 0 0.95"  size="0.08 0.1"        rgba="0.3 0.9 0.3 1"/>
    <geom name="ellipsoid" type="ellipsoid" pos="-0.35 0 0.95" size="0.08 0.12 0.15"  rgba="0.3 0.3 0.9 1"/>
    <geom name="cylinder"  type="cylinder"  pos="0 0 0.95"     size="0.1 0.12"        rgba="0.9 0.9 0.3 1"/>
    <geom name="box"       type="box"       pos="0.35 0 0.95"  size="0.1 0.1 0.12"    rgba="0.9 0.3 0.9 1"/>
    <geom name="mesh"      type="mesh"      pos="0.7 0 0.95"   mesh="tet"             rgba="0.3 0.9 0.9 1"/>
    <geom name="sdf"       type="sdf"       pos="1.05 0 0.95"  mesh="torus" euler="90 0 0" rgba="0.95 0.6 0.3 1"><plugin instance="torus"/></geom>
    <geom name="hfield"    type="hfield"    pos="-0.5 0 0.25"  hfield="hf"            rgba="0.6 0.6 0.95 1"/>
    <geom name="plane"     type="plane"     pos="0.5 0 0.3"    size="0.2 0.2 0.1" euler="-75 0 0" rgba="0.95 0.95 0.6 1"/>
  </worldbody>
</mujoco>"""

NONGEOM = f"""
<mujoco>
  <asset><texture name="sk" type="2d" builtin="checker" width="32" height="32" rgb1="1 1 1" rgb2="0.2 0.6 0.2"/></asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0.3 -1"/>
    {CAM}{ANCHOR}
    <site name="site_sphere" type="sphere" pos="-1.0 0 1.0" size="0.1" rgba="0.9 0.3 0.3 1"/>
    <site name="site_box" type="box" pos="-0.7 0 1.0" size="0.08 0.08 0.08" rgba="0.3 0.9 0.3 1"/>
    <body name="t1" pos="-0.45 0 1.2"><site name="ta" size="0.005" rgba="0 0 0 0"/></body>
    <body name="t2" pos="-0.25 0 0.75"><site name="tb" size="0.005" rgba="0 0 0 0"/></body>
    <body name="bone" pos="1.0 0 1.0"/>
    <flexcomp name="cable" type="grid" count="8 1 1" spacing="0.05 0.05 0.05" pos="0.0 0 0.45" dim="1" radius="0.02"
              rgba="0.95 0.6 0.2 1"><edge equality="true"/></flexcomp>
    <flexcomp name="cloth" type="grid" count="7 7 1" spacing="0.04 0.04 0.04" pos="0.15 0 0.75" euler="90 0 0" dim="2"
              radius="0.005" rgba="0.2 0.5 0.9 1"><edge equality="true"/></flexcomp>
    <flexcomp name="soft" type="grid" count="3 3 3" spacing="0.07 0.07 0.07" pos="0.55 0 0.95" dim="3" radius="0.005"
              rgba="0.6 0.9 0.3 1"><edge equality="true"/></flexcomp>
  </worldbody>
  <tendon><spatial name="tendon" width="0.03" rgba="0.9 0.9 0.2 1"><site site="ta"/><site site="tb"/></spatial></tendon>
  <deformable>
    <skin name="skin" rgba="0.8 0.4 0.9 1"
      vertex="0.9 -0.1 0.9  1.1 -0.1 0.9  1.1 0.1 0.9  0.9 0.1 0.9  0.9 -0.1 1.1  1.1 -0.1 1.1  1.1 0.1 1.1  0.9 0.1 1.1"
      face="0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7">
      <bone body="bone" bindpos="1.0 0 1.0" bindquat="1 0 0 0" vertid="0 1 2 3 4 5 6 7" vertweight="1 1 1 1 1 1 1 1"/>
    </skin>
  </deformable>
</mujoco>"""

TRANSPARENT = f"""
<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0.3 -1"/>
    {CAM}{ANCHOR}
    <geom name="back"  type="box" pos="0 0.4 0.9" size="0.3 0.05 0.3" rgba="0.95 0.1 0.1 1"/>
    <geom name="glass" type="box" pos="0 -0.2 0.9" size="0.45 0.01 0.45" rgba="0.1 0.1 0.95 0.25"/>
  </worldbody>
</mujoco>"""

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


def texture_xml(tmpdir):
    """Emissive (unlit) textured geoms so RGB compares texture mapping only: MuJoCo colour = emission x texture."""
    from PIL import Image
    img = np.zeros((64, 64, 3), np.uint8)
    img[:32] = (230, 40, 40)            # top half of the image file red
    img[32:] = (40, 40, 230)            # bottom half blue
    img[:, :16] = (40, 230, 40)         # left strip green
    p = os.path.join(tmpdir, "asym.png")
    Image.fromarray(img).save(p)
    return f"""
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <asset>
    <texture name="chk" type="2d" builtin="checker" width="64" height="64" rgb1="0.95 0.95 0.95" rgb2="0.1 0.1 0.1"/>
    <texture name="cube" type="cube" builtin="checker" width="64" height="64" rgb1="0.95 0.95 0.95" rgb2="0.1 0.1 0.1"/>
    <texture name="asym" type="2d" file="{p}"/>
    <texture name="sky" type="skybox" builtin="gradient" width="64" height="64" rgb1="0.9 0.9 1" rgb2="0.05 0.05 0.3"/>
    <material name="chk"  texture="chk"  texrepeat="2 2" emission="1" specular="0" shininess="0"/>
    <material name="cube" texture="cube" emission="1" specular="0" shininess="0"/>
    <material name="asym" texture="asym" emission="1" specular="0" shininess="0"/>
    <mesh name="quad" vertex="-0.15 -0.005 -0.15  0.15 -0.005 -0.15  0.15 -0.005 0.15  -0.15 -0.005 0.15
                              -0.15 0.005 -0.15  0.15 0.005 -0.15  0.15 0.005 0.15  -0.15 0.005 0.15"
          face="0 1 2 0 2 3 4 6 5 4 7 6 0 4 5 0 5 1 1 5 6 1 6 2 2 6 7 2 7 3 3 7 4 3 4 0"
          texcoord="0 0  1 0  1 1  0 1  0 0  1 0  1 1  0 1"/>
  </asset>
  <worldbody>
    {CAM}
    <geom name="tex_sphere"   type="sphere"   pos="-1.0 0 1.0" size="0.14"      material="chk"/>
    <geom name="tex_cylinder" type="cylinder" pos="-0.6 0 1.0" size="0.12 0.14" material="chk"/>
    <geom name="tex_capsule"  type="capsule"  pos="-0.2 0 1.0" size="0.1 0.08"  material="chk"/>
    <geom name="tex_box_2d"   type="box"      pos="0.2 0 1.0"  size="0.13 0.13 0.13" euler="0 0 20" material="chk"/>
    <geom name="tex_box_cube" type="box"      pos="0.6 0 1.0"  size="0.13 0.13 0.13" euler="0 0 20" material="cube"/>
    <geom name="tex_mesh_uv"  type="mesh"     pos="1.0 0 1.0"  mesh="quad" material="asym"/>
  </worldbody>
</mujoco>"""


def texture_probe(tmpdir):
    xml = texture_xml(tmpdir)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    rgb_ref = ref_render(m, d).astype(np.float32)
    seg = ref_render(m, d, seg=True)
    outs = {r: ours(m, d, r) for r in ("tier0", "tier2")}
    res = {"probe": "textures", "objects": {}}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        mask = (seg[..., 0] == g) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        a = rgb_ref[mask]
        row = {"px": int(mask.sum()), "ref_lum_std": round(float(a.mean(-1).std()), 1)}
        for r, o in outs.items():
            b = o["rgb"][mask].astype(np.float32)
            la, lb = a.mean(-1), b.mean(-1)
            corr = float(np.corrcoef(la, lb)[0, 1]) if la.std() > 1 and lb.std() > 1 else float("nan")
            row[r] = {"lum_std": round(float(lb.std()), 1), "corr_vs_mujoco": round(corr, 3),
                      "mean_abs_diff": round(float(np.abs(a - b).mean()), 1),
                      "mean_rgb": [round(float(x)) for x in b.mean(0)], "ref_mean_rgb": [round(float(x)) for x in a.mean(0)]}
        res["objects"][name] = row
    bg = seg[..., 0] < 0
    res["skybox_background"] = {"mujoco_lum_std": round(float(rgb_ref[bg].mean(-1).std()), 1),
                                **{r: round(float(o["rgb"][bg].astype(np.float32).mean(-1).std()), 1) for r, o in outs.items()}}
    # orientation of the image file on explicit OBJ-convention UVs: which colour is at the top of the quad?
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tex_mesh_uv")
    mask = (seg[..., 0] == g) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    ys = np.nonzero(mask.any(1))[0]
    top, bot = ys[: max(2, len(ys) // 4)], ys[-max(2, len(ys) // 4):]

    def dom(img, rows):
        sub = img[rows][:, :, :][mask[rows]].astype(np.float32).mean(0)
        return "red" if sub[0] > sub[2] else "blue"
    res["mesh_uv_orientation"] = {"mujoco": {"top": dom(rgb_ref, top), "bottom": dom(rgb_ref, bot)},
                                  **{r: {"top": dom(o["rgb"], top), "bottom": dom(o["rgb"], bot)} for r, o in outs.items()}}
    return res, (rgb_ref.astype(np.uint8), outs)


def transparency_probe():
    m = mujoco.MjModel.from_xml_string(TRANSPARENT)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    rgb_ref = ref_render(m, d).astype(np.float32)
    # region where the back box lies behind the glass: back box mask with the glass hidden
    m2 = mujoco.MjModel.from_xml_string(TRANSPARENT.replace('rgba="0.1 0.1 0.95 0.25"', 'rgba="0.1 0.1 0.95 0"'))
    d2 = mujoco.MjData(m2); mujoco.mj_forward(m2, d2)
    seg2 = ref_render(m2, d2, seg=True)
    back = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "back")
    mask = (seg2[..., 0] == back) & (seg2[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    res = {"probe": "transparency (glass alpha 0.25 in front of a red box)", "px": int(mask.sum()),
           "mujoco_mean_rgb": [round(float(x)) for x in rgb_ref[mask].mean(0)]}
    for r in ("tier0", "tier2"):
        o = ours(m, d, r)
        segv = o["seg"][mask]
        if r == "tier2":
            segv = np.concatenate([[0], np.asarray(o["slot_ids"])])[segv]
        res[r] = {"mean_rgb": [round(float(x)) for x in o["rgb"][mask].astype(np.float32).mean(0)],
                  "seg_is_glass": round(float((segv == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "glass") + 1).mean()), 3),
                  "depth_median_m": round(float(np.median(o["depth"][mask])), 3)}
    rd = ours(m, d, "raydepth")["depth"][mask]
    res["raydepth_median_m"] = round(float(np.median(rd)), 3)
    return res


def far_probe():
    m = mujoco.MjModel.from_xml_string(FAR)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    seg = ref_render(m, d, seg=True)
    rdep = ref_render(m, d, depth=True)
    outs = {r: ours(m, d, r) for r in ("tier0", "tier1", "tier2", "raydepth")}
    res = {"probe": "far plane / infinite plane", "objects": {}}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        mask = (seg[..., 0] == g) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        row = {r: coverage(mask, o, rdep, seg_id=g + 1 if r != "raydepth" else None, slot_ids=o.get("slot_ids")) for r, o in outs.items()}
        if mask.any():
            row["ref_depth_range_m"] = [round(float(rdep[mask].min()), 2), round(float(rdep[mask].max()), 2)]
        res["objects"][name] = row
    return res, (ref_render(m, d), outs)


LIGHTS10 = f"""
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <worldbody>
    <camera name="cam" pos="0 0 6" xyaxes="1 0 0 0 1 0"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
    {''.join(f'<light name="L{i}" pos="{-2.25 + 0.5 * i:.2f} 0 0.4" dir="0 0 -1" diffuse="0.6 0.6 0.6" attenuation="1 0 8" castshadow="false"/>' for i in range(10))}
  </worldbody>
</mujoco>"""

SHADOW = """
<mujoco>
  <visual><headlight active="0" ambient="0.05 0.05 0.05"/></visual>
  <worldbody>
    <camera name="cam" pos="0 -1.6 1.6" xyaxes="1 0 0 0 0.7 0.714"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom name="blocker" type="box" pos="0 0 0.45" size="0.15 0.15 0.02" rgba="0.8 0.3 0.3 1"/>
    {LIGHT}
  </worldbody>
</mujoco>"""


def shadow_probe():
    """Shadow of a plate on the floor under a spot, a point and a directional light: floor brightness inside the
    plate's shadow footprint over the brightness just outside it (1 = no shadow)."""
    out = {"probe": "shadows per light type (ratio shadowed/unshadowed floor luminance; 1.0 = no shadow)"}
    lights = {"spot": '<light pos="0 0 1.5" dir="0 0 -1" cutoff="50" exponent="1" castshadow="true" diffuse="0.9 0.9 0.9"/>',
              "point": '<light type="point" pos="0 0 1.5" castshadow="true" diffuse="0.9 0.9 0.9"/>',
              "directional": '<light directional="true" pos="0 0 3" dir="0 0 -1" castshadow="true" diffuse="0.9 0.9 0.9"/>'}
    for kind, l in lights.items():
        xml = SHADOW.replace("{LIGHT}", l)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        seg = ref_render(m, d, seg=True)
        floor = (seg[..., 0] == 0) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        # footprint: floor pixels whose world point lies under the plate (|x|,|y| < 0.12); ring: 0.25 < r < 0.4
        dep = ref_render(m, d, depth=True)
        fy = H / (2 * np.tan(np.deg2rad(m.cam_fovy[0]) / 2))
        yy, xx = np.mgrid[0:H, 0:W]
        dirs = np.stack([(xx + 0.5 - W / 2) / fy, -(yy + 0.5 - H / 2) / fy, -np.ones_like(xx, float)], -1)
        Rm = d.cam_xmat[0].reshape(3, 3)
        P = d.cam_xpos[0] + (dirs * dep[..., None]) @ Rm.T
        inside = floor & (np.abs(P[..., 0]) < 0.1) & (np.abs(P[..., 1]) < 0.1)
        rr = np.hypot(P[..., 0], P[..., 1])
        ring = floor & (rr > 0.3) & (rr < 0.4)
        row = {}
        imgs = {"mujoco": ref_render(m, d)}
        for r in ("tier0", "tier1", "tier2"):
            imgs[r] = ours(m, d, r)["rgb"]
        for r, img in imgs.items():
            lum = img.astype(np.float32).mean(-1)
            row[r] = round(float(lum[inside].mean() / max(lum[ring].mean(), 1e-3)), 3)
        out[kind] = row
    return out


def lights_probe():
    m = mujoco.MjModel.from_xml_string(LIGHTS10)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    res = {"probe": "10 point lights, one floor patch each (patch luminance per light index)"}
    fy = H / (2 * np.tan(np.deg2rad(m.cam_fovy[0]) / 2))
    imgs = {"mujoco": ref_render(m, d)}
    for r in ("tier0", "tier2"):
        imgs[r] = ours(m, d, r)["rgb"]
    for r, img in imgs.items():
        lum = img.astype(np.float32).mean(-1)
        vals = []
        for i in range(10):
            x = -2.25 + 0.5 * i
            px = int(round(W / 2 + x / 6.0 * fy))
            vals.append(round(float(lum[H // 2 - 2:H // 2 + 3, max(px - 2, 0):px + 3].mean()), 1))
        res[r] = vals
    return res


MESHBOX = f"""
<mujoco>
  <asset><mesh name="mbox" vertex="-0.2 -0.2 -0.2  0.2 -0.2 -0.2  0.2 0.2 -0.2  -0.2 0.2 -0.2  -0.2 -0.2 0.2  0.2 -0.2 0.2  0.2 0.2 0.2  -0.2 0.2 0.2"/></asset>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.2 0.5 -1" castshadow="false"/>
    {CAM}{ANCHOR}
    <geom name="mesh_box" type="mesh" mesh="mbox" pos="-0.35 0 0.9" euler="0 0 25" rgba="0.8 0.8 0.8 1"/>
    <geom name="prim_box" type="box" size="0.2 0.2 0.2" pos="0.35 0 0.9" euler="0 0 25" rgba="0.8 0.8 0.8 1"/>
  </worldbody>
</mujoco>"""


def mesh_normal_probe():
    """A box written as a mesh vs a primitive box, same pose and light: a flat face should be uniformly lit.
    Reports the luminance spread (std) over each object's pixels per renderer."""
    m = mujoco.MjModel.from_xml_string(MESHBOX)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    seg = ref_render(m, d, seg=True)
    imgs = {"mujoco": ref_render(m, d), "tier0": ours(m, d, "tier0")["rgb"], "tier2": ours(m, d, "tier2")["rgb"]}
    res = {"probe": "mesh normals (luminance std inside the object; a box has 3 flat faces visible)"}
    for name in ("mesh_box", "prim_box"):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        mask = (seg[..., 0] == g) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        # within-face spread: luminance std after removing per-face means (faces = clusters of the primitive's normals)
        res[name] = {r: round(float(img.astype(np.float32).mean(-1)[mask].std()), 1) for r, img in imgs.items()}
    return res


def sheet(path, rows):
    from PIL import Image
    tiles = []
    for row in rows:
        tiles.append(np.concatenate([np.asarray(t, np.uint8) for t in row], 1))
    Image.fromarray(np.concatenate(tiles, 0)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/render_coverage")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G, S, T, F = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_TENDON, mujoco.mjtObj.mjOBJ_FLEX
    results = []
    r, m, d, outs = object_audit(GEOMS, "geom types", [(n, G, n) for n in
                                 ("sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh", "sdf", "hfield", "plane")])
    results.append(r); print(json.dumps(r), flush=True)
    sheet(os.path.join(a.out, "geoms.png"), [[ref_render(m, d), outs["tier0"]["rgb"], outs["tier2"]["rgb"]]])
    r, m, d, outs = object_audit(NONGEOM, "non-geom visuals",
                                 [("site_sphere", S, "site_sphere"), ("site_box", S, "site_box"), ("tendon", T, "tendon"),
                                  ("flex_cable_1d", F, "cable"), ("flex_cloth_2d", F, "cloth"), ("flex_soft_3d", F, "soft"),
                                  ("skin", mujoco.mjtObj.mjOBJ_SKIN, "skin")])
    r["model"] = {"nflex": int(m.nflex), "nflexvert": int(m.nflexvert), "nskin": int(m.nskin), "ntendon": int(m.ntendon),
                  "nsite": int(m.nsite), "ngeom": int(m.ngeom), "nbody": int(m.nbody)}
    results.append(r); print(json.dumps(r), flush=True)
    sheet(os.path.join(a.out, "nongeom.png"), [[ref_render(m, d), outs["tier0"]["rgb"], outs["tier2"]["rgb"]]])
    r = transparency_probe(); results.append(r); print(json.dumps(r), flush=True)
    r, imgs = far_probe(); results.append(r); print(json.dumps(r), flush=True)
    sheet(os.path.join(a.out, "far.png"), [[imgs[0], imgs[1]["tier0"]["rgb"], imgs[1]["tier2"]["rgb"]]])
    r, (ref, outs) = texture_probe(a.out); results.append(r); print(json.dumps(r), flush=True)
    sheet(os.path.join(a.out, "textures.png"), [[ref, outs["tier0"]["rgb"], outs["tier2"]["rgb"]]])
    r = lights_probe(); results.append(r); print(json.dumps(r), flush=True)
    r = shadow_probe(); results.append(r); print(json.dumps(r), flush=True)
    r = mesh_normal_probe(); results.append(r); print(json.dumps(r), flush=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
