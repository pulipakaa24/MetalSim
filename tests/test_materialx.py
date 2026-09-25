"""MaterialX materials: .mtlx documents (standard_surface with a texture, open_pbr_surface, a
UsdPreviewSurface node graph) flatten to the expected material-table parameters, a USD scene that
references the document composes and binds them, and tier 0 renders the expected albedo."""
import os

import mujoco
import numpy as np
import pytest

pytest.importorskip("MaterialX")
from pxr import Sdf, Usd, UsdShade  # noqa: E402

from metalsim.scene import materialx as mtx  # noqa: E402

MTLX = """<?xml version="1.0"?>
<materialx version="1.39" fileprefix="tex/">
  <!-- standard_surface, base colour from a tiled texture times a tint -->
  <nodegraph name="NG_tex">
    <tiledimage name="img" type="color3">
      <input name="file" type="filename" value="checker.png" colorspace="srgb_texture"/>
      <input name="uvtiling" type="vector2" value="2, 3"/>
    </tiledimage>
    <multiply name="tint" type="color3">
      <input name="in1" type="color3" nodename="img"/>
      <input name="in2" type="color3" value="1.0, 0.5, 0.5"/>
    </multiply>
    <output name="out" type="color3" nodename="tint"/>
  </nodegraph>
  <standard_surface name="SR_tex" type="surfaceshader">
    <input name="base" type="float" value="0.9"/>
    <input name="base_color" type="color3" nodegraph="NG_tex" output="out"/>
    <input name="metalness" type="float" value="0.3"/>
    <input name="specular_roughness" type="float" value="0.4"/>
    <input name="specular_IOR" type="float" value="1.5"/>
    <input name="coat" type="float" value="0.5"/>
  </standard_surface>
  <surfacematerial name="M_tex" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_tex"/>
  </surfacematerial>

  <!-- open_pbr_surface, constants -->
  <open_pbr_surface name="SR_opbr" type="surfaceshader">
    <input name="base_weight" type="float" value="0.8"/>
    <input name="base_color" type="color3" value="0.1, 0.6, 0.2"/>
    <input name="specular_roughness" type="float" value="0.25"/>
    <input name="specular_ior" type="float" value="1.6"/>
    <input name="fuzz_weight" type="float" value="0.3"/>
    <input name="emission_luminance" type="float" value="100"/>
    <input name="emission_color" type="color3" value="0.1, 0.6, 0.2"/>
  </open_pbr_surface>
  <surfacematerial name="M_opbr" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_opbr"/>
  </surfacematerial>

  <!-- UsdPreviewSurface node graph: UsdUVTexture colour (scaled), roughness from a grey texture -->
  <nodegraph name="NG_prev">
    <UsdPrimvarReader name="st" type="vector2"><input name="varname" type="string" value="st"/></UsdPrimvarReader>
    <UsdUVTexture name="diff" type="multioutput">
      <input name="file" type="filename" value="checker.png"/>
      <input name="st" type="vector2" nodename="st"/>
      <input name="scale" type="vector4" value="0.5, 0.5, 0.5, 1"/>
      <output name="rgb" type="color3"/>
    </UsdUVTexture>
    <UsdUVTexture name="rough" type="multioutput">
      <input name="file" type="filename" value="grey.png"/>
      <input name="st" type="vector2" nodename="st"/>
      <output name="r" type="float"/>
    </UsdUVTexture>
    <output name="diffuse" type="color3" nodename="diff" output="rgb"/>
    <output name="roughness" type="float" nodename="rough" output="r"/>
  </nodegraph>
  <UsdPreviewSurface name="SR_prev" type="surfaceshader">
    <input name="diffuseColor" type="color3" nodegraph="NG_prev" output="diffuse"/>
    <input name="roughness" type="float" nodegraph="NG_prev" output="roughness"/>
    <input name="metallic" type="float" value="1.0"/>
    <input name="clearcoat" type="float" value="0.2"/>
  </UsdPreviewSurface>
  <surfacematerial name="M_prev" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_prev"/>
  </surfacematerial>
</materialx>
"""

# the scene: one cube per material. /World/Looks/MX references the whole document (UsdMtlx layout,
# bindings to /World/Looks/MX/Materials/<name>); /World/Looks/Green references one material onto a
# Material prim; /World/Looks/Inline is a MaterialX network written directly as UsdShade prims.
SCENE = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World"
{
    def Scope "Looks"
    {
        def "MX" (prepend references = @./mats.mtlx@</MaterialX>) {}
        def Material "Green" (prepend references = @./mats.mtlx@</MaterialX/Materials/M_opbr>) {}
        def Material "Inline"
        {
            token outputs:mtlx:surface.connect = </World/Looks/Inline/SR.outputs:out>
            def Shader "SR"
            {
                uniform token info:id = "ND_standard_surface_surfaceshader"
                color3f inputs:base_color = (0.2, 0.3, 0.9)
                float inputs:specular_roughness = 0.6
                token outputs:out
            }
        }
    }
    def Camera "cam"
    {
        float focalLength = 24
        float verticalAperture = 20
        float horizontalAperture = 20
        double3 xformOp:translate = (0, 0, 3)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def DistantLight "sun"
    {
        float inputs:intensity = 1
    }
    def Cube "tex" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        double size = 0.8
        rel material:binding = </World/Looks/MX/Materials/M_tex>
        double3 xformOp:translate = (-0.5, 0.5, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "green" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        double size = 0.8
        rel material:binding = </World/Looks/Green>
        double3 xformOp:translate = (0.5, 0.5, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "prev" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        double size = 0.8
        rel material:binding = </World/Looks/MX/Materials/M_prev>
        double3 xformOp:translate = (-0.5, -0.5, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "inline" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        double size = 0.8
        rel material:binding = </World/Looks/Inline>
        double3 xformOp:translate = (0.5, -0.5, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
"""


def _write_assets(d):
    from PIL import Image
    os.makedirs(d / "tex", exist_ok=True)
    chk = np.zeros((16, 16, 3), np.uint8)
    chk[(np.arange(16)[:, None] // 4 + np.arange(16)[None] // 4) % 2 == 0] = (200, 200, 200)
    chk[(np.arange(16)[:, None] // 4 + np.arange(16)[None] // 4) % 2 == 1] = (40, 80, 120)
    Image.fromarray(chk).save(d / "tex" / "checker.png")
    Image.fromarray(np.full((8, 8, 3), 153, np.uint8)).save(d / "tex" / "grey.png")
    (d / "mats.mtlx").write_text(MTLX)
    (d / "scene.usda").write_text(SCENE)
    return str(d / "mats.mtlx"), str(d / "scene.usda")


def _f0(ior):
    return ((ior - 1) / (ior + 1)) ** 2


def test_mtlx_document_flattens(tmp_path):
    path, _ = _write_assets(tmp_path)
    ps = mtx.load_mtlx(path)
    assert set(ps) == {"M_tex", "M_opbr", "M_prev"}

    t = ps["M_tex"]                                     # standard_surface + tiled texture x tint
    assert t.model == "standard_surface"
    assert t.texture == str(tmp_path / "tex" / "checker.png")      # fileprefix resolved, absolute
    np.testing.assert_allclose(t.rgba, [0.9, 0.45, 0.45, 1.0], atol=1e-6)   # tint x base weight
    np.testing.assert_allclose(t.texrepeat, [2, 3])
    assert (t.metallic, t.roughness) == pytest.approx((0.3, 0.4))
    assert t.specular == pytest.approx(0.5)             # IOR 1.5 -> F0 0.04 -> 0.04 / 0.08
    assert "coat" in t.unmapped

    o = ps["M_opbr"]                                    # open_pbr_surface constants
    assert o.model == "open_pbr_surface" and o.texture is None
    np.testing.assert_allclose(o.rgba[:3], 0.8 * np.array([0.1, 0.6, 0.2]), atol=1e-6)
    assert o.roughness == pytest.approx(0.25) and o.metallic == 0.0
    assert o.specular == pytest.approx(_f0(1.6) / 0.08, rel=1e-5)
    assert o.emission == pytest.approx(0.1 / 0.8, rel=1e-4)     # 100 cd/m^2 / 1000, relative to base
    assert "fuzz_weight" in o.unmapped

    p = ps["M_prev"]                                    # UsdPreviewSurface node graph
    assert p.model == "UsdPreviewSurface"
    assert p.texture == str(tmp_path / "tex" / "checker.png")
    np.testing.assert_allclose(p.rgba[:3], 0.5)         # UsdUVTexture scale
    assert p.metallic == 1.0
    assert p.roughness == pytest.approx(0.6, abs=1e-6)  # grey texture (153/255) reduced to its mean
    assert any("roughness" in a for a in p.approximations)
    assert "clearcoat" in p.unmapped


def test_usd_scene_binds_mtlx(tmp_path):
    from metalsim.scene.usd_to_mjcf import load_usd
    _, scene = _write_assets(tmp_path)
    m = load_usd(scene, lossless=False).compile()

    def mat_of(geom):
        return int(m.geom_matid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom)])
    g = mat_of("green")
    assert g >= 0
    np.testing.assert_allclose(m.mat_rgba[g][:3], [0.08, 0.48, 0.16], atol=1e-6)
    assert m.mat_roughness[g] == pytest.approx(0.25) and m.mat_specular[g] == pytest.approx(_f0(1.6) / 0.08, rel=1e-5)
    t = mat_of("tex")
    assert t >= 0 and m.mat_texid[t][int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] >= 0
    np.testing.assert_allclose(m.mat_texrepeat[t], [2, 3])
    assert m.mat_metallic[t] == pytest.approx(0.3)
    i = mat_of("inline")                                # UsdShade-native MaterialX, no .mtlx
    np.testing.assert_allclose(m.mat_rgba[i][:3], [0.2, 0.3, 0.9], atol=1e-6)
    assert m.mat_roughness[i] == pytest.approx(0.6)
    # the user's layers are untouched; the swap lives in the session layer only
    assert "anon:" not in Sdf.Layer.FindOrOpen(scene).ExportToString()


def test_non_materialx_stage_untouched(tmp_path):
    """A plain UsdPreviewSurface material is not MaterialX; the hook does not trigger."""
    st = Usd.Stage.CreateInMemory()
    mat = UsdShade.Material.Define(st, "/M")
    sh = UsdShade.Shader.Define(st, "/M/S"); sh.CreateIdAttr("UsdPreviewSurface")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    assert not mtx.is_materialx(mat) and not mtx.stage_has_materialx(st)


def _expected_mjcf(tmp_path, params):
    """The same four cubes authored directly in MJCF with the expected parameters."""
    tex = str(tmp_path / "tex" / "checker.png")
    mats, geoms = [], []
    pos = {"tex": (-0.5, 0.5), "green": (0.5, 0.5), "prev": (-0.5, -0.5), "inline": (0.5, -0.5)}
    for name, p in params.items():
        attrs = (f'rgba="{" ".join(map(str, p["rgba"]))}" roughness="{p["roughness"]}" metallic="{p["metallic"]}" '
                 f'specular="{p["specular"]}" emission="{p["emission"]}" shininess="{max(0, 1 - p["roughness"])}"')
        if p.get("tex"):
            attrs += f' texture="t_{name}" texrepeat="{p["repeat"][0]} {p["repeat"][1]}"'
            mats.append(f'<texture name="t_{name}" type="2d" file="{tex}"/>')
        mats.append(f'<material name="m_{name}" {attrs}/>')
        x, y = pos[name]
        geoms.append(f'<geom name="{name}" type="box" size="0.4 0.4 0.4" pos="{x} {y} 0" material="m_{name}" group="2" contype="0" conaffinity="0"/>')
    return f"""<mujoco><asset>{''.join(mats)}</asset><worldbody>
      <light directional="true" pos="0 0 0" dir="0 0 -1" diffuse="1 1 1"/>
      <camera name="cam" pos="0 0 3" fovy="{np.rad2deg(2 * np.arctan(10 / 24))}"/>
      {''.join(geoms)}</worldbody></mujoco>"""


def test_tier0_render_albedo(tmp_path):
    wp = pytest.importorskip("warp")
    if not wp.is_metal_available():
        pytest.skip("needs Metal")
    from metalsim.render.tier0 import Tier0Renderer
    from metalsim.scene.usd_to_mjcf import load_usd
    _, scene = _write_assets(tmp_path)
    m = load_usd(scene, lossless=False).compile()
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    W = 128
    rend = Tier0Renderer(m, 1, width=W, height=W, camera="cam", outputs=("rgb",))
    img = rend.render_host([d])["rgb"][0].astype(float)
    # expected parameters written by hand from the document (not read back from the importer)
    exp = {"tex": dict(rgba=(0.9, 0.45, 0.45, 1), roughness=0.4, metallic=0.3, specular=0.5, emission=0, tex=True, repeat=(2, 3)),
           "green": dict(rgba=(0.08, 0.48, 0.16, 1), roughness=0.25, metallic=0, specular=_f0(1.6) / 0.08, emission=0.125),
           "prev": dict(rgba=(0.5, 0.5, 0.5, 1), roughness=0.6, metallic=1.0, specular=0.5, emission=0, tex=True, repeat=(1, 1)),
           "inline": dict(rgba=(0.2, 0.3, 0.9, 1), roughness=0.6, metallic=0, specular=0.5, emission=0)}
    m_ref = mujoco.MjModel.from_xml_string(_expected_mjcf(tmp_path, exp))
    d_ref = mujoco.MjData(m_ref); mujoco.mj_forward(m_ref, d_ref)
    ref = Tier0Renderer(m_ref, 1, width=W, height=W, camera="cam", outputs=("rgb",)).render_host([d_ref])["rgb"][0].astype(float)
    # compare the top faces (quadrant centres, away from edges)
    q = {"tex": (W // 4, W // 4), "green": (W // 4, 3 * W // 4), "prev": (3 * W // 4, W // 4), "inline": (3 * W // 4, 3 * W // 4)}
    for name, (r, c) in q.items():
        a = img[r - 12:r + 12, c - 12:c + 12].reshape(-1, 3).mean(0)
        b = ref[r - 12:r + 12, c - 12:c + 12].reshape(-1, 3).mean(0)
        print(name, np.round(a, 1), np.round(b, 1))
        np.testing.assert_allclose(a, b, atol=2.0, err_msg=name)
    # the constant materials render with their albedo's hue
    g = img[q["green"][0] - 8:q["green"][0] + 8, q["green"][1] - 8:q["green"][1] + 8].reshape(-1, 3).mean(0)
    assert g[1] > 2 * g[0] and g[1] > 2 * g[2]
    b = img[q["inline"][0] - 8:q["inline"][0] + 8, q["inline"][1] - 8:q["inline"][1] + 8].reshape(-1, 3).mean(0)
    assert b[2] > b[1] > b[0]
