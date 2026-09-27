"""Measure Kit/RTX's DomeLight latlong mapping (image (u, v) -> world direction) on a z-up Isaac stage.

Recipe (all synthetic, generated here with numpy):
  * equirectangular 1024 x 512 RGBE .hdr map, black except three bright squares (24 px) at known file-pixel
    centres: RED at (u, v) = (0.10, 0.25), GREEN at (0.35, 0.25), BLUE at (0.60, 0.10); u = column / width
    (0 = left column of the file), v = row / height (0 = top row of the file). Three non-collinear markers pin the
    pole axis, the azimuth zero AND the handedness (a mirrored map would reverse RED->GREEN's azimuth sense).
  * UsdLux.DomeLight (texture:format latlong, intensity 1, exposure 0), a white diffuse ground plane (z = 0) and a
    white diffuse sphere r = 0.25 m at (0, 0, 0.25).
  * Six 90-degree cameras (cube faces +x -x +y -y +z -z, 256 x 256) at (5, 5, 2) see the dome background directly:
    each marker's centroid pixel is back-projected to a world direction with the camera's authored pose.
  * One top-down camera at (0, 0, 6) looking -z (image up = world +y) renders the sphere's shadows on the ground
    (path tracing): each coloured shadow lies opposite its marker's horizontal direction (a lighting check that the
    illumination uses the same mapping as the background).
  * Variants: dome un-rotated; dome rotateZ +90 deg; dome rotateX +90 deg (UsdLux OrientToStageUpAxis-style); un-rotated again.
Renderer: RTX path tracing (/rtx/rendermode = PathTracing, 64 spp).  Our adopted convention (rendering_vs_rtx
section 6.1): pole +z, u = 0.5 - (atan2(x, -y) - yaw) / 2 pi, v = acos(z) / pi.

    python record_dome_azimuth.py --out ~/parity3/dome_il2      (Isaac Sim python; SimulationApp headless)
"""
import argparse, os, json, math
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--spp", type=int, default=64)
args, _ = parser.parse_known_args()
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 256, "height": 256})

import numpy as np
import carb, omni.usd, omni.kit.app
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdLux, UsdShade, Sdf, Gf

os.makedirs(args.out, exist_ok=True)
W, H, S = 1024, 512, 24
MARKERS = {"red": ((0.10, 0.25), (1, 0, 0)), "green": ((0.35, 0.25), (0, 1, 0)), "blue": ((0.60, 0.10), (0, 0, 1))}
img = np.zeros((H, W, 3), np.float32)
for name, ((u, v), c) in MARKERS.items():
    cx, cy = int(round(u * W)), int(round(v * H))
    img[cy - S // 2:cy + S // 2, cx - S // 2:cx + S // 2] = np.array(c, np.float32) * 200.0

def write_hdr(path, a):
    """Radiance RGBE, top row first ('-Y H +X W')."""
    m = a.max(-1); e = np.zeros(m.shape, np.int32); mant = np.zeros_like(m)
    nz = m > 1e-32; mant[nz], e[nz] = np.frexp(m[nz])
    sc = np.where(nz, mant * 256.0 / np.where(nz, m, 1), 0)
    rgbe = np.zeros(a.shape[:2] + (4,), np.uint8)
    rgbe[..., :3] = np.clip(a * sc[..., None], 0, 255).astype(np.uint8); rgbe[..., 3] = np.where(nz, e + 128, 0).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {a.shape[0]} +X {a.shape[1]}\n".encode()); f.write(rgbe.tobytes())
hdr_path = os.path.join(args.out, "markers_latlong.hdr"); write_hdr(hdr_path, img)
np.save(os.path.join(args.out, "markers_latlong.npy"), img)
try:
    import imageio.v2 as iio
    iio.imwrite(os.path.join(args.out, "markers_latlong_preview.png"), (np.clip(img, 0, 1) * 255).astype(np.uint8))
except Exception as e:
    print("[dome] preview png failed", e)

ctx = omni.usd.get_context(); ctx.new_stage()
for _ in range(5): simulation_app.update()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateTextureFileAttr(hdr_path); dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
dome.CreateIntensityAttr(1.0); dome.CreateExposureAttr(0.0)
dxf = UsdGeom.Xformable(dome); rot_op = None

def white_mat(path):
    m = UsdShade.Material.Define(stage, path); sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface"); sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0); sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface"); return m
mat = white_mat("/World/Looks/white")
ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
L = 20.0
ground.CreatePointsAttr([(-L, -L, 0), (L, -L, 0), (L, L, 0), (-L, L, 0)]); ground.CreateFaceVertexCountsAttr([4]); ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
ground.CreateNormalsAttr([(0, 0, 1)] * 4); ground.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(mat)
sph = UsdGeom.Sphere.Define(stage, "/World/Sphere"); sph.CreateRadiusAttr(0.25); UsdGeom.Xformable(sph).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.25))
UsdShade.MaterialBindingAPI.Apply(sph.GetPrim()).Bind(mat)

def look_rot(fwd, up):
    f = np.array(fwd, float); f /= np.linalg.norm(f); r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    return np.stack([r, u, -f], 1)   # columns: camera local +x, +y, +z(back) in world
def add_cam(path, pos, R, aperture=20.0, focal=10.0):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateHorizontalApertureAttr(aperture); cam.CreateVerticalApertureAttr(aperture); cam.CreateFocalLengthAttr(focal)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1e6))
    M = np.eye(4); M[:3, :3] = R.T; M[3, :3] = pos      # USD row-vector convention
    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(*M.flatten().tolist()))
    return cam
CUBE_POS = np.array([5.0, 5.0, 2.0])
faces = {"px": ([1, 0, 0], [0, 0, 1]), "nx": ([-1, 0, 0], [0, 0, 1]), "py": ([0, 1, 0], [0, 0, 1]), "ny": ([0, -1, 0], [0, 0, 1]),
         "pz": ([0, 0, 1], [0, 1, 0]), "nz": ([0, 0, -1], [0, 1, 0])}
cams = {}
for fn, (fw, up) in faces.items():
    R = look_rot(fw, up); add_cam(f"/World/Cam_{fn}", CUBE_POS, R); cams[fn] = R
TOP_R = look_rot([0, 0, -1], [0, 1, 0]); add_cam("/World/Cam_top", np.array([0, 0, 6.0]), TOP_R, aperture=20.0, focal=14.0)

s = carb.settings.get_settings()
s.set("/rtx/rendermode", "PathTracing"); s.set("/rtx/pathtracing/spp", args.spp); s.set("/rtx/pathtracing/totalSpp", args.spp)
s.set("/rtx/pathtracing/clampSpp", args.spp); s.set("/rtx/post/aa/op", 0)
s.set("/rtx/post/tonemap/op", 0)   # linear-ish, markers saturate anyway
for _ in range(20): simulation_app.update()
rps = {}; anns = {}
for fn in list(faces) + ["top"]:
    res = (256, 256) if fn != "top" else (512, 512)
    rps[fn] = rep.create.render_product(f"/World/Cam_{fn}", res)
    anns[fn] = rep.AnnotatorRegistry.get_annotator("rgb"); anns[fn].attach([rps[fn]])

def cam_dir(R, px, py, w, h, aperture=20.0, focal=10.0):
    t = aperture / 2 / focal
    x = (2 * (px + 0.5) / w - 1) * t; y = (1 - 2 * (py + 0.5) / h) * t
    d = R @ np.array([x, y, -1.0]); return d / np.linalg.norm(d)
def ours(u, v, yaw=0.0):
    phi = (0.5 - u) * 2 * math.pi + yaw; th = v * math.pi
    return np.array([math.sin(th) * math.sin(phi), -math.sin(th) * math.cos(phi), math.cos(th)])

import imageio.v2 as iio
results = {}
for _ in range(3): rep.orchestrator.step(rt_subframes=4)   # warm-up: on Isaac Sim 6.1 the first frames came back before the dome texture was loaded
for variant, op in [("none", None), ("rotZ90", ("Z", 90.0)), ("rotX90", ("X", 90.0)), ("none_repeat", None)]:
    dxf.ClearXformOpOrder()
    for p in list(dome.GetPrim().GetAttributes()):
        if p.GetName().startswith("xformOp:"): dome.GetPrim().RemoveProperty(p.GetName())
    if op is not None:
        (dxf.AddRotateZOp() if op[0] == "Z" else dxf.AddRotateXOp()).Set(op[1])
    for _ in range(10): simulation_app.update()
    rep.orchestrator.step(rt_subframes=max(4, args.spp // 4))
    rep.orchestrator.step(rt_subframes=max(4, args.spp // 4))
    found = {k: [] for k in MARKERS}
    for fn in list(faces) + ["top"]:
        a = np.asarray(anns[fn].get_data())[..., :3].astype(np.float32)
        iio.imwrite(os.path.join(args.out, f"{variant}_{fn}.png"), a.astype(np.uint8))
        if fn == "top": continue
        for k, (_, c) in MARKERS.items():
            ci = int(np.argmax(c)); others = [i for i in range(3) if i != ci]
            m = (a[..., ci] > 128) & (a[..., others[0]] < 0.5 * a[..., ci]) & (a[..., others[1]] < 0.5 * a[..., ci])
            if m.sum() >= 3:
                ys, xs = np.nonzero(m); wts = a[ys, xs, ci]
                found[k].append((int(m.sum()), fn, float((xs * wts).sum() / wts.sum()), float((ys * wts).sum() / wts.sum())))
    res = {}
    for k, lst in found.items():
        if not lst: res[k] = None; continue
        # a marker straddling faces: pixel-count-weighted mean of the per-face directions
        ds = [cam_dir(cams[fn], x, y, 256, 256) * n for n, fn, x, y in lst]; d = np.sum(ds, 0); d /= np.linalg.norm(d)
        (u, v), _ = MARKERS[k]
        yaw = math.radians(op[1]) if op and op[0] == "Z" else 0.0
        pred = ours(u, v, yaw)
        res[k] = {"faces": lst, "measured_dir": d.tolist(), "ours_pred_dir": pred.tolist(),
                  "angle_err_deg": float(math.degrees(math.acos(np.clip(d @ pred, -1, 1)))),
                  "measured_elev_deg": float(math.degrees(math.asin(d[2]))), "measured_atan2_x_negy_deg": float(math.degrees(math.atan2(d[0], -d[1])))}
    # shadow check in the top view: darkest region per channel deficit around the sphere
    top = np.asarray(anns["top"].get_data())[..., :3].astype(np.float32)
    res["top_note"] = "see top png; shadows opposite each marker's horizontal direction"
    results[variant] = res
    print(f"[dome] {variant}:", json.dumps(res), flush=True)
recipe = {"map": {"W": W, "H": H, "square_px": S, "markers_uv": {k: v[0] for k, v in MARKERS.items()}, "radiance": 200.0,
                  "uv_def": "u = column/width (0 = left of file), v = row/height (0 = top row of file)", "file": "markers_latlong.hdr (RGBE, -Y H +X W)"},
          "cube_cams": {"pos": CUBE_POS.tolist(), "fov_deg": 90, "res": 256, "R_cols_right_up_back": {k: v.tolist() for k, v in cams.items()}},
          "top_cam": {"pos": [0, 0, 6], "R": TOP_R.tolist(), "aperture": 20, "focal": 14, "res": 512},
          "renderer": {"rendermode": "PathTracing", "spp": args.spp}, "results": results,
          "kit_version": omni.kit.app.get_app().get_build_version()}
json.dump(recipe, open(os.path.join(args.out, "dome_results.json"), "w"), indent=1)
open(os.path.join(args.out, "DONE"), "w").write("ok")
simulation_app.close()
