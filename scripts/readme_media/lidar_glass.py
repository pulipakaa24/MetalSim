"""README animation: what metalsim's ray-traced sensors do with glass.

One scene, three views, each its own 4:3 frame sequence (150 frames at 30 fps, a 5 s loop; rendered at 640 x 480 and
meant to be shown at 320 x 240, as one row of the README's table):
  rgb    tier-2 path tracer (Tier2Renderer: OmniPBR materials, Poly Haven studio HDRI dome, OIDN, ACES) from the
         sensor's camera; alpha is surface presence, so the glass shows as a tinted, reflecting sheet you see through;
  depth  the ray-traced depth camera (RayTracer.make_depth_camera, transparent_ids="surface", the default) from the same
         camera, pixel for pixel: the first surface with alpha > 0, as MuJoCo's renderer / mj_ray and Isaac's depth on
         glass, so the pane hides what is behind it; inferno, near = bright, fixed 0.4-2.8 m;
  lidar  a 32 x 1024 lidar with two returns (make_lidar, transparency="transmit", the default, max_returns=2, as
         tests/test_sensor_transparency.py), drawn from a fixed 3/4 view: echoes on a pane in cyan (blended over what
         lies behind them), every other echo coloured by range on the depth view's scale; the ring is the sensor.
         Two settings differ from the defaults: hits_per_ray=1 (the default, max_returns, lets a beam pass an opaque
         hit at full power for a second echo, the divergence model of rt.metal lidar_ext; pane passes do not count
         against it) and min_echo_sep=0.1 m (the default 0.4 m merges the pane's echo with the ball's 0.5 m behind it).
The scene is the sensor-transparency diagnostic's (scripts/diagnostics/sensor_transparency_cost.py: a sensor among a
box, cylinder, capsule and sphere, three alpha-0.3 glass panes), rearranged for a camera, with a darker pane colour
(tier 2 only; the sensors read alpha and reflectance); the sphere moves from open floor to behind the front pane and
back.

    scripts/gpu_run.sh lidar-glass render 10 -- .venv/bin/python scripts/readme_media/lidar_glass.py --compare 75

Writes runs/readme_media/assets/lidar_glass_{rgb,depth,lidar}_frames/frame_%04d.png and lidar_glass_{rgb,depth,lidar}.mp4
(H.264, yuv420p, CRF 20, 640 x 480) beside them; into runs/readme_media/scratch/lidar-glass/: lidar_glass.json
(settings, exposure, per-frame timings), preview_320.mp4 (the three views side by side at 320 x 240 each) and, with
``--compare K``, frame K with and without the panes at 320 x 240. ``--no-panes``, ``--frames``, ``--out`` and
``--dump`` are for previews. The path tracer's noise depends on the frame counter, so a frame is bit-identical to the
full run's only when rendered in the same order. Needs the main venv (Metal, pyoidn, Pillow) and the Poly Haven train
assets (scripts/fetch_polyhaven_assets.sh train).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
PH = os.environ.get("POLYHAVEN", os.path.join(ROOT, "assets", "polyhaven"))
ASSETS = os.path.join(ROOT, "runs", "readme_media", "assets")
SCRATCH = os.path.join(ROOT, "runs", "readme_media", "scratch", "lidar-glass")
VIEWS = ("rgb", "depth", "lidar")

# matplotlib's "inferno" (CC0; van der Walt, Smith, Firing), every 8th of its 256 entries (sRGB)
INFERNO = np.array([
    (0.0015, 0.0005, 0.0139), (0.014, 0.0112, 0.0719), (0.0423, 0.0281, 0.1411), (0.082, 0.0433, 0.2153),
    (0.1293, 0.0473, 0.2908), (0.1834, 0.0403, 0.355), (0.2383, 0.0366, 0.3964), (0.2908, 0.0456, 0.4186),
    (0.3415, 0.0623, 0.4294), (0.3915, 0.0809, 0.4331), (0.4412, 0.0993, 0.4316), (0.491, 0.1172, 0.4256),
    (0.5409, 0.1347, 0.4151), (0.5907, 0.1526, 0.4003), (0.6401, 0.1714, 0.3811), (0.6887, 0.1922, 0.3576),
    (0.7357, 0.2159, 0.3302), (0.7751, 0.2397, 0.3035), (0.8173, 0.271, 0.2704), (0.856, 0.307, 0.2351),
    (0.8903, 0.348, 0.1983), (0.9199, 0.3934, 0.1601), (0.9443, 0.4428, 0.1204), (0.9634, 0.4955, 0.0791),
    (0.9771, 0.5508, 0.0391), (0.9853, 0.6084, 0.0242), (0.9879, 0.6677, 0.0583), (0.9849, 0.7284, 0.1208),
    (0.9761, 0.79, 0.196), (0.9625, 0.8515, 0.2855), (0.9487, 0.9105, 0.3953), (0.9517, 0.9606, 0.5242),
    (0.9884, 0.9984, 0.6449)], np.float32)

# --- scene ------------------------------------------------------------------------------------------------------
SENSOR = np.array([0.0, 0.0, 0.5])      # lidar site and camera (one rig, a mocap body)
PITCH_DEG = 10.0                        # camera tilted down
FOVY = 45.0
GLASS_ALPHA = 0.3                       # the diagnostic's panes (its rgba: 0.6 0.8 0.9 0.3)
# panes: name, centre, half sizes (thickness, length, height), yaw (deg)
PANES = [("pane_front", (0.65, 0.40, 0.30), (0.004, 0.45, 0.30), 0.0),
         ("pane_left", (-0.10, 0.95, 0.30), (0.004, 0.50, 0.30), 90.0),
         ("pane_back", (-0.60, -0.45, 0.30), (0.004, 0.40, 0.30), 45.0)]
BALL_X, BALL_R, BALL_Y0, BALL_A = 1.15, 0.13, 0.06, 0.39   # y(t) = Y0 - A cos(2 pi t / T): open floor -> behind the pane


def scene_xml(panes: bool = True, glass_rgb=(0.6, 0.8, 0.9)) -> str:
    floor = os.path.join(PH, "tex", "train", "laminate_floor_02.png")
    if not os.path.exists(floor):
        raise SystemExit(f"missing {floor}: run scripts/fetch_polyhaven_assets.sh train")
    p = np.deg2rad(PITCH_DEG)
    pane_xml = "".join(
        f'<geom name="{n}" type="box" pos="{c[0]} {c[1]} {c[2]}" size="{s[0]} {s[1]} {s[2]}" euler="0 0 {yaw}" '
        f'material="glass"/>\n' for n, c, s, yaw in PANES) if panes else ""
    return f"""
<mujoco model="lidar_glass">
  <visual><headlight active="0"/></visual>
  <asset>
    <texture name="floor" type="2d" file="{floor}" colorspace="sRGB"/>
    <material name="floor" texture="floor" texrepeat="1 1" texuniform="true" roughness="0.55" specular="0.3"/>
    <material name="wall" rgba="0.78 0.78 0.76 1" roughness="0.85" specular="0.2"/>
    <material name="glass" rgba="{glass_rgb[0]} {glass_rgb[1]} {glass_rgb[2]} {GLASS_ALPHA}" roughness="0.05" specular="1"/>
    <material name="red" rgba="0.80 0.16 0.12 1" roughness="0.35" specular="0.5"/>
    <material name="blue" rgba="0.16 0.30 0.80 1" roughness="0.35" specular="0.5"/>
    <material name="yellow" rgba="0.90 0.72 0.12 1" roughness="0.35" specular="0.5"/>
    <material name="green" rgba="0.18 0.66 0.26 1" roughness="0.30" specular="0.5"/>
    <material name="purple" rgba="0.50 0.24 0.66 1" roughness="0.35" specular="0.5"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor"/>
    <geom name="wall" type="box" pos="2.3 0 0.6" size="0.05 2.6 0.6" material="wall"/>
    <body name="rig" mocap="true" pos="{SENSOR[0]} {SENSOR[1]} {SENSOR[2]}">
      <site name="lidar"/>
      <camera name="cam" xyaxes="0 -1 0 {np.sin(p):.6f} 0 {np.cos(p):.6f}" fovy="{FOVY}"/>
    </body>
    {pane_xml}
    <geom name="box" type="box" pos="1.40 0.30 0.2" size="0.13 0.13 0.2" euler="0 0 20" material="red"/>
    <geom name="capsule" type="capsule" pos="1.35 -0.55 0.27" size="0.07 0.2" material="yellow"/>
    <geom name="cylinder" type="cylinder" pos="-0.15 1.45 0.25" size="0.13 0.25" material="blue"/>
    <geom name="cube" type="box" pos="-1.0 -0.85 0.14" size="0.14 0.14 0.14" euler="0 0 30" material="purple"/>
    <body name="ball" mocap="true" pos="{BALL_X} {BALL_Y0 - BALL_A} {BALL_R}">
      <geom name="sphere" type="sphere" size="{BALL_R}" material="green"/>
    </body>
  </worldbody>
</mujoco>"""


def ball_pos(k: int, n_frames: int) -> np.ndarray:
    return np.array([BALL_X, BALL_Y0 - BALL_A * np.cos(2 * np.pi * k / n_frames), BALL_R])


def lidar_beams(channels=32, el=(-25.0, 15.0), columns=1024):
    A, E = np.meshgrid(np.linspace(-np.pi, np.pi, columns, endpoint=False), np.deg2rad(np.linspace(*el, channels)))
    return A.ravel().astype(np.float32), E.ravel().astype(np.float32)


# --- drawing ----------------------------------------------------------------------------------------------------
def cmap(t, lo=0.10, hi=1.0):
    """inferno at lo + (hi - lo) * t (t in [0, 1]); (..., 3) float32 sRGB."""
    x = (lo + (hi - lo) * np.clip(t, 0.0, 1.0)) * (len(INFERNO) - 1)
    i = np.minimum(np.floor(x).astype(int), len(INFERNO) - 2)
    f = (x - i)[..., None]
    return INFERNO[i] * (1 - f) + INFERNO[i + 1] * f


def range_colour(r, rng):
    """Near = bright: the depth panel's and the lidar's shared scale."""
    return cmap(1.0 - (np.asarray(r) - rng[0]) / (rng[1] - rng[0]))


def aces_srgb(hdr, exposure):
    """The film's display transform (metalsim/render/film/render.py): exposure, ACES fit, sRGB."""
    x = np.clip(hdr * exposure, 0, None)
    x = np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def depth_panel(depth, rng, bg):
    img = range_colour(depth, rng)
    img[depth <= 0] = bg
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    f = np.asarray(target, float) - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    return r, np.cross(r, f), f


def splat_layer(points, colours, W, H, eye, target, fovy, radius=1.0, ss=3, fog=0.45, zrange=None):
    """Perspective scatter of discs of ``radius`` output pixels on a ``ss`` x supersampled canvas, z-buffered (nearest
    point per pixel). Colours dimmed by up to ``fog`` with distance from the eye over ``zrange`` (default the points').
    Returns (colour (H ss, W ss, 3), depth (H ss, W ss), inf where empty)."""
    eye = np.asarray(eye, float)
    r, u, f = look_at(eye, target)
    rel = points - eye
    x, y, z = rel @ r, rel @ u, rel @ f
    ok = z > 0.05
    x, y, z, col = x[ok], y[ok], z[ok], colours[ok]
    Ws, Hs = W * ss, H * ss
    canvas = np.zeros((Hs * Ws, 3), np.float32)
    zbuf = np.full(Hs * Ws, np.inf, np.float32)
    if len(z) == 0:
        return canvas.reshape(Hs, Ws, 3), zbuf.reshape(Hs, Ws)
    fp = 0.5 * Hs / np.tan(np.deg2rad(fovy) / 2)
    px = Ws / 2 + fp * x / z
    py = Hs / 2 - fp * y / z
    z0, z1 = zrange if zrange is not None else (z.min(), z.max())
    col = col * (1.0 - fog * np.clip((z - z0) / max(z1 - z0, 1e-6), 0, 1))[:, None]
    R = radius * ss
    k = int(np.ceil(R))
    oy, ox = np.mgrid[-k:k + 1, -k:k + 1]
    disc = (ox ** 2 + oy ** 2) <= R * R
    ox, oy = ox[disc], oy[disc]
    X = (np.round(px)[:, None] + ox[None]).astype(np.int64).ravel()
    Y = (np.round(py)[:, None] + oy[None]).astype(np.int64).ravel()
    Z = np.repeat(z, len(ox))
    idx = np.repeat(np.arange(len(z)), len(ox))
    inside = (X >= 0) & (X < Ws) & (Y >= 0) & (Y < Hs)
    pix = Y[inside] * Ws + X[inside]
    Z, idx = Z[inside], idx[inside]
    order = np.lexsort((Z, pix))                       # per pixel, nearest first
    pix, idx, Z = pix[order], idx[order], Z[order]
    first = np.ones(len(pix), bool); first[1:] = pix[1:] != pix[:-1]
    canvas[pix[first]] = col[idx[first]]
    zbuf[pix[first]] = Z[first]
    return canvas.reshape(Hs, Ws, 3), zbuf.reshape(Hs, Ws)


def cloud_panel(points, colours, glass, W, H, eye, target, fovy, bg, radius=1.0, glass_alpha=0.55, ss=3,
                glass_radius=None):
    """The lidar panel: opaque echoes z-buffered, the glass echoes blended over them where nearer (alpha
    ``glass_alpha``), so what the lidar sees behind a pane stays visible through the pane's own points."""
    rel = points - np.asarray(eye, float)
    zz = rel @ look_at(np.asarray(eye, float), target)[2]
    zr = (np.percentile(zz, 1), np.percentile(zz, 99))
    c0, z0 = splat_layer(points[~glass], colours[~glass], W, H, eye, target, fovy, radius, ss, zrange=zr)
    c1, z1 = splat_layer(points[glass], colours[glass], W, H, eye, target, fovy,
                         radius if glass_radius is None else glass_radius, ss, zrange=zr)
    img = np.where(np.isfinite(z0)[..., None], c0, np.asarray(bg, np.float32))
    a = (np.isfinite(z1) & (z1 < z0))[..., None] * np.float32(glass_alpha)
    img = img * (1 - a) + c1 * a
    return img.reshape(H, ss, W, ss, 3).mean((1, 3))


def marker(img, p, eye, target, fovy, colour=(1.0, 1.0, 1.0), radius=4.0, ring=1.4):
    """A ring at world point p (the sensor) on a float image."""
    H, W = img.shape[:2]
    r, u, f = look_at(np.asarray(eye, float), target)
    rel = np.asarray(p, float) - eye
    z = rel @ f
    fp = 0.5 * H / np.tan(np.deg2rad(fovy) / 2)
    cx, cy = W / 2 + fp * (rel @ r) / z, H / 2 - fp * (rel @ u) / z
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
    a = np.clip(ring / 2 + 0.5 - np.abs(d - radius), 0, 1)[..., None]
    a = np.maximum(a, np.clip(1.6 - d, 0, 1)[..., None])           # and a dot in the middle
    img[:] = img * (1 - a) + np.asarray(colour, np.float32) * a


# --- render -----------------------------------------------------------------------------------------------------
class Stage:
    """The scene with its renderer (tier 2) and sensors (a RayTracer sharing the Metal context)."""

    def __init__(self, a, panes=True):
        import mujoco
        import warp as wp
        wp.config.quiet = True
        from metalsim.render.hdr import load_hdr
        from metalsim.render.tier2 import Tier2Renderer
        from metalsim.sensors.raytrace import RayTracer

        self.a, self.W, self.H = a, a.width, a.height
        self.m = mujoco.MjModel.from_xml_string(scene_xml(panes, a.glass_rgb))
        self.d = mujoco.MjData(self.m)
        self.ball = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.mocap = int(self.m.body_mocapid[self.ball])
        vis = mujoco.MjvOption(); vis.sitegroup[:] = 0          # the lidar site is not a surface
        self.r = Tier2Renderer(self.m, 1, width=self.W, height=self.H, camera="cam", spp=a.spp, max_bounces=a.bounces,
                               material_model="omnipbr", headlight=False, tonemap="rtx", exposure=1.0,
                               denoise="oidn", aux=True, visual=vis)
        hdr = load_hdr(os.path.join(PH, a.hdri))
        self.r.set_environment(hdr, yaw=np.deg2rad(a.hdri_yaw))
        self.rt = RayTracer(self.m, 1, max_range=a.max_range, ctx=self.r.ctx)
        az, el = lidar_beams(a.channels, (a.el_min, a.el_max), a.columns)
        self.lidar = self.rt.make_lidar("lidar", az, el, name="lidar", max_returns=2, hits_per_ray=1,
                                        min_echo_sep=a.min_echo_sep)
        self.depth = self.rt.make_depth_camera("cam", self.W, self.H, name="depth")
        geoms = np.asarray(self.rt.tables.geoms)
        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in geoms]
        self.glass_slot = np.zeros(self.rt.G + 1, bool)
        self.glass_slot[[k for k, nm in enumerate(names) if nm.startswith("pane")]] = True

    def frame(self, k):
        import mujoco
        a = self.a
        self.d.mocap_pos[self.mocap] = ball_pos(k, a.n_frames)
        mujoco.mj_forward(self.m, self.d)
        t0 = time.perf_counter()
        out = self.rt.trace_host([self.d], [self.lidar, self.depth])
        t1 = time.perf_counter()
        img = self.r.render_host([self.d], passes=a.passes, chunk=8)
        t2 = time.perf_counter()
        return out, img, dict(sensors_s=t1 - t0, rgb_s=t2 - t1)


def panels(a, out, img, exposure, glass_slot):
    """The three views (uint8, ``a.height`` x ``a.width`` x 3) and echo counts. ``glass_slot``: per ray-tracer slot,
    True for the panes (their lidar echoes are drawn as glass)."""
    bg = np.asarray(a.bg, np.float32)
    rgb = aces_srgb(np.asarray(img["hdr_denoised"])[0, ..., :3], exposure)
    dep = depth_panel(out["depth"][0], a.range, bg)
    L = out["lidar"]
    n = L["n_returns"][0]
    valid = np.arange(2)[None, :] < n[:, None]                     # (beams, 2): echo k of beam b exists
    P = L["points"][0][valid]                                      # env frame (= world for env 0)
    glass = glass_slot[L["slot"][0][valid]]
    col = range_colour(L["range"][0][valid], a.range)
    col[glass] = a.glass_dot_rgb
    cloud = cloud_panel(P, col.astype(np.float32), glass, a.width, a.height, a.eye, a.target, a.view_fovy, bg,
                        radius=a.dot, glass_alpha=a.glass_alpha)
    marker(cloud, SENSOR, a.eye, a.target, a.view_fovy, radius=a.marker, ring=a.marker * 0.35)
    cloud = (np.clip(cloud, 0, 1) * 255 + 0.5).astype(np.uint8)
    stats = dict(echoes=int(valid.sum()), glass_echoes=int(glass.sum()), two_return_beams=int((n == 2).sum()))
    return dict(rgb=rgb, depth=dep, lidar=cloud), stats


def small(img, size=(320, 240)):
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize(size, Image.LANCZOS))


def row(views, gap=4):
    """Views side by side at 320 x 240 (as the README's table shows them), ``gap`` px apart."""
    tiles = [small(views[v]) for v in VIEWS]
    sep = np.full((tiles[0].shape[0], gap, 3), 255, np.uint8)
    return np.concatenate([tiles[0], sep, tiles[1], sep, tiles[2]], 1)


def parse_frames(spec, n):
    if not spec:
        return list(range(n))
    if ":" in spec:
        p = [int(x) for x in spec.split(":")]
        return list(range(p[0], min(p[1], n), p[2] if len(p) > 2 else 1))
    return [int(x) for x in spec.split(",")]


def auto_exposure(img):
    """The film's rule (metalsim/render/film/render.py): median luminance of the first frame to 0.2 before ACES."""
    hdr = np.asarray(img["hdr_denoised"])[0, ..., :3]
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722])
    return 0.20 / max(float(np.median(lum)), 1e-6)


def ffmpeg():
    return shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def encode(frames_dir, mp4, fps, crf):
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frames_dir, "frame_%04d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4]
    subprocess.run(cmd, check=True)
    return cmd


def encode_preview(dirs, mp4, fps, crf, gap=4):
    """The three views side by side, each scaled to 320 x 240 (lanczos), for checking how the row reads."""
    ins = sum([["-framerate", str(fps), "-i", os.path.join(d, "frame_%04d.png")] for d in dirs], [])
    fc = (";".join(f"[{i}]scale=320:240:flags=lanczos,pad={320 + (gap if i < 2 else 0)}:240:0:0:white[v{i}]"
                   for i in range(3)) + ";[v0][v1][v2]hstack=inputs=3")
    cmd = [ffmpeg(), "-y", "-loglevel", "error", *ins, "-filter_complex", fc, "-c:v", "libx264", "-crf", str(crf),
           "-pix_fmt", "yuv420p", mp4]
    subprocess.run(cmd, check=True)
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=ASSETS, help="directory of the lidar_glass_<view>_frames/ dirs and the mp4s")
    ap.add_argument("--no-encode", dest="encode", action="store_false")
    ap.add_argument("--frames", default="", help="start:stop[:step] or k,k,... (default all; no encode)")
    ap.add_argument("--n-frames", type=int, default=150, help="loop length (the ball's period)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--passes", type=int, default=32, help="tier-2 passes of --spp samples per pixel")
    ap.add_argument("--spp", type=int, default=4)
    ap.add_argument("--bounces", type=int, default=4)
    ap.add_argument("--exposure", type=float, default=0.0, help="0 = automatic from the first rendered frame")
    ap.add_argument("--hdri", default="hdri/train/poly_haven_studio.hdr")
    ap.add_argument("--hdri-yaw", type=float, default=0.0)
    ap.add_argument("--glass-rgb", type=float, nargs=3, default=(0.35, 0.48, 0.55),
                    help="pane colour (tier 2 only: the sensors see alpha and reflectance, not colour)")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--columns", type=int, default=1024)
    ap.add_argument("--el-min", type=float, default=-25.0)
    ap.add_argument("--el-max", type=float, default=15.0)
    ap.add_argument("--min-echo-sep", type=float, default=0.1, help="minDistBetweenEchosM (0.4 merges pane + ball)")
    ap.add_argument("--max-range", type=float, default=10.0)
    ap.add_argument("--range", type=float, nargs=2, default=(0.4, 2.8), help="colour scale of depth and lidar range [m]")
    ap.add_argument("--eye", type=float, nargs=3, default=(-0.8, -0.35, 1.3), help="lidar view: eye")
    ap.add_argument("--target", type=float, nargs=3, default=(1.2, 0.05, 0.2), help="lidar view: target")
    ap.add_argument("--view-fovy", type=float, default=33.0)
    ap.add_argument("--dot", type=float, default=1.8, help="lidar dot radius [px at --width]")
    ap.add_argument("--marker", type=float, default=7.0, help="sensor ring radius [px at --width]")
    ap.add_argument("--glass-alpha", type=float, default=0.6, help="opacity of the glass echoes' dots")
    ap.add_argument("--no-panes", action="store_true", help="the comparison scene: same, without the glass")
    ap.add_argument("--compare", type=int, default=-1, help="also render frame K without the panes (into scratch)")
    ap.add_argument("--dump", default="", help="also save each frame's raw sensor outputs and HDR image (npz) here")
    a = ap.parse_args(argv)
    a.bg, a.glass_dot_rgb = (0.035, 0.035, 0.045), (0.30, 0.88, 1.0)
    dirs = {v: os.path.join(a.out, f"lidar_glass_{v}_frames") for v in VIEWS}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    from PIL import Image

    t_job = time.perf_counter()
    stage = Stage(a, panes=not a.no_panes)
    setup_s = time.perf_counter() - t_job
    frames = parse_frames(a.frames, a.n_frames)
    exposure = a.exposure
    rows, t_frames = [], time.perf_counter()
    for i, k in enumerate(frames):
        tf = time.perf_counter()
        out, img, st = stage.frame(k)
        if exposure <= 0:
            exposure = auto_exposure(img)
            print(f"[lidar_glass] automatic exposure {exposure:.4f}", flush=True)
        views, cs = panels(a, out, img, exposure, stage.glass_slot)
        for v in VIEWS:
            Image.fromarray(views[v]).save(os.path.join(dirs[v], f"frame_{k:04d}.png"))
        if a.dump:
            os.makedirs(a.dump, exist_ok=True)
            np.savez_compressed(os.path.join(a.dump, f"frame_{k:04d}.npz"), depth=out["depth"][0],
                                hdr=np.asarray(img["hdr_denoised"])[0, ..., :3], glass_slot=stage.glass_slot,
                                exposure=exposure, **{f"lidar_{f}": v[0] for f, v in out["lidar"].items()})
        st.update(cs, frame=k, seconds=time.perf_counter() - tf)
        rows.append(st)
        if i == 0 or k % 15 == 0:
            print(f"[lidar_glass] frame {k}: {st['seconds']:.2f} s (rgb {st['rgb_s']:.2f}, sensors {st['sensors_s']:.3f}); "
                  f"{cs['echoes']} echoes, {cs['glass_echoes']} on glass, {cs['two_return_beams']} beams with 2 returns",
                  flush=True)
    frames_s = time.perf_counter() - t_frames
    compare = None
    if a.compare >= 0:
        stage2 = Stage(a, panes=False)
        out, img, _ = stage2.frame(a.compare)
        without, cs2 = panels(a, out, img, exposure, stage2.glass_slot)
        out, img, _ = stage.frame(a.compare)
        with_, _ = panels(a, out, img, exposure, stage.glass_slot)
        p2 = os.path.join(SCRATCH, f"no_panes_frame_{a.compare:04d}_320.png")
        p12 = os.path.join(SCRATCH, f"with_vs_without_frame_{a.compare:04d}_320.png")
        r1, r2 = row(with_), row(without)
        Image.fromarray(r2).save(p2)
        Image.fromarray(np.concatenate([r1, np.full((4, r1.shape[1], 3), 255, np.uint8), r2], 0)).save(p12)
        for v in VIEWS:
            Image.fromarray(without[v]).save(os.path.join(SCRATCH, f"no_panes_frame_{a.compare:04d}_{v}.png"))
        compare = dict(frame=a.compare, no_panes=os.path.relpath(p2, ROOT), pair=os.path.relpath(p12, ROOT), stats=cs2)
    job_s = time.perf_counter() - t_job
    sec = np.array([r["seconds"] for r in rows])
    rel = lambda v: os.path.relpath(v, ROOT) if isinstance(v, str) and os.path.isabs(v) else v   # noqa: E731
    meta = dict(command=" ".join([os.path.relpath(os.path.abspath(sys.argv[0]), ROOT)] + sys.argv[1:]),
                settings={k: rel(v) for k, v in vars(a).items()}, exposure=exposure, frames=len(rows), setup_s=setup_s,
                frames_s=frames_s, job_s=job_s, per_frame_median_s=float(np.median(sec)) if len(sec) else None,
                per_frame_max_s=float(sec.max()) if len(sec) else None, compare=compare, rows=rows)
    try:
        from metalsim.render.film import provenance
        meta.update(software=provenance.software(("warp", "mujoco", "numpy")), hardware=provenance.hardware())
    except Exception as e:  # noqa: BLE001
        meta["provenance_error"] = repr(e)
    full = a.encode and not a.frames
    if full:
        meta["encode"] = {v: [rel(c) for c in encode(dirs[v], os.path.join(a.out, f"lidar_glass_{v}.mp4"), a.fps, a.crf)]
                          for v in VIEWS}
        meta["preview"] = [rel(c) for c in encode_preview([dirs[v] for v in VIEWS],
                                                          os.path.join(SCRATCH, "preview_320.mp4"), a.fps, a.crf)]
    meta_path = os.path.join(SCRATCH if full else a.out, "lidar_glass.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    print(f"[lidar_glass] {len(rows)} frames in {frames_s:.1f} s (median {meta['per_frame_median_s']:.2f} s/frame), "
          f"set-up {setup_s:.1f} s, job {time.perf_counter() - t_job:.1f} s -> {rel(a.out)}", flush=True)


if __name__ == "__main__":
    main()
