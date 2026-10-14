"""Path-traced demo of ANYmal C walking on MPM sand, rendered by metalsim's tier-2 path tracer only.

Inputs: a recording from ``sim_anymal.py`` (Newton 1.5.2 example_mpm_anymal: sand particles + robot body poses), the
MuJoCo Menagerie ANYmal C visual meshes and textures (BSD-3), a Poly Haven HDRI (CC0; its sun is split off into an
analytic sun disk, metalsim.render.hdr.extract_sun) and a Poly Haven sand texture on the ground plane around the bed.

    .venv/bin/python scripts/particle_demo/render_anymal.py --sim runs/demo/anymal_sim --out runs/demo/frames \
        --frames 0:240 --width 1920 --height 1080 --passes 64

Writes ``frame_%04d.png`` (and ``stats.json``: per-frame render time, instance counts). Frame k shows simulation time
``--t0 + k / --fps`` (particles linearly interpolated between the recorded 50 Hz frames, bodies lerp / nlerp).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", default="0:1", help="start:stop[:step] of output frames")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--t0", type=float, default=1.0, help="simulation time of output frame 0 [s]")
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--passes", type=int, default=64, help="path-tracing passes (x --spp samples per pixel)")
ap.add_argument("--spp", type=int, default=4)
ap.add_argument("--bounces", type=int, default=5)
ap.add_argument("--grains", type=int, default=8, help="render grains per particle")
ap.add_argument("--grain-scale", type=float, default=1.2, help="grain radius relative to touching grains")
ap.add_argument("--hdri", default=None)
ap.add_argument("--hdri-yaw", type=float, default=106.0, help="dome rotation about +z [deg] (106: sun 75 deg to the right of the tracking camera)")
ap.add_argument("--sand-tex", default=None)
ap.add_argument("--exposure", type=float, default=0.17, help="linear exposure; 0 = automatic from the first frame")
ap.add_argument("--denoise", default="oidn")
ap.add_argument("--fovy", type=float, default=32.0)
ap.add_argument("--camera", default="track", help="track | still-a | still-b | still-c")
ap.add_argument("--save-hdr", action="store_true")
ap.add_argument("--skip-existing", action="store_true")
ap.add_argument("--ground-planes", type=int, default=4, choices=(1, 2, 4),
                help="4: planes around the robot's path (the demo); 1: one plane under everything; 2: only the left and "
                     "right planes (diagnostics)")
ap.add_argument("--plane-extent", type=float, default=60.0, help="diagnostics: half extent of the ground planes [m]")
ap.add_argument("--plane-shift", type=float, default=0.0,
                help="diagnostics: move the ground planes along +x [m] (e.g. 200: away from the grain region)")
ap.add_argument("--max-particles", type=int, default=0, help="diagnostics: keep every n-th particle so at most this many remain")
ap.add_argument("--no-trace", action="store_true", help="with --timing: the structure build only, no trace pass")
ap.add_argument("--structure", default="two_level", choices=("two_level", "flat"),
                help="grain acceleration-structure layout (particles.InstanceStructure); flat = before 2026-10-13")
ap.add_argument("--timing", action="store_true",
                help="diagnostics: per frame, time the instance-structure build (a pass-free render) and one trace pass "
                     "separately; no denoiser, no image")
ap.add_argument("--surface-depth", type=float, default=0.0,
                help="draw only bed particles within this depth [m] of their local surface (0 = all); grains keep their "
                     "identity through the renderer's stable ids, so more grains per particle fit in memory")
ap.add_argument("--flip-mesh-v", type=int, default=0,
                help="1: mirror mesh UVs (only for renderers before the 2026-10 atlas fix, RENDER_COVERAGE item 11; the "
                     "demo clip of 2026-10-12 was rendered that way)")
ap.add_argument("--cam-eye", type=float, nargs=3, default=None, help="debug camera position (with --cam-target)")
ap.add_argument("--cam-target", type=float, nargs=3, default=None)
ap.add_argument("--ground-z", type=float, default=0.098, help="render-only ground plane height (just under the bed top)")
ap.add_argument("--ground-albedo", type=float, default=0.68,
                help="ground plane albedo relative to the grain palette mean (0.68: matches the grain field's radiance)")
ap.add_argument("--decor-overlap", type=float, default=0.18)
ap.add_argument("--decor-falloff", type=float, default=1.2, help="width of the decoration's thinning border [m]")
ap.add_argument("--decor", type=float, nargs=4, default=[-2.6, -2.5, 2.6, 6.0],
                help="static decoration grains (one layer on the ground plane, outside the bed): x0 y0 x1 y1; 0 0 0 0 = none")
a = ap.parse_args()

import mujoco
import imageio.v3 as iio

from metalsim.render.hdr import extract_sun, load_hdr
from metalsim.render.particles import GrainMaterial
from metalsim.render.tier2 import Tier2Renderer

# MuJoCo Menagerie (git clone https://github.com/google-deepmind/mujoco_menagerie upstream/mujoco_menagerie) and the
# Poly Haven demo assets (scripts/fetch_polyhaven_assets.sh demo); MENAGERIE / POLYHAVEN point elsewhere
MENAGERIE = os.path.join(os.environ.get("MENAGERIE", os.path.join(ROOT, "upstream", "mujoco_menagerie")),
                         "anybotics_anymal_c", "anymal_c.xml")
PH = os.environ.get("POLYHAVEN", os.path.join(ROOT, "assets", "polyhaven"))
hdri = a.hdri or os.path.join(PH, "hdri", "demo", "goegap_8k.hdr")
sand_tex = a.sand_tex or os.path.join(PH, "tex", "demo", "coast_sand_01_diff_2k.png")
os.makedirs(a.out, exist_ok=True)

# ---- sim recording ----
meta = json.load(open(os.path.join(a.sim, "meta.json")))
B = np.load(os.path.join(a.sim, "bodies.npz"))
body_q, labels = B["body_q"], [str(s).split("/")[-1] for s in B["labels"]]
dt, n_sim = meta["frame_dt"], meta["frames"]
p_radius = meta["particle_radius"]

# ---- environment: HDRI sky + analytic sun ----
cache = hdri + ".npy"
hdr = np.load(cache) if os.path.exists(cache) else load_hdr(hdri)
if not os.path.exists(cache):
    np.save(cache, hdr)
yaw = np.deg2rad(a.hdri_yaw)
sky, sun = extract_sun(hdr, yaw=yaw)
print(f"[render] sun dir {np.round(sun['direction'], 3)} irradiance {np.round(sun['irradiance'], 2)} "
      f"radius {np.rad2deg(sun['angular_radius']):.2f} deg", flush=True)

# ---- MuJoCo scene: Menagerie ANYmal C + sand ground plane + sun + camera ----
spec = mujoco.MjSpec.from_file(MENAGERIE)
for l in list(spec.lights):
    spec.delete(l)
tex = spec.add_texture(name="sand_ground", type=mujoco.mjtTexture.mjTEXTURE_2D, file=sand_tex)
# linear colorspace, explicitly: MuJoCo already loads this PNG as linear (tex_colorspace 1), so the atlas samples its
# bytes as they are, before and after the 2026-10 atlas fix (which decodes only textures marked sRGB); the tint below
# divides by that mean (an sRGB-decoded mean clipped the tint and brightened the plane, 2026-10-13)
tex.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR
img = iio.imread(sand_tex)[..., :3].astype(np.float32) / 255.0
tex_mean = img.reshape(-1, 3).mean(0)
grain_mat = GrainMaterial()
pal = np.array([p[:3] for p in grain_mat.palette]); w = np.array([p[3] for p in grain_mat.palette])
grain_mean = (pal * w[:, None]).sum(0) / w.sum()
tint = np.clip(a.ground_albedo * grain_mean / tex_mean, 0, 1)
mat = spec.add_material(name="sand_ground", texrepeat=[0.8, 0.8], texuniform=True,
                        rgba=[float(tint[0]), float(tint[1]), float(tint[2]), 1.0], specular=0.0, shininess=0.0)
mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "sand_ground"
# the ground: four finite planes around the robot's path (the bed interior minus --decor-overlap), so footprints show
# the bed's own sand, never the raised plane; their edges lie under the decoration grains
bx0_, by0_, _, bx1_, by1_, _ = meta["bed"]
hx0, hx1, hy0, hy1 = bx0_ + a.decor_overlap, bx1_ - a.decor_overlap, by0_ + a.decor_overlap, by1_ - a.decor_overlap
E_ = a.plane_extent
grounds = {"ground_l": (-E_, hx0, -E_, E_), "ground_r": (hx1, E_, -E_, E_), "ground_b": (hx0, hx1, -E_, hy0),
           "ground_f": (hx0, hx1, hy1, E_)}
if a.ground_planes == 2:
    grounds = {k: grounds[k] for k in ("ground_l", "ground_r")}
elif a.ground_planes == 1:
    grounds = {"ground": (-E_, E_, -E_, E_)}
grounds = {k: (x0 + a.plane_shift, x1 + a.plane_shift, y0, y1) for k, (x0, x1, y0, y1) in grounds.items()}
for nm, (x0, x1, y0, y1) in grounds.items():
    spec.worldbody.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_PLANE, size=[(x1 - x0) / 2, (y1 - y0) / 2, 0.1],
                            pos=[(x0 + x1) / 2, (y0 + y1) / 2, a.ground_z], material="sand_ground")
sd = sun["direction"]
spec.worldbody.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 10], dir=list(-sd),
                         diffuse=[1, 1, 1], specular=[0, 0, 0], ambient=[0, 0, 0], castshadow=True)
spec.worldbody.add_camera(name="cine", pos=[3, 0, 1], fovy=a.fovy)
model = spec.compile()
if a.flip_mesh_v:
    # renderers before the 2026-10 atlas fix sampled mesh UVs vertically mirrored relative to MuJoCo
    # (docs/RENDER_COVERAGE.md item 11); --flip-mesh-v 1 undoes that for them
    model.mesh_texcoord[:, 1] = 1.0 - model.mesh_texcoord[:, 1]
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# body mapping: Newton body -> MuJoCo body by name, with the constant frame correction measured at the zero pose
# (identity for these two models, both built from ANYbotics' URDF; kept so another robot can be mapped the same way)
mj_bodies = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(model.nbody)}
nb_of_mj = {mj_bodies[l]: i for i, l in enumerate(labels) if l in mj_bodies}
missing = [n for n, b in mj_bodies.items() if b and b not in nb_of_mj]
assert not missing, f"MuJoCo bodies without a Newton body: {missing}"


def qmul(q1, q2):   # wxyz
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def qmat(q):
    m = np.zeros(9); mujoco.mju_quat2Mat(m, np.asarray(q, np.float64)); return m.reshape(3, 3)


def body_poses(t):
    """Newton body poses at sim time t (lerp / nlerp between recorded frames), as {mj body: (pos, R)}."""
    f = min(max(t / dt, 0.0), n_sim - 1.0001)
    i0 = int(f); al = f - i0
    out = {}
    for b, nb in nb_of_mj.items():
        q0, q1 = body_q[i0, nb], body_q[i0 + 1, nb]
        p = (1 - al) * q0[:3] + al * q1[:3]
        r0, r1 = q0[[6, 3, 4, 5]], q1[[6, 3, 4, 5]]          # xyzw -> wxyz
        if np.dot(r0, r1) < 0:
            r1 = -r1
        r = (1 - al) * r0 + al * r1; r /= np.linalg.norm(r)
        out[b] = (p, qmat(r))
    return out


def load_frame(i, wait_s=900.0):
    """A recorded particle frame; waits (up to ``wait_s``) for one still being copied in."""
    path = os.path.join(a.sim, f"particles_{i:05d}.npy")
    t0 = time.time()
    while True:
        try:
            return np.load(path)
        except (FileNotFoundError, ValueError, EOFError):
            if time.time() - t0 > wait_s:
                raise
            print(f"[render] waiting for {path}", flush=True)
            time.sleep(10)


def particles(t):
    f = min(max(t / dt, 0.0), n_sim - 1.0001)
    i0 = int(f); al = np.float32(f - i0)
    p0 = load_frame(i0)
    if al < 1e-6:
        return p0
    p1 = load_frame(i0 + 1)
    return (1 - al) * p0 + al * p1


def surface_keep(P, depth, cell=0.01):
    """Particles within ``depth`` of the local sand surface: the surface height of a 1 cm column is its highest
    particle; a particle is kept if it is above the lowest surface of its 3 x 3 neighbourhood minus ``depth`` (so the
    walls of footprints stay drawn)."""
    lo = P[:, :2].min(0)
    ij = np.floor((P[:, :2] - lo) / cell).astype(np.int64) + 1
    nx, ny = ij.max(0) + 2
    top = np.full((nx, ny), -np.inf, np.float32)
    np.maximum.at(top, (ij[:, 0], ij[:, 1]), P[:, 2])
    pad = np.pad(top, 1, constant_values=np.inf)
    low = np.min([pad[1 + dx:nx + 1 + dx, 1 + dy:ny + 1 + dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)], 0)
    return P[:, 2] >= low[ij[:, 0], ij[:, 1]] - depth


def set_geoms(t):
    poses = body_poses(t)
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b == 0:
            continue
        p, R = poses[b]
        data.geom_xpos[g] = p + R @ model.geom_pos[g]
        data.geom_xmat[g] = (R @ qmat(model.geom_quat[g])).ravel()
    return poses


# ---- camera ----
base_track = body_q[:, nb_of_mj[mj_bodies["base"]], :3]
ts = np.arange(n_sim) * dt
walk = ts > meta["settle"] + 0.5
coef = np.polyfit(ts[walk], base_track[walk, 1], 1)          # smooth forward motion of the robot (y)


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, u, -f], 1)                            # camera x right, y up, looks along -z
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.ravel()); return q


def camera(k, t):
    if a.cam_eye is not None:
        return np.array(a.cam_eye), look_at(np.array(a.cam_eye), a.cam_target, up=(0, 1, 0) if abs(a.cam_eye[0] - a.cam_target[0]) + abs(a.cam_eye[1] - a.cam_target[1]) < 1e-6 else (0, 0, 1))
    """Slow tracking shot: a 3/4 front view orbiting 25 degrees over the clip and easing in."""
    yc = float(np.polyval(coef, t))
    s = k / 240.0
    if a.camera == "track":
        az = np.deg2rad(38 + 22 * s)                     # angle from the walking direction (+y), toward +x
        dist, h = 2.9 - 0.4 * s, 0.85 - 0.12 * s
    elif a.camera == "still-a":
        az, dist, h = np.deg2rad(40), 2.6, 0.8
    elif a.camera == "still-b":                          # low side view: footprints and the sand surface
        az, dist, h = np.deg2rad(95), 2.2, 0.38
    else:                                                # from behind: the trail of footprints
        az, dist, h = np.deg2rad(160), 2.8, 1.25
    target = np.array([0.0, yc + (0.05 if a.camera != "still-c" else 0.4), 0.33])
    eye = target + np.array([np.sin(az) * dist, np.cos(az) * dist, h - 0.33])
    return eye, look_at(eye, target)


# ---- decoration grains: one static particle layer on the raised ground plane around the bed (render only), thinning
# out over its outer --decor-falloff so the grain field fades into the textured plane ----
decor = np.zeros((0, 3), np.float32)
if any(a.decor):
    x0, y0, x1, y1 = a.decor
    sp = 2 * p_radius
    gx, gy = np.meshgrid(np.arange(x0, x1, sp), np.arange(y0, y1, sp), indexing="ij")
    rs = np.random.default_rng(7)
    pts = np.stack([gx.ravel(), gy.ravel()], 1) + rs.uniform(-0.5, 0.5, (gx.size, 2)) * sp
    bx0, by0, _, bx1, by1, _ = meta["bed"]
    # the bed's edges settle below the plane over ~15-20 cm: decoration overlaps that band (where the bed surface is
    # above the plane, the decoration grains are buried in it); the robot's path stays clear of it
    m_ = a.decor_overlap
    inside = (pts[:, 0] > bx0 + m_) & (pts[:, 0] < bx1 - m_) & (pts[:, 1] > by0 + m_) & (pts[:, 1] < by1 - m_)
    edge = np.minimum.reduce([pts[:, 0] - x0, x1 - pts[:, 0], pts[:, 1] - y0, y1 - pts[:, 1]])
    keep = ~inside & (rs.uniform(0, 1, len(pts)) < np.clip(edge / a.decor_falloff, 0, 1))
    pts = pts[keep]
    z = a.ground_z + rs.uniform(0.0, 0.6, len(pts)) * p_radius
    decor = np.concatenate([pts, z[:, None]], 1).astype(np.float32)
    print(f"[render] decoration particles {len(decor)}", flush=True)

# ---- renderer ----
radius_grain = p_radius / a.grains ** (1 / 3) * a.grain_scale
print(f"[render] stage: building the renderer ({model.ngeom} geoms, {model.ntex} textures)", flush=True)
r = Tier2Renderer(model, 1, width=a.width, height=a.height, camera="cine", spp=a.spp, max_bounces=a.bounces,
                  material_model="omnipbr", headlight=False, tonemap="rtx", exposure=1.0,
                  usd_lights=[{"intensity": 1.0, "color": tuple(sun["irradiance"] / np.pi),
                               "angle_deg": 2 * np.rad2deg(max(sun["angular_radius"], 0.0047))}],
                  denoise=None if a.timing else a.denoise, aux=not a.timing, center_sample=False)
print("[render] stage: renderer built; environment map", flush=True)
r.set_environment(sky, yaw=yaw)
print("[render] stage: environment set", flush=True)
# robot materials: painted / plastic shells with a clear-coat gloss (Menagerie leaves specular 0)
mats = r.tables.materials.reshape(r.G, 16).copy()
for slot, g in enumerate(r.tables.geoms):
    if model.geom_bodyid[g] > 0:
        mats[slot, 5] = 0.42 if mats[slot, 0:3].max() > 0.05 or mats[slot, 14] > 0.5 else 0.5   # roughness
        mats[slot, 6] = 0.5                                                                      # specular
r.set_materials(mats)

frames = [int(x) for x in a.frames.split(":")]
rng_frames = range(frames[0], frames[1], frames[2] if len(frames) > 2 else 1)
stats = dict(args=vars(a), sun={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in sun.items()}, frames=[])
exposure = a.exposure
for k in rng_frames:
    t = a.t0 + k / a.fps
    name = f"frame_{k:04d}.png" if a.camera == "track" else f"{a.camera}_{k:04d}.png"
    if a.skip_existing and os.path.exists(os.path.join(a.out, name)):
        continue
    tf = time.time()
    set_geoms(t)
    Ps = particles(t)
    ids = None
    if a.surface_depth > 0:
        keep = surface_keep(Ps, a.surface_depth)
        ids = np.concatenate([np.nonzero(keep)[0], len(Ps) + np.arange(len(decor))]).astype(np.uint32)
        Ps = Ps[keep]
    P = np.concatenate([Ps, decor])
    if a.max_particles and len(P) > a.max_particles:
        P = P[::int(np.ceil(len(P) / a.max_particles))]
    r.set_particles(P, radius=p_radius, grains_per_particle=a.grains, grain_radius=radius_grain, ids=ids,
                    structure=a.structure)
    eye, q = camera(k, t)
    if k == rng_frames[0]:
        print(f"[render] stage: first frame ({len(P)} particles)", flush=True)
    if a.timing:
        t0 = time.time()
        r.render_host([data], passes=0, cam_pos=eye[None], cam_quat=q[None])
        t_build = time.time() - t0
        print(f"[timing] {len(P)} particles, {r.grains.n_grains} grains: upload + structure build {t_build * 1e3:.0f} ms "
              f"(structure {r._ias.as_bytes / 2**20:.0f} MiB)", flush=True)
        t0 = time.time()
        r.render_host([data], passes=0, cam_pos=eye[None], cam_quat=q[None])
        print(f"[timing] rebuild (structure allocated): {(time.time() - t0) * 1e3:.0f} ms", flush=True)
        if a.no_trace:
            continue
        print(f"[timing] tracing one pass of {a.spp} spp at {a.width}x{a.height} ...", flush=True)
        t0 = time.time()
        r.render_host([data], passes=1, cam_pos=eye[None], cam_quat=q[None])
        print(f"[timing] build + one pass of {a.spp} spp at {a.width}x{a.height}: {(time.time() - t0) * 1e3:.0f} ms", flush=True)
        continue
    out = r.render_host([data], passes=a.passes, cam_pos=eye[None], cam_quat=q[None], chunk=4)
    hdr_img = out["hdr_denoised"][0] if "hdr_denoised" in out else out["hdr"][0]
    if exposure <= 0:   # automatic: the median scene luminance to 0.2 before the ACES curve
        lum = hdr_img @ np.array([0.2126, 0.7152, 0.0722])
        exposure = 0.20 / float(np.median(lum))
        print(f"[render] exposure {exposure:.4f}", flush=True)
    x = np.clip(hdr_img * exposure, 0, None)
    x = np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    rgb = (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    iio.imwrite(os.path.join(a.out, name), rgb)
    if a.save_hdr:
        np.save(os.path.join(a.out, name.replace(".png", "_hdr.npy")), hdr_img.astype(np.float16))
    el = time.time() - tf
    st = dict(frame=k, t=t, seconds=el, particles=int(P.shape[0]), grains=int(r.grains.n_grains),
              instances=int(r._ias.n_total), ias_mib=r._ias.as_bytes / 2**20, spp=a.passes * a.spp)
    stats["frames"].append(st)
    stats["exposure"] = exposure
    json.dump(stats, open(os.path.join(a.out, f"stats_{a.camera}_{frames[0]}.json"), "w"), indent=1)
    print(f"[render] {name} t={t:.3f}s {el:.1f}s grains={st['grains']} ias={st['ias_mib']:.0f}MiB", flush=True)
