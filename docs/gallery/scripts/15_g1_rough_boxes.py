"""Tier-2 frame of Isaac's G1 in the inverted-stairs pit of Isaac's rough terrain (seed 0, row 5, column 4) with
``terrain_collision="boxes_local"``: left, the renderer without the per-world box slots (``terrain_slots=False``: the
heightfield only, lowered 5 cm under the box-built cells, which is what the collision heightfield is there); right, the
slots drawn at their per-world size and position (the default). The physics window is 1.2 m around the pelvis; this
still uses a 3.9 m window (BoxWindow(half_width=3.9)) so the slots hold the whole pit's rings (29 boxes, no overflow).

    scripts/gpu_run.sh gallery_rough_boxes render 5 -- python docs/gallery/scripts/15_g1_rough_boxes.py
"""
import numpy as np, mujoco, torch, warp as wp, imageio.v2 as iio
wp.config.quiet = True
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.learn.terrain import isaac_rough_terrain, terrain_boxes, BoxWindow
from metalsim.physics.batch import BatchSim, BatchSimOptions
from metalsim.render.tier2 import Tier2Renderer

ROW, COL = 5, 4
hf = isaac_rough_terrain(seed=0, collision="boxes_local")
B = terrain_boxes(hf["generator"]); cb = B[(B[:, 6] == ROW) & (B[:, 7] == COL)]
o = hf["origin_table"]()[ROW, COL]; rim = float((cb[:, 2] + cb[:, 5]).max())
m, info = build_g1_model("rough", hf, visuals=True)
spec = info["spec"]


def lookat_quat(pos, target):
    f = np.array(target) - np.array(pos); f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r); u = np.cross(r, f)
    R = np.stack([r, u, -f], 1); q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(-1)); return q.tolist()


cpos = [o[0] + 3.4, o[1] - 2.6, rim + 1.3]
cam = spec.worldbody.add_camera(); cam.name = "hero"; cam.pos = cpos; cam.fovy = 50
cam.quat = lookat_quat(cpos, [o[0] - 0.3, o[1] + 0.2, o[2] + 0.35])
l = spec.worldbody.add_light(); l.pos = [o[0], o[1], rim + 5]; l.dir = [-0.45, 0.35, -0.8]; l.diffuse = [0.9, 0.9, 0.85]
l.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
gm = spec.add_material(); gm.name = "terrain"; gm.rgba = [0.62, 0.58, 0.52, 1]; gm.roughness = 0.8
sky = spec.add_texture(); sky.name = "sky"; sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX; sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
sky.rgb1 = [0.55, 0.65, 0.8]; sky.rgb2 = [0.9, 0.92, 0.95]; sky.width = 64; sky.height = 384
for g in spec.geoms:
    if g.name == "ground":
        g.material = "terrain"
    elif not g.name.startswith("tslot") and (g.contype != 0 or g.conaffinity != 0):
        g.group = 4                          # robot collision geometry hidden (the visual meshes are drawn)
m = spec.compile()
sim = BatchSim(m, 1, options=BatchSimOptions(per_world_fields=BoxWindow.FIELDS, njmax=256, nconmax=128))
q = m.key_qpos[0].copy(); q[:2] = o[:2]; q[2] += o[2]
sim.set_state(q.astype(np.float32), np.zeros(m.nv, np.float32))
win = BoxWindow(sim, hf, half_width=3.9); win.launch(); sim.synchronize()
ov = win.overflow.numpy(); assert ov[0] == 0, ov
panels = []
for slots in (False, True):
    r = Tier2Renderer(m, 1, width=960, height=600, camera="hero", spp=16, max_bounces=3, decimate_faces=0, terrain_slots=slots)
    v = r.render(sim, sim.event.value, passes=16); r.after(v); torch.mps.synchronize()
    panels.append(r.out.rgb[0].cpu().numpy())
gap = np.full((600, 8, 3), 255, np.uint8)
img = np.concatenate([panels[0], gap, panels[1]], 1)
iio.imwrite("docs/gallery/g1_rough_boxes_tier2.png", img)
print(f"wrote docs/gallery/g1_rough_boxes_tier2.png; boxes in the window {int(ov[1])}; rim {rim:.3f} m, pit {o[2]:.3f} m")
