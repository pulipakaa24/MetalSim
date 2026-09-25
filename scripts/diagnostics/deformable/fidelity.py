"""Deformable fidelity harness (docs/research/deformables_2026-09-25.md §2). MuJoCo Warp flex vs MuJoCo C (3.14 pip and main), per scene: free-trajectory vertex error, synced one-step velocity
error, and contact-set agreement from the reference states. C vs C' (1e-7 m initial perturbation) is the sensitivity floor.
usage: fidelity.py DEVICE [scene ...]"""
import sys, os, re, subprocess, json, numpy as np, mujoco, warp as wp
wp.config.quiet = True
from metalsim.physics import deformable as dfm
from mujoco_warp._src import collision_flex as _cf
_cf.FLEX_FPS_MODE = os.environ.get("FLEX_FPS_MODE", "c314")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
# reference rollouts are cached here; MuJoCo C main is optional (a python with mujoco built from main)
REF = os.environ.get("DEFORMABLE_FIDELITY_REF", os.path.join(ROOT, "scratch", "deformable", "ref"))
os.makedirs(REF, exist_ok=True)
MAIN_PY = os.environ.get("MUJOCO_MAIN_PYTHON", os.path.join(ROOT, "scratch", "deformable", "venv-mjmain", "bin", "python"))
dev = sys.argv[1]
ITER = 100
SOFT = """
    <flexcomp name="soft" type="grid" count="4 4 4" spacing="0.05 0.05 0.05" pos="0.02 0.01 0.45" dim="3" radius="0.003" mass="0.5" dof="{dof}">
      <elasticity young="5e3" poisson="0.3" damping="0.01"/>
      <contact condim="3" solref="0.01 1" selfcollide="none"/>
    </flexcomp>"""
SCENES = {
    "floor": re.sub(r'<geom name="box"[^>]*/>', '', dfm.box_scene_xml(dfm.ClothCfg(pos=(0, 0, 0.1)), dfm.CableCfg(pos=(0.1, 0.3, 0.1)), iterations=ITER, ls_iterations=50)),
    "box": dfm.box_scene_xml(iterations=ITER, ls_iterations=50),
    "cloth2nd_box": dfm.box_scene_xml(None, dfm.CableCfg(pos=(0.6, 0.6, 0.05)), iterations=ITER, ls_iterations=50).replace("</worldbody>", dfm._cloth_xml(dfm.ClothCfg()) + "</worldbody>"),
    "prims": dfm.box_scene_xml(dfm.ClothCfg(pos=(0, 0, 0.45)), dfm.CableCfg(pos=(-0.15, 0.3, 0.3)), iterations=ITER, ls_iterations=50).replace(
        '<geom name="box" type="box" size="0.15 0.15 0.15" pos="0 0 0.15"',
        '<geom name="ball" type="sphere" size="0.12" pos="0 0 0.2"/><geom name="rod" type="capsule" size="0.04 0.25" pos="0 0.3 0.15" euler="0 90 0"/><geom name="cyl" type="cylinder" size="0.05 0.1" pos="0.3 0 0.1"'),
    "mesh": dfm.box_scene_xml(dfm.ClothCfg(pos=(0, 0, 0.45)), dfm.CableCfg(pos=(-0.15, 0.0, 0.35)), iterations=ITER, ls_iterations=50).replace(
        '<worldbody>', '<asset><mesh name="m" vertex="-.15 -.15 -.15  .15 -.15 -.15  .15 .15 -.15  -.15 .15 -.15  -.15 -.15 .15  .15 -.15 .15  .15 .15 .15  -.15 .15 .15"/></asset><worldbody>').replace(
        '<geom name="box" type="box" size="0.15 0.15 0.15"', '<geom name="box" type="mesh" mesh="m"'),
    "cable_on_cloth": dfm.box_scene_xml(dfm.ClothCfg(), dfm.CableCfg(pos=(-0.15, 0.0, 0.35)), iterations=ITER, ls_iterations=50),
    "soft_full_box": dfm.box_scene_xml(None, None, timestep=0.001, iterations=ITER, ls_iterations=50).replace("</worldbody>", SOFT.format(dof="full") + "</worldbody>"),
    "soft_tri_box": dfm.box_scene_xml(None, None, timestep=0.001, iterations=ITER, ls_iterations=50).replace("</worldbody>", SOFT.format(dof="trilinear") + "</worldbody>"),
    "cable_box": dfm.box_scene_xml(None, dfm.CableCfg(pos=(-0.15, 0.0, 0.35)), iterations=ITER, ls_iterations=50),
}
N = 400
names = sys.argv[2:] or list(SCENES)
res = {}
for name in names:
    xml = SCENES[name]
    xf = os.path.join(REF, name + ".xml"); open(xf, "w").write(xml)
    m = mujoco.MjModel.from_xml_string(xml); d0 = mujoco.MjData(m); mujoco.mj_forward(m, d0)
    q0 = d0.qpos + np.random.default_rng(0).normal(0, 1e-4 if name.startswith('soft') else 2e-3, m.nq); np.save(os.path.join(REF, name + "_q0.npy"), q0)
    q0p = q0 + np.random.default_rng(1).normal(0, 1e-7, m.nq); np.save(os.path.join(REF, name + "_q0p.npy"), q0p)
    refs = {}
    for tag, py in (("c314", sys.executable), ("main", MAIN_PY)):
        if tag == "main" and not os.path.exists(py):
            continue
        for qf, suf in ((name + "_q0.npy", ""), (name + "_q0p.npy", "_p")):
            out = os.path.join(REF, f"{name}_{tag}{suf}.npz")
            if not os.path.exists(out):
                subprocess.run([py, os.path.join(HERE, "ref_main.py"), xf, os.path.join(REF, qf), str(N), out], check=True, capture_output=True)
            refs[tag + suf] = np.load(out, allow_pickle=True)
    sim = dfm.DeformableSim(m, 1, device=dev, capture=False, sizes=dfm.default_sizes(m, 6.0))
    sim.set_qpos(q0[None])
    W = []
    for k in range(N):
        sim.step(); W.append(sim.d.flexvert_xpos.numpy()[0].copy())
    W = np.array(W)
    out = {}
    for tag in [t for t in ("c314", "main") if t in refs]:
        R, Rp = refs[tag], refs[tag + "_p"]
        # synced one-step and contact sets
        dv, same, cnt = [], 0, 0
        for k in range(0, N - 1, 4):
            sim.set_qpos(R["qpos"][k][None], R["qvel"][k][None]); sim.step()
            w = sim.get_world(0)
            dv.append(np.abs(w.qvel - R["qvel"][k + 1]).max())
            cs = R["cs"][k]
            if len(cs) or w.ncon:
                c = w.contact[:w.ncon]
                ws = sorted(map(tuple, np.stack([c.geom[:, 0], c.flex[:, 1], c.elem[:, 1], c.vert[:, 1]], 1).tolist())) if w.ncon else []
                same += ws == sorted(map(tuple, np.asarray(cs).tolist())); cnt += 1
        e_traj = [float(np.abs(W[t - 1] - R["vert"][t - 1]).max()) for t in (50, 100, 200, 400)]
        e_self = [float(np.abs(Rp["vert"][t - 1] - R["vert"][t - 1]).max()) for t in (50, 100, 200, 400)]
        dz = float(W[-1][:, 2].mean() - R["vert"][-1][:, 2].mean())
        out[tag] = {"version": str(R["version"]), "traj_err_m@50/100/200/400": e_traj, "C_vs_Cprime_m": e_self,
                    "mean_z_diff_m@400": dz, "onestep_dqvel_max": float(max(dv)), "onestep_dqvel_median": float(np.median(dv)),
                    "contact_sets_equal": f"{same}/{cnt}"}
    res[name] = out
    print(name, json.dumps(out, indent=1), flush=True)
