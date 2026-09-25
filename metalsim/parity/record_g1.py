"""MetalSim side of the fidelity protocol: replays the exact protocol `isaac_side/record_g1.py` ran
(PD hold, the recorded random action sequence B by joint name, drop) on Isaac's G1 asset with MuJoCo
Warp on Metal, records the same quantities in the same layout (Isaac's joint order), and renders the
same camera under the same lights with tier 2 (and tier 0 for reference).

    python -m metalsim.parity.record_g1 --isaac runs/parity/isaac/rt --out runs/parity/metalsim --physics_dt 0.0025
    python -m metalsim.parity.record_g1 --isaac ... --out ... --contact_tuning tau5_imp99_hardlimits --no_render

Contact forces are Isaac's quantity (net normal contact force per body, world frame) from
metalsim.sensors.contact.ContactSensor; each control step also records the deepest contact
penetration and the largest joint-limit excursion over its substeps (substep hook), and
``--contact_tuning`` applies a metalsim.physics.contact_tuning preset to the model.
"""
import argparse, json, os, math
import numpy as np, torch, mujoco, warp as wp
import imageio.v2 as iio

from metalsim.learn.g1_velocity import build_g1_model, ACTION_SCALE, INIT_POS
from metalsim.physics.batch import BatchSim, BatchSimOptions
from metalsim.render.tier2 import Tier2Renderer
from metalsim.render.tier0 import Tier0Renderer
from metalsim.physics import contact_tuning, solver_presets
from metalsim.sensors.contact import ContactSensor


@wp.kernel
def _substep_penetration(nacon: wp.array(dtype=int), dist: wp.array(dtype=float), worldid: wp.array(dtype=int),
                         efc_address: wp.array2d(dtype=int), pen: wp.array(dtype=float)):
    c = wp.tid()
    if c >= nacon[0] or efc_address[c, 0] < 0:
        return
    wp.atomic_max(pen, worldid[c], -dist[c])


@wp.kernel
def _substep_limit_excursion(qpos: wp.array2d(dtype=float), adr: wp.array(dtype=int), lo: wp.array(dtype=float),
                             hi: wp.array(dtype=float), exc: wp.array(dtype=float)):
    w, j = wp.tid()
    q = qpos[w, adr[j]]
    wp.atomic_max(exc, w, wp.max(lo[j] - q, q - hi[j]))


def lookat_quat(pos, target):
    f = np.array(target) - np.array(pos); f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r); u = np.cross(r, f)
    R = np.stack([r, u, -f], 1); q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(-1)); return q.tolist()


def hero_spec(spec, meta):
    """Camera and lights matching the Isaac recording (sun DistantLight rotated about y, uniform dome)."""
    cam = meta["camera"]
    c = spec.worldbody.add_camera(); c.name = "hero"; c.pos = cam["pos"]; c.quat = lookat_quat(cam["pos"], cam["target"]); c.fovy = cam["vfov_deg"]
    w, x, y, z = meta["lights"]["sun"]["rot_wxyz"]
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.array([w, x, y, z])); d = R.reshape(3, 3) @ np.array([0, 0, -1.0])   # USD distant light emits along -Z
    l = spec.worldbody.add_light(); l.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL; l.pos = [0, 0, 5]; l.dir = d.tolist()
    l.diffuse = [c * 0.9 for c in meta["lights"]["sun"]["color"]]; l.specular = [0.3, 0.3, 0.3]
    sky = spec.add_texture(); sky.name = "sky"; sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX; sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    dc = meta["lights"]["dome"]["color"]; sky.rgb1 = dc; sky.rgb2 = dc; sky.width = 8; sky.height = 48
    # the recording's ground: UsdPreviewSurface diffuseColor 0.5, roughness 0.7, metallic 0, ior 1.5 (F0 0.04 = 0.08 * specular 0.5)
    gm = spec.add_material(); gm.name = "ground_preview"; gm.rgba = [0.5, 0.5, 0.5, 1]; gm.roughness = 0.7; gm.metallic = 0.0
    gm.specular = 0.5; gm.shininess = 0.3
    for g in spec.geoms:
        if g.name == "ground": g.rgba = [0.5, 0.5, 0.5, 1]; g.size = [500.0, 500.0, 0.05]; g.material = "ground_preview"
        elif g.contype != 0 or g.conaffinity != 0: g.group = 4   # collision geometry (Isaac never draws it; the renderer draws groups <= 3); the visual meshes are group 2   # render extent only (a MuJoCo plane is infinite for physics); the ray tracer tessellates size-0 planes to 5 m, which ended before the horizon in the first replay
    return spec


def tier2_parity_kwargs(meta, mode="rtx"):
    """Tier-2 options for the parity scene: USD light units, OmniPBR / UsdPreviewSurface BRDFs, RTX display
    transform and denoiser (docs/research/rendering_vs_rtx_2026-09-25.md); 'legacy' = the renderer as it was."""
    from metalsim.render.rtx_parity import preset_kwargs, RTX_DEFAULT_PRESET
    return preset_kwargs(RTX_DEFAULT_PRESET if mode == "rtx" else mode, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaac", required=True, help="directory written by isaac_side/record_g1.py (meta.json, action_sequence_B.npz)")
    ap.add_argument("--out", required=True); ap.add_argument("--physics_dt", type=float, default=0.0025)
    ap.add_argument("--frame_every", type=int, default=5); ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--contact_tuning", default="default", choices=sorted(contact_tuning.PRESETS),
                    help="metalsim.physics.contact_tuning preset applied to the model")
    ap.add_argument("--solver_cfg", default=None, choices=sorted(solver_presets.PRESETS),
                    help="metalsim.physics.solver_presets preset (e.g. isaaclab3: Isaac Lab 3.0's own MuJoCo Warp settings), "
                         "applied after --contact_tuning")
    ap.add_argument("--no_render", action="store_true", help="physics only (no tier 2 / tier 0 frames)")
    ap.add_argument("--tier2_mode", default="rtx", help="tier-2 preset of metalsim.render.rtx_parity: rtx (the RTX-parity default), "
                    "legacy (MuJoCo-unit lights, linear clamp: the renderer before 2026-09-25), or any preset name")
    a = ap.parse_args(); wp.config.quiet = True
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.isaac, "meta.json")))
    seqB = np.load(os.path.join(a.isaac, "action_sequence_B.npz"))
    isaac_joints = [str(s) for s in seqB["joint_names"]]
    m, info = build_g1_model("flat", visuals=True, physics_dt=a.physics_dt)
    spec = hero_spec(info["spec"], meta); m = spec.compile()
    contact_tuning.apply(m, a.contact_tuning)
    for w in contact_tuning.check_timestep(m, a.contact_tuning):
        print("[metalsim] warning:", w)
    if a.solver_cfg:
        solver_presets.apply(m, a.solver_cfg)
    our_joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    to_isaac = np.array([our_joints.index(n) for n in isaac_joints])            # our joint index for each Isaac joint
    act_of_joint = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0]): i for i in range(m.nu)}
    act_isaac = np.array([act_of_joint[n] for n in isaac_joints])
    n = a.num_envs; dec = int(round(meta["control_dt"] / a.physics_dt))
    bso = dict(substeps=dec, njmax=256, nconmax=32, solver_iterations=10, ls_iterations=20)
    if a.solver_cfg:
        bso = solver_presets.batch_options(a.solver_cfg, **bso)
    sim = BatchSim(m, n, options=BatchSimOptions(**bso))
    if a.solver_cfg:
        solver_presets.install(sim, a.solver_cfg)              # e.g. collision once per 5 ms tick
    sim.synchronize()
    cam = meta["camera"]
    if not a.no_render:
        rend2 = Tier2Renderer(m, n, width=cam["width"], height=cam["height"], camera="hero", spp=16, max_bounces=3, **tier2_parity_kwargs(meta, a.tier2_mode))
        rend0 = Tier0Renderer(m, n, width=cam["width"], height=cam["height"], camera="hero", outputs=("rgb", "depth"))
    # Isaac's contact quantity: net normal force per contact body (bodies without colliders read 0)
    body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    cs_bodies = [nm for nm in meta["contact_bodies"] if nm in body_names]
    csens = ContactSensor(sim, cs_bodies, force_mode="normal")
    cs_index = {nm: i for i, nm in enumerate(cs_bodies)}
    # per-substep penetration / limit excursion (max over the substeps of a control step)
    lim = [j for j in range(1, m.njnt) if m.jnt_limited[j] and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    dev = sim.device
    lim_adr = wp.array(np.array([m.jnt_qposadr[j] for j in lim], np.int32), dtype=int, device=dev)
    lim_lo = wp.array(np.array([m.jnt_range[j][0] for j in lim], np.float32), dtype=float, device=dev)
    lim_hi = wp.array(np.array([m.jnt_range[j][1] for j in lim], np.float32), dtype=float, device=dev)
    pen = wp.zeros(n, dtype=float, device=dev); exc = wp.zeros(n, dtype=float, device=dev)
    def _hook():
        wp.launch(_substep_penetration, dim=sim.d.naconmax, device=dev, inputs=[sim.d.nacon, sim.d.contact.dist, sim.d.contact.worldid, sim.d.contact.efc_address, pen])
        wp.launch(_substep_limit_excursion, dim=(n, len(lim)), device=dev, inputs=[sim.d.qpos, lim_adr, lim_lo, lim_hi, exc])
    sim.add_substep_hook(_hook)
    default = m.key_qpos[0].copy()
    bid = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
    contact_bodies = meta["contact_bodies"]; touch = {"left_ankle_roll_link": 0, "right_ankle_roll_link": 1, "torso_link": 2}
    protocols = {"A_hold": (150, lambda t: np.zeros((n, m.nu), np.float32), None),
                 "B_random": (250, lambda t: np.repeat(seqB["actions"][t][None], n, 0), None),
                 "C_drop": (150, lambda t: np.zeros((n, m.nu), np.float32), 1.0)}
    jr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): (m.jnt_range[j].tolist() if m.jnt_limited[j] else None) for j in range(1, m.njnt)}
    json.dump({"our_joints": our_joints, "isaac_joints": isaac_joints, "physics_dt": a.physics_dt, "decimation": dec,
               "contact_tuning": a.contact_tuning, "tuning": repr(contact_tuning.PRESETS[a.contact_tuning]),
               "solver_cfg": a.solver_cfg, "solver_preset": repr(solver_presets.PRESETS[a.solver_cfg]) if a.solver_cfg else None,
               "joint_range_isaac_order": [jr[nm] for nm in isaac_joints], "contact": "ContactSensor net normal force (Isaac net_forces_w)"},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    for tag, (T, act_fn, drop_z) in protocols.items():
        q = np.tile(default, (n, 1)).astype(np.float32)
        if drop_z is not None: q[:, 2] = drop_z
        sim.set_state(q, np.zeros((n, m.nv), np.float32)); v = sim.forward(); sim.after(v); sim.synchronize()
        rec = {k: [] for k in ("joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel_b", "root_ang_vel_b", "torque", "contact",
                               "contact_touch", "penetration", "limit_excursion", "solver_niter")}
        for t in range(T):
            a_isaac = act_fn(t)                                              # (n, nj) in Isaac joint order
            ctrl = np.tile(default[7:], (n, 1)).astype(np.float32)
            ctrl[:, act_isaac] = default[7:][act_isaac] + ACTION_SCALE * a_isaac
            sim.t.ctrl.copy_(torch.as_tensor(ctrl)); pen.zero_(); exc.zero_(); sim.synchronize()
            vs = sim.step(); sim.synchronize()
            qp = sim.d.qpos.numpy(); qv = sim.d.qvel.numpy(); tau = sim.d.qfrc_actuator.numpy(); sd = sim.d.sensordata.numpy()
            Rm = np.zeros((n, 9)); [mujoco.mju_quat2Mat(Rm[e], qp[e, 3:7]) for e in range(n)]
            R = Rm.reshape(n, 3, 3)
            rec["joint_pos"].append(qp[:, 7:][:, to_isaac]); rec["joint_vel"].append(qv[:, 6:][:, to_isaac])
            rec["root_pos"].append(qp[:, :3].copy()); rec["root_quat"].append(qp[:, 3:7].copy())
            rec["root_lin_vel_b"].append(np.einsum("eji,ej->ei", R, qv[:, :3])); rec["root_ang_vel_b"].append(qv[:, 3:6].copy())
            rec["torque"].append(tau[:, 6:][:, to_isaac])
            cf = np.zeros((n, len(contact_bodies), 3), np.float32); ct_ = np.zeros_like(cf)
            net = csens._net.numpy()
            for k, nm in enumerate(contact_bodies):
                if nm in cs_index: cf[:, k] = net[:, cs_index[nm]]
                if nm in touch: ct_[:, k, 2] = sd[:, touch[nm]]                 # touch sensors (normal force magnitude), previous recordings
            rec["contact"].append(cf); rec["contact_touch"].append(ct_)
            rec["penetration"].append(np.maximum(pen.numpy(), 0.0)); rec["limit_excursion"].append(np.maximum(exc.numpy(), 0.0))
            rec["solver_niter"].append(sim.d.solver_niter.numpy().copy())     # last substep, as Isaac's recording
            if not a.no_render and t % a.frame_every == 0:
                vr = rend2.render(sim, vs, passes=8); rend2.wait_sim_after_render(sim, vr); rend2.after(vr); torch.mps.synchronize()
                iio.imwrite(os.path.join(a.out, f"{tag}_{t:04d}_rgb.png"), rend2.out.rgb[0].cpu().numpy()); np.save(os.path.join(a.out, f"{tag}_{t:04d}_depth.npy"), rend2.out.depth[0].cpu().numpy())
                v0 = rend0.render(sim, vs); rend0.wait_sim_after_render(sim, v0); rend0.after(v0); torch.mps.synchronize()
                iio.imwrite(os.path.join(a.out, f"{tag}_{t:04d}_rgb_tier0.png"), rend0.out.rgb[0].cpu().numpy())
        np.savez_compressed(os.path.join(a.out, f"{tag}.npz"), **{k: np.stack(v) for k, v in rec.items()})
        print(f"[metalsim] {tag}: {T} steps, final root z {rec['root_pos'][-1][:, 2]}", flush=True)


if __name__ == "__main__":
    main()
