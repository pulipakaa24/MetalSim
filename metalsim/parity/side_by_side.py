"""Side-by-side videos: the same policy checkpoint rolled out in Isaac Lab (frames recorded on the VM
by isaac_side/play_policy.py) and in MetalSim (tier 2, same camera, same lights, same initial state
and command), one panel each, for several training stages.

    python -m metalsim.parity.side_by_side --isaac runs/parity/play/isaac_it500 --ckpt runs/policies/g1_flat_dt25_fixed_it500.pt \
        --out docs/gallery/g1_stage_it500.mp4 --label "iteration 500"

The MetalSim rollout uses the exported policy's own joint order (our checkpoints) or an rsl_rl
checkpoint (Isaac's training) mapped by joint name.
"""
import argparse, json, os
import numpy as np, torch, mujoco, warp as wp
import imageio.v2 as iio

from metalsim.learn.g1_velocity import build_g1_model, G1VelocityTask, ACTION_SCALE
from metalsim.learn.warp_policy import ActorCriticMLP
from metalsim.render.tier2 import Tier2Renderer
from metalsim.physics.batch import BatchSim, BatchSimOptions
from metalsim.parity.record_g1 import hero_spec


def load_policy(path, our_joints):
    ck = torch.load(path, map_location="mps", weights_only=False)
    if "model_state_dict" in ck:                     # rsl_rl (Isaac's training): actor.* keys, Isaac joint order
        sd = {k.replace("actor.", ""): v for k, v in ck["model_state_dict"].items() if k.startswith("actor.")}
        isaac_joints = ck.get("joint_names")
        return sd, isaac_joints, "rsl_rl"
    sd = {k.replace("actor.", ""): v for k, v in ck["net"].items() if k.startswith("actor.")}
    return sd, our_joints, "metalsim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaac", required=True, help="directory from isaac_side/play_policy.py (isaac.mp4, traj.npz, meta.json)")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True); ap.add_argument("--label", default="")
    ap.add_argument("--isaac_meta", default=None, help="meta.json of the Isaac recording (camera/lights); defaults to <isaac>/../../rt/meta.json")
    ap.add_argument("--physics_dt", type=float, default=0.0025); ap.add_argument("--spp", type=int, default=16); ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--tier2_mode", default="rtx", help="tier-2 preset (metalsim.render.rtx_parity): rtx = RTX-parity default, legacy = before 2026-09-25")
    a = ap.parse_args(); wp.config.quiet = True
    meta_path = a.isaac_meta or os.path.join(a.isaac, "..", "..", "rt", "meta.json")
    meta = json.load(open(meta_path)); pmeta = json.load(open(os.path.join(a.isaac, "meta.json")))
    traj = np.load(os.path.join(a.isaac, "traj.npz")); isaac_joints = [str(s) for s in traj["joint_names"]]
    steps = int(pmeta["steps"]); fe = int(pmeta["frame_every"])
    task = G1VelocityTask(1, terrain="flat", seed=0, physics_dt=a.physics_dt)
    m, info = build_g1_model("flat", visuals=True, physics_dt=a.physics_dt); hero = hero_spec(info["spec"], meta).compile()
    task.model = hero
    task.sim = BatchSim(hero, 1, options=BatchSimOptions(substeps=task.decimation, njmax=256, nconmax=32, solver_iterations=10, ls_iterations=20)); task.sim.synchronize()
    our_joints = [mujoco.mj_id2name(hero, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, hero.njnt)]
    sd, pol_joints, src = load_policy(a.ckpt, our_joints)
    if pol_joints is None: pol_joints = isaac_joints
    net = ActorCriticMLP(task.obs_dim, task.act_dim, hidden=tuple(sd[k].shape[0] for k in sorted((k for k in sd if k.endswith("weight")), key=lambda s: int(s.split(".")[0]))[:-1])).to("mps")
    net.actor.load_state_dict({k: v for k, v in sd.items()}); net.eval()
    nj = task.nj
    # observation / action permutations between the policy's joint order and ours
    pol_of_ours = np.array([pol_joints.index(n) for n in our_joints]); ours_of_pol = np.array([our_joints.index(n) for n in pol_joints])
    perm_obs = torch.as_tensor(ours_of_pol, device="mps"); perm_act = torch.as_tensor(pol_of_ours, device="mps")
    cam = meta["camera"]
    from metalsim.parity.record_g1 import tier2_parity_kwargs
    rend = Tier2Renderer(hero, 1, width=cam["width"], height=cam["height"], camera="hero", spp=a.spp, max_bounces=3, **tier2_parity_kwargs(meta, a.tier2_mode))
    # deterministic start: Isaac's default state at the origin, command (0.5, 0, 0)
    task.origins.assign(np.zeros((1, 3), np.float32)); task.reset_all()
    q0 = hero.key_qpos[0].astype(np.float32); task.sim.t.qpos.copy_(torch.as_tensor(q0)[None]); task.sim.t.qvel.zero_()
    torch.mps.synchronize()      # the copies run on the MPS queue; forward kinematics (Warp queue) must see them
    v = task.sim.forward(); task.sim.after(v); task.sim.synchronize()
    task.cmd.assign(np.array([[0.5, 0.0, 0.0]], np.float32)); task.resample.assign(np.zeros(1, bool))
    from metalsim.interop import torch_bridge as tb
    step_idx = wp.zeros(1, dtype=int, device="metal:0"); from metalsim.learn.warp_policy import bump
    t_obs = tb.mps_tensor(task.obs)
    ours_frames = []
    for t in range(steps):
        wp.launch(bump, dim=1, inputs=[step_idx], device="metal:0"); task.launch_obs(step_idx); vs = task.sim._signal(); task.sim.after(vs)
        with torch.no_grad():
            o = t_obs.clone()
            if src == "rsl_rl":     # policy expects Isaac's joint order inside the joint blocks
                head = o[:, :12]; jp = o[:, 12:12 + nj][:, perm_obs]; jv = o[:, 12 + nj:12 + 2 * nj][:, perm_obs]; la = o[:, 12 + 2 * nj:][:, perm_obs]
                o = torch.cat([head, jp, jv, la], 1)
            act = net.actor(o)
            if src == "rsl_rl": act = act[:, perm_act]
        task.action_scratch.assign(act.cpu().numpy().astype(np.float32)); task.launch_apply_action(task.action_scratch)
        vstep = task.sim.step()
        if t % fe == 0:
            vr = rend.render(task.sim, vstep, passes=a.passes); rend.wait_sim_after_render(task.sim, vr); rend.after(vr); torch.mps.synchronize()
            ours_frames.append(rend.out.rgb[0].cpu().numpy().copy())
    isaac_frames = [f[..., :3] for f in iio.mimread(os.path.join(a.isaac, "isaac.mp4"), memtest=False)]
    n = min(len(isaac_frames), len(ours_frames))
    from PIL import Image, ImageDraw
    out = []
    for i in range(n):
        strip = np.concatenate([isaac_frames[i], ours_frames[i]], axis=1)
        im = Image.fromarray(strip); d = ImageDraw.Draw(im)
        d.text((12, 10), f"Isaac Sim (RTX) | {a.label}", fill=(255, 255, 255)); d.text((strip.shape[1] // 2 + 12, 10), f"MetalSim tier 2 | {a.label}", fill=(255, 255, 255))
        out.append(np.asarray(im))
    iio.mimwrite(a.out, out, fps=max(1, int(50 / fe)), codec="libx264", quality=8, macro_block_size=None)
    ours_root = task.sim.d.qpos.numpy()[0, :3]; isaac_root = traj["root_pos"][-1, 0]
    print(f"wrote {a.out}: {n} frames; final root z Isaac {isaac_root[2]:.3f} MetalSim {ours_root[2]:.3f}; x travelled Isaac {isaac_root[0]:.2f} MetalSim {ours_root[0]:.2f}")


if __name__ == "__main__":
    main()
