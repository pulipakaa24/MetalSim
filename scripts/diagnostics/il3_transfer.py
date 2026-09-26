"""Transfer of Isaac Lab 3.0-EA's own G1 flat checkpoints (rsl_rl 5.4.1, trained on Isaac's Newton/MuJoCo-Warp or PhysX
backend, runs/parity3/isaac/train/rsl_rl_flat_<backend>/model_<it>.pt) into MetalSim, per physics preset.

Protocol (as scripts/diagnostics/newton_transfer.py, the 2.3.2 transfer): Isaac's default state at the origin, command
(0.5, 0, 0) held, mean action, 400 control steps (8 s), 4 envs, observation noise on (as in training); the task is
reward_cfg="flat_il3" with il3_events=False (3.0's observation semantics: base_lin_vel of the pelvis COM; no pushes,
mass randomization or reset velocities, so the protocol is deterministic). Reports x travelled, final pelvis z, torso
contacts (falls) and, per step, the largest joint-limit excursion. Policy joint order = the backend's articulation
order recorded in runs/parity3/isaac/fidelity/<backend>/meta.json.

usage: python scripts/diagnostics/il3_transfer.py [--its 500,1000,1499] [--backend newton_mjwarp]
                                                  [--presets default,hardlimits,isaaclab3] [--json out.json]
"""
import argparse, json, sys
import numpy as np, torch, warp as wp, mujoco
wp.config.quiet = True
from metalsim.learn.g1_velocity import G1VelocityTask
from metalsim.learn.warp_policy import bump
from metalsim.interop import torch_bridge as tb

N, STEPS = 4, 400


def load_actor(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]
    lin = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    layers = []
    for i, li in enumerate(lin):
        w, b = sd[f"mlp.{li}.weight"], sd[f"mlp.{li}.bias"]
        l = torch.nn.Linear(w.shape[1], w.shape[0]); l.weight.data.copy_(w); l.bias.data.copy_(b); layers.append(l)
        if i < len(lin) - 1:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).to("mps").eval()


def make(preset):
    """preset: a contact_tuning name (applied as the task's contact_cfg, "default" = MuJoCo's defaults) or a solver_presets
    name (applied on top of MuJoCo's defaults)."""
    from metalsim.physics import solver_presets
    sp = preset in solver_presets.PRESETS
    return G1VelocityTask(N, terrain="flat", seed=0, physics_dt=0.0025, reward_cfg="flat_il3", il3_events=False,
                          contact_cfg="default" if sp else preset, solver_cfg=preset if sp else None)


def play(task, actor, pol_joints):
    m = task.model; nj = task.nj
    ours = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    perm_obs = torch.as_tensor(np.array([ours.index(n) for n in pol_joints]), device="mps")
    perm_act = torch.as_tensor(np.array([pol_joints.index(n) for n in ours]), device="mps")
    task.origins.assign(np.zeros((N, 3), np.float32)); task.reset_all()
    q0 = np.tile(m.key_qpos[0], (N, 1)).astype(np.float32)
    task.sim.t.qpos.copy_(torch.as_tensor(q0)); task.sim.t.qvel.zero_(); v = task.sim.forward(); task.sim.after(v)
    task.sim.synchronize()
    task.cmd.assign(np.tile(np.array([0.5, 0.0, 0.0], np.float32), (N, 1))); task.resample.assign(np.zeros(N, bool))
    step_idx = wp.zeros(1, dtype=int, device="metal:0"); t_obs = tb.mps_tensor(task.obs)
    lim = [j for j in range(1, m.njnt) if m.jnt_limited[j]]
    adr = np.array([m.jnt_qposadr[j] for j in lim]); lo = m.jnt_range[lim, 0]; hi = m.jnt_range[lim, 1]
    exc = 0.0; exc_j = None; torso = np.zeros(N, bool)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in lim]
    for _ in range(STEPS):
        wp.launch(bump, dim=1, inputs=[step_idx], device="metal:0"); task.launch_obs(step_idx)
        vs = task.sim._signal(); task.sim.after(vs)
        with torch.no_grad():
            o = t_obs.clone()
            o = torch.cat([o[:, :12], o[:, 12:12 + nj][:, perm_obs], o[:, 12 + nj:12 + 2 * nj][:, perm_obs], o[:, 12 + 2 * nj:][:, perm_obs]], 1)
            act = actor(o)[:, perm_act]
        task.action_scratch.assign(act.cpu().numpy().astype(np.float32)); task.launch_apply_action(task.action_scratch)
        task.sim.step(); task.sim.synchronize()
        q = task.sim.d.qpos.numpy()
        ex = np.maximum(lo - q[:, adr], q[:, adr] - hi).max(0)
        if ex.max() > exc:
            exc = float(ex.max()); exc_j = names[int(ex.argmax())]
        torso |= task.sim.d.sensordata.numpy()[:, int(task.touch_adr[2])] > 1.0
    q = task.sim.d.qpos.numpy()
    return {"x": q[:, 0].tolist(), "z": q[:, 2].tolist(), "falls": int(torso.sum()), "limit_excursion_rad": max(exc, 0.0), "limit_excursion_joint": exc_j}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--its", default="500,1000,1499"); ap.add_argument("--backend", default="newton_mjwarp")
    ap.add_argument("--presets", default="default,hardlimits,isaaclab3"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    pol_joints = json.load(open(f"runs/parity3/isaac/fidelity/{a.backend}/meta.json"))["joint_names"]
    presets = a.presets.split(","); its = [int(x) for x in a.its.split(",")]
    print(f"Isaac Lab 3.0 {a.backend} checkpoints in MetalSim, command (0.5, 0, 0) for {STEPS * 0.02:.0f} s (commanded 4.0 m), "
          f"{N} envs, mean action; x travelled [m] mean (min-max) / final pelvis z [m] / torso contacts / max limit excursion [rad]")
    print("| checkpoint | " + " | ".join(presets) + " |"); print("|---|" + "---|" * len(presets), flush=True)
    tasks = {p: make(p) for p in presets}
    out = {}
    for it in its:
        actor = load_actor(f"runs/parity3/isaac/train/rsl_rl_flat_{a.backend}/model_{it}.pt")
        row = [f"it {it}"]
        for p in presets:
            r = play(tasks[p], actor, pol_joints); out[f"{it}/{p}"] = r
            x = np.array(r["x"]); z = np.array(r["z"])
            row.append(f"{x.mean():.2f} ({x.min():.2f}-{x.max():.2f}) / {z.mean():.3f} / {r['falls']} / {r['limit_excursion_rad']:.3f} ({r['limit_excursion_joint']})")
        print("| " + " | ".join(row) + " |", flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
