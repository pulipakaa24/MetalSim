"""Convert Isaac's rsl_rl G1-flat checkpoint (Isaac joint order) into a MetalSim checkpoint (our joint order) by
folding the joint permutation into the first and last layers, so scripts/diagnostics/g1_reward_terms.py can play
it unchanged. Same mapping as metalsim/parity/side_by_side.py (observation = 12 base terms, then joint pos, joint
vel, last action, each in the policy's joint order); checked here on random observations against that path.

    python scripts/diagnostics/contact_research/isaac_ckpt_to_metalsim.py 1000 [500 ...]
"""
import os, sys
import numpy as np, torch, mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from metalsim.learn.g1_velocity import build_g1_model
from metalsim.learn.warp_policy import ActorCriticMLP

ISAAC_JOINTS = [str(s) for s in np.load(os.path.join(ROOT, "runs/parity/isaac/parity_out2/rt/action_sequence_B.npz"))["joint_names"]]
m, _ = build_g1_model("flat")
ours = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
nj = len(ours)
perm_obs = np.array([ours.index(n) for n in ISAAC_JOINTS])      # isaac-order obs i <- our obs perm_obs[i]
perm_act = np.array([ISAAC_JOINTS.index(n) for n in ours])      # our action j <- isaac action perm_act[j]


def convert(it):
    ck = torch.load(os.path.join(ROOT, f"runs/parity/isaac/ckpt_out/model_{it}.pt"), map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]; out = {}
    cols = np.arange(12 + 3 * nj)
    src_cols = cols.copy()                                       # new column c (our order) takes old column src_cols[c]
    for b in range(3):
        base = 12 + b * nj
        src_cols[base + perm_obs] = base + np.arange(nj)
    for net in ("actor", "critic"):
        for k, v in sd.items():
            if not k.startswith(net + "."): continue
            v = v.clone()
            if k.endswith(".0.weight"): v = v[:, torch.as_tensor(src_cols)]
            if net == "actor" and k in ("actor.6.weight", "actor.6.bias"): v = v[torch.as_tensor(perm_act)]
            out[k] = v
    out["log_std"] = torch.log(sd["std"][torch.as_tensor(perm_act)])
    # check against the side_by_side path
    net_new = ActorCriticMLP(12 + 3 * nj, nj, hidden=(256, 128, 128)); net_new.load_state_dict(out)
    net_old = ActorCriticMLP(12 + 3 * nj, nj, hidden=(256, 128, 128))
    net_old.actor.load_state_dict({k.replace("actor.", ""): v for k, v in sd.items() if k.startswith("actor.")})
    o = torch.randn(64, 12 + 3 * nj); po = torch.as_tensor(perm_obs); pa = torch.as_tensor(perm_act)
    oi = torch.cat([o[:, :12], o[:, 12:12 + nj][:, po], o[:, 12 + nj:12 + 2 * nj][:, po], o[:, 12 + 2 * nj:][:, po]], 1)
    with torch.no_grad():
        err = (net_new.actor(o) - net_old.actor(oi)[:, pa]).abs().max().item()
    assert err < 1e-5, err
    fn = os.path.join(ROOT, f"runs/contact_research/isaac_model_{it}_metalsim.pt")
    torch.save({"net": out, "terrain": "flat", "iterations": it, "obs_dim": 12 + 3 * nj, "act_dim": nj, "hidden": (256, 128, 128),
                "learner": "isaac rsl_rl (PhysX), converted by isaac_ckpt_to_metalsim.py"}, fn)
    print(f"it {it}: max |action diff| vs side_by_side mapping {err:.2e} -> {fn}")


for it in sys.argv[1:] or ["1000"]:
    convert(int(it))
