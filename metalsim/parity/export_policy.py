"""Export a MetalSim G1 checkpoint for isaac_side/play_policy.py: actor state dict + our joint order."""
import sys, torch, mujoco
from metalsim.learn.g1_velocity import build_g1_model
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
m, _ = build_g1_model("flat")
joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
actor = {k.replace("actor.", ""): v.cpu() for k, v in ck["net"].items() if k.startswith("actor.")}
# height-scan ray order (rough): checkpoints before 2026-09-25 carry no key and were trained on the "ij" order
order = ck.get("scan_ordering") or ("ij" if ck.get("terrain", "flat") != "flat" else None)
torch.save({"actor": actor, "joint_names": joints, "iterations": ck.get("iterations"), "hidden": ck.get("hidden"),
            "scan_ordering": order}, dst)
print(f"exported {src} -> {dst}: {len(actor)} tensors, {len(joints)} joints")
