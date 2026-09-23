"""Image-level comparison of tier-0 against mujoco.Renderer on the same scene/state: PSNR and
NVIDIA FLIP (lower is better; 0 = identical). Shading models differ by design (GGX vs Phong), so
these are tracked, not thresholded, until tier 1 is compared against Isaac RTX instead.
Usage: python -m metalsim.bench.render_fidelity [scene.xml] [camera]
"""
import os, sys
import numpy as np, mujoco, imageio.v2 as iio
from metalsim.render.tier0 import Tier0Renderer

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

def flip(a, b):
    try:
        import flip_evaluator
        err, mean, _ = flip_evaluator.evaluate(a.astype(np.float32) / 255.0, b.astype(np.float32) / 255.0, "LDR")
        return float(mean)
    except Exception as e:
        return float("nan")

def compare(model, cam, qpos=None, res=256, shadows=True, label="", out=None, tier=0):
    d = mujoco.MjData(model)
    if qpos is not None:
        d.qpos[:len(qpos)] = qpos
    mujoco.mj_forward(model, d)
    ref = mujoco.Renderer(model, res, res); ref.update_scene(d, camera=cam); ri = ref.render()
    rend = Tier0Renderer(model, 1, width=res, height=res, camera=cam, outputs=("rgb",), shadows=shadows, tier=tier)
    ours = rend.render_host([d])["rgb"][0]
    p = psnr(ri, ours); f = flip(ri, ours)
    print(f"{label:30s} tier={tier} shadows={shadows}: PSNR {p:5.2f} dB  FLIP {f:.4f}  mean intensity ref {ri.mean():.1f} ours {ours.mean():.1f}")
    if out:
        iio.imwrite(out, np.concatenate([ri, ours], axis=1))
    return ri, ours

if __name__ == "__main__":
    os.makedirs("docs/measurements", exist_ok=True)
    scene = sys.argv[1] if len(sys.argv) > 1 else "assets/so101/scene_box_rl.xml"
    cam = sys.argv[2] if len(sys.argv) > 2 else "base_cam"
    m = mujoco.MjModel.from_xml_path(scene)
    compare(m, cam, qpos=[0.4, -0.3, 0.5, 0.2, 0.3, 0.5], shadows=False, label="so101 (no shadows)")
    compare(m, cam, qpos=[0.4, -0.3, 0.5, 0.2, 0.3, 0.5], shadows=True, label="so101 (shadows)", out="docs/measurements/tier0_lit_vs_reference.png")
    compare(m, cam, qpos=[0.4, -0.3, 0.5, 0.2, 0.3, 0.5], label="so101 tier 1 (RT)", out="docs/measurements/tier1_vs_reference.png", tier=1)
