"""Real-time preview (MuJoCo's own OpenGL renderer, not the path tracer) of so101Sim's scripted IK expert on servo
options of metalsim/render/film/so101.py, panels side by side. The env is so101Sim's SO101LiftEnv (unchanged), the
servo attached as in runs/readme_media/scratch/so101-servo/eval_cpu.py (servo_cpu.attach: the "_real" options run
the command delay + firmware profile at every physics step).

Timing: one frame per control step, the frame time measured from d.time (0.02 s: 4 physics steps of 5 ms), so the
video plays in real time at 1 / 0.02 = 50 fps. (An earlier version took env.dt and played 4x too fast.)

  <so101Sim>/urlab_bridge/.venv/bin/python render_expert.py <so101Sim root> <out.mp4> <servo>:<seed> [<servo>:<seed> ...]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
METALSIM = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
root, out, specs = sys.argv[1], sys.argv[2], [s.split(":") for s in sys.argv[3:]]
sys.path.insert(0, os.path.join(root, "rl", "native"))
sys.path.append(METALSIM)
sys.path.insert(1, os.path.join(HERE, "..", "so101-servo"))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from so101_lift import SO101LiftEnv  # noqa: E402

import scripted_expert as SE  # noqa: E402
from servo_cpu import attach  # noqa: E402

W, H, T = 640, 480, 300
LABEL = {"7v4": "old 7.4 V model (Menagerie kp 998)", "7v4_real": "7v4_real (STS3215 under LeRobot's P 16)",
         "12v": "old 12 V model (Menagerie)", "12v_real": "12v_real"}
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
except OSError:
    FONT = ImageFont.load_default()
panels, dts = [], []
for servo, seed in specs:
    seed = int(seed)
    env = SO101LiftEnv()
    attach(env, servo)
    m, d = env.model, env.data
    ex = SE.Expert(env)
    obs, _ = env.reset(seed=seed)
    ex.reset()
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    target = d.xpos[env._box_body].copy()
    cam.lookat[:] = 0.5 * (target + d.xpos[1]) + np.array([0, 0, 0.08])
    cam.distance, cam.azimuth, cam.elevation = 0.62, 135.0, -22.0
    frames, succ_t = [], None
    for k in range(T):
        t0 = d.time
        obs, rew, term, trunc, info = env.step(SE.norm_ctrl(m, ex.act()))
        dts.append(d.time - t0)
        if info["is_success"] and succ_t is None:
            succ_t = d.time
        r.update_scene(d, camera=cam)
        im = Image.fromarray(r.render())
        dr = ImageDraw.Draw(im)
        txt = f"{LABEL.get(servo, servo)}   seed {seed}\nt = {d.time:4.2f} s   phase {ex.phase}" + (
            f"   lifted at {succ_t:.2f} s" if succ_t is not None else "")
        dr.rectangle([0, 0, W, 50], fill=(0, 0, 0))
        dr.text((8, 4), txt, fill=(255, 255, 255), font=FONT)
        frames.append(np.asarray(im))
    panels.append(frames)
    print(f"{servo} seed {seed}: success {succ_t is not None}" + (f" at {succ_t:.2f} s" if succ_t else "")
          + f", box z end {d.xpos[env._box_body][2]:.3f} m, phase end {ex.phase}")
dt = float(np.median(dts))
assert np.allclose(dts, dt), "control period not constant"
fps = 1.0 / dt
print(f"control period from d.time: {dt:.4f} s -> {fps:.1f} fps (real time), {T} frames = {T * dt:.1f} s")
ff = subprocess.Popen(["/opt/homebrew/bin/ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                       f"{W * len(panels)}x{H}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-crf", "20", out], stdin=subprocess.PIPE)
for k in range(T):
    ff.stdin.write(np.concatenate([p[k] for p in panels], axis=1).tobytes())
ff.stdin.close()
ff.wait()
print("wrote", out)
