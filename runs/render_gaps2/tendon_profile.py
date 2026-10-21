"""Per-pass split of the tendon scene's tier-0 frame (catenaries vs straight), and with a coarser tube tessellation."""
import json
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts/diagnostics")
sys.path.insert(0, "tests")
import mujoco
import warp as wp
wp.config.quiet = True
from metalsim.render import gpu_profile
gpu_profile.install()
from render_gaps2_cost import batch, frame_ms, T0
from test_render_tendons import TENDONS
m = mujoco.MjModel.from_xml_string(TENDONS.replace("<worldbody>", '<worldbody><geom type="plane" size="2 2 0.1"/>', 1))
sim = batch(m, 1024)
for label, kw in (("mujoco", {}), ("straight", dict(tendons="straight")), ("mujoco_tube8", dict(tube_slices=8, cap_stacks=2))):
    r = T0(m, 1024, 128, "cam", **kw)
    ms = frame_ms(r, sim)
    prof = gpu_profile.PassProfiler(r.ctx)
    with prof:
        for _ in range(4):
            r.render(sim); prof.next_frame()
        r.ctx.synchronize()
    tris = sum(len(o["tris"]) for o in r.dyn.objects)
    print(json.dumps(dict(config=label, ms=ms, dyn_tris_per_env=tris, passes=prof.table()[:4])), flush=True)
