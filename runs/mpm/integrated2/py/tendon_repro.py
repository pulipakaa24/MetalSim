"""Step MuJoCo Warp on the Franka panda (has tendons) once on Metal; reports success or the load error."""
import os, sys
import mujoco
import warp as wp
import mujoco_warp as mjw

print("warp", wp.__file__, "device_args", wp.config.metal_device_args, flush=True)
m = mujoco.MjModel.from_xml_path("/Users/aditya/robosim/upstream/mujoco_menagerie/franka_emika_panda/panda.xml")
with wp.ScopedDevice("metal:0"):
    mm = mjw.put_model(m)
    d = mjw.make_data(m, nworld=4)
    try:
        mjw.step(mm, d)
        wp.synchronize()
        print("RESULT ok", flush=True)
    except Exception as e:  # noqa: BLE001
        print("RESULT fail", str(e)[:300], flush=True)
