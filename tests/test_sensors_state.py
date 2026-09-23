"""IMU / contact / joint sensors from physics state as zero-copy tensors (MuJoCo sensors in the
MJCF: accelerometer, gyro, touch, force, framequat; MuJoCo Warp evaluates them per step)."""
import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from orchard.physics.batch import BatchSim, BatchSimOptions

pytestmark = pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")

XML = """
<mujoco><option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="b" pos="0 0 0.3"><freejoint/>
      <geom name="g" type="box" size="0.05 0.05 0.05" mass="1"/>
      <site name="imu" pos="0 0 0"/>
      <site name="foot" pos="0 0 -0.05" size="0.06 0.06 0.005" type="box"/>
    </body>
  </worldbody>
  <sensor>
    <accelerometer name="acc" site="imu"/>
    <gyro name="gyro" site="imu"/>
    <framequat name="quat" objtype="site" objname="imu"/>
    <touch name="touch" site="foot"/>
  </sensor>
</mujoco>"""


def test_imu_and_contact_sensors_as_tensors():
    model = mujoco.MjModel.from_xml_string(XML)
    n = 16
    sim = BatchSim(model, n, options=BatchSimOptions(substeps=5, njmax=64))
    sim.synchronize()
    adr = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i): (int(model.sensor_adr[i]), int(model.sensor_dim[i])) for i in range(model.nsensor)}
    sd = sim.t.sensordata                                   # (N, nsensordata) zero-copy MPS tensor
    assert tuple(sd.shape) == (n, model.nsensordata)
    # free fall: accelerometer reads ~0 (specific force) before contact, then ~+g resting on the floor; touch > 0 at rest
    for _ in range(10):
        sim.step()
    sim.synchronize()
    a0, d0 = adr["acc"]
    acc_fall = sim.d.sensordata.numpy()[:, a0:a0 + 3]
    assert np.all(np.abs(acc_fall[:, 2]) < 1.0), acc_fall[:, 2]
    for _ in range(300):
        sim.step()
    sim.synchronize()
    v = sim.d.sensordata.numpy()
    acc_rest = v[:, a0:a0 + 3]; t0, _ = adr["touch"]; touch = v[:, t0]
    assert np.allclose(acc_rest[:, 2], 9.81, atol=0.3), acc_rest[:, 2]
    assert np.all(touch > 5.0), touch          # ~ m g = 9.81 N on the foot site
    q0, _ = adr["quat"]
    assert np.allclose(np.abs(v[:, q0]), 1.0, atol=0.05)
    # per-body contact forces from the physics state as well (Isaac ContactSensor equivalent)
    cf = sim.t.cfrc_ext if hasattr(sim.d, "cfrc_ext") else None
    if cf is not None:
        assert tuple(cf.shape)[:2] == (n, model.nbody)
