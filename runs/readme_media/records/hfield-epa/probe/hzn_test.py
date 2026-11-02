"""collision_gjk_test.test_box_box_horizon's configuration with horizon buffers of several sizes (stream hfield-epa)."""
import sys
import warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import collision_gjk_test as T
from mujoco_warp import test_data
from mujoco_warp._src.types import OverflowType
print(mjw.__file__)
wp.set_device("cpu")
_, _, m, d = test_data.fixture(xml="""<mujoco><worldbody><geom size=".025 .025 .025" type="box"/>
<geom size=".025 .025 .025" type="box"/></worldbody></mujoco>""")
pos1 = wp.vec3(0.065118454396725, -0.125125020742416, 0.124963559210300)
rot1 = wp.mat33(0.996357858181000, 0.085266821086407, -0.000942531623878, -0.085266284644604, 0.996358215808868,
                0.000591202871874, 0.000989508931525, -0.000508683384396, 0.999999582767487)
pos2 = wp.vec3(0.065104484558105, -0.124979749321938, 0.174992129206657)
rot2 = wp.mat33(0.996556758880615, -0.082913912832737, -0.000453041866422, 0.082915119826794, 0.996536433696747,
                0.006357696373016, -0.000075668765930, -0.006373368669301, 0.999979794025421)
for h in [3, 4, 5, 6, 8, 10, 12, 13, 14, 16, 24, 114]:
  ov = wp.zeros(1, dtype=int)
  dist, _, _, _ = T._geom_dist(m, d, 0, 1, multiccd=False, pos1=pos1, mat1=rot1, pos2=pos2, mat2=rot2, horizon_size=h,
                               warn_overflow=False, overflow_out=ov)
  print(f"horizon {h:3d}: dist {dist:.10f} overflow {bool(ov.numpy()[0] & OverflowType.EPA_HORIZON)}")
