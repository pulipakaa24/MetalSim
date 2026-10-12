import sys, traceback, numpy as np, warp as wp
wp.init()
import newton
from newton.solvers import SolverImplicitMPM
print("newton", newton.__file__)
dev = sys.argv[1]; grid = sys.argv[2]
with wp.ScopedDevice(dev):
    b = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(b)
    pts = np.stack(np.meshgrid(*[np.arange(0,0.5,0.05)]*3, indexing="ij"),-1).reshape(-1,3) + [0,0,0.2]
    for p in pts: b.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=0.125*1000*0.05**3, radius=0.025)
    b.add_ground_plane()
    m = b.finalize()
    cfg = SolverImplicitMPM.Config(); cfg.voxel_size = 0.1; cfg.grid_type = grid
    try:
        s = SolverImplicitMPM(m, config=cfg)
        s0 = m.state(); s1 = m.state()
        for i in range(5): s.step(s0, s1, None, None, 1/60); s0, s1 = s1, s0
        q = s0.particle_q.numpy(); print(dev, grid, "OK minz", q[:,2].min(), "meanz", q[:,2].mean())
    except Exception as e:
        print(dev, grid, "FAILED", type(e).__name__, str(e)[:400]); traceback.print_exc(limit=3)
