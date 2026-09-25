"""Eager timing of MuJoCo Warp's flex collision stages on Metal (synchronize around each stage)."""
import sys, time, json, numpy as np, mujoco, warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
from mujoco_warp._src import collision_flex as cf, collision_core, collision_driver
from metalsim.physics import deformable as dfm
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
DEV = "metal:0"
m = mujoco.MjModel.from_xml_string(dfm.box_scene_xml())
sim = dfm.DeformableSim(m, N, device=DEV, capture=False)
sim.randomize(seed=1)
for _ in range(80): sim.step(eager=True)
M, D = sim.m, sim.d
def T(name, fn, n=5):
    with wp.ScopedDevice(DEV):
        fn(); wp.synchronize_device(DEV)
        t = time.perf_counter()
        for _ in range(n): fn()
        wp.synchronize_device(DEV)
    print(json.dumps({"stage": name, "num_envs": N, "ms": (time.perf_counter() - t) / n * 1e3}), flush=True)
with wp.ScopedDevice(DEV):
    ctx = collision_core.create_collision_context(D.naconmax)
T("collision (all)", lambda: mjw.collision(M, D))
T("rigid-only part: collision with flex skipped", lambda: None)
T("flex_broadphase_aabb", lambda: cf.flex_broadphase_aabb(M, D))
T("allocate workspace", lambda: cf._allocate_flex_workspace(M, D))
ws = [None]
def geom_detect():
    w = cf._allocate_flex_workspace(M, D); w.ncand.zero_(); w.flex_num_groups.zero_()
    cf._detect_plane_flex_candidates(M, D, w); cf._detect_1d_geom_candidates(M, D, w); cf._detect_elem_geom_candidates(M, D, w)
    ws[0] = w
T("geom-flex detect (plane+1d+elem)", geom_detect)
T("geom-flex filter+write", lambda: cf._filter_and_write_contacts(M, D, ws[0], enable_fps=True))
T("flex_geom_collision (detect+filter)", lambda: cf._flex_geom_collision(M, D, cf._allocate_flex_workspace(M, D)))
T("sap sort", lambda: cf._run_flex_sap_sort(M, D))
sap = cf._run_flex_sap_sort(M, D)
T("flex-flex sap collision (sweep+narrow+filter)", lambda: cf._flex_sap_collision(M, D, ctx, cf._allocate_flex_workspace(M, D), is_self=False, sap_data=sap))
T("flex_collision (all flex)", lambda: cf.flex_collision(M, D, ctx))
