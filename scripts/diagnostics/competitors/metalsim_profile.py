"""Why is Go2 slow on MetalSim? Same model/settings as metalsim_step.py (matched). Modes:
  kernels : per-kernel GPU time, eager (run with WP_METAL_PROFILE=1)
  variants: synchronized throughput of model variants that isolate collision costs."""
import sys, time, json, mujoco, numpy as np, torch, warp as wp
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *
from metalsim.physics.batch import BatchSim, BatchSimOptions
name, N, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
R = ROBOTS[name]

def build(variant="base"):
    spec = mujoco.MjSpec.from_file(R["scene"])
    LIM = force_limits(mujoco.MjModel.from_xml_path(R["scene"])) if R["kp"] is not None else []
    for a, (lo, hi) in zip(spec.actuators, LIM):
        a.set_to_position(kp=R["kp"], kv=R["kd"]); a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE; a.forcerange = [lo, hi]
    spec.option.timestep = DT
    import os
    if os.environ.get('CONE'): spec.option.cone = {'pyramidal': mujoco.mjtCone.mjCONE_PYRAMIDAL, 'elliptic': mujoco.mjtCone.mjCONE_ELLIPTIC}[os.environ['CONE']]
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    for g in spec.geoms:
        g.margin = 0.0
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE: continue
        if "capsule" in variant and g.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        if "flooronly" in variant and (g.contype or g.conaffinity):
            g.contype, g.conaffinity = 2, 1            # robot geoms touch the floor (contype 1) but not each other
    m = spec.compile()
    return m

def run(m, capture=True):
    home = m.key_qpos[0].copy() if m.nkey else m.qpos0.copy()
    qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    sim = BatchSim(m, N, options=BatchSimOptions(solver_iterations=10, ls_iterations=20, capture=capture, jacobian=__import__("os").environ.get("JAC"), **({"njmax": int(__import__("os").environ["NJMAX"])} if __import__("os").environ.get("NJMAX") else {})))
    sim.set_state(home); sim.forward(); sim.synchronize()
    q_home = torch.tensor(home[qadr], device="mps", dtype=torch.float32)
    g = torch.Generator(device="mps").manual_seed(0)
    def resample():
        sim.t.ctrl.copy_(q_home + AMP * (2 * torch.rand((N, m.nu), device="mps", generator=g) - 1)); torch.mps.synchronize()
    return sim, resample

if mode == "variants":
    for v in ("base", "flooronly", "capsule", "capsule+flooronly"):
        m = build(v); sim, resample = run(m)
        for i in range(100):
            if i % RESAMPLE == 0: resample()
            sim.step()
        sim.synchronize(); t0 = time.perf_counter()
        for i in range(500):
            if i % RESAMPLE == 0: resample()
            sim.step()
        sim.synchronize(); el = time.perf_counter() - t0
        print("VARIANT", json.dumps(dict(robot=name, N=N, variant=v, steps_per_s=round(N * 500 / el),
              contacts_per_world=round(float(sim.t.nacon.flatten()[0].item()) / N, 2), base_z=round(sim.t.qpos[:, 2].mean().item(), 3))), flush=True)
else:
    m = build("base"); sim, resample = run(m, capture=False)
    core = wp._src.context.runtime.core
    for i in range(60):
        if i % RESAMPLE == 0: resample()
        sim.step()
    sim.synchronize(); core.wp_metal_profile_report()
    for i in range(10): sim.step()
    sim.synchronize(); time.sleep(0.2)
    rows = []
    for ln in core.wp_metal_profile_report().decode().strip().splitlines():
        p = ln.split(None, 3); rows.append((float(p[0]) / 10, int(p[2]) // 10, p[3]))
    tot = sum(r[0] for r in rows)
    print(f"KERNELS {name} N={N}: {tot:.2f} ms/step GPU (sum of per-dispatch times), {sum(r[1] for r in rows)} dispatches/step")
    for ms, cnt, k in sorted(rows, reverse=True)[:18]:
        print(f"   {ms:7.3f} ms {100*ms/tot:5.1f}%  x{cnt:3d}  {k[:110]}")
