"""Contacts per world after 200 steps of the metalsim_step.py setup (generated from it: same model, controller, settings)."""
import sys, time, json, numpy as np, mujoco, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *
STEPS = 200
import genesis as gs
name, N, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
R = ROBOTS[name]
m = mujoco.MjModel.from_xml_path(R["scene"]); home = m.key_qpos[0].copy()
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]
LIM = force_limits(m); lo, hi = LIM[:, 0], LIM[:, 1]
qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True, seed=0)
ro = dict(iterations=10, ls_iterations=20) if mode == "matched" else {}
t0 = time.time()
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, substeps=1), rigid_options=gs.options.RigidOptions(**ro), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.MJCF(file=R["robot"]))
scene.build(n_envs=N)
dofs = [robot.get_joint(j).dofs_idx_local[0] for j in jn]
robot.set_dofs_kp(np.full(m.nu, R["kp"]), dofs); robot.set_dofs_kv(np.full(m.nu, R["kd"]), dofs)
robot.set_dofs_force_range(lo, hi, dofs)
robot.set_qpos(torch.tensor(home, device=gs.device, dtype=torch.float32).repeat(N, 1))
t_build = time.time() - t0
q_home = torch.tensor(home[qadr], device=gs.device, dtype=torch.float32)
g = torch.Generator(device=gs.device).manual_seed(0)
def resample():
    robot.control_dofs_position(q_home + AMP * (2 * torch.rand((N, m.nu), device=gs.device, generator=g) - 1), dofs)
for i in range(WARMUP):
    if i % RESAMPLE == 0: resample()
    scene.step()
_ = robot.get_qpos().sum().item(); torch.mps.synchronize()
t0 = time.perf_counter()
for i in range(STEPS):
    if i % RESAMPLE == 0: resample()
    scene.step()
q = robot.get_qpos(); _ = q.sum().item(); torch.mps.synchronize(); el = time.perf_counter() - t0
so = scene.sim.rigid_solver._options if hasattr(scene.sim.rigid_solver, "_options") else None
c = robot.get_contacts(is_padded=True); vm = c["valid_mask"]
cs = robot.get_contacts(with_entity=robot, is_padded=True)["valid_mask"]
print("CONTACTS", name, "genesis per_world", round(vm.sum().item() / N, 2), "self", round(cs.sum().item() / N, 2))
