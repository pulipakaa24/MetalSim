# Research notes: Isaac Lab 3.0-EA's IsaacContrib-Franka-Pour (2026-10-11/12)

What the port (`metalsim/learn/franka_pour*.py`) reproduces, read from Isaac's source. Isaac Lab = `upstream/IsaacLab3`
at `ae37b028` ("[Backport release/3.0.0] Point README at v3.0.0 EA documentation", the commit installed on the L4 VM);
`IL` = `source/`, task files relative to `IL/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/`; Newton = 1.5.2
(`NW` = `newton/_src/`, the Metal fork `metalsim-1.5.2` at `35791a07`); rsl_rl = `rsl-rl-lib` 5.4.1 (Isaac Lab 3.0-EA's
pin, `uv.lock:2084-2085`; read from the 5.4.1 wheel, PyPI).

## 1. Simulation stack per policy step

* `pour_env_cfg.py:597-677`: sim dt 1/120, decimation 4, episode 12 s (`max_episode_length = ceil(12 / (4/120)) = 360`,
  `manager_based_rl_env.py:104`), `NewtonCfg(num_substeps=2, use_cuda_graph=True, collision_cfg=
  NewtonCollisionPipelineCfg(soft_contact_max=0))` around `CouplerProxyCfg` with two entries and one proxy mapping.
* `use_newton_actuators=True` does nothing here: every Franka actuator group is an `ImplicitActuatorCfg`, so
  `prepare_native_actuators` returns before activating Newton actuators (`IL/isaaclab_newton/.../assets/articulation/
  actuator_control.py:54-62`); the env runs the decimation loop itself (`manager_based_rl_env.py:229-244`): per
  physics step `apply_action` -> `write_data_to_sim` -> `sim.step` -> `scene.update`.
* `NewtonManager.step` (`IL/isaaclab_newton/.../physics/newton_manager.py:977-1085`): solver-internal reset and FK for
  dirty worlds only, then the captured `_simulate_physics_only`: **one** `CollisionPipeline.collide` per sim step
  (`:2630-2632`, `collision_decimation=0`), then `num_substeps=2` x `SolverCoupledProxy.step(dt=1/240)` with state
  swaps and `clear_forces` (`:2565-2577`).
* `SolverCoupledProxy.step` (`NW/solvers/coupled/solver_coupled.py:2207-2226`, `solver_coupled_proxy.py:1369-1569`):
  distribute parent state to entries; the `arm` entry (MuJoCo Warp, 3 substeps of 1/720) gets the previous step's
  harvested MPM wrench as `body_f`; staggered sync writes the arm's **end** cup poses and velocities into the media
  view; the `media` entry (implicit MPM, in place, 2 substeps of 1/480) steps; `coupling_harvest_proxy_wrenches`
  (`NW/solvers/implicit_mpm/solver_implicit_mpm.py:1773-1812`) converts collider impulses to proxy wrenches for the
  next step. Per policy step: 4 collides, 8 coupled steps, 24 MuJoCo solves, 16 MPM solves.
* Proxy mass: MuJoCo's effective mass of each cup x `mass_scale` 1000 in the MPM view (`_apply_proxy_body_effective_masses`,
  `solver_coupled_proxy.py:1262-1285`).
* Inferred from the code (not run): the harvest reads only the last MPM substep's `impulse_field` (overwritten per
  substep, `solver_implicit_mpm.py:3266-3268`) and divides by the outer dt (`solver_coupled_proxy.py:1555`), so with
  `substeps=2` the wrench fed back to the cups is about half the average force. The port runs the same Newton code,
  so it has the same behaviour; not a port difference.

## 2. Model (what Isaac Lab hands Newton)

* `NewtonShapeCfg` defaults applied to every builder (`newton_manager_cfg.py:89-120`, `newton_manager.py:1217-1249`):
  margin 0, gap 0.01, ke 2.5e3, kd 100, mu 1.0. Schema resolvers Newton, PhysX, MuJoCo (`newton_manager.py:1921-1933`).
* `MJWarpSolverCfg` defaults reaching `SolverMuJoCo` (`mjwarp_manager_cfg.py`, filtered by `_filter_solver_kwargs`):
  iterations 100, ls_iterations 50, solver newton, cone pyramidal, impratio 1, tolerance 1e-6, ccd_iterations 35,
  update_data_interval 1; task: `use_mujoco_contacts=False`, implicitfast, njmax 2560, nconmax 400.
* `NewtonCollisionPipelineCfg` defaults: broad phase explicit, reduce_contacts, `max_triangle_pairs` 1,000,000,
  soft_contact_margin 0.01 (`newton_collision_cfg.py:97-172`). This scene needs more than 1M triangle pairs from
  ~1,250 worlds (newton-physics/newton#4604, fixed upstream by #4605); the port allocates 4,096 per world.
* Actuators: `FRANKA_PANDA_CFG.actuators` (`IL/isaaclab_assets/.../robots/franka.py:51-73`) with the task's
  `replace(effort_limit_sim=None, velocity_limit_sim=None, stiffness=None, damping=None, armature=None)`
  (`pour_env_cfg.py:344-353`). `effort_limit_sim` is a deprecated alias of `joint_effort_limit`
  (`IL/isaaclab/isaaclab/actuators/actuator_base_cfg.py:73-93`), so `joint_effort_limit` 87 / 12 / 200 stays in force;
  stiffness / damping None resolve to the USD drives (`actuator_collection.py:401-449`): arm 1000/750/300 and
  20/4/2 (set again by the task), finger 350 N/m, 10 N s/m on `panda_finger_joint1`, none on the mimic follower;
  armature from the USD (`physxJoint:armature = 0.1`, physx variant). Newton -> MuJoCo: position actuators
  `gainprm=[kp]`, `biasprm=[0,-kp,-kd]`, `actfrcrange = +-effort` on the joint (`NW/solvers/mujoco/solver_mujoco.py:
  6586-6603, 6560-6563`).
* `MujocoJointCfg(actuatorgravcomp=True)` (`pour_env_cfg.py:364`): `mjc:actuatorgravcomp` on every revolute /
  prismatic joint and `mjc:gravcomp = 1` on each one's child body (`IL/isaaclab_newton/.../sim/schemas/schemas.py:123-161`,
  `IL/isaaclab/isaaclab/sim/schemas/schemas.py:1388-1450`), so gravity compensation goes through the actuators and is
  clamped by `actfrcrange` together with the PD force.
* State read by the terms: `joint_pos` / `joint_vel` = `state_0.joint_q` / `joint_qd`; `body_link_pose_w` =
  `state_0.body_q` from Newton's own FK after each MuJoCo step (`solver_mujoco.py:4869-4931`); Newton's `body_qd` is
  the COM twist (v_com, omega) in world (`NW/sim/state.py:131-135`); `root_link_vel_w` shifts it to the link origin
  (`IL/isaaclab_newton/.../assets/kernels.py:39-61`).

## 3. Assets

Isaac 6.1 asset root (`apps/isaaclab.python.headless.kit:206`), fetched by `scripts/fetch_isaac_assets.sh`:
`Robots/FrankaEmika/franka_panda.usda` (+ payloads; "MuJoCo USD Converter v0.2.0" of Menagerie's `mjx_panda`),
`Contrib/MPM/Pour/franka_pour_cups.usda`, `Contrib/MPM/Pour/reset_dataset.pt` (20,000 rows, schema 12, content
digest `74057f08...`), `Props/Mounts/SeattleLabTable/table_instanceable.usd` + `table.usd`. The 6.2 root serves
byte-identical Franka and cup files.

**The task's spawn hook does not accept the published Franka.** `spawn_franka_with_arm_collisions`
(`pour_env_cfg.py:178-230`) requires the arm collision proxies `link0_c` .. `link7_c` (only under the asset's
`Colliders = convex_hulls` variant; the default is `gripper_only`, and the task selects no variant) and a
`PhysxMimicJointAPI:linear` on `panda_finger_joint2` (absent from every layer; only `NewtonMimicAPI` on
`panda_finger_joint1` exists). Its pinned physics payload hash (`0dc38454...`) does match the published
`payloads/Physics/physics.usda`. The port composes what the hook intends (`franka_pour_scene.compose_robot_stage`).

Spawner behaviour that matters: `create_prim` + `standardize_xform_ops` (`IL/isaaclab/isaaclab/sim/utils/prims.py`,
`transforms.py`) override the referenced root's own transform with the init-state pose, so the SeattleLab table's
authored `xformOp:translate = (0.55, 0, 0)` is replaced by (0.5, 0, 0) and the tabletop spans x in [-0.296, 0.984],
y in [-0.455, 0.455] at z = -0.003 (0 after the task's 3 mm inset).

## 4. MDP

Observations (`mdp/observations.py`, groups in `pour_env_cfg.py:396-464`; scale after clip,
`observation_manager.py:395-410`), actions (`mdp/actions.py`, `binary_joint_actions.py:132-150`,
`joint_actions.py:170-233`), rewards (`mdp/rewards.py`; reward manager multiplies by weight x dt,
`reward_manager.py:146-152`), terminations (`mdp/terminations.py`; evaluated in cfg order, the success term sees the
partial `terminated` buffer, `termination_manager.py:154-181`), reset order (`manager_based_rl_env.py:406-451`:
curriculum, scene reset, reset events, then observation / action / reward / curriculum / command / event /
termination managers; the action manager resets after the reset event, so the gripper's `set_reset_position`
survives and the arm's EMA history is cleared), the curriculum and sampler (`mdp/reset_dataset.py`,
`reset_sampler.py`). `scene.reset` writes each env's default particle state (`IL/isaaclab_newton/.../assets/
mpm_object/mpm_object.py:136-141`), which the reset event then overwrites (positions and velocities), so it has no
observable effect. The rsl_rl wrapper clips actions to [-1, 1] and passes `time_outs` (`vecenv_wrapper.py:178-187`).

## 5. rsl_rl 5.4.1

`algorithms/ppo.py`: time-out bootstrap `r += gamma * V(s_t) * time_out` (:150-155), GAE (:163-188) with
whole-batch advantage normalization, one permutation reused across epochs (`storage/rollout_storage.py:225-258`),
clipped surrogate and value losses, `clip_grad_norm_` separately on actor and critic (:302-303), fixed schedule.
`modules/distribution.py:241-323` `HeteroscedasticGaussianDistribution`: MLP output `(2, A)`, log-std clamped (hard)
to `log(std_range)`, std head initialised with zero weights and bias `log(init_std + 1e-7)`. `modules/mlp.py`:
PyTorch default init. `utils/logger.py`: mean reward / length over the last 100 episodes, `extras["log"]` averaged
per iteration.

No training curve is published for this task (README, task source, Isaac Lab docs at this commit); the README gives
the launch command (`--num_envs 2048 --max_iterations 3000`).
