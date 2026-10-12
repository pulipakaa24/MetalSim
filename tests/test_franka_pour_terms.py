"""Franka-Pour task terms against Isaac Lab 3.0-EA's own code (``tests/franka_pour_isaac_ref.py``: Isaac's term
functions, executed unmodified on this environment's live state), plus the scene settings against ``pour_env_cfg.py``,
the robot model against Isaac's USD, and a smoke run.

Needs Metal, Newton 1.5.2 with the metalsim forks, PyTorch (MPS) and the Isaac Lab 3 checkout (``upstream/IsaacLab3``)
and the fetched Franka-Pour assets (``scripts/fetch_isaac_assets.sh``):
    scripts/gpu_run.sh test_pour check 20 -- .venv-newton152/bin/python -m pytest tests/test_franka_pour_terms.py
"""
from __future__ import annotations

import math
import os
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")
newton = pytest.importorskip("newton")

import franka_pour_isaac_ref as ref  # noqa: E402  (tests/ is on sys.path under pytest)

pytestmark = [
    pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal"),
    pytest.mark.skipif(not newton.__version__.startswith("1.5.2"), reason="needs Newton 1.5.2 (Isaac Lab 3.0-EA's pin)"),
    pytest.mark.skipif(not ref.AVAILABLE, reason="needs upstream/IsaacLab3 (Isaac's task source is the oracle)"),
]

from metalsim.learn import franka_pour as P  # noqa: E402
from metalsim.learn import franka_pour_scene as S  # noqa: E402

pytestmark.append(pytest.mark.skipif(not (os.path.isfile(S.FRANKA_USD) and os.path.isfile(S.RESET_DATASET)),
                                     reason="needs the Franka-Pour assets (scripts/fetch_isaac_assets.sh)"))

N = 8
TOL = dict(rtol=0.0, atol=0.0)          # same formulas on the same tensors: bitwise
LOOSE = dict(rtol=1e-6, atol=1e-6)      # Isaac's own math vs the vendored copy compiled separately (torch.jit)


@pytest.fixture(scope="module")
def env():
    wp.config.quiet = True
    e = P.FrankaPourEnv(P.PourTaskConfig(num_envs=N, seed=3))
    e.reset_all()
    g = torch.Generator(device=e.device).manual_seed(5)
    for _ in range(4):     # leave the reset state: drive with random actions
        e.step(torch.rand(N, 8, device=e.device, generator=g) * 2 - 1)
    return e


def _isaac_obs_group(view, group):
    r = ref.isaac()
    out = []
    for name, kw in ref.task_cfg()[group]:
        fn = r["generic"].get(kw["func"]) or getattr(r["obs"], kw["func"])
        params = dict(kw.get("params", {}))
        if "params_src" in kw:       # SceneEntityCfg("robot", joint_names=["panda_joint.*"]): resolve as Isaac does
            pat = re.search(r'joint_names=\["([^"]+)"\]', kw["params_src"]).group(1)
            names = list(S.ARM_JOINTS + S.FINGER_JOINTS)
            ids = [i for i, nm in enumerate(names) if re.fullmatch(pat, nm)]
            params["asset_cfg"] = type("Cfg", (), {"name": "robot", "joint_ids": ids})()
        v = fn(view, **params).clone()
        if "clip" in kw:
            v = v.clip_(kw["clip"][0], kw["clip"][1])
        if "scale" in kw:
            v = v.mul_(kw["scale"])
        out.append((name, v))
    return out


def test_observation_terms_match_isaac(env):
    """Every observation term of the three groups, in Isaac's order with Isaac's scale and clip, equals Isaac's term
    function evaluated on the same state (84 actor + 2 privileged inputs)."""
    view = ref.IsaacView(env)
    mine = env.observations()
    for group in ("policy", "media", "privileged"):
        terms = _isaac_obs_group(view, group)
        isaac = torch.cat([v for _, v in terms], dim=-1)
        assert mine[group].shape == isaac.shape, (group, mine[group].shape, isaac.shape)
        col = 0
        for name, v in terms:
            w = v.shape[-1]
            torch.testing.assert_close(mine[group][:, col:col + w], v, **LOOSE, msg=f"{group}/{name}")
            col += w
    assert mine["policy"].shape[-1] + mine["media"].shape[-1] == 84


def test_terminations_and_rewards_match_isaac(env):
    """One policy step: Isaac's termination terms (in cfg order, with their side effects on the dwell counters,
    the success latch and the grasp latch) and reward terms x weight equal the environment's, term by term."""
    r = ref.isaac()
    cfg = ref.task_cfg()
    g = torch.Generator(device=env.device).manual_seed(11)
    for _ in range(3):
        env.process_action(torch.rand(N, 8, device=env.device, generator=g) * 2 - 1)
        env.simulate()
        env.episode_length_buf += 1
        env.common_step_counter += 1
        env._invalidate_caches()
        view = ref.IsaacView(env)
        tm = view.termination_manager
        tm.active_terms = [n for n, _ in cfg["terminations"]]
        progress = r["rew"].PourResetLearningProgress(ref.TermCfg(params=cfg["terminations"][7][1]["params"]), view)
        for k in ("_target_potential", "_requires_terminal_success", "ever_success"):
            setattr(progress, k, getattr(env.progress, k).clone())
        isaac_terms = {}
        for name, kw in cfg["terminations"]:
            if name == "learning_progress_context":
                fn = progress
            else:
                fn = getattr(r["term"], kw["func"])
            v = fn(view, **kw.get("params", {}))
            isaac_terms[name] = v
            tm.terms[name] = v
            if kw.get("time_out"):
                tm.truncated |= v
            else:
                tm.terminated |= v
        isaac_rew = {}
        for name, kw in cfg["rewards"]:
            fn = r["generic"].get(kw["func"]) or getattr(r["rew"], kw["func"])
            isaac_rew[name] = fn(view, **kw.get("params", {})) * kw["weight"]
        dones = env._compute_terminations()
        env._compute_rewards()
        assert [n for n, _ in cfg["terminations"]] == list(env.termination_terms)
        for name in env.termination_terms:
            assert torch.equal(env._term_dones[name], isaac_terms[name]), name
        assert torch.equal(env.terminated, tm.terminated) and torch.equal(env.time_outs, tm.truncated)
        for k in ("_success_dwell_count", "_lost_grasp_dwell_count", "_lifted_grasp_seen", "episode_succeeded"):
            assert torch.equal(getattr(env, k), getattr(view, k)), k
        assert torch.equal(env.progress.ever_success, progress.ever_success)
        assert [n for n, _ in cfg["rewards"]] == list(env.reward_terms)
        for j, name in enumerate(env.reward_terms):
            torch.testing.assert_close(env._step_reward[:, j], isaac_rew[name], **LOOSE, msg=name)
        # finish the step as the environment does (resets, observations)
        ids = dones.nonzero().flatten()
        if ids.numel():
            env._reset_idx(ids)
        env.obs_buf = env.observations()


def test_reset_from_dataset_matches_isaac(env):
    """``_reset_from_dataset``: joint state, cup poses and the particle lattice placed in the row's source-cup frame
    (Isaac's formula, its math) after the reset, and the learning-progress baseline of Isaac's term."""
    r = ref.isaac()
    mu = r["math"]
    ids = torch.arange(N, device=env.device)
    env._reset_idx(ids)
    st = env._reset_dataset_states
    rows = env.reset_dataset_row_id
    assert torch.equal(env.joint_pos()[:, :7], st["arm_joint_position"][rows])
    assert torch.equal(env.joint_pos()[:, 7:], st["finger_joint_position"][rows])
    src = st["source_root_pose"][rows].clone()
    src[:, :3] += env.env_origins
    torch.testing.assert_close(env.source_pose_w(), src, rtol=0, atol=2e-6)       # after FK of the free joint
    q = src[:, None, 3:7].expand(-1, env._num_particles, -1)
    expect = mu.quat_apply(q, env._reset_particle_local_position[None].expand(N, -1, -1)) + src[:, None, :3]
    torch.testing.assert_close(env._pq, expect, **LOOSE)
    view = ref.IsaacView(env)
    cfg = ref.task_cfg()
    progress = r["rew"].PourResetLearningProgress(ref.TermCfg(params=cfg["terminations"][7][1]["params"]), view)
    progress.reset(ids)
    torch.testing.assert_close(env.progress._target_potential, progress._target_potential, **LOOSE)
    assert torch.equal(env.progress._requires_terminal_success, progress._requires_terminal_success)
    assert torch.equal(env._lifted_grasp_seen, st["category"][rows] == 1)
    env.obs_buf = env.observations()


def test_sampler_matches_isaac():
    """``_ResetDatasetSampler``: ring buffers, rolling rates, Beta-kernel probabilities, metrics and the sampled rows
    (same torch generator, CPU) equal Isaac's."""
    r = ref.isaac()
    cfg = P.SamplerConfig()
    icfg = r["sampler"].ResetDatasetSamplerCfg(monitored_history_len=50, target_success_rate=0.5, kappa=1.0,
                                               epsilon=1e-4, uniform_fraction=0.5)
    rows = 500
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    mine = P.ResetDatasetSampler(rows, "cpu", cfg, g1)
    isaac = r["sampler"]._ResetDatasetSampler(rows, "cpu", icfg)
    isaac._uniform_order = mine._uniform_order.clone()
    rng = np.random.default_rng(0)
    for k in range(60):
        ids = torch.as_tensor(rng.integers(0, 40 if k % 3 else rows, 64))   # many duplicates every third batch
        vals = torch.as_tensor(rng.random(64) < 0.4)
        mine.record(ids, vals)
        isaac._record_validated(ids, vals)
        a = mine.sample_with_uniform_replay(37)
        b = isaac._sample_with_uniform_replay(37, generator=g2)
        if isaac._uniform_cursor == 0:     # Isaac reshuffles with the passed generator; keep the orders aligned
            isaac._uniform_order = mine._uniform_order.clone()
        assert torch.equal(a, b)
    for k in ("_success_rates", "_success_history", "_history_pointers", "_history_sizes", "_history_success_counts"):
        assert torch.equal(getattr(mine, k), getattr(isaac, k)), k
    assert mine.metrics() == isaac.metrics()


def test_scene_settings_match_isaac_cfg():
    """Constants of ``pour_env_cfg.py`` / ``geometry.py`` / ``media.py`` used by the scene and the task."""
    f = ref.env_cfg_fields()
    assert tuple(f["_ARM_HOME"]) == S.ARM_HOME and f["_GRIPPER_OPEN_POSITION"] == S.GRIPPER_OPEN_POSITION
    assert tuple(f["_SOURCE_CUP_POSITION"]) == S.SOURCE_CUP_POSITION
    assert tuple(f["_TARGET_CUP_POSITION"]) == S.TARGET_CUP_POSITION
    assert f["_SOURCE_CUP_MASS"] == S.SOURCE_CUP_MASS and f["_SOURCE_CUP_GRASP_FRICTION"] == S.SOURCE_CUP_GRASP_FRICTION
    assert f["_TARGET_CUP_FRICTION"] == S.TARGET_CUP_FRICTION
    assert f["_MEDIA_FILL_LEVEL"] == S.MEDIA_FILL_LEVEL and f["_MPM_VOXEL_SIZE"] == S.MPM_VOXEL_SIZE
    assert f["_MPM_PARTICLES_PER_CELL"] == S.MPM_PARTICLES_PER_CELL
    stiff = dict(f["FRANKA_POUR_ARM_DRIVE_STIFFNESS"]) if "FRANKA_POUR_ARM_DRIVE_STIFFNESS" in f else None
    if stiff is None:   # MappingProxyType({...}): read the dict literal
        src = open(os.path.join(ref.TASK, "pour_env_cfg.py")).read()
        stiff = eval(re.search(r"FRANKA_POUR_ARM_DRIVE_STIFFNESS = MappingProxyType\(\s*(\{.*?\})\s*\)", src, re.S)
                     .group(1))
        damp = eval(re.search(r"FRANKA_POUR_ARM_DRIVE_DAMPING = MappingProxyType\(\s*(\{.*?\})\s*\)", src, re.S)
                    .group(1))
    groups = {"panda_joint[1-2]": (0, 1), "panda_joint[3-4]": (2, 3), "panda_joint[5-7]": (4, 5, 6)}
    for key, js in groups.items():
        assert all(S.ARM_STIFFNESS[j] == stiff[key] and S.ARM_DAMPING[j] == damp[key] for j in js), key
    for k in ("tcp_offset_pos", "pour_target_frac", "particle_count_margin", "spill_table_height",
              "max_spill_fraction", "success_dwell_time_s", "success_min_lift_height",
              "state_bound_joint_position_margin", "state_bound_max_joint_velocity",
              "state_bound_max_cup_linear_velocity", "state_bound_max_cup_angular_velocity",
              "particle_workspace_lower_bound", "particle_workspace_upper_bound"):
        assert f[k] == getattr(P.PourTaskConfig(), k), k
    assert f["grasp_contact_ke"] == S.GRASP_CONTACT_KE and f["grasp_contact_kd"] == S.GRASP_CONTACT_KD
    assert f["grasp_contact_kf"] == S.GRASP_CONTACT_KF and f["collider_margin"] == S.COLLIDER_MARGIN
    assert f["particle_max_velocity"] == S.PARTICLE_MAX_VELOCITY
    assert f["mpm_cell_capacity_alignment"] == S.MPM_CELL_CAPACITY_ALIGNMENT
    g = ref.isaac()["geometry"]
    for name in ("SOURCE_CUP_GEOMETRY", "TARGET_CUP_GEOMETRY"):
        a, b = getattr(g, name), getattr(S, name)
        assert (a.inner_width, a.inner_depth, a.cavity_depth, a.wall_thickness, a.bottom_thickness) == \
            (b.inner_width, b.inner_depth, b.cavity_depth, b.wall_thickness, b.bottom_thickness)
    assert g.CUP_GRASP_HEIGHT == S.CUP_GRASP_HEIGHT and g.MPM_COLLIDER_MARGIN == S.MPM_COLLIDER_MARGIN
    assert tuple(g.CUP_GRASP_TCP_QUAT_C) == S.CUP_GRASP_TCP_QUAT_C
    assert S.media_particle_count() == 735 and tuple(S.media_lattice()[1]) == (7, 7, 15)
    src = open(os.path.join(ref.TASK, "pour_env_cfg.py")).read()
    for frag in ('use_mujoco_contacts=False', 'integrator="implicitfast"', "njmax=2560", "nconmax=400",
                 "substeps=3", "voxel_size=_MPM_VOXEL_SIZE", 'grid_type="sparse"', "grid_padding=0",
                 'strain_basis="P0"', 'transfer_scheme="apic"', "max_iterations=24", "tolerance=1.0e-4",
                 'warmstart_mode="auto"', 'velocity_basis="Q1"', 'collider_basis="pic27"',
                 'collider_velocity_mode="forward"', 'solver="jacobi"', "separate_worlds=True", "substeps=2",
                 "in_place=True", "mass_scale=1000.0", 'mode="staggered"', "collision_pipeline=None",
                 "iterations=1", "soft_contact_max=0", "num_substeps=2", "self.decimation = 4",
                 "self.episode_length_s = 12.0", "self.sim.dt = 1.0 / 120.0"):
        assert frag in src, frag
    m = S.mpm_config(2048)
    assert (m.max_iterations, m.tolerance, m.solver, m.voxel_size, m.strain_basis, m.collider_basis,
            m.max_active_cell_count) == (24, 1e-4, "jacobi", 0.015, "P0", "pic27", 1024 * 2048)
    assert (S.SIM_DT, S.DECIMATION, S.NUM_SUBSTEPS, S.ARM_SUBSTEPS, S.MEDIA_SUBSTEPS) == (1 / 120, 4, 2, 3, 2)
    assert P.MAX_EPISODE_LENGTH == 360


def test_robot_model_matches_isaac_asset(env):
    """The Newton model built from Isaac's USD: joint limits, armature (physx variant 0.1), drives as Isaac's actuator
    overrides resolve them, effort limits of FRANKA_PANDA_CFG, actuator gravity compensation, the finger mimic, and
    the collision set (ten arm hull proxies, hand capsule, finger pads; no hand hull)."""
    m, info = env.physics.model, env.info
    dofs = info.arm_dofs + info.finger_dofs
    lo, hi = m.joint_limit_lower.numpy()[dofs], m.joint_limit_upper.numpy()[dofs]
    # physics.usda limits (mjx_panda's ranges)
    np.testing.assert_allclose(lo[:7], [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], atol=1e-4)
    np.testing.assert_allclose(hi[:7], [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], atol=1e-4)
    np.testing.assert_allclose(lo[7:], 0.0)
    np.testing.assert_allclose(hi[7:], 0.04, atol=1e-7)
    np.testing.assert_allclose(m.joint_armature.numpy()[dofs], 0.1, rtol=1e-6)
    np.testing.assert_allclose(m.joint_target_ke.numpy()[dofs], list(S.ARM_STIFFNESS) + [350.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(m.joint_target_kd.numpy()[dofs], list(S.ARM_DAMPING) + [10.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(m.joint_effort_limit.numpy()[dofs], list(S.ARM_EFFORT_LIMIT) + [200.0, 200.0])
    assert m.mujoco.jnt_actgravcomp.numpy()[dofs].all()
    labels = [str(x) for x in m.shape_label]
    flags = m.shape_flags.numpy()
    sw = m.shape_world.numpy()
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    robot = [s for s in range(m.shape_count) if sw[s] == 0 and "/Robot/" in labels[s] and flags[s] & collide]
    names = sorted(lab.rsplit("/", 1)[-1] for lab in (labels[s] for s in robot))
    assert names == sorted(list(S.ARM_COLLISION_PROXIES) + ["hand_capsule", "left_finger_pad", "right_finger_pad"])
    assert int(m.constraint_mimic_joint0.numpy().size) == N


def test_smoke_episode_runs_finite(env):
    """30 policy steps with random actions: finite observations and rewards, the episode counter advancing, and
    resets drawing new dataset rows."""
    g = torch.Generator(device=env.device).manual_seed(7)
    rows0 = env.reset_dataset_row_id.clone()
    for _ in range(30):
        obs, rew, term, tout, extras = env.step(torch.rand(N, 8, device=env.device, generator=g) * 2 - 1)
        assert all(bool(torch.isfinite(v).all()) for v in obs.values())
        assert bool(torch.isfinite(rew).all())
    assert env.common_step_counter > 30
    assert obs["policy"].shape == (N, 62) and obs["media"].shape == (N, 22) and obs["privileged"].shape == (N, 2)
    assert math.isclose(env.step_dt, 1 / 30)
    del rows0
