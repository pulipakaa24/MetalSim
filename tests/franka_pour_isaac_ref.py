"""Isaac Lab 3.0-EA's own Franka-Pour term code, loaded from the upstream checkout (``upstream/IsaacLab3``) without
Isaac Sim, as the oracle for ``metalsim.learn.franka_pour``.

Executed unmodified: ``source/isaaclab/isaaclab/utils/math.py``, the task's ``geometry.py``, ``mdp/observations.py``,
``mdp/rewards.py``, ``mdp/terminations.py``, ``reset_sampler.py``, the ``FrankaPourEnv`` accessor methods of
``pour_env.py`` (extracted with ``ast``: ``_pose_w_to_e`` .. ``_particle_region_masks``) and the generic Isaac Lab
terms the task uses (``joint_pos_rel``, ``joint_vel_rel``, ``last_action``, ``action_l2``, ``action_rate_l2`` from
``source/isaaclab/isaaclab/envs/mdp``). The term configuration (which functions, in which order, with which scale,
clip, weight and parameters) is read from ``pour_env_cfg.py`` with ``ast``. Only the Isaac Lab plumbing those files
import is stubbed (``configclass``, ``ManagerTermBase``, ``TerminationTermCfg``).

``IsaacView(env)`` presents a ``metalsim.learn.franka_pour.FrankaPourEnv`` through the attribute paths Isaac's code
reads (``env._robot.data.joint_pos.torch``, ``env._source_cup.data.root_link_vel_w.torch``,
``env.termination_manager.terminated``, ...), backed by the environment's live state; mutable task buffers are
copies, so Isaac's side effects never touch the environment under test.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import math
import os
import sys
import types

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IL = os.environ.get("ISAACLAB3_SOURCE", os.path.join(os.path.dirname(ROOT) if ".claude" in ROOT else ROOT,
                                                      "upstream", "IsaacLab3", "source"))
if not os.path.isdir(IL):   # worktrees live in <repo>/.claude/worktrees/<name>: upstream/ is the main checkout's
    IL = os.path.join(ROOT.split("/.claude/")[0], "upstream", "IsaacLab3", "source")
TASK = os.path.join(IL, "isaaclab_tasks", "isaaclab_tasks", "contrib", "franka_pour")
AVAILABLE = os.path.isfile(os.path.join(TASK, "pour_env_cfg.py"))


def _load(name, path, package=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if package is not None:
        mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env


@dataclasses.dataclass
class TermCfg:
    func: object = None
    params: dict = dataclasses.field(default_factory=dict)
    time_out: bool = False


def _configclass(cls):
    return dataclasses.dataclass(cls)


_CACHE = {}


def isaac():
    """The loaded Isaac modules (cached): ``math``, ``geometry``, ``obs``, ``rew``, ``term``, ``sampler``, ``generic``
    (Isaac Lab's generic mdp terms) and ``env_methods`` (the pour_env.py accessor functions)."""
    if _CACHE:
        return _CACHE
    saved = {k: sys.modules.get(k) for k in ("isaaclab", "isaaclab.utils", "isaaclab.utils.math", "isaaclab.managers")}
    math_mod = _load("ilref_math", os.path.join(IL, "isaaclab", "isaaclab", "utils", "math.py"))
    il = types.ModuleType("isaaclab")
    il_utils = types.ModuleType("isaaclab.utils")
    il_utils.math = math_mod
    il_utils.configclass = _configclass
    il_mgr = types.ModuleType("isaaclab.managers")
    il_mgr.ManagerTermBase = _ManagerTermBase
    il_mgr.TerminationTermCfg = TermCfg
    il_mgr.CurriculumTermCfg = TermCfg
    il.utils = il_utils
    sys.modules.update({"isaaclab": il, "isaaclab.utils": il_utils, "isaaclab.utils.math": math_mod,
                        "isaaclab.managers": il_mgr})
    try:
        for name in ("ilref", "ilref.mdp"):
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg
        geometry = _load("ilref.geometry", os.path.join(TASK, "geometry.py"), "ilref")
        obs = _load("ilref.mdp.observations", os.path.join(TASK, "mdp", "observations.py"), "ilref.mdp")
        rew = _load("ilref.mdp.rewards", os.path.join(TASK, "mdp", "rewards.py"), "ilref.mdp")
        term = _load("ilref.mdp.terminations", os.path.join(TASK, "mdp", "terminations.py"), "ilref.mdp")
        sampler = _load("ilref.reset_sampler", os.path.join(TASK, "reset_sampler.py"), "ilref")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    generic = {}
    for fname, names in ((os.path.join(IL, "isaaclab", "isaaclab", "envs", "mdp", "observations.py"),
                          ("joint_pos_rel", "joint_vel_rel", "last_action")),
                         (os.path.join(IL, "isaaclab", "isaaclab", "envs", "mdp", "rewards.py"),
                          ("action_l2", "action_rate_l2"))):
        # default-argument stand-in: the test passes the resolved asset_cfg explicitly
        stub_entity = lambda name, **kw: types.SimpleNamespace(name=name, joint_ids=slice(None), **kw)  # noqa: E731
        generic.update(_extract_functions(fname, names, {"torch": torch, "SceneEntityCfg": stub_entity}))
    env_methods = _extract_methods(os.path.join(TASK, "pour_env.py"), "FrankaPourEnv",
                                   ("_pose_w_to_e", "cup_pose_e", "target_pose_e", "tcp_pose_e", "cup_grasp_point_e",
                                    "gripper_width", "particle_pos_e", "_particle_region_masks"),
                                   {"torch": torch, "math_utils": math_mod, "points_inside_box": geometry.points_inside_box,
                                    "MPM_COLLIDER_MARGIN": geometry.MPM_COLLIDER_MARGIN,
                                    "CUP_GRASP_HEIGHT": geometry.CUP_GRASP_HEIGHT})
    _CACHE.update(math=math_mod, geometry=geometry, obs=obs, rew=rew, term=term, sampler=sampler, generic=generic,
                  env_methods=env_methods)
    return _CACHE


def _extract_functions(path, names, ns):
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            for a in node.args.args + node.args.kwonlyargs:   # drop annotations that name Isaac classes
                a.annotation = None
            node.returns = None
            code = compile(ast.Module(body=[node], type_ignores=[]), path, "exec")
            g = dict(ns)
            exec(code, g)
            out[node.name] = g[node.name]
    missing = set(names) - set(out)
    if missing:
        raise RuntimeError(f"{missing} not found in {path}")
    return out


def _extract_methods(path, cls_name, names, ns):
    tree = ast.parse(open(path).read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for n in nodes:
        n.decorator_list = []
        n.returns = None
        for a in n.args.args:
            a.annotation = None
    g = dict(ns)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), g)
    return {n: g[n] for n in names}


# ------------------------------------------------------------------------------------------------ cfg parsing
def _value(node, env):
    """Evaluate a cfg expression: literals, math.*, unary minus, arithmetic, dict/tuple/list."""
    return eval(compile(ast.Expression(body=node), "<cfg>", "eval"), {"math": math, "__builtins__": {}}, env)


def _calls_in_class(tree, cls_name):
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name)
    out = []
    for st in cls.body:
        if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call):
            call = st.value
            kw = {}
            for k in call.keywords:
                if k.arg == "func":
                    v = k.value
                    kw["func"] = v.attr if isinstance(v, ast.Attribute) else v.id
                elif k.arg in ("scale", "weight", "clip", "time_out"):
                    kw[k.arg] = _value(k.value, {})
                elif k.arg == "params":
                    try:
                        kw["params"] = _value(k.value, {})
                    except Exception:  # noqa: BLE001  (e.g. SceneEntityCfg(...)): kept as source text
                        kw["params_src"] = ast.unparse(k.value)
            out.append((st.targets[0].id, kw))
    return out


def task_cfg():
    """Observation groups, reward and termination terms, in declaration order, from ``pour_env_cfg.py``."""
    tree = ast.parse(open(os.path.join(TASK, "pour_env_cfg.py")).read())
    return {"policy": _calls_in_class(tree, "PolicyCfg"), "media": _calls_in_class(tree, "MediaCfg"),
            "privileged": _calls_in_class(tree, "PrivilegedCfg"),
            "rewards": _calls_in_class(tree, "ResetDatasetRewardsCfg"),
            "terminations": _calls_in_class(tree, "TerminationsCfg")}


def env_cfg_fields():
    """``FrankaPourResetDatasetEnvCfg``'s literal-valued fields and the module constants of ``pour_env_cfg.py``."""
    tree = ast.parse(open(os.path.join(TASK, "pour_env_cfg.py")).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = _value(node.value, {})
            except Exception:  # noqa: BLE001
                pass
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "FrankaPourResetDatasetEnvCfg")
    for st in cls.body:
        if isinstance(st, ast.AnnAssign) and st.value is not None:
            try:
                out[st.target.id] = _value(st.value, out)
            except Exception:  # noqa: BLE001
                pass
    return out


# ------------------------------------------------------------------------------------------------ the view
class _P:
    def __init__(self, fn):
        self._fn = fn

    @property
    def torch(self):
        return self._fn()


class _TerminationManager:
    def __init__(self, n, device):
        self.terminated = torch.zeros(n, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(n, dtype=torch.bool, device=device)
        self.terms = {}
        self.active_terms = []

    @property
    def dones(self):
        return self.terminated | self.truncated

    def get_term(self, name):
        return self.terms[name]


class IsaacView:
    """``FrankaPourEnv`` as Isaac's term code sees it, over a metalsim env's live state."""

    def __init__(self, env):
        ref = isaac()
        self._env = env
        n, dev = env.num_envs, env.device
        self.num_envs, self.device = n, dev
        self.step_dt = env.step_dt
        self.max_episode_length = env.max_episode_length
        self.common_step_counter = env.common_step_counter
        self.env_origins = env.env_origins
        bq = lambda: env._bq  # noqa: E731
        fields = env_cfg_fields()
        # PourSceneCfg.robot.init_state.joint_pos (pour_env_cfg.py:365-366): Isaac's default_joint_pos
        default = torch.tensor(list(fields["_ARM_HOME"]) + [fields["_GRIPPER_OPEN_POSITION"]] * 2,
                               device=dev).repeat(n, 1)
        self._robot = types.SimpleNamespace(data=types.SimpleNamespace(
            joint_pos=_P(env.joint_pos), joint_vel=_P(env.joint_vel), body_link_pose_w=_P(bq),
            root_link_pose_w=_P(env.robot_root_pose_w), default_joint_pos=_P(lambda: default),
            default_joint_vel=_P(lambda: torch.zeros_like(default))))
        self.scene = {"robot": self._robot}
        self._source_cup = types.SimpleNamespace(data=types.SimpleNamespace(
            root_link_pose_w=_P(env.source_pose_w), root_link_vel_w=_P(env.source_vel_w)))
        self._target_cup = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pose_w=_P(env.target_pose_w)))
        self._media = types.SimpleNamespace(data=types.SimpleNamespace(particle_pos_w=_P(lambda: env._pq),
                                                                       particle_vel_w=_P(lambda: env._pqd)))
        self._tcp_body_idx = env._tcp_body_idx
        self._finger_joint_ids = env._finger_joint_ids
        for k in ("_tcp_offset_pos", "_tcp_offset_quat", "_desired_grasp_tcp_quat_c", "_gripper_open_width",
                  "_gripper_grasp_width", "_cup_reset_height", "_source_outer_half_t", "_target_outer_half_t",
                  "_particle_workspace_lower_t", "_particle_workspace_upper_t", "_num_particles",
                  "_joint_pos_limits_t", "_source_inner_lo_t", "_source_inner_hi_t", "_target_inner_lo_t",
                  "_target_inner_hi_t", "_reset_dataset_states"):
            setattr(self, k, getattr(env, k))
        # mutable task buffers: copies
        for k in ("_success_dwell_count", "_lost_grasp_dwell_count", "_lifted_grasp_seen", "episode_succeeded",
                  "pour_target_frac", "reset_dataset_row_id", "episode_length_buf"):
            setattr(self, k, getattr(env, k).clone())
        self._particle_region_cache = None
        self._particle_region_cache_step = -1
        self._particle_pos_e_cache = None
        self._particle_pos_e_cache_step = -1
        cfgs = task_cfg()
        lost = dict(cfgs["terminations"])["lost_grasp"]["params"]
        self.cfg = types.SimpleNamespace(
            **{k: fields[k] for k in ("particle_count_margin", "spill_table_height", "max_spill_fraction",
                                      "success_dwell_time_s", "success_min_lift_height",
                                      "state_bound_joint_position_margin", "state_bound_max_joint_velocity",
                                      "state_bound_max_cup_linear_velocity", "state_bound_max_cup_angular_velocity")},
            terminations=types.SimpleNamespace(lost_grasp=types.SimpleNamespace(params=lost)))
        self.termination_manager = _TerminationManager(n, dev)
        g = env.gripper_action
        self.action_manager = types.SimpleNamespace(
            action=env.action, prev_action=env.prev_action, get_term=lambda name: {"gripper_action": g}[name])
        for name, fn in ref["env_methods"].items():
            setattr(self, name, types.MethodType(fn, self))
