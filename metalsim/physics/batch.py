"""Batched GPU physics on MuJoCo Warp (Metal) with event-ordered, zero-copy handoff to PyTorch.

One ``BatchSim`` holds N worlds of one MJCF model on ``metal:0``. Stepping is a replayed graph
(all substeps in one submission). Every state array is reachable as a zero-copy MPS tensor
(``sim.t.qpos``, ``sim.t.ctrl``, ``sim.t.xpos`` ...). Nothing here synchronizes the host: the
caller orders producers and consumers with the events (see ``step`` / ``after`` / ``wait``).

Rollout step contract (GPU timeline, no host waits):

    torch writes sim.t.ctrl  -> tb.signal_event(e_learner, v)   ; sim.wait(e_learner, v)
    v_sim = sim.step()        # graph launch + signal on sim.event
    renderer / torch:  wait sim.event >= v_sim  before reading sim.t.*
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw

from metalsim.interop import torch_bridge as tb
from metalsim.interop import warp_metal as wm


class _TensorViews:
    """Lazy zero-copy MPS views over the fields of a MuJoCo Warp ``Data`` (or ``Model``)."""

    def __init__(self, obj, nworld: int):
        self._obj = obj
        self._nworld = nworld
        self._cache: dict[str, torch.Tensor] = {}

    def __getattr__(self, name: str) -> torch.Tensor:
        if name.startswith("_"):
            raise AttributeError(name)
        t = self._cache.get(name)
        if t is None:
            a = getattr(self._obj, name)
            if not isinstance(a, wp.array):
                raise AttributeError(f"{name} is not a Warp array")
            if a.size == 0:
                t = torch.empty(tuple(a.shape), device="mps")
            else:
                t = tb.mps_tensor(a)
                if a.shape and a.shape[0] == 1 and self._nworld > 1 and t.dim() > 0:
                    t = t.expand((self._nworld,) + tuple(t.shape[1:]))  # shared model field
            self._cache[name] = t
        return t


@dataclass
class BatchSimOptions:
    substeps: int = 1                 # physics steps per call to step()
    capture: bool = True              # replay a recorded graph
    nconmax: int | None = None
    # Constraint rows per world. MuJoCo Warp's default (64) overflows on contact-rich scenes
    # (condim-6 gripper contacts), and an overflow drops rows nondeterministically, which makes
    # the elliptic-cone Newton solver produce NaN. 512 covers every scene in the test corpus.
    njmax: int | None = 512
    solver_iterations: int | None = None
    ls_iterations: int | None = None
    warn_overflow: bool = False       # MuJoCo Warp prints on solver overflow; off for RL loops
    # Model fields given a per-world leading dimension (physics domain randomization): e.g.
    # ("body_mass", "geom_friction", "dof_damping", "actuator_gainprm"). They become (N, ...) tensors
    # in ``sim.tm`` that torch can write per env; call ``sim.recompute_constants()`` after mass changes.
    per_world_fields: tuple = ()
    # Warp-side overrides that leave the MjModel (and MuJoCo C references built from it) untouched:
    # ``jacobian`` "dense" / "sparse" / "auto" selects MuJoCo Warp's constraint-Jacobian layout (the
    # same equations; MuJoCo Warp's own default for nv > 32 is sparse); ``block_dim`` overrides
    # MuJoCo Warp's per-kernel threadgroup widths (``mjw.types.BlockDim`` field -> int).
    jacobian: str | None = None
    block_dim: dict = field(default_factory=dict)
    # Largest diagonal block of M (and of the implicit integrator's M - dt*D) that MuJoCo Warp factors
    # with a dense tile Cholesky; larger blocks use its tree-sparse L'DL (MuJoCo C's mj_factorI
    # algorithm). Same matrix, different arithmetic (float noise). Default 32 (since 2026-09-25): the
    # sparse path wins on the G1's 43-dof tree (+16 % PPO loop) and loses ~1 % on Tron1's 14; every other
    # scene in the repo (trees of 2-18 dofs) keeps its dense tile (scripts/diagnostics/fast_factorization_scenes.py,
    # runs/fastfact/). None or 64 = MuJoCo Warp's M_BLOCK_DENSE_MAX (the previous default); 0 = every
    # block of more than six dofs sparse (G1-only setting before; ~1 % slower on Tron1).
    m_dense_max: int | None = 32
    # Warp Metal: largest matrix that tile_cholesky factors in registers. Default 48 (since 2026-09-25): the
    # G1's nv = 43 in registers instead of the barrier-per-column path; matrices up to 40 take the same path
    # as before. 40 = the Warp fork's own default (the previous behaviour); None leaves warp.config as is.
    # Process-wide (warp.config), read when a module is built (part of the module hash): the last BatchSim
    # constructed sets it.
    metal_register_cholesky_max: int | None = 48
    extra: dict = field(default_factory=dict)


@contextlib.contextmanager
def _m_layout(dense_max):
    """MuJoCo Warp reads M_BLOCK_DENSE_MAX when it lays out M's factor (put_model, put_data,
    get_data_into); hold the override across those calls only."""
    from mujoco_warp._src import types as mjw_types
    old = mjw_types.M_BLOCK_DENSE_MAX
    if dense_max is not None:
        mjw_types.M_BLOCK_DENSE_MAX = int(dense_max)
    try:
        yield
    finally:
        mjw_types.M_BLOCK_DENSE_MAX = old


class BatchSim:
    def __init__(self, model: mujoco.MjModel, num_envs: int, device: str = "metal:0",
                 options: BatchSimOptions | None = None):
        self.opt = options or BatchSimOptions()
        self.mj_model = model
        self.n = num_envs
        self.device = wp.get_device(device)
        self.is_metal = getattr(self.device, "is_metal", False)
        mjd = mujoco.MjData(model)
        mujoco.mj_forward(model, mjd)
        if self.opt.metal_register_cholesky_max is not None:
            wp.config.metal_register_cholesky_max = int(self.opt.metal_register_cholesky_max)
        with wp.ScopedDevice(self.device):
            batch_sizes = {f: num_envs for f in self.opt.per_world_fields} or None
            wmodel = model
            if self.opt.jacobian is not None:     # (efc layout: put_data and get_data_into take this copy too)
                import copy
                wmodel = copy.deepcopy(model)
                wmodel.opt.jacobian = {"dense": mujoco.mjtJacobian.mjJAC_DENSE, "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
                                       "auto": mujoco.mjtJacobian.mjJAC_AUTO}[self.opt.jacobian]
            self._wmodel = wmodel
            with _m_layout(self.opt.m_dense_max):
                self.m = mjw.put_model(wmodel, batch_sizes=batch_sizes)
            for k, v in self.opt.block_dim.items():
                if not hasattr(self.m.block_dim, k):
                    raise ValueError(f"unknown MuJoCo Warp block_dim field {k!r}")
                setattr(self.m.block_dim, k, int(v))
            if self.is_metal:
                # conditional graph nodes read the loop condition back on the host per iteration on Metal
                self.m.opt.graph_conditional = False
            if not self.opt.warn_overflow:
                self.m.opt.warn_overflow = 0
            if self.opt.solver_iterations is not None:
                self.m.opt.iterations = self.opt.solver_iterations
            if self.opt.ls_iterations is not None:
                self.m.opt.ls_iterations = self.opt.ls_iterations
            with _m_layout(self.opt.m_dense_max):
                self.d = mjw.put_data(wmodel, mjd, nworld=num_envs, nconmax=self.opt.nconmax, njmax=self.opt.njmax)
            self._reset_mask = wp.zeros(num_envs, dtype=wp.bool)
            self._graphs = {}
            self._substep_hooks = []    # launched after every physics substep (inside the step graph)
            self._reset_hooks = []      # launched after reset_data with the reset mask (inside the reset graph)
            self._capture()
        self.t = _TensorViews(self.d, num_envs)         # data views
        self.tm = _TensorViews(self.m, num_envs)        # model views (per-world fields have leading dim)
        self.reset_mask = tb.mps_tensor(self._reset_mask)
        self.event = wm.SharedEvent(device, "metalsim.physics")
        self._dt = float(model.opt.timestep) * self.opt.substeps

    # -- graphs ---------------------------------------------------------------------------------

    def _capture(self):
        self._graphs.clear()
        if not self.opt.capture:
            return
        with wp.ScopedCapture(device=self.device) as cap:
            self._launch_substeps()
        self._graphs["step"] = cap.graph
        with wp.ScopedCapture(device=self.device) as cap:
            mjw.forward(self.m, self.d)
        self._graphs["forward"] = cap.graph
        with wp.ScopedCapture(device=self.device) as cap:
            self._launch_reset()
        self._graphs["reset"] = cap.graph

    def _launch_substeps(self) -> None:
        for _ in range(self.opt.substeps):
            mjw.step(self.m, self.d)
            for h in self._substep_hooks:
                h()

    def _launch_reset(self) -> None:
        mjw.reset_data(self.m, self.d, reset=self._reset_mask)
        for h in self._reset_hooks:
            h(self._reset_mask)

    def add_substep_hook(self, fn, reset_fn=None) -> None:
        """Register Warp launches to run after every physics substep (e.g. a contact sensor's
        reduction and history update) and optionally after resets (``reset_fn(mask)``, mask a
        (N,) bool Warp array). The step/reset graphs are re-captured to include them."""
        self._substep_hooks.append(fn)
        if reset_fn is not None:
            self._reset_hooks.append(reset_fn)
        with wp.ScopedDevice(self.device):
            self._capture()

    def launch_step(self) -> None:
        """Launch the substeps directly (no graph replay), for capturing into a larger graph that
        also holds the policy and the rollout bookkeeping."""
        with wp.ScopedDevice(self.device):
            self._launch_substeps()

    def _run(self, name: str):
        with wp.ScopedDevice(self.device):
            g = self._graphs.get(name)
            if g is not None:
                wp.capture_launch(g)
            elif name == "step":
                self._launch_substeps()
            elif name == "forward":
                mjw.forward(self.m, self.d)
            elif name == "reset":
                self._launch_reset()

    # -- ordering ---------------------------------------------------------------------------------

    def wait(self, event: wm.SharedEvent, value: int) -> None:
        """Everything the sim launches afterwards runs after ``event`` reaches ``value``."""
        wm.wait(event, value, self.device)

    def _signal(self) -> int:
        v = self.event.next_value()
        wm.signal(self.event, v, self.device)
        return v

    def after(self, value: int) -> None:
        """Order PyTorch's subsequent kernels after the sim work that signalled ``value``."""
        tb.wait_event(self.event, value)

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)

    def recompute_constants(self, level: str = "set_const") -> int:
        """After per-world changes to masses/inertias (``set_const``), qpos0/armature (``set_const_0``)
        or gravcomp (``set_const_fixed``); launched on the Warp queue, returns the completion value."""
        with wp.ScopedDevice(self.device):
            getattr(mjw, level)(self.m, self.d)
        return self._signal()

    def overflow_flags(self) -> dict[str, int]:
        """Per-flag count of worlds whose overflow bit is set (synchronizes). Use in tests and at
        rollout boundaries: NEFC / NARROWPHASE / CCD overflows mean dropped constraints or contacts
        and must be fixed by raising ``njmax`` / ``nconmax``; LS_ITERATIONS and ITERATIONS are
        solver-convergence warnings (same as MuJoCo C's)."""
        self.synchronize()
        ov = self.d.overflow.numpy()
        return {f.name: int(((ov & int(f)) != 0).sum()) for f in mjw.OverflowType if int(f) not in (0, int(mjw.OverflowType.ALL))
                and ((ov & int(f)) != 0).any()}

    # -- simulation ---------------------------------------------------------------------------------

    @property
    def dt(self) -> float:
        return self._dt

    def step(self) -> int:
        """Advance all worlds by ``substeps`` physics steps; returns the event value of completion."""
        self._run("step")
        return self._signal()

    def forward(self) -> int:
        self._run("forward")
        return self._signal()

    def reset(self, mask: torch.Tensor | None = None) -> int:
        """Reset the worlds where ``mask`` is true (all if None) to the model's qpos0.

        ``mask`` is written on torch's queue and ordered before the reset kernels; state
        randomization is done by the caller writing ``sim.t.qpos`` after ``after(value)`` and
        then calling ``forward()`` after ``wait``-ing on its own event.
        """
        if mask is None:
            self.reset_mask.fill_(True)
        else:
            self.reset_mask.copy_(mask.to(device="mps", dtype=torch.bool))
        v = self.event.next_value()
        tb.signal_event(self.event, v)
        self.wait(self.event, v)
        self._run("reset")
        return self._signal()

    # -- host-side helpers (synchronizing; for tests, tools and resets outside the hot loop) --------

    def get_world(self, world: int, mjd: mujoco.MjData | None = None) -> mujoco.MjData:
        self.synchronize()
        mjd = mjd or mujoco.MjData(self.mj_model)
        with wp.ScopedDevice(self.device), _m_layout(self.opt.m_dense_max):
            mjw.get_data_into(mjd, self._wmodel, self.d, world_id=world)
        return mjd

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray | None = None, worlds=None) -> None:
        """Host write of qpos/qvel (synchronizes); followed by forward()."""
        self.synchronize()
        q = self.d.qpos.numpy()
        idx = range(self.n) if worlds is None else worlds
        for i, w in enumerate(idx):
            q[w] = qpos if qpos.ndim == 1 else qpos[i]
        self.d.qpos.assign(q)
        if qvel is not None:
            v = self.d.qvel.numpy()
            for i, w in enumerate(idx):
                v[w] = qvel if qvel.ndim == 1 else qvel[i]
            self.d.qvel.assign(v)
        self.forward()
