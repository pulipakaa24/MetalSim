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


@wp.kernel
def _latch_capacity_overflow(overflow: wp.array(dtype=int), mask: int, latched: wp.array(dtype=int)):
    """OR this world's capacity overflow bits into ``latched`` (which mjw.reset_data does not clear)."""
    w = wp.tid()
    v = overflow[w] & mask
    if v != 0:
        latched[w] = latched[w] | v


@dataclass
class BatchSimOptions:
    substeps: int = 1                 # physics steps per call to step()
    capture: bool = True              # replay a recorded graph
    # Contacts per world and constraint rows per world. MuJoCo Warp's default njmax (64) overflows on
    # contact-rich scenes (condim-6 gripper contacts), and an overflow drops rows nondeterministically, which
    # makes the elliptic-cone Newton solver produce NaN. 512 covers every scene in the test corpus. "auto" sets
    # each to the model's provable bound (metalsim.physics.capacity: the pairs MuJoCo Warp tests x the largest
    # contact count its narrowphase writes per pair type, rows per contact from condim / cone, plus limit,
    # equality and friction-loss rows); a capacity that cannot be exceeded changes no result and every
    # capacity-sized launch shrinks with it. "auto" raises for models it cannot bound (heightfields, SDF, flex,
    # MULTICCD). check_overflow() is the guard: any capacity overflow since construction raises (latched after every
    # substep, so a task's reset of the world does not erase it).
    nconmax: int | str | None = None
    njmax: int | str | None = 512
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
    # Run-to-run bitwise repeatability (Warp's DeterministicMode.RUN_TO_RUN for MuJoCo Warp's kernels, as Newton's
    # SolverMuJoCo(deterministic=...) does): float atomics accumulate in a fixed order and slot-claiming counters follow
    # thread order (MetalSim's Warp fork implements it on Metal). Off by default: the kernels and results of existing
    # scenes are unchanged. Process-wide once on: it sets the module options of every loaded MuJoCo Warp module.
    deterministic: bool = False
    # Startup compilation on this many CPU threads (metalsim.physics.warmup): before the first graph capture, the
    # step / forward / reset launch code runs once in a discarded capture with every launch recorded instead of run,
    # and the recorded modules (MuJoCo Warp's model-specialised kernels included) compile in parallel. None = the
    # METALSIM_PRELOAD_WORKERS environment variable, else 0 (off: Warp's lazy loading, one module at a time).
    # Same compiled modules either way; with deterministic=True the pass runs inside det_scope like the capture, so
    # the same RUN_TO_RUN modules load. Only with capture=True. Statistics per capture in ``sim.preload_stats``.
    preload_workers: int | None = None
    extra: dict = field(default_factory=dict)


def _deterministic_max_records(model: mujoco.MjModel, njmax: int) -> int:
    """Per-thread record bound for MuJoCo Warp's deterministic atomics (the bound Newton's SolverMuJoCo derives):
    a dense contact-Jacobian thread can touch every constraint row; a sparse Hessian thread every element of a row's
    lower-triangular product (row width = the two longest kinematic chains); spatial tendons walk both chains."""
    nv = int(model.nv)
    chains, seen = [], set()
    for b in range(int(model.nbody)):
        wb = int(model.body_weldid[b])
        if wb in seen:
            continue
        seen.add(wb)
        dof, width = int(model.body_dofadr[wb]) + int(model.body_dofnum[wb]) - 1, 0
        while dof >= 0:
            width += 1
            dof = int(model.dof_parentid[dof])
        chains.append(width)
    chains.sort(reverse=True)
    row = 0
    if nv:
        row = min(nv, max(1, sum(chains[:2]), 2 * max((int(w) for w in model.ten_J_rownnz), default=0),
                          max((int(w) for w in model.flexedge_J_rownnz), default=0), nv if model.nu else 0))
    tendon = max((2 * max(int(n) - 1, 1) * int(w) for n, w in zip(model.tendon_num, model.ten_J_rownnz)), default=0)
    return max(1, int(njmax), row * (row + 1) // 2, tendon)


# MuJoCo Warp's determinism mode is process-wide (module options are shared, generated kernels are cached by MuJoCo
# Warp without the options): the first BatchSim fixes it, a BatchSim asking for the other mode afterwards raises.
# research_notes/MPM in main simulator/deterministic_batchsim.md has the Warp / Newton source reading behind this.
_PROCESS_MODE: str | None = None


def _warp_mujoco_modules():
    """Warp ``Module`` objects of MuJoCo Warp's kernels: its source modules (``mujoco_warp._src.*``) and the
    ``module="unique"`` modules its kernel builders create (named ``<kernel>_<hash>``, found by their kernels' source
    module: the CCD, solver and smooth kernels, among others)."""
    from warp._src import context

    def mjw_kernel(k):
        return str(getattr(getattr(k, "func", None), "__module__", "")).startswith("mujoco_warp")

    return {n: m for n, m in list(context.user_modules.items())
            if n.startswith("mujoco_warp") or any(mjw_kernel(k) for k in list(m.kernels.values()))}


def _claim_process_mode(deterministic: bool) -> None:
    global _PROCESS_MODE
    want = "run_to_run" if deterministic else "default"
    if _PROCESS_MODE == want:
        return
    if _PROCESS_MODE is not None:
        raise RuntimeError(f"this process already runs MuJoCo Warp in {_PROCESS_MODE!r} determinism mode; a BatchSim "
                           f"with deterministic={deterministic} needs its own process (BatchSimOptions.deterministic)")
    if deterministic:
        mode = wp.DeterministicMode.RUN_TO_RUN
        loaded = sorted(n for n, m in _warp_mujoco_modules().items()
                        if m.execs and m.options.get("deterministic") != mode)
        if loaded:
            raise RuntimeError("MuJoCo Warp kernels already ran in default mode in this process (" + ", ".join(loaded[:4])
                               + "): construct the deterministic BatchSim before any other MuJoCo Warp use")
    _PROCESS_MODE = want


def _check_deterministic_support(model: mujoco.MjModel) -> None:
    """Refuse models whose contacts deterministic mode cannot repeat. MuJoCo Warp's convex (GJK/EPA) narrowphase is
    repeatable only through the fork's deterministic CCD split (research_notes/MPM in main simulator/
    deterministic_ccd_divergence.md); its heightfield kernel is not split, so heightfield contacts must take the
    plane path (meshes of at most 256 vertices; primitives against a heightfield raise inside MuJoCo Warp)."""
    from mujoco_warp._src import collision_convex as cc

    if not hasattr(cc, "ccd_det_kernels_builder"):
        raise RuntimeError(f"deterministic BatchSim needs the MuJoCo Warp fork's deterministic CCD split (fork branch "
                           f"metalsim-det-ccd); {cc.__file__} does not have it")
    hfield, mesh = int(mujoco.mjtGeom.mjGEOM_HFIELD), int(mujoco.mjtGeom.mjGEOM_MESH)
    collides = (model.geom_contype | model.geom_conaffinity) != 0
    if np.any((model.geom_type == hfield) & collides):
        big = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g) for g in range(model.ngeom)
               if model.geom_type[g] == mesh and collides[g] and model.mesh_vertnum[model.geom_dataid[g]] > 256]
        if big:
            raise NotImplementedError(f"deterministic BatchSim: meshes of more than 256 vertices against a heightfield "
                                      f"go through GJK/EPA, which is not repeatable in deterministic mode: {big[:4]}")


def _set_mujoco_warp_determinism(smooth_records: int) -> None:
    """RUN_TO_RUN on every loaded MuJoCo Warp module (``smooth`` gets the record bound; options only set where they
    differ, so unchanged modules keep their hash), and drop MuJoCo Warp's generated-kernel cache, whose keys do not
    include module options. Newton's SolverMuJoCo does the same (solver_mujoco.py ``_set_mujoco_warp_module_options``,
    ``_prepare_generated_kernels``)."""
    import sys

    from mujoco_warp._src import warp_util

    mode = wp.DeterministicMode.RUN_TO_RUN
    for name, mod in list(sys.modules.items()):
        if name.startswith("mujoco_warp._src.") and mod is not None:
            want = {"deterministic": mode,
                    "deterministic_max_records": smooth_records if name == "mujoco_warp._src.smooth" else 0}
            wm_ = wp.get_module(name)
            if any(wm_.options.get(k) != v for k, v in want.items()):
                wp.set_module_options(want, module=mod)
    warp_util._KERNEL_CACHE.clear()


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
            # determinism is fixed before MuJoCo Warp's first launch in this sim (put_model / put_data launch kernels):
            # Warp's deterministic launch metadata is taken when a module loads, so a mode set later is not reliable
            _claim_process_mode(self.opt.deterministic)
            if self.opt.deterministic:
                _check_deterministic_support(model)
            self._det_records = None
            if self.opt.deterministic:
                self._det_records = 0
                _set_mujoco_warp_determinism(0)
            with _m_layout(self.opt.m_dense_max), self.det_scope():
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
            self.capacity_bounds = None
            nconmax, njmax = self.opt.nconmax, self.opt.njmax
            if nconmax == "auto" or njmax == "auto":
                from metalsim.physics import capacity
                self.capacity_bounds = capacity.bounds(wmodel, self.m)
                if nconmax == "auto":
                    nconmax = self.capacity_bounds.nconmax
                if njmax == "auto":
                    njmax = self.capacity_bounds.njmax
            if self.opt.solver_iterations is not None:
                self.m.opt.iterations = self.opt.solver_iterations
            if self.opt.ls_iterations is not None:
                self.m.opt.ls_iterations = self.opt.ls_iterations
            with _m_layout(self.opt.m_dense_max), self.det_scope():
                self.d = mjw.put_data(wmodel, mjd, nworld=num_envs, nconmax=nconmax, njmax=njmax)
            if self.opt.deterministic:
                # the record bound needs njmax; smooth's kernels first run in the step (checked by check_determinism)
                self._det_records = _deterministic_max_records(wmodel, self.d.njmax)
                _set_mujoco_warp_determinism(self._det_records)
            self._reset_mask = wp.zeros(num_envs, dtype=wp.bool)
            self._graphs = {}
            self.preload_stats = []     # one entry per capture when startup compilation runs in parallel
            # capacity overflow bits latched after every substep: mjw.reset_data zeroes d.overflow of the worlds it
            # resets, and tasks reset (fallen) worlds every control step, so d.overflow alone loses overflows
            from metalsim.physics import capacity
            self._overflow_latched = wp.zeros(num_envs, dtype=int)
            self._latch_mask = capacity.capacity_mask()
            self._substep_hooks = [self._latch_overflow]   # launched after every physics substep (inside the step graph)
            self._reset_hooks = []      # launched after reset_data with the reset mask (inside the reset graph)
            self._capture()
        self.t = _TensorViews(self.d, num_envs)         # data views
        self.tm = _TensorViews(self.m, num_envs)        # model views (per-world fields have leading dim)
        self.reset_mask = tb.mps_tensor(self._reset_mask)
        self.event = wm.SharedEvent(device, "metalsim.physics")
        self._dt = float(model.opt.timestep) * self.opt.substeps

    # -- graphs ---------------------------------------------------------------------------------

    @contextlib.contextmanager
    def det_scope(self):
        """Hold Warp's deterministic config while MuJoCo Warp kernels may be created (its ``module="unique"`` kernels
        are generated lazily and take the config, not the source module's options). No-op unless deterministic."""
        if self._det_records is None:
            yield
            return
        old = wp.config.deterministic, wp.config.deterministic_max_records
        wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
        wp.config.deterministic_max_records = int(self._det_records)
        try:
            yield
        finally:
            wp.config.deterministic, wp.config.deterministic_max_records = old

    def _capture(self):
        with self.det_scope():
            self._capture_graphs()
        if self._det_records is not None and self._graphs:
            self.check_determinism()

    def check_determinism(self) -> None:
        """With ``deterministic=True``: raise unless every MuJoCo Warp kernel module this sim's graphs replay was built
        in RUN_TO_RUN mode (a module loaded earlier in default mode would replay its non-deterministic kernels)."""
        mode = wp.DeterministicMode.RUN_TO_RUN
        execs = set()
        for g in self._graphs.values():
            execs |= set(getattr(g, "module_execs", ()))
        bad = []
        for name, mod in _warp_mujoco_modules().items():
            used = [e for e in mod.execs.values() if e in execs]
            if used and mod.options.get("deterministic") != mode:
                bad.append(name)
            for e in used:
                if e.module_hash != mod.get_module_hash(e.block_dim):
                    bad.append(f"{name} (stale executable)")
        if bad:
            raise RuntimeError(f"deterministic BatchSim replays non-deterministic MuJoCo Warp modules: {bad}")

    def _capture_graphs(self):
        self._graphs.clear()
        if not self.opt.capture:
            return
        from metalsim.physics import warmup
        workers = warmup.preload_workers(self.opt.preload_workers)
        if workers:
            # every module the three graphs launch, compiled on `workers` threads before they are recorded (inside
            # _capture's det_scope, so generated kernels get the same mode; also on re-captures: loaded modules cost
            # nothing, new substep hooks' kernels are picked up)
            self.preload_stats.append(warmup.preload_capture(self._preload_body, self.device, workers))
        with wp.ScopedCapture(device=self.device) as cap:
            self._launch_substeps()
        self._graphs["step"] = cap.graph
        with wp.ScopedCapture(device=self.device) as cap:
            mjw.forward(self.m, self.d)
        self._graphs["forward"] = cap.graph
        with wp.ScopedCapture(device=self.device) as cap:
            self._launch_reset()
        self._graphs["reset"] = cap.graph

    def _preload_body(self) -> None:
        self._launch_substeps()
        mjw.forward(self.m, self.d)
        self._launch_reset()

    def _launch_substeps(self) -> None:
        for _ in range(self.opt.substeps):
            mjw.step(self.m, self.d)
            for h in self._substep_hooks:
                h()

    def _latch_overflow(self) -> None:
        wp.launch(_latch_capacity_overflow, dim=self.n, inputs=[self.d.overflow, self._latch_mask],
                  outputs=[self._overflow_latched])

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
        with wp.ScopedDevice(self.device), self.det_scope():
            self._launch_substeps()

    def launch_reset(self) -> None:
        """Launch the reset of the worlds selected by ``reset_mask`` directly (no graph replay, no signal), for
        capturing into a policy-step graph (write the mask on torch's queue before replaying it)."""
        with wp.ScopedDevice(self.device), self.det_scope():
            self._launch_reset()

    def _run(self, name: str):
        with wp.ScopedDevice(self.device), self.det_scope():
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
        with wp.ScopedDevice(self.device), self.det_scope():
            getattr(mjw, level)(self.m, self.d)
        return self._signal()

    def overflow_flags(self) -> dict[str, int]:
        """Per-flag count of worlds whose overflow bit is set (synchronizes). Use in tests and at
        rollout boundaries: NEFC / NARROWPHASE / CCD overflows mean dropped constraints or contacts
        and must be fixed by raising ``njmax`` / ``nconmax``; LS_ITERATIONS and ITERATIONS are
        solver-convergence warnings (same as MuJoCo C's). Capacity bits count every world that overflowed
        since construction (latched, survive resets); the solver-cap bits are the live ``d.overflow`` (since
        each world's last reset)."""
        self.synchronize()
        ov = self.d.overflow.numpy() | self._overflow_latched.numpy()
        return {f.name: int(((ov & int(f)) != 0).sum()) for f in mjw.OverflowType if int(f) not in (0, int(mjw.OverflowType.ALL))
                and ((ov & int(f)) != 0).any()}

    def check_overflow(self) -> dict[str, int]:
        """The overflow flags (synchronizes), raising RuntimeError on any capacity overflow since construction (NEFC,
        NARROWPHASE, CCD, ...: rows or contacts were dropped, the physics is wrong from that step on; latched after
        every substep, so a world's reset does not hide it). The solver-cap flags (ITERATIONS, LS_ITERATIONS) are
        returned, not raised. Call at rollout boundaries."""
        from metalsim.physics import capacity
        ov = self.overflow_flags()
        if self._det_records is not None and self.is_metal:
            # Warp's deterministic counters overflow into a sticky flag that graph replays never check
            from warp._src import metal_deterministic

            flag = int(metal_deterministic._state(self.device).overflow.numpy()[0])
            if flag & 2:
                # the Warp fork's tripwire: a kernel's counter calls differ between its counting and execution
                # passes, so slots went to the wrong calls (contacts or rows written into other worlds' slots)
                raise RuntimeError("Warp deterministic-mode counter phase mismatch: a MuJoCo Warp kernel's slot-counter "
                                   "calls differ between its two passes (research_notes/MPM in main simulator/"
                                   "deterministic_ccd_divergence.md)")
            if flag:
                raise RuntimeError("Warp deterministic-mode counter overflow (deterministic_max_records too small)")
        cap = capacity.capacity_overflows(ov)
        if cap:
            raise RuntimeError(f"MuJoCo Warp capacity overflow {cap} (worlds affected): raise njmax / nconmax "
                               f"(current njmax {self.d.njmax}, naconmax {self.d.naconmax} = {self.d.naconmax // self.n} per world)")
        return ov

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
