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

from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw

from orchard.interop import torch_bridge as tb
from orchard.interop import warp_metal as wm


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
    extra: dict = field(default_factory=dict)


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
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(model)
            if self.is_metal:
                # conditional graph nodes read the loop condition back on the host per iteration on Metal
                self.m.opt.graph_conditional = False
            if not self.opt.warn_overflow:
                self.m.opt.warn_overflow = 0
            if self.opt.solver_iterations is not None:
                self.m.opt.iterations = self.opt.solver_iterations
            if self.opt.ls_iterations is not None:
                self.m.opt.ls_iterations = self.opt.ls_iterations
            self.d = mjw.put_data(model, mjd, nworld=num_envs, nconmax=self.opt.nconmax, njmax=self.opt.njmax)
            self._reset_mask = wp.zeros(num_envs, dtype=wp.bool)
            self._graphs = {}
            self._capture()
        self.t = _TensorViews(self.d, num_envs)         # data views
        self.tm = _TensorViews(self.m, num_envs)        # model views (per-world fields have leading dim)
        self.reset_mask = tb.mps_tensor(self._reset_mask)
        self.event = wm.SharedEvent(device, "orchard.physics")
        self._dt = float(model.opt.timestep) * self.opt.substeps

    # -- graphs ---------------------------------------------------------------------------------

    def _capture(self):
        self._graphs.clear()
        if not self.opt.capture:
            return
        with wp.ScopedCapture(device=self.device) as cap:
            for _ in range(self.opt.substeps):
                mjw.step(self.m, self.d)
        self._graphs["step"] = cap.graph
        with wp.ScopedCapture(device=self.device) as cap:
            mjw.forward(self.m, self.d)
        self._graphs["forward"] = cap.graph
        with wp.ScopedCapture(device=self.device) as cap:
            mjw.reset_data(self.m, self.d, reset=self._reset_mask)
        self._graphs["reset"] = cap.graph

    def _run(self, name: str):
        with wp.ScopedDevice(self.device):
            g = self._graphs.get(name)
            if g is not None:
                wp.capture_launch(g)
            elif name == "step":
                for _ in range(self.opt.substeps):
                    mjw.step(self.m, self.d)
            elif name == "forward":
                mjw.forward(self.m, self.d)
            elif name == "reset":
                mjw.reset_data(self.m, self.d, reset=self._reset_mask)

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
        with wp.ScopedDevice(self.device):
            mjw.get_data_into(mjd, self.mj_model, self.d, world_id=world)
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
