"""Per-step evidence for deterministic-mode counter trips (MetalSim batch_18, batch_19).

For every physics step of a deterministic BatchSim this records (a) which deterministic launches set Warp's sticky
counter flag and with which bits, and (b) how many worlds reached the solver's iteration cap in that step (MuJoCo Warp
sets OverflowType.ITERATIONS in d.overflow). Captured: after every deterministic launch recorded into a graph, a tiny
non-deterministic kernel copies the flag into a per-launch slot and clears it; the slots are read and cleared after
each step. Uncaptured: the flag is read after every deterministic launch (synchronize). With the diagnostic Warp build
(scratch/b18 warp-diag) the bits say what kind of mismatch: 2 = a phase-1 counter call with no phase-0 record at all,
16 = beyond a nonzero phase-0 count, 8 = another destination / count, 4 = the thread's phase-0 and phase-1 call
counts differ (either direction: one extra phase-1 call with no phase-0 record sets 2 and 4, flag 6), 1 = overflow.

    python -P det_step_probe.py --scene stack --worlds 64 --iterations 2 --capture 1 --steps 40 --json out.json
    python -P det_step_probe.py --scene g1 --worlds 1024 --capture 1 --steps 60 --json out.json
"""
import argparse
import ctypes
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True


@wp.kernel(enable_backward=False, module="unique",
           module_options={"deterministic": wp.DeterministicMode.NOT_GUARANTEED})
def _snapshot_flag(flag: wp.array(dtype=wp.int32), snaps: wp.array(dtype=wp.int32), slot: int):
    v = flag[0]
    snaps[slot] = v
    if v != 0:
        flag[0] = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=("stack", "g1"), default="stack")
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=None, help="solver iteration cap (stack scene)")
    ap.add_argument("--capture", type=int, default=1)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--slots", type=int, default=1 << 20)
    ap.add_argument("--json")
    a = ap.parse_args()
    import mujoco_warp as mjw
    from warp._src import metal_deterministic as md

    dev = wp.get_device("metal:0")
    snaps = wp.zeros(a.slots, dtype=wp.int32, device=dev)
    dummy = wp.zeros(1, dtype=wp.int32, device=dev)
    wp.launch(_snapshot_flag, dim=1, inputs=[dummy, snaps, 0], device=dev)
    wp.synchronize_device(dev)
    snaps.zero_()
    recorded, eager_hits = [], []
    det_launches = {}   # kernel -> deterministic launches (captured: recorded into a graph; uncaptured: run)
    orig = md.launch

    def launch(kernel, hooks, params, bounds, device, *args, **kw):
        orig(kernel, hooks, params, bounds, device, *args, **kw)
        det_launches[kernel.key] = det_launches.get(kernel.key, 0) + 1
        state = md._state(device)
        if device.is_capturing:
            slot = len(recorded)
            if slot < a.slots:
                recorded.append(kernel.key)
                wp.launch(_snapshot_flag, dim=1, inputs=[state.overflow, snaps, slot], device=device)
            return
        wp.synchronize_device(device)
        flag = ctypes.c_int32.from_address(state.overflow.ptr).value
        if flag:
            eager_hits.append((kernel.key, flag, int(bounds.size)))
            ctypes.c_int32.from_address(state.overflow.ptr).value = 0

    md.launch = launch

    if a.scene == "stack":
        from metalsim.physics.batch import BatchSim, BatchSimOptions
        from metalsim.physics.mpm.probes.rigid_det_diag import model_of

        sim = BatchSim(model_of("stack"), a.worlds, options=BatchSimOptions(
            deterministic=True, capture=bool(a.capture), solver_iterations=a.iterations, warn_overflow=False))
        cap = int(sim.m.opt.iterations)

        def step():
            sim.step()
    else:
        import torch

        from metalsim.interop import torch_bridge as tb
        from metalsim.learn.g1_velocity import G1VelocityTask

        task = G1VelocityTask(a.worlds, terrain="flat", batch_options={"deterministic": True, "capture": bool(a.capture)})
        sim = task.sim
        cap = int(sim.m.opt.iterations)
        action = wp.zeros((a.worlds, task.act_dim), dtype=float, device=dev)
        t_action = tb.mps_tensor(action)
        torch.manual_seed(0)
        task.reset_all()

        def step():
            t_action.uniform_(-1.0, 1.0)
            v = sim.event.next_value(); tb.signal_event(sim.event, v); sim.wait(sim.event, v)
            with wp.ScopedDevice(dev):
                task.launch_apply_action(action)
            sim.step()

    ITER = int(mjw.OverflowType.ITERATIONS)
    per_step = []
    sim.d.overflow.zero_()
    for s in range(a.steps):
        n_eager = len(eager_hits)
        step()
        sim.synchronize()
        ov = sim.d.overflow.numpy()
        cap_worlds = int(((ov & ITER) != 0).sum())
        sim.d.overflow.zero_()
        rec = {"step": s, "cap_worlds": cap_worlds}
        if recorded:
            sv = snaps.numpy()[: len(recorded)]
            nz = np.nonzero(sv)[0]
            rec["captured_hits"] = [(recorded[i], int(sv[i])) for i in nz[:8]]
            rec["n_captured_hits"] = int(len(nz))
            if len(nz):
                snaps.zero_()
        rec["eager_hits"] = eager_hits[n_eager:][:8]
        per_step.append(rec)
    steps_with_cap = [r["step"] for r in per_step if r["cap_worlds"]]
    steps_with_hits = [r["step"] for r in per_step if r.get("n_captured_hits") or r["eager_hits"]]
    kinds, kernels = {}, {}
    for r in per_step:
        for k, f in [(x[0], x[1]) for x in r.get("captured_hits", [])] + [(x[0], x[1]) for x in r["eager_hits"]]:
            kernels[k] = kernels.get(k, 0) + 1
            kinds[str(f)] = kinds.get(str(f), 0) + 1
    res = {"scene": a.scene, "worlds": a.worlds, "capture": a.capture, "iteration_cap": cap, "steps": a.steps,
           "steps_with_cap_worlds": steps_with_cap, "steps_with_hits": steps_with_hits,
           "hits_only_in_cap_steps": set(steps_with_hits) <= set(steps_with_cap),
           "kernels_hit": kernels, "flag_values": kinds, "recorded_det_launches": len(recorded),
           "first_cap_step": steps_with_cap[0] if steps_with_cap else None,
           "steps_without_cap_worlds": a.steps - len(steps_with_cap), "det_launches": det_launches,
           "fused_update": any("fused" in k for k in det_launches),
           "mujoco_warp": mjw.__file__, "warp": wp.__file__}
    print("RESULT " + json.dumps(res))
    if a.json:
        json.dump({**res, "per_step": per_step}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
