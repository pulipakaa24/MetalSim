"""Policy-step-shaped A/B of the coupled step: per-step resets of a fraction of the worlds, the coupled step and an
observation, with the eager API (``A``) or one captured graph per policy step (``B``).

    python policy_step_ab.py --worlds 64 --steps 200 --fraction 0.02 [--json out.json]

``A`` (MPMCoupling as on main before ``launch_step`` / capturable reset): the done mask comes from torch, is read on
the host (``.cpu()``, a synchronization, because ``MPMCoupling.reset`` took a host mask), and when any world is done
``sim.reset(mask)`` and the eager MPM reset run; then ``mpm.step()`` (its own graph) and the observation.
``B``: the mask is written into ``sim.reset_mask`` on torch's queue and one graph per policy step replays
``sim.launch_reset(); mpm.launch_reset(from_sim_mask=True); mpm.launch_step()`` (resets with an all-false mask are
no-ops), then the observation. Both see the same mask sequence; per-step hashes of the robot state, the sand wrench and
the particles must agree bitwise (``bitwise_equal``). Timing: wall time of the whole loop over ``--steps`` policy steps
(one synchronization at the end), interleaved ``--rounds`` times, A then B; run as a timing job on the 16-inch only.
Every run is its own process (``--single``): a 1,024-world coupled scene with its captured graphs holds about 10 GB,
and the scenes of earlier runs in one process were not all released (batch_07 ran out of memory on the eighth).
Scene: scale_coupled.py's box dropped on a sand bed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402


def make(a):
    from metalsim.physics.mpm.probes import scale_coupled

    b = argparse.Namespace(worlds=a.worlds, rigid_only=False, voxel=a.voxel, iterations=a.iterations, solver=a.solver,
                           cells_per_world=a.cells_per_world, capture=1)
    return scale_coupled.build(b)


def old_reset(mpm, mask_np):
    """MPMCoupling.reset as on main before this change: a fresh device mask from the host and eager launches."""
    import warp as wp

    with wp.ScopedDevice(mpm.sim.device):
        wp.copy(mpm._reset_mask, wp.array(np.append(mask_np, False), dtype=wp.bool))
        mpm._launch_reset(False)
    return mpm.sim._signal()


def masks(a):
    rng = np.random.default_rng(1)
    return [rng.random(a.worlds) < a.fraction for _ in range(a.warmup + a.steps)]


def run(a, mode, hashes):
    import torch
    import warp as wp

    from metalsim.interop import torch_bridge as tb

    sim, mpm = make(a)
    ms = masks(a)
    mask_dev = [torch.from_numpy(m).to("mps") for m in ms]   # the "done" flags a policy loop has on the GPU
    out = []
    graph = None

    def hx():
        h = hashlib.sha256()
        for arr in (sim.d.qpos, sim.d.qvel, mpm.coupling_f, mpm.particle_q, mpm.particle_qd):
            h.update(arr.numpy().tobytes())
        return h.hexdigest()[:16]

    # first step and reset eager in both modes (B needs them before capturing)
    mpm.step()
    mpm.reset(np.zeros(a.worlds, bool))
    sim.synchronize()
    t0 = None
    obs = None
    for k in range(a.warmup + a.steps):
        if k == a.warmup:
            sim.synchronize()
            torch.mps.synchronize()
            t0 = time.perf_counter()
        mt = mask_dev[k]
        if mode == "A":
            mh = mt.cpu().numpy()
            if mh.any():
                sim.reset(mt)
                old_reset(mpm, mh)
            v = mpm.step()
        else:
            sim.reset_mask.copy_(mt)
            ev = sim.event.next_value()
            tb.signal_event(sim.event, ev)
            sim.wait(sim.event, ev)
            if graph is None:
                with wp.ScopedDevice(sim.device), wp.ScopedCapture(device=sim.device, force_module_load=False) as cap:
                    sim.launch_reset()
                    mpm.launch_reset(from_sim_mask=True)
                    mpm.launch_step()
                graph = cap.graph
            with wp.ScopedDevice(sim.device):
                wp.capture_launch(graph)
            v = sim._signal()
        sim.after(v)
        obs = sim.t.qpos.clone()          # the policy's observation (torch queue, after the step)
        if hashes:
            out.append(hx())
    sim.synchronize()
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    mpm.check_status()
    return {"ms_per_policy_step": dt * 1e3 / a.steps, "resets": int(sum(m.any() for m in ms[a.warmup:])),
            "obs_finite": bool(torch.isfinite(obs).all())}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--fraction", type=float, default=0.02)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--solver", default="auto")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--cells-per-world", type=int, default=512)
    ap.add_argument("--hash-steps", type=int, default=60, help="steps of the bitwise A/B (hashing synchronizes)")
    ap.add_argument("--json")
    ap.add_argument("--single", choices=("A", "B"), help="internal: one run in this process")
    ap.add_argument("--hash", type=int, default=0, help="internal: with --single, hash every step")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    if a.single:
        with wp.ScopedDevice("metal:0"):
            if a.hash:
                a = argparse.Namespace(**{**vars(a), "steps": a.hash_steps, "warmup": 0})
            r, hashes = run(a, a.single, bool(a.hash))
        r["hashes"] = hashes
        json.dump(r, open(a.json, "w"))
        return
    import subprocess
    import tempfile

    def child(mode, hashes):
        with tempfile.TemporaryDirectory() as d:
            out = f"{d}/r.json"
            argv = [sys.executable, "-P", __file__, "--single", mode, "--hash", str(int(hashes)), "--json", out]
            for k in ("worlds", "steps", "warmup", "fraction", "solver", "voxel", "iterations", "cells_per_world",
                      "hash_steps"):
                argv += [f"--{k.replace('_', '-')}", str(getattr(a, k))]
            p = subprocess.run(argv, capture_output=True, text=True)
            sys.stdout.write(p.stdout[-2000:])
            sys.stderr.write(p.stderr[-4000:])
            if p.returncode != 0:
                raise RuntimeError(f"run {mode} (hashes={hashes}) failed with exit code {p.returncode}")
            return json.load(open(out))

    res = {"worlds": a.worlds, "fraction": a.fraction, "steps": a.steps, "solver": a.solver}
    if True:
        ha, hb_ = child("A", True)["hashes"], child("B", True)["hashes"]
        res["bitwise_equal"] = ha == hb_ and len(ha) == a.hash_steps
        res["first_differing_step"] = next((i for i in range(len(ha)) if ha[i] != hb_[i]), None)
        timings = {"A": [], "B": []}
        for _ in range(a.rounds):
            for mode in ("A", "B"):
                r = child(mode, False)
                timings[mode].append(r["ms_per_policy_step"])
                res["resets_per_run"] = r["resets"]
        for mode, v in timings.items():
            res[f"{mode}_ms_per_policy_step"] = v
            res[f"{mode}_ms_median"] = float(np.median(v))
        res["saving_ms_per_policy_step"] = res["A_ms_median"] - res["B_ms_median"]
        res["speedup"] = res["A_ms_median"] / res["B_ms_median"]
    print(json.dumps(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    sys.exit(0 if res["bitwise_equal"] else 3)


if __name__ == "__main__":
    main()
