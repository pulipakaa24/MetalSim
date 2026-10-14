"""Which deterministic launch trips Warp's counter flag in a CAPTURED G1 deterministic BatchSim?

Under graph capture the host cannot check the flag between launches, so after every deterministic Metal launch recorded
into a graph this probe records one more tiny (non-deterministic) kernel that copies Warp's sticky deterministic flag
into a per-launch slot and clears it. The slots are read after the replays: the first non-zero slot names the launch
(kernel, module, capture order) whose passes set the flag (bit 1 counter overflow, bit 2 phase mismatch). Uncaptured
launches are checked directly after a synchronize, as det_mismatch_probe.py does.

    python -P det_capture_probe.py --worlds 1024 --frames 20 --json out.json
"""
import argparse
import ctypes
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

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
    ap.add_argument("--worlds", type=int, default=1024)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--slots", type=int, default=1 << 20)
    ap.add_argument("--json")
    a = ap.parse_args()
    from warp._src import metal_deterministic as md

    dev = wp.get_device("metal:0")
    snaps = wp.zeros(a.slots, dtype=wp.int32, device=dev)
    dummy = wp.zeros(1, dtype=wp.int32, device=dev)
    wp.launch(_snapshot_flag, dim=1, inputs=[dummy, snaps, 0], device=dev)      # build the module before any capture
    wp.synchronize_device(dev)
    snaps.zero_()
    recorded = []          # slot -> (kernel, module, dim)
    eager = {"launches": 0, "tripped": None}
    orig = md.launch

    def launch(kernel, hooks, params, bounds, device, *args, **kw):
        orig(kernel, hooks, params, bounds, device, *args, **kw)
        state = md._state(device)
        if device.is_capturing:
            slot = len(recorded)
            if slot >= a.slots:
                return
            recorded.append((kernel.key, kernel.module.name, int(bounds.size)))
            wp.launch(_snapshot_flag, dim=1, inputs=[state.overflow, snaps, slot], device=device)
            return
        eager["launches"] += 1
        wp.synchronize_device(device)
        flag = ctypes.c_int32.from_address(state.overflow.ptr).value
        if flag and eager["tripped"] is None:
            eager["tripped"] = {"flag": flag, "kernel": kernel.key, "module": kernel.module.name, "dim": int(bounds.size),
                                "eager_launch": eager["launches"]}
            ctypes.c_int32.from_address(state.overflow.ptr).value = 0

    md.launch = launch
    from metalsim.learn.g1_velocity import G1VelocityTask, benchmark_step

    task = G1VelocityTask(a.worlds, terrain="flat", batch_options={"deterministic": True})
    benchmark_step(task, num_frames=a.frames, warmup=2, capture=True)
    task.sim.synchronize()
    s = snaps.numpy()[: len(recorded)]
    hits = [i for i in range(len(recorded)) if s[i]]
    kernels_hit = {}
    for i in hits:
        k = recorded[i][0]
        kernels_hit[k] = kernels_hit.get(k, 0) + 1
    res = {"worlds": a.worlds, "recorded_det_launches": len(recorded), "eager": eager,
           "first_hit": None if not hits else {"slot": hits[0], "flag": int(s[hits[0]]), "kernel": recorded[hits[0]][0],
                                                "module": recorded[hits[0]][1], "dim": recorded[hits[0]][2]},
           "hit_slots": len(hits), "kernels_hit": kernels_hit,
           "captured_kernels": sorted({r[0] for r in recorded})}
    print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "captured_kernels"}))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
