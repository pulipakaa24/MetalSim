"""Which deterministic launch trips Warp's counter tripwire (or overflows) in a G1 deterministic BatchSim?

Runs the G1 flat velocity task with ``BatchSimOptions(deterministic=True, capture=False)`` and the env step uncaptured,
synchronizes after every deterministic Metal launch and reads Warp's sticky deterministic flag (bit 1 counter
overflow, bit 2 phase mismatch); the first launch that sets it is reported with its kernel and module.

    python -P det_mismatch_probe.py --worlds 64 --frames 40 --json out.json
"""
import argparse
import ctypes
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    from warp._src import metal_deterministic as md

    seen = {"launches": 0, "kernels": {}}
    orig = md.launch

    def launch(kernel, hooks, params, bounds, device, *args, **kw):
        orig(kernel, hooks, params, bounds, device, *args, **kw)
        if device.is_capturing:
            return
        seen["launches"] += 1
        seen["kernels"][kernel.key] = seen["kernels"].get(kernel.key, 0) + 1
        wp.synchronize_device(device)
        flag = ctypes.c_int32.from_address(md._state(device).overflow.ptr).value
        if flag:
            res = {"flag": flag, "phase_mismatch": bool(flag & 2), "overflow": bool(flag & 1), "kernel": kernel.key,
                   "module": kernel.module.name, "source": getattr(getattr(kernel, "func", None), "__module__", "?"),
                   "launch_number": seen["launches"], "dim": int(bounds.size)}
            print("TRIPPED " + json.dumps(res), flush=True)
            if a.json:
                json.dump({**res, "kernels_launched": seen["kernels"]}, open(a.json, "w"), indent=1)
            sys.exit(3)

    md.launch = launch
    from metalsim.learn.g1_velocity import G1VelocityTask, benchmark_step

    task = G1VelocityTask(a.worlds, terrain="flat", batch_options={"deterministic": True, "capture": False})
    benchmark_step(task, num_frames=a.frames, warmup=0, capture=False)
    task.sim.synchronize()
    res = {"flag": 0, "launches": seen["launches"], "kernels_launched": seen["kernels"]}
    print("CLEAN " + json.dumps({"launches": seen["launches"], "kernels": len(seen["kernels"])}))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
