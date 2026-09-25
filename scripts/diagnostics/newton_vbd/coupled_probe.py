"""Newton's coupled rigid/deformable examples (MuJoCo Warp + VBD) headless on a chosen device; CPU vs Metal.

Runs a Newton example module's ``Example`` for N frames with the null viewer and saves the final body and
particle state. Isaac Lab 3.0 EA's coupled tasks use ``SolverCoupledProxy`` (MJWarp source, VBD destination);
``example_mujoco_vbd_coupled_solver`` is Newton's canonical instance of that path, ``example_mujoco_vbd_admm_solver``
the ADMM alternative. Newton source via PYTHONPATH (upstream/newton-1.5.2 = Isaac Lab 3.0 EA pin).

    PYTHONPATH=upstream/newton-1.5.2 .venv-newtonfork/bin/python scripts/diagnostics/newton_vbd/coupled_probe.py \
        --example multiphysics.example_mujoco_vbd_coupled_solver --device cpu --frames 60 --out runs/newton_vbd/x.npz
"""
from __future__ import annotations

import argparse, importlib, json, sys, time, traceback

import numpy as np
import warp as wp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="multiphysics.example_mujoco_vbd_coupled_solver")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--extra", default="", help="extra example args, space separated")
    ap.add_argument("--no-warmup-before-capture", dest="warmup_before_capture", action="store_false")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import newton, newton.examples  # noqa: E401
    rec = {"example": a.example, "device": a.device, "newton_src": newton.__file__.split("/upstream/")[-1].split("/newton/")[0]}
    try:
        mod = importlib.import_module(f"newton.examples.{a.example}")
        if a.warmup_before_capture and hasattr(mod, "_capture_frame_graph"):
            # SolverVBD sizes its contact buffers on the first step; allocating them inside a capture needs a CUDA
            # memory pool (Newton: "Run one uncaptured step ... before capture"). Metal has none, so every device runs
            # one frame eagerly before the capture (the same on CPU, so trajectories stay comparable).
            orig = mod._capture_frame_graph

            def _warm_capture(model, simulate, *, enabled=True, _orig=orig):
                if enabled:
                    simulate()
                return _orig(model, simulate, enabled=enabled)

            mod._capture_frame_graph = _warm_capture
        parser = mod.Example.create_parser()
        argv = ["--viewer", "null", "--device", a.device, "--num-frames", str(a.frames), "--quiet"] + a.extra.split()
        sys.argv = [a.example] + argv
        viewer, args = newton.examples.init(parser)
        t0 = time.perf_counter()
        ex = mod.Example(viewer, args)
        rec["build_s"] = time.perf_counter() - t0
        rec["graph"] = getattr(ex, "graph", None) is not None
        t0 = time.perf_counter()
        ex.step(); wp.synchronize_device()
        rec["first_frame_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(a.frames - 1):
            ex.step()
        wp.synchronize_device()
        rec["loop_s"] = time.perf_counter() - t0
        rec["frames_per_s"] = (a.frames - 1) / rec["loop_s"]
        out = {}
        for k in ("body_q", "particle_q"):
            v = getattr(ex.state_0, k, None)
            if v is not None and v.shape[0]:
                out[k] = v.numpy()
                rec[f"{k}_finite"] = bool(np.isfinite(out[k]).all())
        try:
            ex.test_final(); rec["test_final"] = "pass"
        except Exception as e:  # noqa: BLE001
            rec["test_final"] = f"fail: {e}"
        if a.out:
            np.savez_compressed(a.out, **out)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:6000]
        rec["traceback"] = traceback.format_exc()[-3000:]
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
