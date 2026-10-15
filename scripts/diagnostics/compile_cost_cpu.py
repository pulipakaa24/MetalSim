"""CPU-only measurements behind research_notes/kernel_compile/PARALLEL_COMPILE.md (no Metal device needed).

    python scripts/diagnostics/compile_cost_cpu.py codegen          # serial Python share of a Metal cold load
    python scripts/diagnostics/compile_cost_cpu.py scaling --workers 8   # CPU-target preload, 1/2/4/8 threads

Both record the kernels one G1 flat BatchSim capture body launches (4 substeps, step + forward + reset) with
``wp.preload_launches`` on the CPU device (nothing runs). ``codegen`` then times, per module variant, the module hash
and the Metal-target code generation that ``Module.load`` runs under Warp's codegen lock; ``scaling`` compiles the
same modules for the CPU (LLVM) on N threads. Each run uses a fresh kernel cache under $TMPDIR.
"""
import argparse
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import warp as wp  # noqa: E402

wp.config.kernel_cache_dir = tempfile.mkdtemp(prefix="wp_compile_cost_", dir=os.environ.get("TMPDIR"))
wp.config.quiet = True
wp.init()

import mujoco  # noqa: E402
import mujoco_warp as mjw  # noqa: E402

from metalsim.learn.g1_velocity import build_g1_model  # noqa: E402
from metalsim.physics.batch import _m_layout  # noqa: E402


def record(workers: int, record_only: bool):
    model, _ = build_g1_model("flat")
    wp.config.metal_register_cholesky_max = 48
    mjd = mujoco.MjData(model)
    mujoco.mj_forward(model, mjd)
    with wp.ScopedDevice("cpu"):
        with _m_layout(32):
            m = mjw.put_model(model)
        m.opt.graph_conditional = False
        m.opt.warn_overflow = 0
        m.opt.iterations = 10
        m.opt.ls_iterations = 20
        with _m_layout(32):
            d = mjw.put_data(model, mjd, nworld=4, nconmax=24, njmax=144)
        mask = wp.zeros(4, dtype=wp.bool)
        with wp.preload_launches("cpu", max_workers=workers, kernel_hooks=not record_only) as pl:
            if record_only:
                pl._load = lambda: None
            for _ in range(4):
                mjw.step(m, d)
            mjw.forward(m, d)
            mjw.reset_data(m, d, reset=mask)
    return pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["codegen", "scaling"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "scaling":
        pl = record(a.workers, record_only=False)
        print(f"RESULT workers={a.workers} modules={pl.modules} kernels={pl.kernels} "
              f"discover={pl.discover_seconds:.2f}s load={pl.load_seconds:.2f}s errors={len(pl.errors)}")
        return
    pl = record(0, record_only=True)
    variants = list(pl._variants.values())
    rows, th_sum, tc_sum = [], 0.0, 0.0
    for (module, _device, block_dim), _kernels in variants:
        t0 = time.perf_counter()
        module.get_module_hash(block_dim)
        t1 = time.perf_counter()
        source, *_ = module._run_codegen(module.resolved_options[block_dim] | {"output_arch": None}, "metal")
        t2 = time.perf_counter()
        th_sum += t1 - t0
        tc_sum += t2 - t1
        rows.append((t2 - t0, t1 - t0, t2 - t1, len(source), module.name))
    print(f"discovery {pl.discover_seconds:.3f} s; {len(variants)} module variants, "
          f"{sum(len(k) for _, k in variants)} kernels")
    for tot, th, tc, n, name in sorted(rows, reverse=True)[:15]:
        print(f"{tot * 1e3:8.1f} ms (hash {th * 1e3:6.1f}, codegen {tc * 1e3:7.1f})  {n:8d} B  {name[:70]}")
    print(f"TOTAL hash {th_sum:.2f} s + Metal codegen {tc_sum:.2f} s = {th_sum + tc_sum:.2f} s")


if __name__ == "__main__":
    main()
