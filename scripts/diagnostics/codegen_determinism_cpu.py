"""Is Warp's Metal code generation of the G1 BatchSim modules the same in every process, and does preloading change it?

    python scripts/diagnostics/codegen_determinism_cpu.py          # CPU only, ~1 min

Four fresh processes record or run the G1 flat BatchSim launch code (4 substeps, step + forward + reset) on the CPU
device: two with lazy loading (real CPU launches) and two with ``wp.preload_launches`` (recorded, nothing runs), each
pair under two Python hash seeds (PYTHONHASHSEED 1 and 2). Each prints the SHA-256 of the Metal source of every MuJoCo
Warp module variant (module name + module hash). Lazy vs preload under one seed separates a preload bug (different
modules, hashes or source) from process-to-process nondeterminism (lazy vs lazy under two seeds).

Measured 2026-10-15 (research_notes/kernel_compile/PARALLEL_COMPILE.md): lazy and preload are identical under the same
seed; ``mujoco_warp._src.forward`` differs between seeds in both modes. Warp's
``Adjoint._promote_mutated_iter_constants`` iterates a set of variable names, so the order of the ``wp::copy``
prologue of a dynamic loop (and with it the numbering of two variables) follows string-hash randomization: same
program, different text.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def worker(mode):
    sys.path.insert(0, ROOT)
    import warp as wp

    wp.config.kernel_cache_dir = tempfile.mkdtemp(prefix="wp_codegen_det_", dir=os.environ.get("TMPDIR"))
    wp.config.quiet = True
    wp.init()
    import mujoco
    import mujoco_warp as mjw

    from metalsim.learn.g1_velocity import build_g1_model
    from metalsim.physics.batch import _m_layout
    from warp._src.context import user_modules

    model, _ = build_g1_model("flat")
    wp.config.metal_register_cholesky_max = 48
    mjd = mujoco.MjData(model)
    mujoco.mj_forward(model, mjd)
    variants = {}
    with wp.ScopedDevice("cpu"):
        with _m_layout(32):
            m = mjw.put_model(model)
        m.opt.graph_conditional = False
        m.opt.iterations, m.opt.ls_iterations = 10, 20
        with _m_layout(32):
            d = mjw.put_data(model, mjd, nworld=4, nconmax=24, njmax=144)
        mask = wp.zeros(4, dtype=wp.bool)

        def body():
            for _ in range(4):
                mjw.step(m, d)
            mjw.forward(m, d)
            mjw.reset_data(m, d, reset=mask)

        if mode == "preload":
            with wp.preload_launches("cpu", max_workers=0, kernel_hooks=False) as pl:
                pl._load = lambda: None          # record only
                body()
            for (module, _device, block_dim), _kernels in pl._variants.values():
                variants[(module.name, block_dim)] = module
        else:
            body()
            for module in list(user_modules.values()):
                for _ctx, block_dim in list(module.execs):
                    variants[(module.name, block_dim)] = module
    out = {}
    for (name, block_dim), module in variants.items():
        if not name.startswith("mujoco_warp") and "__locals__" not in name:
            continue
        h = module.get_module_hash(block_dim).hex()[:7]
        source, *_ = module._run_codegen(module.resolved_options[block_dim] | {"output_arch": None}, "metal")
        out[f"{name}_{h}"] = hashlib.sha256(source.encode()).hexdigest()[:16]
    print("JSON " + json.dumps(out))


def main():
    runs = {}
    for mode, seed in [("lazy", "1"), ("lazy", "2"), ("preload", "1"), ("preload", "2")]:
        env = dict(os.environ, PYTHONHASHSEED=seed)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "worker", mode], env=env, capture_output=True,
                           text=True)
        line = [x for x in p.stdout.splitlines() if x.startswith("JSON ")]
        if not line:
            raise SystemExit(p.stdout[-2000:] + p.stderr[-2000:])
        runs[(mode, seed)] = json.loads(line[0][5:])

    def cmp(a, b):
        ra, rb = runs[a], runs[b]
        only = sorted(set(ra) ^ set(rb))
        diff = sorted(k for k in set(ra) & set(rb) if ra[k] != rb[k])
        print(f"{a[0]} seed {a[1]} vs {b[0]} seed {b[1]}: {len(set(ra) & set(rb))} module variants in both, "
              f"only in one: {only}, different Metal source: {diff}")
        return only, diff

    print(f"{len(runs[('lazy', '1')])} MuJoCo Warp module variants")
    preload_vs_lazy = cmp(("lazy", "1"), ("preload", "1"))
    cmp(("preload", "2"), ("lazy", "2"))
    cmp(("lazy", "1"), ("lazy", "2"))
    cmp(("preload", "1"), ("preload", "2"))
    sys.exit(1 if any(preload_vs_lazy) else 0)   # only a preload-vs-lazy difference is a preload bug


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "worker":
        worker(sys.argv[2])
    else:
        main()
