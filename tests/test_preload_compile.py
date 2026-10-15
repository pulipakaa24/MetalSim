"""Parallel startup compilation (metalsim.physics.warmup): same compiled kernels and same results as Warp's lazy,
serial loading. Research: research_notes/kernel_compile/PARALLEL_COMPILE.md."""
import hashlib
import json
import os
import subprocess
import sys
import textwrap

import pytest
import warp as wp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "scripts", "diagnostics", "startup_compile_bench.py")
GO1 = os.path.join(ROOT, "upstream", "mujoco_menagerie", "unitree_go1", "scene.xml")

needs_preload = pytest.mark.skipif(not hasattr(wp, "preload_launches"), reason="needs the Warp fork's preload_launches")


def _cache_files(cache):
    out = {}
    for root, _, files in os.walk(cache):
        for name in files:
            if name.endswith((".metal", ".metal.meta", ".meta", ".cpp")):
                with open(os.path.join(root, name), "rb") as f:
                    out[os.path.relpath(os.path.join(root, name), cache)] = hashlib.sha256(f.read()).hexdigest()
    return out


def _bench_run(tmp_path, tag, workers, scene, extra=()):
    cache, out = tmp_path / f"cache_{tag}", tmp_path / f"out_{tag}"
    # one hash seed for both runs: Warp's codegen orders one loop prologue by string hash (see
    # scripts/diagnostics/codegen_determinism_cpu.py), which would differ between any two processes
    env = dict(os.environ, WARP_CACHE_PATH=str(cache), METALSIM_PRELOAD_WORKERS="0", PYTHONHASHSEED="0",
               METALSIM_COMPILE_MANIFEST_DIR=str(tmp_path / "manifests"))
    p = subprocess.run([sys.executable, BENCH, "run", "--scene", scene, "--workers", str(workers), "--steps", "5",
                        "--out", str(out), *extra], env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    with open(out / "result.json") as f:
        return json.load(f), _cache_files(str(cache)), p.stdout


@needs_preload
@pytest.mark.skipif(not wp.is_metal_available(), reason="needs Metal")
@pytest.mark.skipif(not os.path.exists(GO1), reason="needs mujoco_menagerie")
def test_batchsim_preload_same_kernels_and_state(tmp_path):
    """A BatchSim built with parallel preloading (8 threads, cold kernel cache) writes the same kernel cache files byte
    for byte and, with BatchSimOptions(deterministic=True), reaches the same state bit for bit as one built with lazy
    loading; preloading compiles no module the lazy run did not."""
    lazy, lazy_files, lazy_log = _bench_run(tmp_path, "lazy", 0, f"mjcf:{GO1}", ["--deterministic"])
    pre, pre_files, log = _bench_run(tmp_path, "pre", 8, f"mjcf:{GO1}", ["--deterministic"])
    assert lazy["finite"] and pre["finite"]
    assert pre["result_hash"] == lazy["result_hash"]
    common = set(lazy_files) & set(pre_files)
    assert common and set(pre_files) <= set(lazy_files)
    assert all(lazy_files[k] == pre_files[k] for k in common)
    stats = [s for s in pre["preload"] if s]
    assert stats and stats[0]["modules"] > 10 and stats[0]["errors"] == 0 and stats[0]["error"] is None
    assert 0 < log.count("(compiled)") <= lazy_log.count("(compiled)")


MANIFEST_SCRIPT = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, sys.argv[1])
    import warp as wp
    wp.config.quiet = True
    wp.init()
    import pc_manifest_kernels as K
    from metalsim.physics import warmup
    device = wp.get_device(sys.argv[2])
    m = warmup.CompileManifest("test_manifest")
    m.mark(device)
    stats = m.preload(device, 4)
    module = K.scale_kernel.module
    loaded_before = any(ctx == device.context for ctx, _ in module.execs)
    a = wp.ones(8, dtype=float, device=device)
    wp.launch(K.scale_kernel, dim=8, inputs=[a, 2.0], device=device)
    m.record(device)
    print("STATS", stats, "LOADED_BEFORE", loaded_before, "SUM", float(a.numpy().sum()))
    """
)


@needs_preload
def test_compile_manifest_round_trip(tmp_path):
    """CompileManifest records the named modules an eager step loaded, and a later process loads them before the step
    (Metal if available, else the CPU)."""
    (tmp_path / "pc_manifest_kernels.py").write_text(textwrap.dedent(
        """
        import warp as wp

        @wp.kernel
        def scale_kernel(a: wp.array(dtype=float), s: float):
            i = wp.tid()
            a[i] = a[i] * s
        """))
    script = tmp_path / "manifest_script.py"
    script.write_text(MANIFEST_SCRIPT)
    device = "metal:0" if wp.is_metal_available() else "cpu"
    env = dict(os.environ, WARP_CACHE_PATH=str(tmp_path / "cache"), METALSIM_COMPILE_MANIFEST_DIR=str(tmp_path / "m"),
               PYTHONPATH=os.pathsep.join([ROOT, os.environ.get("PYTHONPATH", "")]))
    outs = []
    for _ in range(2):
        p = subprocess.run([sys.executable, str(script), str(tmp_path), device], env=env, capture_output=True,
                           text=True, timeout=600)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
        outs.append(p.stdout.strip().splitlines()[-1])
    assert "STATS None LOADED_BEFORE False SUM 16.0" in outs[0]     # no manifest yet: lazy load at the launch
    assert "LOADED_BEFORE True SUM 16.0" in outs[1] and "'modules': 1" in outs[1]
    with open(tmp_path / "m" / "test_manifest.json") as f:
        entries = json.load(f)["entries"]
    assert [e["module"] for e in entries] == ["pc_manifest_kernels"]
    assert entries[0]["kernels"] == ["scale_kernel"]
