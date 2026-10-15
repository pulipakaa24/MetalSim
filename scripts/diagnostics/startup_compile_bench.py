"""Startup (kernel compilation) time of a scene, serial vs parallel, with a cold or a warm Warp kernel cache.

One measurement is one fresh process (``run``); ``matrix`` runs the set the report needs, one process after another,
and stops at the first anomaly (a failed run, a preload error, results or cache files that differ from the serial
baseline). Research and design: research_notes/kernel_compile/PARALLEL_COMPILE.md.

    python scripts/diagnostics/startup_compile_bench.py matrix --scene g1 --out runs/parallel_compile/g1
    python scripts/diagnostics/startup_compile_bench.py report runs/parallel_compile/g1 runs/parallel_compile/box_sand

Scenes: ``g1`` = the G1 velocity task's flat-terrain BatchSim (4,096 worlds, its solver settings, 4 substeps), built
and captured as the task builds it; ``box_sand`` = MPMCoupling's box on a sand bed (BatchSim + the vendored MPM
solver, 2 worlds, captured coupled steps; bitwise repeatable run to run). A cold cache is a fresh, empty Warp kernel
cache directory created under the output directory (``WARP_CACHE_PATH``); the user's own cache is never touched.
Apple's Metal shader cache cannot be pointed elsewhere: the first run of a matrix (``prime``) also fills it, and is
reported apart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

T_START = time.perf_counter()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOAD_RE = re.compile(r"^Module (\S+) (?:[0-9a-f]{7} )?load on device '([^']+)' took ([0-9.]+) ms\s+"
                     r"\((compiled|cached|error)\)")


# ---------------------------------------------------------------------------------------------------------------------
# one measurement (a fresh process)


def _hash_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(a.tobytes())
    return h.hexdigest()[:20]


def _run_g1(a, t):
    import numpy as np

    from metalsim.learn.g1_velocity import build_g1_model
    from metalsim.physics.batch import BatchSim, BatchSimOptions

    model, _ = build_g1_model("flat")
    t["model_s"] = time.perf_counter() - t["_t"]
    # the G1 velocity task's BatchSim on flat terrain (g1_velocity.G1VelocityTask): decimation 4, provable capacities
    opt = BatchSimOptions(substeps=4, njmax="auto", nconmax="auto", solver_iterations=10, ls_iterations=20,
                          preload_workers=a.workers, deterministic=a.deterministic)
    t0 = time.perf_counter()
    sim = BatchSim(model, a.worlds, options=opt)
    t["build_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sim.step()
    sim.synchronize()
    t["first_step_s"] = time.perf_counter() - t0
    for _ in range(a.steps - 1):
        sim.step()
    sim.synchronize()
    qpos, qvel = sim.d.qpos.numpy(), sim.d.qvel.numpy()
    out = {"result_hash": _hash_arrays(qpos, qvel), "finite": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
           "preload": sim.preload_stats}
    np.savez_compressed(os.path.join(a.out, "state.npz"), qpos=qpos, qvel=qvel)
    return out


def _run_mjcf(a, t):
    import mujoco
    import numpy as np

    from metalsim.physics.batch import BatchSim, BatchSimOptions

    model = mujoco.MjModel.from_xml_path(a.scene.split(":", 1)[1])
    t["model_s"] = time.perf_counter() - t["_t"]
    t0 = time.perf_counter()
    sim = BatchSim(model, a.worlds, options=BatchSimOptions(substeps=2, preload_workers=a.workers,
                                                            deterministic=a.deterministic))
    t["build_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sim.step()
    sim.synchronize()
    t["first_step_s"] = time.perf_counter() - t0
    for _ in range(a.steps - 1):
        sim.step()
    sim.synchronize()
    qpos, qvel = sim.d.qpos.numpy(), sim.d.qvel.numpy()
    np.savez_compressed(os.path.join(a.out, "state.npz"), qpos=qpos, qvel=qvel)
    return {"result_hash": _hash_arrays(qpos, qvel), "finite": bool(np.isfinite(qpos).all()),
            "preload": sim.preload_stats}


def _load_probe(name):
    import importlib.util

    path = os.path.join(ROOT, "metalsim", "physics", "mpm", "probes", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_probe_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_box_sand(a, t):
    import mujoco
    import numpy as np

    from metalsim.physics.batch import BatchSim, BatchSimOptions
    from metalsim.physics.mpm.coupled import GridEmitter, MPMConfig, MPMCoupling, MPMMaterial
    B = _load_probe("box_on_sand")

    model = mujoco.MjModel.from_xml_string(B.XML)
    t["model_s"] = time.perf_counter() - t["_t"]
    t0 = time.perf_counter()
    sim = BatchSim(model, a.worlds, options=BatchSimOptions(substeps=B.ARM_SUBSTEPS, preload_workers=a.workers))
    cfg = MPMConfig(**B.mpm_settings(0.02), max_active_cell_count="auto", capture=True, proxy_mass="effective",
                    mode="staggered", mpm_substeps=1, preload_workers=a.workers)
    mpm = MPMCoupling(sim, cfg)
    mpm.add_emitter(GridEmitter(lower=B.BED_LO, upper=B.BED_HI, material=MPMMaterial(**B.SAND)))
    mpm.finalize()
    t["build_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    mpm.step()          # eager: sizes the solver's buffers, compiles the MPM modules
    mpm.step()          # captured
    sim.synchronize()
    t["first_step_s"] = time.perf_counter() - t0
    for _ in range(a.steps - 2):
        mpm.step()
    sim.synchronize()
    pos, qvel, wrench = sim.d.xpos.numpy(), sim.d.qvel.numpy(), mpm.wrenches()
    particles = mpm.particles()
    out = {"result_hash": _hash_arrays(pos, qvel, wrench, particles),
           "finite": bool(np.isfinite(particles).all() and np.isfinite(pos).all()),
           "preload": {"batchsim": sim.preload_stats, "mpm_manifest": mpm.preload_stats}}
    np.savez_compressed(os.path.join(a.out, "state.npz"), pos=pos, qvel=qvel, wrench=wrench, particles=particles)
    return out


def cmd_run(a) -> int:
    t = {"_t": time.perf_counter()}
    import warp as wp

    wp.init()   # deterministic runs: BatchSimOptions(deterministic=True), MetalSim's own mode switch
    salt = os.environ.get("METALSIM_BENCH_METAL_SALT")
    if salt:
        # A comment prepended to every Metal source this process compiles, so Apple's own shader cache
        # (DARWIN_USER_CACHE_DIR/com.apple.metal: libraries*.data = MSL -> AIR, functions*.data = AIR -> GPU code,
        # written during the 2026-10-15 matrix) cannot match it: the compile is cold for Apple too, not only for
        # Warp. Applied after Warp read its cache file, so Warp's cache bytes are unchanged.
        core = wp._src.context.runtime.core
        load = core.wp_metal_load_library
        prefix = f"// metalsim bench salt {salt}\n".encode()
        core.wp_metal_load_library = lambda ordinal, source: load(ordinal, prefix + source)
    import mujoco_warp

    import metalsim
    paths = {"warp": wp.__file__, "mujoco_warp": mujoco_warp.__file__, "metalsim": metalsim.__file__}
    if a.expect_warp and not os.path.realpath(wp.__file__).startswith(os.path.realpath(a.expect_warp)):
        raise SystemExit(f"warp imported from {wp.__file__}, expected under {a.expect_warp}")
    t["import_s"] = time.perf_counter() - T_START
    t["_t"] = time.perf_counter()
    os.makedirs(a.out, exist_ok=True)
    run = _run_g1 if a.scene == "g1" else _run_box_sand if a.scene == "box_sand" else _run_mjcf
    out = run(a, t)
    t.pop("_t")
    t["process_s"] = time.perf_counter() - T_START
    res = {"scene": a.scene, "workers": a.workers, "worlds": a.worlds, "steps": a.steps,
           "deterministic": a.deterministic, "cache": os.environ.get("WARP_CACHE_PATH"),
           "maximize": os.environ.get("WP_METAL_MAXIMIZE_CONCURRENT_COMPILATION", "1") != "0",
           "kernel_cache_dir": wp.config.kernel_cache_dir, "paths": paths, **{k: round(v, 4) for k, v in t.items()},
           **out}
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("RESULT " + json.dumps(res))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# matrix driver


def _parse_loads(log_path: str) -> dict:
    n = {"compiled": 0, "cached": 0, "error": 0}
    ms = {"compiled": 0.0, "cached": 0.0, "error": 0.0}
    with open(log_path, errors="replace") as f:
        for line in f:
            m = LOAD_RE.match(line.strip())
            if m:
                n[m.group(4)] += 1
                ms[m.group(4)] += float(m.group(3))
    return {"n_compiled": n["compiled"], "n_cached": n["cached"], "n_error": n["error"],
            "sum_compiled_ms": round(ms["compiled"], 1), "sum_cached_ms": round(ms["cached"], 1)}


def _cache_files(cache: str) -> dict[str, str]:
    """relative path -> sha256 of every generated source / metadata file of a Warp kernel cache."""
    out = {}
    for root, _, files in os.walk(cache):
        for name in files:
            if name.endswith((".metal", ".metal.meta", ".meta", ".cpp")):
                full = os.path.join(root, name)
                with open(full, "rb") as f:
                    out[os.path.relpath(full, cache)] = hashlib.sha256(f.read()).hexdigest()
    return out


def compare_caches(serial: str, other: str) -> dict:
    """Every file both caches hold must be byte-identical; files only one side has are listed (a lazy run can compile
    a module twice when kernels are added to it after its first load; a preload compiles the final version once)."""
    a, b = _cache_files(serial), _cache_files(other)
    common = sorted(set(a) & set(b))
    differ = [p for p in common if a[p] != b[p]]
    return {"common": len(common), "differ": differ, "only_serial": sorted(set(a) - set(b)),
            "only_other": sorted(set(b) - set(a))}


class Matrix:
    def __init__(self, a):
        self.a = a
        self.out = os.path.abspath(a.out)
        os.makedirs(self.out, exist_ok=True)
        self.results = os.path.join(self.out, "results.jsonl")
        self.caches = os.path.join(self.out, "caches")
        os.makedirs(self.caches, exist_ok=True)
        self.manifests = os.path.join(self.out, "manifests")
        self.baseline_hash = {}

    @staticmethod
    def succeeded(rec: dict) -> bool:
        """Only a run that exited 0 and produced a result hash counts as done."""
        return rec.get("rc") == 0 and bool(rec.get("result_hash"))

    def done_ok(self, tag: str) -> bool:
        done = os.path.join(self.out, "runs", tag, "record.json")
        if not os.path.exists(done):
            return False
        with open(done) as f:
            return self.succeeded(json.load(f))

    def clear_run(self, run_dir: str) -> None:
        path = os.path.realpath(run_dir)
        if not path.startswith(os.path.realpath(os.path.join(self.out, "runs")) + os.sep):
            raise SystemExit(f"[bench] refusing to delete {path}: not one of this matrix's run directories")
        shutil.rmtree(path, ignore_errors=True)

    def cache_dir(self, tag: str) -> str:
        """This matrix's own kernel cache for ``tag`` (under the output directory, never the user's cache)."""
        return os.path.join(self.caches, tag)

    def wipe(self, path: str) -> None:
        path = os.path.realpath(path)
        if not path.startswith(os.path.realpath(self.caches) + os.sep):
            raise SystemExit(f"[bench] refusing to delete {path}: not one of this matrix's caches")
        shutil.rmtree(path, ignore_errors=True)

    def run(self, tag: str, workers: int, cache: str, temp: str, maximize: bool = True, deterministic: bool = False,
            reuse: bool = False, salt: bool = False) -> dict:
        run_dir = os.path.join(self.out, "runs", tag)
        done = os.path.join(run_dir, "record.json")
        if reuse and os.path.exists(done):           # resumable: a finished, successful run is not repeated
            with open(done) as f:
                rec = json.load(f)
            if self.succeeded(rec):
                print(f"[bench] {tag:28s} (done before, reused)", flush=True)
                self.check(rec)
                return rec
            print(f"[bench] {tag:28s} failed before (rc={rec.get('rc')}): its run directory is cleared, run again",
                  flush=True)
        self.clear_run(run_dir)                       # nothing from an earlier, unfinished or failed attempt survives
        os.makedirs(run_dir, exist_ok=True)
        if temp.startswith(("cold", "apple_cold")):
            self.wipe(cache)                          # a cold run starts from an empty cache, also on a resume
        os.makedirs(cache, exist_ok=True)
        # PYTHONHASHSEED fixed: Warp's codegen orders one loop prologue by string hash (Adjoint.
        # _promote_mutated_iter_constants iterates a set), so without it two lazy runs already differ in
        # mujoco_warp._src.forward's Metal text (scripts/diagnostics/codegen_determinism_cpu.py); with it the byte
        # comparison sees what preloading changes, nothing else
        env = dict(os.environ, WARP_CACHE_PATH=cache, METALSIM_COMPILE_MANIFEST_DIR=self.manifests,
                   METALSIM_PRELOAD_WORKERS="0", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        if not maximize:
            env["WP_METAL_MAXIMIZE_CONCURRENT_COMPILATION"] = "0"
        if salt:
            env["METALSIM_BENCH_METAL_SALT"] = f"{tag}-{time.time_ns()}"   # unique per attempt
        cmd = [sys.executable, os.path.abspath(__file__), "run", "--scene", self.a.scene, "--workers", str(workers),
               "--worlds", str(self.a.worlds), "--steps", str(self.a.steps), "--out", run_dir]
        if self.a.expect_warp:
            cmd += ["--expect-warp", self.a.expect_warp]
        if deterministic:
            cmd.append("--deterministic")
        log = os.path.join(run_dir, "log.txt")
        t0 = time.perf_counter()
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=self.a.run_timeout).returncode
        wall = time.perf_counter() - t0
        rec = {"tag": tag, "temp": temp, "rc": rc, "wall_s": round(wall, 3), **_parse_loads(log)}
        if rc == 0:
            with open(os.path.join(run_dir, "result.json")) as f:
                rec.update(json.load(f))
        with open(self.results, "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(done, "w") as f:
            json.dump(rec, f)
        print(f"[bench] {tag:28s} rc={rc} wall {wall:6.1f} s  build {rec.get('build_s', float('nan')):6.1f}  "
              f"first {rec.get('first_step_s', float('nan')):6.1f}  compiled {rec['n_compiled']:3d} "
              f"cached {rec['n_cached']:3d}  hash {rec.get('result_hash')}", flush=True)
        self.check(rec)
        return rec

    def check(self, rec: dict) -> None:
        """Stop at the first anomaly."""
        problems = []
        if rec["rc"] != 0:
            problems.append(f"exit code {rec['rc']} (log: runs/{rec['tag']}/log.txt)")
        if rec.get("finite") is False:
            problems.append("non-finite state")
        if rec.get("n_error"):
            problems.append(f"{rec['n_error']} module load errors")
        pre = rec.get("preload")
        stats = []
        if isinstance(pre, list):
            stats = [s for s in pre if s]
        elif isinstance(pre, dict):
            for v in pre.values():
                stats += [s for s in (v if isinstance(v, list) else [v]) if s]
        for s in stats:
            if s.get("errors") or s.get("error"):
                problems.append(f"preload errors: {s}")
        key = (rec.get("deterministic"),)
        if self.a.repeatable or rec.get("deterministic"):
            ref = self.baseline_hash.setdefault(key, rec.get("result_hash"))
            if rec.get("result_hash") != ref:
                problems.append(f"result hash {rec.get('result_hash')} != serial baseline {ref}")
        if problems:
            raise SystemExit(f"[bench] ANOMALY in {rec['tag']}: " + "; ".join(problems))

    def check_cache(self, serial: str, other: str, tag: str) -> None:
        c = compare_caches(serial, other)
        with open(os.path.join(self.out, "cache_compare.jsonl"), "a") as f:
            f.write(json.dumps({"tag": tag, **c}) + "\n")
        print(f"[bench] cache {tag}: {c['common']} files byte-identical check, {len(c['differ'])} differ, "
              f"{len(c['only_serial'])} only in serial, {len(c['only_other'])} only in {tag}", flush=True)
        if c["differ"] or not c["common"]:
            raise SystemExit(f"[bench] ANOMALY: cache files differ from the serial build: {c['differ'][:5]}")


def cmd_matrix(a) -> int:
    m = Matrix(a)
    # (workers, maximize): the shouldMaximizeConcurrentCompilation variant is gone - this Mac reports 16 concurrent
    # compile tasks with and without it (build step of the 2026-10-14 batch), so it cannot change anything here
    variants = [(0, True), (1, True), (2, True), (4, True), (8, True)]

    def vtag(w, mx):
        return f"w{w}" + ("" if mx else "_nomax")

    # 0. manifest (MPMCoupling's first eager step): written by one parallel run, read by the later ones
    if a.scene == "box_sand":
        m.run("manifest_prime", 8, m.cache_dir("manifest_prime"), "cold_manifest", reuse=True)
    # 1. prime: the first compile of these sources on this machine fills Apple's shader cache (reported apart)
    prime_cache = m.cache_dir("prime_w0")
    m.run("prime_w0", 0, prime_cache, "cold_prime", reuse=True)
    # 2. cold Warp cache, interleaved repetitions
    serial_caches = []
    for rep in range(a.reps):
        for w, mx in variants:
            tag = f"cold_{vtag(w, mx)}_r{rep}"
            cache = m.cache_dir(tag)
            reused = m.done_ok(tag)
            m.run(tag, w, cache, "cold", maximize=mx, reuse=True)
            if w == 0:
                serial_caches.append(cache)
            elif not reused:
                m.check_cache(serial_caches[-1], cache, tag)
                if not a.keep_caches:
                    m.wipe(cache)
    # 3. warm Warp cache (the primed one)
    for rep in range(a.reps):
        for w in (0, 2, 4, 8):
            m.run(f"warm_w{w}_r{rep}", w, prime_cache, "warm", reuse=True)
    # 4. deterministic mode (G1: the default mode is not bitwise repeatable run to run; this pair is)
    if a.scene == "g1" and a.deterministic_check:
        d0, d8 = m.cache_dir("det_cold_w0"), m.cache_dir("det_cold_w8")
        m.run("det_cold_w0", 0, d0, "cold_det", deterministic=True, reuse=True)
        m.run("det_cold_w8", 8, d8, "cold_det", deterministic=True, reuse=True)
        if os.path.isdir(d0) and os.path.isdir(d8):
            m.check_cache(d0, d8, "det_cold_w8")
    # 5. cold for Apple too: every source salted, so neither Warp's cache nor Apple's shader cache helps. Rows 2-4's
    # "cold" runs only empty Warp's cache; Apple's (filled by the prime) still served them, and the det_cold_w8 run
    # met sources det_cold_w0 had just compiled. Alternating order, a fresh Warp cache per run.
    if a.apple_cold:
        order = [(0, 4, 8), (8, 4, 0)]
        serial = None
        for rep in range(a.reps):
            for w in order[rep % 2]:
                tag = f"apple_cold_w{w}_r{rep}"
                cache = m.cache_dir(tag)
                reused = m.done_ok(tag)
                m.run(tag, w, cache, "apple_cold", reuse=True, salt=True)
                if w == 0:
                    serial = cache
                elif serial and not reused and os.path.isdir(serial):
                    m.check_cache(serial, cache, tag)
    if not a.keep_caches:
        for name in os.listdir(m.caches):
            m.wipe(os.path.join(m.caches, name))
    print(f"[bench] MATRIX_DONE {a.scene}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# report


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def cmd_report(a) -> int:
    lines = []
    for d in a.dirs:
        recs = [json.loads(line) for line in open(os.path.join(d, "results.jsonl")) if line.strip()]
        # the last record of each run tag, successful runs only (a failed attempt that was rerun stays in the log)
        last = {}
        for r in recs:
            last[r["tag"]] = r
        recs = [r for r in last.values() if Matrix.succeeded(r)]
        if not recs:
            continue
        scene = recs[0]["scene"]
        lines.append(f"\n### {scene} ({recs[0]['worlds']} worlds)\n")
        lines.append("| cache | workers | runs | process wall [s] | build [s] | first step [s] | modules compiled | "
                     "sum of module load timers [s] | speed-up vs serial |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        groups = {}
        for r in recs:
            tag = r["tag"]
            if tag.startswith(("cold_", "warm_", "det_", "prime_", "manifest_", "apple_cold_")):
                temp = r["temp"]
                label = "w" + str(r["workers"]) + ("" if r.get("maximize", True) else " (no maximize)")
                groups.setdefault((temp, label, r["workers"]), []).append(r)
        for temp in ("cold_prime", "cold", "warm", "cold_det", "apple_cold"):
            base = None
            for (t, label, w), rs in sorted(((k, v) for k, v in groups.items() if k[0] == temp),
                                            key=lambda kv: (kv[0][2], kv[0][1])):
                wall = _median([r["process_s"] for r in rs])
                if w == 0 and base is None:
                    base = wall
                sp = f"{base / wall:.2f}x" if base and wall else "-"
                comp = _median([r["n_compiled"] for r in rs])
                lines.append(f"| {t} | {label} | {len(rs)} | {wall:.1f} | {_median([r['build_s'] for r in rs]):.1f} | "
                             f"{_median([r['first_step_s'] for r in rs]):.1f} | {comp:.0f} | "
                             f"{_median([r['sum_compiled_ms'] + r['sum_cached_ms'] for r in rs]) / 1000:.1f} | {sp} |")
        cc = os.path.join(d, "cache_compare.jsonl")
        if os.path.exists(cc):
            comps = [json.loads(line) for line in open(cc) if line.strip()]
            nd = sum(len(c["differ"]) for c in comps)
            lines.append(f"\nKernel cache files compared with the serial build: {len(comps)} runs, "
                         f"{sum(c['common'] for c in comps)} files, {nd} differ.")
        tasks = set()
        for r in recs:
            pre = r.get("preload")
            for st in (pre if isinstance(pre, list) else list((pre or {}).values())):
                for x in (st if isinstance(st, list) else [st]):
                    if x and x.get("metal_compile_tasks") is not None:
                        tasks.add((r.get("maximize", True), x["metal_compile_tasks"]))
        if tasks:
            lines.append("MTLDevice.maximumConcurrentCompilationTaskCount seen by the preloads: "
                         + ", ".join(f"{n} ({'maximized' if mx else 'device default'})" for mx, n in sorted(tasks)))
        hashes = {}
        for r in recs:
            hashes.setdefault(bool(r.get("deterministic")), set()).add(r.get("result_hash"))
        lines.append("Distinct result hashes: " + ", ".join(f"{'deterministic' if k else 'default mode'}: {len(v)}"
                                                           for k, v in hashes.items()))
    text = "\n".join(lines) + "\n"
    print(text)
    if a.write:
        with open(a.write, "w") as f:
            f.write(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--scene", required=True, help="g1, box_sand or mjcf:<path> (a BatchSim of that model)")
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--worlds", type=int, default=None)
    r.add_argument("--steps", type=int, default=10)
    r.add_argument("--deterministic", action="store_true")
    r.add_argument("--expect-warp", default=None, help="fail unless warp is imported from under this directory")
    r.add_argument("--out", required=True)
    mx = sub.add_parser("matrix")
    mx.add_argument("--scene", choices=["g1", "box_sand"], required=True)
    mx.add_argument("--out", required=True)
    mx.add_argument("--worlds", type=int, default=None)
    mx.add_argument("--steps", type=int, default=10)
    mx.add_argument("--reps", type=int, default=2)
    mx.add_argument("--expect-warp", default=None)
    mx.add_argument("--run-timeout", type=float, default=900)
    mx.add_argument("--keep-caches", action="store_true")
    mx.add_argument("--no-deterministic-check", dest="deterministic_check", action="store_false")
    mx.add_argument("--apple-cold", action="store_true", help="add salted runs, cold for Apple's shader cache too")
    rp = sub.add_parser("report")
    rp.add_argument("dirs", nargs="+")
    rp.add_argument("--write", default=None)
    a = ap.parse_args()
    if getattr(a, "worlds", None) is None and a.cmd in ("run", "matrix"):
        a.worlds = 4096 if a.scene == "g1" else 2 if a.scene == "box_sand" else 16
    if a.cmd == "matrix":
        a.repeatable = a.scene == "box_sand"     # bitwise repeatable run to run in its default mode (test_mpm_main)
        return cmd_matrix(a)
    if a.cmd == "report":
        return cmd_report(a)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
