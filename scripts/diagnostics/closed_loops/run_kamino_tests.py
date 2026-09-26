"""Run Newton's Kamino unit tests on a chosen Warp device (Kamino's tests use the Warp default device through
their own ``test_context``; Newton's generic device helpers are not involved).

    .venv-newtonfork/bin/python scripts/diagnostics/closed_loops/run_kamino_tests.py --device cpu
    scripts/gpu_run.sh kamino_tests low 30 -- .venv-newtonfork/bin/python scripts/diagnostics/closed_loops/run_kamino_tests.py --device metal:0

Newton 1.7.0.dev keeps them in ``newton.tests.kamino``; 1.5.2 in ``newton._src.solvers.kamino.tests``.
"""
import argparse, importlib, pkgutil, sys, time, unittest

import warp as wp


def iter_tests(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from iter_tests(t)
        else:
            yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-k", default="", help="substring filter on the test id")
    ap.add_argument("--modules", nargs="*", default=None)
    a = ap.parse_args()
    wp.config.quiet = True
    wp.init()
    wp.set_device(a.device)
    try:
        pkg = importlib.import_module("newton.tests.kamino")
    except ImportError:
        pkg = importlib.import_module("newton._src.solvers.kamino.tests")
    pkg.setup_tests(device=a.device)
    names = a.modules or sorted(m.name for m in pkgutil.iter_modules(pkg.__path__) if m.name.startswith("test_"))
    loader = unittest.TestLoader()
    per_mod = {}
    fails = []
    t00 = time.time()
    for name in names:
        suite = unittest.TestSuite()
        try:
            mod = importlib.import_module(f"{pkg.__name__}.{name}")
        except Exception as e:  # noqa: BLE001
            per_mod[name] = f"IMPORT ERROR {type(e).__name__}: {str(e)[:200]}"
            continue
        for t in iter_tests(loader.loadTestsFromModule(mod)):
            if a.k in t.id():
                suite.addTest(t)
        t0 = time.time()
        r = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
        per_mod[name] = (f"ran={r.testsRun} fail={len(r.failures)} err={len(r.errors)} skip={len(r.skipped)} "
                         f"({time.time() - t0:.0f} s)")
        for t, tb in r.failures + r.errors:
            fails.append((t.id(), tb.strip().splitlines()[-1][:400]))
        print(f"MODULE {name}: {per_mod[name]}", flush=True)
    print(f"\n=== device {a.device}, {time.time() - t00:.0f} s")
    for k, v in per_mod.items():
        print(f"{k:55s} {v}")
    for tid, msg in fails:
        print("FAILED", tid, "::", msg)


if __name__ == "__main__":
    main()
