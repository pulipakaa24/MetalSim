"""Run Newton unit-test modules on a chosen Warp device (Newton only generates CPU/CUDA variants by default).

Replaces Newton's test-device helpers before the test modules are imported, so every ``add_function_test``
also emits a ``metal_0`` variant (tests that filter on ``device.is_cuda`` still skip Metal), then runs only the variants whose name contains
``--device-tag``. Metal runs go through scripts/gpu_run.sh.

    .venv-newton152/bin/python scripts/diagnostics/newton_vbd/run_newton_tests.py --device-tag metal_0 \
        newton.tests.test_solver_vbd newton.tests.test_softbody
"""
import argparse, importlib, sys, unittest

import warp as wp


def iter_tests(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from iter_tests(t)
        else:
            yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="+")
    ap.add_argument("--device-tag", default="metal_0")
    ap.add_argument("-k", default="", help="substring filter on the test id")
    ap.add_argument("--untagged-on", default="", help="also run tests without a device suffix, with this default device")
    a = ap.parse_args()
    wp.config.quiet = True
    from newton.tests import unittest_utils
    unittest_utils.test_mode = "all"
    # modules also ask for mode="basic" or for CUDA devices explicitly: offer metal:0 as the GPU device
    gpus = [d for d in wp.get_devices() if getattr(d, "is_metal", False)]
    unittest_utils.get_test_devices = lambda mode=None: [wp.get_device("cpu")] + gpus
    for name in ("get_selected_cuda_test_devices", "get_cuda_test_devices"):
        if hasattr(unittest_utils, name):
            setattr(unittest_utils, name, lambda mode=None: list(gpus))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in a.modules:
        mod = importlib.import_module(name)
        for t in iter_tests(loader.loadTestsFromModule(mod)):
            tid = t.id()
            untagged = not any(tag in tid for tag in ("cpu", "metal_0", "cuda_"))
            if a.k in tid and (a.device_tag in tid or (a.untagged_on and untagged)):
                suite.addTest(t)
    if a.untagged_on:
        wp.set_device(a.untagged_on)
    print(f"{suite.countTestCases()} tests selected", flush=True)
    r = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    print(f"RESULT ran={r.testsRun} failures={len(r.failures)} errors={len(r.errors)} skipped={len(r.skipped)}", flush=True)
    for t, tb in r.failures + r.errors:
        print("FAILED", t.id(), "::", tb.strip().splitlines()[-1][:400], flush=True)


if __name__ == "__main__":
    main()
