"""Run Newton test modules on every device (CPU and Metal). usage: newton_tests.py module [module...]"""
import sys
import unittest

import newton.tests.unittest_utils as uu

uu.test_mode = "all"
suite = unittest.TestSuite()
for n in sys.argv[1:]:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(n))
r = unittest.TextTestRunner(verbosity=2).run(suite)
print(f"RESULT run={r.testsRun} failures={len(r.failures)} errors={len(r.errors)} skipped={len(r.skipped)}")
sys.exit(0 if r.wasSuccessful() else 1)
