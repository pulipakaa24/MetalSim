"""unittest with Warp's default device set first (Kamino's own tests run on the default device):
ut_dev.py DEVICE module..."""
import sys
import unittest

import warp as wp

wp.set_device(sys.argv[1])
unittest.main(module=None, argv=["ut_dev", "-v", *sys.argv[2:]])
