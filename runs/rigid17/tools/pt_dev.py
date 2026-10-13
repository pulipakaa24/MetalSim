"""pytest with Warp's default device set first (Kamino's own tests run on the default device):
pt_dev.py DEVICE pytest-args..."""
import sys

import warp as wp

wp.set_device(sys.argv[1])
import pytest  # noqa: E402

sys.exit(pytest.main(sys.argv[2:]))
