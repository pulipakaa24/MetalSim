"""pytest with Newton's test device mode "all" (cpu and metal:0)."""
import sys

import newton.tests.unittest_utils as u

u.test_mode = "all"
import pytest  # noqa: E402

sys.exit(pytest.main(sys.argv[1:]))
