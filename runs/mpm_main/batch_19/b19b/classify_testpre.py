"""Class of test_iteration_cap_repeats on the old pin, from its pytest log.

    python3 classify_testpre.py PYTEST_LOG RC     -> prints 0 (passed), 1 (failed on the counter phase mismatch), 2 (other)

Warp's error reads "Warp deterministic-mode counter phase mismatch: a MuJoCo Warp kernel's slot-counter calls differ
between its two passes". pytest's repr truncation can cut either half (batch_19 lost the first), so either matches.
"""
import pathlib
import sys

log, rc = pathlib.Path(sys.argv[1]).read_text(errors="replace"), int(sys.argv[2])
if rc == 0:
    print(0)
elif "counter phase mismatch" in log or "calls differ between its two passes" in log:
    print(1)
else:
    print(2)
