"""run_to.py SECONDS cmd...: run a command with a wall-clock limit (macOS has no coreutils timeout); prints the
exit status or TIMEOUT, never fails the calling script."""
import subprocess, sys, time
t0 = time.time()
try:
    r = subprocess.run(sys.argv[2:], timeout=float(sys.argv[1]))
    print(f"[run_to] exit {r.returncode} after {time.time() - t0:.0f} s: {' '.join(sys.argv[2:])[-160:]}", flush=True)
except subprocess.TimeoutExpired:
    print(f"[run_to] TIMEOUT after {sys.argv[1]} s: {' '.join(sys.argv[2:])[-160:]}", flush=True)
