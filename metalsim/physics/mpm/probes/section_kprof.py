"""Per-section, per-kernel GPU time of the Franka-Pour MPM step (WP_METAL_PROFILE=1, eager): Newton's section
timers are replaced by a context manager that synchronizes and collects Warp's Metal profile report at each section
boundary, so every kernel is attributed to the section that launched it. Command-buffer overhead per dispatch is
estimated with an empty one-thread kernel and reported separately.
    sec_kprof.py WORLDS STEPS [MODE] [JSON]"""
import contextlib, ctypes, json, os, re, sys
from collections import defaultdict
W = int(sys.argv[1]); STEPS = int(sys.argv[2]); MODE = sys.argv[3] if len(sys.argv) > 3 else "eager"
OUT = sys.argv[4] if len(sys.argv) > 4 else None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [HERE + "/pour_bitwise.py", "profile", "--worlds", str(W), "--mode", MODE]
sys.path.insert(0, HERE)
import numpy as np
import pour_bitwise as pb
sp, wp = pb.sp, pb.wp
from warp._src.context import runtime
core = runtime.core
core.wp_metal_profile_report.restype = ctypes.c_char_p


def report():
    rows = []
    for line in core.wp_metal_profile_report().decode().splitlines():
        m = re.match(r"\s*([0-9.]+) ms\s+(\d+)\s+(.*)$", line)
        if m:
            rows.append((m.group(3), float(m.group(1)), int(m.group(2))))
    return rows


@wp.kernel
def _empty(a: wp.array[float]):
    i = wp.tid()
    if i < 0:
        a[0] = 1.0


tot = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
stack = ["(outside)"]


def collect(name):
    wp.synchronize_device()
    for k, ms, c in report():
        e = tot[name][k]
        e[0] += ms; e[1] += c


@contextlib.contextmanager
def timer(name):
    collect(stack[-1])
    stack.append(name)
    try:
        yield
    finally:
        collect(name)
        stack.pop()


with wp.ScopedDevice(pb.DEV):
    model, solver, src, bpw = pb.make_solver()
    state = model.state()
    rng = np.random.default_rng(1)
    phase = wp.array((rng.random(W) * 1.5 - 0.6).astype(np.float32), dtype=float)
    body_idx = wp.array(np.arange(W, dtype=np.int32) * bpw + src, dtype=int)
    base = wp.array(np.asarray(model.body_q.numpy())[body_idx.numpy(), :3].astype(np.float32), dtype=wp.vec3)
    t = [0.0]
    def step():
        wp.launch(sp.drive_source_cup, dim=W, inputs=[t[0], phase, body_idx, base, state.body_q, state.body_qd])
        solver.step(state, state, None, None, sp.MPM_DT)
        t[0] += sp.MPM_DT
    for _ in range(int(round(0.6 / sp.MPM_DT))):
        step()
    wp.synchronize_device(); report()
    solver._timer = timer
    for _ in range(STEPS):
        step()
        collect("(outside)")
    a = wp.zeros(1, dtype=float)
    for _ in range(20): wp.launch(_empty, dim=1, inputs=[a])
    wp.synchronize_device(); report()
    for _ in range(500): wp.launch(_empty, dim=1, inputs=[a])
    wp.synchronize_device()
    cal = [r for r in report() if "_empty" in r[0]]
    ov = cal[0][1] / cal[0][2] if cal else 0.0
    res = {"worlds": W, "steps": STEPS, "empty_cb_ms": ov, "sections": {}}
    for sec, ks in tot.items():
        rows = sorted(((v[0] / STEPS, v[1] / STEPS, k) for k, v in ks.items()), reverse=True)
        raw = sum(r[0] for r in rows); n = sum(r[1] for r in rows)
        res["sections"][sec] = {"raw_ms": raw, "launches": n, "net_ms": raw - n * ov, "kernels": rows}
    print(f"empty command buffer {ov*1e3:.1f} us")
    for sec, d in sorted(res["sections"].items(), key=lambda kv: -kv[1]["net_ms"]):
        print(f"SEC {d['net_ms']:8.2f} net {d['raw_ms']:8.2f} raw {d['launches']:7.1f} launches  {sec}")
        for ms, c, k in d["kernels"][:8]:
            print(f"      {ms - c*ov:8.3f} net {ms:8.3f} raw {c:6.1f}  {k[:80]}")
    if OUT:
        json.dump(res, open(OUT, "w"), indent=1)
