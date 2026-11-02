"""E0: does a command buffer that waits on an MTLSharedEvent time out if the event is signalled late?

PyTorch side: tb.wait_event(ev, v) encodes the wait on torch's current command buffer (as BatchSim.after does at the
end of every PPO rollout), a copy is encoded behind it, the buffer is committed, the host sleeps X s and then signals
the event from the CPU. If the command buffer was killed, the copy never ran and y keeps its -1 sentinel.

Warp side (--warp): wm.wait(ev, v) then a kernel, same protocol; Warp's synchronize reports command buffer errors.

usage: e0_wait_timeout.py [--warp] X1 X2 ...   (seconds)
"""
import sys
import time

import torch
import warp as wp

wp.config.quiet = True
wp.init()
from metalsim.interop import torch_bridge as tb  # noqa: E402
from metalsim.interop import warp_metal as wm  # noqa: E402

args = sys.argv[1:]
use_warp = "--warp" in args
waits = [float(a) for a in args if a != "--warp"]
ev = wm.SharedEvent("metal:0", "e0")


@wp.kernel
def double(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = 2.0 * x[i]


n = 1 << 20
if not use_warp:
    x = torch.arange(n, device="mps", dtype=torch.float32)
    ref = x * 2
    torch.mps.synchronize()
    for X in waits:
        y = torch.full_like(x, -1.0)
        torch.mps.synchronize()
        v = ev.next_value()
        tb.wait_event(ev, v)
        y.copy_(x * 2)
        tb.commit()
        t0 = time.perf_counter()
        time.sleep(X)
        ev.obj.setSignaledValue_(v)
        torch.mps.synchronize()
        el = time.perf_counter() - t0
        ok = bool(torch.equal(y, ref))
        sentinel = bool((y == -1).all())
        print(f"[E0 torch] wait {X:5.1f} s: output correct={ok} untouched(-1)={sentinel} elapsed {el:.2f} s", flush=True)
else:
    x = wp.array(range(n), dtype=float, device="metal:0")
    for X in waits:
        y = wp.full(n, -1.0, dtype=float, device="metal:0")
        wp.synchronize_device("metal:0")
        v = ev.next_value()
        wm.wait(ev, v)
        wp.launch(double, dim=n, inputs=[x, y], device="metal:0")
        wm.flush()
        t0 = time.perf_counter()
        time.sleep(X)
        ev.obj.setSignaledValue_(v)
        err = None
        try:
            wp.synchronize_device("metal:0")
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        el = time.perf_counter() - t0
        yy = y.numpy()
        print(f"[E0 warp] wait {X:5.1f} s: output correct={bool((yy == 2 * x.numpy()).all())} "
              f"untouched(-1)={bool((yy == -1).all())} raised={err} elapsed {el:.2f} s", flush=True)
