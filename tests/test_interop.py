"""WS3 acceptance: zero-copy MPS tensors over Warp Metal arrays and event ordering across queues."""
import time

import numpy as np
import pytest
import torch
import warp as wp

from orchard.interop import torch_bridge as tb
from orchard.interop import warp_metal as wm

pytestmark = pytest.mark.skipif(not wp.is_metal_available() or not torch.backends.mps.is_available(),
                                reason="needs Metal Warp and torch MPS")
DEV = "metal:0"


@wp.kernel
def fill(a: wp.array(dtype=float), v: float):
    i = wp.tid()
    a[i] = v


@wp.kernel
def fill_i(a: wp.array(dtype=float), v: int):
    i = wp.tid()
    a[i] = float(v) + float(i) * 0.5


@wp.kernel
def copy_scaled(src: wp.array(dtype=float), dst: wp.array(dtype=float), s: float):
    i = wp.tid()
    dst[i] = src[i] * s


def test_handles():
    dev = wm.metal_device(DEV)
    assert dev is not None and "Apple" in str(dev.name())
    q = wm.metal_queue(DEV)
    assert q is not None
    tdev = tb.torch_device_ptr()
    print("warp device", wm.objc_ptr(dev), "torch device", tdev, "same object:", wm.objc_ptr(dev) == tdev)


@pytest.mark.parametrize("via", ["blob", "dlpack"])
def test_zero_copy_alias(via):
    n = 4096
    a = wp.zeros(n, dtype=float, device=DEV)
    wp.synchronize_device(DEV)  # zeros() is an async memset on Warp's queue: order it before torch's writes
    t = tb.mps_tensor(a, via=via)
    assert t.device.type == "mps" and t.shape == (n,) and t.dtype == torch.float32
    # torch writes, warp reads
    t.fill_(3.0)
    torch.mps.synchronize()
    assert np.all(a.numpy() == 3.0)
    # warp writes, torch reads
    wp.launch(fill, dim=n, inputs=[a, 7.0], device=DEV)
    wp.synchronize_device(DEV)
    assert torch.all(t == 7.0).item()
    # a slice of the array maps to the right offset
    sub = a[100:200]
    ts = tb.mps_tensor(sub, via=via)
    assert ts.shape == (100,)
    ts.fill_(-1.0)
    torch.mps.synchronize()
    v = a.numpy()
    assert np.all(v[100:200] == -1.0) and np.all(v[:100] == 7.0) and np.all(v[200:] == 7.0)


def test_vec_dtype_and_2d():
    a = wp.zeros((16, 3), dtype=wp.vec3, device=DEV)
    wp.synchronize_device(DEV)
    t = tb.mps_tensor(a)
    assert tuple(t.shape) == (16, 3, 3)
    t[:, 1, 2] = 5.0
    torch.mps.synchronize()
    assert np.all(a.numpy()[:, 1, 2] == 5.0)


def test_tensor_keeps_array_alive():
    a = wp.zeros(256, dtype=float, device=DEV)
    wp.synchronize_device(DEV)
    t = tb.mps_tensor(a)
    del a
    import gc
    gc.collect()
    t.fill_(2.0)
    torch.mps.synchronize()
    assert torch.all(t == 2.0).item()


def test_torch_buffer_readable_by_warp():
    """Reverse direction: a torch MPS tensor's buffer handed to Warp (via host alias of unified memory)."""
    t = torch.arange(1024, dtype=torch.float32, device="mps")
    buf, off = tb.ext().mtlbuffer_of(t)
    assert buf != 0 and off == 0
    t2 = t[10:20]
    buf2, off2 = tb.ext().mtlbuffer_of(t2)
    assert buf2 == buf and off2 == 40


def test_wait_blocks_torch_until_signaled():
    """A wait encoded on torch's command buffer really holds the GPU: signal from the CPU releases it."""
    ev = wm.SharedEvent(DEV, "cpu-gate")
    t = torch.zeros(1 << 20, device="mps")
    tb.wait_event(ev, 1)
    t.add_(1.0)
    tb.commit()
    time.sleep(0.2)
    assert ev.signaled_value == 0
    t0 = time.perf_counter()
    import threading
    threading.Timer(0.3, lambda: ev.obj.setSignaledValue_(1)).start()
    torch.mps.synchronize()
    assert time.perf_counter() - t0 > 0.25, "torch did not wait for the event"
    assert torch.all(t == 1.0).item()


def test_wait_blocks_warp_until_signaled():
    ev = wm.SharedEvent(DEV, "cpu-gate-warp")
    a = wp.zeros(1 << 20, dtype=float, device=DEV)
    wm.wait(ev, 1, DEV)
    wp.launch(fill, dim=a.shape[0], inputs=[a, 4.0], device=DEV)
    wm.flush(DEV)
    time.sleep(0.2)
    import threading
    t0 = time.perf_counter()
    threading.Timer(0.3, lambda: ev.obj.setSignaledValue_(1)).start()
    wp.synchronize_device(DEV)
    assert time.perf_counter() - t0 > 0.25, "warp did not wait for the event"
    assert np.all(a.numpy() == 4.0)


def test_event_ordering_warp_to_torch():
    """Warp produces into a double buffer, torch consumes; back-pressure both ways, no host sync in the loop."""
    n = 1 << 16
    bufs = [wp.zeros(n, dtype=float, device=DEV) for _ in range(2)]
    wp.synchronize_device(DEV)
    ts = [tb.mps_tensor(b) for b in bufs]
    produced = wm.SharedEvent(DEV, "produced")
    consumed = wm.SharedEvent(DEV, "consumed")
    sums = []
    iters = 200
    for it in range(iters):
        b = it % 2
        if it >= 2:
            wm.wait(consumed, it - 1, DEV)      # torch finished reading this buffer's previous contents
        wp.launch(fill_i, dim=n, inputs=[bufs[b], it], device=DEV)
        wm.signal(produced, it + 1, DEV)
        tb.wait_event(produced, it + 1)
        sums.append(ts[b].sum())
        tb.signal_event(consumed, it + 1)
    torch.mps.synchronize()
    expect = np.array([n * it + 0.5 * (n * (n - 1) / 2) for it in range(iters)])
    got = torch.stack(sums).cpu().double().numpy()
    np.testing.assert_allclose(got, expect, rtol=1e-6)


def test_event_ordering_torch_to_warp():
    """torch produces into a double buffer, Warp consumes; results checked once at the end."""
    n = 1 << 16
    bufs = [wp.zeros(n, dtype=float, device=DEV) for _ in range(2)]
    out = wp.zeros((200, n), dtype=float, device=DEV)
    wp.synchronize_device(DEV)
    ts = [tb.mps_tensor(b) for b in bufs]
    produced = wm.SharedEvent(DEV, "produced-t")
    consumed = wm.SharedEvent(DEV, "consumed-w")
    for it in range(200):
        b = it % 2
        if it >= 2:
            tb.wait_event(consumed, it - 1)
        ts[b].fill_(float(it))
        ts[b].add_(1.0)
        tb.signal_event(produced, it + 1)
        wm.wait(produced, it + 1, DEV)
        wp.launch(copy_scaled, dim=n, inputs=[bufs[b], out[it], 2.0], device=DEV)
        wm.signal(consumed, it + 1, DEV)
    wp.synchronize_device(DEV)
    res = out.numpy()
    for it in range(200):
        assert np.all(res[it] == 2.0 * (it + 1)), it


def test_ping_pong_round_trip():
    """Both directions every iteration: the shape of a rollout step (sim -> policy -> sim)."""
    n = 1 << 20
    obs = wp.zeros(n, dtype=float, device=DEV)
    act = wp.zeros(n, dtype=float, device=DEV)
    wp.synchronize_device(DEV)
    t_obs, t_act = tb.mps_tensor(obs), tb.mps_tensor(act)
    e_sim = wm.SharedEvent(DEV, "sim")
    e_pol = wm.SharedEvent(DEV, "policy")
    wp.launch(fill, dim=n, inputs=[obs, 1.0], device=DEV)
    steps = 300
    t0 = time.perf_counter()
    for _ in range(steps):
        vs = e_sim.next_value()
        wm.signal(e_sim, vs, DEV)
        tb.wait_event(e_sim, vs)
        t_act.copy_(t_obs * 0.5 + 0.5)          # "policy"
        vp = e_pol.next_value()
        tb.signal_event(e_pol, vp)
        wm.wait(e_pol, vp, DEV)
        wp.launch(copy_scaled, dim=n, inputs=[act, obs, 2.0], device=DEV)  # "physics": obs = 2*act = obs+1
    wp.synchronize_device(DEV)
    dt = time.perf_counter() - t0
    assert np.allclose(obs.numpy(), 1.0 + steps)
    print(f"round trip: {dt / steps * 1e3:.3f} ms/step host time, {steps / dt:.0f} steps/s")
