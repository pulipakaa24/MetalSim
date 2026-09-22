"""Warp <-> Metal interop: handles and cross-queue ordering.

Wraps the interop entry points added to Warp's Metal runtime (``warp/native/metal.mm``,
branch ``orchard-interop`` of the innate-inc fork):

* ``buffer_of(array)``: the ``id<MTLBuffer>`` behind a Warp array on ``metal:0`` and the byte
  offset of the array's data inside it. No copy; the array must outlive every user.
* ``signal(event, value)``: commit Warp's pending work and signal ``event`` when it completes.
* ``wait(event, value)``: everything Warp launches afterwards waits for ``event`` >= ``value``.

Events are ``MTLSharedEvent`` objects (PyObjC) so that they also order work across Metal
device instances (PyTorch and Warp may hold different ``id<MTLDevice>`` objects) and could
be shared between processes. Values are monotonically increasing 64-bit integers, as with
CUDA events / Vulkan timeline semaphores.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import warp as wp

try:
    import objc
    import Metal  # noqa: F401  (pyobjc-framework-Metal)
except ImportError as e:  # pragma: no cover
    raise ImportError("orchard.interop needs pyobjc-framework-Metal") from e


def _core():
    core = wp._src.context.runtime.core
    if not getattr(core, "_orchard_interop_bound", False):
        core.wp_metal_device_handle.restype = ctypes.c_void_p
        core.wp_metal_device_handle.argtypes = [ctypes.c_int]
        core.wp_metal_queue_handle.restype = ctypes.c_void_p
        core.wp_metal_queue_handle.argtypes = [ctypes.c_int]
        core.wp_metal_buffer_handle.restype = ctypes.c_void_p
        core.wp_metal_buffer_handle.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        core.wp_metal_event_handle.restype = ctypes.c_void_p
        core.wp_metal_event_handle.argtypes = [ctypes.c_int]
        core.wp_metal_event_value.restype = ctypes.c_uint64
        core.wp_metal_event_value.argtypes = [ctypes.c_int]
        core.wp_metal_signal_event.restype = ctypes.c_int
        core.wp_metal_signal_event.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64]
        core.wp_metal_wait_event.restype = ctypes.c_int
        core.wp_metal_wait_event.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64]
        core._orchard_interop_bound = True
    return core


def _ordinal(device) -> int:
    dev = wp.get_device(device)
    if not getattr(dev, "is_metal", False):
        raise ValueError(f"{dev} is not a Metal device")
    return dev.metal_ordinal


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: {wp._src.context.runtime.get_error_string()}")


# -- object <-> pointer -------------------------------------------------------------------

def to_objc(ptr: int):
    """PyObjC proxy for an unretained Objective-C object pointer (id)."""
    if not ptr:
        return None
    return objc.objc_object(c_void_p=ptr)


def objc_ptr(obj) -> int:
    """Raw pointer (id) of a PyObjC object."""
    return objc.pyobjc_id(obj)


# -- handles --------------------------------------------------------------------------------

def metal_device(device="metal:0"):
    """The ``id<MTLDevice>`` Warp uses for ``device`` (PyObjC)."""
    return to_objc(_core().wp_metal_device_handle(_ordinal(device)))


def metal_queue(device="metal:0"):
    """Warp's ``id<MTLCommandQueue>`` on ``device`` (PyObjC)."""
    return to_objc(_core().wp_metal_queue_handle(_ordinal(device)))


@dataclass(frozen=True)
class BufferView:
    buffer: object       # PyObjC id<MTLBuffer>
    buffer_ptr: int      # raw id<MTLBuffer> pointer (what PyTorch's MPS allocator stores as data pointer)
    offset: int          # byte offset of the array's first element inside the buffer
    nbytes: int          # bytes covered by the array (capacity from its first element)


def buffer_of(array: wp.array) -> BufferView:
    """The Metal buffer behind a Warp array on a Metal device. Zero-copy; keep ``array`` alive."""
    ordinal = _ordinal(array.device)
    offset = ctypes.c_size_t(0)
    ptr = _core().wp_metal_buffer_handle(ordinal, ctypes.c_void_p(array.ptr), ctypes.byref(offset))
    if not ptr:
        raise RuntimeError("array memory is not a Metal allocation of this device")
    return BufferView(buffer=to_objc(ptr), buffer_ptr=ptr, offset=int(offset.value),
                      nbytes=int(array.capacity))


# -- ordering -------------------------------------------------------------------------------

class SharedEvent:
    """A monotonically increasing timeline (MTLSharedEvent) that producers signal and consumers wait on."""

    def __init__(self, device="metal:0", label: str = "orchard"):
        dev = metal_device(device)
        self.obj = dev.newSharedEvent()
        self.obj.setLabel_(label)
        self.ptr = objc_ptr(self.obj)
        self.value = 0  # last value handed out by next_value()

    def next_value(self) -> int:
        self.value += 1
        return self.value

    @property
    def signaled_value(self) -> int:
        return int(self.obj.signaledValue())

    def wait_host(self, value: int, timeout_ms: int = 60_000) -> bool:
        """Block the CPU until the event reaches ``value``. Only for rollout boundaries and tests."""
        return bool(self.obj.waitUntilSignaledValue_timeoutMS_(value, timeout_ms))


def signal(event: SharedEvent, value: int, device="metal:0") -> None:
    """Commit Warp's pending work on ``device``; ``event`` reaches ``value`` when it completes."""
    _check(_core().wp_metal_signal_event(_ordinal(device), ctypes.c_void_p(event.ptr), value), "wp_metal_signal_event")


def wait(event: SharedEvent, value: int, device="metal:0") -> None:
    """Order all later Warp work on ``device`` after ``event`` reaches ``value``."""
    _check(_core().wp_metal_wait_event(_ordinal(device), ctypes.c_void_p(event.ptr), value), "wp_metal_wait_event")


def flush(device="metal:0") -> None:
    """Commit Warp's open command buffer without waiting."""
    _check(_core().wp_metal_flush(_ordinal(device)), "wp_metal_flush")


def chain_event(device="metal:0"):
    """Warp's own command-buffer chaining event (MTLEvent) and the value its latest commit signals.

    Waiting for this pair on another queue orders that queue after everything Warp has committed
    so far; ``flush()`` first if the work of interest is still in the open command buffer.
    """
    core = _core()
    o = _ordinal(device)
    return to_objc(core.wp_metal_event_handle(o)), int(core.wp_metal_event_value(o))
