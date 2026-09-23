"""PyTorch MPS side of the Metal bridge.

Two ways to get a zero-copy MPS tensor over a Metal buffer:

* ``tensor_from_buffer`` uses the native extension (``at::from_blob`` with device MPS).
* ``tensor_from_dlpack`` builds a DLPack capsule with device ``kDLMetal`` whose data pointer is
  the ``id<MTLBuffer>`` and whose ``byte_offset`` is the array offset, the convention PyTorch's
  ``DLConvertor.cpp`` uses for MPS in both directions (and MLX uses too). Pure Python.

And the ordering primitives on PyTorch's command buffer: ``wait_event`` before the first use of
a producer's output, ``signal_event`` after the last kernel that writes for a consumer.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch
import warp as wp

from . import warp_metal

_ext = None


def ext():
    """The JIT-built native extension (first call compiles it; cached under ~/.cache/torch_extensions)."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        src = Path(__file__).resolve().parents[1] / "native" / "torch_metal_bridge.mm"
        _ext = load(
            name="metalsim_torch_metal_bridge",
            sources=[str(src)],
            extra_cflags=["-ObjC++", "-fobjc-arc", "-Wno-unused-command-line-argument"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            verbose=bool(os.environ.get("ORCHARD_VERBOSE_BUILD")),
        )
    return _ext


# -- dtype maps -------------------------------------------------------------------------------

_WP_TO_TORCH = {
    wp.float32: torch.float32, wp.float16: torch.float16, wp.int32: torch.int32, wp.int64: torch.int64,
    wp.uint8: torch.uint8, wp.int8: torch.int8, wp.uint32: torch.int32, wp.int16: torch.int16,
    wp.uint16: torch.int16, wp.bool: torch.bool,
}


def _torch_dtype_and_shape(a: wp.array):
    """Torch dtype and the (shape, strides) in elements for a Warp array, flattening vector/matrix dtypes."""
    dtype = a.dtype
    shape = list(a.shape)
    strides = [s // wp.types.type_size_in_bytes(dtype) for s in a.strides]
    scalar = getattr(dtype, "_wp_scalar_type_", None)
    if scalar is not None:  # vec/mat: append the component dims, contiguous
        comp_shape = list(dtype._shape_)
        n = int(np.prod(comp_shape))
        strides = [s * n for s in strides]
        inner = []
        acc = 1
        for d in reversed(comp_shape):
            inner.insert(0, acc)
            acc *= d
        shape += comp_shape
        strides += inner
        dtype = scalar
    if dtype not in _WP_TO_TORCH:
        raise TypeError(f"no torch dtype for {dtype}")
    return _WP_TO_TORCH[dtype], shape, strides


# -- zero-copy tensors -------------------------------------------------------------------------

def tensor_from_buffer(buffer_ptr: int, byte_offset: int, shape, strides, dtype: torch.dtype, keepalive) -> torch.Tensor:
    return ext().tensor_from_mtlbuffer(int(buffer_ptr), int(byte_offset), list(shape), list(strides), dtype, keepalive)


def mps_tensor(a: wp.array, via: str = "blob") -> torch.Tensor:
    """Zero-copy MPS tensor aliasing a Warp array on ``metal:0``.

    ``via="blob"`` uses the native extension; ``via="dlpack"`` uses a kDLMetal DLPack capsule.
    The tensor keeps the Warp array alive. Ordering between Warp and PyTorch is the caller's job
    (see ``wait_event`` / ``signal_event``); nothing here synchronizes.
    """
    view = warp_metal.buffer_of(a)
    dtype, shape, strides = _torch_dtype_and_shape(a)
    if via == "blob":
        return tensor_from_buffer(view.buffer_ptr, view.offset, shape, strides, dtype, a)
    if via == "dlpack":
        return torch.from_dlpack(_DLPackMetal(view.buffer_ptr, view.offset, shape, strides, dtype, a))
    raise ValueError(via)


# DLPack (versioned) structures, per dlpack.h v1.x
class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32), ("dtype", _DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)), ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]


class _DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class _DLManagedTensorVersioned(ctypes.Structure):
    pass


_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensorVersioned))
_DLManagedTensorVersioned._fields_ = [("version", _DLPackVersion), ("manager_ctx", ctypes.c_void_p),
                                      ("deleter", _DELETER), ("flags", ctypes.c_uint64), ("dl_tensor", _DLTensor)]

_KDL_METAL = 8
_DL_CODES = {torch.float32: (2, 32), torch.float16: (2, 16), torch.int32: (0, 32), torch.int64: (0, 64),
             torch.uint8: (1, 8), torch.int8: (0, 8), torch.int16: (0, 16), torch.bool: (6, 8)}

ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

_live = {}  # id(managed) -> (managed struct, arrays, keepalive) until the consumer's deleter runs


class _DLPackMetal:
    """Object with ``__dlpack__`` producing a versioned capsule over a Metal buffer."""

    def __init__(self, buffer_ptr, byte_offset, shape, strides, dtype, keepalive):
        self.buffer_ptr, self.byte_offset, self.shape, self.strides, self.dtype = buffer_ptr, byte_offset, shape, strides, dtype
        self.keepalive = keepalive

    def __dlpack_device__(self):
        return (_KDL_METAL, 0)

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        if max_version is not None and max_version[0] < 1:
            raise RuntimeError("only versioned DLPack capsules are produced")
        n = len(self.shape)
        shape = (ctypes.c_int64 * n)(*self.shape)
        strides = (ctypes.c_int64 * n)(*self.strides)
        m = _DLManagedTensorVersioned()
        m.version.major, m.version.minor = 1, 1
        m.flags = 0
        code, bits = _DL_CODES[self.dtype]
        t = m.dl_tensor
        t.data = self.buffer_ptr
        t.device.device_type, t.device.device_id = _KDL_METAL, 0
        t.ndim, t.dtype.code, t.dtype.bits, t.dtype.lanes = n, code, bits, 1
        t.shape, t.strides, t.byte_offset = shape, strides, self.byte_offset

        def deleter(ptr):
            _live.pop(ctypes.addressof(ptr.contents), None)

        m.deleter = _DELETER(deleter)
        m.manager_ctx = None
        _live[ctypes.addressof(m)] = (m, shape, strides, m.deleter, self.keepalive)
        return ctypes.pythonapi.PyCapsule_New(ctypes.byref(m), b"dltensor_versioned", None)


# -- ordering ---------------------------------------------------------------------------------

def wait_event(event: warp_metal.SharedEvent, value: int) -> None:
    """PyTorch kernels encoded after this call run after ``event`` reaches ``value``."""
    ext().wait_event(event.ptr, int(value))


def signal_event(event: warp_metal.SharedEvent, value: int) -> None:
    """Commit PyTorch's pending kernels; ``event`` reaches ``value`` when they complete."""
    ext().signal_event(event.ptr, int(value))


def commit() -> None:
    ext().commit()


def torch_device_ptr() -> int:
    return int(ext().device_ptr())
