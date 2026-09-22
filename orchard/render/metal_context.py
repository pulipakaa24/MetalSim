"""Metal device context shared with Warp: queue, shader compilation, buffers, textures, ordering.

Uses PyObjC. The device object is the one Warp's Metal runtime uses, so Metal events order the
renderer against physics (Warp queue) and the learner (PyTorch's queue).
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import objc
import Metal
from Foundation import NSString

from orchard.interop import warp_metal as wm

SHADER_DIR = Path(__file__).resolve().parent / "shaders"

# PyObjC bug #690 over-retains instance new* methods before 12.2.3; the leak is bounded (resources
# here live for the renderer's lifetime) and the metadata workaround breaks argument marshalling, so
# it is not applied.


class MetalContext:
    def __init__(self, device="metal:0"):
        self.warp_device = device
        self.device = wm.metal_device(device)
        self.queue = self.device.newCommandQueue()
        self.queue.setLabel_("orchard.render")
        self._libraries = {}
        self._pipelines = {}
        self.event = wm.SharedEvent(device, "orchard.render")
        self._pending = []   # command buffers committed but not known to be complete

    # -- shaders ------------------------------------------------------------------------------------

    def library(self, name: str, defines: dict | None = None):
        key = (name, tuple(sorted((defines or {}).items())))
        lib = self._libraries.get(key)
        if lib is None:
            src = (SHADER_DIR / f"{name}.metal").read_text()
            if defines:
                src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + src
            opts = Metal.MTLCompileOptions.new()
            opts.setFastMathEnabled_(True)
            lib, err = self.device.newLibraryWithSource_options_error_(src, opts, None)
            if lib is None:
                raise RuntimeError(f"Metal compile failed for {name}: {err}")
            self._libraries[key] = lib
        return lib

    def compute_pipeline(self, lib, fn: str):
        key = ("c", id(lib), fn)
        p = self._pipelines.get(key)
        if p is None:
            f = lib.newFunctionWithName_(fn)
            if f is None:
                raise RuntimeError(f"kernel {fn} not found")
            p, err = self.device.newComputePipelineStateWithFunction_error_(f, None)
            if p is None:
                raise RuntimeError(f"pipeline {fn}: {err}")
            self._pipelines[key] = p
        return p

    def render_pipeline(self, desc):
        p, err = self.device.newRenderPipelineStateWithDescriptor_error_(desc, None)
        if p is None:
            raise RuntimeError(f"render pipeline: {err}")
        return p

    # -- resources ----------------------------------------------------------------------------------

    def buffer(self, nbytes: int, data: np.ndarray | None = None, label: str = ""):
        nbytes = max(int(nbytes), 16)
        if data is not None:
            data = np.ascontiguousarray(data)
            nbytes = max(data.nbytes, 16)
        buf = self.device.newBufferWithLength_options_(nbytes, Metal.MTLResourceStorageModeShared)
        if buf is None:
            raise RuntimeError(f"buffer allocation of {nbytes} bytes failed")
        if data is not None:
            self.write_buffer(buf, data)
        if label:
            buf.setLabel_(label)
        return buf

    @staticmethod
    def buffer_array(buf, dtype, shape) -> np.ndarray:
        """Zero-copy numpy view of a shared buffer (synchronize before reading GPU results)."""
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        mv = buf.contents().as_buffer(n)
        return np.frombuffer(mv, dtype=dtype).reshape(shape)

    def write_buffer(self, buf, data: np.ndarray, offset: int = 0):
        data = np.ascontiguousarray(data)
        mv = buf.contents().as_buffer(offset + data.nbytes)
        mv[offset:offset + data.nbytes] = data.tobytes()

    def texture2d(self, width, height, pixel_format, usage, array_length=1, label="", storage=None):
        d = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            pixel_format, width, height, False)
        if array_length > 1:
            d.setTextureType_(Metal.MTLTextureType2DArray)
            d.setArrayLength_(array_length)
        d.setUsage_(usage)
        d.setStorageMode_(storage if storage is not None else Metal.MTLStorageModePrivate)
        t = self.device.newTextureWithDescriptor_(d)
        if t is None:
            raise RuntimeError("texture allocation failed")
        if label:
            t.setLabel_(label)
        return t

    def upload_texture(self, tex, image: np.ndarray, slice_index: int = 0):
        """image: (H, W, C) uint8 (C == 4) or float32."""
        h, w = image.shape[:2]
        bpr = w * image.shape[2] * image.dtype.itemsize
        region = Metal.MTLRegion(Metal.MTLOrigin(0, 0, 0), Metal.MTLSize(w, h, 1))
        tex.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage_(
            region, 0, slice_index, np.ascontiguousarray(image).tobytes(), bpr, bpr * h)

    def sampler(self, linear=True, repeat=True):
        d = Metal.MTLSamplerDescriptor.new()
        f = Metal.MTLSamplerMinMagFilterLinear if linear else Metal.MTLSamplerMinMagFilterNearest
        d.setMinFilter_(f)
        d.setMagFilter_(f)
        mode = Metal.MTLSamplerAddressModeRepeat if repeat else Metal.MTLSamplerAddressModeClampToEdge
        d.setSAddressMode_(mode)
        d.setTAddressMode_(mode)
        return self.device.newSamplerStateWithDescriptor_(d)

    # -- command buffers and ordering ----------------------------------------------------------------

    def command_buffer(self):
        cb = self.queue.commandBuffer()
        self._pending = [p for p in self._pending if p.status() < Metal.MTLCommandBufferStatusCompleted]
        return cb

    def wait_for(self, cb, event: wm.SharedEvent, value: int):
        cb.encodeWaitForEvent_value_(event.obj, value)

    def signal(self, cb) -> int:
        v = self.event.next_value()
        cb.encodeSignalEvent_value_(self.event.obj, v)
        return v

    def commit(self, cb):
        cb.commit()
        self._pending.append(cb)

    def synchronize(self):
        for cb in self._pending:
            cb.waitUntilCompleted()
            if cb.error() is not None:
                raise RuntimeError(f"Metal command buffer failed: {cb.error()}")
        self._pending.clear()


def objc_buffer_from_ptr(ptr: int):
    return wm.to_objc(ptr)
