"""Denoising for tier 2 with Intel Open Image Denoise (OIDN 2.x) on its Metal device.

OIDN is the denoiser Blender Cycles uses on Apple GPUs; its Metal backend runs the U-Net on the GPU.
We call the C API through ctypes (the library ships in the ``pyoidn`` wheel, or set ``OIDN_LIB`` to a
``libOpenImageDenoise`` from Homebrew's ``open-image-denoise``), create the device on the renderer's
own MTLCommandQueue (``oidnNewMetalDevice``) and wrap the renderer's MTLBuffers without copies
(``oidnNewSharedBufferFromMetal``): HDR colour (float4 mean radiance), first-hit albedo and normal
as auxiliary images (``hdr`` mode, ``cleanAux`` since the auxiliary buffers are averaged over the
pixel's samples and nearly noise-free).
"""
from __future__ import annotations

import ctypes, glob, os

import objc

FLOAT3, FLOAT4 = 3, 4
QUALITY = {"default": 0, "fast": 4, "balanced": 5, "high": 6}


def _find_lib():
    p = os.environ.get("OIDN_LIB")
    if p:
        return p
    try:
        import pyoidn
        c = glob.glob(os.path.join(os.path.dirname(pyoidn.__file__), "oidn", "lib", "libOpenImageDenoise.2.dylib"))
        if c:
            return c[0]
    except ImportError:
        pass
    for c in ("/opt/homebrew/lib/libOpenImageDenoise.dylib", "/usr/local/lib/libOpenImageDenoise.dylib"):
        if os.path.exists(c):
            return c
    raise ImportError("Open Image Denoise not found: pip install pyoidn, brew install open-image-denoise, or set OIDN_LIB")


_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(_find_lib())
        vp, sz, i, b, f, cp = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_bool, ctypes.c_float, ctypes.c_char_p
        sig = {"oidnIsMetalDeviceSupported": (b, [vp]), "oidnNewMetalDevice": (vp, [ctypes.POINTER(vp), i]),
               "oidnCommitDevice": (None, [vp]), "oidnSyncDevice": (None, [vp]), "oidnReleaseDevice": (None, [vp]),
               "oidnGetDeviceError": (i, [vp, ctypes.POINTER(cp)]), "oidnGetDeviceInt": (i, [vp, cp]),
               "oidnNewSharedBufferFromMetal": (vp, [vp, vp]), "oidnReleaseBuffer": (None, [vp]),
               "oidnNewFilter": (vp, [vp, cp]), "oidnSetFilterImage": (None, [vp, cp, vp, i, sz, sz, sz, sz, sz]),
               "oidnSetFilterBool": (None, [vp, cp, b]), "oidnSetFilterInt": (None, [vp, cp, i]), "oidnSetFilterFloat": (None, [vp, cp, f]),
               "oidnCommitFilter": (None, [vp]), "oidnExecuteFilter": (None, [vp]), "oidnExecuteFilterAsync": (None, [vp]),
               "oidnReleaseFilter": (None, [vp])}
        for n, (r, a) in sig.items():
            fn = getattr(L, n); fn.restype = r; fn.argtypes = a
        _lib = L
    return _lib


def _id(o):
    return ctypes.c_void_p(objc.pyobjc_id(o))


class OIDNDenoiser:
    """One OIDN "RT" filter per environment image over shared MTLBuffers.

    ``color``/``albedo``/``normal``/``output``: (MTLBuffer, byte offset) of (n, h, w, 4|3) float32 images.
    ``denoise()`` encodes on the queue the device was created with, after anything already committed
    there, and (``sync=True``) waits for completion."""

    def __init__(self, queue, n, w, h, color, output, albedo=None, normal=None, color_ch=4, aux_ch=3, quality="high",
                 clean_aux=True, stacked=None):
        L = lib()
        q = (ctypes.c_void_p * 1)(_id(queue).value)
        self.dev = L.oidnNewMetalDevice(q, 1)
        L.oidnCommitDevice(self.dev); self._check()
        self.n, self.w, self.h = n, w, h
        self._bufs = {}
        def shared(key, mb):
            k = objc.pyobjc_id(mb)
            if k not in self._bufs:
                self._bufs[k] = L.oidnNewSharedBufferFromMetal(self.dev, _id(mb)); self._check()
            return self._bufs[k]
        self.filters = []
        # stacked: all env images as one tall image (one filter, one execution; the network's receptive field then
        # crosses env borders near the top/bottom rows of each image). Default: stacked above 16 envs.
        self.stacked = (n > 16) if stacked is None else stacked
        chunks = [(e, 1) for e in range(n)]
        if self.stacked:                  # OIDN limits the image size: stack at most 16384 rows per filter
            per = max(1, 16384 // h)
            chunks = [(e, min(per, n - e)) for e in range(0, n, per)]
        H = h
        for e, k in chunks:
            h = H * k
            f = L.oidnNewFilter(self.dev, b"RT")
            def img(name, spec, ch):
                mb, off = spec
                L.oidnSetFilterImage(f, name, shared(name, mb), FLOAT3, w, h, off + e * w * H * ch * 4, ch * 4, w * ch * 4)   # RT takes float3; a float4 image is float3 with a 16-byte pixel stride
            img(b"color", color, color_ch); img(b"output", output, color_ch)
            if albedo is not None:
                img(b"albedo", albedo, aux_ch)
                if normal is not None:
                    img(b"normal", normal, aux_ch)
            L.oidnSetFilterBool(f, b"hdr", True)
            L.oidnSetFilterBool(f, b"cleanAux", bool(clean_aux and albedo is not None))
            L.oidnSetFilterInt(f, b"quality", QUALITY[quality])
            L.oidnCommitFilter(f); self._check()
            self.filters.append(f)

    def _check(self):
        msg = ctypes.c_char_p()
        err = lib().oidnGetDeviceError(self.dev, ctypes.byref(msg))
        if err:
            raise RuntimeError(f"OIDN error {err}: {msg.value.decode() if msg.value else ''}")

    def denoise(self, sync=True):
        L = lib()
        for f in self.filters:
            L.oidnExecuteFilterAsync(f)
        if sync:
            L.oidnSyncDevice(self.dev)
        self._check()

    def release(self):
        L = lib()
        for f in self.filters: L.oidnReleaseFilter(f)
        for b in self._bufs.values(): L.oidnReleaseBuffer(b)
        L.oidnReleaseDevice(self.dev); self.filters = []; self._bufs = {}

    def __del__(self):
        try:
            if self.filters: self.release()
        except Exception:
            pass
