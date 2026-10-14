"""3D Gaussian splat file loaders -> ``GaussianCloud`` (one in-memory layout for every format).

Supported: SPZ v1-v3 (gzip; the open, MIT-licensed SPZ format), the SuperSplat/PlayCanvas compressed .ply (InteriorGS), the original 3DGS
training ``.ply`` (Kerbl et al. 2023: ``f_dc_*``, ``f_rest_*`` channel-major, logit opacity, log scales, wxyz
rotation), antimatter15 ``.splat`` (32 bytes per splat, no SH) and NVIDIA Omniverse NuRec ``.nurec`` (gzip msgpack of
3DGRUT/3DGUT tensors, ``load_nurec``). Adding a format = one function returning a
``GaussianCloud`` registered in ``LOADERS``.

Conventions of ``GaussianCloud`` (= the 3DGS .ply semantics, so the renderer has one code path):
* ``means`` (N, 3) float32, metres, in the file's own frame (no axis flip is applied here; ``SplatScene`` applies
  an explicit affine transform);
* ``log_scales`` (N, 3) float32 (natural log of the per-axis standard deviation);
* ``quats`` (N, 4) float32 **wxyz**, unit length;
* ``opacity`` (N,) float32 in [0, 1] (after the sigmoid);
* ``sh_dc`` (N, 3) float32: degree-0 SH coefficient per channel (colour = 0.5 + C0 * dc + higher bands);
* ``sh_rest`` (N, K, 3), K = (deg + 1)^2 - 1, coefficient-major, channel last. SPZ stores these as uint8 with
  value = (b - 128) / 128; they are kept as uint8 (``sh_rest_u8=True``) so no precision is lost and memory is 1/4.
* ``antialiased``: True / False if the file says how it was trained (SPZ flag 0x1: trained with the Mip-Splatting
  2D filter / gsplat ``rasterize_mode="antialiased"``), None if unknown.

SPZ decode formulas are the open-source (MIT) reference implementation's (the ``spz`` library, src/cc/load-spz.cc
and splat-utils.h; reimplemented in numpy, no code copied): 24-bit signed fixed point positions / 2^fractionalBits; alpha byte / 255 (sigmoid already
applied); colour (b / 255 - 0.5) / 0.15; log scale b / 16 - 10; v3 rotation "smallest three" (2-bit index of the
largest component, 3 x (sign bit + 9-bit magnitude scaled by 1/sqrt2)), v2 rotation xyz = b / 127.5 - 1 with w >= 0;
SH (b - 128) / 128. Positions are stored in the RUB frame and returned as stored.
"""
from __future__ import annotations

import gzip
import os
import struct
from dataclasses import dataclass, field

import numpy as np

SH_C0 = 0.28209479177387814
SPZ_MAGIC = 0x5053474E          # "NGSP"


@dataclass
class GaussianCloud:
    means: np.ndarray
    log_scales: np.ndarray
    quats: np.ndarray
    opacity: np.ndarray
    sh_dc: np.ndarray
    sh_rest: np.ndarray | None = None
    sh_rest_u8: bool = False
    sh_degree: int = 0
    antialiased: bool | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    def sh_rest_float(self, idx=slice(None)) -> np.ndarray:
        """Higher-band SH as float32 (N, K, 3) for rows ``idx``."""
        if self.sh_rest is None:
            return np.zeros((self.means[idx].shape[0], 0, 3), np.float32)
        r = self.sh_rest[idx]
        return (r.astype(np.float32) - 128.0) / 128.0 if self.sh_rest_u8 else r.astype(np.float32)

    def subset(self, idx) -> "GaussianCloud":
        return GaussianCloud(self.means[idx], self.log_scales[idx], self.quats[idx], self.opacity[idx], self.sh_dc[idx],
                             None if self.sh_rest is None else self.sh_rest[idx], self.sh_rest_u8, self.sh_degree,
                             self.antialiased, self.source, dict(self.meta))


def sh_dim(degree: int) -> int:
    return (degree + 1) ** 2 - 1


# ------------------------------------------------------------------------------------------------- SPZ

def _quat_smallest_three(r: np.ndarray) -> np.ndarray:
    """(N, 4) uint8 -> (N, 4) float32 xyzw (load-spz unpackQuaternionSmallestThree)."""
    comp = r[:, 0].astype(np.uint32) | (r[:, 1].astype(np.uint32) << 8) | (r[:, 2].astype(np.uint32) << 16) | (r[:, 3].astype(np.uint32) << 24)
    largest = (comp >> 30).astype(np.int64)
    q = np.zeros((len(r), 4), np.float32)
    mask = np.uint32(511)
    s12 = np.float32(np.sqrt(0.5))
    for i in (3, 2, 1, 0):                         # the reference walks i = 3..0 and skips the largest
        sel = largest != i
        mag = (comp & mask).astype(np.float32)
        neg = ((comp >> 9) & 1).astype(bool)
        v = s12 * mag / np.float32(511.0)
        v = np.where(neg, -v, v)
        q[sel, i] = v[sel]
        comp = np.where(sel, comp >> 10, comp)
    ss = (q * q).sum(1)
    q[np.arange(len(r)), largest] = np.sqrt(np.maximum(0.0, 1.0 - ss))
    return q


def _quat_first_three(r: np.ndarray) -> np.ndarray:
    xyz = r.astype(np.float32) / 127.5 - 1.0
    w = np.sqrt(np.maximum(0.0, 1.0 - (xyz * xyz).sum(1)))
    return np.concatenate([xyz, w[:, None]], 1)


def load_spz(path: str) -> GaussianCloud:
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        data = gzip.decompress(raw)
    elif struct.unpack_from("<I", raw, 0)[0] == SPZ_MAGIC:
        return _load_spz_v4(raw, path)
    else:
        raise ValueError(f"{path}: not an SPZ file")
    del raw
    magic, version, n, deg, fb, flags, _ = struct.unpack_from("<IIIBBBB", data, 0)
    if magic != SPZ_MAGIC:
        raise ValueError(f"{path}: bad SPZ magic {magic:#x}")
    if not 1 <= version <= 3:
        raise ValueError(f"{path}: SPZ version {version} not supported by the gzip path")
    dim = sh_dim(deg) if deg > 0 else 0
    off = 16
    pos_b = 6 if version == 1 else 9
    rot_b = 4 if version >= 3 else 3

    def take(nbytes):
        nonlocal off
        a = np.frombuffer(data, np.uint8, nbytes, off)
        off += nbytes
        return a

    pos = take(n * pos_b); alpha = take(n); col = take(n * 3); scl = take(n * 3); rot = take(n * rot_b); sh = take(n * dim * 3)
    if version == 1:
        means = pos.view(np.float16).reshape(n, 3).astype(np.float32)
    else:
        p = pos.reshape(n * 3, 3).astype(np.int32)
        fixed = p[:, 0] | (p[:, 1] << 8) | (p[:, 2] << 16)
        fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
        means = (fixed.astype(np.float32) / np.float32(1 << fb)).reshape(n, 3)
    rot = rot.reshape(n, rot_b)
    q_xyzw = _quat_smallest_three(rot) if version >= 3 else _quat_first_three(rot)
    quats = np.ascontiguousarray(q_xyzw[:, [3, 0, 1, 2]])
    quats /= np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-12)
    cloud = GaussianCloud(
        means=means,
        log_scales=(scl.reshape(n, 3).astype(np.float32) / 16.0 - 10.0),
        quats=quats.astype(np.float32),
        opacity=(alpha.astype(np.float32) / 255.0),
        sh_dc=((col.reshape(n, 3).astype(np.float32) / 255.0 - 0.5) / 0.15),
        sh_rest=(np.array(sh.reshape(n, dim, 3)) if dim else None),
        sh_rest_u8=True, sh_degree=int(deg), antialiased=bool(flags & 1), source=os.path.abspath(path),
        meta=dict(format="spz", version=int(version), fractional_bits=int(fb), flags=int(flags), frame="RUB (as stored)"))
    return cloud


def _load_spz_v4(raw: bytes, path: str) -> GaussianCloud:
    raise NotImplementedError(f"{path}: SPZ v4 (NGSP/zstd multi-stream) is not implemented; re-save as v3 with the spz tool")


# ------------------------------------------------------------------------------------------------- 3DGS .ply

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
              "short": "<i2", "ushort": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def _ply_elements(path):
    """Binary little-endian PLY -> (header length, [(element name, count, [(prop, dtype)])])."""
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            hdr += line
    lines = hdr.decode("latin1").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"{path}: only binary little-endian PLY is supported")
    els = []
    for l in lines:
        t = l.split()
        if t and t[0] == "element":
            els.append((t[1], int(t[2]), []))
        elif t and t[0] == "property" and els:
            if t[1] == "list":
                raise ValueError(f"{path}: list property in element {els[-1][0]}")
            els[-1][2].append((t[2], _PLY_TYPES[t[1]]))
    return len(hdr), els


def load_ply_compressed(path: str) -> GaussianCloud:
    """SuperSplat / PlayCanvas compressed PLY (``splat-transform``; InteriorGS ``3dgs_compressed.ply``): per 256-splat
    chunk min/max of position, log-scale (and colour); per splat packed uint32 position 11-10-11, rotation 2-10-10-10
    (smallest three of the PLY's (w, x, y, z), index of the largest in the top 2 bits), log-scale 11-10-11, colour
    8-8-8-8 (r, g, b = 0.5 + C0 dc in [0, 1] or the chunk's range, a = opacity); optional ``sh`` element: uint8
    f_rest_i (channel-major), value = b * 8 / 255 - 4. Formulas: PlayCanvas engine (MIT)
    src/scene/gsplat/gsplat-compressed-data.js."""
    off, els = _ply_elements(path)
    arrays = {}
    for name, count, props in els:
        dt = np.dtype(props)
        arrays[name] = np.memmap(path, dtype=dt, mode="r", offset=off, shape=(count,))
        off += dt.itemsize * count
    ch, v = arrays["chunk"], arrays["vertex"]
    n = len(v)
    ci = np.arange(n) // 256
    unorm = lambda x, bits: (x & ((1 << bits) - 1)).astype(np.float32) / float((1 << bits) - 1)
    def u111011(x):
        x = np.asarray(x, np.uint32)
        return np.stack([unorm(x >> 21, 11), unorm(x >> 11, 10), unorm(x, 11)], 1)
    lerp = lambda a, b, t: a * (1 - t) + b * t
    c = lambda k: np.asarray(ch[k], np.float32)[ci]
    pt = u111011(v["packed_position"])
    means = np.stack([lerp(c("min_x"), c("max_x"), pt[:, 0]), lerp(c("min_y"), c("max_y"), pt[:, 1]), lerp(c("min_z"), c("max_z"), pt[:, 2])], 1)
    st = u111011(v["packed_scale"])
    ls = np.stack([lerp(c("min_scale_x"), c("max_scale_x"), st[:, 0]), lerp(c("min_scale_y"), c("max_scale_y"), st[:, 1]),
                   lerp(c("min_scale_z"), c("max_scale_z"), st[:, 2])], 1)
    r = np.asarray(v["packed_rotation"], np.uint32)
    small = np.stack([(unorm(r >> 20, 10) - 0.5) * np.sqrt(2.0), (unorm(r >> 10, 10) - 0.5) * np.sqrt(2.0), (unorm(r, 10) - 0.5) * np.sqrt(2.0)], 1)
    m = np.sqrt(np.maximum(0.0, 1.0 - (small * small).sum(1)))
    largest = (r >> 30).astype(np.int64)
    q = np.zeros((n, 4), np.float32)                              # PLY order (w, x, y, z)
    for L in range(4):
        sel = largest == L
        others = [k for k in range(4) if k != L]
        q[np.ix_(sel, others)] = small[sel]
        q[sel, L] = m[sel]
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
    col = np.asarray(v["packed_color"], np.uint32)
    rgba = np.stack([unorm(col >> 24, 8), unorm(col >> 16, 8), unorm(col >> 8, 8), unorm(col, 8)], 1)
    if "min_r" in ch.dtype.names:
        for j, k in enumerate("rgb"):
            rgba[:, j] = lerp(c(f"min_{k}"), c(f"max_{k}"), rgba[:, j])
    sh_rest, deg = None, 0
    if "sh" in arrays:
        names = sorted(arrays["sh"].dtype.names, key=lambda k: int(k.split("_")[-1]))
        K = len(names) // 3
        deg = {3: 1, 8: 2, 15: 3, 24: 4}[K]
        raw = np.stack([np.asarray(arrays["sh"][k]) for k in names], 1).reshape(n, 3, K)
        sh_rest = (raw.transpose(0, 2, 1).astype(np.float32) * (8.0 / 255.0) - 4.0).astype(np.float16)
    return GaussianCloud(means=means.astype(np.float32), log_scales=ls.astype(np.float32), quats=q, opacity=rgba[:, 3].copy(),
                         sh_dc=((rgba[:, :3] - 0.5) / SH_C0).astype(np.float32), sh_rest=sh_rest, sh_rest_u8=False, sh_degree=deg,
                         antialiased=None, source=os.path.abspath(path), meta=dict(format="ply-compressed (SuperSplat)", chunks=len(ch)))


def load_ply(path: str) -> GaussianCloud:
    if any(name == "chunk" for name, _, _ in _ply_elements(path)[1]):
        return load_ply_compressed(path)
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            hdr += line
    lines = hdr.decode("latin1").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"{path}: only binary little-endian PLY is supported")
    n, props, in_vertex = 0, [], False
    for l in lines:
        t = l.split()
        if t[:2] == ["element", "vertex"]:
            n, in_vertex = int(t[2]), True
        elif t and t[0] == "element":
            in_vertex = False
        elif in_vertex and t and t[0] == "property":
            if t[1] == "list":
                raise ValueError(f"{path}: list property in the vertex element")
            props.append((t[2], _PLY_TYPES[t[1]]))
    names = [p[0] for p in props]
    if "f_dc_0" not in names or "rot_0" not in names:
        raise ValueError(f"{path}: not a 3DGS PLY (no f_dc_0 / rot_0)")
    v = np.memmap(path, dtype=np.dtype(props), mode="r", offset=len(hdr), shape=(n,))
    g = lambda k: np.asarray(v[k], np.float32)
    rest = sorted((k for k in names if k.startswith("f_rest_")), key=lambda k: int(k.split("_")[-1]))
    K = len(rest) // 3
    deg = {0: 0, 3: 1, 8: 2, 15: 3, 24: 4}[K]
    sh_rest = None
    if K:
        sr = np.stack([g(k) for k in rest], 1).reshape(n, 3, K)          # channel-major in the file
        sh_rest = np.ascontiguousarray(sr.transpose(0, 2, 1)).astype(np.float16)
    q = np.stack([g(f"rot_{i}") for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
    return GaussianCloud(
        means=np.stack([g("x"), g("y"), g("z")], 1), log_scales=np.stack([g(f"scale_{i}") for i in range(3)], 1),
        quats=q.astype(np.float32), opacity=(1.0 / (1.0 + np.exp(-g("opacity")))).astype(np.float32),
        sh_dc=np.stack([g(f"f_dc_{i}") for i in range(3)], 1), sh_rest=sh_rest, sh_rest_u8=False, sh_degree=deg,
        antialiased=None, source=os.path.abspath(path), meta=dict(format="ply"))


# ------------------------------------------------------------------------------------------------- .splat

def load_splat(path: str) -> GaussianCloud:
    rec = np.dtype([("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    v = np.fromfile(path, rec)
    q = (v["r"].astype(np.float32) - 128.0) / 128.0
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
    c = v["c"].astype(np.float32) / 255.0
    return GaussianCloud(means=v["p"].astype(np.float32), log_scales=np.log(np.maximum(v["s"], 1e-12)).astype(np.float32),
                         quats=q, opacity=c[:, 3].copy(), sh_dc=((c[:, :3] - 0.5) / SH_C0).astype(np.float32),
                         sh_degree=0, antialiased=None, source=os.path.abspath(path), meta=dict(format="splat"))


# ------------------------------------------------------------------------------------------------- NVIDIA NuRec .nurec

NUREC_PREFIX = ".gaussians_nodes.gaussians."


class _Blob:
    """A msgpack bin payload, kept as an (offset, length) view into the decoded buffer (no copy)."""
    __slots__ = ("off", "n")

    def __init__(self, off, n):
        self.off, self.n = off, n


def msgpack_tree(buf) -> object:
    """Minimal msgpack decoder (the full spec except ext types) over any buffer (bytes / mmap). Maps, arrays, strings and
    scalars are decoded; ``bin`` payloads are returned as ``_Blob`` (offset, length) so multi-GB tensors are never copied.
    Recursion depth is bounded by the document's nesting (NuRec: < 10)."""
    mv = memoryview(buf)
    be = lambda p, k: int.from_bytes(mv[p:p + k], "big")

    def rd(p):
        b = mv[p]
        if b <= 0x7F: return b, p + 1
        if b >= 0xE0: return b - 256, p + 1
        if 0x80 <= b <= 0x8F: return rmap(p + 1, b & 0x0F)
        if 0x90 <= b <= 0x9F: return rarr(p + 1, b & 0x0F)
        if 0xA0 <= b <= 0xBF: n = b & 0x1F; return bytes(mv[p + 1:p + 1 + n]).decode("utf-8"), p + 1 + n
        if b == 0xC0: return None, p + 1
        if b == 0xC2: return False, p + 1
        if b == 0xC3: return True, p + 1
        if b in (0xC4, 0xC5, 0xC6):
            k = 1 << (b - 0xC4); n = be(p + 1, k); return _Blob(p + 1 + k, n), p + 1 + k + n
        if b == 0xCA: return struct.unpack(">f", mv[p + 1:p + 5])[0], p + 5
        if b == 0xCB: return struct.unpack(">d", mv[p + 1:p + 9])[0], p + 9
        if 0xCC <= b <= 0xCF: k = 1 << (b - 0xCC); return be(p + 1, k), p + 1 + k
        if 0xD0 <= b <= 0xD3: k = 1 << (b - 0xD0); return int.from_bytes(mv[p + 1:p + 1 + k], "big", signed=True), p + 1 + k
        if 0xD9 <= b <= 0xDB:
            k = 1 << (b - 0xD9); n = be(p + 1, k); return bytes(mv[p + 1 + k:p + 1 + k + n]).decode("utf-8"), p + 1 + k + n
        if b in (0xDC, 0xDD): k = 2 << (b - 0xDC); return rarr(p + 1 + k, be(p + 1, k))
        if b in (0xDE, 0xDF): k = 2 << (b - 0xDE); return rmap(p + 1 + k, be(p + 1, k))
        raise ValueError(f"msgpack: unsupported type byte {b:#x} at offset {p}")

    def rmap(p, n):
        d = {}
        for _ in range(n):
            k, p = rd(p); v, p = rd(p); d[k] = v
        return d, p

    def rarr(p, n):
        a = []
        for _ in range(n):
            v, p = rd(p); a.append(v)
        return a, p

    obj, end = rd(0)
    if end != len(mv):
        raise ValueError(f"msgpack: {len(mv) - end} trailing bytes")
    return obj


def _gunzip_mmap(path: str, chunk=64 << 20):
    """Stream-decompress a gzip file in ``chunk``-sized pieces into an anonymous temp file and memory-map it read-only.
    The temp file is unlinked at once (its pages live until the map is released), so nothing is left on disk and the
    decoded data is page-cache backed instead of resident heap (NuRec scenes are 1-2 GB)."""
    import mmap
    import tempfile
    tmpdir = os.path.expanduser(os.environ.get("METALSIM_SPLAT_CACHE", "~/.cache/metalsim/splat"))
    os.makedirs(tmpdir, exist_ok=True)
    with gzip.open(path, "rb") as f, tempfile.TemporaryFile(dir=tmpdir) as g:
        while True:
            b = f.read(chunk)
            if not b:
                break
            g.write(b)
        g.flush()
        return mmap.mmap(g.fileno(), 0, access=mmap.ACCESS_READ)


def load_nurec(path: str) -> GaussianCloud:
    """NVIDIA Omniverse NuRec Gaussian volume (``*.nurec``, referenced by an ``OmniNuRecFieldAsset`` in the scene USD):
    gzip-compressed msgpack ``{"nre_data": {"version", "model", "config", "state_dict"}}`` written by NVIDIA's 3DGRUT /
    NRE trainer (config.layers.gaussians: ``sh-gaussians``; renderer ``3dgut-nrend``). ``state_dict`` holds raw
    little-endian tensors as msgpack ``bin`` + a ``<name>.shape`` list each, dtype from ``config.layers.gaussians.precision``
    (16 -> float16, 32 -> float32), keys prefixed ``.gaussians_nodes.gaussians.``:

    * ``positions`` (N, 3) world means (USD stage units, the Volume prim's frame);
    * ``rotations`` (N, 4) quaternion **wxyz**, activation ``normalize`` (3DGRUT's convention = the 3DGS .ply's rot_0..3);
    * ``scales`` (N, 3) pre-activation; ``scale_activation`` exp -> log-scales as stored;
    * ``densities`` (N, 1) pre-activation; ``density_activation`` sigmoid -> opacity;
    * ``features_albedo`` (N, 3) SH DC; ``features_specular`` (N, 3K) higher bands, coefficient-major / channel-last
      (r, g, b of band 1, then band 2, ...), i.e. already our ``sh_rest`` layout; ``n_active_features`` (int64) = SH degree;
    * ``extra_signal`` (N, 0) unused. Colour = 0.5 + SH(dir) exactly as 3DGS (verified by the band-correlation test).

    The particle kernel is 3DGUT's: degree-2 (Gaussian) response evaluated in 3D along the ray, with a minimum response
    (``density_kernel_min_response``, 0.0113 ~ the 3-sigma contour) and **no 2D dilation**; ``meta['eps2d_hint'] = 0`` tells
    the rasterizer to drop the 0.3 px^2 3DGS dilation (SPLAT.md §7d). The colour-correction matrix (``omni:nurec:ccm*``)
    and the Volume transform live in the USD, not here (identity in every scene seen so far); apply a non-identity Volume
    transform through ``load_scene``'s ``A, t`` (SPLAT.md §7d: some scenes' Gaussians are +y-up although the stage says Z).
    """
    mm = _gunzip_mmap(path)
    tree = msgpack_tree(mm)
    nre = tree["nre_data"] if "nre_data" in tree else tree
    cfg = nre.get("config", {}); sd = nre["state_dict"]
    layer = cfg.get("layers", {}).get("gaussians", {})
    prec = int(layer.get("precision", 32))
    dt = {16: np.dtype("<f2"), 32: np.dtype("<f4")}[prec]

    def tensor(name, dtype=dt):
        b, shape = sd[NUREC_PREFIX + name], tuple(sd[NUREC_PREFIX + name + ".shape"])
        n_el = int(np.prod(shape)) if shape else 1
        if b.n != n_el * dtype.itemsize:
            raise ValueError(f"{path}: {name}: {b.n} bytes for shape {shape} x {dtype}")
        return np.frombuffer(mm, dtype, n_el, b.off).reshape(shape)

    for act, want in (("density_activation", "sigmoid"), ("scale_activation", "exp"), ("rotation_activation", "normalize")):
        if layer.get(act, want) != want:
            raise ValueError(f"{path}: {act}={layer.get(act)!r} not supported (expected {want!r})")
    part = layer.get("particle", {})
    if int(part.get("density_kernel_degree", 2)) != 2 or part.get("density_kernel_planar", False):
        raise ValueError(f"{path}: only the degree-2 (Gaussian), non-planar particle kernel is supported, got {part}")
    pos = tensor("positions").astype(np.float32)
    n = len(pos)
    deg = int(tensor("n_active_features", np.dtype("<i8")).reshape(-1)[0]) if NUREC_PREFIX + "n_active_features" in sd \
        else int(part.get("radiance_sph_degree", 0))
    spec = tensor("features_specular")
    K = spec.shape[1] // 3
    deg = min(deg, {0: 0, 3: 1, 8: 2, 15: 3, 24: 4}[K])
    sh_rest = spec.reshape(n, K, 3)[:, :sh_dim(deg)] if deg > 0 else None   # zero-copy float16 view (page-cache backed)
    q = tensor("rotations").astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
    dens = tensor("densities").astype(np.float32).reshape(n)
    rend = cfg.get("renderer", {})
    meta = dict(format="nurec", version=nre.get("version"), model=nre.get("model"), layer=layer.get("name"),
                precision=prec, renderer=rend.get("name"), kernel_degree=int(part.get("density_kernel_degree", 2)),
                min_response=float(part.get("density_kernel_min_response", 0.0)),
                density_clamping=bool(part.get("density_kernel_density_clamping", True)),
                transmittance_threshold=float(layer.get("transmittance_threshold", 1e-4)),
                global_z_order=rend.get("global_z_order"), render_mode=rend.get("render", {}).get("mode"),
                appearance_embedding=cfg.get("appearance_embedding", {}).get("name"), background=cfg.get("background", {}).get("name"),
                eps2d_hint=0.0, raw_bytes=len(mm))
    return GaussianCloud(means=pos, log_scales=tensor("scales").astype(np.float32), quats=q,
                         opacity=(1.0 / (1.0 + np.exp(-dens))).astype(np.float32),
                         sh_dc=tensor("features_albedo").astype(np.float32), sh_rest=sh_rest, sh_rest_u8=False, sh_degree=deg,
                         antialiased=False, source=os.path.abspath(path), meta=meta)


LOADERS = {".spz": load_spz, ".ply": load_ply, ".splat": load_splat, ".nurec": load_nurec}


def load(path: str) -> GaussianCloud:
    ext = os.path.splitext(path)[1].lower()
    if ext not in LOADERS:
        raise ValueError(f"no splat loader for {ext!r} (have {sorted(LOADERS)}; .sog: not implemented)")
    return LOADERS[ext](path)


def save_ply(cloud: GaussianCloud, path: str) -> None:
    """Write the cloud as a standard 3DGS .ply (used by tests to round-trip formats)."""
    n, K = cloud.n, sh_dim(cloud.sh_degree)
    rest = cloud.sh_rest_float()
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(3 * K)] + \
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    op = np.clip(cloud.opacity, 1e-7, 1 - 1e-7)
    cols = [cloud.means, cloud.sh_dc, rest.transpose(0, 2, 1).reshape(n, 3 * K),
            np.log(op / (1 - op))[:, None], cloud.log_scales, cloud.quats]
    arr = np.concatenate(cols, 1).astype("<f4")
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" +
                 "".join(f"property float {k}\n" for k in names) + "end_header\n").encode())
        f.write(arr.tobytes())
