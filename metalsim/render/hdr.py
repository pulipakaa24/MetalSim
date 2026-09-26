"""HDR image loading for environment maps: Radiance RGBE (.hdr / .pic; Ward 1991, the format Poly Haven
ships) decoded here in numpy (no OpenCV / FreeImage dependency), other formats through imageio.

RGBE: header lines up to a blank line (``FORMAT=32-bit_rle_rgbe``), a resolution line ``-Y H +X W``, then one
scanline per row: either flat RGBE quadruples or the "new" run-length encoding (a scanline starts with
``2 2 hi lo`` = the width, then the R, G, B and E planes each run-length coded: a count byte > 128 repeats the
next byte count - 128 times, otherwise the next count bytes are literal). Pixel value per channel =
(mantissa + 0.5) / 256 * 2^(e - 128) (Radiance's ``colr_color``), 0 where e = 0.
"""
from __future__ import annotations

import numpy as np


def _decode_rle_scanline(buf, pos, width):
    """Decode one new-style RLE scanline starting at ``pos``; returns (rgbe (W, 4) uint8, new pos)."""
    out = np.empty((4, width), np.uint8)
    for ch in range(4):
        x = 0
        while x < width:
            count = buf[pos]; pos += 1
            if count > 128:
                n = count - 128
                out[ch, x:x + n] = buf[pos]; pos += 1
            else:
                n = count
                out[ch, x:x + n] = np.frombuffer(buf, np.uint8, n, pos); pos += n
            x += n
    return out.T, pos


def load_rgbe(path: str) -> np.ndarray:
    """Radiance RGBE file -> (H, W, 3) float32 linear radiance."""
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(b"#?"):
        raise ValueError(f"{path}: not a Radiance RGBE file")
    pos = 0
    while True:   # header ends at an empty line
        nl = data.index(b"\n", pos)
        line = data[pos:nl]; pos = nl + 1
        if line == b"":
            break
    nl = data.index(b"\n", pos)
    res = data[pos:nl].decode().split(); pos = nl + 1
    if len(res) != 4 or res[0] not in ("-Y", "+Y") or res[2] not in ("+X", "-X"):
        raise ValueError(f"{path}: unsupported resolution line {res}")
    h, w = int(res[1]), int(res[3])
    rows = []
    buf = memoryview(data)
    for _ in range(h):
        if 8 <= w <= 0x7FFF and buf[pos] == 2 and buf[pos + 1] == 2 and (buf[pos + 2] << 8 | buf[pos + 3]) == w:
            row, pos = _decode_rle_scanline(buf, pos + 4, w)
        else:   # flat scanline (old-style RLE is not produced by current writers; not supported)
            row = np.frombuffer(data, np.uint8, w * 4, pos).reshape(w, 4); pos += w * 4
        rows.append(row)
    rgbe = np.stack(rows).astype(np.float32)
    e = rgbe[..., 3]
    scale = np.where(e > 0, np.ldexp(np.float32(1.0), (e - 136).astype(np.int32)), np.float32(0.0)).astype(np.float32)   # 2^(e-128) / 256
    img = (rgbe[..., :3] + 0.5) * scale[..., None]
    img[e <= 0] = 0.0
    if res[0] == "+Y":
        img = img[::-1]
    if res[2] == "-X":
        img = img[:, ::-1]
    return np.ascontiguousarray(img, np.float32)


def load_hdr(path: str) -> np.ndarray:
    """(H, W, 3) float32 linear radiance from .hdr/.pic (RGBE) or any format imageio reads (EXR needs a plugin)."""
    if path.lower().endswith((".hdr", ".pic", ".rgbe")):
        return load_rgbe(path)
    import imageio.v3 as iio
    im = np.asarray(iio.imread(path))
    if im.dtype == np.uint8:
        im = (im.astype(np.float32) / 255.0) ** 2.2   # 8-bit files are sRGB-ish; approximate linearisation
    return np.ascontiguousarray(im[..., :3], np.float32)
