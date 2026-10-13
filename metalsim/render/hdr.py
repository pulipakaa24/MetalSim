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


def env_direction(u, v, yaw=0.0, rotation=None):
    """World direction of equirectangular coordinates (u, v) in [0, 1] under the tier-2 dome convention
    (shaders/pathtrace.metal env_dir: OpenUSD's z-up DomeLight_1 frame, pole +z, u = 0.5 faces -y, ``yaw`` about +z;
    a USD stage's dome as Kit renders it is at ``yaw = prim yaw + tier2.KIT_DOME_AZIMUTH``). ``rotation`` (3x3 or wxyz,
    as ``Tier2Renderer.set_environment``) tilts the dome: the direction is ``rotation @ R_z(yaw) @ d``."""
    lon = (0.5 - np.asarray(u, np.float64)) * 2 * np.pi + yaw
    th = np.asarray(v, np.float64) * np.pi
    d = np.stack([np.sin(th) * np.sin(lon), -np.sin(th) * np.cos(lon), np.cos(th)], -1)
    if rotation is None:
        return d
    from metalsim.render.tier2 import dome_rotation
    return d @ dome_rotation(rotation).T


def extract_sun(hdr, yaw=0.0, window_deg=4.0, rel_threshold=0.05, rotation=None):
    """Split an HDRI's sun from its sky: the pixels within ``window_deg`` of the brightest pixel whose luminance
    exceeds ``rel_threshold`` x the peak are replaced by the surrounding sky (median of a ring around the window), and
    their excess radiance is integrated over solid angle. Rendering the sky map plus an analytic directional light with
    the returned irradiance (a sun disk of the returned angular radius) carries the same energy, but the sun is then
    sampled by next-event estimation instead of being found by BSDF-sampled rays (the source of fireflies on glossy
    surfaces under a sharp HDRI sun).

    Returns (sky (H, W, 3) float32, dict(direction (toward the sun, world, ``yaw`` and ``rotation`` applied as in env_direction), irradiance (rgb, normal
    to the sun direction), angular_radius [rad], pixels))."""
    hdr = np.asarray(hdr, np.float32)[..., :3]
    H, W = hdr.shape[:2]
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    y0, x0 = np.unravel_index(int(np.argmax(lum)), lum.shape)
    wy = max(int(np.ceil(window_deg / 180 * H)), 2)
    th0 = (y0 + 0.5) / H * np.pi
    wx = min(max(int(np.ceil(window_deg / 360 * W / max(np.sin(th0), 0.05))), 2), W // 2)
    ys = np.arange(max(y0 - wy, 0), min(y0 + wy + 1, H))
    xs = (np.arange(x0 - wx, x0 + wx + 1)) % W
    sub = hdr[np.ix_(ys, xs)]
    sl = lum[np.ix_(ys, xs)]
    mask = sl > rel_threshold * lum[y0, x0]
    ring = np.ones_like(mask); ring[1:-1, 1:-1] = False
    bg = np.median(sub[ring], 0)
    th = (ys + 0.5) / H * np.pi
    domega = (2 * np.pi / W) * (np.pi / H) * np.sin(th)[:, None] * np.ones((1, len(xs)))
    excess = np.clip(sub - bg, 0, None) * mask[..., None]
    E = (excess * domega[..., None]).sum((0, 1))
    w = (excess.mean(-1) * domega)
    uu = (xs[None, :] + 0.5) / W * np.ones((len(ys), 1)); vv = ((ys + 0.5) / H)[:, None] * np.ones((1, len(xs)))
    dirs = env_direction(uu, vv, yaw, rotation)
    dvec = (dirs * w[..., None]).sum((0, 1)); dvec /= np.linalg.norm(dvec)
    sky = hdr.copy()
    sub_sky = sub.copy(); sub_sky[mask] = bg
    sky[np.ix_(ys, xs)] = sub_sky
    omega = float((domega * mask).sum())
    return sky, dict(direction=dvec, irradiance=E.astype(np.float64), angular_radius=float(np.sqrt(omega / np.pi)),
                     pixels=int(mask.sum()))
