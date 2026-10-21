"""rsqrt.approx (MUFU.RSQ) regression vectors from the sass-math model (bit-exact on the L4 over [1, 4))."""
import re
import sys

import numpy as np

sys.path.insert(0, "metalsim/physics/mpm/probes")
import mufu_model  # noqa: E402

src = open("_forks/warp/warp/native/metal_cuda_fast_math.h").read()
body = src.split("wp_mufu_rsqrt_packed[128] = {", 1)[1].split("};", 1)[0]
v = np.array([int(h, 16) for h in re.findall(r"0x[0-9a-fA-F]+", body)], dtype=np.uint64).reshape(128, 2)
T = {"rsqrt": np.stack([v[:, 0], v[:, 1] & np.uint64((1 << 17) - 1), v[:, 1] >> np.uint64(17)], axis=1)}
rng = np.random.default_rng(3)
x = (rng.integers(0, 1 << 24, 4096).astype(np.uint32) + np.uint32(0x3F800000))
r = mufu_model.mufu_bits("rsqrt", x, T)
cr = (1.0 / np.sqrt(x.view(np.float32).astype(np.float64))).astype(np.float32).view(np.uint32)
up, down = np.where(r > cr)[0][:5], np.where(r < cr)[0][:5]
eq = np.where(r == cr)[0][:2]
for i in np.concatenate([up, down, eq]):
    print(f"(0x{int(x[i]):08X}, 0x{int(r[i]):08X}),")
