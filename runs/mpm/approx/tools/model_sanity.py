import sys
import numpy as np
sys.path.insert(0, "metalsim/physics/mpm/probes")
from mufu_model import cuda_div, cuda_rcp, cuda_sqrt, load_tables

T = load_tables()
print(hex(cuda_rcp(np.float32(1.1), T).view(np.uint32)[0]), "expect 0x3f68ba2e")
x = np.array([0x3FD6216A, 0x3F875CEE], dtype=np.uint32).view(np.float32)
print([hex(v) for v in cuda_rcp(x, T).view(np.uint32)], "expect 0x3f190736 0x3f721333")
s = np.array([0x3FDBB398, 0x407FCDB3], dtype=np.uint32).view(np.float32)
print([hex(v) for v in cuda_sqrt(s, T).view(np.uint32)], "expect 0x3fa7b210 0x3fffe6d8")
print(cuda_div(np.float32([1, -1, 3, 0, 1]), np.float32([0, np.inf, 1.5 * 2.0**126, 0, -0.0]), T))
