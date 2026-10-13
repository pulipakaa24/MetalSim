"""GPU health probe: compile and run a tiny Metal kernel and wait at most 10 s for it to complete.

    python scripts/particle_demo/gpu_probe.py

Exit 0 = healthy (prints the completion time), 2 = not completed within the timeout (the GPU is stuck or saturated
by non-preemptible work), 3 = wrong result or a Metal error. Run before risky GPU jobs; a hung GPU wedges every
process and can panic the machine hours later.
"""
import sys
import threading
import time

import numpy as np
import Metal

SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void probe(device uint* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = i * 2u + 1u; }
"""
dev = Metal.MTLCreateSystemDefaultDevice()
lib, err = dev.newLibraryWithSource_options_error_(SRC, Metal.MTLCompileOptions.new(), None)
if lib is None:
    print(f"[probe] compile failed: {err}", flush=True)
    sys.exit(3)
pso, err = dev.newComputePipelineStateWithFunction_error_(lib.newFunctionWithName_("probe"), None)
n = 1024
buf = dev.newBufferWithLength_options_(4 * n, Metal.MTLResourceStorageModeShared)
cb = dev.newCommandQueue().commandBuffer()
ce = cb.computeCommandEncoder()
ce.setComputePipelineState_(pso)
ce.setBuffer_offset_atIndex_(buf, 0, 0)
ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(n, 1, 1), Metal.MTLSize(64, 1, 1))
ce.endEncoding()
done = threading.Event()
cb.addCompletedHandler_(lambda _cb: done.set())
t0 = time.time()
cb.commit()
if not done.wait(10.0):
    print("[probe] GPU did NOT complete a 1024-thread kernel within 10 s", flush=True)
    sys.exit(2)
dt = (time.time() - t0) * 1e3
out = np.frombuffer(buf.contents().as_buffer(4 * n), np.uint32)
if cb.error() is not None or not np.array_equal(out, np.arange(n, dtype=np.uint32) * 2 + 1):
    print(f"[probe] wrong result or Metal error: {cb.error()}", flush=True)
    sys.exit(3)
print(f"[probe] healthy: kernel completed in {dt:.1f} ms", flush=True)
