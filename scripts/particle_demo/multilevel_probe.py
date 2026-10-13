"""Metal multi-level instancing probe: what an intersector<..., max_levels<2>> reports for a hit through a nested instance
acceleration structure (tiny: one triangle, 3 + 2 instances, 4 rays).

Scene: primitive structure T (one triangle around the origin, z = 0). Inner instance structure I: 3 instances of T at
x = 10, 20, 30. Top structure: instance 0 = T at x = 0, instance 1 = T at x = 40, instance 2 = I (identity).
Rays straight down at x = 0, 20, 40, 30. Prints per ray: hit, instance_count, instance_id[0], instance_id[1].
"""
import sys
import threading

import numpy as np
import Metal

dev = Metal.MTLCreateSystemDefaultDevice()
q = dev.newCommandQueue()


def buf(a):
    a = np.ascontiguousarray(a)
    b = dev.newBufferWithLength_options_(max(a.nbytes, 16), Metal.MTLResourceStorageModeShared)
    b.contents().as_buffer(a.nbytes)[:] = a.tobytes()
    return b


def run(cb_fill, label):
    cb = q.commandBuffer()
    cb_fill(cb)
    done = threading.Event()
    cb.addCompletedHandler_(lambda _c: done.set())
    cb.commit()
    if not done.wait(10.0):
        print(f"[{label}] did not complete within 10 s"); sys.exit(2)
    if cb.error() is not None:
        print(f"[{label}] error {cb.error()}"); sys.exit(3)


verts = buf(np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32))
g = Metal.MTLAccelerationStructureTriangleGeometryDescriptor.new()
g.setVertexBuffer_(verts); g.setVertexStride_(12); g.setTriangleCount_(1); g.setOpaque_(True)
pd = Metal.MTLPrimitiveAccelerationStructureDescriptor.new(); pd.setGeometryDescriptors_([g])


def make(desc):
    s = dev.accelerationStructureSizesWithDescriptor_(desc)
    return dev.newAccelerationStructureWithSize_(s.accelerationStructureSize), dev.newBufferWithLength_options_(max(s.buildScratchBufferSize, 16), 0)


T, sT = make(pd)


def inst_desc(xs, as_index):
    d = np.zeros((len(xs), 16), np.float32)
    d[:, 0] = d[:, 4] = d[:, 8] = 1.0
    d[:, 9] = xs
    u = d.view(np.uint32); u[:, 12] = 4; u[:, 13] = 0xFFFFFFFF; u[:, 14] = 0; u[:, 15] = as_index
    return d


def ias(prims, descs):
    b = buf(descs)
    d = Metal.MTLInstanceAccelerationStructureDescriptor.new()
    d.setInstancedAccelerationStructures_(prims); d.setInstanceCount_(len(descs))
    d.setInstanceDescriptorBuffer_(b); d.setInstanceDescriptorStride_(64)
    d.setInstanceDescriptorType_(Metal.MTLAccelerationStructureInstanceDescriptorTypeDefault)
    a, s = make(d)
    return d, a, s, b


dI, I, sI, bI = ias([T], inst_desc([10, 20, 30], 0))
top_descs = np.concatenate([inst_desc([0, 40], 0), inst_desc([0], 1)])
dTop, Top, sTop, bTop = ias([T, I], top_descs)


def build(cb):
    e = cb.accelerationStructureCommandEncoder()
    e.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(T, pd, sT, 0)
    e.endEncoding()
    e = cb.accelerationStructureCommandEncoder()
    e.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(I, dI, sI, 0)
    e.endEncoding()
    e = cb.accelerationStructureCommandEncoder()
    e.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(Top, dTop, sTop, 0)
    e.endEncoding()


run(build, "build")
SRC = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;
kernel void k(instance_acceleration_structure accel [[buffer(0)]], device const float* xs [[buffer(1)]],
              device uint* o [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    ray r(float3(xs[i], 0.0, 5.0), float3(0, 0, -1), 0.0, 100.0);
    intersector<triangle_data, instancing, max_levels<2>> isect;
    intersection_result<triangle_data, instancing, max_levels<2>> res = isect.intersect(r, accel);
    o[i * 4 + 0] = res.type == intersection_type::none ? 0u : 1u;
    o[i * 4 + 1] = res.instance_count;
    o[i * 4 + 2] = res.instance_id[0];
    o[i * 4 + 3] = res.instance_id[1];
}
"""
lib, err = dev.newLibraryWithSource_options_error_(SRC, Metal.MTLCompileOptions.new(), None)
if lib is None:
    print("compile failed", err); sys.exit(3)
pso, err = dev.newComputePipelineStateWithFunction_error_(lib.newFunctionWithName_("k"), None)
xs = np.array([0, 20, 40, 30], np.float32)
xb, ob = buf(xs), dev.newBufferWithLength_options_(64, Metal.MTLResourceStorageModeShared)


def trace(cb):
    ce = cb.computeCommandEncoder()
    ce.setComputePipelineState_(pso)
    for a in (Top, I, T):
        ce.useResource_usage_(a, Metal.MTLResourceUsageRead)
    ce.setAccelerationStructure_atBufferIndex_(Top, 0)
    ce.setBuffer_offset_atIndex_(xb, 0, 1); ce.setBuffer_offset_atIndex_(ob, 0, 2)
    ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(4, 1, 1), Metal.MTLSize(4, 1, 1))
    ce.endEncoding()


run(trace, "trace")
o = np.frombuffer(ob.contents().as_buffer(64), np.uint32).reshape(4, 4)
for x, row in zip(xs, o):
    print(f"ray at x={x:4.0f}: hit {row[0]}, instance_count {row[1]}, instance_id[0] {row[2]}, instance_id[1] {row[3]}")
print("expected: x=0 -> top 0; x=20 -> top 2 then inner 1; x=40 -> top 1; x=30 -> top 2 then inner 2")
