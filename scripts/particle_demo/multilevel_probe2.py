"""Metal multi-level instancing probe 2: which instance descriptor types make a max_levels<2> intersector report the
nesting (instance_count, instance_id[], user_instance_id[]). Same scene as multilevel_probe.py (top: T at x = 0,
T at x = 40, inner structure I with T at x = 10, 20, 30), with Default or UserID descriptors at each level
(user ids: top 100 + i, inner 200 + i)."""
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


def run(fill, label):
    cb = q.commandBuffer(); fill(cb)
    done = threading.Event(); cb.addCompletedHandler_(lambda _c: done.set()); cb.commit()
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
run(lambda cb: (lambda e: (e.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(T, pd, sT, 0), e.endEncoding()))(cb.accelerationStructureCommandEncoder()), "T")


def descs(xs, as_index, user=None):
    """Default (64 bytes) or UserID (68 bytes: + user id) instance descriptors."""
    n = len(xs); w = 16 if user is None else 17
    d = np.zeros((n, w), np.float32)
    d[:, 0] = d[:, 4] = d[:, 8] = 1.0; d[:, 9] = xs
    u = d.view(np.uint32); u[:, 12] = 4; u[:, 13] = 0xFFFFFFFF; u[:, 14] = 0; u[:, 15] = as_index
    if user is not None:
        u[:, 16] = np.asarray(user, np.uint32)
    return d


def ias(prims, d, user):
    b = buf(d)
    de = Metal.MTLInstanceAccelerationStructureDescriptor.new()
    de.setInstancedAccelerationStructures_(prims); de.setInstanceCount_(len(d))
    de.setInstanceDescriptorBuffer_(b); de.setInstanceDescriptorStride_(64 if not user else 68)
    de.setInstanceDescriptorType_(Metal.MTLAccelerationStructureInstanceDescriptorTypeUserID if user
                                  else Metal.MTLAccelerationStructureInstanceDescriptorTypeDefault)
    a, s = make(de)
    return de, a, s, b


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
    o[i * 6 + 0] = res.type == intersection_type::none ? 0u : 1u;
    o[i * 6 + 1] = res.instance_count;
    o[i * 6 + 2] = res.instance_id[0]; o[i * 6 + 3] = res.instance_id[1];
    o[i * 6 + 4] = res.user_instance_id[0]; o[i * 6 + 5] = res.user_instance_id[1];
}
"""
lib, err = dev.newLibraryWithSource_options_error_(SRC, Metal.MTLCompileOptions.new(), None)
if lib is None:
    print("compile failed", err); sys.exit(3)
pso, _ = dev.newComputePipelineStateWithFunction_error_(lib.newFunctionWithName_("k"), None)
xs = np.array([0, 20, 40, 30], np.float32)
xb = buf(xs)
for top_user, inner_user in ((False, False), (True, False), (False, True), (True, True)):
    dI, I, sI, bI = ias([T], descs([10, 20, 30], 0, [200, 201, 202] if inner_user else None), inner_user)
    td = np.concatenate([descs([0, 40], 0, [100, 101] if top_user else None), descs([0], 1, [102] if top_user else None)])
    dTop, Top, sTop, bTop = ias([T, I], td, top_user)

    def build(cb):
        for a, d, s in ((I, dI, sI), (Top, dTop, sTop)):
            e = cb.accelerationStructureCommandEncoder()
            e.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_(a, d, s, 0)
            e.endEncoding()
    run(build, "build")
    ob = dev.newBufferWithLength_options_(96, Metal.MTLResourceStorageModeShared)

    def trace(cb):
        ce = cb.computeCommandEncoder(); ce.setComputePipelineState_(pso)
        for a in (Top, I, T):
            ce.useResource_usage_(a, Metal.MTLResourceUsageRead)
        ce.setAccelerationStructure_atBufferIndex_(Top, 0)
        ce.setBuffer_offset_atIndex_(xb, 0, 1); ce.setBuffer_offset_atIndex_(ob, 0, 2)
        ce.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(4, 1, 1), Metal.MTLSize(4, 1, 1)); ce.endEncoding()
    run(trace, "trace")
    o = np.frombuffer(ob.contents().as_buffer(96), np.uint32).reshape(4, 6)
    print(f"top {'UserID' if top_user else 'Default'}, inner {'UserID' if inner_user else 'Default'}:")
    for x, row in zip(xs, o):
        print(f"  x={x:3.0f}: hit {row[0]} count {row[1]} id [{row[2]}, {row[3]}] user [{row[4]}, {row[5]}]")
print("expected nesting: x=0 top 0; x=20 top 2 -> inner 1; x=40 top 1; x=30 top 2 -> inner 2")
