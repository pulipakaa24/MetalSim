import Metal
let dev = MTLCreateSystemDefaultDevice()!
print(dev.name, "apple9:", dev.supportsFamily(.apple9), "metal4:", dev.supportsFamily(.metal4))
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device float *o [[buffer(0)]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup atomic_float t[4];
  if (tid < 4) atomic_store_explicit(&t[tid], 0.0f, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  atomic_fetch_add_explicit(&t[tid % 4], 1.0f, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 4) o[tid] = atomic_load_explicit(&t[tid], memory_order_relaxed);
}
"""
for (name, v) in [("3.0", MTLLanguageVersion.version3_0), ("3.1", .version3_1), ("3.2", .version3_2), ("4.0", .version4_0)] {
  let o = MTLCompileOptions(); o.languageVersion = v
  do { _ = try dev.makeLibrary(source: src, options: o); print("MSL \(name): threadgroup atomic_float compiles") }
  catch { print("MSL \(name): fails:", String(describing: error).split(separator: "\n").filter{ $0.contains("error") }.first ?? "") }
}
