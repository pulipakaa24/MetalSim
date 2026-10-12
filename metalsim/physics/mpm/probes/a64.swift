import Metal
let dev = MTLCreateSystemDefaultDevice()!
let srcs = [
 ("device atomic_ulong max", "kernel void k(device atomic_ulong *o [[buffer(0)]], uint gid [[thread_position_in_grid]]) { atomic_max_explicit(&o[gid % 4], (ulong)gid, memory_order_relaxed); }"),
 ("device atomic_ulong store/load", "kernel void k(device atomic_ulong *o [[buffer(0)]], uint gid [[thread_position_in_grid]]) { atomic_store_explicit(&o[gid % 4], atomic_load_explicit(&o[0], memory_order_relaxed)+1, memory_order_relaxed); }"),
 ("device atomic_ulong cas", "kernel void k(device atomic_ulong *o [[buffer(0)]], uint gid [[thread_position_in_grid]]) { ulong e = 0; atomic_compare_exchange_weak_explicit(&o[gid % 4], &e, 1ul, memory_order_relaxed, memory_order_relaxed); }"),
 ("device atomic_float add (control)", "kernel void k(device atomic_float *o [[buffer(0)]], uint gid [[thread_position_in_grid]]) { atomic_fetch_add_explicit(&o[gid % 4], 1.0f, memory_order_relaxed); }"),
 ("threadgroup atomic_uint add (control)", "kernel void k(device uint *o [[buffer(0)]], uint tid [[thread_index_in_threadgroup]]) { threadgroup atomic_uint t[4]; atomic_fetch_add_explicit(&t[tid % 4], 1u, memory_order_relaxed); threadgroup_barrier(mem_flags::mem_threadgroup); if (tid<4) o[tid] = atomic_load_explicit(&t[tid], memory_order_relaxed); }"),
]
for (vn, v) in [("4.0", MTLLanguageVersion.version4_0)] {
 let opt = MTLCompileOptions(); opt.languageVersion = v
 print("MSL", vn)
 for (name, body) in srcs {
  let s = "#include <metal_stdlib>\nusing namespace metal;\n" + body
  do { let l = try dev.makeLibrary(source: s, options: opt); _ = try dev.makeComputePipelineState(function: l.makeFunction(name: "k")!); print("OK  ", name) }
  catch { print("FAIL", name, "-", String(describing: error).split(separator: "\n").first { $0.contains("error:") } ?? "") }
}
}
