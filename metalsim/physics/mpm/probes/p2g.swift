import Metal
import Foundation

let src = """
#include <metal_stdlib>
using namespace metal;
struct P { float4 x; float4 v; float4 a0; float4 a1; float4 a2; };
struct Params { uint np; uint n; uint N; float dx; float m; float fscale; };

inline void stencil(const P p, float dx, float m, thread int3 &base, thread float3 w[3], thread float3 &fx) {
  float3 xg = p.x.xyz / dx;
  base = int3(floor(xg - 0.5f));
  fx = xg - float3(base);
  w[0] = 0.5f * (1.5f - fx) * (1.5f - fx);
  w[1] = 0.75f - (fx - 1.0f) * (fx - 1.0f);
  w[2] = 0.5f * (fx - 0.5f) * (fx - 0.5f);
}
inline float4 contrib(const P p, float3 fx, int i, int j, int k, float wt, float dx, float m) {
  float3 dpos = (float3(i, j, k) - fx) * dx;
  float3 mom = m * p.v.xyz + float3(dot(p.a0.xyz, dpos), dot(p.a1.xyz, dpos), dot(p.a2.xyz, dpos));
  return wt * float4(m, mom);
}
inline void tg_fadd(threadgroup atomic_uint *a, float v) {
  uint old = atomic_load_explicit(a, memory_order_relaxed);
  while (!atomic_compare_exchange_weak_explicit(a, &old, as_type<uint>(as_type<float>(old) + v), memory_order_relaxed, memory_order_relaxed)) {}
}
inline uint nidx(int3 c, uint N) { return ((uint)(c.z + 1) * N + (uint)(c.y + 1)) * N + (uint)(c.x + 1); }

// V0: non-atomic racy read-modify-write (wrong results; traffic upper bound)
kernel void p2g_racy(device const P *ps [[buffer(0)]], device float4 *g [[buffer(1)]], constant Params &pr [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= pr.np) return;
  P p = ps[gid]; int3 base; float3 w[3]; float3 fx; stencil(p, pr.dx, pr.m, base, w, fx);
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) {
    float wt = w[i].x * w[j].y * w[k].z;
    uint id = nidx(base + int3(i, j, k), pr.N);
    g[id] += contrib(p, fx, i, j, k, wt, pr.dx, pr.m);
  }
}
// V1/V2: plain device float atomics (order decided by the particle buffer)
kernel void p2g_atomic(device const P *ps [[buffer(0)]], device atomic_float *g [[buffer(1)]], constant Params &pr [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= pr.np) return;
  P p = ps[gid]; int3 base; float3 w[3]; float3 fx; stencil(p, pr.dx, pr.m, base, w, fx);
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) {
    float wt = w[i].x * w[j].y * w[k].z;
    uint id = nidx(base + int3(i, j, k), pr.N);
    float4 c = contrib(p, fx, i, j, k, wt, pr.dx, pr.m);
    atomic_fetch_add_explicit(&g[4*id+0], c.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&g[4*id+1], c.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&g[4*id+2], c.z, memory_order_relaxed);
    atomic_fetch_add_explicit(&g[4*id+3], c.w, memory_order_relaxed);
  }
}
// V3: cell-sorted particles + SIMD-group segmented reduction, one device atomic per segment head (Gao 2018 style)
kernel void p2g_simd(device const P *ps [[buffer(0)]], device atomic_float *g [[buffer(1)]], constant Params &pr [[buffer(2)]], uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  bool valid = gid < pr.np;
  P p = ps[valid ? gid : 0]; int3 base; float3 w[3]; float3 fx; stencil(p, pr.dx, pr.m, base, w, fx);
  uint key = valid ? nidx(base, pr.N) : 0xffffffffu;
  // segment tail detection: contiguous equal keys (sorted)
  uint prevkey = simd_shuffle_up(key, 1);
  uint nextkey = simd_shuffle_down(key, 1);
  bool head = (lane == 0) || (prevkey != key);
  bool tail = (lane == 31) || (nextkey != key);
  uint tails = (uint)(ulong)simd_ballot(tail);
  uint run_last = lane + ctz(tails >> lane);
  // segment length to the right
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) {
    float wt = valid ? w[i].x * w[j].y * w[k].z : 0.0f;
    float4 c = contrib(p, fx, i, j, k, wt, pr.dx, pr.m);
    for (ushort off = 1; off < 32; off <<= 1) {
      float4 o = simd_shuffle_down(c, off);
      if (lane + off <= run_last) c += o;
    }
    // c now holds suffix-sum within segment only if segment is contiguous; head holds full segment sum
    if (head && valid) {
      uint id = nidx(base + int3(i, j, k), pr.N);
      atomic_fetch_add_explicit(&g[4*id+0], c.x, memory_order_relaxed);
      atomic_fetch_add_explicit(&g[4*id+1], c.y, memory_order_relaxed);
      atomic_fetch_add_explicit(&g[4*id+2], c.z, memory_order_relaxed);
      atomic_fetch_add_explicit(&g[4*id+3], c.w, memory_order_relaxed);
    }
  }
}
// V4: one threadgroup per 4x4x4 block (512 particles), threadgroup float atomics into a 7^3 x 4 tile, then flush with device atomics
kernel void p2g_tg(device const P *ps [[buffer(0)]], device atomic_float *g [[buffer(1)]], constant Params &pr [[buffer(2)]], uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]], uint tgs [[threads_per_threadgroup]]) {
  threadgroup atomic_uint tile[343*4];
  for (uint t = tid; t < 343*4; t += tgs) atomic_store_explicit(&tile[t], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint nb = pr.n / 4;
  int3 b = int3(tgid % nb, (tgid / nb) % nb, tgid / (nb*nb));
  int3 org = b * 4 - 1;
  if (gid < pr.np) {
    P p = ps[gid]; int3 base; float3 w[3]; float3 fx; stencil(p, pr.dx, pr.m, base, w, fx);
    int3 lb = base - org;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) {
      float wt = w[i].x * w[j].y * w[k].z;
      float4 c = contrib(p, fx, i, j, k, wt, pr.dx, pr.m);
      uint t = (uint)(((lb.z + k) * 7 + (lb.y + j)) * 7 + (lb.x + i));
      tg_fadd(&tile[4*t+0], c.x);
      tg_fadd(&tile[4*t+1], c.y);
      tg_fadd(&tile[4*t+2], c.z);
      tg_fadd(&tile[4*t+3], c.w);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint t = tid; t < 343*4; t += tgs) {
    float v = as_type<float>(atomic_load_explicit(&tile[t], memory_order_relaxed));
    uint node = t / 4, ch = t % 4;
    int3 c = org + int3(node % 7, (node / 7) % 7, node / 49);
    if (v != 0.0f) atomic_fetch_add_explicit(&g[4*nidx(c, pr.N)+ch], v, memory_order_relaxed);
  }
}
// V5: fixed-point int32 device atomics (deterministic, order-independent)
kernel void p2g_fixed(device const P *ps [[buffer(0)]], device atomic_int *g [[buffer(1)]], constant Params &pr [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= pr.np) return;
  P p = ps[gid]; int3 base; float3 w[3]; float3 fx; stencil(p, pr.dx, pr.m, base, w, fx);
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) {
    float wt = w[i].x * w[j].y * w[k].z;
    uint id = nidx(base + int3(i, j, k), pr.N);
    int4 c = int4(rint(contrib(p, fx, i, j, k, wt, pr.dx, pr.m) * pr.fscale));
    atomic_fetch_add_explicit(&g[4*id+0], c.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&g[4*id+1], c.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&g[4*id+2], c.z, memory_order_relaxed);
    atomic_fetch_add_explicit(&g[4*id+3], c.w, memory_order_relaxed);
  }
}
// V6: fixed-point via threadgroup int atomics then device int atomics
kernel void p2g_tg_fixed(device const P *ps [[buffer(0)]], device atomic_int *g [[buffer(1)]], constant Params &pr [[buffer(2)]], uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]], uint tgs [[threads_per_threadgroup]]) {
  threadgroup atomic_int tile[343*4];
  for (uint t = tid; t < 343*4; t += tgs) atomic_store_explicit(&tile[t], 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint nb = pr.n / 4;
  int3 b = int3(tgid % nb, (tgid / nb) % nb, tgid / (nb*nb));
  int3 org = b * 4 - 1;
  if (gid < pr.np) {
    P p = ps[gid]; int3 base; float3 w[3]; float3 fx; stencil(p, pr.dx, pr.m, base, w, fx);
    int3 lb = base - org;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) {
      float wt = w[i].x * w[j].y * w[k].z;
      int4 c = int4(rint(contrib(p, fx, i, j, k, wt, pr.dx, pr.m) * pr.fscale));
      uint t = (uint)(((lb.z + k) * 7 + (lb.y + j)) * 7 + (lb.x + i));
      atomic_fetch_add_explicit(&tile[4*t+0], c.x, memory_order_relaxed);
      atomic_fetch_add_explicit(&tile[4*t+1], c.y, memory_order_relaxed);
      atomic_fetch_add_explicit(&tile[4*t+2], c.z, memory_order_relaxed);
      atomic_fetch_add_explicit(&tile[4*t+3], c.w, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint t = tid; t < 343*4; t += tgs) {
    int v = atomic_load_explicit(&tile[t], memory_order_relaxed);
    uint node = t / 4, ch = t % 4;
    int3 c = org + int3(node % 7, (node / 7) % 7, node / 49);
    if (v != 0) atomic_fetch_add_explicit(&g[4*nidx(c, pr.N)+ch], v, memory_order_relaxed);
  }
}
// V7: base-sorted within block + SIMD segmented reduce + threadgroup int tile + device int flush
kernel void p2g_tg_fixed_simd(device const P *ps [[buffer(0)]], device atomic_int *g [[buffer(1)]], constant Params &pr [[buffer(2)]], uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]], uint tgs [[threads_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_int tile[343*4];
  for (uint t = tid; t < 343*4; t += tgs) atomic_store_explicit(&tile[t], 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint nb = pr.n / 4;
  int3 b = int3(tgid % nb, (tgid / nb) % nb, tgid / (nb*nb));
  int3 org = b * 4 - 1;
  bool valid = gid < pr.np;
  P p = ps[valid ? gid : 0]; int3 base; float3 w[3]; float3 fx; stencil(p, pr.dx, pr.m, base, w, fx);
  int3 lb = base - org;
  uint key = valid ? (uint)((lb.z * 7 + lb.y) * 7 + lb.x) : 0xffffffffu;
  uint prevkey = simd_shuffle_up(key, 1);
  uint nextkey = simd_shuffle_down(key, 1);
  bool head = (lane == 0) || (prevkey != key);
  bool tail = (lane == 31) || (nextkey != key);
  uint tails = (uint)(ulong)simd_ballot(tail);
  uint run_last = lane + ctz(tails >> lane);
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) {
    float wt = valid ? w[i].x * w[j].y * w[k].z : 0.0f;
    int4 c = int4(rint(contrib(p, fx, i, j, k, wt, pr.dx, pr.m) * pr.fscale));
    for (ushort off = 1; off < 32; off <<= 1) {
      int4 o = simd_shuffle_down(c, off);
      if (lane + off <= run_last) c += o;
    }
    if (head && valid) {
      uint t = (uint)(((lb.z + k) * 7 + (lb.y + j)) * 7 + (lb.x + i));
      atomic_fetch_add_explicit(&tile[4*t+0], c.x, memory_order_relaxed);
      atomic_fetch_add_explicit(&tile[4*t+1], c.y, memory_order_relaxed);
      atomic_fetch_add_explicit(&tile[4*t+2], c.z, memory_order_relaxed);
      atomic_fetch_add_explicit(&tile[4*t+3], c.w, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint t = tid; t < 343*4; t += tgs) {
    int v = atomic_load_explicit(&tile[t], memory_order_relaxed);
    uint node = t / 4, ch = t % 4;
    int3 c = org + int3(node % 7, (node / 7) % 7, node / 49);
    if (v != 0) atomic_fetch_add_explicit(&g[4*nidx(c, pr.N)+ch], v, memory_order_relaxed);
  }
}
"""

struct Params { var np: UInt32; var n: UInt32; var N: UInt32; var dx: Float; var m: Float; var fscale: Float }

let dev = MTLCreateSystemDefaultDevice()!
print("device:", dev.name)
let lib = try! dev.makeLibrary(source: src, options: nil)
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let q = dev.makeCommandQueue()!

let sizes = CommandLine.arguments.count > 1 ? CommandLine.arguments[1...].map { Int($0)! } : [32, 64, 128]
for n in sizes {
  let ppc = 8
  let np = n*n*n*ppc
  let N = n + 3
  let dx: Float = 1.0 / Float(n)
  let m: Float = 1.0
  var rng = SystemRandomNumberGenerator()
  // block-major, cell-sorted order: block (4^3) -> cell in block -> particle
  var parts = [Float](repeating: 0, count: np * 20)
  var idx = 0
  let nb = n / 4
  for bz in 0..<nb { for by in 0..<nb { for bx in 0..<nb {
    for cz in 0..<4 { for cy in 0..<4 { for cx in 0..<4 {
      for _ in 0..<ppc {
        let c = [bx*4+cx, by*4+cy, bz*4+cz]
        let o = idx * 20
        for d in 0..<3 { parts[o+d] = (Float(c[d]) + Float.random(in: 0.02..<0.98, using: &rng)) * dx }
        for d in 0..<3 { parts[o+4+d] = Float.random(in: -1..<1, using: &rng) }
        for r in 0..<3 { for d in 0..<3 { parts[o+8+4*r+d] = Float.random(in: -10..<10, using: &rng) } }
        idx += 1
      }
    }}}
  }}}
  // within each block, sort by quadratic-B-spline base node (the P2G "cell" key)
  let per = 64 * ppc
  for blk in 0..<(np / per) {
    var items: [(Int, [Float])] = []
    for i in 0..<per {
      let o = (blk*per + i) * 20
      var key = 0
      for d in [2, 1, 0] { key = key * (n + 3) + Int(floor(parts[o+d] / dx - 0.5)) + 1 }
      items.append((key, Array(parts[o..<(o+20)])))
    }
    items.sort { $0.0 < $1.0 }
    for i in 0..<per { let o = (blk*per + i) * 20; for t in 0..<20 { parts[o+t] = items[i].1[t] } }
  }
  // unsorted copy (random permutation)
  var perm = Array(0..<np); perm.shuffle(using: &rng)
  var shuffled = [Float](repeating: 0, count: np * 20)
  for i in 0..<np { let s = perm[i]; for t in 0..<20 { shuffled[i*20+t] = parts[s*20+t] } }
  let bSorted = dev.makeBuffer(bytes: parts, length: parts.count*4, options: .storageModeShared)!
  let bShuf = dev.makeBuffer(bytes: shuffled, length: shuffled.count*4, options: .storageModeShared)!
  let gbytes = N*N*N*16
  let grid = dev.makeBuffer(length: gbytes, options: .storageModePrivate)!
  let gread = dev.makeBuffer(length: gbytes, options: .storageModeShared)!
  var pr = Params(np: UInt32(np), n: UInt32(n), N: UInt32(N), dx: dx, m: m, fscale: 65536.0)

  func run(_ name: String, _ pbuf: MTLBuffer, tg: Int, reps: Int = 30) -> (Double, Double, [Float]) {
    let p = pso(name)
    var times: [Double] = []
    var cleartimes: [Double] = []
    for r in 0..<(reps+3) {
      let cb0 = q.makeCommandBuffer()!
      let bl = cb0.makeBlitCommandEncoder()!; bl.fill(buffer: grid, range: 0..<gbytes, value: 0); bl.endEncoding()
      cb0.commit(); cb0.waitUntilCompleted()
      let cb = q.makeCommandBuffer()!
      let e = cb.makeComputeCommandEncoder()!
      e.setComputePipelineState(p); e.setBuffer(pbuf, offset: 0, index: 0); e.setBuffer(grid, offset: 0, index: 1)
      e.setBytes(&pr, length: MemoryLayout<Params>.stride, index: 2)
      e.dispatchThreads(MTLSize(width: np, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
      e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
      if r >= 3 { times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000); cleartimes.append((cb0.gpuEndTime - cb0.gpuStartTime)*1000) }
    }
    let cb = q.makeCommandBuffer()!; let bl = cb.makeBlitCommandEncoder()!
    bl.copy(from: grid, sourceOffset: 0, to: gread, destinationOffset: 0, size: gbytes); bl.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let out = [Float](UnsafeBufferPointer(start: gread.contents().assumingMemoryBound(to: Float.self), count: N*N*N*4))
    times.sort(); cleartimes.sort()
    return (times[times.count/2], cleartimes[cleartimes.count/2], out)
  }
  func asFixed(_ a: [Float]) -> [Float] { a.withUnsafeBytes { raw in raw.bindMemory(to: Int32.self).map { Float($0) / Float(1 << 16) } } }
  print("\n== n=\(n)^3 cells, \(np) particles (\(String(format: "%.2f", Double(np)/1e6)) M), grid \(N)^3 nodes x 4 ch (\(gbytes/1048576) MB)")
  let ref = run("p2g_atomic", bSorted, tg: 256)
  let refMass = ref.2.enumerated().filter { $0.offset % 4 == 0 }.map { Double($0.element) }.reduce(0, +)
  print(String(format: "grid clear (blit fill) %.3f ms; total mass check %.1f (expect %d)", ref.1, refMass, np))
  let cases: [(String, String, MTLBuffer, Int, Bool)] = [
    ("V0 racy non-atomic RMW, sorted (wrong; bound)", "p2g_racy", bSorted, 256, false),
    ("V1 device atomics, unsorted particles", "p2g_atomic", bShuf, 256, false),
    ("V2 device atomics, cell-sorted particles", "p2g_atomic", bSorted, 256, false),
    ("V3 sorted + SIMD segmented reduce + device atomics", "p2g_simd", bSorted, 256, false),
    ("V4 sorted by block, tg CAS-float tile + flush", "p2g_tg", bSorted, 512, false),
    ("V5 fixed-point int32 device atomics, sorted", "p2g_fixed", bSorted, 256, true),
    ("V5u fixed-point int32 device atomics, unsorted", "p2g_fixed", bShuf, 256, true),
    ("V6 fixed-point threadgroup tile + flush", "p2g_tg_fixed", bSorted, 512, true),
    ("V6t256 same, 256-thread groups (2 per block)", "p2g_tg_fixed", bSorted, 256, true),
    ("V7 sorted + SIMD reduce + fixed tg tile + flush", "p2g_tg_fixed_simd", bSorted, 512, true),
  ]
  for (label, k, b, tg, fixed) in cases {
    let (t, _, raw) = run(k, b, tg: tg)
    let o = fixed ? asFixed(raw) : raw
    var maxd: Float = 0; var maxv: Float = 0
    for i in 0..<o.count { maxd = max(maxd, abs(o[i]-ref.2[i])); maxv = max(maxv, abs(ref.2[i])) }
    print(String(format: "%-56@ %8.3f ms  %7.2f ms/Mparticle  %6.2f G atomic-adds/s   max|diff| %.2e (max |val| %.2e)", label as NSString, t, t / (Double(np)/1e6), Double(np)*108/(t/1000)/1e9, maxd, maxv))
  }
  // determinism check: run V2 twice and V5 twice
  let a1 = run("p2g_atomic", bShuf, tg: 256, reps: 1).2, a2 = run("p2g_atomic", bShuf, tg: 256, reps: 1).2
  let f1 = run("p2g_fixed", bShuf, tg: 256, reps: 1).2, f2 = run("p2g_fixed", bShuf, tg: 256, reps: 1).2
  func bits(_ a: [Float]) -> [UInt32] { a.map { $0.bitPattern } }
  print("float atomics bitwise-identical across 2 runs: \(bits(a1) == bits(a2)); fixed-point identical: \(bits(f1) == bits(f2))")
}
