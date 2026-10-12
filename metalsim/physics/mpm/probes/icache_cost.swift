// Is instruction fetch a per-dispatch cost for large kernels on this Mac? Same dependent FMA chain (L FMAs per thread)
// written as a loop (small code) or fully unrolled with distinct constants (code size ~ L instructions), 1024 threads,
// dispatched 2000x in an ICB with barriers (Warp's replay), sustained clock; with and without an interleaved
// different small kernel between dispatches (as the contact solve sits between Gauss-Seidel colors).
//   swiftc -O metalsim/physics/mpm/probes/icache_cost.swift -o /tmp/ic && /tmp/ic
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
let out = dev.makeBuffer(length: 8192 * 4, options: .storageModePrivate)!
let N = 1000

func pipeline(_ src: String, _ name: String) -> MTLComputePipelineState {
  let lib = try! dev.makeLibrary(source: src, options: nil)
  let d = MTLComputePipelineDescriptor(); d.computeFunction = lib.makeFunction(name: name)!
  d.supportIndirectCommandBuffers = true
  return try! dev.makeComputePipelineState(descriptor: d, options: [], reflection: nil)
}
let other = pipeline("""
#include <metal_stdlib>
using namespace metal;
kernel void o(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[4096 + (i & 1023)] += 1.0f; }
""", "o")

func timeICB(_ pso: MTLComputePipelineState, interleave: Bool) -> Double {
  let desc = MTLIndirectCommandBufferDescriptor()
  desc.commandTypes = [.concurrentDispatchThreads]; desc.inheritPipelineState = false; desc.inheritBuffers = false
  desc.maxKernelBufferBindCount = 1
  let count = interleave ? 2 * N : N
  let icb = dev.makeIndirectCommandBuffer(descriptor: desc, maxCommandCount: count, options: .storageModeShared)!
  for i in 0..<count {
    let c = icb.indirectComputeCommandAt(i)
    c.setComputePipelineState(interleave && i % 2 == 1 ? other : pso); c.setKernelBuffer(out, offset: 0, at: 0)
    c.concurrentDispatchThreads(MTLSize(width: 1024, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    c.setBarrier()
  }
  func run() -> Double {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.useResource(out, usage: [.read, .write])
    e.executeCommandsInBuffer(icb, range: 0..<count); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(N)
  }
  let t0 = Date(); while Date().timeIntervalSince(t0) < 0.7 { _ = run() }
  return median((0..<7).map { _ in run() })
}
let tOther = timeICB(other, interleave: false)
print(String(format: "small other kernel alone: %.2f us/dispatch", tOther))
for L in [256, 1024, 4096, 8192] {
  let loop = pipeline("""
  #include <metal_stdlib>
  using namespace metal;
  kernel void k(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    float x = o[i & 1023];
    for (uint k = 0; k < \(L); ++k) x = fma(x, 0.999f, 1e-3f);
    o[i & 1023] = x;
  }
  """, "k")
  var body = ""
  for k in 0..<L { body += "x = fma(x, \(String(format: "%.6ff", 0.999 - Double(k) * 1e-7)), \(String(format: "%.6ef", 1e-3 + Double(k) * 1e-9)));\n" }
  let unrolled = pipeline("""
  #include <metal_stdlib>
  using namespace metal;
  kernel void k(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    float x = o[i & 1023];
  \(body)
    o[i & 1023] = x;
  }
  """, "k")
  let a = timeICB(loop, interleave: false), b = timeICB(unrolled, interleave: false)
  let ai = timeICB(loop, interleave: true) - tOther, bi = timeICB(unrolled, interleave: true) - tOther
  print(String(format: "L=%5d  loop %7.2f us  unrolled %7.2f us (%.2f ns/FMA vs %.2f)  | interleaved with another kernel: loop %7.2f  unrolled %7.2f",
               L, a, b, b * 1e3 / Double(L), a * 1e3 / Double(L), ai, bi))
}
