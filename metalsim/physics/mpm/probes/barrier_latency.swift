// Cost of D dependent loads per thread in a chain of dispatches separated by barriers (Warp's ICB replay), with the
// data either tiny (64 KB, cache resident if caches survive the barrier) or 64 MB. 1024 threads in 64-thread groups,
// 2000 dispatches per command buffer, after a 1 s warm-up (sustained clock). us per dispatch above the D = 0 kernel
// tells whether loads after a barrier pay cache-hit or memory latency.
//   swiftc -O metalsim/physics/mpm/probes/barrier_latency.swift -o /tmp/bl && /tmp/bl
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void chase(device const uint *next [[buffer(0)]], device uint *out [[buffer(1)]], constant uint3 &p [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
  uint x = (i * 2654435761u) & p.y;   // p.y = element mask
  for (uint k = 0; k < p.x; ++k) x = next[x];
  out[i] = x + p.z;
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let d = MTLComputePipelineDescriptor(); d.computeFunction = lib.makeFunction(name: "chase")!
d.supportIndirectCommandBuffers = true
let pso = try! dev.makeComputePipelineState(descriptor: d, options: [], reflection: nil)
let out = dev.makeBuffer(length: 4096 * 4, options: .storageModePrivate)!
func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
let N = 2000
for elems in [16384, 16 << 20] {          // 64 KB, 64 MB
  let next = dev.makeBuffer(length: elems * 4, options: .storageModeShared)!
  let p = next.contents().bindMemory(to: UInt32.self, capacity: elems)
  var perm = Array(0..<elems); perm.shuffle()
  for i in 0..<elems { p[perm[i]] = UInt32(perm[(i + 1) % elems]) }
  var row: [String] = []
  var base = 0.0
  for depth in [0, 1, 2, 4, 8] {
    for barrier in [true, false] {
      let par = dev.makeBuffer(length: 16, options: .storageModeShared)!
      par.contents().bindMemory(to: UInt32.self, capacity: 4).update(from: [UInt32(depth), UInt32(elems - 1), 0, 0], count: 4)
      let desc = MTLIndirectCommandBufferDescriptor()
      desc.commandTypes = [.concurrentDispatchThreads]; desc.inheritPipelineState = false; desc.inheritBuffers = false
      desc.maxKernelBufferBindCount = 3
      let icb = dev.makeIndirectCommandBuffer(descriptor: desc, maxCommandCount: N, options: .storageModeShared)!
      for i in 0..<N {
        let c = icb.indirectComputeCommandAt(i)
        c.setComputePipelineState(pso); c.setKernelBuffer(next, offset: 0, at: 0); c.setKernelBuffer(out, offset: 0, at: 1)
        c.setKernelBuffer(par, offset: 0, at: 2)
        c.concurrentDispatchThreads(MTLSize(width: 1024, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        if barrier { c.setBarrier() }
      }
      func run() -> Double {
        let cb = q.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.useResource(next, usage: .read); e.useResource(out, usage: .write); e.useResource(par, usage: .read)
        e.executeCommandsInBuffer(icb, range: 0..<N); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return (cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(N)
      }
      let t0 = Date(); while Date().timeIntervalSince(t0) < 1.0 { _ = run() }
      let t = median((0..<7).map { _ in run() })
      if depth == 0 && barrier { base = t }
      row.append(String(format: "D=%d %@: %6.2f us/dispatch%@", depth, barrier ? "barrier" : "no barrier", t,
                        barrier && depth > 0 ? String(format: " (%.0f ns per dependent load)", (t - base) * 1e3 / Double(depth)) : ""))
    }
  }
  print("data \(elems * 4 / 1024) KB:"); row.forEach { print("  " + $0) }
}
