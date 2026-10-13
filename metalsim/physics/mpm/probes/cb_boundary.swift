// Cost of a command-buffer boundary between dependent GPU work on this Mac, as Warp's Metal runtime commits one:
// C command buffers each executing an ICB range of D dependent dispatches (1024 threads, barrier after each), the
// buffers ordered by (a) Warp's MTLEvent signal/wait chain, (b) an MTLFence updated / waited across them,
// (c) nothing but queue order, against (d) the same C * D dispatches in one command buffer. Reports GPU-side
// (first gpuStartTime to last gpuEndTime) and host-measured time per boundary over (d).
//   swiftc -O metalsim/physics/mpm/probes/cb_boundary.swift -o /tmp/cbb && /tmp/cbb
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = x[i] * 0.5f + 1.0f; }
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let d = MTLComputePipelineDescriptor(); d.computeFunction = lib.makeFunction(name: "k")!
d.supportIndirectCommandBuffers = true
let pso = try! dev.makeComputePipelineState(descriptor: d, options: [], reflection: nil)
let buf = dev.makeBuffer(length: 4096 * 4, options: [.storageModeShared, .hazardTrackingModeUntracked])!
let D = 50, C = 40
let desc = MTLIndirectCommandBufferDescriptor()
desc.commandTypes = [.concurrentDispatchThreads]; desc.inheritPipelineState = false; desc.inheritBuffers = false
desc.maxKernelBufferBindCount = 1
let icb = dev.makeIndirectCommandBuffer(descriptor: desc, maxCommandCount: D, options: .storageModeShared)!
for i in 0..<D {
  let c = icb.indirectComputeCommandAt(i)
  c.setComputePipelineState(pso); c.setKernelBuffer(buf, offset: 0, at: 0)
  c.concurrentDispatchThreads(MTLSize(width: 1024, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
  c.setBarrier()
}
let event = dev.makeEvent()!
var eventValue: UInt64 = 0
let fence = dev.makeFence()!

func run(_ mode: String) -> (gpu: Double, host: Double) {
  var cbs: [MTLCommandBuffer] = []
  let t0 = Date()
  let n = mode == "single" ? 1 : C
  for j in 0..<n {
    let cb = q.makeCommandBuffer()!
    if mode == "event" && eventValue > 0 { cb.encodeWaitForEvent(event, value: eventValue) }
    let e = cb.makeComputeCommandEncoder()!
    e.useResource(buf, usage: [.read, .write])
    if mode == "fence" && j > 0 { e.waitForFence(fence) }
    for _ in 0..<(mode == "single" ? C : 1) { e.executeCommandsInBuffer(icb, range: 0..<D) }
    if mode == "fence" { e.updateFence(fence) }
    e.endEncoding()
    if mode == "event" { eventValue += 1; cb.encodeSignalEvent(event, value: eventValue) }
    cb.commit(); cbs.append(cb)
  }
  cbs.last!.waitUntilCompleted()
  let host = Date().timeIntervalSince(t0) * 1e6
  return ((cbs.last!.gpuEndTime - cbs.first!.gpuStartTime) * 1e6, host)
}

for mode in ["single", "event", "fence", "queue", "single", "event", "fence", "queue"] {
  let t0 = Date(); while Date().timeIntervalSince(t0) < 0.5 { _ = run(mode) }
  let r = (0..<15).map { _ in run(mode) }
  let g = median(r.map { $0.gpu }), h = median(r.map { $0.host })
  print(String(format: "%-7@ %d x %d dispatches: GPU %7.1f us (%5.2f us/dispatch), host %7.1f us", mode, C, D, g, g / Double(C * D), h))
}
