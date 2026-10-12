// Per-dispatch GPU cost of a chain of tiny dependent compute dispatches on this Mac, by encoding mode.
// GPU time from command-buffer GPUStartTime/GPUEndTime (no host timing), median of 7 repetitions.
//   swiftc -O metalsim/physics/mpm/probes/dispatch_modes.swift -o /tmp/dm && /tmp/dm
// Modes: serial encoder (Warp eager: implicit barrier per dispatch), concurrent encoder + memoryBarrier(buffers),
// concurrent encoder without barriers (independent dispatches, lower bound), ICB with setBarrier on every command
// (Warp's graph replay), ICB without barriers, one command buffer per dispatch (WP_METAL_PROFILE mode),
// and a "latency" kernel: each active thread runs a dependent chain of L loads (pointer chase) to calibrate
// memory latency per dependent load.
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void bump(device float *a [[buffer(0)]], constant uint &n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i < n) a[i] = a[i] * 1.0001f + 1.0f;
}
// dependent pointer chase: L loads per thread, each address from the previous load
kernel void chase(device const uint *next [[buffer(0)]], device uint *out [[buffer(1)]], constant uint2 &nl [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
  if (i >= nl.x) return;
  uint p = (i * 7919u) % 1048576u;
  for (uint k = 0; k < nl.y; ++k) p = next[p];
  out[i] = p;
}
// dependent FMA chain: L dependent fused multiply-adds per thread (ALU latency)
kernel void fmachain(device float *out [[buffer(0)]], constant uint2 &nl [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i >= nl.x) return;
  float x = float(i) * 1e-6f;
  for (uint k = 0; k < nl.y; ++k) x = fma(x, 0.999f, 1e-3f);
  out[i] = x;
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let bumpDesc = MTLComputePipelineDescriptor()
bumpDesc.computeFunction = lib.makeFunction(name: "bump")!
bumpDesc.supportIndirectCommandBuffers = true
let bump = try! dev.makeComputePipelineState(descriptor: bumpDesc, options: [], reflection: nil)
let chase = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "chase")!)
let fmac = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "fmachain")!)

func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }

func gpuTime(_ encode: (MTLCommandBuffer) -> Void) -> Double {
  let cb = q.makeCommandBuffer()!
  encode(cb)
  cb.commit(); cb.waitUntilCompleted()
  return cb.gpuEndTime - cb.gpuStartTime
}

let N = 2000
for threads in [64, 1024, 5120] {
  let a = dev.makeBuffer(length: max(threads, 64) * 4, options: .storageModePrivate)!
  var n = UInt32(threads)
  let nbuf = dev.makeBuffer(bytes: &n, length: 4, options: .storageModeShared)!
  let tg = MTLSize(width: 64, height: 1, depth: 1)
  let grid = MTLSize(width: threads, height: 1, depth: 1)
  var res: [(String, Double)] = []
  func run(_ name: String, _ body: (MTLCommandBuffer) -> Void) {
    _ = gpuTime(body)
    res.append((name, median((0..<7).map { _ in gpuTime(body) }) * 1e6 / Double(N)))
  }
  run("serial encoder") { cb in
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(bump); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(nbuf, offset: 0, index: 1)
    for _ in 0..<N { e.dispatchThreads(grid, threadsPerThreadgroup: tg) }
    e.endEncoding()
  }
  run("concurrent + memoryBarrier") { cb in
    let e = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
    e.setComputePipelineState(bump); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(nbuf, offset: 0, index: 1)
    for _ in 0..<N { e.dispatchThreads(grid, threadsPerThreadgroup: tg); e.memoryBarrier(scope: .buffers) }
    e.endEncoding()
  }
  run("concurrent, no barrier") { cb in
    let e = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
    e.setComputePipelineState(bump); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(nbuf, offset: 0, index: 1)
    for _ in 0..<N { e.dispatchThreads(grid, threadsPerThreadgroup: tg) }
    e.endEncoding()
  }
  for barrier in [true, false] {
    let d = MTLIndirectCommandBufferDescriptor()
    d.commandTypes = [.concurrentDispatchThreads]
    d.inheritPipelineState = false; d.inheritBuffers = false; d.maxKernelBufferBindCount = 2
    let icb = dev.makeIndirectCommandBuffer(descriptor: d, maxCommandCount: N, options: .storageModeShared)!
    for i in 0..<N {
      let c = icb.indirectComputeCommandAt(i)
      c.setComputePipelineState(bump); c.setKernelBuffer(a, offset: 0, at: 0); c.setKernelBuffer(nbuf, offset: 0, at: 1)
      c.concurrentDispatchThreads(grid, threadsPerThreadgroup: tg)
      if barrier { c.setBarrier() }
    }
    run(barrier ? "ICB + setBarrier (Warp graph)" : "ICB, no barrier") { cb in
      let e = cb.makeComputeCommandEncoder()!
      e.useResource(a, usage: [.read, .write]); e.useResource(nbuf, usage: .read)
      e.executeCommandsInBuffer(icb, range: 0..<N)
      e.endEncoding()
    }
  }
  // one command buffer per dispatch, chained like Warp's (each committed separately)
  do {
    var ts: [Double] = []
    for _ in 0..<5 {
      var cbs: [MTLCommandBuffer] = []
      for _ in 0..<400 {
        let cb = q.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(bump); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(nbuf, offset: 0, index: 1)
        e.dispatchThreads(grid, threadsPerThreadgroup: tg); e.endEncoding(); cb.commit(); cbs.append(cb)
      }
      cbs.last!.waitUntilCompleted()
      ts.append(cbs.map { $0.gpuEndTime - $0.gpuStartTime }.reduce(0, +) / 400)
      ts.append((cbs.last!.gpuEndTime - cbs.first!.gpuStartTime) / 400)
    }
    res.append(("1 CB per dispatch: in-CB time", median(stride(from: 0, to: ts.count, by: 2).map { ts[$0] }) * 1e6))
    res.append(("1 CB per dispatch: wall/dispatch", median(stride(from: 1, to: ts.count, by: 2).map { ts[$0] }) * 1e6))
  }
  for (k, v) in res { print(String(format: "threads %5d  %-34@ %7.2f us/dispatch", threads, k as NSString, v)) }
}

// memory / ALU latency calibration: 1024 threads (≈ one strain-color's worth), one dispatch, L dependent steps
let M = 1048576
let nextBuf = dev.makeBuffer(length: M * 4, options: .storageModeShared)!
let np = nextBuf.contents().bindMemory(to: UInt32.self, capacity: M)
var perm = Array(0..<M).shuffled()
for i in 0..<M { np[perm[i]] = UInt32(perm[(i + 1) % M]) }   // one random cycle over 4 MB (L2/SLC resident)
let outBuf = dev.makeBuffer(length: 1 << 20, options: .storageModePrivate)!
for (name, pso) in [("pointer chase (4 MB set)", chase), ("dependent FMA", fmac)] {
  var prev = 0.0
  for L in [0, 256, 1024] {
    var nl = SIMD2<UInt32>(1024, UInt32(L))
    let t = median((0..<7).map { _ in gpuTime { cb in
      let e = cb.makeComputeCommandEncoder()!
      e.setComputePipelineState(pso)
      if pso === chase { e.setBuffer(nextBuf, offset: 0, index: 0); e.setBuffer(outBuf, offset: 0, index: 1); e.setBytes(&nl, length: 8, index: 2) }
      else { e.setBuffer(outBuf, offset: 0, index: 0); e.setBytes(&nl, length: 8, index: 1) }
      for _ in 0..<20 { e.dispatchThreads(MTLSize(width: 1024, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)) }
      e.endEncoding() } }) / 20
    if L > 0 { print(String(format: "%@: L=%4d  %8.2f us/dispatch, %6.1f ns per dependent step (vs L=0)", name, L, t * 1e6, (t - prev) * 1e9 / Double(L))) }
    else { prev = t; print(String(format: "%@: L=0  %8.2f us/dispatch", name, t * 1e6)) }
  }
}
