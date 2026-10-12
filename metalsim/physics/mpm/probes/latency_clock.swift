// Does the M4 Max GPU run latency-bound, low-occupancy work at a lower clock? Dependent FMA chain (16 FMAs unrolled
// per loop trip), 1024 threads (32 SIMD groups, the size of one Gauss-Seidel color), timed with command-buffer GPU
// timestamps: (a) cold, one command buffer after 2 s idle; (b) the same dispatch as the last of 2 s of back-to-back
// command buffers of itself (time-busy, occupancy-light); (c) while a second queue keeps every core saturated with
// independent ALU work. The ns per dependent FMA at a fixed cycle latency is inversely proportional to the clock.
//   swiftc -O metalsim/physics/mpm/probes/latency_clock.swift -o /tmp/lc && /tmp/lc
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let q2 = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void chain(device float *out [[buffer(0)]], constant uint2 &nl [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i >= nl.x) return;
  float x = float(i) * 1e-6f;
  for (uint k = 0; k < nl.y; ++k) {
    x = fma(x, 0.999f, 1e-3f); x = fma(x, 0.999f, 1e-3f); x = fma(x, 0.999f, 1e-3f); x = fma(x, 0.999f, 1e-3f);
    x = fma(x, 0.999f, 1e-3f); x = fma(x, 0.999f, 1e-3f); x = fma(x, 0.999f, 1e-3f); x = fma(x, 0.999f, 1e-3f);
    x = fma(x, 0.999f, 1e-3f); x = fma(x, 0.999f, 1e-3f); x = fma(x, 0.999f, 1e-3f); x = fma(x, 0.999f, 1e-3f);
    x = fma(x, 0.999f, 1e-3f); x = fma(x, 0.999f, 1e-3f); x = fma(x, 0.999f, 1e-3f); x = fma(x, 0.999f, 1e-3f);
  }
  out[i] = x;
}
// independent-work load: 8 interleaved chains per thread, many threads
kernel void load(device float *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float a = i, b = a + 1, c = a + 2, d = a + 3, e = a + 4, f = a + 5, g = a + 6, h = a + 7;
  for (uint k = 0; k < 4096; ++k) {
    a = fma(a, 0.999f, 1e-3f); b = fma(b, 0.999f, 1e-3f); c = fma(c, 0.999f, 1e-3f); d = fma(d, 0.999f, 1e-3f);
    e = fma(e, 0.999f, 1e-3f); f = fma(f, 0.999f, 1e-3f); g = fma(g, 0.999f, 1e-3f); h = fma(h, 0.999f, 1e-3f);
  }
  out[i & 1023] = a + b + c + d + e + f + g + h;
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let chain = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "chain")!)
let load = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "load")!)
let out = dev.makeBuffer(length: 4096 * 4, options: .storageModePrivate)!
let out2 = dev.makeBuffer(length: 4096 * 4, options: .storageModePrivate)!

func chainCB(_ trips: UInt32) -> MTLCommandBuffer {
  var nl = SIMD2<UInt32>(1024, trips)
  let cb = q.makeCommandBuffer()!
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(chain); e.setBuffer(out, offset: 0, index: 0); e.setBytes(&nl, length: 8, index: 1)
  e.dispatchThreads(MTLSize(width: 1024, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
  e.endEncoding()
  return cb
}
func nsPerFMA(_ cbs: [MTLCommandBuffer], _ trips: UInt32) -> [Double] {
  cbs.map { ($0.gpuEndTime - $0.gpuStartTime) * 1e9 / Double(trips * 16) }
}
let trips: UInt32 = 256   // 4096 dependent FMAs, ~10-50 us
func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }

// (a) cold
var cold: [Double] = []
for _ in 0..<5 {
  Thread.sleep(forTimeInterval: 2.0)
  let cb = chainCB(trips); cb.commit(); cb.waitUntilCompleted(); cold += nsPerFMA([cb], trips)
}
print(String(format: "(a) cold, after 2 s idle:          %6.2f ns per dependent FMA (samples %@)", median(cold),
             cold.map { String(format: "%.2f", $0) }.joined(separator: " ")))

// (b) sustained: 2 s of back-to-back command buffers of the same dispatch
Thread.sleep(forTimeInterval: 2.0)
var cbs: [MTLCommandBuffer] = []
let t0 = Date()
while Date().timeIntervalSince(t0) < 2.0 {
  let cb = chainCB(trips); cb.commit(); cbs.append(cb)
  if cbs.count % 64 == 0 { cbs.last!.waitUntilCompleted() }
}
cbs.last!.waitUntilCompleted()
let s = nsPerFMA(cbs, trips)
let n = s.count
print(String(format: "(b) back-to-back for 2 s (%d CBs): first 10 %6.2f, 0.5 s in %6.2f, last 100 %6.2f ns per FMA",
             n, median(Array(s[0..<10])), median(Array(s[n / 4..<n / 4 + 50])), median(Array(s[(n - 100)..<n]))))

// (c) with a second queue saturating the GPU with independent work
Thread.sleep(forTimeInterval: 2.0)
var stop = false
let loader = Thread {
  while !stop {
    let cb = q2.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(load); e.setBuffer(out2, offset: 0, index: 0)
    e.dispatchThreads(MTLSize(width: 40 * 1024, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  }
}
loader.start()
Thread.sleep(forTimeInterval: 1.0)
var loaded: [MTLCommandBuffer] = []
for _ in 0..<50 { let cb = chainCB(trips); cb.commit(); cb.waitUntilCompleted(); loaded.append(cb) }
stop = true
Thread.sleep(forTimeInterval: 0.5)
print(String(format: "(c) with the GPU saturated by another queue: %6.2f ns per FMA (shares issue slots)", median(nsPerFMA(loaded, trips))))
