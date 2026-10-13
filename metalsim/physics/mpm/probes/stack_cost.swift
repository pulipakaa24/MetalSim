// Does a kernel's private (stack) memory cost a fixed amount per dispatch on this Mac, even when no thread touches it?
// Synthetic kernels with a dynamically indexed thread-private float array of S elements (forces stack allocation),
// launched with n = 0 (every thread returns before using it), 5120 threads in 64-thread groups, 2000 dispatches in an
// ICB with a barrier per command (Warp's graph replay), sustained clock. Also prints each pipeline's
// maxTotalThreadsPerThreadgroup (falls when registers run out).
// Optional argument: a Warp-generated .metal file and a kernel-name substring: its pipeline stats are printed.
//   swiftc -O metalsim/physics/mpm/probes/stack_cost.swift -o /tmp/sc && /tmp/sc [file.metal name]
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
let N = 2000
let out = dev.makeBuffer(length: 8192 * 4, options: .storageModePrivate)!

func timeICB(_ pso: MTLComputePipelineState, _ n: UInt32) -> Double {
  let par = dev.makeBuffer(length: 16, options: .storageModeShared)!
  par.contents().bindMemory(to: UInt32.self, capacity: 4).update(from: [n, 0, 0, 0], count: 4)
  let desc = MTLIndirectCommandBufferDescriptor()
  desc.commandTypes = [.concurrentDispatchThreads]; desc.inheritPipelineState = false; desc.inheritBuffers = false
  desc.maxKernelBufferBindCount = 2
  let icb = dev.makeIndirectCommandBuffer(descriptor: desc, maxCommandCount: N, options: .storageModeShared)!
  for i in 0..<N {
    let c = icb.indirectComputeCommandAt(i)
    c.setComputePipelineState(pso); c.setKernelBuffer(out, offset: 0, at: 0); c.setKernelBuffer(par, offset: 0, at: 1)
    c.concurrentDispatchThreads(MTLSize(width: 5120, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    c.setBarrier()
  }
  func run() -> Double {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.useResource(out, usage: [.read, .write]); e.useResource(par, usage: .read)
    e.executeCommandsInBuffer(icb, range: 0..<N); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(N)
  }
  let t0 = Date(); while Date().timeIntervalSince(t0) < 0.7 { _ = run() }
  return median((0..<7).map { _ in run() })
}

for S in [0, 8, 32, 128, 512, 2048] {
  let body = S == 0 ? "if (p.x == 0) return; o[i] = float(i * p.x);" : """
    if (p.x == 0) return;
    float a[\(S)];
    for (uint k = 0; k < p.x; ++k) a[(i * 7u + k * 13u) % \(S)u] = float(k) * o[(i + k) & 8191u];
    float s = 0.0f; for (uint k = 0; k < p.x; ++k) s += a[(i + k * 5u) % \(S)u];
    o[i & 8191u] = s;
  """
  let src = """
  #include <metal_stdlib>
  using namespace metal;
  kernel void k(device float *o [[buffer(0)]], constant uint4 &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  \(body)
  }
  """
  let lib = try! dev.makeLibrary(source: src, options: nil)
  let d = MTLComputePipelineDescriptor(); d.computeFunction = lib.makeFunction(name: "k")!
  d.supportIndirectCommandBuffers = true
  let pso = try! dev.makeComputePipelineState(descriptor: d, options: [], reflection: nil)
  let t = timeICB(pso, 0)
  print(String(format: "private array %5d floats (%6d B/thread): zero-work dispatch %6.2f us   maxThreads/TG %d", S, S * 4, t,
               pso.maxTotalThreadsPerThreadgroup))
}

let args = CommandLine.arguments
if args.count >= 3 {
  let src = try! String(contentsOfFile: args[1], encoding: .utf8)
  let o = MTLCompileOptions(); o.languageVersion = .version3_2; o.mathMode = .safe; o.mathFloatingPointFunctions = .precise
  let lib = try! dev.makeLibrary(source: src, options: o)
  for name in lib.functionNames where name.contains(args[2]) {
    let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    print("\(name): maxTotalThreadsPerThreadgroup \(pso.maxTotalThreadsPerThreadgroup), static TG memory \(pso.staticThreadgroupMemoryLength) B, SIMD width \(pso.threadExecutionWidth)")
  }
}
