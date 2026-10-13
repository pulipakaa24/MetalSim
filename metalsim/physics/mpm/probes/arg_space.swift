// Fixed per-dispatch cost of a kernel's argument struct on this Mac, by address space and size: a zero-work kernel
// (every thread reads two offsets and exits, as Newton's colored Gauss-Seidel kernel does for an empty color) whose
// skipped loop body uses A array descriptors (Warp's array_t: data, grad, shape[4], strides[4], ndim/flags = 56 B)
// read from the argument buffer as `constant Args*` (Warp's Metal entry point), `device const Args*`, or with only
// the data pointers packed. 5120 threads in 64-thread groups, 2000 dispatches in an ICB with a barrier after each
// (Warp's graph replay), sustained clock.
//   swiftc -O metalsim/physics/mpm/probes/arg_space.swift -o /tmp/as && /tmp/as
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
let N = 2000
let data = dev.makeBuffer(length: 1 << 20, options: .storageModePrivate)!
let offs = dev.makeBuffer(length: 64, options: .storageModeShared)!
memset(offs.contents(), 0, 64)

func source(_ A: Int, _ space: String, _ packed: Bool) -> String {
  var fields = "", sum = ""
  for k in 0..<A {
    fields += packed ? "  device float* a\(k);\n" : "  Arr a\(k);\n"
    sum += packed ? " + args->a\(k)[j]" : " + args->a\(k).data[j * (args->a\(k).strides[0] >> 2)]"
  }
  return """
  #include <metal_stdlib>
  using namespace metal;
  struct Arr { device float* data; device float* grad; int shape[4]; int strides[4]; ushort ndim; ushort flags; };
  struct Args { int color; int launch_dim; device int* offsets; device float* out;
  \(fields)};
  kernel void k(\(space) Args* args [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    int beg = args->offsets[args->color] + int(i), end = args->offsets[args->color + 1];
    for (int j = beg; j < end; j += args->launch_dim) args->out[j] = 0.0f\(sum);
  }
  """
}

for (A, label) in [(0, "0 arrays"), (4, "4 arrays"), (16, "16 arrays"), (32, "32 arrays")] {
  var row = String(format: "%-10@", label)
  for (space, packed) in [("constant", false), ("device const", false), ("constant", true), ("device const", true)] {
    let lib = try! dev.makeLibrary(source: source(A, space, packed), options: nil)
    let d = MTLComputePipelineDescriptor(); d.computeFunction = lib.makeFunction(name: "k")!
    d.supportIndirectCommandBuffers = true
    let pso = try! dev.makeComputePipelineState(descriptor: d, options: [], reflection: nil)
    // argument struct: color, launch_dim, offsets, out, then A descriptors (all pointing at `data`)
    let stride = packed ? 8 : 56
    let size = 24 + 8 + A * stride + 64
    let argb = dev.makeBuffer(length: size, options: .storageModeShared)!
    let p = argb.contents()
    memset(p, 0, size)
    p.storeBytes(of: Int32(0), toByteOffset: 0, as: Int32.self)
    p.storeBytes(of: Int32(5120), toByteOffset: 4, as: Int32.self)
    p.storeBytes(of: offs.gpuAddress, toByteOffset: 8, as: UInt64.self)
    p.storeBytes(of: data.gpuAddress, toByteOffset: 16, as: UInt64.self)
    for k in 0..<A {
      let o = 24 + k * stride
      p.storeBytes(of: data.gpuAddress, toByteOffset: o, as: UInt64.self)
      if !packed { p.storeBytes(of: Int32(4), toByteOffset: o + 32, as: Int32.self) }
    }
    let desc = MTLIndirectCommandBufferDescriptor()
    desc.commandTypes = [.concurrentDispatchThreads]; desc.inheritPipelineState = false; desc.inheritBuffers = false
    desc.maxKernelBufferBindCount = 2
    let icb = dev.makeIndirectCommandBuffer(descriptor: desc, maxCommandCount: N, options: .storageModeShared)!
    for i in 0..<N {
      let c = icb.indirectComputeCommandAt(i)
      c.setComputePipelineState(pso); c.setKernelBuffer(argb, offset: 0, at: 1)
      c.concurrentDispatchThreads(MTLSize(width: 5120, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      c.setBarrier()
    }
    func run() -> Double {
      let cb = q.makeCommandBuffer()!
      let e = cb.makeComputeCommandEncoder()!
      e.useResource(argb, usage: .read); e.useResource(offs, usage: .read); e.useResource(data, usage: [.read, .write])
      e.executeCommandsInBuffer(icb, range: 0..<N); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
      return (cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(N)
    }
    let t0 = Date(); while Date().timeIntervalSince(t0) < 0.7 { _ = run() }
    let t = median((0..<7).map { _ in run() })
    row += String(format: " | %@%@ %5.2f us", space == "constant" ? "constant" : "device  ", packed ? " packed" : "       ", t)
  }
  print(row)
}
