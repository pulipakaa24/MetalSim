// Where Warp's graph-replay dispatch cost (2.4 us per dependent empty dispatch, fixed_cost.py) goes above raw
// Metal's (1.06 us, barrier_latency.swift): the same zero-work kernel built up one Warp ingredient at a time.
//   raw        kernel(device float*), dispatchThreads, no threadgroup memory
//   +tgmem     threadgroup char* arena [[threadgroup(0)]] with 256 B bound and whole threadgroups dispatched
//              (Warp binds group_size * 4 B for the shared-tile arena's per-lane offsets on every launch)
//   +init      each thread writes its arena offset (wp::tile_shared_storage_t::init)
//   +bounds    constant launch_bounds_t& dim [[buffer(0)]] (shape[4], ndim, size: 32 B) and the size check
//   +args      constant args struct (int + two 56-byte array descriptors) read for the early exit
//   +devargs   the same struct read through a device const pointer
// 5120 threads in 64-thread groups, 2000 dispatches in an ICB with a barrier after each, sustained clock.
//   swiftc -O metalsim/physics/mpm/probes/dispatch_anatomy.swift -o /tmp/da && /tmp/da
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
let N = 2000
let data = dev.makeBuffer(length: 1 << 16, options: .storageModePrivate)!
let offs = dev.makeBuffer(length: 64, options: .storageModeShared)!
memset(offs.contents(), 0, 64)

let header = """
#include <metal_stdlib>
using namespace metal;
struct Bounds { int shape[4]; int ndim; uint size; };
struct Arr { device int* data; device int* grad; int shape[4]; int strides[4]; ushort ndim; ushort flags; };
struct Args { int color; Arr offsets; Arr out; };
"""
struct V { let name: String; let src: String; let tg: Bool; let bounds: Bool }
let variants: [V] = [
  V(name: "raw", src: "kernel void k(device int* out [[buffer(1)]], uint i [[thread_position_in_grid]]) { if (int(i) < 0) out[0] = 1; }", tg: false, bounds: false),
  V(name: "+tgmem", src: "kernel void k(device int* out [[buffer(1)]], threadgroup char* a [[threadgroup(0)]], uint i [[thread_position_in_grid]]) { if (int(i) < 0) out[0] = 1; }", tg: true, bounds: false),
  V(name: "+init", src: "kernel void k(device int* out [[buffer(1)]], threadgroup char* a [[threadgroup(0)]], uint i [[thread_position_in_grid]], uint l [[thread_index_in_threadgroup]]) { ((threadgroup uint*)a)[l] = 0; if (int(i) < 0) out[0] = 1; }", tg: true, bounds: false),
  V(name: "+bounds", src: "kernel void k(constant Bounds& dim [[buffer(0)]], device int* out [[buffer(1)]], threadgroup char* a [[threadgroup(0)]], uint i [[thread_position_in_grid]], uint l [[thread_index_in_threadgroup]]) { if (i >= dim.size) return; ((threadgroup uint*)a)[l] = 0; if (int(i) < 0) out[0] = 1; }", tg: true, bounds: true),
  V(name: "+args", src: "kernel void k(constant Bounds& dim [[buffer(0)]], constant Args* args [[buffer(1)]], threadgroup char* a [[threadgroup(0)]], uint i [[thread_position_in_grid]], uint l [[thread_index_in_threadgroup]]) { if (i >= dim.size) return; ((threadgroup uint*)a)[l] = 0; int b = args->offsets.data[args->color] + int(i), e = args->offsets.data[args->color + 1]; for (int j = b; j < e; j += 5120) args->out.data[j] = 1; }", tg: true, bounds: true),
  V(name: "+devargs", src: "kernel void k(constant Bounds& dim [[buffer(0)]], device const Args* args [[buffer(1)]], threadgroup char* a [[threadgroup(0)]], uint i [[thread_position_in_grid]], uint l [[thread_index_in_threadgroup]]) { if (i >= dim.size) return; ((threadgroup uint*)a)[l] = 0; int b = args->offsets.data[args->color] + int(i), e = args->offsets.data[args->color + 1]; for (int j = b; j < e; j += 5120) args->out.data[j] = 1; }", tg: true, bounds: true),
  V(name: "devargs-tg", src: "kernel void k(constant Bounds& dim [[buffer(0)]], device const Args* args [[buffer(1)]], uint i [[thread_position_in_grid]]) { if (i >= dim.size) return; int b = args->offsets.data[args->color] + int(i), e = args->offsets.data[args->color + 1]; for (int j = b; j < e; j += 5120) args->out.data[j] = 1; }", tg: false, bounds: true),
]

let bb = dev.makeBuffer(length: 64, options: .storageModeShared)!
let bp = bb.contents(); memset(bp, 0, 64)
bp.storeBytes(of: Int32(5120), toByteOffset: 0, as: Int32.self); bp.storeBytes(of: Int32(1), toByteOffset: 16, as: Int32.self)
bp.storeBytes(of: UInt32(5120), toByteOffset: 20, as: UInt32.self)
let ab = dev.makeBuffer(length: 256, options: .storageModeShared)!
let ap = ab.contents(); memset(ap, 0, 256)
ap.storeBytes(of: offs.gpuAddress, toByteOffset: 8, as: UInt64.self)      // Args.offsets.data (after int + pad)
ap.storeBytes(of: data.gpuAddress, toByteOffset: 64, as: UInt64.self)     // Args.out.data

for v in variants {
  let lib = try! dev.makeLibrary(source: header + v.src, options: nil)
  let d = MTLComputePipelineDescriptor(); d.computeFunction = lib.makeFunction(name: "k")!
  d.supportIndirectCommandBuffers = true
  let pso = try! dev.makeComputePipelineState(descriptor: d, options: [], reflection: nil)
  let desc = MTLIndirectCommandBufferDescriptor()
  desc.commandTypes = [.concurrentDispatch, .concurrentDispatchThreads]; desc.inheritPipelineState = false; desc.inheritBuffers = false
  desc.maxKernelBufferBindCount = 2
  desc.maxKernelThreadgroupMemoryBindCount = 1
  let icb = dev.makeIndirectCommandBuffer(descriptor: desc, maxCommandCount: N, options: .storageModeShared)!
  for i in 0..<N {
    let c = icb.indirectComputeCommandAt(i)
    c.setComputePipelineState(pso)
    if v.bounds { c.setKernelBuffer(bb, offset: 0, at: 0) }
    c.setKernelBuffer(v.name == "raw" || v.name == "+tgmem" || v.name == "+init" ? data : ab, offset: 0, at: 1)
    if v.tg {
      c.setThreadgroupMemoryLength(256, index: 0)
      c.concurrentDispatchThreadgroups(MTLSize(width: 80, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    } else {
      c.concurrentDispatchThreads(MTLSize(width: 5120, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }
    c.setBarrier()
  }
  func run() -> Double {
    let cb = q.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.useResource(ab, usage: .read); e.useResource(bb, usage: .read); e.useResource(offs, usage: .read)
    e.useResource(data, usage: [.read, .write])
    e.executeCommandsInBuffer(icb, range: 0..<N); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(N)
  }
  let t0 = Date(); while Date().timeIntervalSince(t0) < 0.7 { _ = run() }
  print(String(format: "%-10@ %5.2f us/dispatch", v.name, median((0..<9).map { _ in run() })))
}
