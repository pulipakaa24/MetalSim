// What each way of asking Metal for "fast math" does to float32 division, square root, denormals and reassociation on
// this GPU, to pick the Metal lowering of Warp's ``fast_math`` module option closest to CUDA's NVRTC --use_fast_math
// (= --ftz=true --prec-div=false --prec-sqrt=false --fmad=true: x/y as __fdividef, 2 ulp for |y| in [2^-126, 2^126];
// sqrtf 1 ulp; f32 denormals flushed; no reassociation, NaN/INF kept).
// Inputs as in fp_semantics.py (a ~ U(-2, 2), x ~ U(1e-3, 10), 65536 each); reference = float64 result rounded once.
//   swiftc -O metalsim/physics/mpm/probes/fast_math_modes.swift -o /tmp/fmm && /tmp/fmm
import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let N = 1 << 16

let body = """
kernel void k(device const float *a [[buffer(0)]], device const float *x [[buffer(1)]], device float *o [[buffer(2)]],
              device float *d [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  o[i] = DIV(a[i], x[i]);
  o[i + N] = SQRT(x[i]);
  o[i + 2 * N] = RSQRT(x[i]);
  o[i + 3 * N] = DIV(1.0f, x[i]);
  if (i == 0) {
    float tiny = a[N - 1];            // 1e-20 (host sets it)
    float sub = a[N - 2];             // 1e-40, a float32 denormal (host sets it)
    float big = a[N - 3];             // 2^24 (host sets it)
    d[0] = tiny * tiny;               // 1e-40 if denormal results are kept, 0 if flushed
    d[1] = sub * 1.0f;                // denormal input through a multiply
    d[2] = sub + 0.0f;                // denormal input through an add
    d[3] = DIV(sub, 2.0f);            // denormal through the division path
    d[4] = SQRT(sub);                 // denormal through the sqrt path (1e-20 if kept)
    d[5] = (x[0] + big) - big;        // reassociation folds this to x[0]
    d[6] = DIV(a[0], 0.0f);           // INF / NaN handling of the division path
    d[7] = SQRT(-1.0f + 0.0f * a[0]); // NaN from sqrt of a negative
    float p = a[1] * x[1];
    d[8] = a[1] * x[1] - p;           // != 0 only if a*b-p is contracted into fma(a, b, -p)
    float nz = -metal::abs(a[2]) * 0.0f;  // -0
    d[9] = metal::copysign(1.0f, nz + 0.0f);  // +1 in IEEE (-0 + +0 = +0); -1 if x + 0 is folded to x (nsz)
  }
}
"""

struct Variant { let name: String; let prelude: String; let mode: MTLMathMode; let fns: MTLMathFloatingPointFunctions }
let plain = "#define DIV(a, b) ((a) / (b))\n#define SQRT(x) metal::sqrt(x)\n#define RSQRT(x) metal::rsqrt(x)\n"
let explicitFast = "#define DIV(a, b) metal::fast::divide(a, b)\n#define SQRT(x) metal::fast::sqrt(x)\n#define RSQRT(x) metal::fast::rsqrt(x)\n"
let variants = [
  Variant(name: "safe+precise, a/b sqrt (Warp today)", prelude: plain, mode: .safe, fns: .precise),
  Variant(name: "safe+precise, fast::divide fast::sqrt", prelude: explicitFast, mode: .safe, fns: .precise),
  Variant(name: "safe, a*fast::divide(1,b), 1/fast::rsqrt (CUDA-shaped)", prelude: "#define DIV(a, b) ((a) * metal::fast::divide(1.0f, (b)))\n#define SQRT(x) metal::fast::divide(1.0f, metal::fast::rsqrt(x))\n#define RSQRT(x) metal::fast::rsqrt(x)\n", mode: .safe, fns: .precise),
  Variant(name: "safe+fast functions, a/b sqrt", prelude: plain, mode: .safe, fns: .fast),
  Variant(name: "safe+precise, #pragma clang fp reciprocal(on)", prelude: "#pragma clang fp reciprocal(on)\n" + plain, mode: .safe, fns: .precise),
  Variant(name: "safe, pragma relaxed + reassociate(off)", prelude: "#pragma METAL fp math_mode(relaxed)\n#pragma clang fp reassociate(off)\n" + plain, mode: .safe, fns: .precise),
  Variant(name: "safe, pragma reassociate(off) + relaxed", prelude: "#pragma clang fp reassociate(off)\n#pragma METAL fp math_mode(relaxed)\n" + plain, mode: .safe, fns: .precise),
  Variant(name: "safe, pragma relaxed + reassoc(off) + contract(off)", prelude: "#pragma METAL fp math_mode(relaxed)\n#pragma clang fp reassociate(off)\n#pragma clang fp contract(off)\n" + plain, mode: .safe, fns: .precise),
  Variant(name: "safe, contract(off)", prelude: "#pragma clang fp contract(off)\n" + plain, mode: .safe, fns: .precise),
  Variant(name: "safe, contract(fast)", prelude: "#pragma clang fp contract(fast)\n" + plain, mode: .safe, fns: .precise),
  Variant(name: "safe, pragma relaxed", prelude: "#pragma METAL fp math_mode(relaxed)\n" + plain, mode: .safe, fns: .precise),
  Variant(name: "relaxed+fast, a/b sqrt", prelude: plain, mode: .relaxed, fns: .fast),
  Variant(name: "fast+fast (Metal default), a/b sqrt", prelude: plain, mode: .fast, fns: .fast),
]

var a = [Float](repeating: 0, count: N), x = [Float](repeating: 0, count: N)
var rng = SystemRandomNumberGenerator()
struct LCG { var s: UInt64; mutating func next() -> Double { s = s &* 6364136223846793005 &+ 1442695040888963407; return Double(s >> 11) / Double(1 << 53) } }
var g = LCG(s: 12345)
for i in 0..<N { a[i] = Float(-2 + 4 * g.next()); x[i] = Float(1e-3 + (10 - 1e-3) * g.next()) }
a[N - 1] = 1e-20; a[N - 2] = Float(bitPattern: 0x000116c2); a[N - 3] = 16777216
_ = rng
let ab = dev.makeBuffer(bytes: a, length: N * 4, options: .storageModeShared)!
let xb = dev.makeBuffer(bytes: x, length: N * 4, options: .storageModeShared)!
let ob = dev.makeBuffer(length: 4 * N * 4, options: .storageModeShared)!
let db = dev.makeBuffer(length: 16 * 4, options: .storageModeShared)!

func ulps(_ r: Float, _ ref: Double) -> Double {
  let rr = Float(ref)
  let u = Double(rr.nextUp) - Double(rr)
  return abs(Double(r) - ref) / u
}

for v in variants {
  let src = "#include <metal_stdlib>\nusing namespace metal;\n#define N \(N)\n" + v.prelude + body
  let opt = MTLCompileOptions()
  opt.languageVersion = .version3_2
  opt.mathMode = v.mode
  opt.mathFloatingPointFunctions = v.fns
  let lib: MTLLibrary
  do { lib = try dev.makeLibrary(source: src, options: opt) } catch {
    print("\(v.name): compile error: \(error.localizedDescription.prefix(300))"); continue
  }
  let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "k")!)
  let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(pso)
  e.setBuffer(ab, offset: 0, index: 0); e.setBuffer(xb, offset: 0, index: 1); e.setBuffer(ob, offset: 0, index: 2)
  e.setBuffer(db, offset: 0, index: 3)
  e.dispatchThreads(MTLSize(width: N - 3, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
  e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  let o = ob.contents().bindMemory(to: Float.self, capacity: 4 * N)
  let d = db.contents().bindMemory(to: Float.self, capacity: 16)
  var line = v.name.padding(toLength: 48, withPad: " ", startingAt: 0)
  for (k, name) in ["div", "sqrt", "rsqrt", "rcp"].enumerated() {
    var cr = 0, maxu = 0.0
    for i in 0..<(N - 3) {
      let ref: Double
      switch k {
      case 0: ref = Double(a[i]) / Double(x[i])
      case 1: ref = Double(x[i]).squareRoot()
      case 2: ref = 1 / Double(x[i]).squareRoot()
      default: ref = 1 / Double(x[i])
      }
      let r = o[k * N + i]
      if r == Float(ref) { cr += 1 }
      maxu = max(maxu, ulps(r, ref))
    }
    line += String(format: " %@ %4.1f%% cr, max %.2f ulp |", name, 100 * Double(cr) / Double(N - 3), maxu)
  }
  for (k, name) in ["div", "sqrt", "rsqrt", "rcp"].enumerated() {
    var h: UInt64 = 1469598103934665603
    for i in 0..<(N - 3) { h = (h ^ UInt64(o[k * N + i].bitPattern)) &* 1099511628211 }
    line += String(format: " %@ %04llx", name, h & 0xffff)
  }
  print(line)
  print("    denormals: tiny*tiny=\(d[0]) sub*1=\(d[1]) sub+0=\(d[2]) sub/2=\(d[3]) sqrt(sub)=\(d[4]) | (x+2^24)-2^24=\(d[5]) (x=\(x[0])) | a/0=\(d[6]) sqrt(-1)=\(d[7]) | fma residual=\(d[8]) | sign(-0+0)=\(d[9])")
}
