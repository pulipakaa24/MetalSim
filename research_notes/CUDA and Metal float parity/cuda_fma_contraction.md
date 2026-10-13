# FMA contraction in the CUDA pipeline (NVRTC/NVVM to PTX to ptxas to SASS) for Warp 1.16 kernels on sm_89

Scope: float32 multiply/add contraction when Warp 1.16 JIT-compiles with NVRTC `--fmad=true` (the default, `fuse_fp=True`) and ptxas assembles for sm_89 (L4). Sources were checked on 2026-10-10. LLVM quotes come from llvm-project `main` as of that date, with LLVM 17.0.6 used where the behaviour differs. NVVM (the front end inside NVRTC) is NVIDIA's closed-source fork of LLVM. Upstream LLVM is evidence about its likely logic, not proof of it.

## 1. What the PTX ISA says about mul/add/sub without a rounding modifier

### Takeaway
PTX ISA 9.4 (current) says that `add`/`sub`/`mul` with no rounding modifier default to round-to-nearest-even and "may be optimized aggressively". In particular, mul/add and mul/sub sequences without a modifier "may be optimized to use fused-multiply-add instructions on the target device". An explicit `.rn` is "treated conservatively". That is ptxas's documented licence to fuse, and `.rn` is the documented way to block it. The ISA gives no rule for *when* ptxas fuses.

### Cited Findings
- The document is titled "Parallel Thread Execution ISA Version 9.4" — [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- Floating-point `add` notes, verbatim: "The default value of rounding modifier is .rn. Note that an add instruction with an explicit rounding modifier is treated conservatively by the code optimizer. An add instruction with no rounding modifier defaults to round-to-nearest-even and may be optimized aggressively by the code optimizer. In particular, mul/add sequences with no rounding modifiers may be optimized to use fused-multiply-add instructions on the target device." — [PTX ISA 9.4, §9.7.3 add](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- `sub` has the same wording: "...In particular, mul/sub sequences with no rounding modifiers may be optimized to use fused-multiply-add instructions on the target device." — [PTX ISA 9.4, §9.7.3 sub](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- `mul` has the same wording: "Note that a mul instruction with an explicit rounding modifier is treated conservatively by the code optimizer. A mul instruction with no rounding modifier defaults to round-to-nearest-even and may be optimized aggressively by the code optimizer. In particular, mul/add and mul/sub sequences with no rounding modifiers may be optimized to use fused-multiply-add instructions on the target device." — [PTX ISA 9.4, §9.7.3.5 mul](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- The summary table (Table 29) gives this note for `{add,sub,mul}.rnd.f32`: "If no rounding modifier is specified, default is .rn and instructions may be folded into a multiply-add." — [PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- `fma` (§9.7.3.6): "Performs a fused multiply-add with no loss of precision in the intermediate product and addition." For sm_20+, `mad.f32` "is the same as" `fma.f32` (infinite-precision product, a single rounding). — [PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

### Inferences
- PTX that already contains `fma.rn.f32` is fixed: ptxas emits it as one FFMA. PTX `mul.f32`/`add.f32`/`sub.f32` with no `.rn` is a candidate for fusion in ptxas. PTX `mul.rn.f32`/`add.rn.f32` is never fused. So reproducing CUDA bit for bit takes two steps: (a) model NVVM's own `fma.rn` decisions, then (b) model ptxas's decisions on the remaining instructions without `.rn`. Step (b) is visible only in SASS.

### Gaps
- The PTX ISA does not say which mul/add pairs ptxas fuses: no single-use rule, no cross-basic-block rule, and no rule for picking an operand when both sides of an add are products.

## 2. How LLVM's NVPTX backend and NVVM decide contraction

### Takeaway
In upstream LLVM, contraction depends on the `contract` fast-math flag on both the fadd and the fmul, or on a global "allow FMA" switch. On NVPTX, two combiners can fire. The NVPTX target combine (`performFADDCombineWithOperands`) handles **FADD only**. It tries operand 0 first, then operand 1. It fuses a product with ≤4 users only if all of its users are FADD (when any user is not an FADD, an IR-order distance of at least 500 and a liveness condition are also required). This matches what was observed. A single-use product is fused. `m=a*b; m+c; m-d` is not, because the FSUB user counts as a non-add user. `a*b + c*d` becomes `fma(a,b, c*d)`, because operand 0 is tried first. The generic DAGCombiner (`visitFADDForFMACombine`) uses the same "N0 first" order, and NVPTX sets `enableAggressiveFMAFusion` = true. Upstream LLVM marks a product that is *not* contracted with `.rn` (mul.rn/add.rn) so that ptxas cannot fuse it. Plain `mul.f32` without `.rn` appears only when contraction was allowed but not done. The PTX described in the task (non-`.rn` mul/sub left in place) is therefore consistent with "contraction allowed globally (`-fmad=true`), but NVVM's heuristic declined it".

### Cited Findings
**Where the decision is made (NVPTXISelLowering.cpp, `main`):**
- Hidden option `nvptx-fma-level` (default `2`, described as "0: don't do it 1: do it 2: do it aggressively"). `allowFMA()` honours this option when it is given on the command line, returns false at `-O0`, and otherwise (on `main`) returns false. Contraction then depends on the `contract` flags. — [NVPTXISelLowering.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp)
- In LLVM 17.0.6, `allowFMA()` also returned true when `TargetOptions.AllowFPOpFusion == FPOpFusion::Fast` (that is, `-fp-contract=fast`) or `UnsafeFPMath` was set. That is the older "global" mechanism that NVVM's `-fmad=true` probably maps onto. — [LLVM 17.0.6 NVPTXISelLowering.cpp](https://github.com/llvm/llvm-project/blob/llvmorg-17.0.6/llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp)
- `NVPTXISelLowering.h`: `bool enableAggressiveFMAFusion(EVT VT) const override { return true; }` — [NVPTXISelLowering.h](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/NVPTX/NVPTXISelLowering.h)

**NVPTX's own FADD combine (`performFADDCombine` → `performFADDCombineWithOperands`):**
- Works on scalar f32/f64 FADD. "First try with the default operand order" `(N0, N1)`, then "try again with the operands commuted" `(N1, N0)`. If N0 is an FMUL and FMA is allowed (`allowFMA` **or** both the FADD and the FMUL carry `contract`), it emits `FMA(N0.op0, N0.op1, N1)`. — [NVPTXISelLowering.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp)
- Use heuristic (source comment): "Do the fusion only when the mul has less than 5 uses and all are add. The heuristic is that if a use is not an add, then that use cannot be fused into fma, therefore mul is still needed anyway. If there are more than 4 uses, even if they are all add, fusing them will increase register pressue." The code returns no combine when `numUses >= 5`. If any user is not an `ISD::FADD`, it fuses only when `orderNo - orderNo2 >= 500` (the IR-order distance between the add and the mul) **and** one of the mul's operands is live past the add (or is a constant). — [NVPTXISelLowering.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/NVPTX/NVPTXISelLowering.cpp)

**Generic DAGCombiner (`visitFADDForFMACombine` / `visitFSUBForFMACombine`, `main`):**
- `AllowFusionGlobally = HasFMAD` on `main`. In LLVM 17 it was `(Options.AllowFPOpFusion == FPOpFusion::Fast || Options.UnsafeFPMath || HasFMAD)`. "If the addition is not contractable, do not combine." — [DAGCombiner.cpp main](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp); [DAGCombiner.cpp 17.0.6](https://github.com/llvm/llvm-project/blob/llvmorg-17.0.6/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp)
- `isContractableFMUL`: the node is an FMUL and (`AllowFusionGlobally` or `N->getFlags().hasAllowContract()`). — [DAGCombiner.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp)
- Choosing an operand for FADD: "If we have two choices trying to fold (fadd (fmul u, v), (fmul x, y)), prefer to fold the multiply with fewer uses." The code is `if (Aggressive && ...) if (N0->use_size() > N1->use_size()) swap`. It then tries "fold (fadd (fmul x, y), z) -> (fma x, y, z)" on N0 first, then N1, with the condition `(Aggressive || N0->hasOneUse())`. — [DAGCombiner.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp)
- Re-association (`fadd (fma A,B,(fmul C,D)), E -> fma A,B,(fma C,D,E)`) happens only if the FADD has `reassoc` ("This requires reassociation because it changes the order of operations"). — [DAGCombiner.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp)
- FSUB: when both operands are contractable FMULs and N0 has more uses than N1, it folds `fma(-c, d, a*b)`. Otherwise it tries `fma(x, y, -z)` (fold the left product) first, then `fma(-y, z, x)`. — [DAGCombiner.cpp](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp)

**What the PTX looks like when not contracted (NVPTXInstrInfo.td):**
- Comment: "we use the '.rn' variant when FMA is disabled, as this behaves just like the non '.rn' op, but prevents ptxas from creating FMAs." The selection predicate for the plain form is `allowFMA() || N->getFlags().hasAllowContract()`. — [NVPTXInstrInfo.td](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/NVPTX/NVPTXInstrInfo.td)
- Test `fp-contract.ll`: ";; Without the contract flag we do not form fma.rn at the PTX level, and emit the explicit add.rn/mul.rn forms to prevent ptxas from fusing the ops." — [fp-contract.ll](https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/NVPTX/fp-contract.ll)

**Front-end defaults:**
- clang sets CUDA's default FP-contract mode to `FPM_Fast` ("Allow fuse across statements disregarding pragmas"). HIP uses `FPM_FastHonorPragmas`. — [clang LangOptions.cpp](https://github.com/llvm/llvm-project/blob/main/clang/lib/Basic/LangOptions.cpp)
- NVRTC documents `--fmad` as: "Enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA, or DFMA). --use_fast_math implies --fmad=true. Default: true" — [NVRTC docs](https://docs.nvidia.com/cuda/nvrtc/index.html)
- njuffa (NVIDIA forums, 2015) describes NVVM contraction this way: "it just walks the DAG constructed from the source expression and contracts FADD dependent on FMUL." He also says "FADD dependent upon FMUL will frequently get contracted to FMA ... in a release build." — [NVIDIA forum: Debug vs Release](https://forums.developer.nvidia.com/t/different-results-in-debug-and-release-mode-compile/39860)

### Inferences
- **The observed behaviour matches upstream NVPTX's FADD combine.**
  - Single-use `a*b + c` → fma: matches.
  - `a*b + c*d` → `fma(a,b, c*d)`: N0 is tried first, both products have one use, and the right-hand product stays a plain (rounded) mul. A front end that builds `a*b + c*d` as `fadd(fmul a b, fmul c d)` fuses the **left** product.
  - `a*b + c*d + e*f` parses as `(a*b + c*d) + e*f`. The inner add becomes `fma(a,b,c*d)`. The outer add's N0 is now an FMA, not an FMUL, so N1 = `e*f` is fused: `fma(e,f, fma(a,b,c*d))`. Matches.
  - `x*x+y*y+z*z` → `fma(z,z, fma(x,x, y*y))`: same pattern, matches.
  - Multi-use `m=a*b; m+c; m-d`: m has an FSUB user, so `nonAddCount>0`. The distance is far below 500, so it is not fused for either user. Matches.
- **Plain non-`.rn` `mul`/`sub` in NVVM's PTX:** `d1*d4 - d3*d2` stayed as `mul, mul, sub` (no `.rn`). Upstream NVPTX's target combine has no FSUB handler. The generic `visitFSUBForFMACombine` would fold it if it ran with contraction allowed. NVVM apparently did not, which suggests its pipeline differs from upstream for FSUB, or that the generic combiner's FMA formation is gated off. Because the PTX ops lack `.rn`, contraction was allowed globally (`-fmad=true`), and **ptxas is free to fuse `d1*d4 - d3*d2` in SASS** (for example to `FFMA(d1, d4, -(d3*d2))` or `FFMA(-d3, d2, d1*d4)`). Only SASS settles which.
- **Practical rule for a matching compiler:** apply the NVPTX FADD heuristic (operand 0 first, all users FADD, ≤4 users) to produce the PTX-level `fma.rn`. Then treat every remaining non-`.rn` mul→add/sub pair as "possibly fused by ptxas", and read the actual choice off the SASS for each kernel. NVVM's real rules may differ from upstream (closed source, different LLVM base version). The heuristic above is a hypothesis that agrees with all five observed cases, not documented NVVM behaviour.
- The order of fadd operands in the IR depends on how Warp's codegen emits expressions (`wp::add(wp::mul(a,b), wp::mul(c,d))` inlined). The "left product is fused" rule therefore applies to the order of operands after inlining.

### Gaps
- NVVM source is not public. I found no NVIDIA document stating NVVM's FADD/FSUB contraction heuristics, or which LLVM version the NVVM in CUDA 12.x/13.x is based on.
- I could not confirm why NVVM leaves FSUB-of-products unfused when upstream's generic combiner would fuse it.

## 3. ptxas rules for contracting non-`.rn` mul/add in SASS

### Takeaway
NVIDIA documents only the switch: ptxas has `--fmad`, "Semantics same as nvcc option --fmad". It does not document the rules ptxas uses to fuse. I found no public reverse-engineered rule set (single use, cross-block, operand choice). The documented ways to prevent fusion are `-fmad=false`, which also makes NVVM emit `.rn` ops, and the `__fmul_rn`/`__fadd_rn` intrinsics, documented as "never ... merged into a single multiply-add instruction". `fma()`/`fmaf()`/`__fmaf_rn` always produce FFMA, even under `-fmad=false`.

### Cited Findings
- ptxas option §4.2.9.1.10: "--fmad (-fmad) Semantics same as nvcc option --fmad." nvcc §4.2.7.11: "This option enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA, or DFMA)... Default: This option is set to true." — [nvcc docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- `__fmul_rn`: "Multiply two floating-point values in round-to-nearest-even mode... Note: This operation will never be merged into a single multiply-add instruction." `__fadd_rn` (and `_rd`/`_ru`/`_rz`) carry the same note. — [CUDA Math API, single intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)
- njuffa: "-fmad=false prevents the compiler from contracting an FMUL and a dependent FADD into an FMA. Calls to the standard math functions fma() and fmaf() will result in FMA instructions ... regardless of the setting of this switch. Such calls can occur inside inlined standard math functions." — [NVIDIA forum: FFMA still there with --fmad=false](https://forums.developer.nvidia.com/t/why-ffma-instructions-are-still-there-even-if-fmad-false-is-set/179934)
- njuffa: "Since FMA-contraction is an optimization, and debug builds are un-optimized, these kind of numerical discrepancies are common." — [NVIDIA forum](https://forums.developer.nvidia.com/t/different-results-in-debug-and-release-mode-compile/39860)
- Upstream LLVM states that ptxas may fuse non-`.rn` ops ("prevents ptxas from creating FMAs"). — [NVPTXInstrInfo.td](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/NVPTX/NVPTXInstrInfo.td)

### Inferences
- Warp's math library (`warp/native/*.h`) calls `fmaf` in some functions. Those calls produce FFMA regardless of `fuse_fp`, so a matching compiler must treat them as explicit FMA.
- To get deterministic CUDA output to match against, the robust options are: (a) `fuse_fp=False` (all `.rn`, no contraction anywhere, except explicit `fmaf`), or (b) record the SASS of the specific Warp kernels and mirror it. ptxas's fusion of leftover non-`.rn` ops is undocumented and may change between ptxas versions.

### Gaps
- No public document or reverse-engineering study of ptxas's FFMA formation rules (use count, basic-block scope, operand choice for `x*y - z*w`, interaction with register pressure). Determining them would require empirical SASS diffs.

## 4. Does NVIDIA document that `-fmad=true` results can differ across architectures or compiler versions?

### Takeaway
Yes, in general terms. The current CUDA Programming Guide (§5.5, Floating-Point Computation) defines bitwise determinism as conditional on the same GPU device *and* "the version of the compiler and the Compiler Flags and Optimizations". It also says contractions are allowed by default and that library results may differ "between host, device, and different device architectures". The floating-point whitepaper says FMA use "depends on ... how the code is compiled" and that intrinsics with an explicit rounding mode are "guaranteed to remain independent". Neither document specifically says that contraction placement changes across architectures, though that follows from the definition.

### Cited Findings
- "Deterministic Results: A deterministic result refers to computing the same bit-wise numerical outputs every time when run with the same inputs under the same specified conditions. Such conditions include: Hardware dependencies, such as execution on the same CPU processor or GPU device. Compiler aspects, such as the version of the compiler and the Compiler Flags and Optimizations..." — [CUDA Programming Guide §5.5](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- "To maximize the portability of results, users are recommended to use the default settings of the nvcc compiler's floating-point options: -ftz=false, -prec-div=true, and -prec-sqrt=true, and not use the --use_fast_math option. Note that floating-point expression re-associations and contractions are allowed by default, similarly to the --fmad=true option." — [CUDA Programming Guide §5.5](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- "CUDA provides the Fused Multiply-Add operation in several ways...: x * y + z when compiled with the flags -fmad=true or --use_fast_math; fma(x, y, z) and fmaf(x, y, z)...; __fmaf_[rd, rn, ru, rz]..." and "-fmad=true ... x * y + z → __fmaf_rn(x, y, z): 0 ULP". — [CUDA Programming Guide §5.5](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- "Library Implementations: Functions defined outside the IEEE-754 standard are not guaranteed to be correctly rounded... results may differ across different platforms, including between host, device, and different device architectures." The guide also recommends comparing optimized builds against `-G` builds. — [CUDA Programming Guide §5.5](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- Whitepaper "Floating Point and IEEE 754 Compliance for NVIDIA GPUs": FMA availability "depends on whether the platform provides the operation and also on how the code is compiled". §4.3 "Controlling Fused Multiply-add": "write the multiplies and additions using intrinsics with explicit rounding mode... Operations written directly as intrinsics are guaranteed to remain independent", and "It is also possible to disable FMA merging via a compiler flag." It also says "minor details such as organization of parentheses or thread counts can affect the final result." — [NVIDIA FP whitepaper](https://docs.nvidia.com/cuda/floating-point/index.html)

### Inferences
- **Warp-specific version risk.** Warp 1.16 by default produces **PTX** whenever the driver version is ≥ the toolkit version (`get_cuda_output_format`: "Older drivers may not be able to handle PTX generated using newer CUDA Toolkits, in which case we fall back on generating CUBIN modules"). It then loads that PTX with `cuModuleLoadDataEx`, so the **driver's JIT ptxas**, not the toolkit's, chooses the SASS-level fusions. The same Warp version and kernel can therefore give different bits after a driver upgrade even on the same L4. Setting `wp.config.cuda_output = "cubin"` pins ptxas to the NVRTC toolkit version. — [warp/_src/context.py v1.16.0](https://github.com/NVIDIA/warp/blob/v1.16.0/warp/_src/context.py)
- Warp's NVRTC options (from `warp/native/warp.cu` v1.16.0): `--gpu-architecture=compute_XX` (PTX) or `sm_XX` (CUBIN), `--std=c++17`, `--Ofast-compile=0` at optimization level 3 (CUDA ≥12.9), `--fmad=true|false` from `fuse_fp`, `--use_fast_math` only if `fast_math`, plus `--extra-device-vectorization`, `--restrict`, `--device-as-default-execution-space`. — [warp/native/warp.cu v1.16.0](https://github.com/NVIDIA/warp/blob/v1.16.0/warp/native/warp.cu)
- Warp documents `fuse_fp` as "Enable floating-point contraction (FMA fusion) during compilation, defaults to True." — [warp/_src/context.py v1.16.0](https://github.com/NVIDIA/warp/blob/v1.16.0/warp/_src/context.py)

### Gaps
- No NVIDIA statement specifically says "contraction placement may differ between sm_80 and sm_89" or "between ptxas versions". This is implied by the determinism conditions but not stated.

## 5. How to inspect, and tools for predicting contraction

### Takeaway
Read the PTX (Warp's kernel cache keeps `<module>.sm89.ptx` or `.cubin`) to see NVVM's `fma.rn` choices and leftover non-`.rn` ops. Read the SASS (`cuobjdump -sass` or `nvdisasm` on a cubin) to see ptxas's final FFMA/FMUL/FADD. No tool predicts contraction. The only reliable method is empirical: compile and disassemble.

### Cited Findings
- `cuobjdump --dump-sass (-sass)`: "Dump CUDA assembly for a single cubin file or all cubin files embedded in the binary". `--dump-ptx (-ptx)` also exists. `nvdisasm` "extracts information from standalone cubin files and presents them in human readable format", with `--print-line-info (-g)` and `--print-life-ranges (-plr)`. — [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
- Warp names cached outputs `{module}.sm{arch}{suffix}.ptx` or `.cubin` under `warp.config.kernel_cache_dir/<module>`. `cuda_output` can be forced to `"ptx"` or `"cubin"`. — [warp/_src/context.py v1.16.0](https://github.com/NVIDIA/warp/blob/v1.16.0/warp/_src/context.py)

### Inferences
- Workflow: set `wp.config.cuda_output="cubin"` to get exactly the SASS Warp runs, with ptxas taken from the NVRTC toolkit. Run `nvdisasm -g` (or `cuobjdump -sass`) on the cached cubin, and compare FFMA operand patterns against the PTX. Alternatively, keep the PTX and run `ptxas -arch=sm_89 -O3` from the same CUDA version as the driver's JIT. That assumes the driver's ptxas matches the toolkit's, which is not guaranteed. The upstream `llc` NVPTX heuristic in §2 can serve as a first-pass predictor for the PTX-level `fma.rn` placement only.

### Gaps
- No tool exists, public or NVIDIA, for predicting ptxas FFMA formation.
