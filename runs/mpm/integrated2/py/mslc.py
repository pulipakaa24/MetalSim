"""Compile an MSL source as Warp does (wp_metal_load_library / wp_metal_get_kernel) and build one pipeline.
usage: mslc.py SOURCE.metal FUNCTION [NAME=VALUE ...]   (each NAME=VALUE becomes a #define at the top)
exit 0: pipeline created; 1: front-end compile error; 2: pipeline creation failed; 3: no such function"""
import ctypes, sys, time
import Metal

src_path, fname = sys.argv[1], sys.argv[2]
defines = [a.split("=", 1) for a in sys.argv[3:] if "=" in a]
src = "".join(f"#define {k} {v}\n" for k, v in defines) + open(src_path).read()
dev = Metal.MTLCreateSystemDefaultDevice()
opts = Metal.MTLCompileOptions.alloc().init()
opts.setLanguageVersion_(Metal.MTLLanguageVersion3_2)
opts.setMathMode_(Metal.MTLMathModeSafe)
opts.setMathFloatingPointFunctions_(Metal.MTLMathFloatingPointFunctionsPrecise)
lib, err = dev.newLibraryWithSource_options_error_(src, opts, None)
if lib is None:
    print("COMPILE_ERROR", str(err)[-1500:]); sys.exit(1)
consts = Metal.MTLFunctionConstantValues.alloc().init()
slot = ctypes.c_uint64(0)
consts.setConstantValue_type_atIndex_(ctypes.addressof(slot), Metal.MTLDataTypeULong, 0)
fn, err = lib.newFunctionWithName_constantValues_error_(fname, consts, None)
if fn is None:
    fn = lib.newFunctionWithName_(fname)
if fn is None:
    print("NO_FUNCTION", err); sys.exit(3)
desc = Metal.MTLComputePipelineDescriptor.alloc().init()
desc.setComputeFunction_(fn)
desc.setSupportIndirectCommandBuffers_(True)
t0 = time.time()
res = dev.newComputePipelineStateWithDescriptor_options_reflection_error_(desc, Metal.MTLPipelineOptionNone, None, None)
if res[0] is None:
    print(f"PIPELINE_FAIL ({time.time() - t0:.1f}s)", str(res[-1])[:400]); sys.exit(2)
print("OK")
