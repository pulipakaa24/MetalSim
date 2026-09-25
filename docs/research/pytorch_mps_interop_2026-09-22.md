All the facts are gathered. Here is the report.

# PyTorch MPS / MLX interop research (as of 2026-09-22)

Sources: PyTorch `main` (raw.githubusercontent.com) and the `v2.14.0` tag, PyPI, GitHub API, MLX `main`, MLX docs. Anything not verified is labelled.

## 1. Latest stable PyTorch, Python coverage, install

- **Latest stable: torch 2.14.0**, GitHub release published **2026-09-02** (tag `v2.14.0`); PyPI wheels uploaded the same day. Previous: 2.13.0 (2026-07-08), 2.12.1 (2026-06-17).
- macOS arm64 wheels for 2.14.0 on PyPI (`https://pypi.org/pypi/torch/2.14.0/json`): `cp310`, `cp311`, `cp312`, `cp313`, `cp314`, `cp314t` — all tagged `macosx_14_0_arm64`. **Python 3.14: yes (incl. free-threaded 3.14t). 3.13: yes (3.13t wheel was dropped after 2.12.0). 3.12: yes.** `requires_python >= 3.10`; classifiers list 3.10–3.14. Note the wheel platform tag moved from `macosx_11_0` (2.11) to `macosx_14_0` (2.12+), so macOS 14+ is required — fine for macOS 26.
- Install: `pip install torch==2.14.0` (plain PyPI; the MPS backend is in the default macOS wheel — there is no separate index for Metal). pytorch.org's selector page is JS-rendered and I could not extract its exact text; conventionally it shows `pip3 install torch torchvision` for Mac/Pip/Default (unverified wording).
- Release-note highlights relevant to you (`https://github.com/pytorch/pytorch/releases/tag/v2.14.0`): native MPS `linalg.svd/eigh/qr/cholesky`, `flex_attention` improvements, "MPS-backed pinned memory correctly appear as a CPU tensor while retaining its shared Metal buffer (#181720)", read-only DLPack export (#188554), "Fix corrupted MPS prefill-attention output on macOS 26 by selecting the correct Metal cooperative-tensor ABI (#191794)", and a rename: `at::mps::is_macos_13_or_newer()` → `at::mps::is_macos_at_least(MacOSVersion::MACOS_15_0)` (#188645).

## 2. DLPack on MPS (`aten/src/ATen/DLConvertor.cpp`, main)

Device mapping is `kDLMetal` both ways:

```cpp
    case DeviceType::MPS:
      ctx.device_type = DLDeviceType::kDLMetal;      // torchDeviceToDLDevice, L152-153
...
    case DLDeviceType::kDLMetal:
      return at::Device(DeviceType::MPS, index);     // dlDeviceToTorchDevice, L190-191
```

**Export**: `DLTensor.data` is the storage base — i.e. the raw `id<MTLBuffer>` handle, *not* a host pointer — and the view offset goes in `byte_offset`:

```cpp
  if (src.device().type()  == kMPS) {
      atDLMTensor->tensor.dl_tensor.data = read_only
          ? const_cast<void*>(src.storage().data())
          : src.storage().mutable_data();
      atDLMTensor->tensor.dl_tensor.byte_offset = src.storage_offset() * c10::elementSize(src.scalar_type());
  } else { ... data = src.mutable_data_ptr(); byte_offset = 0; }
```

PR #182924 explains why: "PyTorch's MPS allocator encodes `id<MTLBuffer>` directly into `c10::DataPtr.data` … so for any sliced/viewed MPS tensor `data_ptr()` returns `id<MTLBuffer> + storage_offset*elemsize` — pointer arithmetic on an Objective-C object pointer, which is not a valid buffer handle." `test/test_dlpack.py` L852: "on MPS, DLTensor.data is an opaque id<MTLBuffer>". Caveat: the fix for the **non-owning** C exchange API (`__dlpack_c_exchange_api__`, `toDLPackNonOwning`) landed on main 2026-09-12 (commit `3f24ada281`), *after* 2.14.0; in 2.14.0 only the owning `__dlpack__` path has the MPS special-case (verified by grepping the tag: L425/L429 present, non-owning fix absent).

**Import** (`fromDLPackImpl`): the data pointer is handed straight to `at::from_blob` with the device forced to MPS; no validation of the handle:

```cpp
  Device device = dlDeviceToTorchDevice(dl_tensor.device.device_type, dl_tensor.device.device_id, dl_tensor.data);
  ...
  return at::from_blob(dl_tensor.data, IntArrayRef(dl_tensor.shape, dl_tensor.ndim),
      IntArrayRef(dl_tensor.strides, dl_tensor.ndim),
      toStorageOffset(dl_tensor.byte_offset, stype), deleter,
      at::device(device).dtype(stype), {device});
```

`TensorMaker::make_tensor` (`aten/src/ATen/templates/Functions.cpp`) just wraps the pointer in a `DataPtr`/`Storage` — so **yes, `at::from_blob` works for MPS with a foreign `id<MTLBuffer>`**, because every MPS kernel reads the buffer via:

```cpp
static inline id<MTLBuffer> getMTLBufferStorage(const TensorBase& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());   // OperationUtils.h L124
}
[encoder setBuffer:getMTLBufferStorage(t) offset:t.storage_offset() * t.element_size() atIndex:idx];
```

Allocator lookups on foreign buffers degrade gracefully: `getUnalignedBufferSize` returns `-1` ("indicates the passed buffer pointer wasn't found"), `isSharedBuffer` returns false, `getSharedBufferPtr` returns `{nullptr,0}`. Exceptions: `_host_alias_storage` raises for non-allocator buffers ("storage was not allocated by the MPSAllocator"), and MLX-style `[buffer contents]` reads require a non-private storage mode. Lifetime: `from_blob`'s deleter is the only thing keeping the producer's buffer alive — retain the `MTLBuffer` and release it in the deleter. DLPack carries **no synchronization** on Metal (the `dlpack.h` stream arg is CUDA-only); see §5.

## 3. `torch.mps` Python API and the C++ API (main == v2.14.0 for everything below)

Python (`torch/mps/__init__.py`):
- `compile_shader(source: str)` → `_mps_ShaderLibrary` (embeds `c10/metal/*.h` headers, compiles via `DynamicMetalShaderLibrary`); kernels are called as `lib.kernel_name(tensor_or_scalar_args..., threads=..., group_size=..., arg_casts=..., error_buf_idx=...)` (binding `_mps_MetalKernel.__call__`; `threads` defaults to first tensor's `numel()`).
- `load_metallib(bytes | path)` → `PrecompiledMetalShaderLibrary` ("compiled ahead of time or generated by external tools (e.g. Triton, MetalASM)").
- `_host_alias_storage(storage)` → CPU `UntypedStorage` aliasing the shared `MTLBuffer` (`[buffer contents]`); requires MPSAllocator-owned, shared-mode buffer; docstring mandates `torch.mps.synchronize()` before and after host access.
- `synchronize()`, `Event(enable_timing)` with `record/wait/query/synchronize/elapsed_time` (wraps an `MTLSharedEvent`, handle not exposed), `profiler.metal_capture(path)` (gputrace, requires `MTL_CAPTURE_ENABLED`).
- **No Python API returns an `MTLBuffer`, command buffer, or queue**; the full `_mps_*` binding list has only: `acquireEvent compileShader currentAllocatedMemory deviceSynchronize driverAllocatedMemory elapsedTimeOfEvents emptyCache get_core_count get_default_generator get_name host_alias_storage is_available is_in_bad_fork is_on_macos_or_newer isCaptureEnabled isCapturing loadMetallibFromPath loadMetalllib maxBufferLength MetalKernel PrecompiledShaderLibrary profilerStartTrace profilerStopTrace queryEvent recommendedMaxMemory recordEvent releaseEvent setMemoryFraction setStream ShaderLibrary startCapture stopCapture synchronizeEvent waitForEvent` (`_mps_allocator_waitForEvents` is main-only, test hook). The raw `id<MTLBuffer>` is reachable from Python as `t.untyped_storage().data_ptr()` (it's the ObjC pointer, per §2) — usable via ctypes but unofficial.

C++ (`torch/csrc/api/include/torch/mps.h`, `#include <torch/mps.h>`):
```cpp
bool is_available(); void manual_seed(uint64_t); void synchronize();
void commit();                             // getDefaultMPSStream()->synchronize(SyncType::COMMIT)
MTLCommandBuffer_t get_command_buffer();   // endKernelCoalescing() then stream->commandBuffer()
DispatchQueue_t get_dispatch_queue();      // default stream's serial dispatch_queue_t
```
`aten/src/ATen/mps/MPSStream.h`: `class MPSStream { commandQueue(); queue(); commandBuffer() /*MPSCommandBuffer*/; commandEncoder(); endKernelCoalescing(); synchronize(SyncType{NONE,COMMIT,COMMIT_AND_WAIT,COMMIT_AND_CONTINUE,COMMIT_ADAPTIVE}); copy(); copy_and_sync(); executeMPSGraph(); addCompletedHandler(MTLCommandBufferHandler); device(); }` — `commit/commitAndWait/commitAndContinue/flush` are **private** ("use synchronize()"). Free functions: `getCurrentMPSStream()`, `setCurrentMPSStream()`, `getDefaultMPSStream()`, `getStreamFromPool()` (32 streams), `getStreamByID()`, `synchronizeAllMPSStreams()`, `dispatch_sync_with_rethrow()`. There is **no `encodeSignalEvent` method on MPSStream**; `aten/src/ATen/mps/MPSEvent.h` has `MPSEvent::record/wait/query/synchronize` on a private `MTLSharedEvent_t m_event`, obtained via `getMPSEventPool()->acquireEvent(enable_timing, stream)`. `IMPSAllocator` (`MPSAllocatorInterface.h`) adds `getHostAliasStorage`, `recordEvents/waitForEvents(buffers)`, `getMPSPinnedAllocator()`, `getMPSPinnedMTLBuffer(host_ptr)`. `MetalShaderLibrary.h` gives extensions `DynamicMetalShaderLibrary(src)` / `PrecompiledMetalShaderLibrary(bytes|path)` → `getKernelFunction(name)` → `MetalKernelFunction::runCommandBlock([&]{ startEncoding(); setArg(i, tensor|scalar|container); dispatch(threads, group); })`.

`torch.utils.cpp_extension` registers `.mm` as a source extension (`cpp_extension.py` L908-914); build with `extra_compile_args={'cxx': ['-ObjC++', ...]}` and `-framework Metal -framework MetalPerformanceShaders`. Official in-tree example: `test/cpp_extensions/mps_extension.mm`.

## 4. Graph capture / torch.compile on MPS

- **No CUDA-Graph equivalent exists for MPS.** No `torch.mps.graph`, no ICB/replay API; `_mps_startCapture/_mps_isCapturing` are Xcode GPU-trace capture (profiler), not graph capture. Inductor explicitly treats MPS as a GPU without streams: `device_need_guard`: "MPS is a GPU but does not [expose streams], so it must be excluded here" (`torch/_inductor/utils.py`), so cudagraph-trees never applies. The internal `MPSGraphCache` (cleared by `empty_cache`) is a per-op MPSGraph kernel cache, not user-facing.
- **Inductor MPS backend exists** (`torch/_inductor/codegen/mps.py`, ~1330 lines, `MetalKernel`/`MetalScheduling`, generates Metal shaders incl. welford reductions) but its own header says: "This is not a feature-complete compiler backend / Just an early prototype that shows that one can compile elementwise ops into a Metal shader". Tracker issue #150121 still lists unfinished items (multi-stage welford, dynamic shapes, argument buffers disabled, matmul decomps disabled); the docs statement "early prototype and attempt to use it to accelerate end-to-end network is likely to fail" is still current. 2.14 notes add fixes for compiled MPS ops (#192020) and `flex_attention` compile on MPS, so `torch.compile(backend="inductor")` on `mps` works for pointwise/reduction fusion but is not a performance path for a full PPO update.

## 5. Custom extension: zero-copy tensor from MTLBuffer and event ordering

(a) **Tensor from existing `MTLBuffer`, no copy** — semi-official, via the exact path DLPack uses:
```objc
id<MTLBuffer> buf = ...;  [buf retain];
auto t = at::from_blob((void*)buf, sizes, strides, /*storage_offset*/ byteOff/elem,
                       [buf](void*){ [buf release]; },
                       at::TensorOptions().device(at::kMPS).dtype(at::kFloat), {at::kMPS});
```
Or produce a `DLManagedTensorVersioned` with `device={kDLMetal,0}`, `data=(void*)buf`, `byte_offset`, and let `torch.from_dlpack` do it — this is what MLX ships today (§6), so the convention is exercised in the wild. No example in `pytorch/extension-cpp` (it has only CPU/CUDA `muladd`); examples: `test/cpp_extensions/mps_extension.mm`, github.com/smrfeld/pytorch-cpp-metal-tutorial, the TDVault "PyTorch MPS – Correct Metal Kernel Setup" note (all use `getMTLBufferStorage`/`get_command_buffer`).

(b) **MTLEvent/MTLSharedEvent on torch's command buffer** — no dedicated API, but composable from the in-tree pattern:
```objc
dispatch_sync(torch::mps::get_dispatch_queue(), ^{
  id<MTLCommandBuffer> cb = torch::mps::get_command_buffer(); // ends torch's open encoder
  [cb encodeWaitForEvent:simEvent value:stepId];              // wait for simulator
  auto enc = [cb computeCommandEncoder]; ... [enc endEncoding];
  [cb encodeSignalEvent:learnerEvent value:stepId];
  torch::mps::commit();
});
```
`get_command_buffer()` calls `endKernelCoalescing()` first so the buffer is in a state where event encoding is legal; `commit()` submits the current buffer (COMMIT sync type). Torch's MPSCommandBuffer is created on the stream's own `MTLCommandQueue` (`stream->commandQueue()`), so cross-queue ordering with your simulator must go through `MTLSharedEvent` (or a shared queue). `MPSStream::addCompletedHandler` gives a completion callback. Alternatively, run the simulator on `torch`'s queue (`getCurrentMPSStream()->commandQueue()`) and encode into `commandEncoder()` directly (the `get_mps_add_output` example) — then ordering is implicit and no events are needed.

## 6. MLX

- Latest **mlx 0.32.2** (PyPI, 2026-08-25). **Zero-copy Metal DLPack with torch MPS landed in MLX PR #3531 (merged 2026-06-23)**; docs (`usage/numpy.html`): "MLX arrays exported to PyTorch with DLPack are exported without a copy on Metal. If a PyTorch MPS tensor is passed to `mx.asarray` or to `mx.from_dlpack` with `copy=None`, MLX imports it without a copy when the underlying Metal buffer is not private" and "DLPack conversion does not synchronize pending Metal work; synchronize or evaluate the producing framework before reading". `python/src/convert.cpp`: export uses `data = arr.buffer().ptr()` (the `MTL::Buffer*`) plus `byte_offset = arr.offset()`; import calls `allocator::can_reuse_alien_buffer(ptr)` = `buf->storageMode() != MTL::StorageModePrivate`, then `out.set_data(mx::allocator::Buffer(data_handle), …, nd_array.byte_offset(), [owner]{})`. MLX #4497 (2026-09) fixed a GIL hang on copying imports. vllm-metal PR #758 (2026-09-20) adopted this bridge in production.
- Python `mlx.core.metal` exposes only `is_available, device_info, start_capture, stop_capture` — **no buffer/queue/event handles from Python**. C++ does: `mlx::core::metal::Device::{mtl_device(), get_command_buffer(), get_command_encoder(), end_encoding(), get_kernel()}` (`mlx/backend/metal/device.h`), `mlx::core::Event::{wait, wait(stream), signal(stream), is_signaled}` (`mlx/event.h`, backed by `MTLSharedEvent` on Metal), `array::buffer().ptr()` → `MTL::Buffer*`, and `allocator::make_buffer(ptr, size)` (wraps host memory via `newBuffer(ptr,size,...)` no-copy).
- Assessment: MLX's lazy graph + `mx.compile` gives kernel fusion that Inductor-MPS doesn't reliably deliver, and its DLPack path is exactly the `MTL::Buffer*` + `byte_offset` convention, so a Metal simulator can feed either framework identically. But MLX has no Python-level event/queue access either, the RL ecosystem (optimizers, distributions, Isaac Lab-style code) is torch-native, and MLX arrays may "rebind to new buffers" on later ops (PR #3531 caveat). Labelled opinion: keep torch as the learner, design the simulator's export as a DLPack-Metal producer so MLX remains a drop-in option.

## Recommended approach (ranked by confidence)

1. **High** — Simulator owns `MTLBuffer`s (shared storage mode); expose them to torch via `DLManagedTensorVersioned{device=kDLMetal, data=id<MTLBuffer>, byte_offset}` → `torch.from_dlpack`, or via `at::from_blob` in a `.mm` extension. Verified by torch source, tests, and MLX's shipping implementation.
2. **High** — Ordering via a `.mm` extension using `get_dispatch_queue()/get_command_buffer()/commit()` with `encodeWaitForEvent`/`encodeSignalEvent` on an `MTLSharedEvent` shared with the simulator; or run the simulator's dispatches on torch's own `MPSStream` command buffer to avoid events entirely.
3. **Medium** — `_host_alias_storage` + MPS pinned memory for bulk host-side observation writes (allocator-owned buffers only; manual `synchronize()` discipline).
4. **Low** — `torch.compile` on MPS for hot pointwise chains only; do not plan on it for whole-step fusion, and there is no graph replay on MPS.
5. **Optional** — MLX as an alternative learner via the same DLPack-Metal handles (mlx ≥ 0.29, PR #3531); no Python event API, so sync stays `mx.eval()`/`torch.mps.synchronize()` based.

Key files (local copies in `/private/tmp/claude-501/-Users-aditya/7ceab845-25d9-459e-a466-a4b793f72f5c/scratchpad/`): `DLConvertor.cpp`, `mps_init.py`, `MPSStream.h`, `MPSEvent.h`, `MetalShaderLibrary.h`, `torch_mps.h`, `mps_Module.cpp`, `mps_extension.mm`, `MPSAllocatorInterface.h`, `mlx_convert.cpp`, `mlx_device.h`, `mlx_event.h`.