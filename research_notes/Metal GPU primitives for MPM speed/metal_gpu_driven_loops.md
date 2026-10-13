# Metal GPU-driven loops: emulating CUDA graph conditional (WHILE) nodes on Apple GPUs

Scope: how to run a data-dependent iterative solver loop (5-iteration Gauss-Seidel chunks of about 45 dispatches, repeated until a GPU-computed convergence flag says stop) and replay a whole simulation step (about 1,500 dispatches) on an M-series GPU without a host readback per check. Researched 2026-10-10. Sources are mostly Apple documentation (.md endpoints), Apple Developer Forums, and a few third-party write-ups. Apple's "Encoding indirect command buffers on the GPU" sample page came back empty when fetched, so its body is not quoted here.

## Q0. The semantics to emulate (CUDA graph conditional nodes)

### Takeaway
A CUDA WHILE node re-runs its body graph on the device for as long as a device-side condition value, set by a kernel through `cudaGraphSetConditional`, stays non-zero. The CPU is never involved. Metal has no equivalent node type, so the loop has to be emulated with a bounded, unrolled structure.

### Cited Findings
- CUDA 12.4 added IF and WHILE conditional nodes. A WHILE node repeats its body while the condition is true. CUDA 12.8 added IF/ELSE and SWITCH. Conditional nodes can nest. — [NVIDIA blog: Dynamic control flow in CUDA graphs with conditional nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes)
- The condition is reached through a handle created with `cudaGraphConditionalHandleCreate`, which can take a default value applied at the start of each graph launch. Kernels set it with `cudaGraphSetConditional`. The body graph can be filled through stream capture (`cudaStreamBeginCaptureToGraph`). — [NVIDIA blog](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes); [CUDA Graphs conditional nodes slides (HiHAT)](https://hihat.opencommons.org/images/c/ca/CUDA_Graphs_conditional_nodes.pdf)
- In cuda-python, a while-loop node runs its body repeatedly while the condition evaluates to non-zero. — [cuda.core ConditionalNode docs](https://nvidia.github.io/cuda-python/cuda-core/1.1.0/generated/cuda.core.graph.ConditionalNode.html)
- NVIDIA's `graphConditionalNodes` sample has a WHILE example in which a kernel decrements a device counter and calls `cudaGraphSetConditional` with the result. — [graphConditionalNodes.cu mirror](https://turing.une.edu.au/~cosc330/lectures/cuda-samples/Samples/3_CUDA_Features/graphConditionalNodes/graphConditionalNodes.cu)

### Inferences
- A CUDA WHILE node is unbounded. On Metal the practical emulation is a bounded loop: unroll to a maximum of N chunks and let GPU-written data turn the chunks after convergence into no-ops. The emulation should record whether the cap was reached (a GPU-side counter) so the caller can detect non-convergence, the way Warp code would check a final iteration count.

### Gaps
- I did not fetch Warp's `wp.capture_while` source or Newton's implicit MPM loop. Their exact iteration-cap and convergence-check placement is not verified here.

## Q1. Indirect command buffers (ICBs) for compute: GPU encoding, GPU-sourced execution range, limits, and early exit

### Takeaway
Metal can take the execution range of an ICB from a GPU buffer: `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:` on `MTLComputeCommandEncoder` (macOS 11+), and `executeCommands(buffer:indirectBuffer:)` on `MTL4ComputeCommandEncoder` (macOS 26+). The range is a `{uint32 location, uint32 length}` struct that can be "calculated in the GPU timeline". This is the documented primitive for GPU-driven skipping. A convergence kernel can write `length = 0` into the range slots of later chunks so those chunks do nothing. The range is capped at 16,384 commands per execute call, and an ICB cannot contain an "execute ICB" command, so loop control has to sit at the encoder level as N unrolled execute calls.

### Cited Findings
- `MTLComputeCommandEncoder executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:` (iOS 13 / macOS 11+) "Encodes an instruction to run commands from an indirect buffer, using another buffer to provide the command range". The range buffer uses the `MTLIndirectCommandBufferExecutionRange` layout. "The maximum length of the range is 16384 commands." The offset must be a multiple of 4. — [Apple docs: executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/executecommandsinbuffer:indirectbuffer:indirectbufferoffset:.md)
- Metal 4 has the same capability as `MTL4ComputeCommandEncoder.executeCommands(buffer:indirectBuffer:)` (macOS 26+), which takes an `MTLGPUAddress` for the range (4-byte aligned): "This allows you to calculate the span of commands Metal executes in the GPU timeline, enabling GPU-driven workflows." The range buffer must be in an `MTLResidencySet`, as must any pipeline states the ICB references, whether set with `setComputePipelineState` on the CPU or `set_compute_pipeline_state()` during GPU encoding. — [Apple docs: MTL4ComputeCommandEncoder executeCommands(buffer:indirectBuffer:)](https://developer.apple.com/documentation/metal/mtl4computecommandencoder/executecommands(buffer:indirectbuffer:).md)
- `MTLIndirectCommandBufferExecutionRange` is a C struct of two `uint32` fields, `location` and `length`. — [objc2-metal generated bindings (Firefox searchfox)](https://arai.searchfox.org/firefox-main/source/third_party/rust/objc2-metal/src/generated/MTLIndirectCommandBuffer.rs)
- The objc2 binding marks the execution range as possibly not bounds-checked, and notes that the ICB may need synchronization and must stay alive while in use. This is a third-party annotation, not Apple's. — [objc2-metal MTL4ComputeCommandEncoder bindings](https://kdab.searchfox.org/firefox-main/source/third_party/rust/objc2-metal/src/generated/MTL4ComputeCommandEncoder.rs)
- The legacy direct-range render method `executeCommandsInBuffer:withRange:` also limits the count to 0x4000 (16,384). — [Apple docs: MTLRenderCommandEncoder executeCommandsInBuffer:withRange:](https://developer.apple.com/documentation/metal/mtlrendercommandencoder/executecommandsinbuffer:withrange:.md)
- ICBs are reusable across submissions, unlike single-use `MTLCommandBuffer`s. They can be encoded on the CPU or the GPU, and the GPU path lets "the output of one pass" drive "a subsequent pass". — [Apple docs: Indirect command encoding](https://developer.apple.com/documentation/metal/indirect-command-encoding.md)
- `MTLIndirectCommandBufferDescriptor` provides `commandTypes`, `inheritPipelineState`, `inheritBuffers`, `maxKernelBufferBindCount`, `maxKernelThreadgroupMemoryBindCount` and `supportRayTracing`. The ICB size (maximum number of commands) is set when the ICB is created (`MTLIndirectCommandBuffer.size`). — [Apple docs: MTLIndirectCommandBufferDescriptor](https://developer.apple.com/documentation/metal/mtlindirectcommandbufferdescriptor.md); [Apple docs: MTLIndirectCommandBuffer](https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer.md)
- `MTLIndirectComputeCommand` supports `setComputePipelineState`, `setKernelBuffer(offset:at:)` (plus an `attributeStride` variant), `setThreadgroupMemoryLength`, `setStageInRegion`, `setImageblockWidth`, `concurrentDispatchThreadgroups` / `concurrentDispatchThreads`, `setBarrier()` / `clearBarrier()` ("ensure that commands executed prior to this command are complete before this command executes") and `reset()`. The ICB also exposes `reset(range)` and a `gpuResourceID` for binding it into argument buffers so GPU kernels can encode into it. — [Apple docs: MTLIndirectComputeCommand](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand.md); [Apple docs: MTLIndirectCommandBuffer](https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer.md)
- Only concurrent dispatch variants exist in ICB compute commands. Ordering between dependent commands is expressed with per-command barriers (`setBarrier`). — [Apple docs: MTLIndirectComputeCommand](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand.md)
- GPU-side encoding in MSL: a forum example uses `compute_command cmd(args->icb, index); cmd.concurrent_dispatch_threads(...)` inside a kernel. The poster first saw no commands in the GPU debugger and then a bad access on replay, and later reported that the commands were in fact being encoded. — [Apple Developer Forums: Encoding indirect thread dispatch on GPU](https://developer.apple.com/forums/thread/677874)
- The `MTLIndirectComputeCommand` method list contains no "execute another ICB" command, so ICBs cannot nest or loop themselves. — [Apple docs: MTLIndirectComputeCommand](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand.md)

### Inferences
- **Recommended loop construction (Metal 3 and Metal 4 alike):**
  1. Encode the 45-dispatch Gauss-Seidel chunk once into an ICB (CPU-encoded is fine: pipelines, buffers and grid sizes are static). Set `setBarrier()` on each dependent command, or use the encoder's serial dispatch if the ICB is executed in a serial encoder. Barrier semantics inside an ICB in a serial encoder still need a check.
  2. On the host, encode a fixed unrolled sequence for N_max chunks: `executeCommands(chunkICB, rangeBuf + 8*i)` followed by a one-thread "check" dispatch that reads the residual and, if converged, writes `length = 0` into all later range slots (i+1 to N_max-1). The check dispatch can also increment an iteration counter.
  3. Reset the range slots to `{0, 45}` at the start of each step with a tiny kernel or blit. This mirrors CUDA's default condition value being re-applied at each launch.
  4. The whole step (about 1,500 commands plus N_max × 46 for the loop) stays inside one command buffer with no host round-trip. The 16,384-commands-per-execute cap is per execute call, not per command buffer.
- With this scheme each skipped chunk costs one executeCommands with a zero-length range plus one tiny check dispatch, instead of 45 empty dispatches. That is likely the cheapest documented early-exit available. The per-skipped-chunk overhead is unmeasured (see Gaps), and should be benchmarked against the current 0.23 ms readback every 5 iterations.
- An alternative to range rewriting is for a GPU "controller" kernel to `reset()` the commands of later chunks inside one big ICB (MSL `compute_command::reset()`), which turns them into empty slots. This also works, but rewriting the range is simpler and avoids encoding PSOs and buffers from the GPU.
- Whole-step capture (CUDA-graph replay) maps to "encode the step once into one or more ICBs and replay with executeCommands each frame". ICBs are reusable, and per-step variable parameters must then live in buffers rather than in setBytes. Host-side utilities that break capture (allocations, host reads) must be hoisted out or replaced with GPU kernels, as on CUDA.

### Gaps
- No Apple document states exactly when the GPU reads the indirect range relative to preceding dispatches, that is, whether a write by the immediately preceding check kernel is guaranteed visible to the next `executeCommands`. Treat it like an indirect-dispatch argument: insert a barrier (Metal 3: serial encoder or `memoryBarrierWithScope:MTLBarrierScopeBuffers`; Metal 4: `barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch`, with the indirect-argument read possibly needing a different stage) and verify it empirically.
- No published latency numbers for ICB execution or zero-length ranges on Apple Silicon were found.
- The authoritative MSL signatures for `command_buffer` / `compute_command` (MSL spec, ICB section) were not retrieved. Only a forum example was.

## Q2. Indirect dispatch (`dispatchThreadgroupsWithIndirectBuffer`) with zero threadgroups to no-op the loop tail

### Takeaway
Indirect dispatch reads threadgroup counts from a GPU buffer (`MTLDispatchThreadgroupsIndirectArguments`, 3 × uint32). A convergence kernel can zero those counts so the remaining dispatches do no work. However, each command is still processed and pays the fixed per-dispatch cost, so 45 × (N_max − n) empty dispatches cost far more than ICB range skipping. No measurement of empty-dispatch cost on M-series was found.

### Cited Findings
- `MTL4ComputeCommandEncoder.dispatchThreadgroups(indirectBuffer:threadsPerThreadgroup:)` takes the threadgroup counts from a buffer laid out as `MTLDispatchThreadgroupsIndirectArguments` at a 4-byte aligned address, which lets the grid size be computed "on the GPU timeline". (The Metal 3 `MTLComputeCommandEncoder` has the equivalent `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`; its doc page was not fetched.) — [Apple docs: MTL4ComputeCommandEncoder dispatchThreadgroups(indirectBuffer:threadsPerThreadgroup:)](https://developer.apple.com/documentation/metal/mtl4computecommandencoder/dispatchthreadgroups(indirectbuffer:threadsperthreadgroup:).md)
- Apple's indirect command encoding topic lists `MTLDispatchThreadgroupsIndirectArguments` as the layout "needed to specify the size of threadgroups". — [Apple docs: Indirect command encoding](https://developer.apple.com/documentation/metal/indirect-command-encoding.md)

### Inferences
- Using the user's measured 3–9 µs per dependent dispatch, 45 empty dispatches per skipped chunk could cost about 0.13–0.4 ms, which is as bad as the 0.23 ms readback it replaces. Zero-size indirect dispatch is therefore a fallback, not the primary mechanism. It is useful only when indirect sizing is needed anyway, or for a single "gate" dispatch. A zero-threadgroup dispatch may cost less than a dependent full dispatch because it has no work to drain, but this is unverified.
- Combining both mechanisms is possible: the ICB range for coarse skipping, and indirect dispatch for data-dependent grid sizes such as active-particle counts.

### Gaps
- I found no measured cost of a zero-threadgroup dispatch on M1–M4. A microbenchmark is needed.

## Q3. Persistent kernels and device-wide barriers; Metal 4 features

### Takeaway
Since Metal 3.2 (macOS 15), MSL has device-scope fences (`atomic_thread_fence` with `thread_scope_device`) and `coherent(device)` memory, so threadgroups can communicate through device memory. Apple documents no threadgroup co-residency or global forward-progress guarantee, though, so a grid-wide spin barrier (cooperative-groups style) can deadlock or trip the GPU watchdog. It should not be used in production. Metal 4 cuts CPU encoding cost (argument tables, residency sets, a unified compute encoder that is concurrent by default with explicit stage barriers) and adds GPU-address-based indirect execution. It does not add GPU-side loops or conditional nodes.

### Cited Findings
- An Apple Graphics and Games engineer (accepted answer, June 2024): "In Metal 3.2, we've introduced `atomic_thread_fence` with a `thread_scope` (§6.15.2 and §6.15.3 of the MSL Specification). You can use a `thread_scope_device` fence along with `device` atomics to synchronize coherent(device) (§4.8) memory operations across thread groups." — [Apple Developer Forums thread 756663](https://developer.apple.com/forums/thread/756663)
- Community discussion of Metal 3.2: the feature lets shaders on different GPU cores synchronize, but efficient work queues also need a forward-progress guarantee. Before 3.2 the same effect required chaining small kernels at a performance cost. — [TechBoards: Metal 3.2 new features](https://techboards.net/threads/metal-3-2-new-features.4732)
- A June 2026 forum question (no replies) reports that the author could find no documented guarantee for threadgroup co-residency, global forward progress, or occupancy-bounded synchronization on Apple GPUs. In their experiments, schemes that assume all threadgroups eventually make progress were unreliable, while local producer/consumer handoffs were more robust. — [Apple Developer Forums thread 831017](https://developer.apple.com/forums/thread/831017)
- An indeterminate `while` loop in a Metal shader caused an IOAF code 5 GPU error (watchdog/timeout) on DTK and M1. — [Apple Developer Forums thread 672532](https://developer.apple.com/forums/thread/672532)
- An MLX bug report describes an orphaned spin-wait "fence" kernel that occupied the GPU compute engine and blocked later Metal work until reboot. Without the env var, the workload hit the macOS GPU watchdog. — [MLX issue 3830](https://upd.dev/ml-explore/mlx/issues/3830) (mirror; not the github.com original)
- Academic work: a general inter-workgroup barrier can deadlock through starvation unless it relies on scheduling guarantees specific to a GPU. — [Imperial College: Inter-workgroup barrier synchronisation on GPUs](https://spiral.imperial.ac.uk/entities/publication/5efadcbf-8cc4-4b55-8e46-7a246ca65229)
- Metal 4 (WWDC25 session 254): two encoder classes. The compute encoder covers dispatches, blits and acceleration-structure builds, and its commands run concurrently by default without extra synchronization. Ordering uses stage-filtered intra-pass barriers, for example `barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice`. Argument tables bind root resources by index. Residency sets attach to a queue or command buffer. Guidance is to reduce encoder count and reuse command-allocator memory. The session gives no per-dispatch overhead numbers and does not cover ICBs. — [WWDC25 session 254](https://developer.apple.com/videos/play/wwdc2025/254/)
- WWDC25 "Discover Metal 4" (session 205): `MTL4ArgumentTable` holds only the bind points the app needs, and in a bindless design one buffer binding is enough. — [WWDC25 session 205](https://developer.apple.com/wwdc25/205); [DEV Community recap](https://dev.to/arshtechpro/wwdc-2025-discover-metal-4-23f2)

### Inferences
- A "persistent kernel with a grid barrier" that is limited to an occupancy-safe number of threadgroups (≤ number of cores × resident threadgroups per core) might work in practice on a given M4. It is undocumented, though, can hang the GPU (watchdog or reboot per the MLX report), and Gauss-Seidel colouring over a large grid would need many grid barriers per iteration. Not recommended over the ICB approach.
- A safer persistent-style variant is to fuse within a threadgroup: if a Gauss-Seidel colour sweep can be tiled so that each tile's iterations only need threadgroup barriers (block-Jacobi / local GS with halo exchange per chunk), several iterations can run in one dispatch. This changes numerics relative to Newton's solver.
- Metal 4's main benefit here is lower CPU-side encode cost and GPU-address-based indirect execution. GPU-side per-dispatch cost and dependency stalls are not addressed by anything in the sources found.

### Gaps
- No Apple statement on forward progress was found. The MSL spec §6.15 text was not retrieved directly.
- No data was found on whether Metal 4 changes GPU-side dispatch latency on M4.

## Q4. Measured dispatch overhead on M-series, and how other projects minimise it

### Takeaway
Public microsecond-level numbers for raw back-to-back dependent dispatches inside one Metal compute encoder are scarce. Most published figures measure command-buffer round-trips (about 120 µs on M2 Max) or WebGPU-over-Metal (about 71 µs/dispatch), not in-encoder dispatch cost. The user's 3–9 µs in-encoder figure is consistent with these orders of magnitude, but I found no independent confirmation. The documented levers are: keep everything in one compute encoder (sub-core concurrency happens only within an encoder), use concurrent dispatch with explicit barriers only where dependencies exist, fuse kernels, and avoid command-buffer round-trips.

### Cited Findings
- An empty kernel with `waitUntilCompleted` took 0.12 ms on an M2 Max, against 0.001 ms with CUDA on an RTX 3080. This is a submit-plus-wait round trip, not in-encoder cost. No replies. — [Apple Developer Forums thread 767853](https://developer.apple.com/forums/thread/767853)
- WebGPU (wgpu) over Metal on M2: about 71 µs per-dispatch overhead against about 25–36 µs on Vulkan. An unfused 7-dispatch MLP took 5.74 ms against 2.85 ms for a 3-dispatch version. This includes the WebGPU layer. — [arXiv 2604.02344: Characterizing WebGPU dispatch overhead](https://arxiv.org/pdf/2604.02344)
- A bitonic sort that created a compute encoder per pass in Swift was dominated by per-dispatch overhead. An Objective-C rewrite was about 25x faster. — [Apple Developer Forums thread 750363](https://developer.apple.com/forums/thread/750363)
- Philip Turner's benchmarks: "The M1 Max has 32 GPU cores, but can perform up to 96 compute commands simultaneously. This sub-core concurrency only happens among commands within the same MTLComputeCommandEncoder. For commands on different Metal command queues, there's only 2x concurrency across the entire GPU." — [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
- `MTLDispatchType.concurrent` lets an encoder run dispatches concurrently, and the app must synchronize shared resources itself. — [Apple docs: MTLDispatchType.concurrent](https://developer.apple.com/documentation/metal/mtldispatchtype/concurrent.md)
- llama.cpp's Metal backend (PR #2358, 2023) analyses the ggml graph once to find which ops can be issued concurrently, then dispatches them with the concurrent dispatch type. The analysis must be rerun when the graph topology changes. — [llama.cpp commit 1aa18ef9 (mirror)](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/commit/1aa18ef994a6a2b531434eb13251ef48e56d345b)
- Google's Dawn (WebGPU) does not use `MTLDispatchTypeConcurrent` because it would require inserting memory barriers between conflicting commands. — [Dawn change 385a40b0](https://dawn.googlesource.com/dawn/+/385a40b0592318cbdc2b7313f10e08a4e51214eb%5E%21)
- The int4 KV-cache paper on Apple Silicon attributes an eager-MPS slowdown to per-call dispatch overhead (single-digit µs per rfft/irfft call), which a fused Metal kernel eliminates. — [arXiv 2605.05699](https://arxiv.org/pdf/2605.05699)

### Inferences
- For the roughly 1,500-dispatch step: using a concurrent encoder with barriers only at true dependencies (as llama.cpp does) can overlap independent kernels, such as per-body or per-field updates, and hide some of the fixed cost. Serial-encoder dependent dispatches pay the full drain-and-fill cost each time.
- Fusing small kernels (as the MPS/WebGPU results show) is probably the largest lever after removing host round-trips. Each dispatch avoided saves several µs.
- The expected win from the ICB-based loop: removing a 0.23 ms readback every 5 iterations, plus whatever CPU idle gaps the readback causes. This should be measured with `MTLCommandBuffer.gpuStartTime/gpuEndTime` and Metal System Trace.

### Gaps
- I found no published in-encoder dependent-dispatch latency for M4, no published cost of ICB `executeCommands` against direct encoding on Apple Silicon, and no MLX design write-up on dispatch minimisation. MLX's own approach (one command buffer per N ops, fused/compiled graphs via `mx.compile`) is not verified from a primary source in these notes.
- Unity/Unreal GPU-driven compute on Metal was not researched (time budget).
