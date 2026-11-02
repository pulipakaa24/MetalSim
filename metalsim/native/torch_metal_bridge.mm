// PyTorch MPS <-> Metal bridge: zero-copy tensors from foreign MTLBuffers and event ordering on
// PyTorch's command buffer. Built with torch.utils.cpp_extension (see metalsim/interop/torch_bridge.py).
//
// Facts this relies on (PyTorch 2.14, aten/src/ATen/mps and torch/csrc/api/include/torch/mps.h):
//  * an MPS tensor's storage data pointer IS the id<MTLBuffer> (OperationUtils.h getMTLBufferStorage
//    bit-casts it back), and the byte offset lives in storage_offset; at::from_blob with device kMPS
//    therefore yields a tensor that every MPS kernel reads from the foreign buffer.
//  * torch::mps::get_command_buffer() ends the open kernel-coalescing encoder and returns the
//    stream's current command buffer; torch::mps::commit() submits it. Both must run on the
//    stream's serial dispatch queue (torch::mps::get_dispatch_queue()).
//  * PyTorch 2.14's MPSStream never reads a command buffer's status or error: a buffer the driver aborts
//    (GPU watchdog timeout, out of memory, "ignored for prior errors") is skipped and its outputs are never
//    written, silently (https://github.com/pytorch/pytorch/pull/198883, in PyTorch 2.15.0, raises for the buffers
//    MPSStream commits; https://github.com/pytorch/pytorch/pull/200398, after 2.15.0, for those MPSGraph commits
//    itself). The command buffers this bridge encodes a cross-queue wait
//    or signal into are the ones a slow producer can get killed (a wait counts against the watchdog, ~4-5 s on
//    an M4 Max), so they are watched here and poll_failures() reports the first one that failed.
#include <torch/extension.h>
#include <torch/mps.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <mutex>
#include <string>
#include <vector>

namespace {

// Command buffers holding a cross-queue wait or signal, until they complete (see poll_failures).
std::mutex g_watch_mutex;
std::vector<id<MTLCommandBuffer>> g_watched;
std::string g_failure;

// The MTLCommandBuffer the next encode lands in: PyTorch's MPSCommandBuffer wraps one and replaces it on every
// commitAndContinue, so the wrapper itself has no single status to read.
id<MTLCommandBuffer> underlying(id<MTLCommandBuffer> cb)
{
    if ([(id)cb respondsToSelector:@selector(commandBuffer)]) {
        id<MTLCommandBuffer> inner = [(MPSCommandBuffer*)cb commandBuffer];
        if (inner)
            return inner;
    }
    return cb;
}

void watch(id<MTLCommandBuffer> cb)
{
    std::lock_guard<std::mutex> lock(g_watch_mutex);
    g_watched.push_back(underlying(cb));
}

// Drops the watched command buffers that finished; returns the first failure since the last call ("" if none).
std::string poll_failures()
{
    std::lock_guard<std::mutex> lock(g_watch_mutex);
    for (auto it = g_watched.begin(); it != g_watched.end();) {
        const MTLCommandBufferStatus status = (*it).status;
        if (status == MTLCommandBufferStatusError) {
            if (g_failure.empty()) {
                NSError* error = (*it).error;
                g_failure = error ? error.localizedDescription.UTF8String : "unknown error";
            }
            it = g_watched.erase(it);
        } else if (status == MTLCommandBufferStatusCompleted) {
            it = g_watched.erase(it);
        } else {
            ++it;
        }
    }
    std::string failure;
    failure.swap(g_failure);
    return failure;
}

id<MTLEvent> as_event(int64_t ptr)
{
    TORCH_CHECK(ptr != 0, "null Metal event handle");
    return (__bridge id<MTLEvent>)(void*)ptr;
}

int64_t device_ptr()
{
    return (int64_t)(__bridge void*)at::mps::MPSDevice::getInstance()->device();
}

int64_t queue_ptr()
{
    return (int64_t)(__bridge void*)at::mps::getCurrentMPSStream()->commandQueue();
}

// Encodes "wait until event >= value" at the current point of PyTorch's command buffer: kernels
// PyTorch encodes afterwards run after the producer's work; kernels already encoded are not held.
void wait_event(int64_t event, int64_t value)
{
    id<MTLEvent> ev = as_event(event);
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        [cb encodeWaitForEvent:ev value:(uint64_t)value];
        watch(cb);
    });
}

// Commits PyTorch's pending work; when it completes the event reaches value.
void signal_event(int64_t event, int64_t value)
{
    id<MTLEvent> ev = as_event(event);
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        [cb encodeSignalEvent:ev value:(uint64_t)value];
        watch(cb);
        torch::mps::commit();
    });
}

void commit()
{
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{ torch::mps::commit(); });
}

// Zero-copy MPS tensor over a foreign MTLBuffer. `keepalive` is any Python object that owns the
// buffer's memory (e.g. the Warp array); it is released when the tensor's storage is freed.
torch::Tensor tensor_from_mtlbuffer(
    int64_t buffer, int64_t byte_offset, std::vector<int64_t> shape, std::vector<int64_t> strides,
    torch::ScalarType dtype, py::object keepalive)
{
    TORCH_CHECK(buffer != 0, "null MTLBuffer handle");
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void*)buffer;
    TORCH_CHECK(buf.storageMode != MTLStorageModePrivate, "private-storage buffers cannot be shared");
    const int64_t elem = (int64_t)c10::elementSize(dtype);
    TORCH_CHECK(byte_offset % elem == 0, "byte offset must be a multiple of the element size");
    int64_t needed = 0;
    for (size_t i = 0; i < shape.size(); ++i)
        if (shape[i] > 0) needed = std::max(needed, (shape[i] - 1) * strides[i] * elem + elem);
    TORCH_CHECK((uint64_t)(byte_offset + needed) <= buf.length, "tensor extends beyond the Metal buffer");
    // retain the buffer for as long as the tensor lives; keepalive holds the owner on the Python side
    void* retained = (void*)CFBridgingRetain(buf);
    auto* owner = new py::object(std::move(keepalive));
    auto deleter = [retained, owner](void*) {
        CFBridgingRelease(retained);
        py::gil_scoped_acquire gil;
        delete owner;
    };
    auto options = torch::TensorOptions().device(at::Device(torch::kMPS, 0)).dtype(dtype);
    return at::from_blob((void*)buffer, shape, strides, byte_offset / elem, deleter, options, at::Device(torch::kMPS, 0));
}

// The id<MTLBuffer> behind an MPS tensor and its byte offset (for renderers / Warp reading torch memory).
std::pair<int64_t, int64_t> mtlbuffer_of(const torch::Tensor& t)
{
    TORCH_CHECK(t.device().is_mps(), "expected an MPS tensor");
    return { (int64_t)t.storage().data(), (int64_t)(t.storage_offset() * t.element_size()) };
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("device_ptr", &device_ptr, "id<MTLDevice> PyTorch's MPS backend uses");
    m.def("queue_ptr", &queue_ptr, "id<MTLCommandQueue> of the current MPS stream");
    m.def("wait_event", &wait_event, "encode a wait for (event, value) on PyTorch's command buffer");
    m.def("signal_event", &signal_event, "commit PyTorch's command buffer, signaling (event, value) on completion");
    m.def("commit", &commit, "commit PyTorch's command buffer without waiting");
    m.def("poll_failures", &poll_failures,
          "first failure of a watched PyTorch command buffer (one holding a bridge wait/signal) since the last call, or ''");
    m.def("tensor_from_mtlbuffer", &tensor_from_mtlbuffer, "zero-copy MPS tensor over an id<MTLBuffer>");
    m.def("mtlbuffer_of", &mtlbuffer_of, "(id<MTLBuffer>, byte offset) of an MPS tensor");
}
