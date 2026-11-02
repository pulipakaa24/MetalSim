"""Compute-heavy, memory-light GPU load: back-to-back 8192^2 fp32 matmuls (~0.8 GB) for S seconds.
Synchronizes every 4 matmuls so no command buffer of its own comes near the GPU watchdog."""
import sys, time, torch
S = float(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else "load"
a = torch.randn(8192, 8192, device="mps"); b = torch.randn(8192, 8192, device="mps"); c = torch.empty_like(a)
torch.mps.synchronize(); t0 = time.perf_counter(); k = 0
while time.perf_counter() - t0 < S:
    for _ in range(4):
        torch.mm(a, b, out=c); k += 1
    torch.mps.synchronize()
el = time.perf_counter() - t0
print(f"[{tag}] {k} matmuls in {el:.1f} s ({k * 2 * 8192**3 / el / 1e12:.1f} TFLOP/s)", flush=True)
