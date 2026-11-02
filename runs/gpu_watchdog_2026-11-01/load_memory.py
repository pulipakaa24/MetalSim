"""Memory-heavy, compute-idle GPU load: allocate and touch GB gigabytes of Metal memory (2 GB MPS tensors), then sleep."""
import sys, time, torch
GB = float(sys.argv[1]); S = float(sys.argv[2])
chunks = []
for _ in range(int(GB // 2)):
    t = torch.empty(int(2e9 // 4), device="mps"); t.fill_(1.0); chunks.append(t)
torch.mps.synchronize()
print(f"[memload] {torch.mps.current_allocated_memory() / 1e9:.1f} GB allocated and touched; driver {torch.mps.driver_allocated_memory() / 1e9:.1f} GB", flush=True)
time.sleep(S)
s = sum(float(t[::1 << 20].sum()) for t in chunks)   # touch again at the end
print(f"[memload] done (checksum {s:.0f})", flush=True)
