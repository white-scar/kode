import torch, time, os
print("  torch", torch.__version__, "| cuda avail:", torch.cuda.is_available(), "| dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
if torch.cuda.is_available():
    a = torch.randn(3000, 3000, device="cuda"); b = torch.randn(3000, 3000, device="cuda")
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20): c = a @ b
    torch.cuda.synchronize()
    print(f"  20x 3000^3 matmul = {time.time()-t0:.2f}s | sum={float(c.sum()):.1f} | mem={torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
