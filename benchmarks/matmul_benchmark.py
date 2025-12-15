import timeit

import torch
import tensorax as ts

times = 100

a_torch = torch.randn((3, 1024, 1024), device='cuda', dtype=torch.float32)
b_torch = torch.randn((3, 1024, 1024), device='cuda', dtype=torch.float32)

a_t = ts.Tensor(a_torch.cpu().numpy(), dtype='float32', device='cuda')
b_t = ts.Tensor(b_torch.cpu().numpy(), dtype='float32', device='cuda')


# Benchmarking matmul with shared memory coalescing
def matmul_shared_memory_coalesced():
    c = a_t.matmul(b_t, method="shared_memory_coalesced")
    return c

# Benchmarking default matmul
def matmul_default():
    c = a_t.matmul(b_t, method="default")
    return c

# Benchmarking tiled matmul
def matmul_tiled():
    c = a_t.matmul(b_t, method="tiled")
    return c

# Benchmarking matmul with shared memory cache blocking
def matmul_cache_blocking():
    c = a_t.matmul(b_t, method="shared_memory_cache_blocking")
    return c

# Benchmarking matmul with 1D block tiling
def matmul_1d_block_tiling():
    c = a_t.matmul(b_t, method="block_tiling_1d")
    return c

# Benchmarking PyTorch matmul
def matmul_pytorch():
    c = torch.matmul(a_torch, b_torch)
    return c

# Warm-up run
print("Warming up...")
matmul_default()
matmul_shared_memory_coalesced()
matmul_tiled()
matmul_cache_blocking()
matmul_1d_block_tiling()
matmul_pytorch()
print("Warm-up done.")

print("Starting benchmarks...")

time_default = timeit.timeit(matmul_default, number=times)
print(f"Default matmul time over {times} runs: {time_default} seconds")

time_shared_memory_coalesced = timeit.timeit(matmul_shared_memory_coalesced, number=times)
print(f"Matmul with shared memory coalescing time over {times} runs: {time_shared_memory_coalesced} seconds")

time_cache_blocking = timeit.timeit(matmul_cache_blocking, number=times)
print(f"Matmul with shared memory cache blocking time over {times} runs: {time_cache_blocking} seconds")

time_tiled = timeit.timeit(matmul_tiled, number=times)
print(f"Tiled matmul time over {times} runs: {time_tiled} seconds")

time_1d_block_tiling = timeit.timeit(matmul_1d_block_tiling, number=times)
print(f"Matmul with 1D block tiling time over {times} runs: {time_1d_block_tiling} seconds")

time_pytorch = timeit.timeit(matmul_pytorch, number=times)
print(f"PyTorch matmul time over {times} runs: {time_pytorch} seconds")