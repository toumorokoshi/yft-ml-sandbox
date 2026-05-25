import sys
import time
import torch
import triton
import triton.language as tl
import triton.runtime.driver as td
from triton.backends.compiler import GPUTarget

# Define GEMM kernel directly for the experiment
@triton.jit
def exp_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def run_experiment():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    if device == "cpu":
        print("Triton requires a GPU to run.")
        sys.exit(1)
        
    M, N, K = 1024, 1024, 1024
    print(f"Benchmarking GEMM M={M}, N={N}, K={K}")
    a = torch.randn((M, K), dtype=torch.float32, device=device)
    b = torch.randn((K, N), dtype=torch.float32, device=device)
    c = torch.empty((M, N), dtype=torch.float32, device=device)
    
    # Warmup and reference
    ref_c = torch.matmul(a, b)
    
    # Use smaller block size to avoid triggering backend workgroup size rounding limits
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    grid = lambda META: (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # Save original target query function
    orig_get_current_target = td.active.get_current_target
    original_target = orig_get_current_target()
    
    # Configurations to test
    configs = [
        # Threads per block = 128
        {"warp_size": 32, "num_warps": 4, "threads": 128},
        {"warp_size": 64, "num_warps": 2, "threads": 128},
        # Threads per block = 256
        {"warp_size": 32, "num_warps": 8, "threads": 256},
        {"warp_size": 64, "num_warps": 4, "threads": 256},
        # Threads per block = 512
        {"warp_size": 32, "num_warps": 16, "threads": 512},
        {"warp_size": 64, "num_warps": 8, "threads": 512},
    ]
    
    results = []
    
    for cfg in configs:
        ws = cfg["warp_size"]
        nw = cfg["num_warps"]
        threads_per_block = cfg["threads"]
        print(f"\n--- Testing warp_size={ws}, num_warps={nw} (threads/block={threads_per_block}) ---")
        
        # Monkeypatch current target to override warp_size
        def patched_get_current_target():
            return GPUTarget(backend=original_target.backend, arch=original_target.arch, warp_size=ws)
            
        td.active.get_current_target = patched_get_current_target
        
        success = False
        error_msg = ""
        latency_ms = 0.0
        
        try:
            # Compiles and runs
            exp_gemm_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE_M=GROUP_SIZE_M,
                num_warps=nw,
            )
            torch.cuda.synchronize()
            
            # Verify correctness
            correct = torch.allclose(c, ref_c, atol=1e-3, rtol=1e-3)
            if not correct:
                raise ValueError("Verification failed: output mismatch")
                
            # Benchmark latency
            trials = 50
            start_time = time.perf_counter()
            for _ in range(trials):
                exp_gemm_kernel[grid](
                    a, b, c,
                    M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                    GROUP_SIZE_M=GROUP_SIZE_M,
                    num_warps=nw,
                )
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latency_ms = ((end_time - start_time) / trials) * 1000.0
            success = True
            
        except Exception as e:
            error_msg = str(e).split('\n')[0]
            
        if success:
            gflops = (2.0 * M * N * K) / (latency_ms / 1000.0) / 1e9
            print(f"  Result: Success! Latency: {latency_ms:.3f} ms, Performance: {gflops:.2f} GFLOPs")
            results.append({
                "warp_size": ws,
                "num_warps": nw,
                "threads": threads_per_block,
                "status": "SUCCESS",
                "latency_ms": latency_ms,
                "gflops": gflops
            })
        else:
            print(f"  Result: FAILED! Error: {error_msg}")
            results.append({
                "warp_size": ws,
                "num_warps": nw,
                "threads": threads_per_block,
                "status": f"FAILED ({error_msg})",
                "latency_ms": float('inf'),
                "gflops": 0.0
            })
            
    # Restore original driver target query function
    td.active.get_current_target = orig_get_current_target
    
    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"{'warp_size':<10} | {'num_warps':<10} | {'threads/block':<15} | {'status':<10} | {'latency (ms)':<15} | {'GFLOPs':<10}")
    print("-" * 85)
    for r in results:
        latency_str = f"{r['latency_ms']:.3f}" if r['latency_ms'] != float('inf') else "N/A"
        gflops_str = f"{r['gflops']:.2f}" if r['gflops'] > 0 else "N/A"
        print(f"{r['warp_size']:<10} | {r['num_warps']:<10} | {r['threads']:<15} | {r['status']:<10} | {latency_str:<15} | {gflops_str:<10}")

if __name__ == "__main__":
    run_experiment()
