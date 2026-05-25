import onnx
import torch
import triton
import triton.language as tl
from onnx import numpy_helper

# Constants for default block sizes and layouts
BLOCK_SIZE_M_GEMM = 64
BLOCK_SIZE_N_GEMM = 64
BLOCK_SIZE_K_GEMM = 32
GROUP_SIZE_M_GEMM = 8

BLOCK_SIZE_M_ELEM = 16
BLOCK_SIZE_N_ELEM = 16


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def elementwise_add_2d_kernel(
    x_ptr, y_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_outm, stride_outn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = tl.load(y_ptrs, mask=mask, other=0.0)

    out = x + y
    tl.store(out_ptrs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def elementwise_mul_2d_kernel(
    x_ptr, y_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_outm, stride_outn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = tl.load(y_ptrs, mask=mask, other=0.0)

    out = x * y
    tl.store(out_ptrs, out, mask=mask)


@triton.autotune(
    configs=[
        # Optimized configurations for RDNA 3/3.5 (warp size 32)
        # Tuning-optimized configurations for small-M (GEMV) and high occupancy (GEMM)
        # the block size being large resulted in needing memory beyond what can be stored in L1 cache, in turn making memory bandwidth the bottleneck.
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),

    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
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


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes elementwise addition of two tensors using Triton."""
    # Ensure inputs are at least 2D for our 2D elementwise kernel
    x_2d = x if x.dim() >= 2 else x.unsqueeze(0)
    y_2d = y if y.dim() >= 2 else y.unsqueeze(0)

    x_b, y_b = torch.broadcast_tensors(x_2d, y_2d)
    M, N = x_b.shape

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    elementwise_add_2d_kernel[grid](
        x_b, y_b, out,
        M, N,
        x_b.stride(0), x_b.stride(1),
        y_b.stride(0), y_b.stride(1),
        out.stride(0), out.stride(1),
    )

    if x.dim() < 2 and y.dim() < 2:
        return out.squeeze(0)
    return out


def triton_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes elementwise multiplication of two tensors using Triton."""
    x_2d = x if x.dim() >= 2 else x.unsqueeze(0)
    y_2d = y if y.dim() >= 2 else y.unsqueeze(0)

    x_b, y_b = torch.broadcast_tensors(x_2d, y_2d)
    M, N = x_b.shape

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    elementwise_mul_2d_kernel[grid](
        x_b, y_b, out,
        M, N,
        x_b.stride(0), x_b.stride(1),
        y_b.stride(0), y_b.stride(1),
        out.stride(0), out.stride(1),
    )

    if x.dim() < 2 and y.dim() < 2:
        return out.squeeze(0)
    return out


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes matrix multiplication of two matrices using Triton."""
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D matrices"
    M, K1 = a.shape
    K2, N = b.shape
    assert K1 == K2, f"Dimension mismatch: {K1} != {K2}"
    K = K1

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c



def load_onnx_model_from_file(file_path: str) -> onnx.ModelProto:
    """IO Wrapper: Loads an ONNX model from a file path."""
    return onnx.load(file_path)


def run_onnx_with_triton(
    model: onnx.ModelProto,
    inputs: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    """Inner function: Executes an ONNX model on PyTorch data structures mapping to Triton ops."""
    env = {}
    for name, tensor in inputs.items():
        env[name] = tensor.to(device)

    for init in model.graph.initializer:
        np_arr = numpy_helper.to_array(init)
        env[init.name] = torch.from_numpy(np_arr).to(device)

    # Process nodes in topological order
    for node in model.graph.node:
        node_inputs = [env[name] for name in node.input]

        if node.op_type == "Add":
            env[node.output[0]] = triton_add(node_inputs[0], node_inputs[1])

        elif node.op_type == "Mul":
            env[node.output[0]] = triton_mul(node_inputs[0], node_inputs[1])

        elif node.op_type == "Gemm":
            # Extract GEMM attributes
            attrs = {attr.name: attr for attr in node.attribute}
            alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
            beta = attrs["beta"].f if "beta" in attrs else 1.0
            transA = attrs["transA"].i if "transA" in attrs else 0
            transB = attrs["transB"].i if "transB" in attrs else 0

            A, B = node_inputs[0], node_inputs[1]
            if transA == 1:
                A = A.t()
            if transB == 1:
                B = B.t()

            out = triton_matmul(A, B)

            if alpha != 1.0:
                alpha_tensor = torch.tensor(alpha, dtype=out.dtype, device=device)
                out = triton_mul(out, alpha_tensor)

            if len(node_inputs) > 2:
                C = node_inputs[2]
                if beta != 1.0:
                    beta_tensor = torch.tensor(beta, dtype=C.dtype, device=device)
                    C = triton_mul(C, beta_tensor)
                out = triton_add(out, C)

            env[node.output[0]] = out
        else:
            raise NotImplementedError(f"ONNX op {node.op_type} is not supported by this Triton interpreter.")

    # Return only the outputs requested by the graph
    outputs = {}
    for out_val in model.graph.output:
        outputs[out_val.name] = env[out_val.name]
    return outputs
