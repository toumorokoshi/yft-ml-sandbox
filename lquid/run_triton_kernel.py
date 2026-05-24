import os
import sys
import ctypes
import tempfile
import numpy as np
import torch
import onnx
from onnx import numpy_helper
import triton
import triton.compiler as tc
import triton.backends.compiler as tbc
import triton.runtime.driver as driver

# Determine HIP library path (prefer PyTorch's bundled one to avoid version conflict)
try:
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    torch_hip_path = os.path.join(torch_lib_dir, "libamdhip64.so")
    if os.path.exists(torch_hip_path):
        HIP_LIB_PATH = torch_hip_path
    else:
        HIP_LIB_PATH = "/opt/rocm/lib/libamdhip64.so"
except Exception:
    HIP_LIB_PATH = "/opt/rocm/lib/libamdhip64.so"

ONNX_PATH = "model.onnx"
TTIR_PATH = "kernel.ttir"

# HIP error checking
def check_hip_err(err_code: int, msg: str) -> None:
    if err_code != 0:
        raise RuntimeError(f"HIP Error {err_code} during {msg}")

# ctypes wrappers
class HipDriver:
    def __init__(self, lib_path: str = HIP_LIB_PATH):
        # Preload HSA runtime globally to resolve symbol conflicts with bundled libraries
        try:
            ctypes.CDLL("/opt/rocm/lib/libhsa-runtime64.so", mode=ctypes.RTLD_GLOBAL)
        except Exception as e:
            print(f"Warning: failed to preload libhsa-runtime64.so: {e}")
            
        self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        
        # Function prototypes
        self.lib.hipInit.argtypes = [ctypes.c_uint]
        self.lib.hipInit.restype = int
        
        self.lib.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        self.lib.hipModuleLoad.restype = int
        
        self.lib.hipModuleUnload.argtypes = [ctypes.c_void_p]
        self.lib.hipModuleUnload.restype = int
        
        self.lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self.lib.hipModuleGetFunction.restype = int
        
        self.lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.hipMalloc.restype = int
        
        self.lib.hipFree.argtypes = [ctypes.c_void_p]
        self.lib.hipFree.restype = int
        
        self.lib.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.hipMemcpy.restype = int
        
        self.lib.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.hipModuleLaunchKernel.restype = int
        
        self.lib.hipDeviceSynchronize.argtypes = []
        self.lib.hipDeviceSynchronize.restype = int

        # Initialize HIP
        check_hip_err(self.lib.hipInit(0), "hipInit")

# Inner pure data validation function
def compare_tensors(torch_out: np.ndarray, triton_out: np.ndarray) -> bool:
    """Pure comparison function for validation."""
    print("PyTorch output:\n", torch_out)
    print("Triton output:\n", triton_out)
    is_close = np.allclose(torch_out, triton_out, atol=1e-4, rtol=1e-4)
    return bool(is_close)

# Load weights from ONNX
def load_onnx_weights(onnx_path: str):
    """IO function to extract parameter values from ONNX file."""
    model = onnx.load(onnx_path)
    weights = {}
    for initializer in model.graph.initializer:
        weights[initializer.name] = numpy_helper.to_array(initializer)
    return weights

# Compilation wrapper
def compile_ttir_to_hsaco(ttir_path: str) -> tuple[bytes, int, int]:
    """IO compilation wrapper calling Triton compiler."""
    try:
        target = driver.active.get_current_target()
    except Exception:
        target = tbc.GPUTarget(backend="hip", arch="gfx1150", warp_size=64)
    print(f"Target: {target}")

    res = triton.compile(ttir_path, target=target)
    if "llir" in res.asm:
        with open("kernel.ll", "w") as f:
            f.write(res.asm["llir"])
    
    # Extract metadata
    print("Metadata attributes:")
    for attr in dir(res.metadata):
        if not attr.startswith("_"):
            try:
                print(f"  {attr}: {getattr(res.metadata, attr)}")
            except Exception as e:
                print(f"  {attr}: Error: {e}")
                
    num_warps = getattr(res.metadata, "num_warps", 4)
    warp_size = getattr(res.metadata, "warp_size", 64)
    shared_mem = getattr(res.metadata, "shared", 0)
    
    return res.asm["hsaco"], num_warps, warp_size, shared_mem

# Main runner logic
def execute_test(onnx_file_path: str, ttir_file_path: str) -> None:
    # 1. Compile
    hsaco_bytes, num_warps, warp_size, shared_mem = compile_ttir_to_hsaco(ttir_file_path)
    print(f"Compiled to HSACO. Size: {len(hsaco_bytes)} bytes. num_warps={num_warps}, warp_size={warp_size}, shared_mem={shared_mem}")

    # 2. Extract weights
    weights = load_onnx_weights(onnx_file_path)
    print("ONNX parameters loaded:", weights.keys())
    
    # Weight: [5, 10], Bias: [5]
    weight = weights["linear.weight"]
    bias = weights["linear.bias"]
    
    M, K = 1, 10
    N = 5
    
    # Generate dummy input A: [1, 10]
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = weight.astype(np.float32) # [5, 10]
    bias = bias.astype(np.float32) # [5]
    
    # 3. Perform PyTorch forward pass equivalent
    torch_A = torch.from_numpy(A)
    torch_W = torch.from_numpy(B)
    torch_b = torch.from_numpy(bias)
    torch_out = torch.nn.functional.linear(torch_A, torch_W, torch_b).numpy()

    # 4. Perform Triton kernel execution on GPU
    hip = HipDriver()
    
    # Load module
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as tmp:
        tmp.write(hsaco_bytes)
        tmp_name = tmp.name
    
    module = ctypes.c_void_p()
    try:
        check_hip_err(hip.lib.hipModuleLoad(ctypes.byref(module), tmp_name.encode('utf-8')), "hipModuleLoad")
    finally:
        os.unlink(tmp_name)
        
    kernel = ctypes.c_void_p()
    check_hip_err(hip.lib.hipModuleGetFunction(ctypes.byref(kernel), module, b"matmul_kernel"), "hipModuleGetFunction")

    # Allocate GPU memory
    d_A = ctypes.c_void_p()
    d_B = ctypes.c_void_p()
    d_C = ctypes.c_void_p()
    d_bias = ctypes.c_void_p()
    
    check_hip_err(hip.lib.hipMalloc(ctypes.byref(d_A), A.nbytes), "hipMalloc A")
    check_hip_err(hip.lib.hipMalloc(ctypes.byref(d_B), B.nbytes), "hipMalloc B")
    check_hip_err(hip.lib.hipMalloc(ctypes.byref(d_C), torch_out.nbytes), "hipMalloc C")
    check_hip_err(hip.lib.hipMalloc(ctypes.byref(d_bias), bias.nbytes), "hipMalloc bias")

    # Copy input memory
    # hipMemcpyHostToDevice = 1
    check_hip_err(hip.lib.hipMemcpy(d_A, A.ctypes.data, A.nbytes, 1), "hipMemcpy A")
    check_hip_err(hip.lib.hipMemcpy(d_B, B.ctypes.data, B.nbytes, 1), "hipMemcpy B")
    check_hip_err(hip.lib.hipMemcpy(d_bias, bias.ctypes.data, bias.nbytes, 1), "hipMemcpy bias")
    
    print(f"Allocated: d_A={d_A.value:#x}, d_B={d_B.value:#x}, d_C={d_C.value:#x}, d_bias={d_bias.value:#x}")
    print(f"Shapes: A={A.shape}, B={B.shape}, C={torch_out.shape}, bias={bias.shape}")
    print(f"Params: M={M}, N={N}, K={K}")
    
    # Zero C memory
    zero_C = np.zeros_like(torch_out)
    check_hip_err(hip.lib.hipMemcpy(d_C, zero_C.ctypes.data, zero_C.nbytes, 1), "hipMemcpy zero C")

    # Strides calculation
    stride_am = K
    stride_ak = 1
    stride_bk = 1 # transposed row dimension
    stride_bn = K # input dimension
    stride_cm = N
    stride_cn = 1
    
    # Kernel args mapping
    a_arg = ctypes.c_void_p(d_A.value)
    b_arg = ctypes.c_void_p(d_B.value)
    c_arg = ctypes.c_void_p(d_C.value)
    bias_arg = ctypes.c_void_p(d_bias.value)
    
    m_arg = ctypes.c_int(M)
    n_arg = ctypes.c_int(N)
    k_arg = ctypes.c_int(K)
    
    sam_arg = ctypes.c_int(stride_am)
    sak_arg = ctypes.c_int(stride_ak)
    sbk_arg = ctypes.c_int(stride_bk)
    sbn_arg = ctypes.c_int(stride_bn)
    scm_arg = ctypes.c_int(stride_cm)
    scn_arg = ctypes.c_int(stride_cn)

    # Extra readnone pointers expected by Triton ROCm backend
    extra_1 = ctypes.c_void_p(0)
    extra_2 = ctypes.c_void_p(0)

    args = [
        a_arg, b_arg, c_arg, bias_arg,
        m_arg, n_arg, k_arg,
        sam_arg, sak_arg,
        sbk_arg, sbn_arg,
        scm_arg, scn_arg,
        extra_1, extra_2
    ]
    
    args_ptrs = (ctypes.c_void_p * len(args))()
    for i, arg in enumerate(args):
        args_ptrs[i] = ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p)

    # Launch kernel
    gridDimX = 1
    gridDimY = 1
    gridDimZ = 1
    
    blockDimX = num_warps * warp_size
    blockDimY = 1
    blockDimZ = 1
    
    print(f"Launching matmul_kernel with grid=({gridDimX},{gridDimY},{gridDimZ}), block=({blockDimX},{blockDimY},{blockDimZ}), shared_mem={shared_mem}")
    
    check_hip_err(hip.lib.hipModuleLaunchKernel(
        kernel,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        shared_mem, None, args_ptrs, None
    ), "hipModuleLaunchKernel")
    
    # Synchronize
    check_hip_err(hip.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")
    
    # Copy output memory back
    # hipMemcpyDeviceToHost = 2
    triton_out = np.zeros_like(torch_out)
    check_hip_err(hip.lib.hipMemcpy(triton_out.ctypes.data, d_C, triton_out.nbytes, 2), "hipMemcpy C back")

    # Free memory
    hip.lib.hipFree(d_A)
    hip.lib.hipFree(d_B)
    hip.lib.hipFree(d_C)
    hip.lib.hipFree(d_bias)
    hip.lib.hipModuleUnload(module)

    # 5. Assert match
    is_close = compare_tensors(torch_out, triton_out)
    if is_close:
        print("SUCCESS: Triton output matches PyTorch output!")
    else:
        print("ERROR: Triton output does not match PyTorch output!")
        sys.exit(1)

def main():
    print("PyTorch CUDA available:", torch.cuda.is_available())
    print("PyTorch device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("PyTorch current device:", torch.cuda.current_device())
        print("PyTorch device name:", torch.cuda.get_device_name(0))
        
        
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")
    onnx_file = os.path.join(workspace_dir, ONNX_PATH)
    ttir_file = os.path.join(workspace_dir, TTIR_PATH)
    
    if not os.path.exists(onnx_file):
        print(f"Error: ONNX file not found at {onnx_file}")
        sys.exit(1)
    if not os.path.exists(ttir_file):
        print(f"Error: TTIR file not found at {ttir_file}")
        sys.exit(1)
        
    execute_test(onnx_file, ttir_file)

if __name__ == "__main__":
    main()
