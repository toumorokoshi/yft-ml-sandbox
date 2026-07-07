#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <mma.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>

using namespace nvcuda;

// CUDA Error Checking Macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// cuBLAS Error Checking Macro
#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << "Status Code " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper Kernels for FP32 <-> FP16 Conversion
__global__ void floatToHalfKernel(const float* src, __half* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void halfToFloatKernel(const __half* src, float* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __half2float(src[idx]);
    }
}

// Helper Kernels for FP32 <-> FP8 Conversion
__global__ void floatToFp8Kernel(const float* src, __nv_fp8_storage_t* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __nv_cvt_float_to_fp8(src[idx], __NV_NOSAT, __NV_E4M3);
    }
}

__global__ void floatToFp8TransposedKernel(const float* src, __nv_fp8_storage_t* dst, int K_dim, int N_dim) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < K_dim && c < N_dim) {
        float val = src[r * N_dim + c];
        dst[c * K_dim + r] = __nv_cvt_float_to_fp8(val, __NV_NOSAT, __NV_E4M3);
    }
}

// Helper Kernels for FP32 <-> FP4 Conversion
__global__ void floatToFp4Kernel(const float* src, __nv_fp4_storage_t* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size / 2) {
        __nv_fp4_storage_t fp4_1 = __nv_cvt_float_to_fp4(src[2 * idx], __NV_E2M1, cudaRoundNearest);
        __nv_fp4_storage_t fp4_2 = __nv_cvt_float_to_fp4(src[2 * idx + 1], __NV_E2M1, cudaRoundNearest);
        dst[idx] = (fp4_2 << 4) | (fp4_1 & 0x0F);
    }
}

__global__ void floatToFp4TransposedKernel(const float* src, __nv_fp4_storage_t* dst, int K_dim, int N_dim) {
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 2; // processes 2 rows
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < K_dim && c < N_dim) {
        __nv_fp4_storage_t fp4_1 = __nv_cvt_float_to_fp4(src[r * N_dim + c], __NV_E2M1, cudaRoundNearest);
        __nv_fp4_storage_t fp4_2 = __nv_cvt_float_to_fp4(src[(r + 1) * N_dim + c], __NV_E2M1, cudaRoundNearest);
        dst[c * (K_dim / 2) + (r / 2)] = (fp4_2 << 4) | (fp4_1 & 0x0F);
    }
}

__global__ void fillScalesKernel(__nv_fp8_storage_t* scales, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scales[idx] = __nv_cvt_float_to_fp8(1.0f, __NV_NOSAT, __NV_E4M3);
    }
}

// Custom Warp-level C++ WMMA Matrix Multiplication Kernel (FP16 inputs, FP32 Accumulation)
// WMMA shape used is 16x16x16.
__global__ void gemm_wmma_fp16(
    const __half* A, const __half* B, float* C,
    int M, int N, int K) {
    
    // Each block contains blockDim.x = 32 threads (one warp in X) and blockDim.y warps.
    // threadIdx.y identifies the warp index within the block.
    int warpRow = blockIdx.y * blockDim.y + threadIdx.y;
    int warpCol = blockIdx.x;
    
    int row = warpRow * 16;
    int col = warpCol * 16;
    
    if (row >= M || col >= N) return;
    
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Initialize the accumulator fragment
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop over the inner K dimension
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store the output back to global memory
    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}

// Host-based sub-matrix verification
bool verify_results(const float* A, const float* B, const float* C, int M, int N, int K, int check_size, float rel_tolerance = 1e-4f) {
    int limit_M = std::min(M, check_size);
    int limit_N = std::min(N, check_size);

    for (int r = 0; r < limit_M; ++r) {
        for (int c = 0; c < limit_N; ++c) {
            double expected = 0.0;
            for (int k = 0; k < K; ++k) {
                expected += (double)A[r * K + k] * (double)B[k * N + c];
            }
            float actual = C[r * N + c];
            double diff = std::abs(expected - actual);
            double rel_error = diff / (std::abs(expected) + 1e-9);
            if (diff > 1e-1 && rel_error > rel_tolerance) {
                std::cerr << "Verification failed at (" << r << ", " << c
                          << "): expected " << expected << ", got " << actual
                          << ", diff = " << diff << ", rel_error = " << rel_error << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Main benchmark runner
void run_benchmark(cublasHandle_t handle, cublasLtHandle_t ltHandle, int N_size) {
    int M = N_size;
    int N = N_size;
    int K = N_size;

    std::cout << "\n============================================\n";
    std::cout << "Benchmarking Matrix Size: " << M << "x" << N << "x" << K << "\n";
    std::cout << "============================================\n";

    // Host matrices
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    // Initialize inputs
    for (size_t i = 0; i < (size_t)M * K; ++i) h_A[i] = static_cast<float>(rand() % 100) / 100.0f;
    for (size_t i = 0; i < (size_t)K * N; ++i) h_B[i] = static_cast<float>(rand() % 100) / 100.0f;

    // Device pointers
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    __half *d_A_half = nullptr, *d_B_half = nullptr;
    __nv_fp8_storage_t *d_A_fp8 = nullptr, *d_B_fp8 = nullptr;
    __nv_fp4_storage_t *d_A_fp4 = nullptr, *d_B_fp4 = nullptr;
    __nv_fp8_storage_t *d_scale_A = nullptr, *d_scale_B = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // Allocate half-precision memory for FP16 and WMMA benchmarks
    size_t size_A_half = (size_t)M * K * sizeof(__half);
    size_t size_B_half = (size_t)K * N * sizeof(__half);

    CHECK_CUDA(cudaMalloc(&d_A_half, size_A_half));
    CHECK_CUDA(cudaMalloc(&d_B_half, size_B_half));

    // Allocate FP8 precision memory
    size_t size_A_fp8 = (size_t)M * K * sizeof(__nv_fp8_storage_t);
    size_t size_B_fp8 = (size_t)K * N * sizeof(__nv_fp8_storage_t);
    CHECK_CUDA(cudaMalloc(&d_A_fp8, size_A_fp8));
    CHECK_CUDA(cudaMalloc(&d_B_fp8, size_B_fp8));

    // Allocate FP4 precision memory
    size_t size_A_fp4 = ((size_t)M * K) / 2 * sizeof(__nv_fp4_storage_t);
    size_t size_B_fp4 = ((size_t)K * N) / 2 * sizeof(__nv_fp4_storage_t);
    CHECK_CUDA(cudaMalloc(&d_A_fp4, size_A_fp4));
    CHECK_CUDA(cudaMalloc(&d_B_fp4, size_B_fp4));

    // Allocate scales for FP4 VEC16 block scaling
    int scale_A_size = M * (K / 16);
    int scale_B_size = N * (K / 16);
    CHECK_CUDA(cudaMalloc(&d_scale_A, scale_A_size));
    CHECK_CUDA(cudaMalloc(&d_scale_B, scale_B_size));

    // Convert inputs on device
    int threads_convert = 256;
    int blocks_convert_A = ((M * K) + threads_convert - 1) / threads_convert;
    int blocks_convert_B = ((K * N) + threads_convert - 1) / threads_convert;

    floatToHalfKernel<<<blocks_convert_A, threads_convert>>>(d_A, d_A_half, M * K);
    floatToHalfKernel<<<blocks_convert_B, threads_convert>>>(d_B, d_B_half, K * N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Performance measurements setup
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int warmup_runs = 10;
    const int benchmark_runs = 50;
    double gflops_base = 2.0 * M * N * K;

    // ----------------------------------------------------
    // 1. cuBLAS FP32 GEMM (SGEMM baseline)
    // ----------------------------------------------------
    float alpha = 1.0f;
    float beta = 0.0f;

    for (int i = 0; i < warmup_runs; ++i) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < benchmark_runs; ++i) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_cublas_fp32 = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_cublas_fp32, start, stop));
    float avg_sec_cublas_fp32 = (ms_cublas_fp32 / benchmark_runs) / 1000.0f;
    double tflops_cublas_fp32 = (gflops_base / avg_sec_cublas_fp32) / 1e12;

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    bool cublas_fp32_ok = verify_results(h_A.data(), h_B.data(), h_C.data(), M, N, K, 128);

    // ----------------------------------------------------
    // 2. Custom WMMA GEMM (FP16 Inputs, FP32 Accumulation)
    // ----------------------------------------------------
    // Grid maps: each block has 32 threads in X (1 warp) and 4 warps in Y.
    // warpRow computes 16 rows. block computes 4 * 16 = 64 rows.
    // warpCol computes 16 columns. block computes 1 * 16 = 16 columns.
    dim3 block_wmma(32, 4);
    dim3 grid_wmma((N + 15) / 16, (M + 63) / 64);

    for (int i = 0; i < warmup_runs; ++i) {
        gemm_wmma_fp16<<<grid_wmma, block_wmma>>>(d_A_half, d_B_half, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < benchmark_runs; ++i) {
        gemm_wmma_fp16<<<grid_wmma, block_wmma>>>(d_A_half, d_B_half, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_custom_wmma = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_custom_wmma, start, stop));
    float avg_sec_custom_wmma = (ms_custom_wmma / benchmark_runs) / 1000.0f;
    double tflops_custom_wmma = (gflops_base / avg_sec_custom_wmma) / 1e12;

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    bool custom_wmma_ok = verify_results(h_A.data(), h_B.data(), h_C.data(), M, N, K, 128, 1e-2f);

    // ----------------------------------------------------
    // 3. cuBLAS HGEMM (FP16 baseline)
    // ----------------------------------------------------
    bool fp16_tested = false;
    float ms_cublas_fp16 = 0.0f;
    double tflops_cublas_fp16 = 0.0f;
    bool cublas_fp16_ok = false;

    cublasStatus_t fp16_status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_half, CUDA_R_16F, N, d_A_half, CUDA_R_16F, K, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (fp16_status == CUBLAS_STATUS_SUCCESS) {
        fp16_tested = true;
        for (int i = 1; i < warmup_runs; ++i) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_half, CUDA_R_16F, N, d_A_half, CUDA_R_16F, K, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < benchmark_runs; ++i) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_half, CUDA_R_16F, N, d_A_half, CUDA_R_16F, K, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&ms_cublas_fp16, start, stop));
        float avg_sec_cublas_fp16 = (ms_cublas_fp16 / benchmark_runs) / 1000.0f;
        tflops_cublas_fp16 = (gflops_base / avg_sec_cublas_fp16) / 1e12;

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
        cublas_fp16_ok = verify_results(h_A.data(), h_B.data(), h_C.data(), M, N, K, 128, 1e-2f);
    }

    // ----------------------------------------------------
    // 4. cuBLASLt FP8 Tensor Core GEMM
    // ----------------------------------------------------
    floatToFp8Kernel<<<blocks_convert_A, threads_convert>>>(d_A, d_A_fp8, M * K);
    dim3 blockDimTrans(16, 16);
    dim3 gridDimTrans((N + 15) / 16, (K + 15) / 16);
    floatToFp8TransposedKernel<<<gridDimTrans, blockDimTrans>>>(d_B, d_B_fp8, K, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    bool fp8_tested = false;
    float ms_cublas_fp8 = 0.0f;
    double tflops_cublas_fp8 = 0.0f;
    bool cublas_fp8_ok = false;

    cublasLtMatrixLayout_t Adesc_fp8 = nullptr, Bdesc_fp8 = nullptr, Cdesc_fp8 = nullptr, Ddesc_fp8 = nullptr;
    cublasLtMatmulDesc_t opDesc_fp8 = nullptr;
    cublasLtMatmulPreference_t pref_fp8 = nullptr;
    void* workspace_fp8 = nullptr;
    uint64_t workspaceSize_fp8 = 4 * 1024 * 1024;

    do {
        cublasStatus_t status;
        status = cublasLtMatrixLayoutCreate(&Adesc_fp8, CUDA_R_8F_E4M3, K, N, K);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatrixLayoutCreate(&Bdesc_fp8, CUDA_R_8F_E4M3, K, M, K);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatrixLayoutCreate(&Cdesc_fp8, CUDA_R_32F, N, M, N);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatrixLayoutCreate(&Ddesc_fp8, CUDA_R_32F, N, M, N);
        if (status != CUBLAS_STATUS_SUCCESS) break;

        status = cublasLtMatmulDescCreate(&opDesc_fp8, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (status != CUBLAS_STATUS_SUCCESS) break;

        cublasOperation_t transA = CUBLAS_OP_T;
        cublasOperation_t transB = CUBLAS_OP_N;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp8, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp8, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
        if (status != CUBLAS_STATUS_SUCCESS) break;

        status = cublasLtMatmulPreferenceCreate(&pref_fp8);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulPreferenceSetAttribute(pref_fp8, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize_fp8, sizeof(workspaceSize_fp8));
        if (status != CUBLAS_STATUS_SUCCESS) break;

        CHECK_CUDA(cudaMalloc(&workspace_fp8, workspaceSize_fp8));

        std::vector<cublasLtMatmulHeuristicResult_t> heuristicResults(1);
        int returnedAlgoCount = 0;
        status = cublasLtMatmulAlgoGetHeuristic(
            ltHandle, opDesc_fp8, Adesc_fp8, Bdesc_fp8, Cdesc_fp8, Ddesc_fp8, pref_fp8, 1, heuristicResults.data(), &returnedAlgoCount
        );
        if (status != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) break;

        fp8_tested = true;
        for (int i = 0; i < warmup_runs; ++i) {
            cublasLtMatmul(ltHandle, opDesc_fp8, &alpha, d_B_fp8, Adesc_fp8, d_A_fp8, Bdesc_fp8, &beta, d_C, Cdesc_fp8, d_C, Ddesc_fp8, &heuristicResults[0].algo, workspace_fp8, workspaceSize_fp8, nullptr);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < benchmark_runs; ++i) {
            cublasLtMatmul(ltHandle, opDesc_fp8, &alpha, d_B_fp8, Adesc_fp8, d_A_fp8, Bdesc_fp8, &beta, d_C, Cdesc_fp8, d_C, Ddesc_fp8, &heuristicResults[0].algo, workspace_fp8, workspaceSize_fp8, nullptr);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&ms_cublas_fp8, start, stop));
        float avg_sec_cublas_fp8 = (ms_cublas_fp8 / benchmark_runs) / 1000.0f;
        tflops_cublas_fp8 = (gflops_base / avg_sec_cublas_fp8) / 1e12;

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
        cublas_fp8_ok = verify_results(h_A.data(), h_B.data(), h_C.data(), M, N, K, 128, 5e-2f);
    } while (0);

    if (workspace_fp8) cudaFree(workspace_fp8);
    if (pref_fp8) cublasLtMatmulPreferenceDestroy(pref_fp8);
    if (opDesc_fp8) cublasLtMatmulDescDestroy(opDesc_fp8);
    if (Adesc_fp8) cublasLtMatrixLayoutDestroy(Adesc_fp8);
    if (Bdesc_fp8) cublasLtMatrixLayoutDestroy(Bdesc_fp8);
    if (Cdesc_fp8) cublasLtMatrixLayoutDestroy(Cdesc_fp8);
    if (Ddesc_fp8) cublasLtMatrixLayoutDestroy(Ddesc_fp8);

    // ----------------------------------------------------
    // 5. cuBLASLt FP4 (NVFP4) Tensor Core GEMM
    // ----------------------------------------------------
    int blocks_convert_A_fp4 = ((M * K / 2) + threads_convert - 1) / threads_convert;
    floatToFp4Kernel<<<blocks_convert_A_fp4, threads_convert>>>(d_A, d_A_fp4, M * K);
    dim3 gridDimTransFp4((N + 15) / 16, (K / 2 + 15) / 16);
    floatToFp4TransposedKernel<<<gridDimTransFp4, blockDimTrans>>>(d_B, d_B_fp4, K, N);
    fillScalesKernel<<< (scale_A_size + threads_convert - 1) / threads_convert, threads_convert>>>(d_scale_A, scale_A_size);
    fillScalesKernel<<< (scale_B_size + threads_convert - 1) / threads_convert, threads_convert>>>(d_scale_B, scale_B_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    bool fp4_tested = false;
    float ms_cublas_fp4 = 0.0f;
    double tflops_cublas_fp4 = 0.0f;
    bool cublas_fp4_ok = false;

    cublasLtMatrixLayout_t Adesc_fp4 = nullptr, Bdesc_fp4 = nullptr, Cdesc_fp4 = nullptr, Ddesc_fp4 = nullptr;
    cublasLtMatmulDesc_t opDesc_fp4 = nullptr;
    cublasLtMatmulPreference_t pref_fp4 = nullptr;
    void* workspace_fp4 = nullptr;
    uint64_t workspaceSize_fp4 = 4 * 1024 * 1024;

    do {
        cublasStatus_t status;
        status = cublasLtMatrixLayoutCreate(&Adesc_fp4, CUDA_R_4F_E2M1, K, N, K);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatrixLayoutCreate(&Bdesc_fp4, CUDA_R_4F_E2M1, K, M, K);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatrixLayoutCreate(&Cdesc_fp4, CUDA_R_32F, N, M, N);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatrixLayoutCreate(&Ddesc_fp4, CUDA_R_32F, N, M, N);
        if (status != CUBLAS_STATUS_SUCCESS) break;

        status = cublasLtMatmulDescCreate(&opDesc_fp4, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (status != CUBLAS_STATUS_SUCCESS) break;

        cublasOperation_t transA = CUBLAS_OP_T;
        cublasOperation_t transB = CUBLAS_OP_N;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
        if (status != CUBLAS_STATUS_SUCCESS) break;

        int32_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_A, sizeof(d_scale_A));
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_B, sizeof(d_scale_B));
        if (status != CUBLAS_STATUS_SUCCESS) break;

        status = cublasLtMatmulPreferenceCreate(&pref_fp4);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulPreferenceSetAttribute(pref_fp4, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize_fp4, sizeof(workspaceSize_fp4));
        if (status != CUBLAS_STATUS_SUCCESS) break;

        CHECK_CUDA(cudaMalloc(&workspace_fp4, workspaceSize_fp4));

        std::vector<cublasLtMatmulHeuristicResult_t> heuristicResults(1);
        int returnedAlgoCount = 0;
        status = cublasLtMatmulAlgoGetHeuristic(
            ltHandle, opDesc_fp4, Adesc_fp4, Bdesc_fp4, Cdesc_fp4, Ddesc_fp4, pref_fp4, 1, heuristicResults.data(), &returnedAlgoCount
        );
        if (status != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) break;

        fp4_tested = true;
        for (int i = 0; i < warmup_runs; ++i) {
            cublasLtMatmul(ltHandle, opDesc_fp4, &alpha, d_B_fp4, Adesc_fp4, d_A_fp4, Bdesc_fp4, &beta, d_C, Cdesc_fp4, d_C, Ddesc_fp4, &heuristicResults[0].algo, workspace_fp4, workspaceSize_fp4, nullptr);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < benchmark_runs; ++i) {
            cublasLtMatmul(ltHandle, opDesc_fp4, &alpha, d_B_fp4, Adesc_fp4, d_A_fp4, Bdesc_fp4, &beta, d_C, Cdesc_fp4, d_C, Ddesc_fp4, &heuristicResults[0].algo, workspace_fp4, workspaceSize_fp4, nullptr);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&ms_cublas_fp4, start, stop));
        float avg_sec_cublas_fp4 = (ms_cublas_fp4 / benchmark_runs) / 1000.0f;
        tflops_cublas_fp4 = (gflops_base / avg_sec_cublas_fp4) / 1e12;

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
        cublas_fp4_ok = verify_results(h_A.data(), h_B.data(), h_C.data(), M, N, K, 128, 5e-1f);
    } while (0);

    if (workspace_fp4) cudaFree(workspace_fp4);
    if (pref_fp4) cublasLtMatmulPreferenceDestroy(pref_fp4);
    if (opDesc_fp4) cublasLtMatmulDescDestroy(opDesc_fp4);
    if (Adesc_fp4) cublasLtMatrixLayoutDestroy(Adesc_fp4);
    if (Bdesc_fp4) cublasLtMatrixLayoutDestroy(Bdesc_fp4);
    if (Cdesc_fp4) cublasLtMatrixLayoutDestroy(Cdesc_fp4);
    if (Ddesc_fp4) cublasLtMatrixLayoutDestroy(Ddesc_fp4);

    // Print Results Table
    std::cout << std::left << std::setw(30) << "Implementation"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "TFLOPS"
              << std::setw(12) << "Correct?" << "\n";
    std::cout << "---------------------------------------------------------------------------\n";

    std::cout << std::setw(30) << "cuBLAS SGEMM (FP32)"
              << std::setw(15) << std::fixed << std::setprecision(3) << (ms_cublas_fp32 / benchmark_runs)
              << std::setw(18) << std::setprecision(4) << tflops_cublas_fp32
              << std::setw(12) << (cublas_fp32_ok ? "PASS" : "FAIL") << "\n";

    std::cout << std::setw(30) << "Custom WMMA (FP16)"
              << std::setw(15) << (ms_custom_wmma / benchmark_runs)
              << std::setw(18) << tflops_custom_wmma
              << std::setw(12) << (custom_wmma_ok ? "PASS" : "FAIL") << "\n";

    if (fp16_tested) {
        std::cout << std::setw(30) << "cuBLAS HGEMM (FP16)"
                  << std::setw(15) << (ms_cublas_fp16 / benchmark_runs)
                  << std::setw(18) << tflops_cublas_fp16
                  << std::setw(12) << (cublas_fp16_ok ? "PASS" : "FAIL") << "\n";
    }

    if (fp8_tested) {
        std::cout << std::setw(30) << "cuBLASLt GEMM (FP8)"
                  << std::setw(15) << (ms_cublas_fp8 / benchmark_runs)
                  << std::setw(18) << tflops_cublas_fp8
                  << std::setw(12) << (cublas_fp8_ok ? "PASS" : "FAIL") << "\n";
    }

    if (fp4_tested) {
        std::cout << std::setw(30) << "cuBLASLt GEMM (FP4/NVFP4)"
                  << std::setw(15) << (ms_cublas_fp4 / benchmark_runs)
                  << std::setw(18) << tflops_cublas_fp4
                  << std::setw(12) << (cublas_fp4_ok ? "PASS" : "FAIL") << "\n";
    }

    // Clean up device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_half));
    CHECK_CUDA(cudaFree(d_B_half));
    CHECK_CUDA(cudaFree(d_A_fp8));
    CHECK_CUDA(cudaFree(d_B_fp8));
    CHECK_CUDA(cudaFree(d_A_fp4));
    CHECK_CUDA(cudaFree(d_B_fp4));
    CHECK_CUDA(cudaFree(d_scale_A));
    CHECK_CUDA(cudaFree(d_scale_B));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int clockRateKHz = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device));

    std::cout << "============================================\n";
    std::cout << "WMMA & NVFP4 Tensor Core GEMM Benchmark\n";
    std::cout << "============================================\n";
    std::cout << "GPU Device ID: " << device << "\n";
    std::cout << "GPU Name: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Multiprocessors (SMs): " << prop.multiProcessorCount << "\n";
    std::cout << "GPU Clock Rate: " << clockRateKHz / 1000.0f << " MHz\n";

    // Estimate peak FLOPS
    int cores_per_sm = 128;
    double peak_flops = (double)prop.multiProcessorCount * cores_per_sm * ((double)clockRateKHz * 1e3) * 2.0;
    std::cout << "Theoretical Peak FP32 GFLOPS: " << std::fixed << std::setprecision(1) << (peak_flops / 1e9) << " GFLOPS (" << (peak_flops / 1e12) << " TFLOPS)\n";
    std::cout << "============================================\n";

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // Matrix sizes to run
    std::vector<int> sizes = {1024, 2048, 4096, 8192};
    for (int size : sizes) {
        run_benchmark(handle, ltHandle, size);
    }

    CHECK_CUBLAS(cublasLtDestroy(ltHandle));
    CHECK_CUBLAS(cublasDestroy(handle));
    std::cout << "\nBenchmark finished successfully.\n";
    return 0;
}
