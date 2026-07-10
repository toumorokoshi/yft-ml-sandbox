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
#include <unordered_map>
#include <string>

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

__global__ void fillScalesKernel(__nv_fp8_storage_t* scales, int size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scales[idx] = __nv_cvt_float_to_fp8(val, __NV_NOSAT, __NV_E4M3);
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

// Custom Warp-level C++ WMMA GEMM with Larger Tile Sizes
// Each block computes a 64x64 tile of C, and each warp computes a 32x32 tile using 4 accumulator fragments.
__global__ void gemm_wmma_fp16_tiled_32x32(
    const __half* A, const __half* B, float* C,
    int M, int N, int K) {

    // Each block contains blockDim.x = 32 (1 warp) and blockDim.y = 4 (4 warps total)
    // We arrange the 4 warps as a 2x2 grid inside the 64x64 block tile.
    int warpId = threadIdx.y;
    int warpRow = warpId / 2; // 0 or 1
    int warpCol = warpId % 2; // 0 or 1

    int row = (blockIdx.y * 2 + warpRow) * 32;
    int col = (blockIdx.x * 2 + warpCol) * 32;

    if (row >= M || col >= N) return;

    // Declare 4 accumulator fragments for the 32x32 output tile
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
    wmma::fill_fragment(c_frag[0][0], 0.0f);
    wmma::fill_fragment(c_frag[0][1], 0.0f);
    wmma::fill_fragment(c_frag[1][0], 0.0f);
    wmma::fill_fragment(c_frag[1][1], 0.0f);

    // Loop over the K dimension in steps of 16
    for (int k = 0; k < K; k += 16) {
        // Load two A fragments along the rows (row and row + 16)
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2];
        wmma::load_matrix_sync(a_frag[0], A + row * K + k, K);
        wmma::load_matrix_sync(a_frag[1], A + (row + 16) * K + k, K);

        // Load two B fragments along the columns (col and col + 16)
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2];
        wmma::load_matrix_sync(b_frag[0], B + k * N + col, N);
        wmma::load_matrix_sync(b_frag[1], B + k * N + col + 16, N);

        // Perform 4 matrix multiplications (2x2 MMA tiling)
        wmma::mma_sync(c_frag[0][0], a_frag[0], b_frag[0], c_frag[0][0]);
        wmma::mma_sync(c_frag[0][1], a_frag[0], b_frag[1], c_frag[0][1]);
        wmma::mma_sync(c_frag[1][0], a_frag[1], b_frag[0], c_frag[1][0]);
        wmma::mma_sync(c_frag[1][1], a_frag[1], b_frag[1], c_frag[1][1]);
    }

    // Store the 4 accumulator fragments back to global memory
    wmma::store_matrix_sync(C + row * N + col, c_frag[0][0], N, wmma::mem_row_major);
    wmma::store_matrix_sync(C + row * N + col + 16, c_frag[0][1], N, wmma::mem_row_major);
    wmma::store_matrix_sync(C + (row + 16) * N + col, c_frag[1][0], N, wmma::mem_row_major);
    wmma::store_matrix_sync(C + (row + 16) * N + col + 16, c_frag[1][1], N, wmma::mem_row_major);
}

// Convert FP4 back to float on host for verification
void fp4ToFloat(const __nv_fp4_storage_t* src, float* dst, size_t size, float scale_D) {
    for (size_t i = 0; i < size / 2; ++i) {
        uint8_t byte = src[i];
        __nv_fp4_storage_t raw1 = byte & 0x0F;
        __nv_fp4_storage_t raw2 = (byte >> 4) & 0x0F;
        __half_raw h1 = __nv_cvt_fp4_to_halfraw(raw1, __NV_E2M1);
        __half_raw h2 = __nv_cvt_fp4_to_halfraw(raw2, __NV_E2M1);
        half h1_val = *reinterpret_cast<half*>(&h1);
        half h2_val = *reinterpret_cast<half*>(&h2);
        dst[2 * i] = __half2float(h1_val) * scale_D;
        dst[2 * i + 1] = __half2float(h2_val) * scale_D;
    }
}

// Host-based FP4 writeback verification (handling quantization steps)
bool verify_fp4_writeback(const float* A, const float* B, const float* C_reconstructed, int M, int N, int K, int check_size) {
    int limit_M = std::min(M, check_size);
    int limit_N = std::min(N, check_size);

    for (int r = 0; r < limit_M; ++r) {
        for (int c = 0; c < limit_N; ++c) {
            double expected = 0.0;
            for (int k = 0; k < K; ++k) {
                expected += (double)A[r * K + k] * (double)B[k * N + c];
            }
            float actual = C_reconstructed[r * N + c];
            double diff = std::abs(expected - actual);
            double max_allowed_error = (double)K / 8.0;
            if (diff > max_allowed_error) {
                std::cerr << "FP4 Writeback verification failed at (" << r << ", " << c
                          << "): expected " << expected << ", got " << actual
                          << ", diff = " << diff << ", max_allowed_error = " << max_allowed_error << std::endl;
                return false;
            }
        }
    }
    return true;
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

struct FP4WritebackResult {
    bool tested = false;
    float time_ms = 0.0f;
    double tflops = 0.0;
    bool ok = false;
};

FP4WritebackResult run_cublaslt_fp4_writeback(
    cublasLtHandle_t ltHandle,
    int M, int N, int K,
    const __nv_fp4_storage_t* d_A_fp4,
    const __nv_fp4_storage_t* d_B_fp4,
    __nv_fp4_storage_t* d_D_fp4,
    __half* d_C_half,
    __nv_fp8_storage_t* d_scale_A,
    __nv_fp8_storage_t* d_scale_B,
    __nv_fp8_storage_t* d_scale_D,
    const std::vector<float>& h_A,
    const std::vector<float>& h_B,
    int warmup_runs,
    int benchmark_runs,
    double gflops_base,
    cudaEvent_t start,
    cudaEvent_t stop
) {
    FP4WritebackResult res;

    cublasLtMatrixLayout_t Adesc_fp4_wb = nullptr, Bdesc_fp4_wb = nullptr, Cdesc_fp4_wb = nullptr, Ddesc_fp4_wb = nullptr;
    cublasLtMatmulDesc_t opDesc_fp4_wb = nullptr;
    cublasLtMatmulPreference_t pref_fp4_wb = nullptr;
    void* workspace_fp4_wb = nullptr;
    uint64_t workspaceSize_fp4_wb = 128 * 1024 * 1024;

    do {
        cublasStatus_t status;
        status = cublasLtMatrixLayoutCreate(&Adesc_fp4_wb, CUDA_R_4F_E2M1, K, N, K);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatrixLayoutCreate(&Bdesc_fp4_wb, CUDA_R_4F_E2M1, K, M, K);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatrixLayoutCreate(&Cdesc_fp4_wb, CUDA_R_16F, N, M, N);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatrixLayoutCreate(&Ddesc_fp4_wb, CUDA_R_4F_E2M1, N, M, N);
        if (status != CUBLAS_STATUS_SUCCESS) break;

        status = cublasLtMatmulDescCreate(&opDesc_fp4_wb, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (status != CUBLAS_STATUS_SUCCESS) break;

        cublasOperation_t transA = CUBLAS_OP_T;
        cublasOperation_t transB = CUBLAS_OP_N;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4_wb, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4_wb, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
        if (status != CUBLAS_STATUS_SUCCESS) break;

        int32_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4_wb, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4_wb, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4_wb, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_A, sizeof(d_scale_A));
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4_wb, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_B, sizeof(d_scale_B));
        if (status != CUBLAS_STATUS_SUCCESS) break;

        // Set D output scale mode and pointer
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4_wb, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &scale_mode, sizeof(scale_mode));
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulDescSetAttribute(opDesc_fp4_wb, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_scale_D, sizeof(d_scale_D));
        if (status != CUBLAS_STATUS_SUCCESS) break;

        status = cublasLtMatmulPreferenceCreate(&pref_fp4_wb);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        status = cublasLtMatmulPreferenceSetAttribute(pref_fp4_wb, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize_fp4_wb, sizeof(workspaceSize_fp4_wb));
        if (status != CUBLAS_STATUS_SUCCESS) break;

        CHECK_CUDA(cudaMalloc(&workspace_fp4_wb, workspaceSize_fp4_wb));

        int scale_D_size = N * (M / 16);
        float scale_D_val = (float)K / 20.0f;
        int threads_convert = 256;
        fillScalesKernel<<< (scale_D_size + threads_convert - 1) / threads_convert, threads_convert>>>(d_scale_D, scale_D_size, scale_D_val);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<cublasLtMatmulHeuristicResult_t> heuristicResults(40);
        int returnedAlgoCount = 0;
        status = cublasLtMatmulAlgoGetHeuristic(
            ltHandle, opDesc_fp4_wb, Adesc_fp4_wb, Bdesc_fp4_wb, Cdesc_fp4_wb, Ddesc_fp4_wb, pref_fp4_wb, 40, heuristicResults.data(), &returnedAlgoCount
        );
        if (status != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) break;

        float best_ms = 1e9f;
        double best_tflops = 0.0;
        bool best_ok = false;
        cublasLtMatmulAlgo_t best_algo;
        bool found_any = false;

        float alpha = 1.0f;
        float beta = 0.0f;

        for (int a = 0; a < returnedAlgoCount; ++a) {
            std::vector<int32_t> tiles_to_try = {-1, CUBLASLT_MATMUL_TILE_128x128};
            std::vector<int32_t> splitk_vals = {1, 2, 4, 8};
            for (int32_t tileId : tiles_to_try) {
                for (int32_t splitk : splitk_vals) {
                    cublasLtMatmulAlgo_t algo = heuristicResults[a].algo;
                    if (tileId != -1) {
                        status = cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId));
                        if (status != CUBLAS_STATUS_SUCCESS) continue;
                    }
                    status = cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk));
                    if (status != CUBLAS_STATUS_SUCCESS) continue;

                    // Verify compatibility
                    cublasStatus_t run_status = cublasLtMatmul(
                        ltHandle, opDesc_fp4_wb, &alpha, d_B_fp4, Adesc_fp4_wb, d_A_fp4, Bdesc_fp4_wb, &beta,
                        d_C_half, Cdesc_fp4_wb, d_D_fp4, Ddesc_fp4_wb, &algo, workspace_fp4_wb, workspaceSize_fp4_wb, nullptr
                    );
                    if (run_status != CUBLAS_STATUS_SUCCESS) continue;

                    // Warmup
                    for (int i = 0; i < warmup_runs; ++i) {
                        cublasLtMatmul(ltHandle, opDesc_fp4_wb, &alpha, d_B_fp4, Adesc_fp4_wb, d_A_fp4, Bdesc_fp4_wb, &beta, d_C_half, Cdesc_fp4_wb, d_D_fp4, Ddesc_fp4_wb, &algo, workspace_fp4_wb, workspaceSize_fp4_wb, nullptr);
                    }
                    CHECK_CUDA(cudaDeviceSynchronize());

                    CHECK_CUDA(cudaEventRecord(start));
                    for (int i = 0; i < benchmark_runs; ++i) {
                        cublasLtMatmul(ltHandle, opDesc_fp4_wb, &alpha, d_B_fp4, Adesc_fp4_wb, d_A_fp4, Bdesc_fp4_wb, &beta, d_C_half, Cdesc_fp4_wb, d_D_fp4, Ddesc_fp4_wb, &algo, workspace_fp4_wb, workspaceSize_fp4_wb, nullptr);
                    }
                    CHECK_CUDA(cudaEventRecord(stop));
                    CHECK_CUDA(cudaEventSynchronize(stop));

                    float ms = 0.0f;
                    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

                    // Verification
                    std::vector<__nv_fp4_storage_t> h_D_fp4(M * N / 2);
                    CHECK_CUDA(cudaMemcpy(h_D_fp4.data(), d_D_fp4, M * N / 2, cudaMemcpyDeviceToHost));
                    std::vector<float> h_D_float(M * N);
                    fp4ToFloat(h_D_fp4.data(), h_D_float.data(), M * N, scale_D_val);
                    bool ok = verify_fp4_writeback(h_A.data(), h_B.data(), h_D_float.data(), M, N, K, 128);

                    if (ok && ms < best_ms) {
                        best_ms = ms;
                        best_tflops = (gflops_base / ((ms / benchmark_runs) / 1000.0f)) / 1e12;
                        best_ok = true;
                        best_algo = algo;
                        found_any = true;
                    }
                }
            }
        }

        if (found_any) {
            res.tested = true;
            res.time_ms = best_ms;
            res.tflops = best_tflops;
            res.ok = best_ok;

            // Re-run best to ensure state correctness
            cublasLtMatmul(
                ltHandle, opDesc_fp4_wb, &alpha, d_B_fp4, Adesc_fp4_wb, d_A_fp4, Bdesc_fp4_wb, &beta,
                d_C_half, Cdesc_fp4_wb, d_D_fp4, Ddesc_fp4_wb, &best_algo, workspace_fp4_wb, workspaceSize_fp4_wb, nullptr
            );
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    } while (0);

    if (workspace_fp4_wb) cudaFree(workspace_fp4_wb);
    if (pref_fp4_wb) cublasLtMatmulPreferenceDestroy(pref_fp4_wb);
    if (opDesc_fp4_wb) cublasLtMatmulDescDestroy(opDesc_fp4_wb);
    if (Adesc_fp4_wb) cublasLtMatrixLayoutDestroy(Adesc_fp4_wb);
    if (Bdesc_fp4_wb) cublasLtMatrixLayoutDestroy(Bdesc_fp4_wb);
    if (Cdesc_fp4_wb) cublasLtMatrixLayoutDestroy(Cdesc_fp4_wb);
    if (Ddesc_fp4_wb) cublasLtMatrixLayoutDestroy(Ddesc_fp4_wb);

    return res;
}

// Main benchmark runner
void run_benchmark(cublasHandle_t handle, cublasLtHandle_t ltHandle, int N_size) {
    int M = N_size;
    int N = N_size;
    int K = N_size;

    struct FP4TileResult {
        std::string name;
        float time_ms;
        double tflops;
        bool ok;
    };
    std::vector<FP4TileResult> fp4_tile_results;

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
    __half *d_A_half = nullptr, *d_B_half = nullptr, *d_C_half = nullptr;
    __nv_fp8_storage_t *d_A_fp8 = nullptr, *d_B_fp8 = nullptr;
    __nv_fp4_storage_t *d_A_fp4 = nullptr, *d_B_fp4 = nullptr, *d_D_fp4 = nullptr;
    __nv_fp8_storage_t *d_scale_A = nullptr, *d_scale_B = nullptr, *d_scale_D = nullptr;

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

    size_t size_C_half = (size_t)M * N * sizeof(__half);
    size_t size_D_fp4 = ((size_t)M * N) / 2 * sizeof(__nv_fp4_storage_t);
    int scale_D_size = M * (N / 16);
    CHECK_CUDA(cudaMalloc(&d_C_half, size_C_half));
    CHECK_CUDA(cudaMalloc(&d_D_fp4, size_D_fp4));
    CHECK_CUDA(cudaMalloc(&d_scale_D, scale_D_size));
    CHECK_CUDA(cudaMemset(d_C_half, 0, size_C_half));

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
    // 2b. Custom WMMA GEMM V2 (32x32 Warp Tile, 64x64 Block Tile)
    // ----------------------------------------------------
    dim3 block_wmma_tiled(32, 4);
    dim3 grid_wmma_tiled((N + 63) / 64, (M + 63) / 64);

    for (int i = 0; i < warmup_runs; ++i) {
        gemm_wmma_fp16_tiled_32x32<<<grid_wmma_tiled, block_wmma_tiled>>>(d_A_half, d_B_half, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < benchmark_runs; ++i) {
        gemm_wmma_fp16_tiled_32x32<<<grid_wmma_tiled, block_wmma_tiled>>>(d_A_half, d_B_half, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_custom_wmma_tiled = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_custom_wmma_tiled, start, stop));
    float avg_sec_custom_wmma_tiled = (ms_custom_wmma_tiled / benchmark_runs) / 1000.0f;
    double tflops_custom_wmma_tiled = (gflops_base / avg_sec_custom_wmma_tiled) / 1e12;

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    bool custom_wmma_tiled_ok = verify_results(h_A.data(), h_B.data(), h_C.data(), M, N, K, 128, 1e-2f);

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
    uint64_t workspaceSize_fp8 = 128 * 1024 * 1024;

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
    fillScalesKernel<<< (scale_A_size + threads_convert - 1) / threads_convert, threads_convert>>>(d_scale_A, scale_A_size, 1.0f);
    fillScalesKernel<<< (scale_B_size + threads_convert - 1) / threads_convert, threads_convert>>>(d_scale_B, scale_B_size, 1.0f);
    fillScalesKernel<<< (scale_D_size + threads_convert - 1) / threads_convert, threads_convert>>>(d_scale_D, scale_D_size, 1.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    bool fp4_tested = false;

    cublasLtMatrixLayout_t Adesc_fp4 = nullptr, Bdesc_fp4 = nullptr, Cdesc_fp4 = nullptr, Ddesc_fp4 = nullptr;
    cublasLtMatmulDesc_t opDesc_fp4 = nullptr;
    cublasLtMatmulPreference_t pref_fp4 = nullptr;
    void* workspace_fp4 = nullptr;
    uint64_t workspaceSize_fp4 = 128 * 1024 * 1024;

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

        std::vector<cublasLtMatmulHeuristicResult_t> heuristicResults(50);
        int returnedAlgoCount = 0;
        status = cublasLtMatmulAlgoGetHeuristic(
            ltHandle, opDesc_fp4, Adesc_fp4, Bdesc_fp4, Cdesc_fp4, Ddesc_fp4, pref_fp4, 50, heuristicResults.data(), &returnedAlgoCount
        );
        if (status != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) break;

        // Map tile IDs to string representations
        auto get_tile_name = [](int32_t tileId) -> std::string {
            switch (tileId) {
                case CUBLASLT_MATMUL_TILE_8x8: return "8x8";
                case CUBLASLT_MATMUL_TILE_8x16: return "8x16";
                case CUBLASLT_MATMUL_TILE_16x8: return "16x8";
                case CUBLASLT_MATMUL_TILE_8x32: return "8x32";
                case CUBLASLT_MATMUL_TILE_16x16: return "16x16";
                case CUBLASLT_MATMUL_TILE_32x8: return "32x8";
                case CUBLASLT_MATMUL_TILE_8x64: return "8x64";
                case CUBLASLT_MATMUL_TILE_16x32: return "16x32";
                case CUBLASLT_MATMUL_TILE_32x16: return "32x16";
                case CUBLASLT_MATMUL_TILE_64x8: return "64x8";
                case CUBLASLT_MATMUL_TILE_32x32: return "32x32";
                case CUBLASLT_MATMUL_TILE_32x64: return "32x64";
                case CUBLASLT_MATMUL_TILE_64x32: return "64x32";
                case CUBLASLT_MATMUL_TILE_32x128: return "32x128";
                case CUBLASLT_MATMUL_TILE_64x64: return "64x64";
                case CUBLASLT_MATMUL_TILE_128x32: return "128x32";
                case CUBLASLT_MATMUL_TILE_64x128: return "64x128";
                case CUBLASLT_MATMUL_TILE_128x64: return "128x64";
                case CUBLASLT_MATMUL_TILE_64x256: return "64x256";
                case CUBLASLT_MATMUL_TILE_128x128: return "128x128";
                case CUBLASLT_MATMUL_TILE_256x64: return "256x64";
                case CUBLASLT_MATMUL_TILE_64x512: return "64x512";
                case CUBLASLT_MATMUL_TILE_128x256: return "128x256";
                case CUBLASLT_MATMUL_TILE_256x128: return "256x128";
                case CUBLASLT_MATMUL_TILE_512x64: return "512x64";
                default: return "Unknown (" + std::to_string(tileId) + ")";
            }
        };

        struct FP4Run {
            std::string tile_name;
            float time_ms;
            double tflops;
            bool ok;
        };
        std::unordered_map<std::string, FP4Run> best_runs;

        fp4_tested = true;
        // 1. Run over all compatible configurations returned by the heuristic search
        for (int idx = 0; idx < returnedAlgoCount; ++idx) {
            cublasLtMatmulAlgo_t algo = heuristicResults[idx].algo;
            int32_t tileId = 0;
            size_t sizeWritten = 0;
            status = cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId), &sizeWritten);
            if (status != CUBLAS_STATUS_SUCCESS) continue;

            std::string tile_name = get_tile_name(tileId);

            cublasStatus_t run_status = cublasLtMatmul(
                ltHandle, opDesc_fp4, &alpha, d_B_fp4, Adesc_fp4, d_A_fp4, Bdesc_fp4, &beta,
                d_C, Cdesc_fp4, d_C, Ddesc_fp4, &algo, workspace_fp4, workspaceSize_fp4, nullptr
            );
            if (run_status != CUBLAS_STATUS_SUCCESS) continue;

            // Warmup
            for (int i = 0; i < warmup_runs; ++i) {
                cublasLtMatmul(ltHandle, opDesc_fp4, &alpha, d_B_fp4, Adesc_fp4, d_A_fp4, Bdesc_fp4, &beta, d_C, Cdesc_fp4, d_C, Ddesc_fp4, &algo, workspace_fp4, workspaceSize_fp4, nullptr);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < benchmark_runs; ++i) {
                cublasLtMatmul(ltHandle, opDesc_fp4, &alpha, d_B_fp4, Adesc_fp4, d_A_fp4, Bdesc_fp4, &beta, d_C, Cdesc_fp4, d_C, Ddesc_fp4, &algo, workspace_fp4, workspaceSize_fp4, nullptr);
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            float avg_sec = (ms / benchmark_runs) / 1000.0f;
            double tflops = (gflops_base / avg_sec) / 1e12;

            CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
            bool ok = verify_results(h_A.data(), h_B.data(), h_C.data(), M, N, K, 128, 5e-1f);

            if (best_runs.find(tile_name) == best_runs.end() || tflops > best_runs[tile_name].tflops) {
                best_runs[tile_name] = {tile_name, ms, tflops, ok};
            }
        }

        // 2. Try manual configurations on the top algorithm to find others
        struct TileConfig {
            const char* name;
            int32_t tileId;
        };
        std::vector<TileConfig> manual_tiles = {
            {"8x8", CUBLASLT_MATMUL_TILE_8x8},
            {"16x16", CUBLASLT_MATMUL_TILE_16x16},
            {"32x32", CUBLASLT_MATMUL_TILE_32x32},
            {"64x64", CUBLASLT_MATMUL_TILE_64x64},
            {"128x64", CUBLASLT_MATMUL_TILE_128x64},
            {"64x128", CUBLASLT_MATMUL_TILE_64x128},
            {"128x128", CUBLASLT_MATMUL_TILE_128x128},
            {"256x64", CUBLASLT_MATMUL_TILE_256x64},
            {"64x256", CUBLASLT_MATMUL_TILE_64x256},
            {"256x128", CUBLASLT_MATMUL_TILE_256x128},
            {"128x256", CUBLASLT_MATMUL_TILE_128x256},
        };

        if (M == 1024) {
            std::cout << "\nScanning cuBLASLt FP4 manual tile layout compatibility:\n";
        }
        for (const auto& tile : manual_tiles) {
            std::string tile_name = tile.name;

            cublasLtMatmulAlgo_t algo = heuristicResults[0].algo;
            status = cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile.tileId, sizeof(tile.tileId));
            if (status != CUBLAS_STATUS_SUCCESS) {
                if (M == 1024) std::cout << "  Tile " << tile_name << ": Config Set failed\n";
                continue;
            }

            cublasStatus_t run_status = cublasLtMatmul(
                ltHandle, opDesc_fp4, &alpha, d_B_fp4, Adesc_fp4, d_A_fp4, Bdesc_fp4, &beta,
                d_C, Cdesc_fp4, d_C, Ddesc_fp4, &algo, workspace_fp4, workspaceSize_fp4, nullptr
            );
            if (run_status != CUBLAS_STATUS_SUCCESS) {
                if (M == 1024) std::cout << "  Tile " << tile_name << ": Execution failed (Status: " << run_status << ")\n";
                continue;
            }

            if (M == 1024) std::cout << "  Tile " << tile_name << ": SUCCESS!\n";
            if (best_runs.find(tile_name) != best_runs.end()) continue;

            // Warmup
            for (int i = 0; i < warmup_runs; ++i) {
                cublasLtMatmul(ltHandle, opDesc_fp4, &alpha, d_B_fp4, Adesc_fp4, d_A_fp4, Bdesc_fp4, &beta, d_C, Cdesc_fp4, d_C, Ddesc_fp4, &algo, workspace_fp4, workspaceSize_fp4, nullptr);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < benchmark_runs; ++i) {
                cublasLtMatmul(ltHandle, opDesc_fp4, &alpha, d_B_fp4, Adesc_fp4, d_A_fp4, Bdesc_fp4, &beta, d_C, Cdesc_fp4, d_C, Ddesc_fp4, &algo, workspace_fp4, workspaceSize_fp4, nullptr);
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            float avg_sec = (ms / benchmark_runs) / 1000.0f;
            double tflops = (gflops_base / avg_sec) / 1e12;

            CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
            bool ok = verify_results(h_A.data(), h_B.data(), h_C.data(), M, N, K, 128, 5e-1f);

            best_runs[tile_name] = {tile_name, ms, tflops, ok};
        }

        // Copy consolidated runs to final array
        for (const auto& pair : best_runs) {
            fp4_tile_results.push_back({pair.second.tile_name, pair.second.time_ms, pair.second.tflops, pair.second.ok});
        }
    } while (0);

    if (workspace_fp4) cudaFree(workspace_fp4);
    if (pref_fp4) cublasLtMatmulPreferenceDestroy(pref_fp4);
    if (opDesc_fp4) cublasLtMatmulDescDestroy(opDesc_fp4);
    if (Adesc_fp4) cublasLtMatrixLayoutDestroy(Adesc_fp4);
    if (Bdesc_fp4) cublasLtMatrixLayoutDestroy(Bdesc_fp4);
    if (Cdesc_fp4) cublasLtMatrixLayoutDestroy(Cdesc_fp4);
    if (Ddesc_fp4) cublasLtMatrixLayoutDestroy(Ddesc_fp4);

    // ----------------------------------------------------
    // 6. cuBLASLt FP4 Tensor Core GEMM (FP4 Writeback)
    // ----------------------------------------------------
    FP4WritebackResult fp4_wb_res = run_cublaslt_fp4_writeback(
        ltHandle, M, N, K,
        d_A_fp4, d_B_fp4, d_D_fp4, d_C_half,
        d_scale_A, d_scale_B, d_scale_D,
        h_A, h_B,
        warmup_runs, benchmark_runs, gflops_base,
        start, stop
    );

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

    std::cout << std::setw(30) << "Custom WMMA V2 (FP16, 32x32)"
              << std::setw(15) << (ms_custom_wmma_tiled / benchmark_runs)
              << std::setw(18) << tflops_custom_wmma_tiled
              << std::setw(12) << (custom_wmma_tiled_ok ? "PASS" : "FAIL") << "\n";

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
        for (const auto& res : fp4_tile_results) {
            std::cout << std::setw(30) << ("cuBLASLt FP4 (Tile " + res.name + ")")
                      << std::setw(15) << std::fixed << std::setprecision(3) << (res.time_ms / benchmark_runs)
                      << std::setw(18) << std::setprecision(4) << res.tflops
                      << std::setw(12) << (res.ok ? "PASS" : "FAIL") << "\n";
        }
    }

    if (fp4_wb_res.tested) {
        std::cout << std::setw(30) << "cuBLASLt FP4 (FP4 Writeback)"
                  << std::setw(15) << std::fixed << std::setprecision(3) << (fp4_wb_res.time_ms / benchmark_runs)
                  << std::setw(18) << std::setprecision(4) << fp4_wb_res.tflops
                  << std::setw(12) << (fp4_wb_res.ok ? "PASS" : "FAIL") << "\n";
    }

    // Clean up device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_half));
    CHECK_CUDA(cudaFree(d_B_half));
    CHECK_CUDA(cudaFree(d_C_half));
    CHECK_CUDA(cudaFree(d_A_fp8));
    CHECK_CUDA(cudaFree(d_B_fp8));
    CHECK_CUDA(cudaFree(d_A_fp4));
    CHECK_CUDA(cudaFree(d_B_fp4));
    CHECK_CUDA(cudaFree(d_D_fp4));
    CHECK_CUDA(cudaFree(d_scale_A));
    CHECK_CUDA(cudaFree(d_scale_B));
    CHECK_CUDA(cudaFree(d_scale_D));
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
