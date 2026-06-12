#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>

#define TILE_SIZE 32

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

// Custom Tiled Matrix Multiplication (FP32)
__global__ void gemm_custom_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A into shared memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B into shared memory
        if ((t * TILE_SIZE + threadIdx.y) < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute product of the loaded tiles
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host-based sub-matrix verification
bool verify_results(const float* A, const float* B, const float* C, int M, int N, int K, int check_size, float rel_tolerance = 1e-4f) {
    // Only check a smaller corner of the matrix to save host execution time
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
            // Allow small absolute difference or relative error within threshold
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
void run_benchmark(cublasHandle_t handle, int N_size) {
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

    // Initialize inputs with reasonable values
    for (size_t i = 0; i < (size_t)M * K; ++i) h_A[i] = static_cast<float>(rand() % 100) / 100.0f;
    for (size_t i = 0; i < (size_t)K * N; ++i) h_B[i] = static_cast<float>(rand() % 100) / 100.0f;

    // Device pointers
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    __half *d_A_half = nullptr, *d_B_half = nullptr, *d_C_half = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // Allocate half-precision memory for FP16 benchmarks
    size_t size_A_half = (size_t)M * K * sizeof(__half);
    size_t size_B_half = (size_t)K * N * sizeof(__half);
    size_t size_C_half = (size_t)M * N * sizeof(__half);

    CHECK_CUDA(cudaMalloc(&d_A_half, size_A_half));
    CHECK_CUDA(cudaMalloc(&d_B_half, size_B_half));
    CHECK_CUDA(cudaMalloc(&d_C_half, size_C_half));

    // Convert inputs to half on device
    int threads_convert = 256;
    int blocks_convert_A = ((M * K) + threads_convert - 1) / threads_convert;
    int blocks_convert_B = ((K * N) + threads_convert - 1) / threads_convert;

    floatToHalfKernel<<<blocks_convert_A, threads_convert>>>(d_A, d_A_half, M * K);
    floatToHalfKernel<<<blocks_convert_B, threads_convert>>>(d_B, d_B_half, K * N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Performance measurements
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int warmup_runs = 10;
    const int benchmark_runs = 50;
    double gflops_base = 2.0 * M * N * K;

    // ----------------------------------------------------
    // 1. Custom Tiled GEMM (FP32)
    // ----------------------------------------------------
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        gemm_custom_tiled<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < benchmark_runs; ++i) {
        gemm_custom_tiled<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_custom = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_custom, start, stop));
    float avg_sec_custom = (ms_custom / benchmark_runs) / 1000.0f;
    double tflops_custom = (gflops_base / avg_sec_custom) / 1e12;

    // Verify Custom results
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    bool custom_ok = verify_results(h_A.data(), h_B.data(), h_C.data(), M, N, K, 128);

    // ----------------------------------------------------
    // 2. cuBLAS FP32 GEMM (SGEMM)
    // ----------------------------------------------------
    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
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

    // Verify cuBLAS FP32 results
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    bool cublas_fp32_ok = verify_results(h_A.data(), h_B.data(), h_C.data(), M, N, K, 128);

    // ----------------------------------------------------
    // 3. cuBLAS TF32 GEMM (using cublasGemmEx)
    // ----------------------------------------------------
    bool tf32_tested = false;
    float ms_cublas_tf32 = 0.0f;
    double tflops_cublas_tf32 = 0.0f;
    bool cublas_tf32_ok = false;

    // We use cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32
    // Warmup
    cublasStatus_t tf32_status = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_32F, N,
        d_A, CUDA_R_32F, K,
        &beta,
        d_C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT
    );

    if (tf32_status == CUBLAS_STATUS_SUCCESS) {
        tf32_tested = true;
        for (int i = 1; i < warmup_runs; ++i) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_32F, N, d_A, CUDA_R_32F, K, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < benchmark_runs; ++i) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_32F, N, d_A, CUDA_R_32F, K, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&ms_cublas_tf32, start, stop));
        float avg_sec_cublas_tf32 = (ms_cublas_tf32 / benchmark_runs) / 1000.0f;
        tflops_cublas_tf32 = (gflops_base / avg_sec_cublas_tf32) / 1e12;

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
        cublas_tf32_ok = verify_results(h_A.data(), h_B.data(), h_C.data(), M, N, K, 128, 5e-3f);
    }

    // ----------------------------------------------------
    // 4. cuBLAS FP16 Tensor Core GEMM (using cublasGemmEx)
    // ----------------------------------------------------
    // Warmup
    cublasStatus_t fp16_status = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B_half, CUDA_R_16F, N,
        d_A_half, CUDA_R_16F, K,
        &beta,
        d_C_half, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    bool fp16_tested = false;
    float ms_cublas_fp16 = 0.0f;
    double tflops_cublas_fp16 = 0.0f;
    bool cublas_fp16_ok = false;

    if (fp16_status == CUBLAS_STATUS_SUCCESS) {
        fp16_tested = true;
        for (int i = 1; i < warmup_runs; ++i) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_half, CUDA_R_16F, N, d_A_half, CUDA_R_16F, K, &beta, d_C_half, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < benchmark_runs; ++i) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_half, CUDA_R_16F, N, d_A_half, CUDA_R_16F, K, &beta, d_C_half, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&ms_cublas_fp16, start, stop));
        float avg_sec_cublas_fp16 = (ms_cublas_fp16 / benchmark_runs) / 1000.0f;
        tflops_cublas_fp16 = (gflops_base / avg_sec_cublas_fp16) / 1e12;

        // Convert result back to FP32 for verification
        floatToHalfKernel<<<blocks_convert_A, threads_convert>>>(d_A, d_A_half, M * K); // reuse or convert
        int blocks_convert_C = ((M * N) + threads_convert - 1) / threads_convert;
        halfToFloatKernel<<<blocks_convert_C, threads_convert>>>(d_C_half, d_C, M * N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
        // Higher tolerance for FP16
        cublas_fp16_ok = verify_results(h_A.data(), h_B.data(), h_C.data(), M, N, K, 128, 1e-2f);
    }

    // Print Results
    std::cout << std::left << std::setw(30) << "Implementation" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(18) << "TFLOPS" 
              << std::setw(12) << "Correct?" << "\n";
    std::cout << "---------------------------------------------------------------------------\n";

    std::cout << std::setw(30) << "Custom Tiled (FP32)"
              << std::setw(15) << std::fixed << std::setprecision(3) << (ms_custom / benchmark_runs)
              << std::setw(18) << std::setprecision(4) << tflops_custom
              << std::setw(12) << (custom_ok ? "PASS" : "FAIL") << "\n";

    std::cout << std::setw(30) << "cuBLAS SGEMM (FP32)"
              << std::setw(15) << (ms_cublas_fp32 / benchmark_runs)
              << std::setw(18) << tflops_cublas_fp32
              << std::setw(12) << (cublas_fp32_ok ? "PASS" : "FAIL") << "\n";

    if (tf32_tested) {
        std::cout << std::setw(30) << "cuBLAS TF32"
                  << std::setw(15) << (ms_cublas_tf32 / benchmark_runs)
                  << std::setw(18) << tflops_cublas_tf32
                  << std::setw(12) << (cublas_tf32_ok ? "PASS" : "FAIL") << "\n";
    } else {
        std::cout << std::setw(30) << "cuBLAS TF32"
                  << std::setw(15) << "N/A"
                  << std::setw(18) << "N/A"
                  << std::setw(12) << "N/A" << "\n";
    }

    if (fp16_tested) {
        std::cout << std::setw(30) << "cuBLAS HGEMM (FP16)"
                  << std::setw(15) << (ms_cublas_fp16 / benchmark_runs)
                  << std::setw(18) << tflops_cublas_fp16
                  << std::setw(12) << (cublas_fp16_ok ? "PASS" : "FAIL") << "\n";
    } else {
        std::cout << std::setw(30) << "cuBLAS HGEMM (FP16)"
                  << std::setw(15) << "N/A"
                  << std::setw(18) << "N/A"
                  << std::setw(12) << "N/A" << "\n";
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_half));
    CHECK_CUDA(cudaFree(d_B_half));
    CHECK_CUDA(cudaFree(d_C_half));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    // Set device
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    // Query Device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int clockRateKHz = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device));

    std::cout << "============================================\n";
    std::cout << "CUDA GEMM Performance Benchmark\n";
    std::cout << "============================================\n";
    std::cout << "GPU Device ID: " << device << "\n";
    std::cout << "GPU Name: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Multiprocessors (SMs): " << prop.multiProcessorCount << "\n";
    std::cout << "GPU Clock Rate: " << clockRateKHz / 1000.0f << " MHz\n";
    
    // Estimate peak FLOPS
    // Ampere/Ada/Blackwell have 128 FP32 CUDA cores per SM
    int cores_per_sm = 128;
    double peak_flops = (double)prop.multiProcessorCount * cores_per_sm * ((double)clockRateKHz * 1e3) * 2.0; // 2 for FMA
    std::cout << "Theoretical Peak FP32 GFLOPS: " << std::fixed << std::setprecision(1) << (peak_flops / 1e9) << " GFLOPS (" << (peak_flops / 1e12) << " TFLOPS)\n";
    std::cout << "============================================\n";

    // Initialize cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Matrix sizes to run
    std::vector<int> sizes = {1024, 2048, 4096, 8192};
    for (int size : sizes) {
        run_benchmark(handle, size);
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    std::cout << "\nBenchmark finished successfully.\n";
    return 0;
}
