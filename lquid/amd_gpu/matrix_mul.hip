#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_HIP_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != hipSuccess) {
        std::cerr << "HIP Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << hipGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// HIP Kernel for matrix multiplication
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    // Dimensions
    const int M = 512;
    const int N = 512;
    const int K = 512;

    std::cout << "Initializing matrices of size " << M << "x" << N << " on host..." << std::endl;

    // Host matrices
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    // Initialize with simple values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(i % 100) / 10.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(i % 50) / 5.0f;
    }

    // Device pointers
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    // Allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&d_A, M * K * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, K * N * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, M * N * sizeof(float)));

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));

    // Execution configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "Launching HIP matrix multiplication kernel on AMD GPU..." << std::endl;
    
    // Launch kernel
    hipLaunchKernelGGL(matrixMulKernel, numBlocks, threadsPerBlock, 0, 0, d_A, d_B, d_C, M, N, K);

    // Check for kernel launch errors
    CHECK_HIP_ERROR(hipGetLastError());

    // Synchronize device
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    std::cout << "Kernel execution completed. Copying results back to host..." << std::endl;

    // Copy results back to host
    CHECK_HIP_ERROR(hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify results on host
    std::cout << "Validating results..." << std::endl;
    bool success = true;
    for (int r = 0; r < 10; ++r) {
        for (int c = 0; c < 10; ++c) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) {
                expected += h_A[r * K + k] * h_B[k * N + c];
            }
            float actual = h_C[r * N + c];
            if (std::abs(expected - actual) > 1e-3f) {
                std::cerr << "Mismatch at (" << r << ", " << c << "): expected " << expected << ", got " << actual << std::endl;
                success = false;
                break;
            }
        }
        if (!success) break;
    }

    if (success) {
        std::cout << "Verification SUCCESSFUL! GPU results match host reference computation." << std::endl;
        std::cout << "Sample output [0,0]: " << h_C[0] << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
    }

    // Free device memory
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));

    return success ? 0 : 1;
}
