// peak_mma.cu - pure tensor-core peak microbenchmark
//
// Copyright (c) 2026, Alfonso De Gregorio
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Derived from: https://github.com/secYOUre/nvfp4bench/blob/main/src/peak_mma.cu
//
// Issues many INDEPENDENT back-to-back FP4 MMAs from registers (no memory traffic,
// no shared, no L2 effects) to measure the raw MMA issue rate -> peak TFLOPS, for
// DENSE (m16n8k32).
//
//   nvcc -gencode=arch=compute_121a,code=sm_121a -O3 -o peak_mma src/peak_mma.cu && ./peak_mma
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

#define NACC 16   // independent accumulators per warp -> hide MMA latency

__global__ void peak_dense(int iters, float* sink) {
  uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0,b0=a0,b1=a0;
  const uint32_t sf=0x81u; const uint16_t z=0;
  float d[NACC][4];
  #pragma unroll
  for (int j=0;j<NACC;++j) d[j][0]=d[j][1]=d[j][2]=d[j][3]=0.f;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  for (int i=0;i<iters;++i) {
    #pragma unroll
    for (int j=0;j<NACC;++j) {
      asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col."
        "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d[j][0]),"=f"(d[j][1]),"=f"(d[j][2]),"=f"(d[j][3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(d[j][0]),"f"(d[j][1]),"f"(d[j][2]),"f"(d[j][3]),
          "r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
    }
  }
#endif
  float s=0;
  #pragma unroll
  for (int j=0;j<NACC;++j) s += d[j][0]+d[j][1]+d[j][2]+d[j][3];
  if (s < 0) sink[threadIdx.x] = s;
}

// --- native PACKED NVFP4: 2 codes/byte -> m16n8k64 dense (2x the K of mxf8f6f4) ---
__global__ void peak_mxf4_dense(int iters, float* sink) {
  uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0, b0=a0,b1=a0;
  uint32_t sf=0x00008181u; const uint16_t z=0;
  float d[NACC][4];
  #pragma unroll
  for (int j=0;j<NACC;++j) d[j][0]=d[j][1]=d[j][2]=d[j][3]=0.f;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  for (int i=0;i<iters;++i) {
    #pragma unroll
    for (int j=0;j<NACC;++j) {
      asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col."
        "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d[j][0]),"=f"(d[j][1]),"=f"(d[j][2]),"=f"(d[j][3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(d[j][0]),"f"(d[j][1]),"f"(d[j][2]),"f"(d[j][3]),
          "r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
    }
  }
#endif
  float s=0;
  #pragma unroll
  for (int j=0;j<NACC;++j) s += d[j][0]+d[j][1]+d[j][2]+d[j][3];
  if (s < 0) sink[threadIdx.x] = s;
}

int main() {
  int dev=0; cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
  int blocks = prop.multiProcessorCount * 8;   // saturate the GPU
  int block  = 128;                            // 4 warps/block
  int iters  = 700;                            // short bursts -> capture un-throttled peak
  long warps = long(blocks) * (block / 32);
  float* sink; cudaMalloc(&sink, sizeof(float) * block);

  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  enum Kern { K_DENSE, K_MXF4_DENSE };
  auto launch = [&](Kern k){
    if      (k==K_DENSE)        peak_dense<<<blocks,block>>>(iters, sink);
    else                        peak_mxf4_dense<<<blocks,block>>>(iters, sink);
  };
  auto bench = [&](const char* name, Kern k, double flops_per_mma, double spec) {
    launch(k); cudaDeviceSynchronize();              // warmup
    double best = 1e30;
    for (int r=0;r<10;++r) {
      cudaEventRecord(s); launch(k); cudaEventRecord(e); cudaEventSynchronize(e);
      float ms=0; cudaEventElapsedTime(&ms,s,e); if (ms<best) best=ms;
    }
    double mmas = double(warps) * iters * NACC;
    double tflops = mmas * flops_per_mma / (best*1e-3) / 1e12;
    std::printf("%-16s peak: %.3f ms, %.1f TFLOPS  (%.1f%% of %.1f spec)\n",
                name, best, tflops, 100.0*tflops/spec, spec);
    return tflops;
  };

  std::printf("Device: %s, %d SMs, %ld warps, %d iters x %d acc\n",
              prop.name, prop.multiProcessorCount, warps, iters, NACC);
  double dense  = bench("mxf8f6f4 dense",   K_DENSE,       2.0*16*8*32,  500.0);
  double mxf4   = bench("mxf4nvf4 dense",   K_MXF4_DENSE,  2.0*16*8*64,  500.0);
  std::printf("\nladder: mxf8f6f4 dense %.0f -> packed %.0f TFLOPS\n",
              dense, mxf4);
  std::printf("packed=%.2fx  |  500 TFLOPS = %.1f%% reached\n",
              mxf4/dense, 100.0*mxf4/500.0);
  cudaFree(sink);
  return 0;
}
