# lquid

## high level design

1. user writes an op in rust, that takes in the following:

- host profile (number of cores, cache sizes, ...)
- device profile (L1 / L2 cache sizes, warp size, ...)
- any parameters specific to that op (e.g. for a gemm A, B, and C shapes)

2. the function outputs IR.
3. IR has a compilation pipeline for the target device
   1. this uses the native compiler toolchain:
      1. nvidia: nvvm IR -> cubin
      2. amd:k1G

## Milestones

### M0

The first goal is to get to _some_ op that is optimized based on the target
architecture.

For the first milestone, we will be re-using the triton IR to generate various
kernels for a target device. The steps here will look like the following:

1. the front end consumes an ONNX model
2. map an ONNX GEMM to a rust function that will generate optimized triton IR for
   that particular target profile.
3. under the hood, use the triton compiler to compile the triton IR to the
   target architecture.
4. have an example script that will load the compiled artifact and run inference
   on the target.

### M1

- optimization on kernels that require host <-> device transfers

## Terminology

The terminology in this document largely aligns with NVidia's nomenclature for
CUDA.

- thread: a single unit of execution on the GPU.
- warps: a collection of threads will be executed in parallel, using a
  single-instruction-multiple-threads (SIMT paradigm).
- block: a collection of threads that has the constraint of running on a single
  multiprocessor.
- grid: a collection of blocks.

## design thoughts

### general model execution workflow

For many cases, the workflow for a ML model can be generalized:

1. copy data from host to device
2. execute one or more kernels on device
3. copy data from device to host

2 can be more complicated, as it may involve multiple transfers betweeen the
host and the device, depending on the size of the model that needs to be
executed and the fact that the device may not have sufficient memory to hold the
entire model. In this case, latency hiding, via performing reads simultaneously
while executing a kernel, is valuable.

### warp throughput

warp throughput can be maximized by minimizing the number of control flow
instructions within a warp. This is because control flow instructions can cause
warp divergence, where threads in a warp execute different instructions, in many cases forcing some warps to stall
waiting for others to finish.

### types of operations to implement

- Layer fusion: fusion of various layers helps enables better memory locality,
  avoiding the need for expensive transfers. The IR requires a way to identify
  and map these fusions to optimized kernels.
- Tiling: the sizes of the tiling that occurs on a given op can also be
  optimized for a given architecture. For example, if there is an optimal warp
  (collection of thread) size for a given architecture due to the number of
  simultaneous multiprocessors, the size of the block should be hand-picked for
  that given architecture, and the orientation of the data should be selected
  such that the cache locality is optimized.

### ONNX

ONNX is a protobuf format that represents a model at a graph of nodes.

lquid reads onnx graphs as it provides a level of abstraction from which a
compiler could produce an optimized runtime.

### kernels

A common GPU paradigm is that kernels are compiled, and then executed directly
on the device. so part of the compilation pipeline should include the
compilation of kernels that will run on the GPU.

### Nvidia GPU compilation path

1. author NVVM IR
2. use nvvc to compile the IR to a .cubin file
3. the .cubin file must be executed via a program that is using the CUDA driver API.

### Usage of memory models

Even during an execution of a model, we have to be able to possibly leverage
every possible storage of temporary that we can. This includes

- registers / SRAM
- L1 / L2 / L3 cached
- high-bandwidth-memory (HBM)
- system DRAM

Many kernels will only need ram that resides on the device GPU. In some cases
there is a need to move between host and device memory (things like the Tensor
memory accelerator can help with this).

### The optimizer needs to have device-specific parameters

An AOT optimizer will require some understanding of both host and device to
optimize.

- a hand-written optimization algorithm is probably best.
- generate_kernel(host_profile, device_profile) -> IR
- IR --> native_compiler --> assembly
