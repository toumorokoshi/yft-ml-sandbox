# lquid

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
