use objc2_foundation::ns_string;
use objc2_metal::{
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLCommandQueue, MTLBuffer,
    MTLCommandBuffer, MTLComputeCommandEncoder, MTLResourceOptions, MTLSize,
    MTLCommandEncoder,
};

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

#[link(name = "Metal", kind = "framework")]
extern "C" {}

fn main() {
    let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
    let name = device.name();
    println!("Got Metal device: {}", name);

    let queue = device.newCommandQueue().expect("Failed to create command queue");
    println!("Created command queue");

    let source = ns_string!(
        "#include <metal_stdlib>\n\
        using namespace metal;\n\
        kernel void add_arrays(device const float* inA [[buffer(0)]],\n\
                               device const float* inB [[buffer(1)]],\n\
                               device float* out [[buffer(2)]],\n\
                               uint id [[thread_position_in_grid]]) {\n\
            out[id] = inA[id] + inB[id];\n\
        }"
    );

    let library = device.newLibraryWithSource_options_error(source, None)
        .expect("Failed to compile shader library");
    println!("Shader library compiled successfully!");

    let function = library.newFunctionWithName(ns_string!("add_arrays"))
        .expect("Failed to find add_arrays function in library");
    println!("Found kernel function");

    let pipeline_state = device.newComputePipelineStateWithFunction_error(&function)
        .expect("Failed to create compute pipeline state");
    println!("Created compute pipeline state");

    const NUM_ELEMENTS: usize = 1024;
    let buffer_size = NUM_ELEMENTS * std::mem::size_of::<f32>();

    let buffer_a = device.newBufferWithLength_options(buffer_size, MTLResourceOptions::StorageModeShared)
        .expect("Failed to allocate buffer A");
    let buffer_b = device.newBufferWithLength_options(buffer_size, MTLResourceOptions::StorageModeShared)
        .expect("Failed to allocate buffer B");
    let buffer_out = device.newBufferWithLength_options(buffer_size, MTLResourceOptions::StorageModeShared)
        .expect("Failed to allocate buffer OUT");
    println!("Allocated inputs and output buffers of size: {} bytes", buffer_size);

    // Initialize input data
    let ptr_a = buffer_a.contents().as_ptr() as *mut f32;
    let ptr_b = buffer_b.contents().as_ptr() as *mut f32;
    unsafe {
        let slice_a = std::slice::from_raw_parts_mut(ptr_a, NUM_ELEMENTS);
        let slice_b = std::slice::from_raw_parts_mut(ptr_b, NUM_ELEMENTS);
        for i in 0..NUM_ELEMENTS {
            slice_a[i] = i as f32;
            slice_b[i] = (i * 2) as f32;
        }
    }
    println!("Initialized input buffers");

    // Command buffer and encoding
    let command_buffer = queue.commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    let threads_per_threadgroup = MTLSize {
        width: 64,
        height: 1,
        depth: 1,
    };
    let threadgroups_per_grid = MTLSize {
        width: (NUM_ELEMENTS + 63) / 64,
        height: 1,
        depth: 1,
    };

    unsafe {
        encoder.setComputePipelineState(&pipeline_state);
        encoder.setBuffer_offset_atIndex(Some(&buffer_a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&buffer_b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&buffer_out), 0, 2);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups_per_grid, threads_per_threadgroup);
        encoder.endEncoding();
    }
    println!("Encoded commands and finished encoding");

    // Commit and wait for execution
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    println!("Submitted command buffer and waited for execution");

    // Read and verify results
    let ptr_out = buffer_out.contents().as_ptr() as *const f32;
    unsafe {
        let slice_out = std::slice::from_raw_parts(ptr_out, NUM_ELEMENTS);
        for i in 0..NUM_ELEMENTS {
            let expected = (i + i * 2) as f32;
            assert_eq!(slice_out[i], expected, "Mismatch at index {}: got {}, expected {}", i, slice_out[i], expected);
        }
    }

    println!("Verification SUCCESSFUL! All {} element additions computed correctly on GPU using Metal!", NUM_ELEMENTS);
}
