use crate::triton_ir::{TritonFunction, TritonInstruction, TritonModule, TritonType};
use onnx_protobuf::{ModelProto, GraphProto, NodeProto, AttributeProto};

#[derive(Debug, Clone, PartialEq)]
pub struct GemmParams {
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub trans_a: bool,
    pub trans_b: bool,
    pub alpha: f32,
    pub beta: f32,
    pub has_bias: bool,
}

/// Helper function to retrieve the shape of a named tensor from the ONNX graph.
/// Follows Rule 1: pure functional data transformation.
fn get_tensor_shape(graph: &GraphProto, name: &str) -> Option<Vec<i64>> {
    // 1. Search initializers
    for init in &graph.initializer {
        if init.name == name {
            return Some(init.dims.clone());
        }
    }

    // 2. Search inputs
    for input in &graph.input {
        if input.name == name {
            if let Some(type_proto) = input.type_.as_ref() {
                if type_proto.has_tensor_type() {
                    let tensor_type = type_proto.tensor_type();
                    if let Some(shape_proto) = tensor_type.shape.as_ref() {
                        let dims: Vec<i64> = shape_proto.dim.iter().map(|d| d.dim_value()).collect();
                        return Some(dims);
                    }
                }
            }
        }
    }

    // 3. Search value_info
    for vi in &graph.value_info {
        if vi.name == name {
            if let Some(type_proto) = vi.type_.as_ref() {
                if type_proto.has_tensor_type() {
                    let tensor_type = type_proto.tensor_type();
                    if let Some(shape_proto) = tensor_type.shape.as_ref() {
                        let dims: Vec<i64> = shape_proto.dim.iter().map(|d| d.dim_value()).collect();
                        return Some(dims);
                    }
                }
            }
        }
    }

    None
}

/// Helper to get an attribute by name.
fn get_attribute<'a>(node: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    node.attribute.iter().find(|attr| attr.name == name)
}

/// Extracts GemmParams from the first Gemm node in the ModelProto.
/// Follows Rule 1: pure functional logic.
pub fn extract_gemm_params(model: &ModelProto) -> Result<GemmParams, String> {
    let graph = model.graph.as_ref().ok_or("Model has no graph")?;
    
    let gemm_node = graph.node.iter()
        .find(|node| node.op_type == "Gemm")
        .ok_or("No Gemm node found in the graph")?;

    if gemm_node.input.len() < 2 {
        return Err("Gemm node must have at least 2 inputs".to_string());
    }

    let input_a = &gemm_node.input[0];
    let input_b = &gemm_node.input[1];
    let has_bias = gemm_node.input.len() >= 3 && !gemm_node.input[2].is_empty();

    let shape_a = get_tensor_shape(graph, input_a)
        .ok_or_else(|| format!("Could not find shape for input A: {}", input_a))?;
    let shape_b = get_tensor_shape(graph, input_b)
        .ok_or_else(|| format!("Could not find shape for input B: {}", input_b))?;

    if shape_a.len() != 2 || shape_b.len() != 2 {
        return Err("Gemm inputs A and B must be 2D matrices".to_string());
    }

    let trans_a = get_attribute(gemm_node, "transA")
        .map(|attr| attr.i != 0)
        .unwrap_or(false);

    let trans_b = get_attribute(gemm_node, "transB")
        .map(|attr| attr.i != 0)
        .unwrap_or(false);

    let alpha = get_attribute(gemm_node, "alpha")
        .map(|attr| attr.f)
        .unwrap_or(1.0);

    let beta = get_attribute(gemm_node, "beta")
        .map(|attr| attr.f)
        .unwrap_or(1.0);

    let m = if trans_a { shape_a[1] } else { shape_a[0] };
    let k = if trans_a { shape_a[0] } else { shape_a[1] };
    let n = if trans_b { shape_b[0] } else { shape_b[1] };

    Ok(GemmParams {
        m,
        n,
        k,
        trans_a,
        trans_b,
        alpha,
        beta,
        has_bias,
    })
}

/// Generates a TritonModule representing the GEMM kernel.
/// Follows Rule 1: pure functional data transformation.
pub fn generate_gemm_module(params: &GemmParams) -> TritonModule {
    let mut args = vec![
        ("a_ptr".to_string(), TritonType::Ptr(Box::new(TritonType::F32))),
        ("b_ptr".to_string(), TritonType::Ptr(Box::new(TritonType::F32))),
        ("c_ptr".to_string(), TritonType::Ptr(Box::new(TritonType::F32))),
    ];

    if params.has_bias {
        args.push(("bias_ptr".to_string(), TritonType::Ptr(Box::new(TritonType::F32))));
    }

    args.extend(vec![
        ("M".to_string(), TritonType::I32),
        ("N".to_string(), TritonType::I32),
        ("K".to_string(), TritonType::I32),
        ("stride_am".to_string(), TritonType::I32),
        ("stride_ak".to_string(), TritonType::I32),
        ("stride_bk".to_string(), TritonType::I32),
        ("stride_bn".to_string(), TritonType::I32),
        ("stride_cm".to_string(), TritonType::I32),
        ("stride_cn".to_string(), TritonType::I32),
    ]);

    let mut insts = Vec::new();

    // Constant definitions:
    insts.push(TritonInstruction::Constant { dest: "c31_i32".to_string(), value: "31".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::Constant { dest: "c127_i32".to_string(), value: "127".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::Constant { dest: "c1_i32".to_string(), value: "1".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::Constant { dest: "c0_i32".to_string(), value: "0".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::Constant { dest: "cst".to_string(), value: "0.000000e+00".to_string(), ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::F32)) });
    insts.push(TritonInstruction::Constant { dest: "cst_0".to_string(), value: "0.000000e+00".to_string(), ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::F32)) });
    insts.push(TritonInstruction::Constant { dest: "cst_1".to_string(), value: "0.000000e+00".to_string(), ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::F32)) });
    insts.push(TritonInstruction::Constant { dest: "c32_i32".to_string(), value: "32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::Constant { dest: "c128_i32".to_string(), value: "128".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::Constant { dest: "c8_i32".to_string(), value: "8".to_string(), ty: TritonType::I32 });

    // Program ID and group routing:
    insts.push(TritonInstruction::GetProgramId { dest: "pid".to_string(), axis: 'x', ty: TritonType::I32 });
    insts.push(TritonInstruction::AddI { dest: "num_pid_m".to_string(), lhs: "M".to_string(), rhs: "c127_i32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::DivSI { dest: "num_pid_m_2".to_string(), lhs: "num_pid_m".to_string(), rhs: "c128_i32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::AddI { dest: "num_pid_n".to_string(), lhs: "N".to_string(), rhs: "c127_i32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::DivSI { dest: "num_pid_n_3".to_string(), lhs: "num_pid_n".to_string(), rhs: "c128_i32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::MulI { dest: "num_pid_in_group".to_string(), lhs: "num_pid_n_3".to_string(), rhs: "c8_i32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::DivSI { dest: "group_id".to_string(), lhs: "pid".to_string(), rhs: "num_pid_in_group".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::MulI { dest: "first_pid_m".to_string(), lhs: "group_id".to_string(), rhs: "c8_i32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::SubI { dest: "group_size_m".to_string(), lhs: "num_pid_m_2".to_string(), rhs: "first_pid_m".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::MinSI { dest: "group_size_m_4".to_string(), lhs: "group_size_m".to_string(), rhs: "c8_i32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::RemSI { dest: "pid_m".to_string(), lhs: "pid".to_string(), rhs: "group_size_m_4".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::AddI { dest: "pid_m_5".to_string(), lhs: "first_pid_m".to_string(), rhs: "pid_m".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::RemSI { dest: "pid_n".to_string(), lhs: "pid".to_string(), rhs: "num_pid_in_group".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::DivSI { dest: "pid_n_6".to_string(), lhs: "pid_n".to_string(), rhs: "group_size_m_4".to_string(), ty: TritonType::I32 });

    // Tile offsets:
    insts.push(TritonInstruction::MulI { dest: "offs_am".to_string(), lhs: "pid_m_5".to_string(), rhs: "c128_i32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::MakeRange { dest: "offs_am_7".to_string(), start: 0, end: 128, ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Splat { dest: "offs_am_8".to_string(), src: "offs_am".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::AddI { dest: "offs_am_9".to_string(), lhs: "offs_am_8".to_string(), rhs: "offs_am_7".to_string(), ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Splat { dest: "offs_am_10".to_string(), src: "M".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::RemSI { dest: "offs_am_11".to_string(), lhs: "offs_am_9".to_string(), rhs: "offs_am_10".to_string(), ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::MulI { dest: "offs_bn".to_string(), lhs: "pid_n_6".to_string(), rhs: "c128_i32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::Splat { dest: "offs_bn_12".to_string(), src: "offs_bn".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::AddI { dest: "offs_bn_13".to_string(), lhs: "offs_bn_12".to_string(), rhs: "offs_am_7".to_string(), ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Splat { dest: "offs_bn_14".to_string(), src: "N".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::RemSI { dest: "offs_bn_15".to_string(), lhs: "offs_bn_13".to_string(), rhs: "offs_bn_14".to_string(), ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::MakeRange { dest: "offs_k".to_string(), start: 0, end: 32, ty: TritonType::Tensor(vec![32], Box::new(TritonType::I32)) });

    // A pointer calculations:
    insts.push(TritonInstruction::ExpandDims { dest: "a_ptrs".to_string(), src: "offs_am_11".to_string(), axis: 1, src_ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)), dest_ty: TritonType::Tensor(vec![128, 1], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Splat { dest: "a_ptrs_16".to_string(), src: "stride_am".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![128, 1], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::MulI { dest: "a_ptrs_17".to_string(), lhs: "a_ptrs".to_string(), rhs: "a_ptrs_16".to_string(), ty: TritonType::Tensor(vec![128, 1], Box::new(TritonType::I32)) });
    
    insts.push(TritonInstruction::ExpandDims { dest: "a_ptrs_18".to_string(), src: "offs_k".to_string(), axis: 0, src_ty: TritonType::Tensor(vec![32], Box::new(TritonType::I32)), dest_ty: TritonType::Tensor(vec![1, 32], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Splat { dest: "a_ptrs_19".to_string(), src: "stride_ak".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![1, 32], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::MulI { dest: "a_ptrs_20".to_string(), lhs: "a_ptrs_18".to_string(), rhs: "a_ptrs_19".to_string(), ty: TritonType::Tensor(vec![1, 32], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::Broadcast { dest: "a_ptrs_21".to_string(), src: "a_ptrs_17".to_string(), src_ty: TritonType::Tensor(vec![128, 1], Box::new(TritonType::I32)), dest_ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Broadcast { dest: "a_ptrs_22".to_string(), src: "a_ptrs_20".to_string(), src_ty: TritonType::Tensor(vec![1, 32], Box::new(TritonType::I32)), dest_ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::AddI { dest: "a_ptrs_23".to_string(), lhs: "a_ptrs_21".to_string(), rhs: "a_ptrs_22".to_string(), ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::Splat { dest: "a_ptrs_24".to_string(), src: "a_ptr".to_string(), src_ty: TritonType::Ptr(Box::new(TritonType::F32)), dest_ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))) });
    insts.push(TritonInstruction::AddPtr { dest: "a_ptrs_25".to_string(), ptr: "a_ptrs_24".to_string(), offset: "a_ptrs_23".to_string(), ptr_ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))), offset_ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::I32)) });

    // B pointer calculations:
    insts.push(TritonInstruction::ExpandDims { dest: "b_ptrs".to_string(), src: "offs_k".to_string(), axis: 1, src_ty: TritonType::Tensor(vec![32], Box::new(TritonType::I32)), dest_ty: TritonType::Tensor(vec![32, 1], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Splat { dest: "b_ptrs_26".to_string(), src: "stride_bk".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![32, 1], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::MulI { dest: "b_ptrs_27".to_string(), lhs: "b_ptrs".to_string(), rhs: "b_ptrs_26".to_string(), ty: TritonType::Tensor(vec![32, 1], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::ExpandDims { dest: "b_ptrs_28".to_string(), src: "offs_bn_15".to_string(), axis: 0, src_ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)), dest_ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Splat { dest: "b_ptrs_29".to_string(), src: "stride_bn".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::MulI { dest: "b_ptrs_30".to_string(), lhs: "b_ptrs_28".to_string(), rhs: "b_ptrs_29".to_string(), ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::Broadcast { dest: "b_ptrs_31".to_string(), src: "b_ptrs_27".to_string(), src_ty: TritonType::Tensor(vec![32, 1], Box::new(TritonType::I32)), dest_ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Broadcast { dest: "b_ptrs_32".to_string(), src: "b_ptrs_30".to_string(), src_ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::I32)), dest_ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::AddI { dest: "b_ptrs_33".to_string(), lhs: "b_ptrs_31".to_string(), rhs: "b_ptrs_32".to_string(), ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::Splat { dest: "b_ptrs_34".to_string(), src: "b_ptr".to_string(), src_ty: TritonType::Ptr(Box::new(TritonType::F32)), dest_ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))) });
    insts.push(TritonInstruction::AddPtr { dest: "b_ptrs_35".to_string(), ptr: "b_ptrs_34".to_string(), offset: "b_ptrs_33".to_string(), ptr_ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))), offset_ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::I32)) });

    // Loop bounds and increments:
    insts.push(TritonInstruction::AddI { dest: "loop_bound_0".to_string(), lhs: "K".to_string(), rhs: "c31_i32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::DivSI { dest: "loop_bound_1".to_string(), lhs: "loop_bound_0".to_string(), rhs: "c32_i32".to_string(), ty: TritonType::I32 });

    insts.push(TritonInstruction::MulI { dest: "a_ptrs_36".to_string(), lhs: "stride_ak".to_string(), rhs: "c32_i32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::Splat { dest: "a_ptrs_37".to_string(), src: "a_ptrs_36".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::MulI { dest: "b_ptrs_38".to_string(), lhs: "stride_bk".to_string(), rhs: "c32_i32".to_string(), ty: TritonType::I32 });
    insts.push(TritonInstruction::Splat { dest: "b_ptrs_39".to_string(), src: "b_ptrs_38".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::I32)) });

    // The scf.for loop:
    let loop_body = vec![
        TritonInstruction::MulI { dest: "k_offset".to_string(), lhs: "k".to_string(), rhs: "c32_i32".to_string(), ty: TritonType::I32 },
        TritonInstruction::SubI { dest: "k_remaining".to_string(), lhs: "K".to_string(), rhs: "k_offset".to_string(), ty: TritonType::I32 },
        
        TritonInstruction::Splat { dest: "k_rem_splat_a".to_string(), src: "k_remaining".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![1, 32], Box::new(TritonType::I32)) },
        TritonInstruction::Cmpi { dest: "a_mask_cmp".to_string(), predicate: "slt".to_string(), lhs: "a_ptrs_18".to_string(), rhs: "k_rem_splat_a".to_string(), ty: TritonType::Tensor(vec![1, 32], Box::new(TritonType::I32)) },
        TritonInstruction::Broadcast { dest: "a_mask_2d".to_string(), src: "a_mask_cmp".to_string(), src_ty: TritonType::Tensor(vec![1, 32], Box::new(TritonType::I1)), dest_ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::I1)) },
        TritonInstruction::Load { dest: "a_tile".to_string(), ptr: "a_ptrs_loop".to_string(), mask: Some("a_mask_2d".to_string()), other: Some("cst_1".to_string()), ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))) },

        TritonInstruction::Splat { dest: "k_rem_splat_b".to_string(), src: "k_remaining".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![32, 1], Box::new(TritonType::I32)) },
        TritonInstruction::Cmpi { dest: "b_mask_cmp".to_string(), predicate: "slt".to_string(), lhs: "b_ptrs".to_string(), rhs: "k_rem_splat_b".to_string(), ty: TritonType::Tensor(vec![32, 1], Box::new(TritonType::I32)) },
        TritonInstruction::Broadcast { dest: "b_mask_2d".to_string(), src: "b_mask_cmp".to_string(), src_ty: TritonType::Tensor(vec![32, 1], Box::new(TritonType::I1)), dest_ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::I1)) },
        TritonInstruction::Load { dest: "b_tile".to_string(), ptr: "b_ptrs_loop".to_string(), mask: Some("b_mask_2d".to_string()), other: Some("cst_0".to_string()), ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))) },

        TritonInstruction::Dot { dest: "acc_next".to_string(), a: "a_tile".to_string(), b: "b_tile".to_string(), accumulator: "acc_loop".to_string(), a_ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::F32)), b_ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::F32)), dest_ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::F32)) },
        
        TritonInstruction::AddPtr { dest: "a_ptrs_next".to_string(), ptr: "a_ptrs_loop".to_string(), offset: "a_ptrs_37".to_string(), ptr_ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))), offset_ty: TritonType::Tensor(vec![128, 32], Box::new(TritonType::I32)) },
        TritonInstruction::AddPtr { dest: "b_ptrs_next".to_string(), ptr: "b_ptrs_loop".to_string(), offset: "b_ptrs_39".to_string(), ptr_ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))), offset_ty: TritonType::Tensor(vec![32, 128], Box::new(TritonType::I32)) },
    ];

    let loop_iter_args = vec![
        ("a_ptrs_loop".to_string(), "a_ptrs_25".to_string(), TritonType::Tensor(vec![128, 32], Box::new(TritonType::Ptr(Box::new(TritonType::F32))))),
        ("b_ptrs_loop".to_string(), "b_ptrs_35".to_string(), TritonType::Tensor(vec![32, 128], Box::new(TritonType::Ptr(Box::new(TritonType::F32))))),
        ("acc_loop".to_string(), "cst".to_string(), TritonType::Tensor(vec![128, 128], Box::new(TritonType::F32))),
    ];

    insts.push(TritonInstruction::For {
        loop_var: "k".to_string(),
        start: "c0_i32".to_string(),
        end: "loop_bound_1".to_string(),
        step: "c1_i32".to_string(),
        iter_args: loop_iter_args,
        dest_prefix: "accumulator".to_string(),
        body: loop_body,
        yield_vals: vec!["a_ptrs_next".to_string(), "b_ptrs_next".to_string(), "acc_next".to_string()],
        yield_types: vec![
            TritonType::Tensor(vec![128, 32], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))),
            TritonType::Tensor(vec![32, 128], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))),
            TritonType::Tensor(vec![128, 128], Box::new(TritonType::F32)),
        ],
    });

    let final_acc = if params.has_bias {
        // Load bias, broadcast, and add it
        insts.push(TritonInstruction::Splat {
            dest: "bias_ptrs".to_string(),
            src: "bias_ptr".to_string(),
            src_ty: TritonType::Ptr(Box::new(TritonType::F32)),
            dest_ty: TritonType::Tensor(vec![128], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))),
        });
        insts.push(TritonInstruction::AddPtr {
            dest: "bias_ptrs_add".to_string(),
            ptr: "bias_ptrs".to_string(),
            offset: "offs_bn_15".to_string(),
            ptr_ty: TritonType::Tensor(vec![128], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))),
            offset_ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)),
        });
        insts.push(TritonInstruction::Splat {
            dest: "c_mask_n_splat".to_string(),
            src: "N".to_string(),
            src_ty: TritonType::I32,
            dest_ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)),
        });
        insts.push(TritonInstruction::Cmpi {
            dest: "bias_mask".to_string(),
            predicate: "slt".to_string(),
            lhs: "offs_bn_13".to_string(),
            rhs: "c_mask_n_splat".to_string(),
            ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)),
        });
        insts.push(TritonInstruction::Constant {
            dest: "bias_cst_zero".to_string(),
            value: "0.000000e+00".to_string(),
            ty: TritonType::Tensor(vec![128], Box::new(TritonType::F32)),
        });
        insts.push(TritonInstruction::Load {
            dest: "bias_val".to_string(),
            ptr: "bias_ptrs_add".to_string(),
            mask: Some("bias_mask".to_string()),
            other: Some("bias_cst_zero".to_string()),
            ty: TritonType::Tensor(vec![128], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))),
        });
        insts.push(TritonInstruction::ExpandDims {
            dest: "bias_2d".to_string(),
            src: "bias_val".to_string(),
            axis: 0,
            src_ty: TritonType::Tensor(vec![128], Box::new(TritonType::F32)),
            dest_ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::F32)),
        });
        insts.push(TritonInstruction::Broadcast {
            dest: "bias_broadcast".to_string(),
            src: "bias_2d".to_string(),
            src_ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::F32)),
            dest_ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::F32)),
        });
        insts.push(TritonInstruction::AddF {
            dest: "acc_with_bias".to_string(),
            lhs: "accumulator#2".to_string(),
            rhs: "bias_broadcast".to_string(),
            ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::F32)),
        });
        "acc_with_bias".to_string()
    } else {
        "accumulator#2".to_string()
    };

    // C pointer calculations:
    insts.push(TritonInstruction::ExpandDims { dest: "c_ptrs".to_string(), src: "offs_am_9".to_string(), axis: 1, src_ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)), dest_ty: TritonType::Tensor(vec![128, 1], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Splat { dest: "c_ptrs_40".to_string(), src: "stride_cm".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![128, 1], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::MulI { dest: "c_ptrs_41".to_string(), lhs: "c_ptrs".to_string(), rhs: "c_ptrs_40".to_string(), ty: TritonType::Tensor(vec![128, 1], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::ExpandDims { dest: "c_ptrs_42".to_string(), src: "offs_bn_13".to_string(), axis: 0, src_ty: TritonType::Tensor(vec![128], Box::new(TritonType::I32)), dest_ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Splat { dest: "c_ptrs_43".to_string(), src: "stride_cn".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::MulI { dest: "c_ptrs_44".to_string(), lhs: "c_ptrs_42".to_string(), rhs: "c_ptrs_43".to_string(), ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::Broadcast { dest: "c_ptrs_45".to_string(), src: "c_ptrs_41".to_string(), src_ty: TritonType::Tensor(vec![128, 1], Box::new(TritonType::I32)), dest_ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Broadcast { dest: "c_ptrs_46".to_string(), src: "c_ptrs_44".to_string(), src_ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::I32)), dest_ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::AddI { dest: "c_ptrs_47".to_string(), lhs: "c_ptrs_45".to_string(), rhs: "c_ptrs_46".to_string(), ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::Splat { dest: "c_ptrs_48".to_string(), src: "c_ptr".to_string(), src_ty: TritonType::Ptr(Box::new(TritonType::F32)), dest_ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))) });
    insts.push(TritonInstruction::AddPtr { dest: "c_ptrs_49".to_string(), ptr: "c_ptrs_48".to_string(), offset: "c_ptrs_47".to_string(), ptr_ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))), offset_ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::I32)) });

    // Store mask:
    insts.push(TritonInstruction::Splat { dest: "c_mask_m".to_string(), src: "M".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![128, 1], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Cmpi { dest: "c_mask_50".to_string(), predicate: "slt".to_string(), lhs: "c_ptrs".to_string(), rhs: "c_mask_m".to_string(), ty: TritonType::Tensor(vec![128, 1], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::Splat { dest: "c_mask_n".to_string(), src: "N".to_string(), src_ty: TritonType::I32, dest_ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::I32)) });
    insts.push(TritonInstruction::Cmpi { dest: "c_mask_52".to_string(), predicate: "slt".to_string(), lhs: "c_ptrs_42".to_string(), rhs: "c_mask_n".to_string(), ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::I32)) });

    insts.push(TritonInstruction::Broadcast { dest: "c_mask_53".to_string(), src: "c_mask_50".to_string(), src_ty: TritonType::Tensor(vec![128, 1], Box::new(TritonType::I1)), dest_ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::I1)) });
    insts.push(TritonInstruction::Broadcast { dest: "c_mask_54".to_string(), src: "c_mask_52".to_string(), src_ty: TritonType::Tensor(vec![1, 128], Box::new(TritonType::I1)), dest_ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::I1)) });
    insts.push(TritonInstruction::And { dest: "c_mask_55".to_string(), lhs: "c_mask_53".to_string(), rhs: "c_mask_54".to_string(), ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::I1)) });

    insts.push(TritonInstruction::Store { ptr: "c_ptrs_49".to_string(), value: final_acc, mask: Some("c_mask_55".to_string()), ty: TritonType::Tensor(vec![128, 128], Box::new(TritonType::Ptr(Box::new(TritonType::F32)))) });
    
    insts.push(TritonInstruction::Return);

    TritonModule {
        functions: vec![TritonFunction {
            name: "matmul_kernel".to_string(),
            args,
            instructions: insts,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_protobuf::{GraphProto, ModelProto, NodeProto, TypeProto, ValueInfoProto};
    use onnx_protobuf::type_proto::{Tensor, Value as TypeValue};
    use onnx_protobuf::tensor_shape_proto::{Dimension, dimension::Value as DimValue};
    use onnx_protobuf::TensorShapeProto;
    use protobuf::MessageField;

    fn make_dimension(val: i64) -> Dimension {
        Dimension {
            denotation: String::new(),
            value: Some(DimValue::DimValue(val)),
            special_fields: protobuf::SpecialFields::new(),
        }
    }

    fn make_tensor_type(dims: Vec<i64>) -> TypeProto {
        let mut t = Tensor::new();
        t.elem_type = 1; // FLOAT
        
        let mut shape = TensorShapeProto::new();
        shape.dim = dims.into_iter().map(make_dimension).collect();
        t.shape = MessageField::some(shape);

        TypeProto {
            denotation: String::new(),
            value: Some(TypeValue::TensorType(t)),
            special_fields: protobuf::SpecialFields::new(),
        }
    }

    #[test]
    fn test_extract_gemm_params_simple() {
        let mut model = ModelProto::new();
        let mut graph = GraphProto::new();

        // Add Gemm node
        let mut node = NodeProto::new();
        node.op_type = "Gemm".to_string();
        node.input = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        node.output = vec!["Y".to_string()];
        
        // transA = 0, transB = 1
        let mut attr_a = AttributeProto::new();
        attr_a.name = "transA".to_string();
        attr_a.i = 0;
        node.attribute.push(attr_a);

        let mut attr_b = AttributeProto::new();
        attr_b.name = "transB".to_string();
        attr_b.i = 1;
        node.attribute.push(attr_b);

        graph.node.push(node);

        // Add shapes to graph.input
        let mut input_a = ValueInfoProto::new();
        input_a.name = "A".to_string();
        input_a.type_ = MessageField::some(make_tensor_type(vec![16, 32])); // M=16, K=32
        graph.input.push(input_a);

        let mut input_b = ValueInfoProto::new();
        input_b.name = "B".to_string();
        // Since transB is 1, shape of B is [N, K], i.e. [64, 32] -> N=64, K=32
        input_b.type_ = MessageField::some(make_tensor_type(vec![64, 32]));
        graph.input.push(input_b);

        // Optional bias input C shape [64]
        let mut input_c = ValueInfoProto::new();
        input_c.name = "C".to_string();
        input_c.type_ = MessageField::some(make_tensor_type(vec![64]));
        graph.input.push(input_c);

        model.graph = MessageField::some(graph);

        let params = extract_gemm_params(&model).unwrap();
        assert_eq!(params.m, 16);
        assert_eq!(params.n, 64);
        assert_eq!(params.k, 32);
        assert_eq!(params.trans_a, false);
        assert_eq!(params.trans_b, true);
        assert_eq!(params.has_bias, true);
    }

    #[test]
    fn test_generate_gemm_module() {
        let params = GemmParams {
            m: 16,
            n: 64,
            k: 32,
            trans_a: false,
            trans_b: true,
            alpha: 1.0,
            beta: 1.0,
            has_bias: true,
        };
        let module = generate_gemm_module(&params);
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.functions[0].name, "matmul_kernel");
        
        // Assert some specific instructions are present
        let inst_names: Vec<&str> = module.functions[0].instructions.iter()
            .map(|inst| match inst {
                TritonInstruction::Constant { dest, .. } => dest.as_str(),
                TritonInstruction::GetProgramId { dest, .. } => dest.as_str(),
                TritonInstruction::Load { dest, .. } => dest.as_str(),
                TritonInstruction::Dot { dest, .. } => dest.as_str(),
                TritonInstruction::Store { .. } => "store",
                _ => "",
            })
            .filter(|n| !n.is_empty())
            .collect();
        
        assert!(inst_names.contains(&"cst"));
        assert!(inst_names.contains(&"pid"));
        assert!(inst_names.contains(&"store"));
    }
}
