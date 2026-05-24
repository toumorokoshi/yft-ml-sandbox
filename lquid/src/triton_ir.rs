use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TritonType {
    I32,
    F32,
    I1,
    Ptr(Box<TritonType>),
    Tensor(Vec<usize>, Box<TritonType>),
}

impl fmt::Display for TritonType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TritonType::I32 => write!(f, "i32"),
            TritonType::F32 => write!(f, "f32"),
            TritonType::I1 => write!(f, "i1"),
            TritonType::Ptr(t) => write!(f, "!tt.ptr<{}>", t),
            TritonType::Tensor(shape, t) => {
                let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
                write!(f, "tensor<{}x{}>", shape_str.join("x"), t)
            }
        }
    }
}

/// Represents an instruction in the Triton IR Dialect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TritonInstruction {
    /// Defines a constant value (integer, float, or dense tensor constant).
    Constant { dest: String, value: String, ty: TritonType },
    /// Retrieves the program ID along a specific grid axis ('x', 'y', or 'z').
    GetProgramId { dest: String, axis: char, ty: TritonType },
    /// Computes the integer addition of two operands.
    AddI { dest: String, lhs: String, rhs: String, ty: TritonType },
    /// Computes the floating-point addition of two operands.
    AddF { dest: String, lhs: String, rhs: String, ty: TritonType },
    /// Computes the integer subtraction of two operands.
    SubI { dest: String, lhs: String, rhs: String, ty: TritonType },
    /// Computes the floating-point subtraction of two operands.
    SubF { dest: String, lhs: String, rhs: String, ty: TritonType },
    /// Computes the integer multiplication of two operands.
    MulI { dest: String, lhs: String, rhs: String, ty: TritonType },
    /// Computes the floating-point multiplication of two operands.
    MulF { dest: String, lhs: String, rhs: String, ty: TritonType },
    /// Computes the signed integer division of two operands.
    DivSI { dest: String, lhs: String, rhs: String, ty: TritonType },
    /// Computes the floating-point division of two operands.
    DivF { dest: String, lhs: String, rhs: String, ty: TritonType },
    /// Computes the signed integer remainder of two operands.
    RemSI { dest: String, lhs: String, rhs: String, ty: TritonType },
    /// Computes the signed integer minimum of two operands.
    MinSI { dest: String, lhs: String, rhs: String, ty: TritonType },
    /// Performs a logical/bitwise AND of two operands.
    And { dest: String, lhs: String, rhs: String, ty: TritonType },
    /// Performs an integer comparison based on a predicate (e.g. "slt", "eq").
    Cmpi { dest: String, predicate: String, lhs: String, rhs: String, ty: TritonType },
    /// Creates a 1D tensor representing a range of integers from start (inclusive) to end (exclusive).
    MakeRange { dest: String, start: i32, end: i32, ty: TritonType },
    /// Splats a scalar value into a tensor of the specified destination type.
    Splat { dest: String, src: String, src_ty: TritonType, dest_ty: TritonType },
    /// Expands the dimensions of a tensor by inserting a new unit dimension at the specified axis.
    ExpandDims { dest: String, src: String, axis: i32, src_ty: TritonType, dest_ty: TritonType },
    /// Broadcasts a tensor to a compatible destination type of a larger shape.
    Broadcast { dest: String, src: String, src_ty: TritonType, dest_ty: TritonType },
    /// Adds an offset to a pointer or a tensor of pointers.
    AddPtr { dest: String, ptr: String, offset: String, ptr_ty: TritonType, offset_ty: TritonType },
    /// Loads a value or tensor of values from memory, with an optional mask and other padding.
    Load { dest: String, ptr: String, mask: Option<String>, other: Option<String>, ty: TritonType },
    /// Stores a value or tensor of values into memory under an optional mask.
    Store { ptr: String, value: String, mask: Option<String>, ty: TritonType },
    /// Performs a matrix multiplication of a and b, adding it to the accumulator.
    Dot { dest: String, a: String, b: String, accumulator: String, a_ty: TritonType, b_ty: TritonType, dest_ty: TritonType },
    /// Represents a structured loop (scf.for) with loop variable, bounds, step, and loop-carried variables.
    For {
        loop_var: String,
        start: String,
        end: String,
        step: String,
        iter_args: Vec<(String, String, TritonType)>, // (arg_name, init_val, type)
        dest_prefix: String,
        body: Vec<TritonInstruction>,
        yield_vals: Vec<String>,
        yield_types: Vec<TritonType>,
    },
    /// Returns control back from the function.
    Return,
}

pub struct TritonFunction {
    pub name: String,
    pub args: Vec<(String, TritonType)>,
    pub instructions: Vec<TritonInstruction>,
}

pub struct TritonModule {
    pub functions: Vec<TritonFunction>,
}

fn indent(level: usize) -> String {
    "  ".repeat(level)
}

impl TritonInstruction {
    pub fn format(&self, level: usize) -> String {
        let ind = indent(level);
        match self {
            TritonInstruction::Constant { dest, value, ty } => {
                match ty {
                    TritonType::Tensor(..) => {
                        format!("{}%{} = arith.constant dense<{}> : {}\n", ind, dest, value, ty)
                    }
                    _ => {
                        format!("{}%{} = arith.constant {} : {}\n", ind, dest, value, ty)
                    }
                }
            }
            TritonInstruction::GetProgramId { dest, axis, ty } => {
                format!("{}%{} = tt.get_program_id {} : {}\n", ind, dest, axis, ty)
            }
            TritonInstruction::AddI { dest, lhs, rhs, ty } => {
                format!("{}%{} = arith.addi %{}, %{} : {}\n", ind, dest, lhs, rhs, ty)
            }
            TritonInstruction::AddF { dest, lhs, rhs, ty } => {
                format!("{}%{} = arith.addf %{}, %{} : {}\n", ind, dest, lhs, rhs, ty)
            }
            TritonInstruction::SubI { dest, lhs, rhs, ty } => {
                format!("{}%{} = arith.subi %{}, %{} : {}\n", ind, dest, lhs, rhs, ty)
            }
            TritonInstruction::SubF { dest, lhs, rhs, ty } => {
                format!("{}%{} = arith.subf %{}, %{} : {}\n", ind, dest, lhs, rhs, ty)
            }
            TritonInstruction::MulI { dest, lhs, rhs, ty } => {
                format!("{}%{} = arith.muli %{}, %{} : {}\n", ind, dest, lhs, rhs, ty)
            }
            TritonInstruction::MulF { dest, lhs, rhs, ty } => {
                format!("{}%{} = arith.mulf %{}, %{} : {}\n", ind, dest, lhs, rhs, ty)
            }
            TritonInstruction::DivSI { dest, lhs, rhs, ty } => {
                format!("{}%{} = arith.divsi %{}, %{} : {}\n", ind, dest, lhs, rhs, ty)
            }
            TritonInstruction::DivF { dest, lhs, rhs, ty } => {
                format!("{}%{} = arith.divf %{}, %{} : {}\n", ind, dest, lhs, rhs, ty)
            }
            TritonInstruction::RemSI { dest, lhs, rhs, ty } => {
                format!("{}%{} = arith.remsi %{}, %{} : {}\n", ind, dest, lhs, rhs, ty)
            }
            TritonInstruction::MinSI { dest, lhs, rhs, ty } => {
                format!("{}%{} = arith.minsi %{}, %{} : {}\n", ind, dest, lhs, rhs, ty)
            }
            TritonInstruction::And { dest, lhs, rhs, ty } => {
                format!("{}%{} = arith.andi %{}, %{} : {}\n", ind, dest, lhs, rhs, ty)
            }
            TritonInstruction::Cmpi { dest, predicate, lhs, rhs, ty } => {
                format!("{}%{} = arith.cmpi {}, %{}, %{} : {}\n", ind, dest, predicate, lhs, rhs, ty)
            }
            TritonInstruction::MakeRange { dest, start, end, ty } => {
                format!("{}%{} = tt.make_range {{end = {} : i32, start = {} : i32}} : {}\n", ind, dest, end, start, ty)
            }
            TritonInstruction::Splat { dest, src, src_ty, dest_ty } => {
                format!("{}%{} = tt.splat %{} : {} -> {}\n", ind, dest, src, src_ty, dest_ty)
            }
            TritonInstruction::ExpandDims { dest, src, axis, src_ty, dest_ty } => {
                format!("{}%{} = tt.expand_dims %{} {{axis = {} : i32}} : {} -> {}\n", ind, dest, src, axis, src_ty, dest_ty)
            }
            TritonInstruction::Broadcast { dest, src, src_ty, dest_ty } => {
                format!("{}%{} = tt.broadcast %{} : {} -> {}\n", ind, dest, src, src_ty, dest_ty)
            }
            TritonInstruction::AddPtr { dest, ptr, offset, ptr_ty, offset_ty } => {
                format!("{}%{} = tt.addptr %{}, %{} : {}, {}\n", ind, dest, ptr, offset, ptr_ty, offset_ty)
            }
            TritonInstruction::Load { dest, ptr, mask, other, ty } => {
                match (mask, other) {
                    (Some(m), Some(o)) => {
                        format!("{}%{} = tt.load %{}, %{}, %{} : {}\n", ind, dest, ptr, m, o, ty)
                    }
                    _ => {
                        format!("{}%{} = tt.load %{} : {}\n", ind, dest, ptr, ty)
                    }
                }
            }
            TritonInstruction::Store { ptr, value, mask, ty } => {
                match mask {
                    Some(m) => {
                        format!("{}tt.store %{}, %{}, %{} : {}\n", ind, ptr, value, m, ty)
                    }
                    None => {
                        format!("{}tt.store %{}, %{} : {}\n", ind, ptr, value, ty)
                    }
                }
            }
            TritonInstruction::Dot { dest, a, b, accumulator, a_ty, b_ty, dest_ty } => {
                format!("{}%{} = tt.dot %{}, %{}, %{} : {} * {} -> {}\n", ind, dest, a, b, accumulator, a_ty, b_ty, dest_ty)
            }
            TritonInstruction::For {
                loop_var,
                start,
                end,
                step,
                iter_args,
                dest_prefix,
                body,
                yield_vals,
                yield_types,
            } => {
                let mut out = String::new();
                if !yield_types.is_empty() {
                    if yield_types.len() == 1 {
                        out.push_str(&format!("{}%{} = ", ind, dest_prefix));
                    } else {
                        out.push_str(&format!("{}%{}:{} = ", ind, dest_prefix, yield_types.len()));
                    }
                } else {
                    out.push_str(&ind);
                }

                out.push_str(&format!("scf.for %{} = %{} to %{} step %{} iter_args(", loop_var, start, end, step));
                let args_str: Vec<String> = iter_args.iter()
                    .map(|(name, val, _ty)| format!("%{} = %{}", name, val))
                    .collect();

                out.push_str(&args_str.join(", "));
                out.push_str(") -> (");
                let types_str: Vec<String> = yield_types.iter().map(|ty| ty.to_string()).collect();
                out.push_str(&types_str.join(", "));
                out.push_str(")  : i32 {\n");

                for inst in body {
                    out.push_str(&inst.format(level + 1));
                }

                out.push_str(&format!("{}scf.yield ", indent(level + 1)));
                let yield_str: Vec<String> = yield_vals.iter().map(|v| format!("%{}", v)).collect();
                out.push_str(&yield_str.join(", "));
                out.push_str(" : ");
                out.push_str(&types_str.join(", "));
                out.push_str("\n");

                out.push_str(&format!("{}}}\n", ind));
                out
            }
            TritonInstruction::Return => {
                format!("{}tt.return\n", ind)
            }
        }
    }
}

pub fn format_module(module: &TritonModule) -> String {
    let mut out = String::new();
    out.push_str("module {\n");
    for func in &module.functions {
        out.push_str(&format!("  tt.func public @{}(", func.name));
        let args_str: Vec<String> = func.args.iter()
            .map(|(name, ty)| format!("%{}: {}", name, ty))
            .collect();
        out.push_str(&args_str.join(", "));
        out.push_str(") attributes {noinline = false} {\n");

        for inst in &func.instructions {
            out.push_str(&inst.format(2));
        }

        out.push_str("  }\n");
    }
    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_types() {
        assert_eq!(TritonType::I32.to_string(), "i32");
        assert_eq!(TritonType::F32.to_string(), "f32");
        assert_eq!(
            TritonType::Ptr(Box::new(TritonType::F32)).to_string(),
            "!tt.ptr<f32>"
        );
        assert_eq!(
            TritonType::Tensor(vec![128, 32], Box::new(TritonType::F32)).to_string(),
            "tensor<128x32xf32>"
        );
    }

    #[test]
    fn test_format_constant() {
        let inst = TritonInstruction::Constant {
            dest: "c1024_i32".to_string(),
            value: "1024".to_string(),
            ty: TritonType::I32,
        };
        assert_eq!(
            inst.format(1),
            "  %c1024_i32 = arith.constant 1024 : i32\n"
        );
    }
}
