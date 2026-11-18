use std::collections::HashMap;

use oxide_ml_core::{DenseTensor, Scalar, ValueId};
use oxide_ml_graph::{ConstantData, ConstantTensor, Graph, Op, OpAttrs, ValueKind};
use oxide_ml_kernel::KernelCall;

use crate::pass::{GraphPass, PassContext, PassResult};

pub struct ConstantFolding;

impl<T: Scalar> GraphPass<T> for ConstantFolding {
    fn name(&self) -> &'static str {
        "constant-folding"
    }

    fn run(
        &self,
        graph: &mut Graph,
        context: &PassContext<'_, T>,
    ) -> oxide_ml_core::Result<PassResult> {
        let mut constants: HashMap<ValueId, ConstantTensor> = HashMap::new();
        for node in graph.nodes() {
            if let Some(value) = node_constant(node.op.clone(), node.attrs.clone()) {
                for &output in &node.outputs {
                    constants.insert(output, value.clone());
                }
            }
        }

        let mut folded = 0usize;
        let mut skipped = Vec::new();
        let node_count = graph.nodes.len();
        for index in 0..node_count {
            let (op, attrs, inputs, outputs) = {
                let node = &graph.nodes[index];
                (
                    node.op.clone(),
                    node.attrs.clone(),
                    node.inputs.clone(),
                    node.outputs.clone(),
                )
            };
            if matches!(op, Op::Constant | Op::ConstantScalar) || outputs.len() != 1 {
                continue;
            }
            if !inputs.iter().all(|input| constants.contains_key(input)) {
                continue;
            }
            let output_shape = graph.value(outputs[0])?.shape.clone();
            let Some(static_shape) = output_shape.static_dims() else {
                skipped.push(format!(
                    "{}: symbolic output cannot be folded",
                    graph.nodes[index].id
                ));
                continue;
            };
            let mut tensors = Vec::with_capacity(inputs.len());
            let mut compatible = true;
            for input in &inputs {
                let constant = &constants[input];
                match (&constant.data, T::DTYPE) {
                    (ConstantData::F32(values), oxide_ml_core::DType::F32) => {
                        tensors.push(
                            DenseTensor::<T>::from_f64_vec(
                                constant.shape.static_dims().unwrap(),
                                values.iter().map(|value| *value as f64).collect(),
                            )
                            .expect("validated constant"),
                        );
                    }
                    (ConstantData::F64(values), oxide_ml_core::DType::F64) => {
                        tensors.push(
                            DenseTensor::<T>::from_f64_vec(
                                constant.shape.static_dims().unwrap(),
                                values.clone(),
                            )
                            .expect("validated constant"),
                        );
                    }
                    _ => {
                        compatible = false;
                        break;
                    }
                }
            }
            if !compatible {
                skipped.push(format!(
                    "{}: operand dtype differs from compiler scalar",
                    graph.nodes[index].id
                ));
                continue;
            }
            let output_spec = oxide_ml_kernel::TensorSpec::contiguous(
                static_shape.clone(),
                T::DTYPE,
                oxide_ml_core::Device::Cpu,
            );
            let input_specs = tensors
                .iter()
                .map(|tensor| {
                    oxide_ml_kernel::TensorSpec::contiguous(
                        tensor.dims(),
                        T::DTYPE,
                        oxide_ml_core::Device::Cpu,
                    )
                })
                .collect::<Vec<_>>();
            let call = KernelCall {
                op: &op,
                attrs: &attrs,
                inputs: &tensors,
                input_specs: &input_specs,
                output_specs: std::slice::from_ref(&output_spec),
                aliasing: &[],
                workspace: oxide_ml_kernel::WorkspaceSpec::none(),
                context: oxide_ml_kernel::ExecutionContext::default(),
            };
            let Ok(results) = context.backend.execute(call) else {
                skipped.push(format!(
                    "{}: backend could not evaluate constant expression",
                    graph.nodes[index].id
                ));
                continue;
            };
            let Some(result) = results.into_iter().next() else {
                skipped.push(format!(
                    "{}: backend returned no value",
                    graph.nodes[index].id
                ));
                continue;
            };
            let constant = match T::DTYPE {
                oxide_ml_core::DType::F32 => ConstantTensor {
                    shape: output_shape.clone(),
                    data: ConstantData::F32(
                        result
                            .data()
                            .iter()
                            .map(|value| value.to_f64() as f32)
                            .collect(),
                    ),
                },
                oxide_ml_core::DType::F64 => ConstantTensor {
                    shape: output_shape.clone(),
                    data: ConstantData::F64(
                        result.data().iter().map(|value| value.to_f64()).collect(),
                    ),
                },
                _ => {
                    skipped.push(format!(
                        "{}: unsupported scalar dtype",
                        graph.nodes[index].id
                    ));
                    continue;
                }
            };
            constant.validate()?;
            let node = &mut graph.nodes[index];
            node.op = Op::Constant;
            node.attrs = OpAttrs::Constant {
                value: constant.clone(),
            };
            node.inputs.clear();
            let value = graph.value_mut(outputs[0])?;
            value.kind = ValueKind::Constant;
            constants.insert(outputs[0], constant);
            folded += 1;
        }

        if folded == 0 {
            Ok(PassResult {
                changed: false,
                notes: skipped,
            })
        } else {
            let mut notes = vec![format!("folded {folded} node(s)")];
            notes.extend(skipped);
            Ok(PassResult {
                changed: true,
                notes,
            })
        }
    }
}

fn node_constant(op: Op, attrs: OpAttrs) -> Option<ConstantTensor> {
    match (op, attrs) {
        (Op::Constant, OpAttrs::Constant { value }) => Some(value),
        (Op::ConstantScalar, OpAttrs::ConstantScalar { value }) => {
            Some(ConstantTensor::f32(oxide_ml_core::Shape::scalar(), vec![value]).ok()?)
        }
        _ => None,
    }
}
