use std::collections::HashMap;
use std::sync::Arc;

use oxide_ml_core::{
    DType, DenseTensor, Device, Dim, Error, MemoryPlan, ParameterStore, Result, Scalar,
    SymbolBindings, ValueId,
};
use oxide_ml_graph::{Graph, GraphVerifier, OpAttrs};
use oxide_ml_kernel::{CustomKernelRegistry, KernelBackend, KernelCall, TensorSpec};

/// Input tensors keyed by graph value id.
#[derive(Clone, Debug, Default)]
pub struct ExecutionInputs<T: Scalar = f32> {
    values: HashMap<ValueId, DenseTensor<T>>,
}

impl<T: Scalar> ExecutionInputs<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: ValueId, tensor: DenseTensor<T>) {
        self.values.insert(id, tensor);
    }

    pub fn get(&self, id: ValueId) -> Option<&DenseTensor<T>> {
        self.values.get(&id)
    }

    /// Build inputs by graph input name.
    pub fn from_named(
        graph: &Graph,
        tensors: impl IntoIterator<Item = (impl AsRef<str>, DenseTensor<T>)>,
    ) -> Result<Self> {
        let mut inputs = Self::new();
        for (name, tensor) in tensors {
            let name = name.as_ref();
            let value = graph
                .value_by_name(name)
                .ok_or_else(|| Error::invalid_graph(format!("unknown input name {name:?}")))?;
            if !graph.inputs.contains(&value.id) {
                return Err(Error::invalid_graph(format!(
                    "{name:?} is not a graph input"
                )));
            }
            inputs.insert(value.id, tensor);
        }
        Ok(inputs)
    }
}

/// Materialized outputs and the full value environment of one execution.
#[derive(Clone, Debug)]
pub struct ExecutionResult<T: Scalar = f32> {
    values: Vec<Option<DenseTensor<T>>>,
    outputs: Vec<ValueId>,
}

impl<T: Scalar> ExecutionResult<T> {
    pub fn value(&self, id: ValueId) -> Result<&DenseTensor<T>> {
        self.values
            .get(id.0 as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::execution(format!("value {id} was not materialized")))
    }

    pub fn value_owned(&self, id: ValueId) -> Result<DenseTensor<T>> {
        self.value(id).cloned()
    }

    pub fn output(&self, index: usize) -> Result<&DenseTensor<T>> {
        let id = self
            .outputs
            .get(index)
            .copied()
            .ok_or_else(|| Error::execution(format!("output index {index} is out of range")))?;
        self.value(id)
    }

    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    pub fn values(&self) -> &[Option<DenseTensor<T>>] {
        &self.values
    }

    pub fn grad(&self, gradient: ValueId) -> Result<&DenseTensor<T>> {
        self.value(gradient)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionStats {
    pub buffer_allocations: usize,
    pub planned_peak_bytes: usize,
    pub materialized_values: usize,
    pub reused_buffers: usize,
}

/// Static graph executor.
///
/// The executor specializes symbolic shapes from concrete inputs, injects
/// parameter tensors, and dispatches nodes in SSA order. It is deliberately
/// not a tape: the same graph can be run for many batches without rebuilding.
pub struct Executor<T: Scalar = f32> {
    graph: Graph,
    backend: Box<dyn KernelBackend<T>>,
    custom: Option<Arc<CustomKernelRegistry<T>>>,
}

impl<T: Scalar> Executor<T> {
    pub fn new(graph: Graph) -> Result<Self> {
        GraphVerifier::verify(&graph)?;
        Ok(Executor {
            graph,
            backend: Box::new(crate::CpuBackend::<T>::new()),
            custom: None,
        })
    }

    pub fn with_backend(graph: Graph, backend: Box<dyn KernelBackend<T>>) -> Result<Self> {
        GraphVerifier::verify(&graph)?;
        Ok(Executor {
            graph,
            backend,
            custom: None,
        })
    }

    /// Attach a registry of user-defined kernels to the reference backend.
    pub fn with_custom(graph: Graph, registry: Arc<CustomKernelRegistry<T>>) -> Result<Self> {
        let mut executor = Self::new(graph)?;
        executor.custom = Some(registry);
        Ok(executor)
    }

    /// Attach both a backend and a custom-kernel registry.
    pub fn with_backend_and_custom(
        graph: Graph,
        backend: Box<dyn KernelBackend<T>>,
        registry: Arc<CustomKernelRegistry<T>>,
    ) -> Result<Self> {
        let mut executor = Self::with_backend(graph, backend)?;
        executor.custom = Some(registry);
        Ok(executor)
    }

    pub fn custom_registry(&self) -> Option<&Arc<CustomKernelRegistry<T>>> {
        self.custom.as_ref()
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Execute with a compiled memory plan. Only graph outputs are retained;
    /// intermediate values may share buffers and are intentionally not exposed.
    pub fn run_planned(
        &self,
        parameters: &ParameterStore<T>,
        inputs: &ExecutionInputs<T>,
        plan: &MemoryPlan,
    ) -> Result<(ExecutionResult<T>, ExecutionStats)> {
        for assignment in plan.assignments.values() {
            if assignment.slot.0 as usize >= plan.slots.len() {
                return Err(Error::execution(format!(
                    "memory plan references missing buffer {}",
                    assignment.slot.0
                )));
            }
            if assignment.dtype != T::DTYPE {
                return Err(Error::DTypeMismatch {
                    context: "memory plan".into(),
                    expected: T::DTYPE,
                    got: assignment.dtype,
                });
            }
        }
        let mut bindings = SymbolBindings::new();
        let mut external: HashMap<ValueId, DenseTensor<T>> = HashMap::new();
        for &input_id in self.graph.inputs() {
            let value = self.graph.value(input_id)?;
            let tensor = inputs.get(input_id).ok_or_else(|| {
                Error::execution(format!(
                    "missing input {:?} ({input_id})",
                    value.name.as_deref().unwrap_or("<unnamed>")
                ))
            })?;
            ensure_cpu(value.device, "input")?;
            ensure_dtype::<T>(value.dtype, "input")?;
            ensure_storage(tensor, "input")?;
            if tensor.rank() != value.shape.rank() {
                return Err(Error::RankMismatch {
                    context: format!("input {input_id}"),
                    expected: value.shape.rank(),
                    got: tensor.rank(),
                });
            }
            bind_shape(&value.shape, &tensor.dims(), &mut bindings)?;
            external.insert(input_id, tensor.clone());
        }
        for parameter in self.graph.parameters() {
            let stored = parameters.get(parameter.id)?;
            let parameter_value = self.graph.value(parameter.value)?;
            ensure_cpu(parameter_value.device, "parameter")?;
            ensure_dtype::<T>(parameter.dtype, "parameter")?;
            ensure_storage(&stored.data, "parameter")?;
            let expected = parameter
                .shape
                .static_dims()
                .ok_or_else(|| Error::invalid_graph("parameter shape is not static"))?;
            if stored.data.dims() != expected {
                return Err(Error::shape_mismatch(
                    format!("parameter {:?}", parameter.name),
                    parameter.shape.clone(),
                    stored.data.shape().clone(),
                ));
            }
            external.insert(parameter.value, stored.data.clone());
        }

        let mut slots: Vec<Option<DenseTensor<T>>> = vec![None; plan.slots.len()];
        let mut owned: HashMap<ValueId, DenseTensor<T>> = HashMap::new();
        let mut stats = ExecutionStats {
            buffer_allocations: plan.allocation_count(),
            planned_peak_bytes: plan.peak_bytes(),
            ..ExecutionStats::default()
        };

        for node in self.graph.nodes() {
            let mut input_tensors = Vec::with_capacity(node.inputs.len());
            for &input in &node.inputs {
                let tensor = external
                    .get(&input)
                    .or_else(|| {
                        plan.assignment(input)
                            .and_then(|assignment| slots[assignment.slot.0 as usize].as_ref())
                    })
                    .or_else(|| owned.get(&input))
                    .ok_or_else(|| {
                        Error::execution(format!("{} missing operand {input}", node.id))
                    })?;
                input_tensors.push(tensor.clone());
            }
            let mut output_shapes = Vec::with_capacity(node.outputs.len());
            for &output in &node.outputs {
                let value = self.graph.value(output)?;
                output_shapes.push(value.shape.bind(&bindings)?);
            }
            let input_specs = node
                .inputs
                .iter()
                .zip(&input_tensors)
                .map(|(&input, tensor)| {
                    let value = self.graph.value(input).expect("validated input");
                    TensorSpec::contiguous(tensor.dims(), value.dtype, value.device)
                })
                .collect::<Vec<_>>();
            let output_specs = node
                .outputs
                .iter()
                .zip(&output_shapes)
                .map(|(&output, shape)| {
                    let value = self.graph.value(output).expect("validated output");
                    TensorSpec::contiguous(shape.clone(), value.dtype, value.device)
                })
                .collect::<Vec<_>>();
            let call = KernelCall {
                op: &node.op,
                attrs: &node.attrs,
                inputs: &input_tensors,
                input_specs: &input_specs,
                output_specs: &output_specs,
                aliasing: &[],
                workspace: oxide_ml_kernel::WorkspaceSpec::none(),
                context: oxide_ml_kernel::ExecutionContext::default(),
            };
            let outputs = if matches!(node.op, oxide_ml_graph::Op::Custom) {
                let registry = self.custom.as_ref().ok_or_else(|| {
                    Error::execution(
                        "graph contains a custom kernel but the executor has no \
                         custom kernel registry",
                    )
                })?;
                let name = match &node.attrs {
                    OpAttrs::Custom { name, .. } => name.as_str(),
                    _ => {
                        return Err(Error::execution("Custom op is missing OpAttrs::Custom"));
                    }
                };
                let device = self
                    .graph
                    .value(node.outputs[0])
                    .map(|value| value.device)
                    .unwrap_or(Device::Cpu);
                registry.execute(name, &input_tensors, &output_shapes, device)?
            } else {
                self.backend.execute(call)?
            };
            if outputs.len() != node.outputs.len() {
                return Err(Error::execution(format!(
                    "{} returned {} tensors for {} outputs",
                    node.op.name(),
                    outputs.len(),
                    node.outputs.len()
                )));
            }
            for ((&value_id, tensor), expected_shape) in
                node.outputs.iter().zip(outputs).zip(&output_shapes)
            {
                let value = self.graph.value(value_id)?;
                ensure_cpu(value.device, node.op.name())?;
                ensure_dtype::<T>(value.dtype, node.op.name())?;
                ensure_storage(&tensor, node.op.name())?;
                if tensor.dims() != *expected_shape {
                    return Err(Error::shape_mismatch(
                        format!("{} output {value_id}", node.op.name()),
                        oxide_ml_core::Shape::from_static(expected_shape),
                        tensor.shape().clone(),
                    ));
                }
                stats.materialized_values += 1;
                if let Some(assignment) = plan.assignment(value_id) {
                    let slot = &mut slots[assignment.slot.0 as usize];
                    if let Some(existing) = slot {
                        if existing.numel() == tensor.numel() {
                            existing.copy_from(&tensor);
                            stats.reused_buffers += 1;
                        } else {
                            *slot = Some(tensor);
                        }
                    } else {
                        *slot = Some(tensor);
                    }
                } else {
                    owned.insert(value_id, tensor);
                }
            }
        }

        let mut values: Vec<Option<DenseTensor<T>>> = vec![None; self.graph.value_count()];
        for &output in self.graph.outputs() {
            let tensor = external
                .get(&output)
                .or_else(|| {
                    plan.assignment(output)
                        .and_then(|assignment| slots[assignment.slot.0 as usize].as_ref())
                })
                .or_else(|| owned.get(&output))
                .ok_or_else(|| Error::execution(format!("output {output} was not computed")))?;
            values[output.0 as usize] = Some(tensor.clone());
        }
        Ok((
            ExecutionResult {
                values,
                outputs: self.graph.outputs().to_vec(),
            },
            stats,
        ))
    }

    pub fn run(
        &self,
        parameters: &ParameterStore<T>,
        inputs: &ExecutionInputs<T>,
    ) -> Result<ExecutionResult<T>> {
        let mut bindings = SymbolBindings::new();
        let mut values: Vec<Option<DenseTensor<T>>> = vec![None; self.graph.value_count()];

        for &input_id in self.graph.inputs() {
            let value = self.graph.value(input_id)?;
            let tensor = inputs.get(input_id).ok_or_else(|| {
                Error::execution(format!(
                    "missing input {:?} ({input_id})",
                    value.name.as_deref().unwrap_or("<unnamed>")
                ))
            })?;
            ensure_cpu(value.device, "input")?;
            ensure_dtype::<T>(value.dtype, "input")?;
            ensure_storage(tensor, "input")?;
            if tensor.rank() != value.shape.rank() {
                return Err(Error::RankMismatch {
                    context: format!("input {input_id}"),
                    expected: value.shape.rank(),
                    got: tensor.rank(),
                });
            }
            bind_shape(&value.shape, &tensor.dims(), &mut bindings)?;
            values[input_id.0 as usize] = Some(tensor.clone());
        }

        for parameter in self.graph.parameters() {
            let stored = parameters.get(parameter.id)?;
            let parameter_value = self.graph.value(parameter.value)?;
            ensure_cpu(parameter_value.device, "parameter")?;
            ensure_dtype::<T>(parameter.dtype, "parameter")?;
            ensure_storage(&stored.data, "parameter")?;
            let expected = parameter
                .shape
                .static_dims()
                .ok_or_else(|| Error::invalid_graph("parameter shape is not static"))?;
            if stored.data.dims() != expected {
                return Err(Error::shape_mismatch(
                    format!("parameter {:?}", parameter.name),
                    parameter.shape.clone(),
                    stored.data.shape().clone(),
                ));
            }
            values[parameter.value.0 as usize] = Some(stored.data.clone());
        }

        for node in self.graph.nodes() {
            let mut input_tensors = Vec::with_capacity(node.inputs.len());
            for &input in &node.inputs {
                let tensor = values[input.0 as usize].as_ref().ok_or_else(|| {
                    Error::execution(format!("{} missing operand {input}", node.id))
                })?;
                input_tensors.push(tensor.clone());
            }
            let mut output_shapes = Vec::with_capacity(node.outputs.len());
            for &output in &node.outputs {
                let value = self.graph.value(output)?;
                let dims = value.shape.bind(&bindings)?;
                output_shapes.push(dims);
            }
            let input_specs = node
                .inputs
                .iter()
                .zip(&input_tensors)
                .map(|(&input, tensor)| {
                    let value = self.graph.value(input).expect("validated input");
                    TensorSpec::contiguous(tensor.dims(), value.dtype, value.device)
                })
                .collect::<Vec<_>>();
            let output_specs = node
                .outputs
                .iter()
                .zip(&output_shapes)
                .map(|(&output, shape)| {
                    let value = self.graph.value(output).expect("validated output");
                    TensorSpec::contiguous(shape.clone(), value.dtype, value.device)
                })
                .collect::<Vec<_>>();
            let call = KernelCall {
                op: &node.op,
                attrs: &node.attrs,
                inputs: &input_tensors,
                input_specs: &input_specs,
                output_specs: &output_specs,
                aliasing: &[],
                workspace: oxide_ml_kernel::WorkspaceSpec::none(),
                context: oxide_ml_kernel::ExecutionContext::default(),
            };
            let outputs = if matches!(node.op, oxide_ml_graph::Op::Custom) {
                let registry = self.custom.as_ref().ok_or_else(|| {
                    Error::execution(
                        "graph contains a custom kernel but the executor has no \
                         custom kernel registry",
                    )
                })?;
                let name = match &node.attrs {
                    OpAttrs::Custom { name, .. } => name.as_str(),
                    _ => {
                        return Err(Error::execution("Custom op is missing OpAttrs::Custom"));
                    }
                };
                let device = self
                    .graph
                    .value(node.outputs[0])
                    .map(|value| value.device)
                    .unwrap_or(Device::Cpu);
                registry.execute(name, &input_tensors, &output_shapes, device)?
            } else {
                self.backend.execute(call)?
            };
            if outputs.len() != node.outputs.len() {
                return Err(Error::execution(format!(
                    "{} returned {} tensors for {} outputs",
                    node.op.name(),
                    outputs.len(),
                    node.outputs.len()
                )));
            }
            for ((&value_id, tensor), expected_shape) in
                node.outputs.iter().zip(outputs).zip(&output_shapes)
            {
                ensure_dtype::<T>(self.graph.value(value_id)?.dtype, node.op.name())?;
                ensure_storage(&tensor, node.op.name())?;
                if tensor.dims() != *expected_shape {
                    return Err(Error::shape_mismatch(
                        format!("{} output {value_id}", node.op.name()),
                        oxide_ml_core::Shape::from_static(expected_shape),
                        tensor.shape().clone(),
                    ));
                }
                values[value_id.0 as usize] = Some(tensor);
            }
        }

        let mut outputs = Vec::with_capacity(self.graph.outputs().len());
        for &id in self.graph.outputs() {
            if values[id.0 as usize].is_none() {
                return Err(Error::execution(format!("output {id} was not computed")));
            }
            outputs.push(id);
        }
        Ok(ExecutionResult { values, outputs })
    }
}

fn ensure_cpu(device: Device, context: &str) -> Result<()> {
    if !device.is_cpu() {
        return Err(Error::unsupported(format!(
            "{context}: device {device} cannot run on the CPU reference backend"
        )));
    }
    Ok(())
}

fn ensure_dtype<T: Scalar>(dtype: DType, context: &str) -> Result<()> {
    if dtype != T::DTYPE {
        return Err(Error::DTypeMismatch {
            context: context.to_string(),
            expected: T::DTYPE,
            got: dtype,
        });
    }
    Ok(())
}

fn ensure_storage<T: Scalar>(tensor: &DenseTensor<T>, context: &str) -> Result<()> {
    // DenseTensor is f32-only today; the explicit check keeps the invariant at
    // the backend boundary visible and makes adding other dtypes a local change.
    if tensor.numel() != tensor.data().len() {
        return Err(Error::execution(format!(
            "{context}: tensor storage length does not match element count"
        )));
    }
    Ok(())
}

fn bind_shape(
    symbolic: &oxide_ml_core::Shape,
    concrete: &[usize],
    bindings: &mut SymbolBindings,
) -> Result<()> {
    if symbolic.rank() != concrete.len() {
        return Err(Error::RankMismatch {
            context: "symbolic shape binding".into(),
            expected: symbolic.rank(),
            got: concrete.len(),
        });
    }
    for (index, (dim, &actual)) in symbolic.dims().iter().zip(concrete).enumerate() {
        match dim {
            Dim::Static(expected) => {
                if *expected != actual {
                    return Err(Error::shape_mismatch(
                        format!("dimension {index}"),
                        oxide_ml_core::Shape::from_static([*expected]),
                        oxide_ml_core::Shape::from_static([actual]),
                    ));
                }
            }
            Dim::Symbol(symbol) => {
                if let Some(previous) = bindings.insert(*symbol, actual)
                    && previous != actual
                {
                    return Err(Error::shape_mismatch(
                        format!("symbol {symbol:?} bound inconsistently"),
                        oxide_ml_core::Shape::from_static([previous]),
                        oxide_ml_core::Shape::from_static([actual]),
                    ));
                }
            }
            Dim::Dynamic => {
                return Err(Error::invalid_shape(
                    "symbolic shape binding",
                    format!("dynamic dimension {index} cannot be specialized"),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use oxide_ml_core::{DType, ParameterId, Shape};
    use oxide_ml_graph::GraphBuilder;

    use super::*;

    #[test]
    fn executes_symbolic_batch() {
        let mut builder = GraphBuilder::new();
        let batch = builder.symbol("batch");
        let x = builder.input(
            "x",
            Shape::new(vec![Dim::Symbol(batch), Dim::Static(2)]),
            DType::F32,
        );
        let w = builder
            .try_parameter("w", Shape::from_static([2, 1]), DType::F32, true)
            .unwrap();
        let y = builder.matmul(x, w).unwrap();
        let graph = builder.finish(&[y]).unwrap();
        let mut params = ParameterStore::<f32>::new();
        params
            .add(
                ParameterId(0),
                "w",
                DenseTensor::from_vec([2, 1], vec![2.0, 3.0]).unwrap(),
                true,
            )
            .unwrap();
        let mut inputs = ExecutionInputs::<f32>::new();
        inputs.insert(
            graph.inputs()[0],
            DenseTensor::from_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        let executor = Executor::<f32>::new(graph).unwrap();
        let result = executor.run(&params, &inputs).unwrap();
        assert_eq!(result.output(0).unwrap().data(), &[8.0, 18.0]);
    }
}
