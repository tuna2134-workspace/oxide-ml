use std::collections::HashMap;
use std::sync::Arc;

use cuda_core::CudaStream;
use oxide_ml_core::{DType, DenseTensor, Shape, ValueId};
use oxide_ml_graph::{Graph, Op, OpAttrs, Reduction};

use crate::buffer::{CudaBufferPool, CudaTensor};
use crate::device::CudaDevice;
use crate::error::{CudaError, Result};
use crate::event::{CudaEvent, EventPool};
use crate::kernels::CudaKernels;
use crate::memplan::CudaMemoryPlan;
use crate::param::CudaParameterStore;

pub struct CudaRun {
    pub values: HashMap<ValueId, CudaTensor>,
    pub outputs: Vec<ValueId>,
}

impl CudaRun {
    pub fn value(&self, id: ValueId) -> Result<&CudaTensor> {
        self.values
            .get(&id)
            .ok_or_else(|| CudaError::Internal(format!("CUDA value {id} was not materialized")))
    }

    pub fn output(&self, index: usize) -> Result<&CudaTensor> {
        let id = *self
            .outputs
            .get(index)
            .ok_or_else(|| CudaError::Internal(format!("output index {index} out of range")))?;
        self.value(id)
    }

    pub fn download(&self, stream: &CudaStream, id: ValueId) -> Result<DenseTensor<f32>> {
        let tensor = self.value(id)?;
        let mut host = tensor.to_host(stream)?;
        host.truncate(tensor.numel());
        Ok(DenseTensor::from_vec(tensor.shape(), host)?)
    }

    pub fn download_outputs(
        &self,
        stream: &CudaStream,
    ) -> Result<HashMap<ValueId, DenseTensor<f32>>> {
        let mut out = HashMap::new();
        for &id in &self.outputs {
            out.insert(id, self.download(stream, id)?);
        }
        Ok(out)
    }
}

pub struct CudaExecution {
    run: CudaRun,
    stream: Arc<CudaStream>,
}

impl CudaExecution {
    pub fn output_tensor(&self, index: usize) -> Result<CudaTensor> {
        let id = *self
            .run
            .outputs
            .get(index)
            .ok_or_else(|| CudaError::Internal(format!("output index {index} out of range")))?;
        Ok(self.run.value(id)?.clone())
    }

    pub fn is_complete(&self) -> Result<bool> {
        Ok(self.stream.query()?)
    }

    pub fn wait(&self) -> Result<HashMap<ValueId, DenseTensor<f32>>> {
        self.stream.synchronize()?;
        self.run.download_outputs(&self.stream)
    }

    pub fn collect(&mut self) -> Result<Option<HashMap<ValueId, DenseTensor<f32>>>> {
        if self.stream.query()? {
            Ok(Some(self.run.download_outputs(&self.stream)?))
        } else {
            Ok(None)
        }
    }
}

// Drop intentionally does not synchronize: every buffer used by a node is
// tagged with that node's completion event, so PooledBuffer moves to the
// deferred-release queue instead of returning to the free list. The pending
// events and their context are owned by that queue.

pub struct CudaExecutor {
    device: CudaDevice,
    pool: CudaBufferPool,
    kernels: CudaKernels,
    events: EventPool,
}

impl CudaExecutor {
    pub fn new(device: CudaDevice) -> Result<Self> {
        let stream = device.default_stream();
        let pool = CudaBufferPool::new(device.context().clone(), stream);
        let kernels = CudaKernels::load(device.context())?;
        let events = EventPool::new(device.context().clone());
        Ok(CudaExecutor {
            device,
            pool,
            kernels,
            events,
        })
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn pool_stats(&self) -> crate::buffer::PoolStats {
        self.pool.stats()
    }

    pub fn pool(&self) -> &CudaBufferPool {
        &self.pool
    }

    pub fn kernels(&self) -> &CudaKernels {
        &self.kernels
    }

    pub fn event_stats(&self) -> crate::event::EventPoolStats {
        self.events.stats()
    }

    pub fn event_pool(&self) -> &crate::event::EventPool {
        &self.events
    }

    pub fn shutdown_events(&self) -> crate::event::CleanupReport {
        self.events.shutdown()
    }

    pub fn run(
        &self,
        graph: &Graph,
        parameters: &CudaParameterStore,
        inputs: &HashMap<ValueId, DenseTensor<f32>>,
    ) -> Result<CudaRun> {
        graph.validate()?;
        let stream = self.device.default_stream();
        let mut values: HashMap<ValueId, CudaTensor> = HashMap::new();

        for &input_id in graph.inputs() {
            let value = graph.value(input_id)?;
            ensure_f32(value.dtype)?;
            let shape = static_shape(value.shape.clone())?;
            let host = inputs.get(&input_id).ok_or_else(|| {
                CudaError::Shape(format!(
                    "missing host input {:?} ({input_id})",
                    value.name.as_deref().unwrap_or("<unnamed>")
                ))
            })?;
            if host.dims() != shape {
                return Err(CudaError::Shape(format!(
                    "input {input_id} shape mismatch: host {:?}, graph {shape:?}",
                    host.dims()
                )));
            }
            values.insert(
                input_id,
                CudaTensor::from_host(&self.pool, shape, host.data())?,
            );
        }

        for parameter in graph.parameters() {
            values.insert(parameter.value, parameters.get(parameter.id)?.clone());
        }

        for node in graph.nodes() {
            let mut operands = Vec::with_capacity(node.inputs.len());
            for &input in &node.inputs {
                operands.push(values.get(&input).cloned().ok_or_else(|| {
                    CudaError::Internal(format!("{} missing operand {input}", node.id))
                })?);
            }
            let output_shapes = node
                .outputs
                .iter()
                .map(|&output| static_shape(graph.value(output)?.shape.clone()))
                .collect::<Result<Vec<_>>>()?;
            let stream = self.device.default_stream();
            let outputs = self.execute_node(
                &stream,
                &node.op,
                &node.attrs,
                &operands,
                &output_shapes,
                None,
            )?;
            let lease = self.events.acquire()?;
            lease.record(&stream)?;
            let event = lease.arc()?;
            for operand in &operands {
                operand.mark_used(event.clone())?;
            }
            for tensor in &outputs {
                tensor.mark_used(event.clone())?;
            }
            drop(lease);
            for (&output, tensor) in node.outputs.iter().zip(outputs) {
                values.insert(output, tensor);
            }
        }
        stream.synchronize()?;
        Ok(CudaRun {
            values,
            outputs: graph.outputs().to_vec(),
        })
    }

    pub fn execute_operation(
        &self,
        op: &Op,
        attrs: &OpAttrs,
        operands: &[CudaTensor],
        output_shapes: &[Vec<usize>],
    ) -> Result<Vec<CudaTensor>> {
        let stream = self.device.context().default_stream();
        self.execute_node(&stream, op, attrs, operands, output_shapes, None)
    }

    /// Execute with a CUDA adapter for the Graph Compiler's MemoryPlan.
    ///
    /// Values with a planned slot execute into that slot's preallocated device
    /// buffer. Values without an assignment (dynamic shapes or an incomplete
    /// plan) use the ordinary pool path.
    pub fn run_planned(
        &self,
        graph: &Graph,
        parameters: &CudaParameterStore,
        inputs: &HashMap<ValueId, DenseTensor<f32>>,
        plan: &CudaMemoryPlan,
    ) -> Result<CudaRun> {
        let stream = self.device.default_stream();
        let run = self.run_planned_on(graph, parameters, inputs, plan, &stream)?;
        stream.synchronize()?;
        Ok(run)
    }

    pub fn execute_async_planned(
        &self,
        graph: &Graph,
        parameters: &CudaParameterStore,
        inputs: &HashMap<ValueId, DenseTensor<f32>>,
        plan: &CudaMemoryPlan,
        stream: Arc<CudaStream>,
    ) -> Result<CudaExecution> {
        let run = self.run_planned_on(graph, parameters, inputs, plan, &stream)?;
        Ok(CudaExecution { run, stream })
    }

    fn run_planned_on(
        &self,
        graph: &Graph,
        parameters: &CudaParameterStore,
        inputs: &HashMap<ValueId, DenseTensor<f32>>,
        plan: &CudaMemoryPlan,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaRun> {
        graph.validate()?;
        let mut values: HashMap<ValueId, CudaTensor> = HashMap::new();
        for &input_id in graph.inputs() {
            let value = graph.value(input_id)?;
            ensure_f32(value.dtype)?;
            let shape = static_shape(value.shape.clone())?;
            let host = inputs
                .get(&input_id)
                .ok_or_else(|| CudaError::Shape(format!("missing host input {input_id}")))?;
            values.insert(
                input_id,
                CudaTensor::from_host(&self.pool, shape, host.data())?,
            );
        }
        for parameter in graph.parameters() {
            values.insert(parameter.value, parameters.get(parameter.id)?.clone());
        }

        let mut slots: Vec<Option<CudaTensor>> = Vec::with_capacity(plan.slots.len());
        for slot in &plan.slots {
            slots.push(Some(CudaTensor::from_pool(
                &self.pool,
                vec![slot.elements_f32],
            )?));
        }
        let mut owners: Vec<Option<ValueId>> = vec![None; plan.slots.len()];
        let mut slot_events: Vec<Vec<Arc<CudaEvent>>> = vec![Vec::new(); plan.slots.len()];
        let mut slot_streams: Vec<Option<Arc<CudaStream>>> = vec![None; plan.slots.len()];
        let mut planned_reuse = 0usize;
        let mut fallback_allocations = 0usize;

        for node in graph.nodes() {
            let mut operands = Vec::with_capacity(node.inputs.len());
            for &input in &node.inputs {
                operands.push(values.get(&input).cloned().ok_or_else(|| {
                    CudaError::Internal(format!("{} missing operand {input}", node.id))
                })?);
            }
            let output_shapes = node
                .outputs
                .iter()
                .map(|&output| static_shape(graph.value(output)?.shape.clone()))
                .collect::<Result<Vec<_>>>()?;
            let mut preallocated = Vec::with_capacity(node.outputs.len());
            for (&output, shape) in node.outputs.iter().zip(&output_shapes) {
                if let Some(assignment) = plan.assignment(output) {
                    let slot_index = assignment.slot as usize;
                    if let Some(previous) = owners[slot_index].take() {
                        values.remove(&previous);
                        planned_reuse += 1;
                    }
                    let same_stream = slot_streams[slot_index]
                        .as_ref()
                        .is_some_and(|previous| Arc::ptr_eq(previous, stream));
                    if !same_stream {
                        for event in &slot_events[slot_index] {
                            event.wait_on(stream)?;
                        }
                    }
                    let mut tensor = slots[assignment.slot as usize]
                        .take()
                        .ok_or_else(|| CudaError::InvalidPlan("slot is already in use".into()))?;
                    tensor.set_shape(shape.clone())?;
                    preallocated.push(tensor);
                } else {
                    fallback_allocations += 1;
                    preallocated.push(CudaTensor::from_pool(&self.pool, shape.clone())?);
                }
            }
            let outputs = self.execute_node(
                stream,
                &node.op,
                &node.attrs,
                &operands,
                &output_shapes,
                Some(preallocated),
            )?;
            let lease = self.events.acquire()?;
            lease.record(stream)?;
            let event = lease.arc()?;
            for operand in &operands {
                operand.mark_used(event.clone())?;
            }
            for tensor in &outputs {
                tensor.mark_used(event.clone())?;
            }
            drop(lease);
            for (&output, tensor) in node.outputs.iter().zip(outputs) {
                if let Some(assignment) = plan.assignment(output) {
                    let slot_index = assignment.slot as usize;
                    slots[slot_index] = Some(tensor.clone());
                    owners[slot_index] = Some(output);
                    slot_events[slot_index].push(event.clone());
                    slot_streams[slot_index] = Some(stream.clone());
                }
                values.insert(output, tensor);
            }
        }
        self.pool.record_plan(
            plan.slot_count(),
            planned_reuse,
            fallback_allocations,
            plan.logical_bytes,
        );
        stream.synchronize()?;
        Ok(CudaRun {
            values,
            outputs: graph.outputs().to_vec(),
        })
    }
    fn execute_node(
        &self,
        stream: &Arc<CudaStream>,
        op: &Op,
        attrs: &OpAttrs,
        operands: &[CudaTensor],
        output_shapes: &[Vec<usize>],
        preallocated: Option<Vec<CudaTensor>>,
    ) -> Result<Vec<CudaTensor>> {
        let mut prepared = preallocated.unwrap_or_default();
        let mut next_index = 0usize;
        let mut next_output = || -> Result<CudaTensor> {
            let index = next_index;
            next_index += 1;
            if prepared.is_empty() {
                CudaTensor::from_pool(
                    &self.pool,
                    output_shapes
                        .get(index)
                        .cloned()
                        .ok_or_else(|| CudaError::Internal("missing output shape".into()))?,
                )
            } else {
                Ok(prepared.remove(0))
            }
        };
        match op {
            Op::Constant => {
                let value = match attrs {
                    OpAttrs::Constant { value } => value,
                    _ => return Err(CudaError::Internal("Constant attrs missing".into())),
                };
                let data = match &value.data {
                    oxide_ml_graph::ConstantData::F32(values) => values.clone(),
                    oxide_ml_graph::ConstantData::F64(_) => {
                        return Err(CudaError::Unsupported(
                            "CUDA backend currently supports F32 constants only".into(),
                        ));
                    }
                };
                let mut out = next_output()?;
                out.copy_from_host(stream, &data)?;
                Ok(vec![out])
            }
            Op::ConstantScalar => {
                let value = match attrs {
                    OpAttrs::ConstantScalar { value } => *value,
                    _ => return Err(CudaError::Internal("ConstantScalar attrs missing".into())),
                };
                let mut out = next_output()?;
                out.copy_from_host(stream, &[value])?;
                Ok(vec![out])
            }
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::AddRelu => {
                let opcode = match op {
                    Op::Add => 0,
                    Op::Sub => 1,
                    Op::Mul => 2,
                    Op::Div => 3,
                    Op::AddRelu => 4,
                    _ => unreachable!(),
                };
                let mut out = next_output()?;
                self.kernels
                    .binary(stream, opcode, &operands[0], &operands[1], &mut out)?;
                Ok(vec![out])
            }
            Op::Neg | Op::ReLU | Op::GELU | Op::Sigmoid | Op::Tanh => {
                let opcode = match op {
                    Op::ReLU => 0,
                    Op::GELU => 1,
                    Op::Sigmoid => 2,
                    Op::Tanh => 3,
                    Op::Neg => 4,
                    _ => unreachable!(),
                };
                let mut out = next_output()?;
                self.kernels.unary(stream, opcode, &operands[0], &mut out)?;
                Ok(vec![out])
            }
            Op::Reshape => {
                let mut out = next_output()?;
                self.kernels.broadcast(stream, &operands[0], &mut out)?;
                Ok(vec![out])
            }
            Op::Transpose => {
                let axes = match attrs {
                    OpAttrs::Transpose { axes } => axes,
                    _ => return Err(CudaError::Internal("Transpose attrs missing".into())),
                };
                let mut out = next_output()?;
                self.kernels
                    .transpose(stream, &operands[0], &mut out, axes)?;
                Ok(vec![out])
            }
            Op::BroadcastTo => {
                let mut out = next_output()?;
                self.kernels.broadcast(stream, &operands[0], &mut out)?;
                Ok(vec![out])
            }
            Op::ReduceSum | Op::ReduceMean => {
                let (axes, _) = match attrs {
                    OpAttrs::Reduce { axes, keep_dims } => (axes, *keep_dims),
                    _ => return Err(CudaError::Internal("Reduce attrs missing".into())),
                };
                let mut out = next_output()?;
                self.kernels.reduce(
                    stream,
                    &operands[0],
                    &mut out,
                    axes,
                    matches!(op, Op::ReduceMean),
                )?;
                Ok(vec![out])
            }
            Op::MatMul => {
                let (transpose_a, transpose_b) = match attrs {
                    OpAttrs::MatMul {
                        transpose_a,
                        transpose_b,
                    } => (*transpose_a, *transpose_b),
                    _ => return Err(CudaError::Internal("MatMul attrs missing".into())),
                };
                if operands[0].shape().len() != 2 || operands[1].shape().len() != 2 {
                    return Err(CudaError::Unsupported(
                        "CUDA matmul currently supports rank-2 matrices only".into(),
                    ));
                }
                let a = operands[0].shape();
                let b = operands[1].shape();
                let (m, k) = if transpose_a {
                    (a[1], a[0])
                } else {
                    (a[0], a[1])
                };
                let (k2, n) = if transpose_b {
                    (b[1], b[0])
                } else {
                    (b[0], b[1])
                };
                if k != k2 {
                    return Err(CudaError::Shape("matmul K mismatch".into()));
                }
                let lda = a[1];
                let ldb = b[1];
                let mut out = next_output()?;
                if !transpose_a && !transpose_b && m >= 32 && n >= 32 && k >= 32 {
                    self.kernels.gemm_tiled(
                        stream,
                        &operands[0],
                        &operands[1],
                        &mut out,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.kernels.gemm_naive(
                        stream,
                        &operands[0],
                        &operands[1],
                        &mut out,
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        transpose_a,
                        transpose_b,
                    )?;
                }
                Ok(vec![out])
            }
            Op::MSELoss => {
                let mean = matches!(
                    attrs,
                    OpAttrs::Loss {
                        reduction: Reduction::Mean
                    }
                );
                let mut out = next_output()?;
                self.kernels
                    .mse(stream, &operands[0], &operands[1], &mut out, mean)?;
                Ok(vec![out])
            }
            Op::MSELossBackward => {
                let mean = matches!(
                    attrs,
                    OpAttrs::Loss {
                        reduction: Reduction::Mean
                    }
                );
                let mut dpred = next_output()?;
                let mut dtarget = next_output()?;
                self.kernels.mse_backward(
                    stream,
                    &operands[0],
                    &operands[1],
                    &operands[2],
                    &mut dpred,
                    &mut dtarget,
                    mean,
                )?;
                Ok(vec![dpred, dtarget])
            }
            Op::ActivationBackward(kind) => {
                let opcode = match kind {
                    oxide_ml_graph::ActivationKind::ReLU => 0,
                    oxide_ml_graph::ActivationKind::GELU => 1,
                    oxide_ml_graph::ActivationKind::Sigmoid => 2,
                    oxide_ml_graph::ActivationKind::Tanh => 3,
                };
                let mut out = next_output()?;
                self.kernels.activation_backward(
                    stream,
                    opcode,
                    &operands[0],
                    &operands[1],
                    &mut out,
                )?;
                Ok(vec![out])
            }
            Op::AddReluBackward => {
                let mut out = next_output()?;
                self.kernels.add_relu_backward(
                    stream,
                    &operands[0],
                    &operands[1],
                    &operands[2],
                    &mut out,
                )?;
                Ok(vec![out])
            }
            Op::Unbroadcast => {
                let target = match attrs {
                    OpAttrs::Unbroadcast { shape } => static_shape(shape.clone())?,
                    _ => return Err(CudaError::Internal("Unbroadcast attrs missing".into())),
                };
                let axes = unbroadcast_axes(operands[0].shape(), &target)?;
                let mut out = next_output()?;
                self.kernels
                    .reduce(stream, &operands[0], &mut out, &axes, false)?;
                Ok(vec![out])
            }
            Op::Softmax => {
                let axis = match attrs {
                    OpAttrs::Softmax { axis } => {
                        oxide_ml_graph::normalize_axis(*axis, operands[0].shape().len())?
                    }
                    _ => return Err(CudaError::Internal("Softmax attrs missing".into())),
                };
                let mut out = next_output()?;
                self.kernels.softmax(stream, &operands[0], &mut out, axis)?;
                Ok(vec![out])
            }
            Op::LayerNorm => {
                let normalized = match attrs {
                    OpAttrs::LayerNorm {
                        normalized_shape, ..
                    } => normalized_shape.iter().product::<usize>(),
                    _ => return Err(CudaError::Internal("LayerNorm attrs missing".into())),
                };
                let mut out = next_output()?;
                self.kernels.layer_norm(
                    stream,
                    &operands[0],
                    &operands[1],
                    &operands[2],
                    &mut out,
                    normalized as u32,
                )?;
                Ok(vec![out])
            }
            other => Err(CudaError::Unsupported(format!(
                "operation {} is not implemented by the CUDA backend",
                other.name()
            ))),
        }
    }
}

fn ensure_f32(dtype: DType) -> Result<()> {
    if dtype != DType::F32 {
        return Err(CudaError::Unsupported(format!(
            "CUDA backend currently supports F32 only, got {dtype}"
        )));
    }
    Ok(())
}

fn static_shape(shape: Shape) -> Result<Vec<usize>> {
    shape
        .static_dims()
        .ok_or_else(|| CudaError::Shape(format!("shape {shape} is not fully static")))
}

fn unbroadcast_axes(input: &[usize], target: &[usize]) -> Result<Vec<usize>> {
    if target.len() > input.len() {
        return Err(CudaError::Shape(
            "unbroadcast target rank exceeds input".into(),
        ));
    }
    let offset = input.len() - target.len();
    let mut axes = Vec::new();
    for axis in 0..input.len() {
        if axis < offset {
            axes.push(axis);
        } else {
            let target_dim = target[axis - offset];
            if target_dim == 1 && input[axis] != 1 {
                axes.push(axis);
            } else if target_dim != input[axis] {
                return Err(CudaError::Shape(format!(
                    "unbroadcast {input:?} -> {target:?} is invalid"
                )));
            }
        }
    }
    Ok(axes)
}

// Keep `Arc` imported for the public `CudaTensor::storage` type in signatures
// used by downstream code.
#[allow(dead_code)]
fn _arc_marker(_: Arc<()>) {}

impl CudaExecutor {
    /// Execute a planned graph using a persistent arena slot set.
    ///
    /// The returned `CudaRun` contains only graph outputs, copied out of the
    /// arena slots so the slots can be leased again immediately.
    pub fn run_arena(
        &self,
        graph: &Graph,
        parameters: &CudaParameterStore,
        inputs: &HashMap<ValueId, DenseTensor<f32>>,
        arena: &crate::arena::CudaPlannedArena,
        stream: Arc<CudaStream>,
    ) -> Result<CudaRun> {
        let mut lease = arena.lease(&stream)?;
        let (slots, mut slot_events, mut slot_streams) = lease.take()?;
        let (run, slots, slot_events, slot_streams) = self.run_arena_on(
            graph,
            parameters,
            inputs,
            arena.plan(),
            slots,
            &mut slot_events,
            &mut slot_streams,
            &stream,
        )?;
        lease.finish(slots, slot_events, slot_streams)?;
        stream.synchronize()?;
        Ok(run)
    }

    fn run_arena_on(
        &self,
        graph: &Graph,
        parameters: &CudaParameterStore,
        inputs: &HashMap<ValueId, DenseTensor<f32>>,
        plan: &CudaMemoryPlan,
        slot_buffers: Vec<CudaTensor>,
        slot_events: &mut Vec<Vec<Arc<CudaEvent>>>,
        slot_streams: &mut Vec<Option<Arc<CudaStream>>>,
        stream: &Arc<CudaStream>,
    ) -> Result<(
        CudaRun,
        Vec<CudaTensor>,
        Vec<Vec<Arc<CudaEvent>>>,
        Vec<Option<Arc<CudaStream>>>,
    )> {
        graph.validate()?;
        let mut values: HashMap<ValueId, CudaTensor> = HashMap::new();
        for &input_id in graph.inputs() {
            let value = graph.value(input_id)?;
            ensure_f32(value.dtype)?;
            let shape = static_shape(value.shape.clone())?;
            let host = inputs
                .get(&input_id)
                .ok_or_else(|| CudaError::Shape(format!("missing host input {input_id}")))?;
            values.insert(
                input_id,
                CudaTensor::from_host(&self.pool, shape, host.data())?,
            );
        }
        for parameter in graph.parameters() {
            values.insert(parameter.value, parameters.get(parameter.id)?.clone());
        }

        let mut slots: Vec<Option<CudaTensor>> = slot_buffers.into_iter().map(Some).collect();
        let mut owners: Vec<Option<ValueId>> = vec![None; slots.len()];
        let mut planned_reuse = 0usize;

        for node in graph.nodes() {
            let mut operands = Vec::with_capacity(node.inputs.len());
            for &input in &node.inputs {
                operands.push(values.get(&input).cloned().ok_or_else(|| {
                    CudaError::Internal(format!("{} missing operand {input}", node.id))
                })?);
            }
            let output_shapes = node
                .outputs
                .iter()
                .map(|&output| static_shape(graph.value(output)?.shape.clone()))
                .collect::<Result<Vec<_>>>()?;
            let mut preallocated = Vec::with_capacity(node.outputs.len());
            for (&output, shape) in node.outputs.iter().zip(&output_shapes) {
                if let Some(assignment) = plan.assignment(output) {
                    let slot_index = assignment.slot as usize;
                    if let Some(previous) = owners[slot_index].take() {
                        values.remove(&previous);
                        planned_reuse += 1;
                    }
                    let same_stream = slot_streams[slot_index]
                        .as_ref()
                        .is_some_and(|previous| Arc::ptr_eq(previous, stream));
                    if !same_stream {
                        for event in &slot_events[slot_index] {
                            event.wait_on(stream)?;
                        }
                    }
                    let mut tensor = slots[slot_index]
                        .take()
                        .ok_or_else(|| CudaError::InvalidPlan("slot is already in use".into()))?;
                    tensor.set_shape(shape.clone())?;
                    preallocated.push(tensor);
                } else {
                    preallocated.push(CudaTensor::from_pool(&self.pool, shape.clone())?);
                }
            }
            let outputs = self.execute_node(
                stream,
                &node.op,
                &node.attrs,
                &operands,
                &output_shapes,
                Some(preallocated),
            )?;
            let lease = self.events.acquire()?;
            lease.record(stream)?;
            let event = lease.arc()?;
            for operand in &operands {
                operand.mark_used(event.clone())?;
            }
            for tensor in &outputs {
                tensor.mark_used(event.clone())?;
            }
            drop(lease);
            for (&output, tensor) in node.outputs.iter().zip(outputs) {
                if let Some(assignment) = plan.assignment(output) {
                    let slot_index = assignment.slot as usize;
                    slots[slot_index] = Some(tensor.clone());
                    owners[slot_index] = Some(output);
                    slot_events[slot_index].push(event.clone());
                    slot_streams[slot_index] = Some(stream.clone());
                }
                values.insert(output, tensor);
            }
        }

        // Copy outputs so arena slots are not externally referenced.
        let mut out_values = HashMap::new();
        for &output in graph.outputs() {
            let src = values
                .get(&output)
                .cloned()
                .ok_or_else(|| CudaError::Internal(format!("missing graph output {output}")))?;
            let shape = src.shape().to_vec();
            let mut dst = CudaTensor::from_pool(&self.pool, shape)?;
            self.kernels.broadcast(stream, &src, &mut dst)?;
            let lease = self.events.acquire()?;
            lease.record(stream)?;
            let event = lease.arc()?;
            src.mark_used(event.clone())?;
            dst.mark_used(event.clone())?;
            drop(lease);
            out_values.insert(output, dst);
        }
        self.pool
            .record_plan(plan.slot_count(), planned_reuse, 0, plan.logical_bytes);
        let slots = slots
            .into_iter()
            .map(|slot| {
                slot.ok_or_else(|| CudaError::Internal("arena slot missing after execution".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((
            CudaRun {
                values: out_values,
                outputs: graph.outputs().to_vec(),
            },
            slots,
            std::mem::take(slot_events),
            std::mem::take(slot_streams),
        ))
    }
}
