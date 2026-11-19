use std::collections::HashMap;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream};
use oxide_ml_core::{DType, DenseTensor, Dim, Shape, SymbolId, ValueId};
use oxide_ml_execution_safety::{
    ExecutionDiagnostics, ExecutionDiagnosticsSnapshot, ExecutionState, ExecutionStateError,
    SharedExecutionLifecycle,
};
use oxide_ml_graph::{Graph, Op, OpAttrs, Reduction, ValueInfo, infer_outputs};

use crate::arena::SlotSetLease;
use crate::buffer::{CudaBufferPool, CudaTensor};
use crate::device::CudaDevice;
use crate::error::{CudaError, Result};
use crate::event::{CudaEvent, EventLease, EventPool};
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

struct ArenaLeaseState {
    lease: SlotSetLease,
    slots: Option<Vec<CudaTensor>>,
    events: Vec<Vec<Arc<CudaEvent>>>,
    streams: Vec<Option<Arc<CudaStream>>>,
}

impl ArenaLeaseState {
    /// Finishes the arena lease without dropping slot buffers on failure.
    ///
    /// A failure leaves the resources in `self`, so the caller can keep them
    /// alive until `ExecutionResources::drop` rather than prematurely freeing
    /// memory the GPU may still be using.
    fn finish(&mut self, diagnostics: &ExecutionDiagnostics) -> bool {
        if self.lease.is_finished() {
            self.slots = None;
            return true;
        }
        let Some(slots) = self.slots.take() else {
            return true;
        };
        let events = std::mem::take(&mut self.events);
        let streams = std::mem::take(&mut self.streams);
        match self.lease.finish_or_return(slots, events, streams) {
            Ok(()) => true,
            Err((slots, events, streams)) => {
                self.slots = Some(slots);
                self.events = events;
                self.streams = streams;
                diagnostics.record_cleanup_failure(
                    "arena lease finish failed; resources retained for deferred release",
                );
                false
            }
        }
    }
}

/// Resources owned by one asynchronous execution.
///
/// Field order is intentional: `drop` first defers the arena lease, then
/// fields drop in declaration order. Buffers carry their own completion
/// events, so `CudaRun` can drop before the stream/context/event lease without
/// releasing GPU-used memory early.
pub struct ExecutionResources {
    run: Option<CudaRun>,
    stream: Arc<CudaStream>,
    context: Arc<CudaContext>,
    completion: Option<EventLease>,
    arena: Option<ArenaLeaseState>,
    diagnostics: Arc<ExecutionDiagnostics>,
}

impl ExecutionResources {
    fn new(
        run: CudaRun,
        stream: Arc<CudaStream>,
        context: Arc<CudaContext>,
        completion: Option<EventLease>,
        arena: Option<ArenaLeaseState>,
        diagnostics: Arc<ExecutionDiagnostics>,
    ) -> Self {
        ExecutionResources {
            run: Some(run),
            stream,
            context,
            completion,
            arena,
            diagnostics,
        }
    }

    fn run(&self) -> Result<&CudaRun> {
        self.run
            .as_ref()
            .ok_or_else(|| CudaError::Internal("execution run is no longer available".into()))
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    fn completion(&self) -> Result<&EventLease> {
        self.completion
            .as_ref()
            .ok_or_else(|| CudaError::Internal("execution completion event is missing".into()))
    }

    fn output_tensor(&self, index: usize) -> Result<CudaTensor> {
        Ok(self.run()?.output(index)?.clone())
    }

    fn finish_arena(&mut self) {
        if let Some(mut arena) = self.arena.take()
            && !arena.finish(&self.diagnostics)
        {
            // Keep the resources alive; the final drop will try once more.
            self.arena = Some(arena);
        }
    }

    fn diagnostics(&self) -> &Arc<ExecutionDiagnostics> {
        &self.diagnostics
    }
}

impl Drop for ExecutionResources {
    fn drop(&mut self) {
        // No host synchronization. Arena slots move to PendingCompletion;
        // buffers move to deferred reclamation/quarantine through their own
        // Drop implementations.
        self.finish_arena();
    }
}

/// Handle for one asynchronous execution.
///
/// The execution owns a `CudaRun` (all device buffers and their completion
/// events), the submission stream, an explicit `Arc<CudaContext>`, a final
/// completion `EventLease`, and an optional arena lease. It does not hold the
/// `CudaExecutor`.
pub struct CudaExecution {
    resources: ExecutionResources,
    lifecycle: SharedExecutionLifecycle,
}

impl CudaExecution {
    fn new(resources: ExecutionResources, lifecycle: SharedExecutionLifecycle) -> Self {
        CudaExecution {
            resources,
            lifecycle,
        }
    }

    pub fn state(&self) -> ExecutionState {
        self.lifecycle.state()
    }

    pub fn diagnostics(&self) -> ExecutionDiagnosticsSnapshot {
        self.resources.diagnostics().snapshot()
    }

    pub fn output_tensor(&self, index: usize) -> Result<CudaTensor> {
        self.resources.output_tensor(index)
    }

    /// Returns an independent external output owner.
    pub fn collect_output(&mut self, index: usize) -> Result<CudaTensor> {
        self.resources.output_tensor(index)
    }

    /// Non-blocking CUDA Event query. Never synchronizes the host.
    pub fn is_complete(&self) -> Result<bool> {
        match self.lifecycle.state() {
            ExecutionState::Collected => return Ok(true),
            ExecutionState::Failed => {
                return Err(execution_state_error(ExecutionStateError::Failed));
            }
            ExecutionState::Completed => return Ok(true),
            ExecutionState::Submitted | ExecutionState::Running => {}
        }
        let event = self.resources.completion()?;
        match event.is_complete() {
            Ok(true) => {
                self.lifecycle
                    .mark_completed()
                    .map_err(execution_state_error)?;
                Ok(true)
            }
            Ok(false) => {
                self.lifecycle
                    .note_pending()
                    .map_err(execution_state_error)?;
                Ok(false)
            }
            Err(error) => {
                let _ = self.lifecycle.mark_failed();
                Err(error)
            }
        }
    }

    /// Synchronizes the execution completion event and downloads outputs.
    ///
    /// The execution remains alive and its output buffers remain owned. The
    /// legacy method name and output map are preserved for existing callers.
    pub fn wait(&mut self) -> Result<HashMap<ValueId, DenseTensor<f32>>> {
        match self.lifecycle.state() {
            ExecutionState::Collected => {
                return Err(execution_state_error(ExecutionStateError::AlreadyCollected));
            }
            ExecutionState::Failed => {
                return Err(execution_state_error(ExecutionStateError::Failed));
            }
            _ => {}
        }

        if let Err(error) = self.resources.completion()?.synchronize() {
            let _ = self.lifecycle.mark_failed();
            return Err(error);
        }
        self.lifecycle
            .mark_completed()
            .map_err(execution_state_error)?;
        self.resources.finish_arena();

        let result = {
            let run = self.resources.run()?;
            run.download_outputs(self.resources.stream())
        };
        match result {
            Ok(outputs) => Ok(outputs),
            Err(error) => {
                self.resources.diagnostics().record_d2h_failure();
                let _ = self.lifecycle.mark_failed();
                Err(error)
            }
        }
    }

    /// Non-blocking collection. Returns `None` while the GPU is still running.
    pub fn collect(&mut self) -> Result<Option<HashMap<ValueId, DenseTensor<f32>>>> {
        match self.lifecycle.state() {
            ExecutionState::Collected => {
                return Err(execution_state_error(ExecutionStateError::AlreadyCollected));
            }
            ExecutionState::Failed => {
                return Err(execution_state_error(ExecutionStateError::Failed));
            }
            ExecutionState::Completed => {}
            ExecutionState::Submitted | ExecutionState::Running => {
                if !self.is_complete()? {
                    return Ok(None);
                }
            }
        }

        self.lifecycle
            .begin_collect()
            .map_err(execution_state_error)?;
        let result = {
            let run = self.resources.run()?;
            run.download_outputs(self.resources.stream())
        };
        match result {
            Ok(outputs) => {
                self.lifecycle
                    .mark_collected()
                    .map_err(execution_state_error)?;
                self.resources.finish_arena();
                Ok(Some(outputs))
            }
            Err(error) => {
                self.resources.diagnostics().record_d2h_failure();
                let _ = self.lifecycle.mark_failed();
                Err(error)
            }
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
    custom: Option<std::sync::Arc<crate::custom::CudaKernelRegistry>>,
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
            custom: None,
        })
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Attach a registry of user-defined CUDA kernels.
    pub fn with_custom_kernels(
        mut self,
        registry: std::sync::Arc<crate::custom::CudaKernelRegistry>,
    ) -> Self {
        self.custom = Some(registry);
        self
    }

    pub fn custom_kernels(&self) -> Option<&std::sync::Arc<crate::custom::CudaKernelRegistry>> {
        self.custom.as_ref()
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

    /// Runtime shape signature for cache identity. This is separate from the
    /// canonical graph/plan hash and must not be used to reuse a different
    /// static plan layout.
    pub fn shape_signature(
        &self,
        graph: &Graph,
        inputs: &HashMap<ValueId, DenseTensor<f32>>,
    ) -> Result<CudaShapeSignature> {
        let (bindings, _) = bind_inputs(graph, inputs)?;
        Ok(CudaShapeSignature::new(&bindings))
    }

    /// Creates a static specialization of a symbolic graph for one runtime
    /// shape. The signature is the cache identity; the canonical graph hash is
    /// intentionally not changed in place.
    pub fn specialize_graph(
        &self,
        graph: &Graph,
        inputs: &HashMap<ValueId, DenseTensor<f32>>,
    ) -> Result<(Graph, CudaShapeSignature)> {
        let (bindings, _) = bind_inputs(graph, inputs)?;
        self.specialize_graph_with_bindings(graph, &bindings)
    }

    pub fn specialize_graph_with_bindings(
        &self,
        graph: &Graph,
        bindings: &HashMap<SymbolId, usize>,
    ) -> Result<(Graph, CudaShapeSignature)> {
        let mut specialized = graph.clone();
        for value in &mut specialized.values {
            let dims = value
                .shape
                .dims()
                .iter()
                .map(|dim| match dim {
                    Dim::Static(value) => Ok(Dim::Static(*value)),
                    Dim::Symbol(symbol) => bindings
                        .get(symbol)
                        .copied()
                        .map(Dim::Static)
                        .ok_or_else(|| {
                            CudaError::Shape(format!(
                                "shape {} uses unbound symbol {symbol:?}",
                                value.shape
                            ))
                        }),
                    Dim::Dynamic => Err(CudaError::Capacity(format!(
                        "shape {} contains an unbounded dynamic dimension; \
                         bounded specialization requires a symbol or static dimension",
                        value.shape
                    ))),
                })
                .collect::<Result<Vec<_>>>()?;
            value.shape = Shape::new(dims);
        }
        specialized.validate()?;
        Ok((specialized, CudaShapeSignature::new(bindings)))
    }

    /// Creates a bounded dynamic plan at maximum capacity.
    ///
    /// The resulting plan has maximum-capacity slots. Execution validates the
    /// actual runtime bytes against those capacities before any launch.
    pub fn plan_bounded(
        &self,
        graph: &Graph,
        max_bindings: &HashMap<SymbolId, usize>,
    ) -> Result<(CudaMemoryPlan, CudaShapeSignature)> {
        if max_bindings.is_empty() {
            return Err(CudaError::Capacity(
                "bounded plan requires at least one symbol maximum".into(),
            ));
        }
        let (specialized, signature) = self.specialize_graph_with_bindings(graph, max_bindings)?;
        let plan = oxide_ml_compiler::MemoryPlanner::default().plan(&specialized)?;
        let plan = CudaMemoryPlan::from_bounded_memory_plan(&specialized, &plan, max_bindings)?;
        Ok((plan, signature))
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

        let (bindings, input_shapes) = bind_inputs(graph, inputs)?;
        let shapes = runtime_shapes(graph, &input_shapes)?;
        for &input_id in graph.inputs() {
            let value = graph.value(input_id)?;
            ensure_f32(value.dtype)?;
            let shape = input_shapes
                .iter()
                .find(|(id, _)| *id == input_id)
                .map(|(_, shape)| shape.clone())
                .ok_or_else(|| CudaError::Internal("input shape was not bound".into()))?;
            let host = inputs.get(&input_id).ok_or_else(|| {
                CudaError::Shape(format!(
                    "missing host input {:?} ({input_id})",
                    value.name.as_deref().unwrap_or("<unnamed>")
                ))
            })?;
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
                .map(|&output| {
                    shapes.get(&output).cloned().ok_or_else(|| {
                        CudaError::Internal(format!("missing runtime shape for {output}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let stream = self.device.default_stream();
            let outputs = self.execute_node(
                &stream,
                &node.op,
                &node.attrs,
                &operands,
                &output_shapes,
                None,
                &bindings,
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
        let bindings = HashMap::new();
        self.execute_node(&stream, op, attrs, operands, output_shapes, None, &bindings)
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
        let (run, _) = self.run_planned_on(graph, parameters, inputs, plan, &stream, true)?;
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
        let (run, completion) =
            self.run_planned_on(graph, parameters, inputs, plan, &stream, false)?;
        let completion = completion.ok_or_else(|| {
            CudaError::Internal("async execution did not record a completion event".into())
        })?;
        let context = Arc::clone(stream.context());
        let resources = ExecutionResources::new(
            run,
            stream,
            context,
            Some(completion),
            None,
            Arc::new(ExecutionDiagnostics::new()),
        );
        Ok(CudaExecution::new(
            resources,
            SharedExecutionLifecycle::new(),
        ))
    }

    fn run_planned_on(
        &self,
        graph: &Graph,
        parameters: &CudaParameterStore,
        inputs: &HashMap<ValueId, DenseTensor<f32>>,
        plan: &CudaMemoryPlan,
        stream: &Arc<CudaStream>,
        synchronize: bool,
    ) -> Result<(CudaRun, Option<EventLease>)> {
        graph.validate()?;
        let mut values: HashMap<ValueId, CudaTensor> = HashMap::new();
        let (bindings, input_shapes) = bind_inputs(graph, inputs)?;
        let shapes = runtime_shapes(graph, &input_shapes)?;
        validate_runtime_capacity(graph, plan, &bindings, &shapes)?;
        for &input_id in graph.inputs() {
            let value = graph.value(input_id)?;
            ensure_f32(value.dtype)?;
            let shape = input_shapes
                .iter()
                .find(|(id, _)| *id == input_id)
                .map(|(_, shape)| shape.clone())
                .ok_or_else(|| CudaError::Internal("input shape was not bound".into()))?;
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
                .map(|&output| {
                    shapes.get(&output).cloned().ok_or_else(|| {
                        CudaError::Internal(format!("missing runtime shape for {output}"))
                    })
                })
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
                &bindings,
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
        let run = CudaRun {
            values,
            outputs: graph.outputs().to_vec(),
        };
        if synchronize {
            stream.synchronize()?;
            Ok((run, None))
        } else {
            let completion = self.events.acquire()?;
            completion.record(stream)?;
            Ok((run, Some(completion)))
        }
    }
    fn execute_node(
        &self,
        stream: &Arc<CudaStream>,
        op: &Op,
        attrs: &OpAttrs,
        operands: &[CudaTensor],
        output_shapes: &[Vec<usize>],
        preallocated: Option<Vec<CudaTensor>>,
        bindings: &HashMap<SymbolId, usize>,
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
            Op::Neg
            | Op::ReLU
            | Op::GELU
            | Op::Sigmoid
            | Op::Tanh
            | Op::Sqrt
            | Op::Exp
            | Op::Log
            | Op::Abs => {
                let opcode = match op {
                    Op::ReLU => 0,
                    Op::GELU => 1,
                    Op::Sigmoid => 2,
                    Op::Tanh => 3,
                    Op::Neg => 4,
                    Op::Sqrt => 5,
                    Op::Exp => 6,
                    Op::Log => 7,
                    Op::Abs => 8,
                    _ => unreachable!(),
                };
                let mut out = next_output()?;
                self.kernels.unary(stream, opcode, &operands[0], &mut out)?;
                Ok(vec![out])
            }
            Op::AbsBackward => {
                let mut out = next_output()?;
                self.kernels
                    .binary(stream, 5, &operands[0], &operands[1], &mut out)?;
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
            Op::MatMulBias => {
                let (transpose_a, transpose_b) = match attrs {
                    OpAttrs::MatMul {
                        transpose_a,
                        transpose_b,
                    } => (*transpose_a, *transpose_b),
                    _ => return Err(CudaError::Internal("MatMulBias attrs missing".into())),
                };
                if operands[0].shape().len() != 2 || operands[1].shape().len() != 2 {
                    return Err(CudaError::Unsupported(
                        "CUDA matmul_bias currently supports rank-2 matrices only".into(),
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
                    return Err(CudaError::Shape("matmul_bias K mismatch".into()));
                }
                let lda = a[1];
                let ldb = b[1];
                let dims = output_shapes.first().cloned().ok_or_else(|| {
                    CudaError::Internal("missing matmul_bias output shape".into())
                })?;
                let mut product = CudaTensor::from_pool(&self.pool, dims)?;
                if !transpose_a && !transpose_b && m >= 32 && n >= 32 && k >= 32 {
                    self.kernels.gemm_tiled(
                        stream,
                        &operands[0],
                        &operands[1],
                        &mut product,
                        m,
                        n,
                        k,
                    )?;
                } else {
                    self.kernels.gemm_naive(
                        stream,
                        &operands[0],
                        &operands[1],
                        &mut product,
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        transpose_a,
                        transpose_b,
                    )?;
                }
                let mut out = next_output()?;
                self.kernels
                    .binary(stream, 0, &product, &operands[2], &mut out)?;
                Ok(vec![out])
            }
            Op::MulAdd => {
                let dims = output_shapes
                    .first()
                    .cloned()
                    .ok_or_else(|| CudaError::Internal("missing mul_add output shape".into()))?;
                let mut product = CudaTensor::from_pool(&self.pool, dims)?;
                self.kernels
                    .binary(stream, 2, &operands[0], &operands[1], &mut product)?;
                let mut out = next_output()?;
                self.kernels
                    .binary(stream, 0, &product, &operands[2], &mut out)?;
                Ok(vec![out])
            }
            Op::Cast => {
                // The CUDA executor is F32-only; a same-dtype cast is a copy.
                let target = match attrs {
                    OpAttrs::Cast { dtype } => *dtype,
                    _ => return Err(CudaError::Internal("Cast attrs missing".into())),
                };
                if target != DType::F32 {
                    return Err(CudaError::Unsupported(format!(
                        "CUDA cast to {target} is not implemented; the device executor is F32"
                    )));
                }
                let mut out = next_output()?;
                self.kernels.broadcast(stream, &operands[0], &mut out)?;
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
                    OpAttrs::Unbroadcast { shape } => resolve_shape(shape, bindings)?,
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
            Op::Custom => {
                let name = match attrs {
                    OpAttrs::Custom { name, .. } => name.as_str(),
                    _ => {
                        return Err(CudaError::Internal(
                            "Custom op is missing OpAttrs::Custom".into(),
                        ));
                    }
                };
                let registry = self.custom.as_ref().ok_or_else(|| {
                    CudaError::Unsupported(
                        "graph contains a custom kernel but CudaExecutor has no kernel registry"
                            .into(),
                    )
                })?;
                let outputs = registry.execute(name, self, stream, operands, output_shapes)?;
                Ok(outputs)
            }
            other => Err(CudaError::Unsupported(format!(
                "operation {} is not implemented by the CUDA backend",
                other.name()
            ))),
        }
    }
}

fn execution_state_error(error: ExecutionStateError) -> CudaError {
    CudaError::Execution(error.to_string())
}

fn ensure_f32(dtype: DType) -> Result<()> {
    if dtype != DType::F32 {
        return Err(CudaError::Unsupported(format!(
            "CUDA backend currently supports F32 only, got {dtype}"
        )));
    }
    Ok(())
}

/// Runtime symbol bindings produced from graph inputs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CudaShapeSignature {
    bindings: Vec<(u32, usize)>,
}

impl CudaShapeSignature {
    pub fn new(bindings: &HashMap<SymbolId, usize>) -> Self {
        let mut entries = bindings
            .iter()
            .map(|(symbol, value)| (symbol.0, *value))
            .collect::<Vec<_>>();
        entries.sort_unstable();
        CudaShapeSignature { bindings: entries }
    }

    pub fn as_entries(&self) -> &[(u32, usize)] {
        &self.bindings
    }
}

fn bind_inputs(
    graph: &Graph,
    inputs: &HashMap<ValueId, DenseTensor<f32>>,
) -> Result<(HashMap<SymbolId, usize>, Vec<(ValueId, Vec<usize>)>)> {
    let mut bindings: HashMap<SymbolId, usize> = HashMap::new();
    let mut resolved = Vec::with_capacity(graph.inputs().len());
    for &input_id in graph.inputs() {
        let value = graph.value(input_id)?;
        let host = inputs.get(&input_id).ok_or_else(|| {
            CudaError::Shape(format!(
                "missing host input {:?} ({input_id})",
                value.name.as_deref().unwrap_or("<unnamed>")
            ))
        })?;
        if host.dims().len() != value.shape.rank() {
            return Err(CudaError::Shape(format!(
                "input {input_id} rank mismatch: host {:?}, graph {}",
                host.dims(),
                value.shape
            )));
        }
        for (dim, actual) in value.shape.dims().iter().zip(host.dims()) {
            match dim {
                Dim::Static(expected) => {
                    if *expected != actual {
                        return Err(CudaError::Shape(format!(
                            "input {input_id} shape mismatch: host {:?}, graph {}",
                            host.dims(),
                            value.shape
                        )));
                    }
                }
                Dim::Symbol(symbol) => {
                    if let Some(previous) = bindings.insert(*symbol, actual)
                        && previous != actual
                    {
                        return Err(CudaError::Shape(format!(
                            "symbol {symbol:?} bound inconsistently ({previous} vs {actual})"
                        )));
                    }
                }
                Dim::Dynamic => {
                    // Unbounded dynamic dimensions are accepted from the
                    // actual input; they have no symbol binding and cannot
                    // participate in a static memory plan.
                }
            }
        }
        resolved.push((input_id, host.dims().to_vec()));
    }
    Ok((bindings, resolved))
}

fn checked_numel(dims: &[usize]) -> Result<usize> {
    let mut total = 1usize;
    for &dim in dims {
        total = total.checked_mul(dim).ok_or_else(|| {
            CudaError::Capacity(format!("shape {dims:?} overflows usize element count"))
        })?;
    }
    Ok(total)
}

fn validate_runtime_capacity(
    graph: &Graph,
    plan: &CudaMemoryPlan,
    bindings: &HashMap<SymbolId, usize>,
    shapes: &HashMap<ValueId, Vec<usize>>,
) -> Result<()> {
    for &(symbol, maximum) in plan.capacity_bindings() {
        let actual = bindings
            .get(&SymbolId(symbol))
            .copied()
            .ok_or_else(|| CudaError::Capacity(format!("bounded symbol {symbol} was not bound")))?;
        if actual > maximum {
            return Err(CudaError::Capacity(format!(
                "symbol {symbol} runtime {actual} exceeds maximum {maximum}"
            )));
        }
    }
    for (&value, assignment) in &plan.assignments {
        let shape = shapes
            .get(&value)
            .ok_or_else(|| CudaError::Capacity(format!("missing runtime shape for {value}")))?;
        let elements = checked_numel(shape)?;
        let dtype = graph.value(value)?.dtype;
        let bytes = elements
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| CudaError::Capacity(format!("value {value} byte size overflows")))?;
        if bytes > assignment.size_bytes {
            return Err(CudaError::Capacity(format!(
                "value {value} needs {bytes} bytes at runtime but capacity is {}",
                assignment.size_bytes
            )));
        }
        let slot = plan
            .slots
            .get(assignment.slot as usize)
            .ok_or_else(|| CudaError::InvalidPlan(format!("missing slot {}", assignment.slot)))?;
        if assignment.offset % slot.alignment != 0 {
            return Err(CudaError::Capacity(format!(
                "value {value} offset {} violates alignment {}",
                assignment.offset, slot.alignment
            )));
        }
        if assignment.offset + bytes > slot.size_bytes {
            return Err(CudaError::Capacity(format!(
                "value {value} range {}..{} exceeds slot capacity {}",
                assignment.offset,
                assignment.offset + bytes,
                slot.size_bytes
            )));
        }
    }
    Ok(())
}

fn runtime_shapes(
    graph: &Graph,
    inputs: &[(ValueId, Vec<usize>)],
) -> Result<HashMap<ValueId, Vec<usize>>> {
    let mut shapes = HashMap::new();
    for (value, shape) in inputs {
        shapes.insert(*value, shape.clone());
    }
    for parameter in graph.parameters() {
        let dims = parameter.shape.static_dims().ok_or_else(|| {
            CudaError::Shape(format!(
                "parameter {} has a non-static shape",
                parameter.name
            ))
        })?;
        shapes.insert(parameter.value, dims);
    }
    for node in graph.nodes() {
        let mut operands = Vec::with_capacity(node.inputs.len());
        for input in &node.inputs {
            let info = graph.value(*input)?;
            let dims = shapes.get(input).ok_or_else(|| {
                CudaError::Shape(format!("runtime shape for value {input} is missing"))
            })?;
            let mut concrete = info.clone();
            concrete.shape = Shape::from_static(dims);
            operands.push(ValueInfo {
                id: concrete.id,
                name: concrete.name,
                shape: concrete.shape,
                dtype: concrete.dtype,
                device: concrete.device,
                kind: concrete.kind,
            });
        }
        let inferred = infer_outputs(&node.op, &node.attrs, &operands)?;
        for (output, inferred) in node.outputs.iter().zip(inferred) {
            let dims = inferred.shape.static_dims().ok_or_else(|| {
                CudaError::Shape(format!(
                    "output {output} could not be specialized to a concrete shape"
                ))
            })?;
            shapes.insert(*output, dims);
        }
    }
    Ok(shapes)
}

fn resolve_shape(shape: &Shape, bindings: &HashMap<SymbolId, usize>) -> Result<Vec<usize>> {
    shape
        .dims()
        .iter()
        .map(|dim| match dim {
            Dim::Static(value) => Ok(*value),
            Dim::Symbol(symbol) => bindings.get(symbol).copied().ok_or_else(|| {
                CudaError::Shape(format!(
                    "shape {shape} uses unbound symbol {symbol:?}; bind it from a graph input"
                ))
            }),
            Dim::Dynamic => Err(CudaError::Shape(format!(
                "shape {shape} contains a dynamic dimension that cannot be specialized; \
                 use an input symbol or a fully static output shape"
            ))),
        })
        .collect()
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

    /// Submit a persistent-arena execution without synchronizing the host.
    ///
    /// The returned execution owns the `SlotSetLease` and the slot buffers.
    /// Dropping it before completion moves the slots to PendingCompletion
    /// instead of returning them to the free list.
    pub fn execute_async_arena(
        &self,
        graph: &Graph,
        parameters: &CudaParameterStore,
        inputs: &HashMap<ValueId, DenseTensor<f32>>,
        arena: &crate::arena::CudaPlannedArena,
        stream: Arc<CudaStream>,
    ) -> Result<CudaExecution> {
        let mut lease = arena.lease(&stream)?;
        let (slots, mut slot_events, mut slot_streams) = lease.take()?;
        let (run, slots, mut returned_events, returned_streams) = self.run_arena_on(
            graph,
            parameters,
            inputs,
            arena.plan(),
            slots,
            &mut slot_events,
            &mut slot_streams,
            &stream,
        )?;

        let diagnostics = Arc::new(ExecutionDiagnostics::new());
        let completion = match self.events.acquire() {
            Ok(completion) => completion,
            Err(error) => {
                // Submission failed after slot work was enqueued. Never leave
                // the slot set permanently Leased; finish it so the arena can
                // defer reclamation through its slot events.
                let mut state = ArenaLeaseState {
                    lease,
                    slots: Some(slots),
                    events: returned_events,
                    streams: returned_streams,
                };
                state.finish(&diagnostics);
                return Err(error);
            }
        };
        if let Err(error) = completion.record(&stream) {
            let mut state = ArenaLeaseState {
                lease,
                slots: Some(slots),
                events: returned_events,
                streams: returned_streams,
            };
            state.finish(&diagnostics);
            return Err(error);
        }
        let barrier = match completion.arc() {
            Ok(barrier) => barrier,
            Err(error) => {
                let mut state = ArenaLeaseState {
                    lease,
                    slots: Some(slots),
                    events: returned_events,
                    streams: returned_streams,
                };
                state.finish(&diagnostics);
                return Err(error);
            }
        };
        for events in &mut returned_events {
            events.push(barrier.clone());
        }

        let arena_state = ArenaLeaseState {
            lease,
            slots: Some(slots),
            events: returned_events,
            streams: returned_streams,
        };
        let context = Arc::clone(stream.context());
        let resources = ExecutionResources::new(
            run,
            stream,
            context,
            Some(completion),
            Some(arena_state),
            diagnostics,
        );
        Ok(CudaExecution::new(
            resources,
            SharedExecutionLifecycle::new(),
        ))
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
        let (bindings, input_shapes) = bind_inputs(graph, inputs)?;
        let shapes = runtime_shapes(graph, &input_shapes)?;
        validate_runtime_capacity(graph, plan, &bindings, &shapes)?;
        for &input_id in graph.inputs() {
            let value = graph.value(input_id)?;
            ensure_f32(value.dtype)?;
            let shape = input_shapes
                .iter()
                .find(|(id, _)| *id == input_id)
                .map(|(_, shape)| shape.clone())
                .ok_or_else(|| CudaError::Internal("input shape was not bound".into()))?;
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
                .map(|&output| {
                    shapes.get(&output).cloned().ok_or_else(|| {
                        CudaError::Internal(format!("missing runtime shape for {output}"))
                    })
                })
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
                &bindings,
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
